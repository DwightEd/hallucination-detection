"""Main extraction loop module.

负责核心的特征提取循环逻辑，包括：
- 模型加载
- 特征提取器创建
- 逐样本提取和保存
- 内存管理
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from src.core import (
    Sample, ModelConfig, FeaturesConfig, ExtractedFeatures,
)
from src.models import get_model, unload_all_models
from src.features import create_extractor_from_requirements
from src.utils import (
    clear_gpu_memory, log_gpu_memory, MemoryTracker,
    is_multi_gpu_model, log_device_distribution, synchronize_all_gpus,
)

from utils.feature_manager import create_feature_manager
from utils.checkpoint import CheckpointManager
from utils.async_saver import MemoryEfficientSaver

from .progress import ProgressTracker
from .output_manager import (
    extract_features_dict, save_sample_answer, 
    save_answers, finalize_outputs,
)

logger = logging.getLogger(__name__)


def run_extraction_loop(
    cfg: DictConfig,
    samples: List[Sample],
    output_dir: Path,
    checkpoint_manager: CheckpointManager,
    saver: MemoryEfficientSaver,
    is_resume: bool = False,
) -> None:
    """运行主特征提取循环。
    
    Args:
        cfg: Hydra 配置对象
        samples: 待处理的样本列表
        output_dir: 输出目录
        checkpoint_manager: 断点管理器
        saver: 异步保存器
        is_resume: 是否为续传模式
    """
    # =========================================================================
    # 1. 创建特征管理器
    # =========================================================================
    methods = cfg.get("methods", ["lapeigvals"])
    if isinstance(methods, str):
        methods = [methods]
    
    allow_full_attention = cfg.get("allow_full_attention", False)
    
    feature_manager = create_feature_manager(
        methods=methods,
        allow_full_attention=allow_full_attention,
    )
    
    logger.info(feature_manager.describe())
    
    # 内存预估
    max_length = cfg.features.get("max_length", 4096)
    mem_estimate = feature_manager.estimate_memory_per_sample(
        seq_len=max_length,
        n_layers=cfg.model.get("n_layers", 32),
        n_heads=cfg.model.get("n_heads", 32),
        hidden_size=cfg.model.get("hidden_size", 4096),
    )
    
    logger.info(f"Estimated memory per sample: {mem_estimate['total_mb']:.1f} MB")
    
    if mem_estimate['total_gb'] > 10:
        logger.warning(
            f"⚠️ High memory usage expected: {mem_estimate['total_gb']:.1f} GB/sample"
        )
    
    # =========================================================================
    # 2. 加载模型
    # =========================================================================
    model_config = ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    logger.info(f"Loading model: {model_config.name}")
    log_gpu_memory("Before model load")
    
    model = get_model(model_config)
    log_gpu_memory("After model load")
    
    if is_multi_gpu_model(model.model):
        logger.info("Model loaded in multi-GPU mode")
        log_device_distribution(model.model, logger)
    else:
        logger.info(f"Model loaded on: {model.get_device()}")
    
    # =========================================================================
    # 3. 创建特征提取器
    # =========================================================================
    features_config_dict = OmegaConf.to_container(cfg.features, resolve=True)
    features_config_dict.update(feature_manager.to_features_config())
    features_config = FeaturesConfig(**features_config_dict)
    
    extractor = create_extractor_from_requirements(
        model=model,
        config=features_config,
        feature_requirements=feature_manager.get_combined_requirements().to_dict(),
        allow_full_attention=allow_full_attention,
    )
    
    extractor_mem = extractor.get_memory_estimate(seq_len=max_length)
    logger.info(f"Extractor memory estimate: {extractor_mem['total_mb']:.1f} MB/sample")
    
    # =========================================================================
    # 4. 主提取循环
    # =========================================================================
    logger.info(f"Extracting features (batch_size=1, async_save=True)...")
    progress = ProgressTracker(len(samples), "Extracting")
    
    answers = []
    
    for sample in samples:
        try:
            with MemoryTracker(f"Sample {sample.id}"):
                features = extractor.extract(sample)
            
            features_dict = extract_features_dict(features)
            metadata = {
                "model_name": model_config.name,
                "label": sample.label,
                "source_model": sample.metadata.get("source_model"),
            }
            
            saver.save_and_release(sample.id, features_dict, metadata, force_gc=True)
            
            answers.append(save_sample_answer(sample, features))
            
            checkpoint_manager.mark_completed(sample.id)
            
            del features
            
        except Exception as e:
            logger.warning(f"Failed to process {sample.id}: {e}")
            checkpoint_manager.mark_failed(sample.id, str(e))
        
        progress.update()
        
        # 定期清理内存
        if progress.current % 10 == 0:
            clear_gpu_memory()
            
            if torch.cuda.device_count() > 1:
                synchronize_all_gpus()
            
            log_gpu_memory(f"After {progress.current} samples")
    
    # =========================================================================
    # 5. 完成保存
    # =========================================================================
    logger.info("Waiting for async saves to complete...")
    save_stats = saver.finalize()
    logger.info(f"Save stats: {save_stats}")
    saver.shutdown()
    
    # 保存 answers
    save_answers(output_dir, answers, is_resume)
    
    # 清理资源
    unload_all_models()
    clear_gpu_memory()


def setup_extraction_environment(
    cfg: DictConfig,
    output_dir: Path,
    all_samples: List[Sample],
) -> tuple:
    """设置提取环境，返回待处理样本和管理器。
    
    核心逻辑：
    1. 扫描 features_individual/ 中已存在的特征文件
    2. 与 all_samples 对比，找出缺失的样本
    3. 如果全部齐全 → pending_samples 为空
    4. 如果有缺失 → 只返回缺失的样本
    
    Args:
        cfg: 配置对象
        output_dir: 输出目录
        all_samples: 所有样本列表
        
    Returns:
        (pending_samples, checkpoint_manager, saver, is_resume)
    """
    from utils.checkpoint import get_pending_samples, get_failed_samples, clear_failed_samples
    from .output_manager import compute_stable_config_hash
    
    split_name = cfg.dataset.get("split_name", "train")
    
    checkpoint_manager = CheckpointManager(output_dir)
    saver = MemoryEfficientSaver(output_dir, max_workers=2)
    
    # 配置哈希仅用于日志记录
    config_for_hash = compute_stable_config_hash(cfg, split_name)
    resume = cfg.get("resume", True)
    
    # 获取所有样本的ID列表
    all_sample_ids = [s.id for s in all_samples]
    
    # 初始化时传入样本ID列表，用于完整性检查
    is_resume = checkpoint_manager.initialize(
        total_samples=len(all_samples),
        config=config_for_hash,
        force_restart=not resume,
        sample_ids=all_sample_ids,  # 传入期望的样本列表
    )
    
    # 获取待处理的样本
    pending_samples = get_pending_samples(all_samples, checkpoint_manager)
    
    # 处理 retry_failed 选项
    retry_failed = cfg.get("retry_failed", False)
    if retry_failed:
        failed_samples = get_failed_samples(all_samples, checkpoint_manager)
        if failed_samples:
            logger.info(f"Retrying {len(failed_samples)} failed samples")
            clear_failed_samples(checkpoint_manager)
            pending_ids = {s.id for s in pending_samples}
            for s in failed_samples:
                if s.id not in pending_ids:
                    pending_samples.append(s)
    
    # 打印状态
    failed_count = len(checkpoint_manager.state.failed_ids)
    if failed_count > 0:
        logger.warning(f"⚠️ {failed_count} samples previously failed. Use retry_failed=true to retry them.")
    
    if len(pending_samples) == 0:
        logger.info(f"✅ All {len(all_samples)} samples already extracted, nothing to do")
    else:
        logger.info(
            f"Pending samples: {len(pending_samples)} "
            f"(skipping {len(all_samples) - len(pending_samples)} already extracted)"
        )
    
    return pending_samples, checkpoint_manager, saver, is_resume
