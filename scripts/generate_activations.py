#!/usr/bin/env python3
"""Generate activations (features) for hallucination detection.

内存优化版本，支持：
- 逐样本保存 (batch_size=1)
- 每个样本后立即释放GPU内存
- 异步保存 (ThreadPoolExecutor)
- 检查点/恢复支持
- 进度跟踪
- 多GPU支持
- 源模型过滤
- 内存预估

Usage:
    # 基本使用
    python scripts/generate_activations.py dataset.name=ragtruth model=mistral_7b
    
    # 按源模型过滤
    python scripts/generate_activations.py dataset.models=[gpt-4]
    
    # 按任务类型过滤（单个）
    python scripts/generate_activations.py dataset.task_type=QA
    
    # 按任务类型过滤（多个）
    python scripts/generate_activations.py dataset.task_types=[QA,Summary]
    
    # 多方法（自动计算特征需求）
    python scripts/generate_activations.py methods=[lapeigvals,entropy]
    
    # 启用完整注意力（hypergraph方法）
    python scripts/generate_activations.py methods=[hypergraph] \\
        features=with_full_attentions allow_full_attention=true
    
    # 从检查点恢复
    python scripts/generate_activations.py resume=true
    
    # 强制重新开始
    python scripts/generate_activations.py resume=false
"""
import sys
import re
import json
import logging
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import (
    DatasetConfig, ModelConfig, FeaturesConfig, Sample, ExtractedFeatures,
    set_seed, setup_logging,
)
from src.data import get_dataset
from src.models import get_model, unload_all_models
from src.features import create_extractor, create_extractor_from_requirements
from src.utils import (
    clear_gpu_memory, log_gpu_memory, MemoryTracker,
    get_device_info, get_all_cuda_devices, synchronize_all_gpus,
    is_multi_gpu_model, log_device_distribution,
)

# Import optimization utilities
from utils.checkpoint import CheckpointManager, get_pending_samples
from utils.async_saver import MemoryEfficientSaver
from utils.feature_manager import (
    FeatureManager, compute_union_requirements, create_feature_manager
)

logger = logging.getLogger(__name__)


# =============================================================================
# 系统信息
# =============================================================================

def log_system_info():
    """记录系统和GPU信息。"""
    logger.info("=" * 70)
    logger.info("System Information")
    logger.info("=" * 70)
    
    # PyTorch版本
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # GPU信息
    device_info = get_device_info()
    
    if device_info["cuda_available"]:
        logger.info(f"CUDA version: {device_info.get('cuda_version', 'N/A')}")
        logger.info(f"GPU count: {device_info['device_count']}")
        
        for i in range(device_info['device_count']):
            dev_key = f"device_{i}"
            if dev_key in device_info:
                dev = device_info[dev_key]
                logger.info(
                    f"  GPU {i}: {dev['name']} | "
                    f"Total: {dev['total_memory_gb']:.1f}GB | "
                    f"Free: {dev.get('free_memory_gb', 'N/A'):.1f}GB"
                )
    else:
        logger.warning("CUDA not available - using CPU")
    
    logger.info("=" * 70)


def log_model_filter_info(cfg: DictConfig, samples: List[Sample]):
    """记录模型过滤信息。"""
    # 收集所有源模型
    source_models = set()
    for sample in samples:
        src_model = sample.metadata.get("source_model")
        if src_model:
            source_models.add(src_model)
    
    # 检查配置中的过滤器
    model_filter = cfg.dataset.get("models", None)
    
    logger.info("-" * 50)
    logger.info("Data Source Models")
    logger.info("-" * 50)
    
    if model_filter:
        logger.info(f"Filter applied: {model_filter}")
    else:
        logger.info("Filter: None (all models)")
    
    logger.info(f"Models in loaded data: {sorted(source_models)}")
    
    # 按模型统计样本数
    model_counts = {}
    for sample in samples:
        src_model = sample.metadata.get("source_model", "unknown")
        model_counts[src_model] = model_counts.get(src_model, 0) + 1
    
    for model, count in sorted(model_counts.items()):
        logger.info(f"  {model}: {count} samples")
    
    logger.info("-" * 50)


# =============================================================================
# Task Type 解析函数（修复 DVC 路径问题）
# =============================================================================

def parse_task_types_to_list(task_types: Any) -> Optional[List[str]]:
    """将各种格式的 task_types 解析为标准列表。
    
    支持的输入格式：
    - None, 'null', 'None' -> None
    - 'QA' -> ['QA']
    - ['QA'] -> ['QA']
    - ['QA', 'Summary'] -> ['QA', 'Summary']
    - '[QA]' (字符串) -> ['QA']
    - "['QA']" (字符串) -> ['QA']
    - ListConfig(['QA']) -> ['QA']
    
    Returns:
        List[str] 或 None
    """
    if task_types is None:
        return None
    
    # 转换为字符串检查特殊值
    task_str = str(task_types).strip()
    if task_str.lower() in ('null', 'none', '[]', ''):
        return None
    
    # 如果是 ListConfig 或 list，直接处理
    if isinstance(task_types, (list, ListConfig)):
        if len(task_types) == 0:
            return None
        # 清理每个元素
        cleaned = [str(t).strip().strip("'\"") for t in task_types]
        return [c for c in cleaned if c]
    
    # 如果是字符串形式的列表，如 "['QA']" 或 "[QA]"
    if task_str.startswith('[') and task_str.endswith(']'):
        inner = task_str[1:-1].strip()
        if not inner:
            return None
        # 分割并清理
        parts = re.split(r'[,\s]+', inner)
        cleaned = [p.strip().strip("'\"") for p in parts if p.strip().strip("'\"")]
        return cleaned if cleaned else None
    
    # 普通字符串，如 'QA'
    return [task_str.strip("'\"")]


def get_task_types_from_config(cfg: DictConfig) -> Optional[List[str]]:
    """从配置中获取 task_types 列表。
    
    优先级：
    1. dataset.task_type (单个字符串，DVC matrix 使用)
    2. dataset.task_types (列表)
    
    Returns:
        List[str] 或 None
    """
    # 优先使用 task_type (单个) - DVC matrix 传入的格式
    task_type = cfg.dataset.get('task_type', None)
    if task_type is not None:
        parsed = parse_task_types_to_list(task_type)
        if parsed:
            return parsed
    
    # 其次使用 task_types (列表)
    task_types = cfg.dataset.get('task_types', None)
    return parse_task_types_to_list(task_types)


def get_task_suffix(cfg: DictConfig) -> str:
    """获取用于目录名的 task 后缀。
    
    Examples:
        None -> 'all'
        ['QA'] -> 'QA'
        ['QA', 'Summary'] -> 'QA_Summary'
    """
    task_types = get_task_types_from_config(cfg)
    
    if task_types is None or len(task_types) == 0:
        return "all"
    
    return "_".join(task_types)


# =============================================================================
# 辅助函数
# =============================================================================

def get_model_short_name(cfg: DictConfig) -> str:
    """获取模型短名称。"""
    if hasattr(cfg.model, 'short_name') and cfg.model.short_name:
        return cfg.model.short_name
    return cfg.model.name.split("/")[-1]


def build_output_dir(cfg: DictConfig) -> Path:
    """构建输出目录路径。
    
    新目录结构: {base}/{dataset}/{model}/seed_{seed}/{task_type}/
    
    修复：正确处理 task_type/task_types 的各种格式，
    包括 DVC 传入的 '[QA]' 字符串格式。
    """
    base_dir = Path(cfg.get("features_dir", "outputs/features"))
    dataset_name = cfg.dataset.name
    
    # 使用修复后的函数获取 task 后缀
    task_suffix = get_task_suffix(cfg)
    
    model_name = get_model_short_name(cfg)
    
    # 新目录结构: {base}/{dataset}/{model}/seed_{seed}/{task_type}/
    return base_dir / dataset_name / model_name / f"seed_{cfg.seed}" / task_suffix


def prepare_dataset_config(cfg: DictConfig) -> Dict[str, Any]:
    """准备 dataset 配置字典。
    
    处理 task_type/task_types 的兼容性，确保传给 DatasetConfig 的格式正确。
    """
    dataset_dict = OmegaConf.to_container(cfg.dataset, resolve=True)
    
    # 获取标准化的 task_types 列表
    task_types_list = get_task_types_from_config(cfg)
    
    # 设置 task_types（覆盖原有的任何格式）
    dataset_dict['task_types'] = task_types_list
    
    # 移除 task_type（避免传给 DatasetConfig 时出错）
    if 'task_type' in dataset_dict:
        del dataset_dict['task_type']
    
    return dataset_dict


def extract_features_dict(features: ExtractedFeatures) -> Dict[str, Any]:
    """将ExtractedFeatures转换为字典格式。"""
    result = {
        "sample_id": features.sample_id,
        "prompt_len": features.prompt_len,
        "response_len": features.response_len,
        "label": features.label,
        "layers": features.layers,
    }
    
    # 使用 getattr 安全访问可选属性
    # 注意：ExtractedFeatures类使用full_attention，但存储时使用full_attentions
    attr_mapping = {
        'attn_diags': 'attn_diags',
        'laplacian_diags': 'laplacian_diags', 
        'attn_entropy': 'attn_entropy',
        'full_attention': 'full_attentions',  # 类使用单数，存储使用复数
        'hidden_states': 'hidden_states',
        'token_probs': 'token_probs',
        'token_entropy': 'token_entropy',
    }
    for class_attr, storage_key in attr_mapping.items():
        value = getattr(features, class_attr, None)
        if value is not None:
            result[storage_key] = value
    
    return result


def save_sample_answer(sample: Sample, features: ExtractedFeatures) -> Dict[str, Any]:
    """创建样本答案条目。"""
    return {
        "id": sample.id,
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
        "task_type": sample.task_type.value if sample.task_type else None,
        "split": sample.split.value if sample.split else None,
        "source_model": sample.metadata.get("source_model"),
        "labels": sample.metadata.get("hallucination_spans", []),
        "prompt_len": features.prompt_len,
        "response_len": features.response_len,
    }


class ProgressTracker:
    """进度跟踪器。"""
    
    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = datetime.now()
    
    def update(self, n: int = 1):
        self.current += n
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.current / max(elapsed, 1)
        remaining = (self.total - self.current) / max(rate, 0.001)
        
        pct = 100 * self.current / max(self.total, 1)
        logger.info(
            f"[{self.desc}] {self.current}/{self.total} ({pct:.1f}%) "
            f"| {rate:.2f} samples/sec | ETA: {remaining:.0f}s"
        )


# =============================================================================
# 主函数
# =============================================================================

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """内存优化的特征提取主入口。"""
    setup_logging(level=logging.INFO)
    set_seed(cfg.seed)
    
    logger.info("=" * 70)
    logger.info("Generate Activations (Memory-Optimized)")
    logger.info("=" * 70)
    
    # 记录系统信息
    log_system_info()
    
    # 构建输出目录
    output_dir = build_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # 初始化检查点管理器
    checkpoint_manager = CheckpointManager(output_dir)
    
    # 初始化异步保存器
    saver = MemoryEfficientSaver(output_dir, max_workers=2)
    
    # =========================================================================
    # 加载数据集（使用修复后的配置处理）
    # =========================================================================
    dataset_dict = prepare_dataset_config(cfg)
    
    dataset_config = DatasetConfig(**dataset_dict)
    logger.info(f"Loading dataset: {dataset_config.name}")
    logger.info(f"Task types filter: {dataset_config.task_types}")
    
    dataset = get_dataset(config=dataset_config)
    all_samples = dataset.load(max_samples=dataset_config.max_samples)
    
    # 检查是否有样本
    if len(all_samples) == 0:
        logger.error("No samples loaded! Check your filters (dataset.models, dataset.task_types)")
        logger.error("Available filters can be viewed by examining the raw data")
        return
    
    logger.info(f"Loaded {len(all_samples)} samples")
    
    # 记录模型过滤信息
    log_model_filter_info(cfg, all_samples)
    
    # =========================================================================
    # 初始化检查点
    # =========================================================================
    config_for_hash = OmegaConf.to_container(cfg, resolve=True)
    resume = cfg.get("resume", True)
    is_resume = checkpoint_manager.initialize(
        total_samples=len(all_samples),
        config=config_for_hash,
        force_restart=not resume
    )
    
    # 获取待处理样本
    pending_samples = get_pending_samples(all_samples, checkpoint_manager)
    logger.info(
        f"Pending samples: {len(pending_samples)} "
        f"(skipping {len(all_samples) - len(pending_samples)} already processed)"
    )
    
    if len(pending_samples) == 0:
        logger.info("All samples already processed!")
        finalize_outputs(output_dir, checkpoint_manager, all_samples, cfg)
        return
    
    # 保存配置
    OmegaConf.save(cfg, output_dir / "config.yaml")
    
    # =========================================================================
    # 确定特征需求
    # =========================================================================
    methods = cfg.get("methods", ["lapeigvals"])
    if isinstance(methods, str):
        methods = [methods]
    
    # 获取全局 allow_full_attention 设置
    allow_full_attention = cfg.get("allow_full_attention", False)
    
    # 创建特征管理器
    feature_manager = create_feature_manager(
        methods=methods,
        allow_full_attention=allow_full_attention,
    )
    
    # 显示特征需求
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
    # 加载模型
    # =========================================================================
    model_config = ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    logger.info(f"Loading model: {model_config.name}")
    log_gpu_memory("Before model load")
    
    model = get_model(model_config)
    log_gpu_memory("After model load")
    
    # 记录设备分布
    if is_multi_gpu_model(model.model):
        logger.info("Model loaded in multi-GPU mode")
        log_device_distribution(model.model, logger)
    else:
        logger.info(f"Model loaded on: {model.get_device()}")
    
    # =========================================================================
    # 创建特征提取器
    # =========================================================================
    features_config_dict = OmegaConf.to_container(cfg.features, resolve=True)
    features_config_dict.update(feature_manager.to_features_config())
    features_config = FeaturesConfig(**features_config_dict)
    
    # 使用安全创建函数
    extractor = create_extractor_from_requirements(
        model=model,
        config=features_config,
        feature_requirements=feature_manager.get_combined_requirements().to_dict(),
        allow_full_attention=allow_full_attention,
    )
    
    # 显示内存预估
    extractor_mem = extractor.get_memory_estimate(seq_len=max_length)
    logger.info(f"Extractor memory estimate: {extractor_mem['total_mb']:.1f} MB/sample")
    
    # =========================================================================
    # 处理样本
    # =========================================================================
    logger.info(f"Extracting features (batch_size=1, async_save=True)...")
    progress = ProgressTracker(len(pending_samples), "Extracting")
    
    answers = []
    
    for sample in pending_samples:
        try:
            # 提取特征
            with MemoryTracker(f"Sample {sample.id}"):
                features = extractor.extract(sample)
            
            # 转换为字典并异步保存
            features_dict = extract_features_dict(features)
            metadata = {
                "model_name": model_config.name,
                "label": sample.label,
                "source_model": sample.metadata.get("source_model"),
            }
            
            saver.save_and_release(sample.id, features_dict, metadata, force_gc=True)
            
            # 保存答案信息
            answers.append(save_sample_answer(sample, features))
            
            # 标记为已完成
            checkpoint_manager.mark_completed(sample.id)
            
            # 清理特征对象
            del features
            
        except Exception as e:
            logger.warning(f"Failed to process {sample.id}: {e}")
            checkpoint_manager.mark_failed(sample.id, str(e))
        
        progress.update()
        
        # 定期清理GPU内存
        if progress.current % 10 == 0:
            clear_gpu_memory()
            
            # 多GPU同步
            if torch.cuda.device_count() > 1:
                synchronize_all_gpus()
            
            log_gpu_memory(f"After {progress.current} samples")
    
    # =========================================================================
    # 完成保存
    # =========================================================================
    logger.info("Waiting for async saves to complete...")
    save_stats = saver.finalize()
    logger.info(f"Save stats: {save_stats}")
    saver.shutdown()
    
    # 保存答案
    answers_path = output_dir / "answers.json"
    
    # 如果恢复，加载现有答案
    existing_answers = []
    if is_resume and answers_path.exists():
        try:
            with open(answers_path) as f:
                existing_answers = json.load(f)
            existing_ids = {a["id"] for a in existing_answers}
            answers = [a for a in answers if a["id"] not in existing_ids]
        except Exception:
            pass
    
    all_answers = existing_answers + answers
    with open(answers_path, "w", encoding="utf-8") as f:
        json.dump(all_answers, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(all_answers)} answers to {answers_path}")
    
    # 整合输出
    finalize_outputs(output_dir, checkpoint_manager, all_samples, cfg)
    
    # 清理
    unload_all_models()
    clear_gpu_memory()
    
    logger.info("=" * 70)
    logger.info("Done!")
    logger.info("=" * 70)


def finalize_outputs(
    output_dir: Path,
    checkpoint_manager: CheckpointManager,
    samples: List[Sample],
    cfg: DictConfig
):
    """整合单个特征文件为合并格式。"""
    logger.info("Consolidating features...")
    
    # 整合到特征目录
    features_dir = output_dir / "features"
    features_dir.mkdir(exist_ok=True)
    
    consolidated = checkpoint_manager.consolidate_features()
    
    # 保存各类特征
    feature_types = [
        ("attn_diags", "attn_diags.pt"),
        ("laplacian_diags", "laplacian_diags.pt"),
        ("attn_entropy", "attn_entropy.pt"),
        ("hidden_states", "hidden_states.pt"),
        ("token_probs", "token_probs.pt"),
        ("full_attentions", "full_attentions.pt"),
    ]
    
    for key, filename in feature_types:
        if consolidated.get(key):
            torch.save(consolidated[key], features_dir / filename)
            logger.info(f"Saved {key}: {len(consolidated[key])} samples")
    
    # 保存标签
    labels = torch.tensor([s.label if s.label is not None else -1 for s in samples])
    torch.save(labels, output_dir / "labels.pt")
    
    # 保存元数据
    progress = checkpoint_manager.get_progress()
    metadata = {
        "n_samples": len(samples),
        "n_processed": progress["processed"],
        "n_failed": progress["failed"],
        "sample_ids": consolidated.get("sample_ids", []),
        "config": OmegaConf.to_container(cfg, resolve=True) if cfg else {},
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Consolidated features saved to {features_dir}")


if __name__ == "__main__":
    main()