#!/usr/bin/env python3
"""Generate activations (features) for hallucination detection.

重构版本：使用统一的 PathManager 管理路径。

核心优化：在加载模型之前检查特征文件完整性，
如果 features_individual/ 中的样本齐全，直接跳过提取。

Usage:
    # 训练集
    python scripts/generate_activations.py dataset.name=ragtruth dataset.split_name=train
    
    # 测试集
    python scripts/generate_activations.py dataset.name=ragtruth dataset.split_name=test
    
    # 强制重新提取
    python scripts/generate_activations.py resume=false
"""
import sys
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import set_seed, setup_logging, PathManager
from src.utils import get_device_info

# 导入提取模块
from scripts.extraction import (
    load_samples_from_splits,
    get_task_types_from_config,
    finalize_outputs,
)
from scripts.extraction.extraction_loop import (
    run_extraction_loop,
    setup_extraction_environment,
)
from scripts.extraction.data_loader import log_model_filter_info

logger = logging.getLogger(__name__)


def check_features_complete(output_dir: Path, expected_count: int) -> bool:
    """快速检查 features_individual/ 中的特征文件是否齐全。
    
    Args:
        output_dir: 输出目录
        expected_count: 期望的样本数量
        
    Returns:
        True 如果特征文件数量 >= 期望数量
    """
    individual_dir = output_dir / "features_individual"
    
    if not individual_dir.exists():
        return False
    
    # 快速统计 .pt 文件数量（不加载内容）
    pt_files = list(individual_dir.glob("*.pt"))
    actual_count = len(pt_files)
    
    logger.info(f"Feature files check: {actual_count}/{expected_count} samples")
    
    return actual_count >= expected_count


def log_system_info() -> None:
    """记录系统和GPU信息。"""
    logger.info("=" * 70)
    logger.info("System Information")
    logger.info("=" * 70)
    
    logger.info(f"PyTorch version: {torch.__version__}")
    
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


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """特征提取主入口。
    
    流程：
    1. 初始化环境
    2. 加载样本
    3. 设置断点续传
    4. 运行提取循环
    5. 整合输出
    """
    # =========================================================================
    # 1. 初始化
    # =========================================================================
    setup_logging(level=logging.INFO)
    set_seed(cfg.seed)
    
    split_name = cfg.dataset.get("split_name", "train")
    
    # 创建路径管理器
    pm = PathManager.from_config(cfg)
    
    logger.info("=" * 70)
    logger.info(f"Generate Activations - Split: {split_name}")
    logger.info(f"Dataset: {pm.dataset_name}")
    logger.info(f"Model: {pm.model_short_name}")
    logger.info(f"Task: {pm.task_suffix}")
    logger.info("=" * 70)
    
    log_system_info()
    
    # =========================================================================
    # 2. 构建输出目录 (使用 PathManager)
    # =========================================================================
    output_dir = pm.get_features_dir(split=split_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # =========================================================================
    # 3. 加载样本
    # =========================================================================
    dataset_name = cfg.dataset.name
    task_types_filter = get_task_types_from_config(cfg)
    
    model_filter = cfg.dataset.get("models", None)
    if hasattr(model_filter, '__iter__') and not isinstance(model_filter, str):
        model_filter = list(model_filter)
    
    logger.info(f"Loading dataset: {dataset_name}")
    logger.info(f"Task types filter: {task_types_filter}")
    logger.info(f"Model filter: {model_filter}")
    logger.info(f"Split: {split_name}")
    
    try:
        all_samples = load_samples_from_splits(
            dataset_name=dataset_name,
            task_types_filter=task_types_filter,
            model_filter=model_filter,
            split=split_name
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    if len(all_samples) == 0:
        logger.error("No samples loaded! Check your filters (dataset.models, dataset.task_types)")
        return
    
    logger.info(f"Loaded {len(all_samples)} samples")
    log_model_filter_info(cfg, all_samples)
    
    # =========================================================================
    # 4. 设置断点续传
    # =========================================================================
    pending_samples, checkpoint_manager, saver, is_resume = setup_extraction_environment(
        cfg, output_dir, all_samples
    )
    
    if len(pending_samples) == 0:
        logger.info("All samples already processed!")
        finalize_outputs(output_dir, checkpoint_manager, all_samples, cfg)
        return
    
    # 保存配置
    OmegaConf.save(cfg, output_dir / "config.yaml")
    
    # =========================================================================
    # 5. 运行提取循环
    # =========================================================================
    run_extraction_loop(
        cfg=cfg,
        samples=pending_samples,
        output_dir=output_dir,
        checkpoint_manager=checkpoint_manager,
        saver=saver,
        is_resume=is_resume,
    )
    
    # =========================================================================
    # 6. 整合输出
    # =========================================================================
    finalize_outputs(output_dir, checkpoint_manager, all_samples, cfg)
    
    logger.info("=" * 70)
    logger.info("Done!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
