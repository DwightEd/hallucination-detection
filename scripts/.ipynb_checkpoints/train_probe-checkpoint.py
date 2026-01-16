#!/usr/bin/env python3
"""Train hallucination detection probe.

重构版本：
- 使用统一的 PathManager 管理所有路径
- 使用统一的 FeatureLoader 加载特征
- 使用统一的 level 字段（替代 training_level）

Usage:
    python scripts/train_probe.py method=lapeigvals
    python scripts/train_probe.py method=haloscope method.level=token
"""
import sys
import json
import logging
from pathlib import Path
from typing import Optional, List

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import (
    MethodConfig, SplitType, PathManager,
    set_seed, setup_logging,
)
from src.methods import create_method
from src.utils.metrics_tracker import MetricsTracker
from src.features.loader import (
    load_features_for_method,
    split_features_by_split,
)

logger = logging.getLogger(__name__)


def compute_metrics(method, features, labels, prefix=""):
    """Compute AUROC and AUPR.
    
    Args:
        method: 训练好的方法实例
        features: 特征列表
        labels: 标签列表
        prefix: 指标名称前缀
        
    Returns:
        指标字典
    """
    if not features:
        return {}
    
    predictions = method.predict_batch(features)
    scores = np.array([p.score for p in predictions])
    y_true = np.array(labels)
    
    metrics = {
        f"{prefix}n_samples": len(features),
        f"{prefix}n_positive": int(sum(labels)),
        f"{prefix}n_negative": len(labels) - int(sum(labels)),
    }
    
    if len(np.unique(y_true)) >= 2:
        metrics[f"{prefix}auroc"] = float(roc_auc_score(y_true, scores))
        metrics[f"{prefix}aupr"] = float(average_precision_score(y_true, scores))
    
    return metrics


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """训练主入口。"""
    setup_logging(level=logging.INFO)
    set_seed(cfg.seed)

    # 创建路径管理器
    pm = PathManager.from_config(cfg)
    
    logger.info("=" * 60)
    logger.info(f"Train Probe: {cfg.method.name}")
    logger.info(f"Dataset: {pm.dataset_name}")
    logger.info(f"Model: {pm.model_short_name}")
    logger.info(f"Task: {pm.task_suffix}")
    logger.info("=" * 60)

    # 使用 PathManager 获取特征目录
    features_dir = pm.get_features_dir(split="train")
    
    if not features_dir.exists():
        logger.error(f"Features not found: {features_dir}")
        logger.info("Please run generate_activations.py first.")
        return

    # 获取方法配置
    method_name = cfg.method.name
    method_config = MethodConfig(**OmegaConf.to_container(cfg.method, resolve=True))
    level = method_config.level
    load_token_labels = level in ("token", "both")
    
    # 使用统一的 FeatureLoader 加载特征
    features_list, samples = load_features_for_method(
        features_dir,
        method_name=method_name,
        load_token_labels=load_token_labels,
    )
    logger.info(f"Loaded {len(features_list)} samples from {features_dir}")

    # 按 split 分割数据
    train_features, train_labels, test_features, test_labels = split_features_by_split(
        features_list, samples
    )
    
    if not train_features:
        logger.warning("No train split found, using all data for training")
        train_features = features_list
        train_labels = [f.label or 0 for f in features_list]

    logger.info(f"Train: {len(train_features)} ({sum(train_labels)} positive)")
    logger.info(f"Test:  {len(test_features)} ({sum(test_labels)} positive)")

    # 创建方法实例
    method = create_method(method_config.name, config=method_config)
    
    # 检查 level 和方法兼容性
    logger.info(f"Level: {level}")
    
    if level in ("token", "both") and not method.supports_token_level:
        logger.warning(
            f"Method '{method_config.name}' does not support token-level training. "
            f"Falling back to sample-level."
        )
        level = "sample"
    
    # 统计 token-level labels 可用性
    if level in ("token", "both"):
        n_with_labels = sum(1 for f in train_features if f.hallucination_labels is not None)
        n_hallucinated = sum(1 for f in train_features if f.label == 1)
        logger.info(f"Token-level labels: {n_with_labels}/{n_hallucinated} hallucinated samples")
        
        if n_with_labels == 0 and level == "token":
            logger.error("No token-level labels available! Cannot train at token level.")
            logger.error("Re-extract features with the latest code to get hallucination_labels.")
            return

    # 性能追踪
    tracker = MetricsTracker(method_name=method_name)
    tracker.set_sample_info(n_samples=len(train_features))
    
    logger.info("Training...")
    tracker.start()
    
    # 训练
    fit_metrics = method.fit(train_features, train_labels, cv=False)
    tracker.stop()
    
    # 计算训练集指标
    train_metrics = compute_metrics(method, train_features, train_labels, "train_")
    metrics = {**fit_metrics, **train_metrics}
    metrics["level"] = level

    logger.info("=" * 40)
    logger.info(f"Train AUROC: {metrics.get('train_auroc', 0):.4f}")
    logger.info(f"Train AUPR:  {metrics.get('train_aupr', 0):.4f}")
    logger.info("=" * 40)

    # 使用 PathManager 获取输出目录
    output_dir = pm.get_model_dir(method=method_name, level=level)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    model_path = output_dir / "model.pkl"
    method.save(model_path)
    
    # 记录性能指标
    tracker.set_model_path(model_path)
    perf_metrics = tracker.get_metrics()
    
    logger.info("")
    logger.info("Performance Metrics:")
    logger.info(f"  Training Time: {perf_metrics.training_time_seconds:.2f}s")
    logger.info(f"  Peak CPU Memory: {perf_metrics.peak_cpu_memory_mb:.1f} MB")
    logger.info(f"  Peak GPU Memory: {perf_metrics.peak_gpu_memory_mb:.1f} MB")
    logger.info(f"  Model Size: {perf_metrics.model_size_mb:.2f} MB")
    
    metrics["performance"] = perf_metrics.to_dict()
    
    # 保存训练指标和配置
    with open(output_dir / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Metrics saved to: {output_dir / 'train_metrics.json'}")


if __name__ == "__main__":
    main()
