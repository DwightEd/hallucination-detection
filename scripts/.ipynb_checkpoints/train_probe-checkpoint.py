#!/usr/bin/env python3
"""Train hallucination detection probe.

重构版本：
- 使用统一的 PathManager 管理所有路径
- 使用统一的 FeatureLoader 加载特征
- 使用统一的 level 字段（替代 training_level）
- ⚠️ 当 level="both" 时，分别训练 sample 和 token 两个级别

Usage:
    python scripts/train_probe.py method=lapeigvals
    python scripts/train_probe.py method=haloscope method.level=token
    python scripts/train_probe.py method=lookback_lens  # level=both 会训练两次
"""
import sys
import json
import logging
from pathlib import Path
from typing import Optional, List, Tuple

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


def train_single_level(
    cfg: DictConfig,
    pm: PathManager,
    method_config: MethodConfig,
    train_features: List,
    train_labels: List,
    level: str,
) -> Tuple[bool, dict]:
    """训练单个级别的探针。
    
    Args:
        cfg: Hydra 配置
        pm: PathManager 实例
        method_config: 方法配置
        train_features: 训练特征
        train_labels: 训练标签
        level: 训练级别 ("sample" 或 "token")
        
    Returns:
        (success, metrics) 元组
    """
    method_name = method_config.name
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Training at level: {level}")
    logger.info("=" * 60)
    
    # 创建新的方法实例（每个级别需要独立的模型）
    # 创建一个修改了 level 的配置副本
    level_config = MethodConfig(
        name=method_config.name,
        level=level,
        feature_requirements=method_config.feature_requirements,
        hyperparameters=method_config.hyperparameters,
    )
    method = create_method(method_name, config=level_config)
    
    # 检查方法兼容性
    if level == "token" and not method.supports_token_level:
        logger.warning(
            f"Method '{method_name}' does not support token-level training. "
            f"Skipping token-level."
        )
        return False, {}
    
    # 检查 token-level labels 可用性
    if level == "token":
        n_with_labels = sum(1 for f in train_features if f.hallucination_labels is not None)
        n_hallucinated = sum(1 for f in train_features if f.label == 1)
        logger.info(f"Token-level labels: {n_with_labels}/{n_hallucinated} hallucinated samples")
        
        if n_with_labels == 0:
            logger.error("No token-level labels available! Skipping token-level training.")
            return False, {}
    
    # 性能追踪
    tracker = MetricsTracker(method_name=f"{method_name}_{level}")
    tracker.set_sample_info(n_samples=len(train_features))
    
    logger.info(f"Training {method_name} at {level} level...")
    tracker.start()
    
    # 训练
    fit_metrics = method.fit(train_features, train_labels, cv=False)
    tracker.stop()
    
    # 计算训练集指标
    train_metrics = compute_metrics(method, train_features, train_labels, "train_")
    metrics = {**fit_metrics, **train_metrics}
    metrics["level"] = level

    logger.info("-" * 40)
    logger.info(f"[{level}] Train AUROC: {metrics.get('train_auroc', 0):.4f}")
    logger.info(f"[{level}] Train AUPR:  {metrics.get('train_aupr', 0):.4f}")
    logger.info("-" * 40)

    # 获取输出目录（使用具体的 level，不是 "both"）
    output_dir = pm.get_model_dir(method=method_name, level=level)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    model_path = output_dir / "model.pkl"
    method.save(model_path)
    
    # 记录性能指标
    tracker.set_model_path(model_path)
    perf_metrics = tracker.get_metrics()
    
    logger.info("")
    logger.info(f"[{level}] Performance Metrics:")
    logger.info(f"  Training Time: {perf_metrics.training_time_seconds:.2f}s")
    logger.info(f"  Peak CPU Memory: {perf_metrics.peak_cpu_memory_mb:.1f} MB")
    logger.info(f"  Peak GPU Memory: {perf_metrics.peak_gpu_memory_mb:.1f} MB")
    logger.info(f"  Model Size: {perf_metrics.model_size_mb:.2f} MB")
    
    metrics["performance"] = perf_metrics.to_dict()
    
    # 保存训练指标和配置
    with open(output_dir / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    logger.info(f"[{level}] Model saved to: {model_path}")
    logger.info(f"[{level}] Metrics saved to: {output_dir / 'train_metrics.json'}")
    
    return True, metrics


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """训练主入口。
    
    当 level="both" 时，分别训练 sample 和 token 两个级别，
    保存到不同的目录。
    """
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
    config_level = method_config.level  # 原始配置的 level
    
    # 确定需要训练的级别列表
    if config_level == "both":
        # ⚠️ level="both" 时，分别训练 sample 和 token
        levels_to_train = ["sample", "token"]
        logger.info(f"Level config is 'both' → will train both sample and token levels")
    else:
        levels_to_train = [config_level]
    
    # 加载特征时，如果要训练 token 级别，需要加载 token labels
    load_token_labels = "token" in levels_to_train
    
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
    
    # 检查方法是否支持 token 级别
    temp_method = create_method(method_name, config=method_config)
    if "token" in levels_to_train and not temp_method.supports_token_level:
        logger.warning(
            f"Method '{method_name}' does not support token-level training. "
            f"Will only train at sample level."
        )
        levels_to_train = ["sample"]

    # 训练每个级别
    all_results = {}
    for level in levels_to_train:
        success, metrics = train_single_level(
            cfg=cfg,
            pm=pm,
            method_config=method_config,
            train_features=train_features,
            train_labels=train_labels,
            level=level,
        )
        if success:
            all_results[level] = metrics
    
    # 汇总
    logger.info("")
    logger.info("=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    for level, metrics in all_results.items():
        auroc = metrics.get('train_auroc', 0)
        aupr = metrics.get('train_aupr', 0)
        logger.info(f"  [{level}] AUROC: {auroc:.4f}, AUPR: {aupr:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
