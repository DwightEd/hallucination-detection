#!/usr/bin/env python3
"""评估幻觉检测模型。

重构版本：
- 使用统一的 PathManager 管理所有路径
- 使用统一的 FeatureLoader 加载特征
- 使用统一的 level 字段（替代 classification_level）

支持 sample-level 和 token-level 评估。
通过 level 参数决定评估级别：
- sample: 评估 sample 级别分类器
- token: 评估 token 级别分类器

Usage:
    python scripts/evaluate.py method=lapeigvals
    python scripts/evaluate.py method=lapeigvals method.level=token
"""
import sys
import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Dict

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, 
    precision_score, recall_score, roc_curve
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import (
    MethodConfig, SplitType, Sample, ExtractedFeatures,
    PathManager, set_seed, setup_logging,
)
from src.methods import create_method, BaseMethod
from src.evaluation import find_optimal_threshold
from src.features.loader import (
    load_features_for_method,
    split_features_by_split,
)

logger = logging.getLogger(__name__)


def spans_to_token_labels(
    hallucination_spans: List[Dict],
    response_text: str,
    response_len: int,
    prompt_len: int = 0,
) -> List[int]:
    """Convert hallucination spans to token-level labels.
    
    Args:
        hallucination_spans: 幻觉区间列表
        response_text: 响应文本
        response_len: 响应 token 数量
        prompt_len: 提示 token 数量
        
    Returns:
        Token 级别标签列表
    """
    total_len = prompt_len + response_len
    full_labels = [0] * total_len
    
    if not hallucination_spans or response_len == 0:
        return full_labels
    
    char_len = len(response_text) if response_text else 1
    char_labels = [0] * char_len
    
    for span in hallucination_spans:
        if isinstance(span, dict):
            start = span.get('start', 0)
            end = span.get('end', 0)
        elif isinstance(span, (list, tuple)) and len(span) >= 2:
            start, end = span[0], span[1]
        else:
            continue
        
        for i in range(max(0, start), min(end, char_len)):
            char_labels[i] = 1
    
    for token_idx in range(response_len):
        char_start = int(token_idx * char_len / response_len)
        char_end = int((token_idx + 1) * char_len / response_len)
        
        if any(char_labels[i] for i in range(char_start, min(char_end, char_len))):
            full_labels[prompt_len + token_idx] = 1
    
    return full_labels


def compute_sample_metrics(
    method: BaseMethod,
    features: List[ExtractedFeatures],
    labels: List[int],
    threshold: float = 0.5,
) -> tuple:
    """计算 sample-level 指标。
    
    Args:
        method: 方法实例
        features: 特征列表
        labels: 标签列表
        threshold: 分类阈值
        
    Returns:
        (metrics_dict, scores_list)
    """
    if not features:
        return {}, []
    
    predictions = method.predict_batch(features)
    scores = [p.score for p in predictions]
    y_true = np.array(labels)
    y_scores = np.array(scores)
    y_pred = (y_scores >= threshold).astype(int)
    
    metrics = {
        "n_samples": len(features),
        "n_positive": int(y_true.sum()),
        "n_negative": int(len(y_true) - y_true.sum()),
        "threshold": threshold,
    }
    
    if len(np.unique(y_true)) >= 2:
        metrics["auroc"] = float(roc_auc_score(y_true, y_scores))
        metrics["aupr"] = float(average_precision_score(y_true, y_scores))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    
    return metrics, scores


def compute_token_metrics(
    method: BaseMethod,
    features: List[ExtractedFeatures],
    threshold: float = 0.5,
) -> tuple:
    """计算 token-level 指标。
    
    Args:
        method: 方法实例
        features: 特征列表
        threshold: 分类阈值
        
    Returns:
        (metrics_dict, scores_list, per_sample_results)
    """
    all_scores = []
    all_labels = []
    per_sample_results = []
    
    for feat in features:
        if feat.label != 1 or feat.hallucination_labels is None:
            continue
        
        try:
            token_scores = method.predict_token_scores(feat)
            if token_scores is None:
                continue
            
            full_labels = feat.hallucination_labels
            prompt_len = feat.prompt_len
            response_len = feat.response_len
            response_labels = full_labels[prompt_len:prompt_len + response_len]
            
            min_len = min(len(token_scores), len(response_labels))
            
            if min_len > 0:
                sample_scores = token_scores[:min_len]
                sample_labels = response_labels[:min_len]
                
                all_scores.extend(sample_scores)
                all_labels.extend(sample_labels)
                
                per_sample_results.append({
                    "sample_id": feat.sample_id,
                    "token_scores": sample_scores,
                    "token_labels": sample_labels,
                    "n_tokens": min_len,
                    "n_hallucinated": sum(sample_labels),
                })
        except Exception as e:
            logger.debug(f"Token prediction failed for {feat.sample_id}: {e}")
    
    if not all_labels:
        return {}, [], []
    
    y_true = np.array(all_labels)
    y_scores = np.array(all_scores)
    y_pred = (y_scores >= threshold).astype(int)
    
    metrics = {
        "n_tokens": len(y_true),
        "n_positive": int(y_true.sum()),
        "n_negative": int(len(y_true) - y_true.sum()),
        "n_samples_evaluated": len(per_sample_results),
        "threshold": threshold,
    }
    
    if len(np.unique(y_true)) >= 2:
        metrics["auroc"] = float(roc_auc_score(y_true, y_scores))
        metrics["aupr"] = float(average_precision_score(y_true, y_scores))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    
    return metrics, y_scores.tolist(), per_sample_results


def get_level(cfg: DictConfig) -> str:
    """从配置获取级别。
    
    Args:
        cfg: 配置对象
        
    Returns:
        "sample" 或 "token"
    """
    # 优先使用 method.level
    if hasattr(cfg.method, 'level') and cfg.method.level:
        return cfg.method.level
    # 向后兼容：检查旧字段名
    if hasattr(cfg, 'level') and cfg.level:
        return cfg.level
    return "sample"


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """评估主入口。"""
    setup_logging(level=logging.INFO)
    set_seed(cfg.seed)
    
    # 创建路径管理器
    pm = PathManager.from_config(cfg)
    
    level = get_level(cfg)
    method_name = cfg.method.name
    
    logger.info("=" * 60)
    logger.info(f"Evaluate: {method_name}")
    logger.info(f"Dataset: {pm.dataset_name}")
    logger.info(f"Model: {pm.model_short_name}")
    logger.info(f"Task: {pm.task_suffix}")
    logger.info(f"Level: {level}")
    logger.info("=" * 60)

    # 使用 PathManager 获取路径
    model_path = pm.get_model_path(method=method_name, level=level)
    train_features_dir = pm.get_features_dir(split="train")
    test_features_dir = pm.get_features_dir(split="test")

    # 检查模型是否存在
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.info(f"Please train the model first:")
        logger.info(f"  python scripts/train_probe.py method={method_name}")
        return
    
    if not train_features_dir.exists():
        logger.error(f"Train features not found: {train_features_dir}")
        return

    # 加载模型
    method_config = MethodConfig(**OmegaConf.to_container(cfg.method, resolve=True))
    method = create_method(method_config.name, config=method_config)
    method.load(model_path)
    logger.info(f"Loaded model from {model_path}")

    load_token_labels = level == "token"
    
    # 使用统一的 FeatureLoader 加载训练特征
    train_features, train_samples = load_features_for_method(
        train_features_dir,
        method_name=method_name,
        load_token_labels=load_token_labels,
    )
    train_labels = [f.label for f in train_features]
    logger.info(f"Loaded {len(train_features)} train samples from {train_features_dir}")
    
    # 加载测试特征
    test_features, test_samples = [], []
    test_labels = []
    
    if test_features_dir.exists():
        test_features, test_samples = load_features_for_method(
            test_features_dir,
            method_name=method_name,
            load_token_labels=load_token_labels,
        )
        test_labels = [f.label for f in test_features]
        logger.info(f"Loaded {len(test_features)} test samples from {test_features_dir}")
    else:
        logger.info(f"Test features directory not found: {test_features_dir}")
        logger.info("Falling back to split field in train features...")
        
        train_features_split, train_labels_split, test_features, test_labels = split_features_by_split(
            train_features, train_samples
        )
        
        if test_features:
            logger.info(f"Found {len(test_features)} test samples from split field")
            train_features = train_features_split
            train_labels = train_labels_split
        else:
            logger.warning("No test data found! Only train metrics will be reported.")

    logger.info(f"Train: {len(train_features)}, Test: {len(test_features)}")

    # 使用 PathManager 获取输出目录
    output_dir = pm.get_model_dir(method=method_name, level=level)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "level": level,
        "config": {
            "dataset": pm.dataset_name,
            "model": pm.model_short_name,
            "method": method_name,
            "seed": pm.seed,
            "task": pm.task_suffix,
        }
    }

    # ==================== Sample Level Evaluation ====================
    if level == "sample":
        threshold = 0.5
        if test_features and len(np.unique(test_labels)) > 1:
            test_preds = method.predict_batch(test_features)
            threshold, _ = find_optimal_threshold(test_preds, test_labels)

        train_metrics, _ = compute_sample_metrics(method, train_features, train_labels, threshold)
        test_metrics, test_scores = compute_sample_metrics(method, test_features, test_labels, threshold)

        logger.info("")
        logger.info(">>> [Sample Level] Train (In-sample):")
        logger.info(f"    AUROC: {train_metrics.get('auroc', 0):.4f}")
        logger.info(f"    AUPR:  {train_metrics.get('aupr', 0):.4f}")
        logger.info("")
        logger.info(">>> [Sample Level] Test (Out-of-sample):")
        logger.info(f"    AUROC: {test_metrics.get('auroc', 0):.4f}")
        logger.info(f"    AUPR:  {test_metrics.get('aupr', 0):.4f}")
        logger.info(f"    F1:    {test_metrics.get('f1', 0):.4f}")
        logger.info("")
        logger.info(f"Threshold: {threshold:.4f}")

        results["train_metrics"] = train_metrics
        results["test_metrics"] = test_metrics
        results["threshold"] = threshold

        # ROC curve
        if test_scores and len(np.unique(test_labels)) > 1:
            fpr, tpr, _ = roc_curve(test_labels, test_scores)
            roc_data = [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr, tpr)]
            with open(output_dir / "roc_curve.json", "w") as f:
                json.dump(roc_data, f, indent=2)

    # ==================== Token Level Evaluation ====================
    elif level == "token":
        if not hasattr(method, 'is_token_fitted') or not method.is_token_fitted:
            logger.error("Model does not have token-level classifier fitted!")
            return
        
        threshold = 0.5
        
        train_metrics, _, train_per_sample = compute_token_metrics(method, train_features, threshold)
        test_metrics, test_scores, test_per_sample = compute_token_metrics(method, test_features, threshold)

        logger.info("")
        logger.info(">>> [Token Level] Train (In-sample):")
        logger.info(f"    AUROC: {train_metrics.get('auroc', 0):.4f}")
        logger.info(f"    AUPR:  {train_metrics.get('aupr', 0):.4f}")
        logger.info(f"    F1:    {train_metrics.get('f1', 0):.4f}")
        logger.info(f"    Tokens: {train_metrics.get('n_tokens', 0)}")
        logger.info("")
        logger.info(">>> [Token Level] Test (Out-of-sample):")
        logger.info(f"    AUROC: {test_metrics.get('auroc', 0):.4f}")
        logger.info(f"    AUPR:  {test_metrics.get('aupr', 0):.4f}")
        logger.info(f"    F1:    {test_metrics.get('f1', 0):.4f}")
        logger.info(f"    Precision: {test_metrics.get('precision', 0):.4f}")
        logger.info(f"    Recall: {test_metrics.get('recall', 0):.4f}")
        logger.info(f"    Tokens: {test_metrics.get('n_tokens', 0)}")
        logger.info("")
        logger.info(f"Threshold: {threshold:.4f}")

        results["train_metrics"] = train_metrics
        results["test_metrics"] = test_metrics
        results["threshold"] = threshold

        if test_per_sample:
            with open(output_dir / "token_predictions.json", "w") as f:
                json.dump(test_per_sample, f, indent=2)
            logger.info(f"Saved token predictions for {len(test_per_sample)} samples")

    # Save results
    eval_results_path = output_dir / "eval_results.json"
    with open(eval_results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {eval_results_path}")


if __name__ == "__main__":
    main()
