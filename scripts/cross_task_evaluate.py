#!/usr/bin/env python3
"""跨任务评估脚本

支持在不同任务上训练的模型在其他任务上评估，测试泛化能力。

用法:
    # 单个跨任务评估
    python scripts/cross_task_evaluate.py \
        dataset.name=ragtruth \
        model=llama2_7b_chat \
        model.short_name=Llama-2-7b-chat-hf \
        method=lookback_lens \
        method.level=sample \
        seed=42 \
        train_task=QA \
        eval_task=Summary
    
    # 或使用 --cross-task 模式
    python scripts/quick_eval.py --cross_task --train_task QA --all_tasks "QA Summary Data2txt"
"""
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, 
    accuracy_score, roc_curve, precision_recall_curve
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import setup_logging, set_seed, MethodConfig
from src.methods import create_method
from src.features.loader import load_features_for_method

logger = logging.getLogger(__name__)


def find_optimal_threshold(
    predictions: List,
    labels: List[int],
) -> Tuple[float, float]:
    """寻找最优阈值 (基于 F1 分数)"""
    scores = np.array([p.score for p in predictions])
    labels = np.array(labels)
    
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        preds = (scores > threshold).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def compute_metrics(
    method,
    features: List,
    labels: List[int],
    threshold: float = 0.5,
) -> Tuple[Dict[str, Any], List[float]]:
    """计算评估指标"""
    if not features:
        return {}, []
    
    predictions = method.predict_batch(features)
    scores = np.array([p.score for p in predictions])
    y_true = np.array(labels[:len(scores)])
    y_pred = (scores > threshold).astype(int)
    
    metrics = {
        "n_samples": len(scores),
        "n_positive": int(y_true.sum()),
        "n_negative": len(y_true) - int(y_true.sum()),
        "threshold": float(threshold),
    }
    
    if len(np.unique(y_true)) >= 2:
        metrics["auroc"] = float(roc_auc_score(y_true, scores))
        metrics["aupr"] = float(average_precision_score(y_true, scores))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        
        # 计算 precision 和 recall
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        
        metrics["precision"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        metrics["recall"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    else:
        metrics["auroc"] = 0.5
        metrics["aupr"] = 0.0
        metrics["f1"] = 0.0
        metrics["accuracy"] = 0.0
        metrics["warning"] = "Only one class in dataset"
    
    return metrics, scores.tolist()


def evaluate_cross_task(
    dataset_name: str,
    model_short: str,
    method_name: str,
    level: str,
    seed: int,
    train_task: str,
    eval_task: str,
    base_path: Path = Path("outputs"),
) -> Dict[str, Any]:
    """执行跨任务评估
    
    Args:
        dataset_name: 数据集名称
        model_short: 模型简称
        method_name: 方法名称
        level: 训练级别 (sample/token)
        seed: 随机种子
        train_task: 训练任务
        eval_task: 评估任务
        base_path: 输出基础路径
        
    Returns:
        评估结果字典
    """
    logger.info("=" * 60)
    logger.info(f"Cross-Task Evaluation")
    logger.info(f"Train Task: {train_task} -> Eval Task: {eval_task}")
    logger.info(f"Method: {method_name} ({level})")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Model: {model_short}")
    logger.info("=" * 60)
    
    # 模型路径 (从 train_task 加载)
    model_path = (
        base_path / "models" / dataset_name / model_short / 
        f"seed_{seed}" / train_task / method_name / level / "model.pkl"
    )
    
    # 评估特征路径 (使用 eval_task)
    eval_features_dir = (
        base_path / "features" / dataset_name / model_short /
        f"seed_{seed}" / f"{eval_task}_test"
    )
    
    # 结果输出路径
    output_dir = (
        base_path / "results" / dataset_name / model_short /
        f"seed_{seed}" / f"train_{train_task}_eval_{eval_task}" /
        method_name / level
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查路径
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return {"error": f"Model not found: {model_path}"}
    
    if not eval_features_dir.exists():
        # 尝试不带 _test 后缀
        eval_features_dir_alt = (
            base_path / "features" / dataset_name / model_short /
            f"seed_{seed}" / eval_task
        )
        if eval_features_dir_alt.exists():
            eval_features_dir = eval_features_dir_alt
        else:
            logger.error(f"Eval features not found: {eval_features_dir}")
            return {"error": f"Eval features not found: {eval_features_dir}"}
    
    # 加载模型
    method_config = MethodConfig(name=method_name, level=level)
    method = create_method(method_name, config=method_config)
    method.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    
    # 检查 token-level 状态
    if level == "token":
        if hasattr(method, 'is_token_fitted'):
            logger.info(f"Token-level fitted: {method.is_token_fitted}")
            if not method.is_token_fitted:
                logger.warning("Token classifier not fitted, falling back to sample-level")
    
    # 加载评估特征
    load_token_labels = (level == "token")
    eval_features, _ = load_features_for_method(
        eval_features_dir,
        method_name=method_name,
        load_token_labels=load_token_labels,
    )
    eval_labels = [f.label for f in eval_features]
    logger.info(f"Loaded {len(eval_features)} eval samples from {eval_features_dir}")
    
    # 计算最优阈值
    threshold = 0.5
    if len(np.unique(eval_labels)) > 1:
        predictions = method.predict_batch(eval_features)
        threshold, _ = find_optimal_threshold(predictions, eval_labels)
    
    # 评估
    metrics, scores = compute_metrics(method, eval_features, eval_labels, threshold)
    
    logger.info("")
    logger.info(f">>> Results (train={train_task}, eval={eval_task}):")
    logger.info(f"    AUROC:    {metrics.get('auroc', 0):.4f}")
    logger.info(f"    AUPR:     {metrics.get('aupr', 0):.4f}")
    logger.info(f"    F1:       {metrics.get('f1', 0):.4f}")
    logger.info(f"    Accuracy: {metrics.get('accuracy', 0):.4f}")
    logger.info(f"    Threshold: {threshold:.4f}")
    
    # ROC curve
    if scores and len(np.unique(eval_labels)) > 1:
        fpr, tpr, _ = roc_curve(eval_labels, scores)
        roc_data = [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr, tpr)]
        with open(output_dir / "roc_curve.json", "w") as f:
            json.dump(roc_data, f, indent=2)
    
    # 构建结果
    results = {
        "config": {
            "dataset": dataset_name,
            "model": model_short,
            "method": method_name,
            "level": level,
            "seed": seed,
            "train_task": train_task,
            "eval_task": eval_task,
            "is_cross_task": train_task != eval_task,
        },
        "metrics": metrics,
    }
    
    # 保存结果
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_dir / 'eval_results.json'}")
    
    return results


def run_full_cross_task_matrix(
    dataset_name: str,
    model_short: str,
    methods: List[str],
    levels: Dict[str, List[str]],
    task_types: List[str],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """运行完整的跨任务评估矩阵
    
    Args:
        dataset_name: 数据集名称
        model_short: 模型简称
        methods: 方法列表
        levels: 每个方法支持的级别
        task_types: 任务类型列表
        seed: 随机种子
        
    Returns:
        所有评估结果列表
    """
    all_results = []
    
    for train_task in task_types:
        for eval_task in task_types:
            for method in methods:
                for level in levels.get(method, ["sample"]):
                    result = evaluate_cross_task(
                        dataset_name=dataset_name,
                        model_short=model_short,
                        method_name=method,
                        level=level,
                        seed=seed,
                        train_task=train_task,
                        eval_task=eval_task,
                    )
                    all_results.append(result)
    
    return all_results


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """跨任务评估主入口"""
    setup_logging(level=logging.INFO)
    set_seed(cfg.seed)
    
    # 获取参数
    dataset_name = cfg.dataset.name
    model_short = cfg.model.short_name
    method_name = cfg.method.name
    level = cfg.method.level
    seed = cfg.seed
    
    # 从配置获取训练和评估任务
    train_task = cfg.get("train_task", cfg.dataset.task_type)
    eval_task = cfg.get("eval_task", cfg.dataset.task_type)
    
    result = evaluate_cross_task(
        dataset_name=dataset_name,
        model_short=model_short,
        method_name=method_name,
        level=level,
        seed=seed,
        train_task=train_task,
        eval_task=eval_task,
    )
    
    if "error" in result:
        logger.error(f"Evaluation failed: {result['error']}")
        return
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
