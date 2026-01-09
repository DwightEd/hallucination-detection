#!/usr/bin/env python3
"""Evaluate a trained method on test data.

新目录结构:
- 特征: {features_dir}/{dataset}/{model}/seed_{seed}/{task_type}/
- 模型: {models_dir}/{dataset}/{model}/seed_{seed}/{task_type}/{method}/probe/
- 结果: {models_dir}/{dataset}/{model}/seed_{seed}/{task_type}/{method}/
"""
import sys
import re
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import (
    MethodConfig, EvalMetrics, Prediction, SplitType, Sample, TaskType,
    ExtractedFeatures, set_seed, setup_logging,
)
from src.methods import create_method, BaseMethod
from src.evaluation import compute_metrics, find_optimal_threshold

logger = logging.getLogger(__name__)


# =============================================================================
# Task Type 解析函数（与 generate_activations.py 保持一致）
# =============================================================================

def parse_task_types_to_list(task_types: Any) -> Optional[List[str]]:
    """将各种格式的 task_types 解析为标准列表。"""
    if task_types is None:
        return None
    
    task_str = str(task_types).strip()
    if task_str.lower() in ('null', 'none', '[]', ''):
        return None
    
    if isinstance(task_types, (list, ListConfig)):
        if len(task_types) == 0:
            return None
        cleaned = [str(t).strip().strip("'\"") for t in task_types]
        return [c for c in cleaned if c]
    
    if task_str.startswith('[') and task_str.endswith(']'):
        inner = task_str[1:-1].strip()
        if not inner:
            return None
        parts = re.split(r'[,\s]+', inner)
        cleaned = [p.strip().strip("'\"") for p in parts if p.strip().strip("'\"")]
        return cleaned if cleaned else None
    
    return [task_str.strip("'\"")]


def get_task_types_from_config(cfg: DictConfig) -> Optional[List[str]]:
    """从配置中获取 task_types 列表。"""
    task_type = cfg.dataset.get('task_type', None)
    if task_type is not None:
        parsed = parse_task_types_to_list(task_type)
        if parsed:
            return parsed
    
    task_types = cfg.dataset.get('task_types', None)
    return parse_task_types_to_list(task_types)


def get_task_suffix(cfg: DictConfig) -> str:
    """获取用于目录名的 task 后缀。"""
    task_types = get_task_types_from_config(cfg)
    if task_types is None or len(task_types) == 0:
        return "all"
    return "_".join(task_types)


# =============================================================================
# 辅助函数
# =============================================================================

def get_model_short_name(cfg: DictConfig) -> str:
    """Get short name for model."""
    if hasattr(cfg.model, 'short_name') and cfg.model.short_name:
        return cfg.model.short_name
    return cfg.model.name.split("/")[-1]


def get_base_output_dir(cfg: DictConfig) -> Path:
    """Get the base output directory (without probe subdirectory).
    
    新目录结构: {base}/{dataset}/{model}/seed_{seed}/{task_type}/{method}/
    """
    base_dir = Path(cfg.models_dir)
    dataset_name = cfg.dataset.name
    task_suffix = get_task_suffix(cfg)
    model_name = get_model_short_name(cfg)
    method_name = cfg.method.name

    # 新目录结构
    return base_dir / dataset_name / model_name / f"seed_{cfg.seed}" / task_suffix / method_name


def find_model_path(cfg: DictConfig) -> Path:
    """Find the trained model path based on config.
    
    尝试新旧两种目录结构。
    """
    # 新目录结构
    base_output_dir = get_base_output_dir(cfg)
    model_path_new = base_output_dir / "probe" / "model.pkl"
    
    if model_path_new.exists():
        return model_path_new
    
    # 旧目录结构
    base_dir = Path(cfg.models_dir)
    dataset_name = cfg.dataset.name
    task_suffix = get_task_suffix(cfg)
    model_name = get_model_short_name(cfg)
    method_name = cfg.method.name
    
    model_path_old = base_dir / f"{dataset_name}_{task_suffix}" / model_name / method_name / f"seed_{cfg.seed}" / "probe" / "model.pkl"
    
    if model_path_old.exists():
        logger.info(f"Using legacy model path: {model_path_old}")
        return model_path_old
    
    # 返回新路径（用于错误信息）
    return model_path_new


def find_features_dir(cfg: DictConfig) -> Path:
    """Find the features directory based on config.
    
    新目录结构: {base}/{dataset}/{model}/seed_{seed}/{task_type}/
    """
    base_dir = Path(cfg.features_dir)
    dataset_name = cfg.dataset.name
    task_suffix = get_task_suffix(cfg)
    model_name = get_model_short_name(cfg)
    seed_str = f"seed_{cfg.seed}"

    # 新目录结构
    features_dir_new = base_dir / dataset_name / model_name / seed_str / task_suffix
    
    if features_dir_new.exists():
        return features_dir_new
    
    # 旧目录结构
    features_dir_old = base_dir / f"{dataset_name}_{task_suffix}" / model_name / seed_str
    
    if features_dir_old.exists():
        logger.info(f"Using legacy features path: {features_dir_old}")
        return features_dir_old
    
    return features_dir_new


def load_model(model_path: Path, method_config: MethodConfig) -> BaseMethod:
    """Load trained model."""
    method = create_method(method_config.name, config=method_config)
    method.load(model_path)
    return method


def load_features(features_dir: Path) -> tuple:
    """Load features from directory.
    
    支持两种格式：
    1. features.pkl - 旧格式，单个pickle文件
    2. features/ 目录 - 新格式，包含多个 .pt 文件
    """
    features_path = features_dir / "features.pkl"
    
    if features_path.exists():
        # 旧格式
        with open(features_path, "rb") as f:
            data = pickle.load(f)
        features_list = data.get("features", [])
        samples = data.get("samples", [])
        return features_list, samples
    
    # 新格式：从目录加载
    if (features_dir / "metadata.json").exists():
        return load_features_from_dir(features_dir)
    
    raise FileNotFoundError(f"No features found in {features_dir}")


def load_features_from_dir(features_dir: Path) -> tuple:
    """从新格式目录加载特征。"""
    import json
    
    # 加载元数据
    metadata_path = features_dir / "metadata.json"
    answers_path = features_dir / "answers.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {features_dir}")
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # 加载样本信息
    samples = []
    if answers_path.exists():
        with open(answers_path) as f:
            answers = json.load(f)
        
        for ans in answers:
            # 转换 task_type
            task_type = None
            if ans.get("task_type"):
                try:
                    task_type = TaskType(ans["task_type"])
                except (ValueError, KeyError):
                    pass
            
            # 转换 split
            split = None
            if ans.get("split"):
                try:
                    split = SplitType(ans["split"])
                except (ValueError, KeyError):
                    pass
            
            sample = Sample(
                id=ans["id"],
                prompt=ans.get("prompt", ""),
                response=ans.get("response", ""),
                label=ans.get("label", 0),
                task_type=task_type,
                split=split,
                metadata={
                    "source_model": ans.get("source_model"),
                    "hallucination_spans": ans.get("labels", []),
                    "prompt_len": ans.get("prompt_len", 0),
                    "response_len": ans.get("response_len", 0),
                }
            )
            samples.append(sample)
    
    # 加载特征文件
    features_subdir = features_dir / "features"
    if not features_subdir.exists():
        features_subdir = features_dir
    
    feature_files = {
        "attn_diags": "attn_diags.pt",
        "laplacian_diags": "laplacian_diags.pt",
        "attn_entropy": "attn_entropy.pt",
        "hidden_states": "hidden_states.pt",
        "token_probs": "token_probs.pt",
        "token_entropy": "token_entropy.pt",
        "full_attentions": "full_attentions.pt",  # 存储使用复数命名
    }
    
    loaded_features = {}
    for key, filename in feature_files.items():
        filepath = features_subdir / filename
        if filepath.exists():
            loaded_features[key] = torch.load(filepath, weights_only=False)
            if isinstance(loaded_features[key], dict):
                logger.info(f"Loaded {key}: {len(loaded_features[key])} samples (dict format)")
            elif isinstance(loaded_features[key], list):
                logger.info(f"Loaded {key}: {len(loaded_features[key])} samples (list format)")
            else:
                logger.info(f"Loaded {key}")
    
    # 加载标签
    labels_path = features_dir / "labels.pt"
    if labels_path.exists():
        labels = torch.load(labels_path, weights_only=False)
    else:
        labels = torch.tensor([s.label if s.label is not None else 0 for s in samples])
    
    # 构建 ExtractedFeatures 列表
    sample_ids = metadata.get("sample_ids", [s.id for s in samples])
    n_samples = len(sample_ids)
    
    features_list = []
    for i in range(n_samples):
        # 获取该样本的各类特征
        sample_features = {}
        sample_id = sample_ids[i] if i < len(sample_ids) else str(i)
        
        for key, data in loaded_features.items():
            # 将存储键映射回类属性名
            class_attr = key if key != "full_attentions" else "full_attention"
            
            if isinstance(data, dict):
                # 新格式：dict 以 sample_id 为键
                if sample_id in data:
                    sample_features[class_attr] = data[sample_id]
            elif isinstance(data, list):
                # 旧格式：list 按索引访问
                if i < len(data):
                    sample_features[class_attr] = data[i]
        
        # 获取样本信息
        sample = samples[i] if i < len(samples) else None
        label = int(labels[i]) if i < len(labels) else 0
        
        # 创建 ExtractedFeatures 对象
        feat = ExtractedFeatures(
            sample_id=sample_id,
            prompt_len=sample.metadata.get("prompt_len", 0) if sample else 0,
            response_len=sample.metadata.get("response_len", 0) if sample else 0,
            label=label,
            attn_diags=sample_features.get("attn_diags"),
            laplacian_diags=sample_features.get("laplacian_diags"),
            attn_entropy=sample_features.get("attn_entropy"),
            hidden_states=sample_features.get("hidden_states"),
            token_probs=sample_features.get("token_probs"),
            token_entropy=sample_features.get("token_entropy"),
            full_attention=sample_features.get("full_attention"),  # 映射为单数形式
            metadata={
                "task_type_str": sample.task_type.value if sample and sample.task_type else "unknown",
            } if sample else {},
        )
        features_list.append(feat)
    
    return features_list, samples


def get_test_data(features_list, samples):
    """Get test split data. 

    Samples should already have split field set.
    """
    test_features = []
    test_labels = []

    for feat, sample in zip(features_list, samples):
        # 使用 test split (或非 train 的都算 test)
        if sample.split is None or sample.split != SplitType.TRAIN:
            label = feat.label if feat.label is not None else (sample.label if sample.label is not None else 0)
            test_features.append(feat)
            test_labels.append(label)

    # 如果没有 test split，使用所有数据
    if len(test_features) == 0:
        logger.warning("No test split found, using all data for evaluation")
        test_features = features_list
        test_labels = [f.label if f.label is not None else 0 for f in features_list]

    return test_features, test_labels


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation."""

    setup_logging(level=logging.INFO)
    set_seed(cfg.seed)

    logger.info("=" * 60)
    logger.info("Evaluate")
    logger.info("=" * 60)
    logger.info(f"Task types: {get_task_types_from_config(cfg)}")

    # Find paths
    model_path = find_model_path(cfg)
    features_dir = find_features_dir(cfg)

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.error("Please run train_probe.py first")
        return

    if not features_dir.exists():
        logger.error(f"Features directory not found: {features_dir}")
        logger.error("Please run generate_activations.py first")
        return

    logger.info(f"Model: {model_path}")
    logger.info(f"Features: {features_dir}")

    # Load model
    method_config = MethodConfig(**OmegaConf.to_container(cfg.method, resolve=True))
    method = load_model(model_path, method_config)
    logger.info(f"Loaded method: {method_config.name}")

    # Load features
    try:
        features_list, samples = load_features(features_dir)
    except Exception as e:
        logger.error(f"Failed to load features: {e}")
        import traceback
        traceback.print_exc()
        return

    # Get test data
    test_features, test_labels = get_test_data(features_list, samples)
    logger.info(f"Test samples: {len(test_features)}")

    n_pos = sum(1 for l in test_labels if l == 1)
    n_neg = sum(1 for l in test_labels if l == 0)
    logger.info(f"Test labels: {n_pos} positive, {n_neg} negative")

    # Predict
    logger.info("Predicting...")
    predictions = method.predict_batch(test_features)
    logger.info(f"Generated {len(predictions)} predictions")

    # Get threshold
    threshold = cfg.get("evaluation", {}).get("threshold", None)
    scores = [p.score for p in predictions]

    if threshold is None:
        threshold, _ = find_optimal_threshold(scores, test_labels)
        logger.info(f"Optimal threshold: {threshold:.4f}")
    else:
        logger.info(f"Using specified threshold: {threshold:.4f}")

    # Compute metrics
    metrics = compute_metrics(scores, test_labels, threshold=threshold)

    logger.info("=" * 40)
    logger.info("Evaluation Results:")
    logger.info("=" * 40)
    logger.info(f"  AUROC:      {metrics.auroc:.4f}")
    logger.info(f"  AUPRC:      {metrics.auprc:.4f}")
    logger.info(f"  F1:         {metrics.f1:.4f}")
    logger.info(f"  Precision:  {metrics.precision:.4f}")
    logger.info(f"  Recall:     {metrics.recall:.4f}")
    logger.info(f"  Accuracy:   {metrics.accuracy:.4f}")
    logger.info(f"  Threshold:  {threshold:.4f}")
    logger.info("=" * 40)

    # Save results
    output_dir = get_base_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "eval_results.json"

    # 按 task_type 分组的结果
    by_task = {}
    for i, (feat, pred) in enumerate(zip(test_features, predictions)):
        task = feat.metadata.get("task_type_str", "unknown") if feat.metadata else "unknown"
        if task not in by_task:
            by_task[task] = {"scores": [], "labels": []}
        by_task[task]["scores"].append(pred.score)
        by_task[task]["labels"].append(test_labels[i])

    task_metrics = {}
    for task, data in by_task.items():
        if len(set(data["labels"])) > 1:
            task_m = compute_metrics(data["scores"], data["labels"], threshold=threshold)
            task_metrics[task] = {
                "auroc": task_m.auroc,
                "auprc": task_m.auprc,
                "f1": task_m.f1,
                "n_samples": len(data["labels"]),
                "n_positive": sum(1 for l in data["labels"] if l == 1),
            }

    results = {
        "metrics": metrics.to_dict(),
        "by_task_type": task_metrics,
        "threshold": threshold,
        "n_samples": len(predictions),
        "config": {
            "dataset": cfg.dataset.name,
            "model": get_model_short_name(cfg),
            "method": cfg.method.name,
            "seed": cfg.seed,
        }
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")

    # Save ROC curve data
    roc_path = output_dir / "roc_curve.json"
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(test_labels, scores)
        roc_data = [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr, tpr)]
        with open(roc_path, "w") as f:
            json.dump(roc_data, f, indent=2)
        logger.info(f"ROC curve saved to {roc_path}")
    except Exception as e:
        logger.warning(f"Failed to save ROC curve: {e}")

    # Print by task type
    if task_metrics:
        logger.info("\nResults by Task Type:")
        logger.info("-" * 60)
        for task, m in task_metrics.items():
            logger.info(f"  {task}: AUROC={m['auroc']:.4f}, F1={m['f1']:.4f}, N={m['n_samples']}")

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__": 
    main()