#!/usr/bin/env python3
"""评估幻觉检测模型"""
import sys
import re
import json
import logging
from pathlib import Path
from typing import Any, List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import (
    MethodConfig, SplitType, Sample, TaskType, ExtractedFeatures,
    set_seed, setup_logging,
)
from src.methods import create_method, BaseMethod
from src.evaluation import find_optimal_threshold

logger = logging.getLogger(__name__)


def parse_task_types(task_types: Any) -> Optional[List[str]]:
    """解析 task_types 配置为列表"""
    if task_types is None:
        return None
    task_str = str(task_types).strip()
    if task_str.lower() in ('null', 'none', '[]', ''):
        return None
    if isinstance(task_types, (list, ListConfig)):
        if len(task_types) == 0:
            return None
        return [str(t).strip().strip("'\"") for t in task_types if str(t).strip()]
    if task_str.startswith('[') and task_str.endswith(']'):
        inner = task_str[1:-1].strip()
        if not inner:
            return None
        parts = re.split(r'[,\s]+', inner)
        return [p.strip().strip("'\"") for p in parts if p.strip().strip("'\"")]
    return [task_str.strip("'\"")]


def get_task_suffix(cfg: DictConfig) -> str:
    """获取 task 后缀"""
    task_type = cfg.dataset.get('task_type', None)
    if task_type:
        parsed = parse_task_types(task_type)
        if parsed:
            return "_".join(parsed)
    task_types = cfg.dataset.get('task_types', None)
    parsed = parse_task_types(task_types)
    return "_".join(parsed) if parsed else "all"


def get_model_short_name(cfg: DictConfig) -> str:
    if hasattr(cfg.model, 'short_name') and cfg.model.short_name:
        return cfg.model.short_name
    return cfg.model.name.split("/")[-1]


def get_features_dir(cfg: DictConfig, split: str = "train") -> Path:
    """特征目录"""
    base_dir = Path(cfg.features_dir)
    task_suffix = get_task_suffix(cfg)
    
    if split == "test":
        task_suffix = f"{task_suffix}_test"
    
    return base_dir / cfg.dataset.name / get_model_short_name(cfg) / f"seed_{cfg.seed}" / task_suffix


def get_output_dir(cfg: DictConfig) -> Path:
    """输出目录"""
    base_dir = Path(cfg.models_dir)
    return base_dir / cfg.dataset.name / get_model_short_name(cfg) / f"seed_{cfg.seed}" / get_task_suffix(cfg) / cfg.method.name


def get_model_path(cfg: DictConfig) -> Path:
    return get_output_dir(cfg) / "probe" / "model.pkl"


def load_features(features_dir: Path, split_name: str = "train") -> tuple:
    """加载特征文件"""
    metadata_path = features_dir / "metadata.json"
    answers_path = features_dir / "answers.json"
    
    if not metadata_path.exists():
        logger.warning(f"metadata.json not found in {features_dir}")
        return [], []
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    samples = []
    if answers_path.exists():
        with open(answers_path) as f:
            answers = json.load(f)
        for ans in answers:
            task_type = None
            if ans.get("task_type"):
                try:
                    task_type = TaskType(ans["task_type"])
                except (ValueError, KeyError):
                    pass
            split = None
            if ans.get("split"):
                try:
                    split = SplitType(ans["split"])
                except (ValueError, KeyError):
                    pass
            samples.append(Sample(
                id=ans["id"],
                prompt=ans.get("prompt", ""),
                response=ans.get("response", ""),
                label=ans.get("label", 0),
                task_type=task_type,
                split=split,
                metadata={
                    "prompt_len": ans.get("prompt_len", 0),
                    "response_len": ans.get("response_len", 0),
                }
            ))
    
    features_subdir = features_dir / "features"
    if not features_subdir.exists():
        features_subdir = features_dir
    
    feature_files = {
        "attn_diags": "attn_diags.pt",
        "laplacian_diags": "laplacian_diags.pt",
        "attn_entropy": "attn_entropy.pt",
        "token_probs": "token_probs.pt",
        "token_entropy": "token_entropy.pt",
    }
    
    large_feature_indexes = {
        "hidden_states": "hidden_states_index.json",
        "full_attentions": "full_attentions_index.json",
    }
    
    loaded_features = {}
    for key, filename in feature_files.items():
        filepath = features_subdir / filename
        if filepath.exists():
            loaded_features[key] = torch.load(filepath, weights_only=False)
    
    feature_indexes = {}
    for key, filename in large_feature_indexes.items():
        filepath = features_subdir / filename
        if filepath.exists():
            with open(filepath) as f:
                feature_indexes[key] = json.load(f)
    
    labels_path = features_dir / "labels.pt"
    labels = torch.load(labels_path, weights_only=False) if labels_path.exists() else torch.tensor([s.label or 0 for s in samples])
    
    sample_ids = metadata.get("sample_ids", [s.id for s in samples])
    features_list = []
    
    for i, sample_id in enumerate(sample_ids):
        sample_features = {}
        for key, data in loaded_features.items():
            class_attr = "full_attention" if key == "full_attentions" else key
            if isinstance(data, dict) and sample_id in data:
                sample_features[class_attr] = data[sample_id]
            elif isinstance(data, list) and i < len(data):
                sample_features[class_attr] = data[i]
        
        # 准备大特征的懒加载路径
        feature_paths = {}
        for feature_key in ["hidden_states", "full_attentions"]:
            if feature_key in feature_indexes:
                index_data = feature_indexes[feature_key]
                if "index" in index_data and sample_id in index_data["index"]:
                    feature_paths[feature_key] = index_data["index"][sample_id]
        
        sample = samples[i] if i < len(samples) else None
        
        # 构建完整的 metadata，包含 _feature_paths
        sample_metadata = {
            "task_type_str": sample.task_type.value if sample and sample.task_type else "unknown",
            "_feature_paths": feature_paths,  # 关键：包含懒加载路径
        }
        
        features_list.append(ExtractedFeatures(
            sample_id=sample_id,
            prompt_len=sample.metadata.get("prompt_len", 0) if sample else 0,
            response_len=sample.metadata.get("response_len", 0) if sample else 0,
            label=int(labels[i]) if i < len(labels) else 0,
            attn_diags=sample_features.get("attn_diags"),
            laplacian_diags=sample_features.get("laplacian_diags"),
            attn_entropy=sample_features.get("attn_entropy"),
            hidden_states=None,
            token_probs=sample_features.get("token_probs"),
            token_entropy=sample_features.get("token_entropy"),
            full_attention=None,
            metadata=sample_metadata,  # 使用包含 _feature_paths 的完整 metadata
        ))
    
    logger.info(f"Loaded {len(features_list)} {split_name} samples from {features_dir}")
    return features_list, samples


def split_data(features_list, samples):
    """按 split 字段分割数据"""
    train_features, train_labels = [], []
    test_features, test_labels = [], []
    
    for feat, sample in zip(features_list, samples):
        label = feat.label if feat.label is not None else (sample.label or 0)
        if sample.split == SplitType.TRAIN:
            train_features.append(feat)
            train_labels.append(label)
        else:
            test_features.append(feat)
            test_labels.append(label)
    
    return train_features, train_labels, test_features, test_labels


def compute_metrics(method, features, labels, threshold=0.5):
    """计算评估指标"""
    if not features:
        return {}, []
    
    predictions = method.predict_batch(features)
    scores = np.array([p.score for p in predictions])
    y_true = np.array(labels)
    y_pred = (scores >= threshold).astype(int)
    
    metrics = {
        "n_samples": len(features),
        "n_positive": int(sum(labels)),
        "n_negative": len(labels) - int(sum(labels)),
    }
    
    if len(np.unique(y_true)) >= 2:
        metrics["auroc"] = float(roc_auc_score(y_true, scores))
        metrics["aupr"] = float(average_precision_score(y_true, scores))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    
    return metrics, scores.tolist()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging(level=logging.INFO)
    set_seed(cfg.seed)

    logger.info("=" * 60)
    logger.info(f"Evaluate: {cfg.method.name}")
    logger.info("=" * 60)

    model_path = get_model_path(cfg)
    train_features_dir = get_features_dir(cfg, split="train")
    test_features_dir = get_features_dir(cfg, split="test")

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
    if not train_features_dir.exists():
        logger.error(f"Train features not found: {train_features_dir}")
        return

    method_config = MethodConfig(**OmegaConf.to_container(cfg.method, resolve=True))
    method = create_method(method_config.name, config=method_config)
    method.load(model_path)

    train_features, train_samples = load_features(train_features_dir, "train")
    train_labels = [f.label for f in train_features]
    
    test_features, test_samples = [], []
    test_labels = []
    
    if test_features_dir.exists():
        test_features, test_samples = load_features(test_features_dir, "test")
        test_labels = [f.label for f in test_features]
        logger.info(f"Loaded test features from: {test_features_dir}")
    else:
        logger.info(f"Test features directory not found: {test_features_dir}")
        logger.info("Falling back to split field in train features...")
        
        train_features_split, train_labels_split, test_features, test_labels = split_data(
            train_features, train_samples
        )
        
        if test_features:
            logger.info(f"Found {len(test_features)} test samples from split field")
            train_features = train_features_split
            train_labels = train_labels_split
        else:
            logger.warning("No test data found! Only train metrics will be reported.")

    logger.info(f"Train: {len(train_features)}, Test: {len(test_features)}")

    threshold = 0.5
    if test_features and len(np.unique(test_labels)) > 1:
        test_preds = method.predict_batch(test_features)
        threshold, _ = find_optimal_threshold([p.score for p in test_preds], test_labels)

    train_metrics, _ = compute_metrics(method, train_features, train_labels, threshold)
    test_metrics, test_scores = compute_metrics(method, test_features, test_labels, threshold)

    logger.info("")
    logger.info(">>> Train (In-sample):")
    logger.info(f"    AUROC: {train_metrics.get('auroc', 0):.4f}")
    logger.info(f"    AUPR:  {train_metrics.get('aupr', 0):.4f}")
    logger.info("")
    logger.info(">>> Test (Out-of-sample):")
    logger.info(f"    AUROC: {test_metrics.get('auroc', 0):.4f}")
    logger.info(f"    AUPR:  {test_metrics.get('aupr', 0):.4f}")
    logger.info(f"    F1:    {test_metrics.get('f1', 0):.4f}")
    logger.info("")
    logger.info(f"Threshold: {threshold:.4f}")

    output_dir = get_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "threshold": threshold,
        "config": {
            "dataset": cfg.dataset.name,
            "model": get_model_short_name(cfg),
            "method": cfg.method.name,
            "seed": cfg.seed,
        }
    }

    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if test_scores and len(np.unique(test_labels)) > 1:
        fpr, tpr, _ = roc_curve(test_labels, test_scores)
        roc_data = [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr, tpr)]
        with open(output_dir / "roc_curve.json", "w") as f:
            json.dump(roc_data, f, indent=2)

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
