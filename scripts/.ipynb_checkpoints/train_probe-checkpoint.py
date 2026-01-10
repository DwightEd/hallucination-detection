#!/usr/bin/env python3
"""Train hallucination detection probe."""
import sys
import json
import logging
from pathlib import Path
from typing import Optional, List

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import (
    MethodConfig, ExtractedFeatures, SplitType, Sample, TaskType,
    set_seed, setup_logging,
)
from src.methods import create_method
from src.utils.metrics_tracker import MetricsTracker

logger = logging.getLogger(__name__)


def parse_task_types(task_types) -> Optional[List[str]]:
    """Parse task_types config to list."""
    if task_types is None:
        return None
    task_str = str(task_types).strip()
    if task_str.lower() in ('null', 'none', '[]', ''):
        return None
    if isinstance(task_types, (list,)):
        if len(task_types) == 0:
            return None
        return [str(t).strip().strip("'\"") for t in task_types if str(t).strip()]
    if task_str.startswith('[') and task_str.endswith(']'):
        inner = task_str[1:-1].strip()
        if not inner:
            return None
        import re
        parts = re.split(r'[,\s]+', inner)
        return [p.strip().strip("'\"") for p in parts if p.strip().strip("'\"")]
    return [task_str.strip("'\"")]


def get_task_suffix(cfg: DictConfig) -> str:
    """Get task suffix for directory naming."""
    task_type = cfg.dataset.get('task_type', None)
    if task_type:
        parsed = parse_task_types(task_type)
        if parsed:
            return "_".join(parsed)
    task_types = cfg.dataset.get('task_types', None)
    parsed = parse_task_types(task_types)
    return "_".join(parsed) if parsed else "all"


def get_model_short_name(cfg: DictConfig) -> str:
    """Get model short name."""
    if hasattr(cfg.model, 'short_name') and cfg.model.short_name:
        return cfg.model.short_name
    return cfg.model.name.split("/")[-1]


def get_features_dir(cfg: DictConfig) -> Path:
    """Features directory."""
    base_dir = Path(cfg.features_dir)
    return base_dir / cfg.dataset.name / get_model_short_name(cfg) / f"seed_{cfg.seed}" / get_task_suffix(cfg)


def get_output_dir(cfg: DictConfig) -> Path:
    """Output directory."""
    base_dir = Path(cfg.models_dir)
    return base_dir / cfg.dataset.name / get_model_short_name(cfg) / f"seed_{cfg.seed}" / get_task_suffix(cfg) / cfg.method.name / "probe"


def load_features(features_dir: Path) -> tuple:
    """Load feature files with proper lazy loading support."""
    metadata_path = features_dir / "metadata.json"
    answers_path = features_dir / "answers.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {features_dir}")
    
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
                    "source_model": ans.get("source_model"),
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
            logger.info(f"Loaded {key}")
    
    feature_indexes = {}
    for key, filename in large_feature_indexes.items():
        filepath = features_subdir / filename
        if filepath.exists():
            with open(filepath) as f:
                feature_indexes[key] = json.load(f)
            logger.info(f"Loaded {key} index ({feature_indexes[key].get('sample_count', 0)} samples)")
    
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
        
        # 准备大特征的懒加载路径 - 这是关键修复
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
    
    return features_list, samples


def split_data(features_list, samples):
    """Split data by split field."""
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


def compute_metrics(method, features, labels, prefix=""):
    """Compute AUROC and AUPR."""
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
    setup_logging(level=logging.INFO)
    set_seed(cfg.seed)

    logger.info("=" * 60)
    logger.info(f"Train Probe: {cfg.method.name}")
    logger.info("=" * 60)

    features_dir = get_features_dir(cfg)
    if not features_dir.exists():
        logger.error(f"Features not found: {features_dir}")
        return

    features_list, samples = load_features(features_dir)
    logger.info(f"Loaded {len(features_list)} samples")

    train_features, train_labels, test_features, test_labels = split_data(features_list, samples)
    
    if not train_features:
        logger.warning("No train split, using all data")
        train_features = features_list
        train_labels = [f.label or 0 for f in features_list]

    logger.info(f"Train: {len(train_features)} ({sum(train_labels)} pos), Test: {len(test_features)}")

    method_config = MethodConfig(**OmegaConf.to_container(cfg.method, resolve=True))
    method = create_method(method_config.name, config=method_config)

    tracker = MetricsTracker(method_name=cfg.method.name)
    tracker.set_sample_info(n_samples=len(train_features))
    
    logger.info("Training...")
    tracker.start()
    fit_metrics = method.fit(train_features, train_labels, cv=False)
    tracker.stop()
    
    train_metrics = compute_metrics(method, train_features, train_labels, "train_")
    metrics = {**fit_metrics, **train_metrics}

    logger.info("=" * 40)
    logger.info(f"Train AUROC: {metrics.get('train_auroc', 0):.4f}")
    logger.info(f"Train AUPR:  {metrics.get('train_aupr', 0):.4f}")
    logger.info("=" * 40)

    output_dir = get_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "model.pkl"
    method.save(model_path)
    
    tracker.set_model_path(model_path)
    perf_metrics = tracker.get_metrics()
    
    logger.info("")
    logger.info("Performance Metrics:")
    logger.info(f"  Training Time: {perf_metrics.training_time_seconds:.2f}s")
    logger.info(f"  Peak CPU Memory: {perf_metrics.peak_cpu_memory_mb:.1f} MB")
    logger.info(f"  Peak GPU Memory: {perf_metrics.peak_gpu_memory_mb:.1f} MB")
    logger.info(f"  Model Size: {perf_metrics.model_size_mb:.2f} MB")
    
    metrics["performance"] = perf_metrics.to_dict()
    
    with open(output_dir / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    logger.info(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
