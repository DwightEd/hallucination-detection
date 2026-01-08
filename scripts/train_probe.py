#!/usr/bin/env python3
"""Train probe/classifier for hallucination detection.

新目录结构:
- 特征: {features_dir}/{dataset}/{model}/seed_{seed}/{task_type}/
- 模型: {models_dir}/{dataset}/{model}/seed_{seed}/{task_type}/{method}/probe/
"""
import sys
import re
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import (
    MethodConfig, ExtractedFeatures, SplitType,
    set_seed, setup_logging,
)
from src.methods import create_method

logger = logging.getLogger(__name__)


# =============================================================================
# Task Type 解析函数（与 generate_activations.py 保持一致）
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
    """
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


def find_features_dir(cfg: DictConfig) -> Path:
    """Find the features directory based on config.
    
    新目录结构: {base}/{dataset}/{model}/seed_{seed}/{task_type}/
    旧目录结构: {base}/{dataset}_{task_suffix}/{model}/seed_{seed}/
    
    会尝试两种格式以保持向后兼容。
    """
    base_dir = Path(cfg.features_dir)
    dataset_name = cfg.dataset.name
    task_suffix = get_task_suffix(cfg)
    model_name = get_model_short_name(cfg)
    seed_str = f"seed_{cfg.seed}"

    # 新目录结构: {base}/{dataset}/{model}/seed_{seed}/{task_type}/
    features_dir_new = base_dir / dataset_name / model_name / seed_str / task_suffix
    
    # 旧目录结构: {base}/{dataset}_{task_suffix}/{model}/seed_{seed}/
    features_dir_old = base_dir / f"{dataset_name}_{task_suffix}" / model_name / seed_str
    
    # 优先使用新结构
    if features_dir_new.exists():
        logger.info(f"Using new directory structure: {features_dir_new}")
        return features_dir_new
    
    # 回退到旧结构
    if features_dir_old.exists():
        logger.info(f"Using legacy directory structure: {features_dir_old}")
        return features_dir_old
    
    # 尝试搜索
    for pattern_dir in [
        base_dir / dataset_name / model_name,
        base_dir / f"{dataset_name}_{task_suffix}" / model_name
    ]:
        if pattern_dir.exists():
            for subdir in pattern_dir.iterdir():
                if subdir.is_dir() and subdir.name.startswith("seed_"):
                    # 检查新结构
                    task_dir = subdir / task_suffix
                    if task_dir.exists():
                        logger.warning(f"Using found features directory: {task_dir}")
                        return task_dir
                    # 检查旧结构
                    if (subdir / "features.pkl").exists():
                        logger.warning(f"Using found features directory: {subdir}")
                        return subdir
    
    # 返回新结构路径（即使不存在，后面会报错）
    return features_dir_new


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
    
    # 新格式：从 features/ 目录加载
    features_subdir = features_dir / "features"
    if features_subdir.exists():
        return load_features_from_dir(features_dir)
    
    # 尝试直接从目录加载（如果目录本身就是特征目录）
    if (features_dir / "metadata.json").exists():
        return load_features_from_dir(features_dir)
    
    raise FileNotFoundError(f"No features found in {features_dir}")


def load_features_from_dir(features_dir: Path) -> tuple:
    """从新格式目录加载特征。
    
    新格式目录结构:
    - metadata.json: 元数据
    - answers.json: 样本信息
    - features/: 特征文件
        - attn_diags.pt
        - laplacian_diags.pt
        - hidden_states.pt
        - etc.
    """
    import json
    from src.core import Sample, TaskType, SplitType
    
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
        features_subdir = features_dir  # 直接在目录中查找
    
    feature_files = {
        "attn_diags": "attn_diags.pt",
        "laplacian_diags": "laplacian_diags.pt",
        "attn_entropy": "attn_entropy.pt",
        "hidden_states": "hidden_states.pt",
        "token_probs": "token_probs.pt",
        "full_attentions": "full_attentions.pt",  # 存储使用复数命名
    }
    
    loaded_features = {}
    for key, filename in feature_files.items():
        filepath = features_subdir / filename
        if filepath.exists():
            loaded_features[key] = torch.load(filepath, weights_only=False)
            logger.info(f"Loaded {key}: {len(loaded_features[key])} samples")
    
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
        for key, data_dict in loaded_features.items():
            if isinstance(data_dict, dict):
                sample_id = sample_ids[i] if i < len(sample_ids) else str(i)
                if sample_id in data_dict:
                    # 将存储键映射回类属性名
                    class_attr = key if key != "full_attentions" else "full_attention"
                    sample_features[class_attr] = data_dict[sample_id]
        
        # 获取样本信息
        sample = samples[i] if i < len(samples) else None
        label = int(labels[i]) if i < len(labels) else 0
        
        # 创建 ExtractedFeatures 对象
        feat = ExtractedFeatures(
            sample_id=sample_ids[i] if i < len(sample_ids) else str(i),
            prompt_len=sample.metadata.get("prompt_len", 0) if sample else 0,
            response_len=sample.metadata.get("response_len", 0) if sample else 0,
            label=label,
            attn_diags=sample_features.get("attn_diags"),
            laplacian_diags=sample_features.get("laplacian_diags"),
            attn_entropy=sample_features.get("attn_entropy"),
            hidden_states=sample_features.get("hidden_states"),
            token_probs=sample_features.get("token_probs"),
            full_attention=sample_features.get("full_attention"),  # 使用单数形式
            metadata={
                "task_type_str": sample.task_type.value if sample and sample.task_type else "unknown",
            } if sample else {},
        )
        features_list.append(feat)
    
    return features_list, samples


def split_by_dataset_split(
    features_list: List[ExtractedFeatures],
    samples: list
) -> tuple:
    """Split features by dataset split (train/test).

    Samples should already have split field set.
    """
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    for feat, sample in zip(features_list, samples):
        label = feat.label if feat.label is not None else (sample.label if sample.label is not None else 0)

        if sample.split and sample.split == SplitType.TRAIN: 
            train_features.append(feat)
            train_labels.append(label)
        else:  # test or validation or None
            test_features.append(feat)
            test_labels.append(label)

    return train_features, train_labels, test_features, test_labels


def build_output_dir(cfg: DictConfig) -> Path:
    """Build output directory for trained model.
    
    新目录结构: {base}/{dataset}/{model}/seed_{seed}/{task_type}/{method}/probe/
    """
    base_dir = Path(cfg.models_dir)
    dataset_name = cfg.dataset.name
    task_suffix = get_task_suffix(cfg)
    model_name = get_model_short_name(cfg)
    method_name = cfg.method.name

    # 新目录结构
    output_dir = base_dir / dataset_name / model_name / f"seed_{cfg.seed}" / task_suffix / method_name / "probe"
    return output_dir


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for training."""

    setup_logging(level=logging.INFO)
    set_seed(cfg.seed)

    logger.info("=" * 60)
    logger.info("Train Probe")
    logger.info("=" * 60)
    logger.info(f"Method: {cfg.method.name}")
    logger.info(f"Classifier: {cfg.method.classifier}")
    logger.info(f"Task types: {get_task_types_from_config(cfg)}")

    # Find features
    features_dir = find_features_dir(cfg)
    logger.info(f"Features directory: {features_dir}")

    if not features_dir.exists():
        logger.error(f"Features directory not found: {features_dir}")
        logger.error("Please run generate_activations.py first")
        return

    # Load features
    try:
        features_list, samples = load_features(features_dir)
        logger.info(f"Loaded {len(features_list)} feature sets")
    except Exception as e:
        logger.error(f"Failed to load features: {e}")
        import traceback
        traceback.print_exc()
        return

    # Split by dataset split
    train_features, train_labels, test_features, test_labels = split_by_dataset_split(
        features_list, samples
    )
    logger.info(f"Train: {len(train_features)}, Test: {len(test_features)}")

    # 如果没有 split 信息，使用所有数据训练
    if len(train_features) == 0: 
        logger.warning("No train split found, using all data for training")
        train_features = features_list
        train_labels = [f.label if f.label is not None else 0 for f in features_list]

    n_pos = sum(1 for l in train_labels if l == 1)
    n_neg = sum(1 for l in train_labels if l == 0)
    logger.info(f"Train labels: {n_pos} positive, {n_neg} negative")

    # Build method config
    method_config = MethodConfig(**OmegaConf.to_container(cfg.method, resolve=True))

    # Create method
    logger.info(f"Creating method: {method_config.name}")
    method = create_method(method_config.name, config=method_config)

    # Train
    logger.info("Training...")
    try:
        metrics = method.fit(train_features, train_labels, cv=True)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return

    logger.info("Training complete!")
    logger.info(f"Metrics: {metrics}")

    # Build output directory
    output_dir = build_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / "model.pkl"
    method.save(model_path)
    logger.info(f"Model saved to {model_path}")

    # Save metrics
    metrics_path = output_dir / "train_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save config
    OmegaConf.save(cfg, output_dir / "config.yaml")

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
