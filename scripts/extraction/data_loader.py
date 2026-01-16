"""Data loading module for feature extraction.

重构版本：使用统一的 parse_task_types 函数。

负责从 splits 目录加载样本，支持：
- Task type 过滤
- Model 过滤
- 标准化 Sample 对象创建
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List, Optional, Any

from omegaconf import DictConfig

from src.core import Sample, TaskType, SplitType, parse_task_types

logger = logging.getLogger(__name__)


def get_task_types_from_config(cfg: DictConfig) -> Optional[List[str]]:
    """从配置中获取 task_types 列表。
    
    使用统一的 parse_task_types 函数。
    
    按优先级尝试：
    1. cfg.dataset.task_type
    2. cfg.dataset.task_types
    
    Args:
        cfg: Hydra 配置对象
        
    Returns:
        task_types 列表或 None
    """
    task_type = cfg.dataset.get('task_type', None)
    if task_type is not None:
        parsed = parse_task_types(task_type)
        if parsed:
            return parsed
    
    task_types = cfg.dataset.get('task_types', None)
    return parse_task_types(task_types)


def load_samples_from_splits(
    dataset_name: str,
    task_types_filter: Optional[List[str]] = None,
    model_filter: Optional[List[str]] = None,
    split: str = "train"
) -> List[Sample]:
    """从 outputs/splits/{dataset}/train.json 或 test.json 加载样本。
    
    Args:
        dataset_name: 数据集名称
        task_types_filter: 要保留的 task type 列表
        model_filter: 要保留的模型名称列表（部分匹配）
        split: "train" 或 "test"
        
    Returns:
        Sample 对象列表
        
    Raises:
        FileNotFoundError: 如果 split 文件不存在
    """
    splits_dir = Path("outputs/splits") / dataset_name
    split_file = splits_dir / f"{split}.json"
    
    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}\n"
            f"Please run 'dvc repro split_dataset' first."
        )
    
    logger.info(f"Loading samples from {split_file}")
    
    with open(split_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    for item in data:
        # Task type 过滤
        item_task_type = item.get('task_type', '')
        if task_types_filter and item_task_type not in task_types_filter:
            continue
        
        # Model 过滤（部分匹配）
        if model_filter:
            item_model = item.get('model', '')
            matched = any(
                filter_model.lower() in item_model.lower()
                for filter_model in model_filter
            )
            if not matched:
                continue
        
        # 解析 TaskType 枚举
        task_type = None
        if item_task_type:
            try:
                task_type = TaskType(item_task_type)
            except (ValueError, KeyError):
                pass
        
        # 解析 SplitType 枚举
        split_type = None
        split_str = item.get('split', '')
        if split_str:
            split_str_lower = split_str.lower()
            if split_str_lower in ['train', 'training']:
                split_type = SplitType.TRAIN
            elif split_str_lower in ['test', 'testing', 'val', 'validation']:
                split_type = SplitType.TEST
        
        metadata = item.get('metadata', {})
        sample = Sample(
            id=str(item.get('id', '')),
            prompt=item.get('prompt', ''),
            response=item.get('response', ''),
            label=item.get('label', 0),
            task_type=task_type,
            split=split_type,
            metadata={
                'source_model': item.get('model', ''),
                'hallucination_spans': metadata.get('hallucination_spans', []),
                'source_id': metadata.get('source_id', ''),
            }
        )
        samples.append(sample)
    
    logger.info(f"Loaded {len(samples)} samples from {split} split (after filtering)")
    return samples


def log_model_filter_info(cfg: DictConfig, samples: List[Sample]) -> None:
    """记录模型过滤信息。
    
    Args:
        cfg: 配置对象
        samples: 样本列表
    """
    source_models = set()
    for sample in samples:
        src_model = sample.metadata.get("source_model")
        if src_model:
            source_models.add(src_model)
    
    model_filter = cfg.dataset.get("models", None)
    
    logger.info("-" * 50)
    logger.info("Data Source Models")
    logger.info("-" * 50)
    
    if model_filter:
        logger.info(f"Filter applied: {model_filter}")
    else:
        logger.info("Filter: None (all models)")
    
    logger.info(f"Models in loaded data: {sorted(source_models)}")
    
    model_counts = {}
    for sample in samples:
        src_model = sample.metadata.get("source_model", "unknown")
        model_counts[src_model] = model_counts.get(src_model, 0) + 1
    
    for model, count in sorted(model_counts.items()):
        logger.info(f"  {model}: {count} samples")
    
    logger.info("-" * 50)
