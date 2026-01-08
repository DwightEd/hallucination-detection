"""Data loading and processing utilities."""

from .base import (
    BaseDataset,
    JsonlDataset,
    JsonDataset,
    create_dataset,
    load_samples,
)

from .halueval import HaluEvalDataset
from .ragtruth import RAGTruthDataset
from .truthfulqa import TruthfulQADataset

from .splitter import (
    DatasetSplitter,
    auto_split_dataset,
    split_features,
)


def get_dataset(config, split=None, **kwargs):
    """Get dataset by config.

    Args:
        config: DatasetConfig instance
        split: Optional split to filter
        **kwargs: Additional arguments

    Returns:
        Dataset instance
    """
    from src.core import DATASETS

    name = config.name.lower()

    if DATASETS.contains(name):
        dataset = DATASETS.create(name, path=config.path, config=config, **kwargs)
    else:
        dataset = create_dataset(config.path, config=config, **kwargs)

    return dataset


__all__ = [
    "BaseDataset",
    "JsonlDataset",
    "JsonDataset",
    "create_dataset",
    "load_samples",
    "get_dataset",
    "HaluEvalDataset",
    "RAGTruthDataset",
    "TruthfulQADataset",
    "DatasetSplitter",
    "auto_split_dataset",
    "split_features",
]
