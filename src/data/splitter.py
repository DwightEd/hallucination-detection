"""Dataset splitting utilities for hallucination detection. 

Provides train/test splitting for datasets that don't have split information. 
Reads split ratio from dataset config, defaults to 0.9 (90% train, 10% test).

This module is designed to be run as a preprocessing step before feature extraction.
After splitting, all samples will have a `split` field (train/test), making them
compatible with datasets that already have split information. 

Example:
    from src.data import DatasetSplitter, auto_split_dataset, split_features

    # Split samples
    splitter = DatasetSplitter(train_ratio=0.9, seed=42)
    samples = splitter.split(samples)

    # Or use auto_split_dataset
    samples = auto_split_dataset(samples, train_ratio=0.9, seed=42)
"""
from __future__ import annotations
import random
import logging
from typing import List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from src.core import Sample, ExtractedFeatures, DatasetConfig

from src.core import SplitType

logger = logging. getLogger(__name__)


# Default train ratio if not specified in config
DEFAULT_TRAIN_RATIO = 0.9


@dataclass
class DatasetSplitter: 
    """Splitter for dividing datasets into train/test splits. 

    Attributes:
        train_ratio:  Ratio of samples to use for training (0. 0 to 1.0)
        seed: Random seed for reproducibility
        stratify: Whether to stratify by label (ensures balanced splits)
    """
    train_ratio:  float = DEFAULT_TRAIN_RATIO
    seed: int = 42
    stratify: bool = True

    def __post_init__(self):
        if not 0.0 < self.train_ratio < 1.0:
            raise ValueError(f"train_ratio must be between 0 and 1, got {self.train_ratio}")

    def split(self, samples: List["Sample"]) -> List["Sample"]: 
        """Split samples into train and test sets. 

        Updates the split field of each sample in-place style (returns new list).

        Args:
            samples: List of samples to split

        Returns:
            List of samples with split field set to TRAIN or TEST
        """
        if not samples:
            return []

        # Set random seed for reproducibility
        rng = random.Random(self.seed)

        if self.stratify:
            return self._stratified_split(samples, rng)
        else:
            return self._random_split(samples, rng)

    def _random_split(
        self,
        samples: List["Sample"],
        rng: random. Random
    ) -> List["Sample"]:
        """Simple random split."""
        indices = list(range(len(samples)))
        rng.shuffle(indices)

        n_train = int(len(samples) * self.train_ratio)
        train_indices = set(indices[:n_train])

        result = []
        for i, sample in enumerate(samples):
            split_type = SplitType.TRAIN if i in train_indices else SplitType.TEST
            new_sample = self._copy_sample_with_split(sample, split_type)
            result.append(new_sample)

        return result

    def _stratified_split(
        self,
        samples: List["Sample"],
        rng:  random.Random
    ) -> List["Sample"]:
        """Stratified split to maintain label distribution."""
        # Group by label
        by_label = {}
        for i, sample in enumerate(samples):
            label = sample.label if sample.label is not None else -1
            if label not in by_label:
                by_label[label] = []
            by_label[label].append(i)

        train_indices = set()

        # Split each label group proportionally
        for label, indices in by_label. items():
            rng.shuffle(indices)
            n_train = max(1, int(len(indices) * self.train_ratio))
            train_indices.update(indices[:n_train])

        result = []
        for i, sample in enumerate(samples):
            split_type = SplitType.TRAIN if i in train_indices else SplitType.TEST
            new_sample = self._copy_sample_with_split(sample, split_type)
            result.append(new_sample)

        return result

    def _copy_sample_with_split(self, sample: "Sample", split:  SplitType) -> "Sample": 
        """Create a copy of sample with updated split."""
        from src.core import Sample

        return Sample(
            id=sample.id,
            prompt=sample.prompt,
            response=sample.response,
            reference=sample.reference,
            label=sample.label,
            task_type=sample.task_type,
            split=split,
            metadata=sample.metadata. copy() if sample.metadata else {},
        )


def get_train_ratio_from_config(config: Optional["DatasetConfig"] = None) -> float:
    """Get train ratio from dataset config. 

    Args: 
        config: Dataset configuration object

    Returns: 
        Train ratio (defaults to DEFAULT_TRAIN_RATIO if not specified)
    """
    if config is None:
        return DEFAULT_TRAIN_RATIO

    # Check for train_ratio in config
    if hasattr(config, 'train_ratio') and config.train_ratio is not None: 
        return config. train_ratio

    # Check for split_ratio in config (alternative name)
    if hasattr(config, 'split_ratio') and config.split_ratio is not None: 
        return config. split_ratio

    return DEFAULT_TRAIN_RATIO


def auto_split_dataset(
    samples: List["Sample"],
    train_ratio: Optional[float] = None,
    seed:  int = 42,
    config: Optional["DatasetConfig"] = None,
    force_split: bool = False,
) -> List["Sample"]:
    """Automatically split dataset if samples don't have split information.

    This function checks if samples already have split information: 
    - If all samples have splits and force_split=False, return as-is
    - If no samples have splits, perform splitting
    - If some samples have splits and force_split=False, only split those without

    Args:
        samples: List of samples to potentially split
        train_ratio: Ratio for train split (0.0 to 1.0). If None, read from config.
        seed: Random seed for reproducibility
        config: Dataset configuration to read train_ratio from
        force_split: If True, re-split even if samples already have splits

    Returns:
        List of samples with split information
    """
    if not samples:
        return samples

    # Count samples with/without splits
    n_with_split = sum(1 for s in samples if s. split is not None)
    n_without_split = len(samples) - n_with_split

    logger.debug(f"Samples with split:  {n_with_split}, without:  {n_without_split}")

    # All samples have splits and we're not forcing
    if n_with_split == len(samples) and not force_split: 
        logger.info("All samples already have split information, skipping auto-split")
        return samples

    # Get train ratio
    if train_ratio is None:
        train_ratio = get_train_ratio_from_config(config)

    # Some samples have splits - only split those without
    if 0 < n_with_split < len(samples) and not force_split:
        logger. warning(
            f"Mixed split information:  {n_with_split} with splits, "
            f"{n_without_split} without.  Only splitting samples without split info."
        )
        return _partial_split(samples, train_ratio, seed)

    # No samples have splits OR force_split is True - do full split
    logger.info(f"Performing auto-split with train_ratio={train_ratio}, seed={seed}")

    splitter = DatasetSplitter(train_ratio=train_ratio, seed=seed, stratify=True)
    result = splitter.split(samples)

    n_train = sum(1 for s in result if s. split == SplitType. TRAIN)
    n_test = sum(1 for s in result if s. split == SplitType.TEST)
    logger.info(f"Split complete:  {n_train} train, {n_test} test")

    return result


def _partial_split(
    samples: List["Sample"],
    train_ratio: float,
    seed:  int,
) -> List["Sample"]: 
    """Split only samples that don't have split information."""
    from src.core import Sample

    # Separate samples with and without splits
    result = []
    without_split_indices = []

    for i, sample in enumerate(samples):
        if sample. split is not None: 
            result.append(sample)
        else:
            without_split_indices. append(i)
            result.append(None)  # Placeholder

    # Get samples without split
    samples_to_split = [samples[i] for i in without_split_indices]

    if not samples_to_split:
        return [s for s in result if s is not None]

    # Split them
    splitter = DatasetSplitter(train_ratio=train_ratio, seed=seed, stratify=True)
    split_samples = splitter.split(samples_to_split)

    # Put back
    for idx, split_sample in zip(without_split_indices, split_samples):
        result[idx] = split_sample

    n_train = sum(1 for s in split_samples if s. split == SplitType.TRAIN)
    n_test = sum(1 for s in split_samples if s.split == SplitType.TEST)
    logger.info(f"Partial split: {n_train} train, {n_test} test (from {len(samples_to_split)} samples)")

    return result


def split_features(
    features_list: List["ExtractedFeatures"],
    samples: List["Sample"],
) -> Tuple[
    List["ExtractedFeatures"],
    List[int],
    List["ExtractedFeatures"],
    List[int]
]:
    """Split features based on existing sample splits.

    This function assumes samples already have split information. 
    Use auto_split_dataset first if samples don't have splits.

    Args:
        features_list: List of extracted features
        samples:  List of corresponding samples (must have split field)

    Returns:
        Tuple of (train_features, train_labels, test_features, test_labels)
    """
    if len(features_list) != len(samples):
        raise ValueError(
            f"features_list and samples must have same length, "
            f"got {len(features_list)} and {len(samples)}"
        )

    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    for feat, sample in zip(features_list, samples):
        label = feat.label if feat. label is not None else (sample.label if sample.label is not None else 0)

        if sample.split == SplitType.TRAIN: 
            train_features. append(feat)
            train_labels. append(label)
        else:  # TEST or VALIDATION or None -> treat as test
            test_features.append(feat)
            test_labels.append(label)

    return train_features, train_labels, test_features, test_labels