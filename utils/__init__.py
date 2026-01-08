"""Utility modules for hallucination detection framework.

This package provides:
- checkpoint: Checkpoint management for resume support
- async_saver: Asynchronous feature saving
- feature_manager: Unified feature extraction management
"""

from .checkpoint import (
    CheckpointManager,
    CheckpointState,
    get_pending_samples,
)
from .async_saver import (
    AsyncFeatureSaver,
    MemoryEfficientSaver,
    create_feature_saver,
)
from .feature_manager import (
    FeatureManager,
    FeatureRequirements,
    get_method_requirements,
    compute_union_requirements,
    validate_features_for_method,
    METHOD_FEATURE_REQUIREMENTS,
)

__all__ = [
    # Checkpoint
    "CheckpointManager",
    "CheckpointState",
    "get_pending_samples",
    # Async saver
    "AsyncFeatureSaver",
    "MemoryEfficientSaver",
    "create_feature_saver",
    # Feature manager
    "FeatureManager",
    "FeatureRequirements",
    "get_method_requirements",
    "compute_union_requirements",
    "validate_features_for_method",
    "METHOD_FEATURE_REQUIREMENTS",
]
