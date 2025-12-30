"""Core module for hallucination detection framework."""

from .types import (
    TaskType, SplitType, ExtractionMode, StorageMode,
    Sample, ExtractedFeatures, Prediction, JudgeResult, EvalMetrics,
)

from .registry import (
    Registry, DATASETS, METHODS, MODELS, EXTRACTORS, LLM_APIS, list_available,
)

from .config import (
    parse_layers, DatasetConfig, ModelConfig, FeaturesConfig,
    GenerationConfig, MethodConfig, LLMAPIConfig, Config,
    load_config, save_config, print_config,
)

from .utils import (
    setup_logging, get_logger, HallucDetectError, DatasetError,
    ModelError, FeatureError, MethodError, APIError,
    Progress, timer, ensure_dir, batch_iter, set_seed, get_device,
)

__all__ = [
    "TaskType", "SplitType", "ExtractionMode", "StorageMode",
    "Sample", "ExtractedFeatures", "Prediction", "JudgeResult", "EvalMetrics",
    "Registry", "DATASETS", "METHODS", "MODELS", "EXTRACTORS", "LLM_APIS", "list_available",
    "parse_layers", "DatasetConfig", "ModelConfig", "FeaturesConfig",
    "GenerationConfig", "MethodConfig", "LLMAPIConfig", "Config",
    "load_config", "save_config", "print_config",
    "setup_logging", "get_logger", "HallucDetectError", "DatasetError",
    "ModelError", "FeatureError", "MethodError", "APIError",
    "Progress", "timer", "ensure_dir", "batch_iter", "set_seed", "get_device",
]
