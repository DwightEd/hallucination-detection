"""Core module for hallucination detection framework.

Provides:
- Type definitions (Sample, ExtractedFeatures, Prediction, etc.)
- Configuration classes (DatasetConfig, ModelConfig, etc.)
- Component registry for extensibility
- Utility functions (logging, errors, progress tracking)
"""

from .types import (
    TaskType, SplitType, ExtractionMode, StorageMode,
    Sample, ExtractedFeatures, Prediction, HallucinationSpan, JudgeResult, EvalMetrics,
)

from .registry import (
    Registry, DATASETS, METHODS, MODELS, EXTRACTORS, LLM_APIS, list_available,
)

from .config import (
    parse_layers,
    DatasetConfig, ModelConfig, PromptConfig, QaPromptConfig, RAGTruthPromptConfig,
    FeaturesConfig, GenerationConfig, MethodConfig, LLMAPIConfig, Config,
    load_config_from_hydra, save_config, print_config,
)

from .utils import (
    setup_logging, get_logger, HallucDetectError, DatasetError,
    ModelError, FeatureError, MethodError, APIError,
    Progress, timer, ensure_dir, batch_iter, set_seed, get_device,
    import_class_from_path, import_function_from_path,
)

__all__ = [
    # Types
    "TaskType", "SplitType", "ExtractionMode", "StorageMode",
    "Sample", "ExtractedFeatures", "Prediction", "HallucinationSpan", "JudgeResult", "EvalMetrics",
    # Registry
    "Registry", "DATASETS", "METHODS", "MODELS", "EXTRACTORS", "LLM_APIS", "list_available",
    # Config
    "parse_layers",
    "DatasetConfig", "ModelConfig", "PromptConfig", "QaPromptConfig", "RAGTruthPromptConfig",
    "FeaturesConfig", "GenerationConfig", "MethodConfig", "LLMAPIConfig", "Config",
    "load_config_from_hydra", "save_config", "print_config",
    # Utils
    "setup_logging", "get_logger", "HallucDetectError", "DatasetError",
    "ModelError", "FeatureError", "MethodError", "APIError",
    "Progress", "timer", "ensure_dir", "batch_iter", "set_seed", "get_device",
    "import_class_from_path", "import_function_from_path",
]
