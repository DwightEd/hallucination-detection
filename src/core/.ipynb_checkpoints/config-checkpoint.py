"""Configuration management using Pydantic models.

Design aligned with lapeigvals for compatibility.
All configs use Pydantic BaseModel for validation and serialization.

Key Changes:
- Added `models` field to DatasetConfig for model filtering
- Added `store_full_attention` to FeaturesConfig with safety controls
- Added `multi_gpu` to ModelConfig for multi-GPU support
- Added `required_features` to MethodConfig
- Unified `level` field (replaces training_level and classification_level)
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Literal
import re

from pydantic import BaseModel, Field, model_validator

from .types import ExtractionMode, StorageMode, TaskType


# ==============================================================================
# Layer Selection - Unified parsing
# ==============================================================================

def parse_layers(spec: Union[str, List[int], None], n_layers: int) -> List[int]:
    """Parse layer specification to concrete indices.

    Args:
        spec: Layer specification, can be:
            - "all": All layers [0, 1, ..., n_layers-1]
            - "first": First layer only [0]
            - "last": Last layer only [n_layers-1]
            - "first_n:k": First k layers [0, 1, ..., k-1]
            - "last_n:k": Last k layers [n_layers-k, ..., n_layers-1]
            - [0, 4, 8, ...]: Explicit list of indices
            - "[0, 4, 8]": String representation of list
            - None: Defaults to "all"
        n_layers: Total number of layers in model

    Returns:
        List of layer indices (0-indexed, validated)
    """
    if spec is None:
        spec = "all"

    if isinstance(spec, (list, tuple)):
        return [i for i in spec if 0 <= i < n_layers]

    spec = str(spec).strip().lower()

    if spec == "all":
        return list(range(n_layers))
    if spec == "first":
        return [0]
    if spec == "last":
        return [n_layers - 1]

    match = re.match(r"(first|last)_n[:\s]*(\d+)", spec)
    if match:
        mode, n = match.groups()
        n = int(n)
        if mode == "first":
            return list(range(min(n, n_layers)))
        else:
            return list(range(max(0, n_layers - n), n_layers))

    try:
        cleaned = spec.strip("[]() ")
        if cleaned:
            indices = [int(x.strip()) for x in cleaned.split(",")]
            return [i for i in indices if 0 <= i < n_layers]
    except ValueError:
        pass

    return list(range(n_layers))


# ==============================================================================
# Pydantic Configuration Models
# ==============================================================================

class CacheConfig(BaseModel, extra="forbid"):
    """Cache configuration for dataset loading."""
    enabled: bool = False
    path: Optional[str] = None


class DatasetConfig(BaseModel, extra="ignore"):
    """Base dataset configuration.
    
    使用 extra="ignore" 允许子类或 YAML 配置包含额外字段。
    每个数据集可以继承此基类并添加特定字段。
    
    Attributes:
        name: Dataset name (required). Options: 'ragtruth', 'halueval', 'truthfulqa'
        path: Path to dataset files. Required for local datasets.
        cls_path: Custom dataset class path (optional)
        subset: Dataset subset name (optional)
        
        splits: Filter by data splits. Options: ['train'], ['test'], ['train', 'test']
        task_types: Filter by task types.
        max_samples: Maximum number of samples to load (optional)
        
        max_answer_tokens: Maximum tokens in answer (default: 256)
        target_column_name: Name of target column (default: 'answer')
        test_split_name: Custom name for test split (optional)
        
        train_ratio: Train/test split ratio (default: 0.9)
        split_seed: Seed for split randomization (optional)
        force_split: Force re-split even if samples have splits (default: False)
    """
    name: str
    path: Optional[Path] = None
    cls_path: Optional[str] = None
    subset: Optional[str] = None

    # Filtering options
    splits: Optional[List[str]] = None
    task_types: Optional[List[str]] = None
    models: Optional[List[str]] = None 
    max_samples: Optional[int] = None

    max_answer_tokens: int = 256
    target_column_name: str = "answer"
    test_split_name: Optional[str] = None

    # Train/test split ratio (for datasets without split information)
    train_ratio: Optional[float] = 0.9
    split_seed: Optional[int] = None
    force_split: bool = False


class RAGTruthDatasetConfig(DatasetConfig):
    """RAGTruth specific dataset configuration.
    
    Additional Attributes:
        models: Filter by source models (e.g., ['gpt-4', 'llama-2-7b-chat'])
        exclude_quality: Quality labels to exclude (default: ['incorrect_refusal', 'truncated'])
        include_metadata: Whether to include span annotations (default: True)
        validate_data: Whether to validate data integrity (default: True)
        cache: Cache configuration
    """
    models: Optional[List[str]] = None
    exclude_quality: Optional[List[str]] = None
    include_metadata: bool = True
    validate_data: bool = True
    cache: Optional[CacheConfig] = None


class HaluEvalDatasetConfig(DatasetConfig):
    """HaluEval specific dataset configuration.
    
    Additional Attributes:
        categories: Filter by categories (e.g., ['qa', 'dialogue', 'summarization'])
        include_knowledge: Whether to include knowledge context (default: True)
    """
    categories: Optional[List[str]] = None
    include_knowledge: bool = True


class TruthfulQADatasetConfig(DatasetConfig):
    """TruthfulQA specific dataset configuration.
    
    Additional Attributes:
        include_best_answer: Whether to include best answer (default: True)
        include_incorrect_answers: Whether to include incorrect answers (default: False)
    """
    include_best_answer: bool = True
    include_incorrect_answers: bool = False


# Registry for dataset configs - maps dataset name to config class
DATASET_CONFIG_REGISTRY: Dict[str, type] = {
    "ragtruth": RAGTruthDatasetConfig,
    "halueval": HaluEvalDatasetConfig,
    "truthfulqa": TruthfulQADatasetConfig,
}


def get_dataset_config(config_dict: Dict[str, Any]) -> DatasetConfig:
    """Factory function to create the appropriate dataset config.
    
    Args:
        config_dict: Dictionary containing dataset configuration
        
    Returns:
        Appropriate DatasetConfig subclass instance
        
    Example:
        >>> cfg_dict = {"name": "ragtruth", "models": ["gpt-4"], ...}
        >>> config = get_dataset_config(cfg_dict)  # Returns RAGTruthDatasetConfig
    """
    name = config_dict.get("name", "").lower()
    config_cls = DATASET_CONFIG_REGISTRY.get(name, DatasetConfig)
    return config_cls(**config_dict)


class MultiGPUConfig(BaseModel, extra="forbid"):
    """Multi-GPU configuration.
    
    Attributes:
        enabled: Whether to enable multi-GPU (default: True if multiple GPUs available)
        strategy: Distribution strategy. Options: 'auto', 'balanced', 'sequential'
        max_memory: Maximum memory per GPU in GB (optional)
        memory_fraction: Fraction of GPU memory to use (0.0-1.0, optional)
        main_device: Main device for non-distributed operations (default: 0)
    """
    enabled: bool = True
    strategy: Literal["auto", "balanced", "sequential"] = "auto"
    max_memory: Optional[Dict[int, str]] = None  # e.g., {0: "20GB", 1: "20GB"}
    memory_fraction: Optional[float] = None  # e.g., 0.9 = use 90% of each GPU
    main_device: int = 0


class ModelConfig(BaseModel, extra="forbid"):
    """Model configuration.
    
    Attributes:
        name: Model name or HuggingFace path (required)
        short_name: Short name for outputs (auto-generated if not provided)
        
        n_layers: Number of transformer layers (auto-detected from model)
        n_heads: Number of attention heads (auto-detected from model)
        hidden_size: Hidden dimension size (default: 4096)
        context_size: Maximum context length (default: 8192)
        
        dtype: Model data type. Options: 'float32', 'float16', 'bfloat16'
        device_map: Device mapping strategy. Options: 'auto', 'cuda:0', etc.
        trust_remote_code: Trust remote code when loading (default: True)
        attn_implementation: Attention implementation. Options: 'eager', 'flash_attention_2'
        load_in_4bit: Load model in 4-bit quantization (default: False)
        load_in_8bit: Load model in 8-bit quantization (default: False)
        
        tokenizer_name: Tokenizer name (defaults to model name)
        tokenizer_padding_side: Padding side. Options: 'left', 'right'
        
        quantization: Custom quantization config (optional)
        multi_gpu: Multi-GPU configuration (optional)
    """
    name: str
    short_name: Optional[str] = None
    
    n_layers: int = 32
    n_heads: int = 32
    hidden_size: int = 4096
    context_size: int = 8192

    dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    attn_implementation: str = "eager"
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    tokenizer_name: Optional[str] = None
    tokenizer_padding_side: Literal["left", "right"] = "left"

    quantization: Optional[Dict[str, Any]] = None
    multi_gpu: Optional[MultiGPUConfig] = None  # NEW: Multi-GPU config

    @model_validator(mode="after")
    def set_defaults(self) -> "ModelConfig":
        if self.tokenizer_name is None:
            self.tokenizer_name = self.name
        if self.short_name is None:
            self.short_name = self.name.split("/")[-1].replace("-", "_").lower()
        return self


class PromptConfig(BaseModel, extra="forbid"):
    """Base prompt configuration."""
    name: str = "default"
    cls_path: Optional[str] = None
    content: str = "{question}"


class QaPromptConfig(PromptConfig, extra="forbid"):
    """QA prompt configuration."""
    question_key: str = "question"
    context_key: Optional[str] = None
    num_few_shot_examples: Optional[int] = None


class RAGTruthPromptConfig(PromptConfig, extra="forbid"):
    """RAGTruth specific prompt configuration."""
    use_original_prompt: bool = True
    question_key: str = "prompt"
    context_key: Optional[str] = None


class FeaturesConfig(BaseModel, extra="forbid"):
    """Feature extraction configuration.
    
    Attributes:
        mode: Extraction mode. Options: 'teacher_forcing', 'generation'
        stored_features: Features to store. Options: 'attention_diags', 'all'
        
        attention_enabled: Enable attention extraction (default: True)
        attention_layers: Layers to extract. Options: 'all', 'last', 'first_n:8', [0,4,8]
        attention_storage: Storage mode. Options: 'diagonal', 'full'
        store_full_attention: Store full attention matrices (default: False)
                             ⚠️ WARNING: Requires ~8-128GB per sample!
        extract_attention_row_sums: Extract attention row sums for Laplacian (default: True)
        
        hidden_states_enabled: Enable hidden state extraction (default: True)
        hidden_states_layers: Layers. Options: 'last', 'last_n:4', [28,29,30,31]
        hidden_states_pooling: Pooling method. Options: 'last_token', 'mean', 'max', 'none'
        
        token_probs_enabled: Enable token probability extraction (default: True)
        token_probs_top_k: Number of top-k probs to store (default: 10)
        
        max_length: Maximum sequence length (default: 4096)
        batch_size: Batch size (default: 1, recommended for memory efficiency)
        
        use_fp16: Use mixed precision (default: True)
        clear_cache_per_sample: Clear GPU cache after each sample (default: True)
        checkpoint_interval: Save checkpoint every N samples (default: 100)
    """
    mode: str = "teacher_forcing"
    stored_features: str = "attention_diags"

    attention_enabled: bool = True
    attention_layers: Union[str, List[int]] = "all"
    attention_storage: str = "diagonal"
    store_full_attention: bool = False  # NEW: Control full attention storage (OOM risk!)
    extract_attention_row_sums: bool = True  # NEW: For lapeigvals method

    hidden_states_enabled: bool = True
    hidden_states_layers: Optional[Union[str, List[int]]] = "last_n:4"
    hidden_states_pooling: Optional[str] = "last_token"

    token_probs_enabled: bool = True
    token_probs_top_k: int = 10

    max_length: int = 4096
    batch_size: int = 1

    # Advanced options
    use_fp16: bool = True
    clear_cache_per_sample: bool = True
    checkpoint_interval: int = 100

    def get_attention_layers(self, n_layers: int) -> List[int]:
        if not self.attention_enabled:
            return []
        return parse_layers(self.attention_layers, n_layers)

    def get_hidden_layers(self, n_layers: int) -> List[int]:
        if not self.hidden_states_enabled:
            return []
        return parse_layers(self.hidden_states_layers, n_layers)
    
    def should_store_full_attention(self) -> bool:
        """Check if full attention should be stored.
        
        This is a safety method - full attention requires enormous memory
        (~8-128GB per sample depending on sequence length).
        
        Returns:
            True only if explicitly enabled via store_full_attention
        """
        return self.store_full_attention and self.attention_enabled


class GenerationConfig(BaseModel, extra="forbid"):
    """Generation configuration."""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.0


class MethodFeaturesConfig(BaseModel, extra="forbid"):
    """Method feature configuration (v6.3+).
    
    Defines base features (extracted from model) and derived features (computed from base).
    """
    base: List[str] = Field(default_factory=list)
    derived: Dict[str, Any] = Field(default_factory=dict)


class MethodConfig(BaseModel, extra="ignore"):
    """Detection method configuration.
    
    Attributes:
        name: Method name. Options: 'lapeigvals', 'entropy', 'hypergraph', 'lookback_lens'
        cls_path: Custom method class path (optional)
        
        classifier: Classifier type. Options: 'logistic', 'svm', 'rf', 'xgboost'
        cv_folds: Number of cross-validation folds (default: 5)
        random_seed: Random seed for reproducibility (default: 42)
        
        level: Training/classification granularity (unified field). Options:
            - 'sample': Sample-level labels only (default)
            - 'token': Token-level labels (requires hallucination_labels)
            - 'both': Method handles both levels internally
        
        params: Method-specific parameters
        features: Feature configuration (base + derived) - v6.3+
        required_features: Required features for this method (legacy, optional)
    
    Note: Uses extra="ignore" to allow additional fields from YAML configs for forward compatibility.
    """
    name: str = "lapeigvals"
    cls_path: Optional[str] = None

    classifier: str = "logistic"
    cv_folds: int = 5
    random_seed: int = 42
    
    # Unified level field (replaces training_level and classification_level)
    level: str = "sample"  # "sample" | "token" | "both"

    params: Dict[str, Any] = Field(default_factory=dict)
    
    # v6.3+ feature configuration
    features: Optional[MethodFeaturesConfig] = None
    
    # Legacy field (for backward compatibility)
    required_features: Optional[Dict[str, bool]] = None
    
    def get_base_features(self) -> List[str]:
        """Get list of required base features."""
        if self.features and self.features.base:
            return self.features.base
        return []
    
    def get_derived_features(self) -> Dict[str, Any]:
        """Get derived feature specifications."""
        if self.features and self.features.derived:
            return self.features.derived
        return {}


class LLMAPIConfig(BaseModel, extra="forbid"):
    """LLM API configuration for judge."""
    provider: str = "qwen"
    model: str = "qwen-plus"
    api_key_env: str = "DASHSCOPE_API_KEY"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    system_prompt: Optional[str] = None

    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: int = 60
    max_retries: int = 3
    rate_limit: int = 60


class Config(BaseModel):
    """Main configuration container."""
    dataset: DatasetConfig
    model: ModelConfig
    prompt: Union[PromptConfig, QaPromptConfig, RAGTruthPromptConfig] = Field(
        default_factory=PromptConfig
    )
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    generation_config: GenerationConfig = Field(default_factory=GenerationConfig)
    method: MethodConfig = Field(default_factory=lambda: MethodConfig(name="lapeigvals"))
    llm_api: LLMAPIConfig = Field(default_factory=LLMAPIConfig)

    seed: int = 42
    device: str = "cuda"
    output_dir: str = "outputs"
    results_dir: str = "outputs/results"
    features_dir: str = "outputs/features"
    models_dir: str = "outputs/models"

    def get_output_path(self) -> Path:
        """Get output path based on config."""
        return Path(self.results_dir) / self.dataset.name / self.model.short_name


# ==============================================================================
# Config Loading Utilities
# ==============================================================================

def load_config_from_hydra(cfg) -> Config:
    """Convert Hydra DictConfig to typed Config."""
    from omegaconf import OmegaConf
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    return Config(**cfg_dict)


def save_config(cfg: Config, path: Path) -> None:
    """Save config to YAML file."""
    import yaml
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(cfg.model_dump(), f, default_flow_style=False, sort_keys=False)


def print_config(cfg: Config) -> None:
    """Print configuration."""
    import json
    print(json.dumps(cfg.model_dump(), indent=2, default=str))