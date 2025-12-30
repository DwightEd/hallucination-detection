"""Configuration management for hallucination detection framework.

Design principles:
- Flat, simple configuration structures
- Layer selection unified via parse_layers() function
- All modes (all/first/last/first_n/last_n) converted to specific indices
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import re

from .types import ExtractionMode, StorageMode, TaskType


# ==============================================================================
# Layer Selection - Unified parsing
# ==============================================================================

def parse_layers(spec: Union[str, List[int], None], n_layers: int) -> List[int]:
    """Parse layer specification to concrete indices.
    
    This is THE function for layer selection. All modes are handled here,
    downstream code only deals with List[int].
    
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
    
    # Already a list
    if isinstance(spec, (list, tuple)):
        return [i for i in spec if 0 <= i < n_layers]
    
    spec = str(spec).strip().lower()
    
    # Keyword modes
    if spec == "all":
        return list(range(n_layers))
    if spec == "first":
        return [0]
    if spec == "last":
        return [n_layers - 1]
    
    # first_n:k or last_n:k
    match = re.match(r"(first|last)_n[:\s]*(\d+)", spec)
    if match:
        mode, n = match.groups()
        n = int(n)
        if mode == "first":
            return list(range(min(n, n_layers)))
        else:
            return list(range(max(0, n_layers - n), n_layers))
    
    # Try parsing as list string "[0, 1, 2]" or "0,1,2"
    try:
        cleaned = spec.strip("[]() ")
        if cleaned:
            indices = [int(x.strip()) for x in cleaned.split(",")]
            return [i for i in indices if 0 <= i < n_layers]
    except ValueError:
        pass
    
    return list(range(n_layers))


# ==============================================================================
# Configuration Dataclasses
# ==============================================================================

@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str = "ragtruth"
    path: str = "./data/RAGTruth"
    splits: List[str] = field(default_factory=lambda: ["test"])
    task_types: Optional[List[str]] = None
    max_samples: Optional[int] = None
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "Qwen/Qwen2.5-7B-Instruct"
    n_layers: int = 28
    n_heads: int = 28
    hidden_size: int = 3584
    dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    attn_implementation: str = "eager"  # MUST be "eager" for attention extraction
    load_in_4bit: bool = False
    load_in_8bit: bool = False


@dataclass
class FeaturesConfig:
    """Feature extraction configuration."""
    mode: str = "teacher_forcing"
    max_length: int = 4096
    attention_enabled: bool = True
    attention_layers: Union[str, List[int]] = "all"
    attention_storage: str = "diagonal"
    hidden_states_enabled: bool = True
    hidden_states_layers: Union[str, List[int]] = "last_n:4"
    hidden_states_pooling: str = "last_token"
    token_probs_enabled: bool = True
    token_probs_top_k: int = 10
    
    def get_attention_layers(self, n_layers: int) -> List[int]:
        return parse_layers(self.attention_layers, n_layers)
    
    def get_hidden_layers(self, n_layers: int) -> List[int]:
        return parse_layers(self.hidden_states_layers, n_layers)


@dataclass
class GenerationConfig:
    """Generation configuration."""
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.0


@dataclass
class MethodConfig:
    """Detection method configuration."""
    name: str = "lapeigvals"
    classifier: str = "logistic"
    cv_folds: int = 5
    val_split: float = 0.2
    random_seed: int = 42
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMAPIConfig:
    """LLM API configuration for judge."""
    provider: str = "qwen"
    model: str = "qwen-plus"
    api_key_env: str = "DASHSCOPE_API_KEY"
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: int = 60
    max_retries: int = 3


@dataclass
class Config:
    """Main configuration container."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    llm_api: LLMAPIConfig = field(default_factory=LLMAPIConfig)
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "./outputs"
    
    def get_output_path(self) -> Path:
        model_name = self.model.name.replace("/", "_")
        return Path(self.output_dir) / self.dataset.name / model_name / f"seed_{self.seed}"


# ==============================================================================
# Config Loading Utilities
# ==============================================================================

def load_config(cfg: DictConfig) -> Config:
    """Convert Hydra DictConfig to typed Config."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    return _dict_to_config(cfg_dict)


def _dict_to_config(d: Dict[str, Any]) -> Config:
    return Config(
        dataset=_dict_to_dataclass(DatasetConfig, d.get("dataset", {})),
        model=_dict_to_dataclass(ModelConfig, d.get("model", {})),
        features=_dict_to_dataclass(FeaturesConfig, d.get("features", {})),
        generation=_dict_to_dataclass(GenerationConfig, d.get("generation", {})),
        method=_dict_to_dataclass(MethodConfig, d.get("method", {})),
        llm_api=_dict_to_dataclass(LLMAPIConfig, d.get("llm_api", {})),
        seed=d.get("seed", 42),
        device=d.get("device", "cuda"),
        output_dir=d.get("output_dir", "./outputs"),
    )


def _dict_to_dataclass(cls, d: Dict[str, Any]):
    if not d:
        return cls()
    valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in d.items() if k in valid_fields}
    return cls(**filtered)


def save_config(cfg: Config, path: Path) -> None:
    """Save config to YAML file."""
    import yaml
    from dataclasses import asdict
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False, sort_keys=False)


def print_config(cfg: Config) -> None:
    """Print configuration."""
    from dataclasses import asdict
    import json
    print(json.dumps(asdict(cfg), indent=2, default=str))
