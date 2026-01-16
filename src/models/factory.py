"""Model factory for loading and managing models.

Provides unified interface for model loading following lapeigvals pattern.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import logging

import torch

from src.core import ModelConfig
from src.core.utils import import_class_from_path
from .loader import LoadedModel, load_model, ModelManager, get_model_manager

logger = logging.getLogger(__name__)


def get_model(config: ModelConfig, device: Optional[str] = None) -> LoadedModel:
    """Get model instance from config.
    
    Uses global model manager for caching.
    
    Args:
        config: Model configuration
        device: Override device
        
    Returns:
        LoadedModel instance
    """
    manager = get_model_manager()
    return manager.get(config, device)


def create_model(config: ModelConfig, device: Optional[str] = None) -> LoadedModel:
    """Create new model instance (no caching).
    
    Args:
        config: Model configuration
        device: Override device
        
    Returns:
        LoadedModel instance
    """
    return load_model(config, device)


def get_model_from_hydra(cfg) -> LoadedModel:
    """Get model from Hydra config.
    
    Args:
        cfg: Hydra DictConfig with model section
        
    Returns:
        LoadedModel instance
    """
    from omegaconf import OmegaConf
    
    model_dict = OmegaConf.to_container(cfg.model, resolve=True)
    config = ModelConfig(**model_dict)
    
    return get_model(config)


def unload_model(model_name: str) -> None:
    """Unload model from cache.
    
    Args:
        model_name: Model name or path
    """
    manager = get_model_manager()
    manager.unload(model_name)


def unload_all_models() -> None:
    """Unload all cached models."""
    manager = get_model_manager()
    manager.unload_all()


def get_model_info(config: ModelConfig) -> Dict[str, Any]:
    """Get model architecture info without loading.
    
    Args:
        config: Model configuration
        
    Returns:
        Dict with model info
    """
    from transformers import AutoConfig
    
    try:
        model_config = AutoConfig.from_pretrained(
            config.name,
            trust_remote_code=config.trust_remote_code,
        )
        
        return {
            "name": config.name,
            "n_layers": getattr(model_config, "num_hidden_layers", config.n_layers),
            "n_heads": getattr(model_config, "num_attention_heads", config.n_heads),
            "hidden_size": getattr(model_config, "hidden_size", config.hidden_size),
            "vocab_size": getattr(model_config, "vocab_size", None),
            "model_type": getattr(model_config, "model_type", None),
        }
    except Exception as e:
        logger.warning(f"Could not get model info: {e}")
        return {
            "name": config.name,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "hidden_size": config.hidden_size,
        }