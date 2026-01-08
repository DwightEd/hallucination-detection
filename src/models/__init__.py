"""Model loading utilities."""

from .loader import (
    LoadedModel,
    load_model,
    ModelManager,
    get_model_manager,
    unload_all_models,
)


def get_model(config):
    """Get model from config using global manager."""
    return get_model_manager().get(config)


__all__ = [
    "LoadedModel",
    "load_model",
    "get_model",
    "ModelManager",
    "get_model_manager",
    "unload_all_models",
]