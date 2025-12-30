"""Model loading module for hallucination detection.

Example:
    from src.models import load_model, LoadedModel
    from src.core import ModelConfig
    
    model = load_model(ModelConfig(
        name="Qwen/Qwen2.5-7B-Instruct",
        attn_implementation="eager",  # Required for attention extraction
    ))
    
    # Forward pass
    outputs = model.forward(input_ids, output_attentions=True)
    
    # Generation
    outputs = model.generate(input_ids, max_new_tokens=256)
"""

from .loader import (
    LoadedModel,
    load_model,
    ModelManager,
    get_model_manager,
    unload_model,
    unload_all_models,
)

__all__ = [
    "LoadedModel",
    "load_model",
    "ModelManager",
    "get_model_manager",
    "unload_model",
    "unload_all_models",
]
