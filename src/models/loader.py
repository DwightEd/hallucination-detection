"""Model loading utilities for hallucination detection.

Key requirements:
- attn_implementation MUST be "eager" for attention extraction
- Support quantization (4bit/8bit) for large models
- Unified interface for generation and forward pass
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.core import ModelConfig, ModelError

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """Container for loaded model and tokenizer.
    
    Provides unified interface for model operations.
    """
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    config: ModelConfig
    device: str
    
    @property
    def num_layers(self) -> int:
        """Get number of transformer layers."""
        return self.config.n_layers
    
    @property
    def num_heads(self) -> int:
        """Get number of attention heads."""
        return self.config.n_heads
    
    @property
    def hidden_size(self) -> int:
        """Get hidden dimension size."""
        return self.config.hidden_size
    
    def encode(self, text: str, add_special_tokens: bool = True) -> torch.Tensor:
        """Encode text to token ids."""
        return self.tokenizer.encode(text, return_tensors="pt", add_special_tokens=add_special_tokens)
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode token ids to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = True,
        output_hidden_states: bool = True,
    ) -> Dict[str, Any]:
        """Forward pass with attention and hidden states output.
        
        Args:
            input_ids: Input token ids [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            
        Returns:
            Dictionary with logits, attentions, hidden_states
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        
        result = {"logits": outputs.logits}
        
        if output_attentions and outputs.attentions is not None:
            # attentions: tuple of [batch, heads, seq, seq] for each layer
            result["attentions"] = outputs.attentions
        
        if output_hidden_states and outputs.hidden_states is not None:
            # hidden_states: tuple of [batch, seq, hidden] for each layer
            result["hidden_states"] = outputs.hidden_states
        
        return result
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict_in_generate: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate text with optional attention/hidden states output.
        
        Args:
            input_ids: Input token ids [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            
        Returns:
            Dictionary with generated_ids, attentions (optional), hidden_states (optional)
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict_in_generate=return_dict_in_generate,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **kwargs,
            )
        
        if return_dict_in_generate:
            result = {"generated_ids": outputs.sequences}
            if output_attentions and hasattr(outputs, "attentions") and outputs.attentions:
                result["attentions"] = outputs.attentions
            if output_hidden_states and hasattr(outputs, "hidden_states") and outputs.hidden_states:
                result["hidden_states"] = outputs.hidden_states
            return result
        
        return {"generated_ids": outputs}
    
    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get logits for input tokens (for probability computation)."""
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, return_dict=True)
        return outputs.logits


def load_model(config: ModelConfig, device: Optional[str] = None) -> LoadedModel:
    """Load model and tokenizer from config.
    
    Args:
        config: Model configuration
        device: Override device (default: from config)
        
    Returns:
        LoadedModel instance
        
    Raises:
        ModelError: If loading fails
    """
    device = device or config.device_map
    logger.info(f"Loading model: {config.name}")
    
    # Validate attention implementation
    if config.attn_implementation != "eager":
        logger.warning(
            f"attn_implementation={config.attn_implementation}, "
            "but 'eager' is required for attention extraction. Forcing 'eager'."
        )
        config.attn_implementation = "eager"
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.name,
            trust_remote_code=config.trust_remote_code,
        )
        
        # Ensure pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare model kwargs
        model_kwargs = {
            "trust_remote_code": config.trust_remote_code,
            "attn_implementation": config.attn_implementation,
            "device_map": config.device_map,
        }
        
        # Set dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if config.dtype in dtype_map:
            model_kwargs["torch_dtype"] = dtype_map[config.dtype]
        
        # Quantization config
        if config.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype_map.get(config.dtype, torch.float16),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif config.load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(config.name, **model_kwargs)
        model.eval()
        
        # Determine actual device
        if hasattr(model, "hf_device_map"):
            actual_device = next(iter(model.hf_device_map.values()))
        elif hasattr(model, "device"):
            actual_device = str(model.device)
        else:
            actual_device = device
        
        # Update config with actual model architecture
        model_config = model.config
        if hasattr(model_config, "num_hidden_layers"):
            config.n_layers = model_config.num_hidden_layers
        if hasattr(model_config, "num_attention_heads"):
            config.n_heads = model_config.num_attention_heads
        if hasattr(model_config, "hidden_size"):
            config.hidden_size = model_config.hidden_size
        
        logger.info(f"Model loaded: {config.n_layers} layers, {config.n_heads} heads, {config.hidden_size} hidden")
        
        return LoadedModel(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=str(actual_device),
        )
        
    except Exception as e:
        raise ModelError(f"Failed to load model: {e}", details={"model": config.name})


class ModelManager:
    """Manage multiple loaded models with caching."""
    
    def __init__(self):
        self._cache: Dict[str, LoadedModel] = {}
    
    def get(self, config: ModelConfig, device: Optional[str] = None) -> LoadedModel:
        """Get model, loading if not cached."""
        key = f"{config.name}:{config.dtype}:{config.load_in_4bit}:{config.load_in_8bit}"
        
        if key not in self._cache:
            self._cache[key] = load_model(config, device)
        
        return self._cache[key]
    
    def unload(self, model_name: str) -> None:
        """Unload specific model from cache."""
        keys_to_remove = [k for k in self._cache if k.startswith(model_name)]
        for key in keys_to_remove:
            del self._cache[key]
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def unload_all(self) -> None:
        """Unload all models."""
        self._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Global model manager
_model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """Get global model manager."""
    return _model_manager


def unload_model(model_name: str) -> None:
    """Unload model from global cache."""
    _model_manager.unload(model_name)


def unload_all_models() -> None:
    """Unload all models from global cache."""
    _model_manager.unload_all()
