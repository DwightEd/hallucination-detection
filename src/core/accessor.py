"""Feature accessor for convenient access to ExtractedFeatures data.

Provides a context manager for accessing features with automatic
lazy loading and memory management.
"""
from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import logging

import torch
import numpy as np

if TYPE_CHECKING:
    from .types import ExtractedFeatures

logger = logging.getLogger(__name__)


class FeatureAccessor:
    """Context manager for accessing ExtractedFeatures data.
    
    Provides convenient methods to access various feature types with:
    - Automatic lazy loading for large features
    - Automatic memory cleanup on exit
    - Consistent numpy array output
    
    Example:
        with FeatureAccessor(features, prefer_fast=True) as accessor:
            diag = accessor.get_attention_diags()
            entropy = accessor.get_attention_entropy()
    """
    
    def __init__(
        self,
        features: 'ExtractedFeatures',
        prefer_fast: bool = True,
        allow_lazy_load: bool = True,
    ):
        """Initialize accessor.
        
        Args:
            features: ExtractedFeatures object to access
            prefer_fast: If True, prefer pre-computed features over computing from raw data
            allow_lazy_load: If True, allow loading large features from disk
        """
        self.features = features
        self.prefer_fast = prefer_fast
        self.allow_lazy_load = allow_lazy_load
        self._loaded_large_features = False
    
    def __enter__(self) -> 'FeatureAccessor':
        """Enter context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context, clean up large features."""
        if self._loaded_large_features:
            self.features.release_large_features()
    
    def _to_numpy(self, tensor: Optional[torch.Tensor]) -> Optional[np.ndarray]:
        """Convert tensor to numpy array."""
        if tensor is None:
            return None
        if isinstance(tensor, np.ndarray):
            return tensor
        return tensor.cpu().numpy()
    
    def get_attention_diags(self) -> Optional[np.ndarray]:
        """Get attention diagonal values.
        
        Returns:
            Attention diagonals as numpy array, or None if not available
        """
        if self.features.attn_diags is not None:
            return self._to_numpy(self.features.attn_diags)
        return None
    
    def get_laplacian_diags(self) -> Optional[np.ndarray]:
        """Get Laplacian diagonal values.
        
        If not available but attn_diags is available, computes it as (1 - attn_diags).
        
        Returns:
            Laplacian diagonals as numpy array, or None if not available
        """
        if self.features.laplacian_diags is not None:
            return self._to_numpy(self.features.laplacian_diags)
        
        # Compute from attn_diags if available
        if self.features.attn_diags is not None:
            attn = self._to_numpy(self.features.attn_diags)
            return 1.0 - attn
        
        return None
    
    def get_attention_entropy(self) -> Optional[np.ndarray]:
        """Get attention entropy values.
        
        Returns:
            Attention entropy as numpy array, or None if not available
        """
        if self.features.attn_entropy is not None:
            return self._to_numpy(self.features.attn_entropy)
        return None
    
    def get_token_probs(self) -> Optional[np.ndarray]:
        """Get token probabilities.
        
        Returns:
            Token probabilities as numpy array, or None if not available
        """
        if self.features.token_probs is not None:
            return self._to_numpy(self.features.token_probs)
        return None
    
    def get_token_entropy(self) -> Optional[np.ndarray]:
        """Get token entropy values.
        
        Returns:
            Token entropy as numpy array, or None if not available
        """
        if self.features.token_entropy is not None:
            return self._to_numpy(self.features.token_entropy)
        return None
    
    def get_hidden_states(self) -> Optional[np.ndarray]:
        """Get hidden states (may lazy load).
        
        Returns:
            Hidden states as numpy array, or None if not available
        """
        if self.features.hidden_states is not None:
            return self._to_numpy(self.features.hidden_states)
        
        if self.allow_lazy_load:
            loaded = self.features.get_hidden_states()
            if loaded is not None:
                self._loaded_large_features = True
                return self._to_numpy(loaded)
        
        return None
    
    def get_full_attention(self) -> Optional[torch.Tensor]:
        """Get full attention matrix (may lazy load).
        
        Note: Returns torch.Tensor for compatibility with hypergraph builder.
        
        Returns:
            Full attention tensor, or None if not available
        """
        if self.features.full_attention is not None:
            return self.features.full_attention
        
        if self.allow_lazy_load:
            loaded = self.features.get_full_attention()
            if loaded is not None:
                self._loaded_large_features = True
                return loaded
        
        return None
    
    def get_response_features(
        self,
        feature_name: str,
    ) -> Optional[np.ndarray]:
        """Get features for response tokens only.
        
        Args:
            feature_name: Name of feature to get
                Options: "attn_diags", "laplacian_diags", "attn_entropy", 
                         "token_probs", "token_entropy"
        
        Returns:
            Feature array sliced to response portion, or None
        """
        getter_map = {
            "attn_diags": self.get_attention_diags,
            "laplacian_diags": self.get_laplacian_diags,
            "attn_entropy": self.get_attention_entropy,
            "token_probs": self.get_token_probs,
            "token_entropy": self.get_token_entropy,
        }
        
        if feature_name not in getter_map:
            logger.warning(f"Unknown feature: {feature_name}")
            return None
        
        data = getter_map[feature_name]()
        if data is None:
            return None
        
        prompt_len = self.features.prompt_len
        response_len = self.features.response_len
        
        if response_len <= 0:
            return data
        
        # Handle different tensor shapes
        if len(data.shape) == 1:
            # [seq_len]
            end = min(prompt_len + response_len, len(data))
            return data[prompt_len:end]
        elif len(data.shape) == 2:
            # [n_layers, seq_len] or [n_heads, seq_len]
            end = min(prompt_len + response_len, data.shape[-1])
            return data[..., prompt_len:end]
        elif len(data.shape) == 3:
            # [n_layers, n_heads, seq_len]
            end = min(prompt_len + response_len, data.shape[-1])
            return data[..., prompt_len:end]
        
        return data
