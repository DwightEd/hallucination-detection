"""Lookback Lens: Attention Ratio-based Hallucination Detection.

Based on the concept that hallucinated content shows different
attention patterns - specifically, how much attention is paid
to the context vs. to previously generated tokens.

Key metric: Lookback ratio = attention_to_context / attention_to_generated
"""
from __future__ import annotations
from typing import Optional, List
import logging
import numpy as np
import torch

from src.core import ExtractedFeatures, MethodConfig, METHODS
from .base import BaseMethod

logger = logging.getLogger(__name__)


def compute_lookback_ratio(
    attn_diags: torch.Tensor,
    prompt_len: int,
    response_len: int,
) -> torch.Tensor:
    """Compute lookback ratio from attention diagonals.
    
    Vectorized implementation for better performance.
    
    Args:
        attn_diags: Attention diagonals [n_layers, n_heads, seq_len]
        prompt_len: Length of prompt
        response_len: Length of response
        
    Returns:
        Lookback features [n_layers * n_heads * 4]
    """
    n_layers, n_heads, seq_len = attn_diags.shape
    
    # Convert to float for computation
    attn_diags = attn_diags.float()
    
    # Split into prompt and response portions
    prompt_diag = attn_diags[:, :, :prompt_len]  # [n_layers, n_heads, prompt_len]
    
    if prompt_len + response_len <= seq_len:
        response_diag = attn_diags[:, :, prompt_len:prompt_len + response_len]
    else:
        response_diag = attn_diags[:, :, prompt_len:]
    
    # Vectorized computation across all layers and heads
    # [n_layers, n_heads]
    p_mean = prompt_diag.mean(dim=-1) if prompt_diag.shape[-1] > 0 else torch.zeros(n_layers, n_heads)
    r_mean = response_diag.mean(dim=-1) if response_diag.shape[-1] > 0 else torch.zeros(n_layers, n_heads)
    
    # Lookback ratio (higher = more attention to prompt)
    ratio = p_mean / (r_mean + 1e-8)
    
    # Difference
    diff = p_mean - r_mean
    
    # Stack and flatten: [n_layers, n_heads, 4] -> [n_layers * n_heads * 4]
    features = torch.stack([p_mean, r_mean, ratio, diff], dim=-1)  # [n_layers, n_heads, 4]
    
    return features.cpu().numpy().flatten().astype(np.float32)


def compute_attention_patterns(
    attn_diags: torch.Tensor,
    attn_entropy: Optional[torch.Tensor],
    prompt_len: int,
    response_len: int,
) -> np.ndarray:
    """Compute comprehensive attention pattern features.
    
    Vectorized implementation for better performance.
    
    Args:
        attn_diags: Attention diagonals [n_layers, n_heads, seq_len]
        attn_entropy: Attention entropy [n_layers, n_heads, seq_len]
        prompt_len: Length of prompt
        response_len: Length of response
        
    Returns:
        Feature vector
    """
    n_layers, n_heads, seq_len = attn_diags.shape
    
    # Response start and end
    resp_start = prompt_len
    resp_end = min(prompt_len + response_len, seq_len)
    
    if resp_end <= resp_start:
        resp_start = 0
        resp_end = seq_len
    
    # Extract response region [n_layers, n_heads, resp_len]
    resp_diag = attn_diags[:, :, resp_start:resp_end].float()
    resp_len_actual = resp_diag.shape[-1]
    
    if resp_len_actual == 0:
        # Return zeros if no response
        n_features = 5 * n_layers * n_heads
        if attn_entropy is not None:
            n_features += 2 * n_layers * n_heads
        return np.zeros(n_features, dtype=np.float32)
    
    # Vectorized statistics [n_layers, n_heads]
    diag_mean = resp_diag.mean(dim=-1)
    diag_std = resp_diag.std(dim=-1)
    diag_max = resp_diag.max(dim=-1).values
    diag_min = resp_diag.min(dim=-1).values
    
    # Vectorized trend computation using linear regression
    # y = ax + b, we want 'a' (slope)
    # a = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
    if resp_len_actual > 1:
        x = torch.arange(resp_len_actual, dtype=resp_diag.dtype, device=resp_diag.device)
        x_mean = x.mean()
        y_mean = resp_diag.mean(dim=-1, keepdim=True)
        
        # Compute slope: sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean)^2)
        x_centered = x - x_mean
        y_centered = resp_diag - y_mean
        
        numerator = (x_centered * y_centered).sum(dim=-1)
        denominator = (x_centered ** 2).sum() + 1e-8
        trend = numerator / denominator
    else:
        trend = torch.zeros(n_layers, n_heads)
    
    # Stack features [n_layers, n_heads, 5]
    diag_features = torch.stack([diag_mean, diag_std, diag_max, diag_min, trend], dim=-1)
    
    # Entropy features if available
    if attn_entropy is not None:
        resp_entropy = attn_entropy[:, :, resp_start:resp_end].float()
        if resp_entropy.shape[-1] > 0:
            ent_mean = resp_entropy.mean(dim=-1)
            ent_std = resp_entropy.std(dim=-1)
        else:
            ent_mean = torch.zeros(n_layers, n_heads)
            ent_std = torch.zeros(n_layers, n_heads)
        
        ent_features = torch.stack([ent_mean, ent_std], dim=-1)
        all_features = torch.cat([diag_features, ent_features], dim=-1)  # [n_layers, n_heads, 7]
    else:
        all_features = diag_features  # [n_layers, n_heads, 5]
    
    return all_features.cpu().numpy().flatten().astype(np.float32)


@METHODS.register("lookback_lens", aliases=["lookback", "attention_ratio"])
class LookbackLensMethod(BaseMethod):
    """Lookback Lens hallucination detection.
    
    Analyzes attention patterns to detect hallucinations:
    - Ratio of attention to context vs generated tokens
    - Attention distribution statistics
    - Attention entropy patterns
    """
    
    def __init__(self, config: Optional[MethodConfig] = None):
        super().__init__(config)
        
        params = self.config.params or {}
        self.use_entropy = params.get("use_entropy", True)
        self.use_trends = params.get("use_trends", True)
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """Extract lookback ratio and attention pattern features.
        
        Args:
            features: Extracted features with attn_diags
            
        Returns:
            Feature vector
        """
        if features.attn_diags is None:
            raise ValueError("LookbackLens requires attn_diags")
        
        attn_diags = features.attn_diags
        attn_entropy = features.attn_entropy if self.use_entropy else None
        
        # Convert to tensor if needed
        if isinstance(attn_diags, np.ndarray):
            attn_diags = torch.from_numpy(attn_diags)
        if attn_entropy is not None and isinstance(attn_entropy, np.ndarray):
            attn_entropy = torch.from_numpy(attn_entropy)
        
        # Compute lookback ratio features
        lookback_features = compute_lookback_ratio(
            attn_diags,
            features.prompt_len,
            features.response_len,
        )
        
        # Compute attention pattern features
        pattern_features = compute_attention_patterns(
            attn_diags,
            attn_entropy,
            features.prompt_len,
            features.response_len,
        )
        
        # Combine
        feat_vec = np.concatenate([lookback_features, pattern_features])
        
        # Handle NaN/Inf
        if np.any(~np.isfinite(feat_vec)):
            feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return feat_vec


@METHODS.register("attention_stats", aliases=["attn_stats"])
class AttentionStatsMethod(BaseMethod):
    """Simple attention statistics-based detection.
    
    Uses basic statistics of attention patterns without
    complex ratio computations.
    """
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """Extract simple attention statistics.
        
        Args:
            features: Extracted features with attn_diags
            
        Returns:
            Feature vector
        """
        if features.attn_diags is None:
            raise ValueError("AttentionStats requires attn_diags")
        
        attn_diags = features.attn_diags
        if isinstance(attn_diags, np.ndarray):
            attn_diags = torch.from_numpy(attn_diags)
        
        # Focus on response portion
        resp_start = features.prompt_len
        resp_end = features.prompt_len + features.response_len
        
        if resp_end <= attn_diags.shape[-1]:
            resp_attn = attn_diags[:, :, resp_start:resp_end]
        else:
            resp_attn = attn_diags[:, :, resp_start:]
        
        # Flatten and compute statistics
        flat = resp_attn.float().cpu().numpy().flatten()
        
        features_list = [
            np.mean(flat),
            np.std(flat),
            np.max(flat),
            np.min(flat),
            np.median(flat),
            np.percentile(flat, 25),
            np.percentile(flat, 75),
            np.percentile(flat, 90),
            np.percentile(flat, 10),
        ]
        
        # Per-layer statistics
        n_layers = attn_diags.shape[0]
        for layer in range(n_layers):
            layer_flat = resp_attn[layer].float().cpu().numpy().flatten()
            features_list.extend([
                np.mean(layer_flat),
                np.std(layer_flat),
            ])
        
        return np.array(features_list, dtype=np.float32)
