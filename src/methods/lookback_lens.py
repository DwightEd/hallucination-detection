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
    
    Approximation: Since we only have diagonals, we estimate the ratio
    using the diagonal values in prompt vs response regions.
    
    Args:
        attn_diags: Attention diagonals [n_layers, n_heads, seq_len]
        prompt_len: Length of prompt
        response_len: Length of response
        
    Returns:
        Lookback features [n_layers, n_heads, 4]
    """
    n_layers, n_heads, seq_len = attn_diags.shape
    
    # Split into prompt and response portions
    prompt_diag = attn_diags[:, :, :prompt_len]  # [n_layers, n_heads, prompt_len]
    
    if prompt_len + response_len <= seq_len:
        response_diag = attn_diags[:, :, prompt_len:prompt_len + response_len]
    else:
        response_diag = attn_diags[:, :, prompt_len:]
    
    features = []
    
    for layer in range(n_layers):
        for head in range(n_heads):
            p_diag = prompt_diag[layer, head].float()
            r_diag = response_diag[layer, head].float()
            
            # Compute statistics
            p_mean = p_diag.mean().item() if p_diag.numel() > 0 else 0
            r_mean = r_diag.mean().item() if r_diag.numel() > 0 else 0
            
            # Lookback ratio (higher = more attention to prompt)
            ratio = p_mean / (r_mean + 1e-8)
            
            # Difference
            diff = p_mean - r_mean
            
            features.extend([p_mean, r_mean, ratio, diff])
    
    return np.array(features, dtype=np.float32)


def compute_attention_patterns(
    attn_diags: torch.Tensor,
    attn_entropy: Optional[torch.Tensor],
    prompt_len: int,
    response_len: int,
) -> np.ndarray:
    """Compute comprehensive attention pattern features.
    
    Args:
        attn_diags: Attention diagonals [n_layers, n_heads, seq_len]
        attn_entropy: Attention entropy [n_layers, n_heads, seq_len]
        prompt_len: Length of prompt
        response_len: Length of response
        
    Returns:
        Feature vector
    """
    features = []
    
    n_layers, n_heads, seq_len = attn_diags.shape
    
    # Response start and end
    resp_start = prompt_len
    resp_end = min(prompt_len + response_len, seq_len)
    
    if resp_end <= resp_start:
        resp_start = 0
        resp_end = seq_len
    
    for layer in range(n_layers):
        for head in range(n_heads):
            # Attention diagonal in response region
            resp_diag = attn_diags[layer, head, resp_start:resp_end].float().cpu().numpy()
            
            if len(resp_diag) > 0:
                # Self-attention statistics
                features.extend([
                    np.mean(resp_diag),
                    np.std(resp_diag),
                    np.max(resp_diag),
                    np.min(resp_diag),
                ])
                
                # Trend: is attention increasing or decreasing?
                if len(resp_diag) > 1:
                    trend = np.polyfit(np.arange(len(resp_diag)), resp_diag, 1)[0]
                    features.append(trend)
                else:
                    features.append(0.0)
            else:
                features.extend([0.0] * 5)
            
            # Entropy features if available
            if attn_entropy is not None:
                resp_entropy = attn_entropy[layer, head, resp_start:resp_end].float().cpu().numpy()
                if len(resp_entropy) > 0:
                    features.extend([
                        np.mean(resp_entropy),
                        np.std(resp_entropy),
                    ])
                else:
                    features.extend([0.0] * 2)
    
    return np.array(features, dtype=np.float32)


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
