"""Token Entropy-based Hallucination Detection.

Uses various token-level entropy measures to detect hallucinations:
- Token probability entropy: Uncertainty in token predictions
- Attention entropy: How spread out attention is
- Perplexity: Overall model confidence

注意：这与 Semantic Entropy Probes 不同。
- Token Entropy: 直接使用 token 预测概率的熵
- Semantic Entropy Probes: 使用 hidden states 训练 probe 来预测 semantic entropy

Key insight: Hallucinated content often shows different entropy patterns,
either too high (uncertain) or too low (overconfident without grounding).
"""
from __future__ import annotations
from typing import Optional, List
import logging
import numpy as np
import torch

from src.core import ExtractedFeatures, MethodConfig, METHODS
from .base import BaseMethod

logger = logging.getLogger(__name__)


def compute_token_entropy_features(
    token_entropy: torch.Tensor,
    token_probs: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Compute features from token-level entropy.
    
    Args:
        token_entropy: Entropy at each token position [seq_len]
        token_probs: Probability of each token [seq_len]
        
    Returns:
        Feature vector
    """
    if isinstance(token_entropy, torch.Tensor):
        token_entropy = token_entropy.float().cpu().numpy()
    
    features = []
    
    if len(token_entropy) > 0:
        # Basic statistics
        features.extend([
            np.mean(token_entropy),
            np.std(token_entropy),
            np.max(token_entropy),
            np.min(token_entropy),
            np.median(token_entropy),
        ])
        
        # Percentiles
        features.extend([
            np.percentile(token_entropy, 25),
            np.percentile(token_entropy, 75),
            np.percentile(token_entropy, 90),
            np.percentile(token_entropy, 10),
        ])
        
        # High entropy ratio (tokens with entropy > mean + std)
        threshold = np.mean(token_entropy) + np.std(token_entropy)
        high_entropy_ratio = np.mean(token_entropy > threshold)
        features.append(high_entropy_ratio)
        
        # Low entropy ratio (tokens with entropy < mean - std)
        low_threshold = np.mean(token_entropy) - np.std(token_entropy)
        low_entropy_ratio = np.mean(token_entropy < low_threshold)
        features.append(low_entropy_ratio)
        
        # Entropy trend (increasing or decreasing over sequence)
        if len(token_entropy) > 1:
            x = np.arange(len(token_entropy))
            trend = np.polyfit(x, token_entropy, 1)[0]
            features.append(trend)
            
            # Entropy variance in windows
            window_size = max(len(token_entropy) // 4, 1)
            windows = [token_entropy[i:i+window_size] for i in range(0, len(token_entropy), window_size)]
            window_means = [np.mean(w) for w in windows if len(w) > 0]
            if len(window_means) > 1:
                features.append(np.std(window_means))
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0])
    else:
        features.extend([0.0] * 13)
    
    # Token probability features
    if token_probs is not None:
        if isinstance(token_probs, torch.Tensor):
            token_probs = token_probs.float().cpu().numpy()
        
        if len(token_probs) > 0:
            features.extend([
                np.mean(token_probs),
                np.std(token_probs),
                np.min(token_probs),
                np.mean(token_probs < 0.1),  # Low confidence ratio
                np.mean(token_probs > 0.9),  # High confidence ratio
            ])
        else:
            features.extend([0.0] * 5)
    
    return np.array(features, dtype=np.float32)


def compute_attention_entropy_features(
    attn_entropy: torch.Tensor,
    prompt_len: int,
    response_len: int,
) -> np.ndarray:
    """Compute features from attention entropy.
    
    Args:
        attn_entropy: Attention entropy [n_layers, n_heads, seq_len]
        prompt_len: Length of prompt
        response_len: Length of response
        
    Returns:
        Feature vector
    """
    if isinstance(attn_entropy, torch.Tensor):
        attn_entropy = attn_entropy.float().cpu().numpy()
    
    n_layers, n_heads, seq_len = attn_entropy.shape
    
    # Focus on response portion
    resp_start = prompt_len
    resp_end = min(prompt_len + response_len, seq_len)
    
    if resp_end <= resp_start:
        resp_start = 0
        resp_end = seq_len
    
    resp_entropy = attn_entropy[:, :, resp_start:resp_end]
    
    features = []
    
    # Global statistics
    flat = resp_entropy.flatten()
    features.extend([
        np.mean(flat),
        np.std(flat),
        np.max(flat),
        np.min(flat),
    ])
    
    # Per-layer statistics (limit to avoid huge feature vectors)
    n_layers_to_use = min(n_layers, 8)
    layer_indices = np.linspace(0, n_layers - 1, n_layers_to_use, dtype=int)
    
    for layer in layer_indices:
        layer_entropy = resp_entropy[layer].flatten()
        features.extend([
            np.mean(layer_entropy),
            np.std(layer_entropy),
        ])
    
    # Layer-wise trend
    layer_means = [np.mean(resp_entropy[l]) for l in range(n_layers)]
    if len(layer_means) > 1:
        trend = np.polyfit(np.arange(len(layer_means)), layer_means, 1)[0]
        features.append(trend)
    else:
        features.append(0.0)
    
    return np.array(features, dtype=np.float32)


# 注册为 token_entropy（主名称）和 entropy（别名，保持向后兼容）
@METHODS.register("token_entropy", aliases=["entropy"])
class TokenEntropyMethod(BaseMethod):
    """Token entropy-based hallucination detection.
    
    Uses token prediction entropy and attention entropy
    to detect hallucinations.
    
    注意：这与 Semantic Entropy Probes (SEPs) 不同：
    - Token Entropy: 直接使用 token 级别的熵统计量
    - SEPs: 从 hidden states 预测 semantic entropy
    """
    
    def __init__(self, config: Optional[MethodConfig] = None):
        super().__init__(config)
        
        params = self.config.params or {}
        self.use_token_entropy = params.get("use_token_entropy", True)
        self.use_attention_entropy = params.get("use_attention_entropy", True)
        self.use_perplexity = params.get("use_perplexity", True)
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """Extract entropy-based features.
        
        Args:
            features: Extracted features
            
        Returns:
            Feature vector
        """
        all_features = []
        
        # Token entropy features
        if self.use_token_entropy and features.token_entropy is not None:
            token_feats = compute_token_entropy_features(
                features.token_entropy,
                features.token_probs,
            )
            all_features.append(token_feats)
        
        # Attention entropy features
        if self.use_attention_entropy and features.attn_entropy is not None:
            attn_feats = compute_attention_entropy_features(
                features.attn_entropy,
                features.prompt_len,
                features.response_len,
            )
            all_features.append(attn_feats)
        
        # Perplexity
        if self.use_perplexity and features.perplexity is not None:
            all_features.append(np.array([features.perplexity], dtype=np.float32))
        
        if len(all_features) == 0:
            raise ValueError("TokenEntropy method requires token_entropy or attn_entropy")
        
        return np.concatenate(all_features)


@METHODS.register("perplexity", aliases=["ppl"])
class PerplexityMethod(BaseMethod):
    """Simple perplexity-based detection.
    
    Uses only perplexity and basic token probability statistics.
    """
    
    def extract_method_features(self, features: ExtractedFeatures) -> np.ndarray:
        """Extract perplexity-based features.
        
        Args:
            features: Extracted features
            
        Returns:
            Feature vector
        """
        feat_list = []
        
        # Perplexity
        if features.perplexity is not None:
            feat_list.append(features.perplexity)
            feat_list.append(np.log(features.perplexity + 1))
        else:
            feat_list.extend([0.0, 0.0])
        
        # Token probabilities
        if features.token_probs is not None:
            probs = features.token_probs
            if isinstance(probs, torch.Tensor):
                probs = probs.float().cpu().numpy()
            
            if len(probs) > 0:
                feat_list.extend([
                    np.mean(probs),
                    np.std(probs),
                    np.min(probs),
                    np.mean(-np.log(probs + 1e-10)),  # Average negative log prob
                ])
            else:
                feat_list.extend([0.0] * 4)
        else:
            feat_list.extend([0.0] * 4)
        
        return np.array(feat_list, dtype=np.float32)