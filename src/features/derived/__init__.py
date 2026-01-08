"""Derived Feature Extractors - 派生特征提取器。

按基础特征分类：
- attention_derived: 基于 full_attention 的派生特征
- token_probs_derived: 基于 token_probs 的派生特征
- hidden_states_derived: 基于 hidden_states 的派生特征

Usage:
    from src.features.derived import (
        # Attention-based
        compute_attention_diags,
        compute_laplacian_diags,
        compute_attention_entropy,
        compute_lookback_ratio,
        compute_mva_features,
        
        # Token probs-based
        compute_token_entropy,
        compute_token_confidence,
        compute_sequence_perplexity,
        
        # Hidden states-based
        compute_pooled_states,
        compute_layer_similarity,
        compute_svd_features,
    )
"""

# =============================================================================
# Attention-based Derived Features
# =============================================================================
from .attention_derived import (
    # Attention Diagonals
    compute_attention_diags,
    compute_attention_diags_direct,
    AttentionDiagsConfig,
    
    # Laplacian Diagonals
    compute_laplacian_diags,
    compute_laplacian_diags_direct,
    
    # Attention Entropy
    compute_attention_entropy,
    compute_attention_entropy_direct,
    
    # Lookback Ratio
    compute_lookback_ratio,
    compute_lookback_ratio_direct,
    
    # MVA Features
    compute_mva_features,
    compute_mva_features_direct,
)

# =============================================================================
# Token Probability-based Derived Features
# =============================================================================
from .token_probs_derived import (
    # Token Entropy
    compute_token_entropy,
    compute_token_entropy_from_probs,
    
    # Token Confidence
    compute_token_confidence,
    compute_top_k_confidence,
    
    # Perplexity
    compute_sequence_perplexity,
    compute_token_perplexity,
    
    # Uncertainty Metrics
    compute_uncertainty_metrics,
)

# =============================================================================
# Hidden States-based Derived Features
# =============================================================================
from .hidden_states_derived import (
    # Pooling
    PoolingStrategy,
    compute_pooled_states,
    compute_layerwise_pooled_states,
    
    # Layer Similarity
    compute_layer_similarity,
    compute_layer_similarity_matrix,
    
    # Representation Statistics
    compute_representation_stats,
    compute_representation_drift,
    
    # SVD Features
    compute_svd_features,
)

__all__ = [
    # =========================
    # Attention-based
    # =========================
    "compute_attention_diags",
    "compute_attention_diags_direct",
    "AttentionDiagsConfig",
    
    "compute_laplacian_diags",
    "compute_laplacian_diags_direct",
    
    "compute_attention_entropy",
    "compute_attention_entropy_direct",
    
    "compute_lookback_ratio",
    "compute_lookback_ratio_direct",
    
    "compute_mva_features",
    "compute_mva_features_direct",
    
    # =========================
    # Token Probability-based
    # =========================
    "compute_token_entropy",
    "compute_token_entropy_from_probs",
    
    "compute_token_confidence",
    "compute_top_k_confidence",
    
    "compute_sequence_perplexity",
    "compute_token_perplexity",
    
    "compute_uncertainty_metrics",
    
    # =========================
    # Hidden States-based
    # =========================
    "PoolingStrategy",
    "compute_pooled_states",
    "compute_layerwise_pooled_states",
    
    "compute_layer_similarity",
    "compute_layer_similarity_matrix",
    
    "compute_representation_stats",
    "compute_representation_drift",
    
    "compute_svd_features",
]
