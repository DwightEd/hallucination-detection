"""Feature extraction module for hallucination detection.

Provides:
- FeatureExtractor: Main class for extracting features
- Utility functions for attention, hidden states, and token probabilities

Example:
    from src.features import FeatureExtractor, create_extractor
    from src.models import load_model
    from src.core import FeaturesConfig
    
    extractor = FeatureExtractor(model, FeaturesConfig(
        mode="teacher_forcing",
        attention_layers="last_n:4",
        hidden_states_layers="last_n:4",
    ))
    
    features = extractor.extract(sample)
    # features.attn_diags: [n_layers, n_heads, seq_len]
    # features.laplacian_diags: [n_layers, n_heads, seq_len]
    # features.hidden_states: [n_layers, hidden_size]
    # features.token_probs: [response_len]
"""

from .extractor import (
    # Main extractor
    FeatureExtractor,
    create_extractor,
    
    # Attention utilities
    extract_attention_diagonal,
    compute_laplacian_diagonal,
    compute_attention_entropy,
    stack_layer_attentions,
    
    # Hidden state utilities
    pool_hidden_states,
    stack_layer_hidden_states,
    
    # Token probability utilities
    compute_token_probs,
    compute_token_entropy,
    compute_top_k_probs,
)

__all__ = [
    "FeatureExtractor",
    "create_extractor",
    "extract_attention_diagonal",
    "compute_laplacian_diagonal",
    "compute_attention_entropy",
    "stack_layer_attentions",
    "pool_hidden_states",
    "stack_layer_hidden_states",
    "compute_token_probs",
    "compute_token_entropy",
    "compute_top_k_probs",
]
