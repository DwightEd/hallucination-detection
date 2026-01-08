"""Base Feature Extractors - 基础特征提取器。

基础特征需要模型推理才能获得：
- full_attention: 完整注意力矩阵
- hidden_states: 隐藏状态
- token_probs: Token 概率

Usage:
    from src.features.base import (
        extract_full_attention,
        extract_hidden_states,
        extract_token_probs,
    )
"""

from .attention import (
    extract_full_attention,
    extract_attention_for_layers,
    AttentionExtractor,
)

from .hidden_states import (
    extract_hidden_states,
    extract_hidden_states_for_layers,
    HiddenStatesExtractor,
)

from .token_probs import (
    extract_token_probs,
    compute_token_entropy,
    TokenProbsExtractor,
)

__all__ = [
    # Attention
    "extract_full_attention",
    "extract_attention_for_layers",
    "AttentionExtractor",
    
    # Hidden States
    "extract_hidden_states",
    "extract_hidden_states_for_layers",
    "HiddenStatesExtractor",
    
    # Token Probs
    "extract_token_probs",
    "compute_token_entropy",
    "TokenProbsExtractor",
]
