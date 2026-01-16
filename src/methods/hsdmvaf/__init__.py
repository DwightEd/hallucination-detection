"""HSDMVAF - Hallucinated Span Detection with Multi-View Attention Features.

基于多视角注意力特征的幻觉检测方法。

模块结构：
- features.py: Multi-View Attention特征计算
- model.py: Transformer+CRF模型
- method.py: HSDMVAFMethod检测方法
"""

from .method import HSDMVAFMethod, HSDMVAFDetector
from .model import MultiViewAttentionEncoder, CRFLayer, HSDMVAFModel
from .features import (
    compute_multi_view_attention_features,
    compute_mva_features_from_diags,
)

__all__ = [
    "HSDMVAFMethod",
    "HSDMVAFDetector",
    "MultiViewAttentionEncoder",
    "CRFLayer",
    "HSDMVAFModel",
    "compute_multi_view_attention_features",
    "compute_mva_features_from_diags",
]
