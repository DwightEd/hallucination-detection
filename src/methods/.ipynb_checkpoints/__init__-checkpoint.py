"""Detection methods for hallucination detection. 

Available methods:
- lapeigvals: Laplacian eigenvalue-based (EMNLP 2025)
- lookback_lens: Attention ratio analysis
- token_entropy: Token/attention entropy-based (原 entropy)
- semantic_entropy_probes: SEPs from hidden states (OATML)
- ensemble: Combined methods (voting, stacking, concat)
- hypergraph: Hypergraph neural network (HyperCHARM)
- haloscope: SVD-based unsupervised detection (NeurIPS'24)
- hsdmvaf: Multi-view attention with Transformer+CRF
- act: Hidden state probing (LLMsKnow, arXiv'24)
- tsv: Truthfulness Separator Vector (ICML'25)

方法对比:
- token_entropy: 使用 token 概率熵和注意力熵
- semantic_entropy_probes: 从 hidden states 预测 semantic entropy
- hsdmvaf: 使用 Multi-View Attention Features (avg_in, div_in, div_out)
- act: 使用 exact_answer 最后 token 位置的 hidden states
- tsv: 学习 steering vector 重塑表示空间, 使用 vMF 分布分类

Example:
    from src.methods import create_method, LapEigvalsMethod, HypergraphMethod, ACTMethod, TSVMethod
    
    # By name
    method = create_method("hypergraph")
    method = create_method("semantic_entropy_probes")  # 新增
    method = create_method("token_entropy")  # 原 entropy
    method = create_method("haloscope")  # HaloScope方法
    method = create_method("act")  # ACT/LLMsKnow方法
    method = create_method("tsv")  # TSV方法 (ICML'25)
    
    # Direct instantiation
    method = HypergraphMethod()
    method = ACTMethod()
    method = TSVMethod()
    
    # Train
    method.fit(features_list)
    
    # Predict
    prediction = method.predict(features)
"""

from .base import BaseMethod, create_method
from .lapeigvals import LapEigvalsMethod, LapEigvalsFullMethod
from .lookback_lens import LookbackLensMethod, AttentionStatsMethod
from .token_entropy import TokenEntropyMethod, PerplexityMethod
from .semantic_entropy_probes import SemanticEntropyProbesMethod, HiddenStateProbeMethod
from .ensemble import EnsembleMethod, AutoEnsembleMethod
from .hypergraph import HypergraphMethod, HypergraphTokenMethod, HyperCHARMModel, HypergraphData, HypergraphBuilder
from .haloscope import HaloScopeDetector, HaloScopeMethod  # 添加 HaloScopeMethod
from .hsdmvaf import HSDMVAFMethod, HSDMVAFDetector, MultiViewAttentionEncoder
from .act import ACTMethod, ACTMultiLayerMethod, ACTExactAnswerMethod, ACTDetector  # ACT方法
from .tsv import TSVMethod, TSVDetector

# 向后兼容别名
EntropyMethod = TokenEntropyMethod

__all__ = [
    # Base
    "BaseMethod",
    "create_method",
    
    # LapEigvals
    "LapEigvalsMethod",
    "LapEigvalsFullMethod",
    
    # Lookback Lens
    "LookbackLensMethod",
    "AttentionStatsMethod",
    
    # Token Entropy (原 Entropy)
    "TokenEntropyMethod",
    "EntropyMethod",  # 向后兼容别名
    "PerplexityMethod",
    
    # Semantic Entropy Probes (新增)
    "SemanticEntropyProbesMethod",
    "HiddenStateProbeMethod",
    
    # Ensemble
    "EnsembleMethod",
    "AutoEnsembleMethod",
    
    # Hypergraph
    "HypergraphMethod",
    "HypergraphTokenMethod",
    
    # HaloScope (NeurIPS'24)
    "HaloScopeDetector",
    "HaloScopeMethod",  # 添加 HaloScopeMethod
    
    # HSDMVAF (修复后)
    "HSDMVAFMethod",
    "HSDMVAFDetector",
    "MultiViewAttentionEncoder",
    
    # ACT / LLMsKnow (arXiv'24)
    "ACTMethod",
    "ACTMultiLayerMethod",
    "ACTExactAnswerMethod",
    "ACTDetector",
    
    # TSV (ICML'25)
    "TSVMethod",
    "TSVDetector",
]