"""Unified Feature Requirements Registry - 统一的特征需求注册中心。

重构说明 (v6.2):
================
采用"基础特征 → 衍生特征"两阶段架构：

1. 基础特征 (Base Features): 
   - 直接从模型提取的原始特征
   - 不做预处理（如 pooling、层选择）
   - 存储完整数据供后续计算

2. 衍生特征 (Derived Features):
   - 从基础特征计算得到
   - 由各方法自己定义和计算
   - 计算参数在方法配置文件中指定

设计原则：
1. 基础特征定义在此文件
2. 衍生特征定义在 config/method/*.yaml
3. 衍生特征计算由 DerivedFeatureComputer 统一处理

Usage:
    from src.features.registry import (
        BASE_FEATURES,
        get_method_base_features,
        get_combined_base_features,
    )
"""
from __future__ import annotations
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# 基础特征定义
# =============================================================================

class BaseFeature(Enum):
    """基础特征枚举。
    
    这些是直接从模型提取的原始特征，不做预处理。
    """
    # 注意力相关
    ATTENTION_DIAGS = "attention_diags"      # [n_layers, n_heads, seq_len]
    ATTENTION_ROW_SUMS = "attention_row_sums" # [n_layers, n_heads, seq_len] 用于计算 degree
    FULL_ATTENTION = "full_attention"        # [n_layers, n_heads, seq_len, seq_len] (可选，高内存)
    
    # 隐藏状态
    HIDDEN_STATES = "hidden_states"          # [n_layers, seq_len, hidden_dim] (完整，不做pooling)
    
    # Token 概率
    TOKEN_PROBS = "token_probs"              # [seq_len]


# 基础特征描述
BASE_FEATURES: Dict[str, Dict[str, Any]] = {
    "attention_diags": {
        "description": "注意力对角线值 A[i,i]",
        "shape": "[n_layers, n_heads, seq_len]",
        "dtype": "float16",
        "memory_estimate_per_token": 2,  # bytes per token per head per layer
    },
    "attention_row_sums": {
        "description": "注意力矩阵每行的和（出度）",
        "shape": "[n_layers, n_heads, seq_len]",
        "dtype": "float16",
        "memory_estimate_per_token": 2,
    },
    "full_attention": {
        "description": "完整注意力矩阵（⚠️ 高内存）",
        "shape": "[n_layers, n_heads, seq_len, seq_len]",
        "dtype": "float16",
        "memory_estimate_per_token": "O(seq_len * n_heads * n_layers * 2)",
        "high_memory": True,
    },
    "hidden_states": {
        "description": "完整隐藏状态（不做 pooling）",
        "shape": "[n_layers, seq_len, hidden_dim]",
        "dtype": "float16",
        "memory_estimate_per_token": "hidden_dim * n_layers * 2",
    },
    "token_probs": {
        "description": "每个 token 的生成概率",
        "shape": "[seq_len]",
        "dtype": "float32",
        "memory_estimate_per_token": 4,
    },
}


# =============================================================================
# 衍生特征定义
# =============================================================================

DERIVED_FEATURES: Dict[str, Dict[str, Any]] = {
    "laplacian_diags": {
        "description": "Laplacian 对角线 L[i,i] = D[i,i] - A[i,i]",
        "compute_fn": "compute_laplacian_from_diags",
        "inputs": ["attention_diags", "attention_row_sums"],
        "output_shape": "[n_layers, n_heads, seq_len]",
    },
    "attention_entropy": {
        "description": "注意力熵 H = -sum(A * log(A))",
        "compute_fn": "compute_attention_entropy",
        "inputs": ["full_attention"],
        "output_shape": "[n_layers, n_heads, seq_len]",
    },
    "attention_entropy_approx": {
        "description": "注意力熵近似（从对角线）",
        "compute_fn": "compute_attention_entropy_approx",
        "inputs": ["attention_diags"],
        "output_shape": "[n_layers, n_heads, seq_len]",
    },
    "token_entropy": {
        "description": "Token 熵 H = -p * log(p)",
        "compute_fn": "compute_token_entropy",
        "inputs": ["token_probs"],
        "output_shape": "[seq_len]",
    },
    "last_token_embedding": {
        "description": "最后一个 token 的 embedding（用于 HaloScope）",
        "compute_fn": "extract_last_token_embedding",
        "inputs": ["hidden_states"],
        "output_shape": "[n_selected_layers, hidden_dim]",
    },
    "mva_features": {
        "description": "Multi-View Attention 特征（用于 HSDMVAF）",
        "compute_fn": "compute_mva_features",
        "inputs": ["full_attention"],
        "output_shape": "[n_tokens, n_features]",
    },
    "lookback_ratio": {
        "description": "Lookback 比率",
        "compute_fn": "compute_lookback_ratio",
        "inputs": ["full_attention"],
        "output_shape": "[n_layers, n_heads, response_len]",
    },
}


# =============================================================================
# 方法基础特征需求映射
# 
# 定义每个方法需要的基础特征（最小集合）
# 衍生特征的计算参数在 config/method/*.yaml 中定义
# =============================================================================

METHOD_BASE_FEATURES: Dict[str, Set[str]] = {
    # LapEigvals: 需要对角线和行和来计算 Laplacian
    "lapeigvals": {"attention_diags", "attention_row_sums"},
    
    # Lookback Lens: 需要完整注意力来计算 lookback ratio
    # 如果只有对角线，使用降级模式
    "lookback_lens": {"attention_diags"},  # 最小需求，full_attention 可选
    
    # HaloScope: 需要完整隐藏状态
    "haloscope": {"hidden_states"},
    
    # HSDMVAF: 需要完整注意力矩阵
    "hsdmvaf": {"full_attention"},
    
    # Hypergraph: 需要完整注意力矩阵
    "hypergraph": {"full_attention"},
    
    # Token Entropy: 需要 token 概率
    "token_entropy": {"token_probs"},
    
    # ACT: 需要对角线和隐藏状态
    "act": {"attention_diags", "hidden_states"},
    
    # Semantic Entropy Probes: 需要隐藏状态
    "semantic_entropy_probes": {"hidden_states"},
    
    # Ensemble: 需要所有基础特征
    "ensemble": {"attention_diags", "attention_row_sums", "hidden_states", "token_probs"},
    "auto_ensemble": {"attention_diags", "attention_row_sums", "hidden_states", "token_probs"},
}


# =============================================================================
# 向后兼容的 FeatureRequirements（将逐步废弃）
# =============================================================================

@dataclass
class FeatureRequirements:
    """方法的特征需求定义。
    
    ⚠️ 此类将在 v7.0 中废弃，请使用 METHOD_BASE_FEATURES
    
    保留此类是为了向后兼容现有代码。
    """
    attention_diags: bool = False
    attention_row_sums: bool = False  # NEW: for lapeigvals
    laplacian_diags: bool = False  # 衍生特征，将移除
    attention_entropy: bool = False  # 衍生特征，将移除
    full_attention: bool = False
    hidden_states: bool = False
    hidden_states_layers: List[int] = field(default_factory=list)  # 移到方法配置
    token_probs: bool = False
    token_entropy: bool = False  # 衍生特征，将移除
    
    def __or__(self, other: "FeatureRequirements") -> "FeatureRequirements":
        """合并两个特征需求（取并集）。"""
        return FeatureRequirements(
            attention_diags=self.attention_diags or other.attention_diags,
            attention_row_sums=self.attention_row_sums or other.attention_row_sums,
            laplacian_diags=self.laplacian_diags or other.laplacian_diags,
            attention_entropy=self.attention_entropy or other.attention_entropy,
            full_attention=self.full_attention or other.full_attention,
            hidden_states=self.hidden_states or other.hidden_states,
            hidden_states_layers=list(set(self.hidden_states_layers + other.hidden_states_layers)),
            token_probs=self.token_probs or other.token_probs,
            token_entropy=self.token_entropy or other.token_entropy,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "attention_diags": self.attention_diags,
            "attention_row_sums": self.attention_row_sums,
            "laplacian_diags": self.laplacian_diags,
            "attention_entropy": self.attention_entropy,
            "full_attention": self.full_attention,
            "hidden_states": self.hidden_states,
            "hidden_states_layers": self.hidden_states_layers,
            "token_probs": self.token_probs,
            "token_entropy": self.token_entropy,
        }
    
    def to_feature_set(self) -> Set[str]:
        features = set()
        if self.attention_diags:
            features.add("attn_diags")
        if self.attention_row_sums:
            features.add("attn_row_sums")
        if self.laplacian_diags:
            features.add("laplacian_diags")
        if self.attention_entropy:
            features.add("attn_entropy")
        if self.full_attention:
            features.add("full_attention")
        if self.hidden_states:
            features.add("hidden_states")
        if self.token_probs:
            features.add("token_probs")
        if self.token_entropy:
            features.add("token_entropy")
        return features
    
    def to_base_features(self) -> Set[str]:
        """转换为基础特征集合（新API）。"""
        features = set()
        if self.attention_diags or self.laplacian_diags:
            features.add("attention_diags")
            features.add("attention_row_sums")
        if self.full_attention or self.attention_entropy:
            features.add("full_attention")
        if self.hidden_states:
            features.add("hidden_states")
        if self.token_probs or self.token_entropy:
            features.add("token_probs")
        return features
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureRequirements":
        return cls(
            attention_diags=data.get("attention_diags", False),
            attention_row_sums=data.get("attention_row_sums", False),
            laplacian_diags=data.get("laplacian_diags", False),
            attention_entropy=data.get("attention_entropy", False),
            full_attention=data.get("full_attention", False),
            hidden_states=data.get("hidden_states", False),
            hidden_states_layers=data.get("hidden_states_layers", []),
            token_probs=data.get("token_probs", False),
            token_entropy=data.get("token_entropy", False),
        )


# 向后兼容：旧的 METHOD_FEATURE_REQUIREMENTS
METHOD_FEATURE_REQUIREMENTS: Dict[str, FeatureRequirements] = {
    "lapeigvals": FeatureRequirements(
        attention_diags=True,
        attention_row_sums=True,
        laplacian_diags=True,
    ),
    "lookback_lens": FeatureRequirements(
        attention_diags=True,
    ),
    "haloscope": FeatureRequirements(
        hidden_states=True,
    ),
    "hsdmvaf": FeatureRequirements(
        attention_diags=True,
        attention_entropy=True,
        full_attention=True,
    ),
    "hypergraph": FeatureRequirements(
        attention_diags=True,
        full_attention=True,
    ),
    "token_entropy": FeatureRequirements(
        token_probs=True,
        token_entropy=True,
    ),
    "act": FeatureRequirements(
        attention_diags=True,
        hidden_states=True,
    ),
    "semantic_entropy_probes": FeatureRequirements(
        hidden_states=True,
    ),
    "ensemble": FeatureRequirements(
        attention_diags=True,
        laplacian_diags=True,
        attention_entropy=True,
        hidden_states=True,
        token_probs=True,
        token_entropy=True,
    ),
    "auto_ensemble": FeatureRequirements(
        attention_diags=True,
        laplacian_diags=True,
        attention_entropy=True,
        hidden_states=True,
        token_probs=True,
        token_entropy=True,
    ),
}

# 向后兼容
DERIVED_FEATURE_DEPS = {
    "laplacian_diags": ["attn_diags"],
    "attn_entropy": ["full_attention"],
    "attn_entropy_approx": ["attn_diags"],
    "token_entropy": ["token_probs"],
}


# =============================================================================
# 新 API 函数
# =============================================================================

def get_method_base_features(method_name: str) -> Set[str]:
    """获取方法需要的基础特征集合。
    
    Args:
        method_name: 方法名称
        
    Returns:
        基础特征名称集合
    """
    return METHOD_BASE_FEATURES.get(method_name, set())


def get_combined_base_features(methods: List[str]) -> Set[str]:
    """获取多个方法的基础特征并集。
    
    Args:
        methods: 方法名称列表
        
    Returns:
        基础特征名称并集
    """
    combined = set()
    for method in methods:
        combined |= get_method_base_features(method)
    return combined


def needs_full_attention(methods: List[str]) -> bool:
    """检查是否需要提取完整注意力矩阵。
    
    Args:
        methods: 方法名称列表
        
    Returns:
        是否需要 full_attention
    """
    combined = get_combined_base_features(methods)
    return "full_attention" in combined


def needs_hidden_states(methods: List[str]) -> bool:
    """检查是否需要提取隐藏状态。
    
    Args:
        methods: 方法名称列表
        
    Returns:
        是否需要 hidden_states
    """
    combined = get_combined_base_features(methods)
    return "hidden_states" in combined


# =============================================================================
# 向后兼容函数
# =============================================================================

def get_method_requirements(method_name: str) -> FeatureRequirements:
    """获取方法的详细特征需求。
    
    ⚠️ 此函数将在 v7.0 中废弃，请使用 get_method_base_features
    """
    return METHOD_FEATURE_REQUIREMENTS.get(
        method_name,
        FeatureRequirements()
    )


def get_method_feature_set(method_name: str) -> Set[str]:
    """获取方法需要的简化特征集合。
    
    ⚠️ 此函数将在 v7.0 中废弃，请使用 get_method_base_features
    """
    req = get_method_requirements(method_name)
    return req.to_feature_set()


def get_combined_requirements(methods: List[str]) -> FeatureRequirements:
    """获取多个方法的合并特征需求。
    
    ⚠️ 此函数将在 v7.0 中废弃，请使用 get_combined_base_features
    """
    combined = FeatureRequirements()
    for method in methods:
        combined = combined | get_method_requirements(method)
    return combined


def is_feature_derived(feature_name: str) -> bool:
    """检查特征是否为衍生特征。"""
    return feature_name in DERIVED_FEATURES


def get_feature_dependencies(feature_name: str) -> List[str]:
    """获取衍生特征的依赖。"""
    if feature_name in DERIVED_FEATURES:
        return DERIVED_FEATURES[feature_name].get("inputs", [])
    return []
