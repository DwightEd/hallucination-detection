"""Feature Registry - 特征需求注册与依赖管理。

定义基础特征和派生特征的关系，以及各方法的具体需求。

基础特征（需要模型推理）：
- full_attention: 完整注意力矩阵 [n_layers, n_heads, seq_len, seq_len]
- hidden_states: 隐藏状态 [n_layers, seq_len, hidden_dim]
- token_probs: Token 概率 [seq_len]

派生特征（从基础特征计算）：
- attention_diags: 注意力对角线（从 full_attention）
- laplacian_diags: Laplacian 对角线（从 full_attention）
- attention_entropy: 注意力熵（从 full_attention）
- lookback_ratio: Lookback 比率（从 full_attention，需要 prompt/response 分离）
- mva_features: Multi-View Attention 特征（从 full_attention）

Usage:
    from src.features.feature_registry import (
        get_method_requirements,
        compute_union_requirements,
        get_base_features_needed,
        FeatureScope,
    )
    
    # 获取方法需求
    req = get_method_requirements("lapeigvals")
    
    # 计算多方法的需求并集
    union_req = compute_union_requirements(["lapeigvals", "lookback_lens"])
    
    # 获取需要的基础特征
    base_features = get_base_features_needed(union_req)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# 枚举定义
# =============================================================================

class FeatureScope(Enum):
    """特征范围：使用 prompt、response 还是全部。"""
    FULL = auto()          # 使用完整序列
    PROMPT_ONLY = auto()   # 只使用 prompt 部分
    RESPONSE_ONLY = auto() # 只使用 response 部分
    BOTH_SEPARATE = auto() # prompt 和 response 分别处理


class LayerSelection(Enum):
    """层选择策略。"""
    ALL = auto()           # 所有层
    LAST = auto()          # 最后一层
    LAST_4 = auto()        # 最后 4 层
    LAST_HALF = auto()     # 后半部分层
    FIRST_HALF = auto()    # 前半部分层
    SPECIFIC = auto()      # 指定层


class HeadSelection(Enum):
    """注意力头选择策略。"""
    ALL = auto()           # 所有头
    MEAN = auto()          # 平均所有头
    SPECIFIC = auto()      # 指定头


# =============================================================================
# 基础特征类型
# =============================================================================

class BaseFeatureType(Enum):
    """基础特征类型（需要模型推理）。"""
    FULL_ATTENTION = "full_attention"
    HIDDEN_STATES = "hidden_states"
    TOKEN_PROBS = "token_probs"


# =============================================================================
# 派生特征类型
# =============================================================================

class DerivedFeatureType(Enum):
    """派生特征类型（从基础特征计算）。"""
    ATTENTION_DIAGS = "attention_diags"
    LAPLACIAN_DIAGS = "laplacian_diags"
    ATTENTION_ENTROPY = "attention_entropy"
    LOOKBACK_RATIO = "lookback_ratio"
    MVA_FEATURES = "mva_features"  # Multi-View Attention


# 派生特征到基础特征的依赖映射
DERIVED_TO_BASE_DEPS: Dict[DerivedFeatureType, BaseFeatureType] = {
    DerivedFeatureType.ATTENTION_DIAGS: BaseFeatureType.FULL_ATTENTION,
    DerivedFeatureType.LAPLACIAN_DIAGS: BaseFeatureType.FULL_ATTENTION,
    DerivedFeatureType.ATTENTION_ENTROPY: BaseFeatureType.FULL_ATTENTION,
    DerivedFeatureType.LOOKBACK_RATIO: BaseFeatureType.FULL_ATTENTION,
    DerivedFeatureType.MVA_FEATURES: BaseFeatureType.FULL_ATTENTION,
}


# =============================================================================
# 特征需求配置
# =============================================================================

@dataclass
class DerivedFeatureConfig:
    """派生特征的详细配置。"""
    feature_type: DerivedFeatureType
    scope: FeatureScope = FeatureScope.RESPONSE_ONLY
    layer_selection: LayerSelection = LayerSelection.ALL
    specific_layers: List[int] = field(default_factory=list)
    head_selection: HeadSelection = HeadSelection.ALL
    specific_heads: List[int] = field(default_factory=list)
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaseFeatureConfig:
    """基础特征的详细配置。"""
    feature_type: BaseFeatureType
    layer_selection: LayerSelection = LayerSelection.ALL
    specific_layers: List[int] = field(default_factory=list)
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class MethodFeatureRequirements:
    """方法的完整特征需求。"""
    method_name: str
    
    # 需要的基础特征
    base_features: List[BaseFeatureConfig] = field(default_factory=list)
    
    # 需要的派生特征（及其详细配置）
    derived_features: List[DerivedFeatureConfig] = field(default_factory=list)
    
    # 是否使用 prompt 部分的 attention
    uses_prompt_attention: bool = False
    
    # 是否使用 response 部分的 attention
    uses_response_attention: bool = True
    
    # 备注
    notes: str = ""
    
    def needs_full_attention(self) -> bool:
        """检查是否需要完整注意力矩阵。"""
        # 检查基础特征
        for bf in self.base_features:
            if bf.feature_type == BaseFeatureType.FULL_ATTENTION:
                return True
        
        # 检查派生特征的依赖
        for df in self.derived_features:
            if DERIVED_TO_BASE_DEPS.get(df.feature_type) == BaseFeatureType.FULL_ATTENTION:
                return True
        
        return False
    
    def needs_hidden_states(self) -> bool:
        """检查是否需要隐藏状态。"""
        for bf in self.base_features:
            if bf.feature_type == BaseFeatureType.HIDDEN_STATES:
                return True
        return False
    
    def needs_token_probs(self) -> bool:
        """检查是否需要 token 概率。"""
        for bf in self.base_features:
            if bf.feature_type == BaseFeatureType.TOKEN_PROBS:
                return True
        return False
    
    def get_required_base_features(self) -> Set[BaseFeatureType]:
        """获取所有需要的基础特征。"""
        required = set()
        
        # 直接需要的基础特征
        for bf in self.base_features:
            required.add(bf.feature_type)
        
        # 派生特征依赖的基础特征
        for df in self.derived_features:
            base_dep = DERIVED_TO_BASE_DEPS.get(df.feature_type)
            if base_dep:
                required.add(base_dep)
        
        return required
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "method_name": self.method_name,
            "needs_full_attention": self.needs_full_attention(),
            "needs_hidden_states": self.needs_hidden_states(),
            "needs_token_probs": self.needs_token_probs(),
            "uses_prompt_attention": self.uses_prompt_attention,
            "uses_response_attention": self.uses_response_attention,
            "derived_features": [df.feature_type.value for df in self.derived_features],
            "notes": self.notes,
        }


# =============================================================================
# 各方法的特征需求定义
# =============================================================================

METHOD_REQUIREMENTS: Dict[str, MethodFeatureRequirements] = {
    # =========================================================================
    # LapEigvals (EMNLP 2025)
    # 只使用 response 部分的 attention 对角线来计算 Laplacian 特征值
    # =========================================================================
    "lapeigvals": MethodFeatureRequirements(
        method_name="lapeigvals",
        derived_features=[
            DerivedFeatureConfig(
                feature_type=DerivedFeatureType.ATTENTION_DIAGS,
                scope=FeatureScope.RESPONSE_ONLY,
                layer_selection=LayerSelection.ALL,
            ),
            DerivedFeatureConfig(
                feature_type=DerivedFeatureType.LAPLACIAN_DIAGS,
                scope=FeatureScope.RESPONSE_ONLY,
                layer_selection=LayerSelection.ALL,
            ),
        ],
        uses_prompt_attention=False,
        uses_response_attention=True,
        notes="Original paper only uses response tokens for Laplacian eigenvalue computation",
    ),
    
    # =========================================================================
    # Lookback Lens
    # 需要对比 prompt 和 response 的 attention 模式
    # =========================================================================
    "lookback_lens": MethodFeatureRequirements(
        method_name="lookback_lens",
        derived_features=[
            DerivedFeatureConfig(
                feature_type=DerivedFeatureType.ATTENTION_DIAGS,
                scope=FeatureScope.BOTH_SEPARATE,  # 需要分别处理
                layer_selection=LayerSelection.ALL,
            ),
            DerivedFeatureConfig(
                feature_type=DerivedFeatureType.LOOKBACK_RATIO,
                scope=FeatureScope.BOTH_SEPARATE,
                layer_selection=LayerSelection.ALL,
            ),
        ],
        uses_prompt_attention=True,  # 需要 prompt 的 attention
        uses_response_attention=True,
        notes="Compares attention to context vs generated tokens",
    ),
    
    # =========================================================================
    # HaloScope (NeurIPS'24)
    # 只使用 hidden states，不需要 attention
    # =========================================================================
    "haloscope": MethodFeatureRequirements(
        method_name="haloscope",
        base_features=[
            BaseFeatureConfig(
                feature_type=BaseFeatureType.HIDDEN_STATES,
                layer_selection=LayerSelection.LAST_HALF,
            ),
        ],
        uses_prompt_attention=False,
        uses_response_attention=False,
        notes="SVD-based detection using hidden states only",
    ),
    
    # =========================================================================
    # HSDMVAF (Multi-View Attention Features)
    # 需要完整 attention 矩阵来计算 MVA 特征
    # =========================================================================
    "hsdmvaf": MethodFeatureRequirements(
        method_name="hsdmvaf",
        base_features=[
            BaseFeatureConfig(
                feature_type=BaseFeatureType.FULL_ATTENTION,
                layer_selection=LayerSelection.LAST_4,  # 限制层数减少内存
            ),
        ],
        derived_features=[
            DerivedFeatureConfig(
                feature_type=DerivedFeatureType.MVA_FEATURES,
                scope=FeatureScope.RESPONSE_ONLY,
                layer_selection=LayerSelection.LAST_4,
            ),
        ],
        uses_prompt_attention=True,  # MVA 需要看 incoming attention from all tokens
        uses_response_attention=True,
        notes="Multi-View Attention: avg_in, div_in, div_out",
    ),
    
    # =========================================================================
    # Hypergraph (HyperCHARM)
    # 需要完整 attention 矩阵来构建超图
    # =========================================================================
    "hypergraph": MethodFeatureRequirements(
        method_name="hypergraph",
        base_features=[
            BaseFeatureConfig(
                feature_type=BaseFeatureType.FULL_ATTENTION,
                layer_selection=LayerSelection.ALL,
            ),
        ],
        uses_prompt_attention=True,  # 需要完整图结构
        uses_response_attention=True,
        notes="Requires full attention for hypergraph construction",
    ),
    
    # =========================================================================
    # Token Entropy
    # 使用 token 概率和 attention entropy
    # =========================================================================
    "token_entropy": MethodFeatureRequirements(
        method_name="token_entropy",
        base_features=[
            BaseFeatureConfig(
                feature_type=BaseFeatureType.TOKEN_PROBS,
            ),
        ],
        derived_features=[
            DerivedFeatureConfig(
                feature_type=DerivedFeatureType.ATTENTION_ENTROPY,
                scope=FeatureScope.RESPONSE_ONLY,
                layer_selection=LayerSelection.ALL,
            ),
        ],
        uses_prompt_attention=False,
        uses_response_attention=True,
        notes="Uses token probabilities and attention entropy",
    ),
    
    # =========================================================================
    # Semantic Entropy Probes
    # 使用 hidden states 预测 semantic entropy
    # =========================================================================
    "semantic_entropy_probes": MethodFeatureRequirements(
        method_name="semantic_entropy_probes",
        base_features=[
            BaseFeatureConfig(
                feature_type=BaseFeatureType.HIDDEN_STATES,
                layer_selection=LayerSelection.ALL,
            ),
            BaseFeatureConfig(
                feature_type=BaseFeatureType.TOKEN_PROBS,
            ),
        ],
        uses_prompt_attention=False,
        uses_response_attention=False,
        notes="Predicts semantic entropy from hidden states",
    ),
}


# =============================================================================
# 工具函数
# =============================================================================

def get_method_requirements(method_name: str) -> MethodFeatureRequirements:
    """获取方法的特征需求。
    
    Args:
        method_name: 方法名称
        
    Returns:
        方法的特征需求配置
    """
    method_name = method_name.lower()
    
    if method_name in METHOD_REQUIREMENTS:
        return METHOD_REQUIREMENTS[method_name]
    
    # 默认需求：只需要 attention 对角线
    logger.warning(f"Unknown method '{method_name}', using default requirements")
    return MethodFeatureRequirements(
        method_name=method_name,
        derived_features=[
            DerivedFeatureConfig(
                feature_type=DerivedFeatureType.ATTENTION_DIAGS,
                scope=FeatureScope.RESPONSE_ONLY,
            ),
        ],
    )


def compute_union_requirements(
    method_names: List[str]
) -> MethodFeatureRequirements:
    """计算多个方法的特征需求并集。
    
    Args:
        method_names: 方法名称列表
        
    Returns:
        合并后的特征需求
    """
    if not method_names:
        return get_method_requirements("lapeigvals")
    
    # 收集所有需求
    all_base_features: Dict[BaseFeatureType, BaseFeatureConfig] = {}
    all_derived_features: Dict[DerivedFeatureType, DerivedFeatureConfig] = {}
    uses_prompt = False
    uses_response = False
    
    for name in method_names:
        req = get_method_requirements(name)
        
        # 合并基础特征（取层的并集）
        for bf in req.base_features:
            if bf.feature_type not in all_base_features:
                all_base_features[bf.feature_type] = bf
            else:
                # 如果已存在，选择更宽的层范围
                existing = all_base_features[bf.feature_type]
                if bf.layer_selection == LayerSelection.ALL:
                    all_base_features[bf.feature_type] = bf
        
        # 合并派生特征
        for df in req.derived_features:
            if df.feature_type not in all_derived_features:
                all_derived_features[df.feature_type] = df
            else:
                # 选择更宽的范围
                existing = all_derived_features[df.feature_type]
                if df.scope == FeatureScope.FULL or df.scope == FeatureScope.BOTH_SEPARATE:
                    all_derived_features[df.feature_type] = df
        
        uses_prompt = uses_prompt or req.uses_prompt_attention
        uses_response = uses_response or req.uses_response_attention
    
    return MethodFeatureRequirements(
        method_name=f"union({','.join(method_names)})",
        base_features=list(all_base_features.values()),
        derived_features=list(all_derived_features.values()),
        uses_prompt_attention=uses_prompt,
        uses_response_attention=uses_response,
        notes=f"Union of requirements for: {', '.join(method_names)}",
    )


def get_base_features_needed(
    requirements: MethodFeatureRequirements
) -> Dict[BaseFeatureType, BaseFeatureConfig]:
    """获取需要提取的基础特征。
    
    根据需求确定：
    - 如果有派生特征需要 full_attention，则需要提取 full_attention
    - 否则可能只需要提取派生特征
    
    Args:
        requirements: 特征需求
        
    Returns:
        需要的基础特征及其配置
    """
    needed: Dict[BaseFeatureType, BaseFeatureConfig] = {}
    
    # 直接需要的基础特征
    for bf in requirements.base_features:
        needed[bf.feature_type] = bf
    
    # 派生特征依赖的基础特征
    for df in requirements.derived_features:
        base_dep = DERIVED_TO_BASE_DEPS.get(df.feature_type)
        if base_dep and base_dep not in needed:
            # 使用派生特征的层配置
            needed[base_dep] = BaseFeatureConfig(
                feature_type=base_dep,
                layer_selection=df.layer_selection,
                specific_layers=df.specific_layers,
            )
    
    return needed


def should_store_full_attention(
    requirements: MethodFeatureRequirements
) -> bool:
    """判断是否需要存储完整的注意力矩阵。
    
    规则：
    - 如果任何方法直接需要 full_attention -> 存储
    - 如果需要多个派生特征，且都依赖 full_attention -> 存储（避免重复计算）
    - 如果只需要一个简单的派生特征 -> 可以直接计算并存储派生特征
    
    Args:
        requirements: 特征需求
        
    Returns:
        是否需要存储完整注意力矩阵
    """
    # 直接需要 full_attention
    for bf in requirements.base_features:
        if bf.feature_type == BaseFeatureType.FULL_ATTENTION:
            return True
    
    # 检查派生特征：如果有 2+ 个依赖 full_attention 的派生特征，则存储
    attention_dependent_count = sum(
        1 for df in requirements.derived_features
        if DERIVED_TO_BASE_DEPS.get(df.feature_type) == BaseFeatureType.FULL_ATTENTION
    )
    
    # 如果需要 lookback_ratio 或 mva_features，这些需要完整矩阵来计算
    for df in requirements.derived_features:
        if df.feature_type in [DerivedFeatureType.LOOKBACK_RATIO, DerivedFeatureType.MVA_FEATURES]:
            return True
    
    return attention_dependent_count >= 2


def describe_requirements(requirements: MethodFeatureRequirements) -> str:
    """生成需求的人类可读描述。"""
    lines = ["=" * 60]
    lines.append(f"Feature Requirements: {requirements.method_name}")
    lines.append("=" * 60)
    
    # 基础特征
    if requirements.base_features:
        lines.append("\n📦 Base Features (require model inference):")
        for bf in requirements.base_features:
            lines.append(f"  • {bf.feature_type.value} (layers: {bf.layer_selection.name})")
    
    # 派生特征
    if requirements.derived_features:
        lines.append("\n🔧 Derived Features (computed from base):")
        for df in requirements.derived_features:
            base_dep = DERIVED_TO_BASE_DEPS.get(df.feature_type)
            dep_str = f" ← {base_dep.value}" if base_dep else ""
            lines.append(f"  • {df.feature_type.value}{dep_str}")
            lines.append(f"      scope: {df.scope.name}, layers: {df.layer_selection.name}")
    
    # Attention 使用情况
    lines.append("\n🎯 Attention Usage:")
    lines.append(f"  • Uses prompt attention: {'✓' if requirements.uses_prompt_attention else '✗'}")
    lines.append(f"  • Uses response attention: {'✓' if requirements.uses_response_attention else '✗'}")
    
    # 总结
    lines.append("\n📊 Summary:")
    lines.append(f"  • Needs full_attention: {'✓' if requirements.needs_full_attention() else '✗'}")
    lines.append(f"  • Needs hidden_states: {'✓' if requirements.needs_hidden_states() else '✗'}")
    lines.append(f"  • Needs token_probs: {'✓' if requirements.needs_token_probs() else '✗'}")
    
    if requirements.notes:
        lines.append(f"\n📝 Notes: {requirements.notes}")
    
    lines.append("=" * 60)
    return "\n".join(lines)
