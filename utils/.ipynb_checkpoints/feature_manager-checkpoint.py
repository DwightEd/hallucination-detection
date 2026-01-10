"""Feature Manager - Unified feature extraction based on method requirements.

特征管理器，负责：
- 根据方法需求统一管理特征提取
- 计算多方法的特征需求并集
- 内存预估和安全控制
- 一次前向传播提取所有所需特征

Design principles:
1. One forward pass extracts all required features
2. Method configs define their own feature requirements
3. Features are computed as union of all method requirements
4. Avoid redundant computation across methods
5. Full attention disabled by default for memory safety

Usage:
    from utils.feature_manager import FeatureManager, create_feature_manager
    
    manager = create_feature_manager(methods=["lapeigvals", "entropy"])
    requirements = manager.get_combined_requirements()
    config = manager.to_features_config()
    
    # 检查内存需求
    mem_estimate = manager.estimate_memory_per_sample(seq_len=2048)
"""
from __future__ import annotations
import logging
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# 内存估算常量（单位：bytes per element, float16）
# =============================================================================

MEMORY_ESTIMATES = {
    "attention_diags": 2,       # per token per head per layer
    "laplacian_diags": 2,       # per token per head per layer
    "attention_entropy": 2,     # per token per head per layer
    "hidden_states": 2,         # per token per hidden_dim per layer
    "token_probs": 4,           # per token (float32)
    "full_attention": 2,        # per token * seq_len * heads per layer (HUGE!)
}


# =============================================================================
# 特征需求数据类
# =============================================================================

@dataclass
class FeatureRequirements:
    """方法的特征需求定义。
    
    Attributes:
        attention_diags: 注意力对角线值
        laplacian_diags: 拉普拉斯特征值
        attention_entropy: 注意力熵
        full_attention: 完整注意力矩阵（⚠️ 高内存消耗）
        hidden_states: 隐藏状态向量
        hidden_states_layers: 要提取的隐藏层索引
        token_probs: Token概率
        token_entropy: Token级熵
    """
    attention_diags: bool = False
    laplacian_diags: bool = False
    attention_entropy: bool = False
    full_attention: bool = False  # ⚠️ 高内存，默认禁用
    hidden_states: bool = False
    hidden_states_layers: List[int] = field(default_factory=list)
    token_probs: bool = False
    token_entropy: bool = False
    
    def __or__(self, other: "FeatureRequirements") -> "FeatureRequirements":
        """合并两个特征需求（取并集）。"""
        return FeatureRequirements(
            attention_diags=self.attention_diags or other.attention_diags,
            laplacian_diags=self.laplacian_diags or other.laplacian_diags,
            attention_entropy=self.attention_entropy or other.attention_entropy,
            full_attention=self.full_attention or other.full_attention,
            hidden_states=self.hidden_states or other.hidden_states,
            hidden_states_layers=list(set(self.hidden_states_layers + other.hidden_states_layers)),
            token_probs=self.token_probs or other.token_probs,
            token_entropy=self.token_entropy or other.token_entropy,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "attention_diags": self.attention_diags,
            "laplacian_diags": self.laplacian_diags,
            "attention_entropy": self.attention_entropy,
            "full_attention": self.full_attention,
            "hidden_states": self.hidden_states,
            "hidden_states_layers": self.hidden_states_layers,
            "token_probs": self.token_probs,
            "token_entropy": self.token_entropy,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureRequirements":
        """从字典创建。"""
        return cls(
            attention_diags=data.get("attention_diags", False),
            laplacian_diags=data.get("laplacian_diags", False),
            attention_entropy=data.get("attention_entropy", False),
            full_attention=data.get("full_attention", False),
            hidden_states=data.get("hidden_states", False),
            hidden_states_layers=data.get("hidden_states_layers", []),
            token_probs=data.get("token_probs", False),
            token_entropy=data.get("token_entropy", False),
        )


# =============================================================================
# 方法特征需求定义
# 
# 根据原始论文的要求定义每个方法需要的特征：
# - lapeigvals: 使用attention对角线计算Laplacian特征值 (EMNLP 2025)
# - lookback_lens: 使用attention分析context vs generated比率
# - haloscope: 使用hidden states进行SVD分析 (NeurIPS'24)
# - hsdmvaf: Multi-view attention特征 (需要full_attention以获得最佳效果)
# - hypergraph: 超图神经网络 (需要full_attention)
# =============================================================================

METHOD_FEATURE_REQUIREMENTS: Dict[str, FeatureRequirements] = {
    # =========================================================================
    # LapEigvals - Laplacian Eigenvalue Analysis (EMNLP 2025)
    # 论文: Hallucination Detection in LLMs Using Spectral Features of Attention Maps
    # 只需要 attention 对角线来计算 Laplacian 特征值
    # =========================================================================
    "lapeigvals": FeatureRequirements(
        attention_diags=True,
        laplacian_diags=True,
        attention_entropy=False,
        full_attention=False,  # 只需要对角线，不需要完整矩阵
        hidden_states=False,
        token_probs=False,
        token_entropy=False,
    ),
    
    # =========================================================================
    # Lookback Lens - Attention Ratio Analysis
    # 论文: Looking Backwards: Attention-based Hallucination Detection
    # 使用 attention 对角线分析 context vs generated token 的注意力比率
    # =========================================================================
    "lookback_lens": FeatureRequirements(
        attention_diags=True,
        laplacian_diags=False,
        attention_entropy=False,
        full_attention=False,  # 可以从对角线计算基本比率
        hidden_states=False,
        token_probs=False,
        token_entropy=False,
    ),
    
    # =========================================================================
    # HaloScope - SVD-based Unsupervised Detection (NeurIPS'24)
    # 论文: Harnessing Unlabeled LLM Generations for Hallucination Detection
    # 只需要 hidden states 进行 SVD 分析，不需要 attention
    # =========================================================================
    "haloscope": FeatureRequirements(
        attention_diags=False,
        laplacian_diags=False,
        attention_entropy=False,
        full_attention=False,  # 不需要 attention
        hidden_states=True,    # 核心特征
        token_probs=False,
        token_entropy=False,
    ),
    
    # =========================================================================
    # HSDMVAF - Hallucinated Span Detection with Multi-View Attention Features
    # 论文: https://aclanthology.org/2025.starsem-1.31/
    # 需要完整 attention 矩阵来计算 Multi-View Attention 特征:
    # - avg_in: 入向平均注意力
    # - div_in: 入向注意力多样性
    # - div_out: 出向注意力多样性
    # =========================================================================
    "hsdmvaf": FeatureRequirements(
        attention_diags=True,      # 备选：可以近似计算
        laplacian_diags=False,
        attention_entropy=True,    # 用于计算 div_out
        full_attention=True,       # 最佳：完整 MVA 特征
        hidden_states=False,
        token_probs=False,
        token_entropy=False,
    ),
    
    # =========================================================================
    # Hypergraph - Hypergraph Neural Network for Hallucination Detection
    # 论文: HyperCHARM - Hypergraph-based Hallucination Detection
    # 需要完整 attention 矩阵来构建超图结构
    # =========================================================================
    "hypergraph": FeatureRequirements(
        attention_diags=True,      # 辅助特征
        laplacian_diags=False,
        attention_entropy=False,
        full_attention=True,       # 核心：构建超图需要完整矩阵
        hidden_states=False,
        token_probs=False,
        token_entropy=False,
    ),
    
    # =========================================================================
    # 其他方法
    # =========================================================================
    
    # Token Entropy-based Detection
    "token_entropy": FeatureRequirements(
        attention_diags=False,
        laplacian_diags=False,
        attention_entropy=True,
        full_attention=False,
        hidden_states=False,
        token_probs=True,
        token_entropy=True,
    ),
    
    # Semantic Entropy Probes
    "semantic_entropy_probes": FeatureRequirements(
        attention_diags=False,
        laplacian_diags=False,
        attention_entropy=False,
        full_attention=False,
        hidden_states=True,
        token_probs=True,
        token_entropy=False,
    ),
    
    # Ensemble Method
    "ensemble": FeatureRequirements(
        attention_diags=True,
        laplacian_diags=True,
        attention_entropy=True,
        full_attention=False,
        hidden_states=True,
        token_probs=True,
        token_entropy=True,
    ),
    
    # Auto Ensemble (combination of multiple methods)
    "auto_ensemble": FeatureRequirements(
        attention_diags=True,
        laplacian_diags=True,
        attention_entropy=True,
        full_attention=False,
        hidden_states=True,
        token_probs=True,
        token_entropy=True,
    ),
}


def get_method_requirements(method_name: str) -> FeatureRequirements:
    """获取指定方法的特征需求。
    
    Args:
        method_name: 方法名称
        
    Returns:
        该方法的特征需求
    """
    return METHOD_FEATURE_REQUIREMENTS.get(
        method_name.lower(),
        FeatureRequirements(attention_diags=True)  # 默认
    )


def compute_union_requirements(method_names: List[str]) -> FeatureRequirements:
    """计算多个方法的特征需求并集。
    
    Args:
        method_names: 方法名称列表
        
    Returns:
        所有方法需求的并集
    """
    if not method_names:
        return FeatureRequirements(attention_diags=True)
    
    requirements = get_method_requirements(method_names[0])
    for name in method_names[1:]:
        requirements = requirements | get_method_requirements(name)
    
    return requirements


# =============================================================================
# 特征管理器
# =============================================================================

class FeatureManager:
    """特征管理器 - 根据方法需求统一管理特征提取。
    
    关键安全特性：
    - `allow_full_attention` 默认为 False
    - 即使方法需要 full_attention，也必须显式允许才会启用
    - 提供内存预估功能
    
    Usage:
        manager = FeatureManager(methods=["lapeigvals", "entropy"])
        requirements = manager.get_combined_requirements()
        config = manager.to_features_config()
        
        # 检查内存需求
        mem = manager.estimate_memory_per_sample(seq_len=2048, n_layers=32, n_heads=32)
    """
    
    def __init__(
        self,
        methods: Optional[List[str]] = None,
        config_path: Optional[Path] = None,
        allow_full_attention: bool = False,
    ):
        """初始化特征管理器。
        
        Args:
            methods: 方法名称列表
            config_path: 配置文件路径
            allow_full_attention: 是否允许完整注意力提取
                ⚠️ 默认禁用以防止OOM
        """
        self.methods: List[str] = methods or []
        self.allow_full_attention = allow_full_attention
        self._requirements_cache: Dict[str, FeatureRequirements] = {}
        
        if config_path:
            self._load_config(config_path)
        
        if not self.allow_full_attention:
            logger.info(
                "Full attention extraction DISABLED (memory safety). "
                "Set allow_full_attention=True to enable."
            )
    
    def _load_config(self, config_path: Path):
        """从配置文件加载方法列表。"""
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            if "methods" in config:
                methods = config["methods"]
                if isinstance(methods, list):
                    for m in methods:
                        if isinstance(m, str):
                            self.methods.append(m)
                        elif isinstance(m, dict) and "name" in m:
                            self.methods.append(m["name"])
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
    
    def add_method(
        self,
        method_name: str,
        requirements: Optional[FeatureRequirements] = None
    ):
        """添加方法及其特征需求。
        
        Args:
            method_name: 方法名称
            requirements: 自定义需求（None则使用默认）
        """
        self.methods.append(method_name)
        if requirements:
            self._requirements_cache[method_name] = requirements
    
    def get_method_requirements(self, method_name: str) -> FeatureRequirements:
        """获取指定方法的特征需求。"""
        if method_name in self._requirements_cache:
            return self._requirements_cache[method_name]
        return get_method_requirements(method_name)
    
    def get_combined_requirements(self) -> FeatureRequirements:
        """获取所有方法的特征需求并集。"""
        if not self.methods:
            return FeatureRequirements(attention_diags=True)
        
        reqs = [self.get_method_requirements(m) for m in self.methods]
        combined = reqs[0]
        for r in reqs[1:]:
            combined = combined | r
        
        logger.info(f"Combined requirements for {self.methods}: {combined.to_dict()}")
        return combined
    
    def has_high_memory_features(self) -> bool:
        """检查是否包含高内存消耗的特征需求。
        
        Returns:
            是否需要 full_attention
        """
        combined = self.get_combined_requirements()
        return combined.full_attention
    
    def estimate_memory_per_sample(
        self,
        seq_len: int = 2048,
        n_layers: int = 32,
        n_heads: int = 32,
        hidden_size: int = 4096,
    ) -> Dict[str, float]:
        """估算每个样本的内存需求。
        
        Args:
            seq_len: 序列长度
            n_layers: 模型层数
            n_heads: 注意力头数
            hidden_size: 隐藏维度
            
        Returns:
            各特征类型的内存估算（单位：MB）
        """
        req = self.get_combined_requirements()
        estimates = {}
        
        n_attn_layers = n_layers  # 假设提取所有层
        n_hidden_layers = len(req.hidden_states_layers) if req.hidden_states_layers else n_layers
        
        # 注意力相关特征
        if req.attention_diags:
            # [n_layers, seq_len, n_heads]
            bytes_needed = n_attn_layers * seq_len * n_heads * MEMORY_ESTIMATES["attention_diags"]
            estimates["attention_diags_mb"] = bytes_needed / 1024**2
        
        if req.laplacian_diags:
            bytes_needed = n_attn_layers * seq_len * n_heads * MEMORY_ESTIMATES["laplacian_diags"]
            estimates["laplacian_diags_mb"] = bytes_needed / 1024**2
        
        if req.attention_entropy:
            bytes_needed = n_attn_layers * seq_len * n_heads * MEMORY_ESTIMATES["attention_entropy"]
            estimates["attention_entropy_mb"] = bytes_needed / 1024**2
        
        # ⚠️ 完整注意力（非常大！）
        if req.full_attention and self.allow_full_attention:
            # [n_layers, n_heads, seq_len, seq_len]
            bytes_needed = n_attn_layers * n_heads * seq_len * seq_len * MEMORY_ESTIMATES["full_attention"]
            estimates["full_attention_mb"] = bytes_needed / 1024**2
        
        # 隐藏状态
        if req.hidden_states:
            bytes_needed = n_hidden_layers * seq_len * hidden_size * MEMORY_ESTIMATES["hidden_states"]
            estimates["hidden_states_mb"] = bytes_needed / 1024**2
        
        # Token概率
        if req.token_probs or req.token_entropy:
            bytes_needed = seq_len * MEMORY_ESTIMATES["token_probs"]
            estimates["token_probs_mb"] = bytes_needed / 1024**2
        
        # 总计
        estimates["total_mb"] = sum(estimates.values())
        estimates["total_gb"] = estimates["total_mb"] / 1024
        
        return estimates
    
    def to_features_config(self) -> Dict[str, Any]:
        """转换为 FeaturesConfig 兼容的字典格式。
        
        Returns:
            可用于配置 FeatureExtractor 的字典
        """
        req = self.get_combined_requirements()
        
        # ⚠️ 安全控制：仅在显式允许时启用 full_attention
        store_full = req.full_attention and self.allow_full_attention
        if req.full_attention and not self.allow_full_attention:
            logger.warning(
                "⚠️ Method(s) request full_attention but allow_full_attention=False. "
                "Full attention will NOT be extracted to prevent OOM."
            )
        
        return {
            "attention_enabled": (
                req.attention_diags or
                req.laplacian_diags or
                req.attention_entropy or
                req.full_attention
            ),
            "attention_layers": "all",
            "store_full_attention": store_full,
            "hidden_states_enabled": req.hidden_states,
            "hidden_states_layers": (
                req.hidden_states_layers if req.hidden_states_layers else "last_4"
            ),
            "token_probs_enabled": req.token_probs or req.token_entropy,
        }
    
    def describe(self) -> str:
        """获取人类可读的需求描述。"""
        lines = ["=" * 50]
        lines.append("Feature Requirements Summary")
        lines.append("=" * 50)
        lines.append(f"Methods: {', '.join(self.methods)}")
        lines.append(f"Allow full attention: {self.allow_full_attention}")
        
        combined = self.get_combined_requirements()
        lines.append("\nRequired features:")
        
        if combined.attention_diags:
            lines.append("  ✓ Attention diagonals")
        if combined.laplacian_diags:
            lines.append("  ✓ Laplacian diagonals")
        if combined.attention_entropy:
            lines.append("  ✓ Attention entropy")
        if combined.full_attention:
            status = "✓" if self.allow_full_attention else "✗ (disabled)"
            lines.append(f"  {status} Full attention matrices (⚠️ high memory)")
        if combined.hidden_states:
            layers = combined.hidden_states_layers or ["last_4"]
            lines.append(f"  ✓ Hidden states (layers: {layers})")
        if combined.token_probs:
            lines.append("  ✓ Token probabilities")
        if combined.token_entropy:
            lines.append("  ✓ Token entropy")
        
        # 内存估算
        mem = self.estimate_memory_per_sample()
        lines.append(f"\nEstimated memory per sample: {mem['total_mb']:.1f} MB")
        
        lines.append("=" * 50)
        return "\n".join(lines)


# =============================================================================
# 工厂函数
# =============================================================================

def create_feature_manager(
    methods: Optional[List[str]] = None,
    config_path: Optional[Path] = None,
    allow_full_attention: bool = False,
) -> FeatureManager:
    """创建特征管理器。
    
    Args:
        methods: 方法名称列表
        config_path: 配置文件路径
        allow_full_attention: 是否允许完整注意力
        
    Returns:
        配置好的 FeatureManager 实例
    """
    return FeatureManager(
        methods=methods,
        config_path=config_path,
        allow_full_attention=allow_full_attention,
    )


# =============================================================================
# 从配置加载方法需求
# =============================================================================

def load_method_requirements_from_config(
    config_dir: Path
) -> Dict[str, FeatureRequirements]:
    """从方法配置目录加载特征需求。
    
    查找每个方法配置文件中的 'required_features' 部分。
    
    Args:
        config_dir: 配置目录路径
        
    Returns:
        方法名称到需求的映射
    """
    method_dir = config_dir / "method"
    if not method_dir.exists():
        return {}
    
    requirements = {}
    
    for config_file in method_dir.glob("*.yaml"):
        try:
            with open(config_file) as f:
                config = yaml.safe_load(f)
            
            method_name = config.get("name", config_file.stem)
            
            if "required_features" in config:
                rf = config["required_features"]
                requirements[method_name] = FeatureRequirements.from_dict(rf)
                logger.debug(f"Loaded requirements for {method_name} from config")
        except Exception as e:
            logger.warning(f"Failed to load {config_file}: {e}")
    
    return requirements


def validate_features_for_method(
    features: Dict[str, Any],
    method_name: str
) -> tuple[bool, List[str]]:
    """验证提取的特征是否满足方法需求。
    
    Args:
        features: 已提取的特征字典
        method_name: 方法名称
        
    Returns:
        (是否有效, 缺失特征列表)
    """
    requirements = get_method_requirements(method_name)
    missing = []
    
    if requirements.attention_diags and features.get("attn_diags") is None:
        missing.append("attention_diags")
    
    if requirements.laplacian_diags and features.get("laplacian_diags") is None:
        missing.append("laplacian_diags")
    
    if requirements.attention_entropy and features.get("attn_entropy") is None:
        missing.append("attention_entropy")
    
    if requirements.full_attention and features.get("full_attention") is None:
        missing.append("full_attention")
    
    if requirements.hidden_states and features.get("hidden_states") is None:
        missing.append("hidden_states")
    
    if requirements.token_probs and features.get("token_probs") is None:
        missing.append("token_probs")
    
    return len(missing) == 0, missing
