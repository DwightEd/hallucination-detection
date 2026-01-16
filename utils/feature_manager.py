"""Feature Manager - Unified feature extraction based on method requirements.

重构说明 (v6.2):
================
采用"基础特征 → 衍生特征"两阶段架构：

1. 特征提取阶段：
   - 读取方法列表的 base_features 需求
   - 计算并集，提取所有需要的基础特征
   - 基础特征提取时不做预处理（如 pooling）

2. 训练/预测阶段：
   - 各方法根据自己的配置计算衍生特征
   - 衍生特征配置在 config/method/*.yaml 中定义
   - 由 DerivedFeatureComputer 统一处理

Usage:
    from utils.feature_manager import FeatureManager, create_feature_manager
    
    # 创建特征管理器
    manager = create_feature_manager(methods=["lapeigvals", "haloscope"])
    
    # 获取基础特征需求
    base_features = manager.get_required_base_features()
    
    # 转换为提取器配置
    config = manager.to_features_config()
"""
from __future__ import annotations
import logging
from typing import Dict, List, Set, Any, Optional
from pathlib import Path
import yaml

# 从统一注册中心导入
from src.features.registry import (
    BASE_FEATURES,
    METHOD_BASE_FEATURES,
    get_method_base_features,
    get_combined_base_features,
    needs_full_attention,
    needs_hidden_states,
    # 向后兼容（仅内部使用，不再导出）
    FeatureRequirements,
    get_method_requirements,
    get_combined_requirements as compute_union_requirements,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 内存估算常量（单位：bytes per element, float16）
# =============================================================================

MEMORY_ESTIMATES = {
    "attention_diags": 2,       # per token per head per layer
    "attention_row_sums": 2,    # per token per head per layer
    "full_attention": 2,        # per token * seq_len * heads per layer (HUGE!)
    "hidden_states": 2,         # per token per hidden_dim per layer
    "token_probs": 4,           # per token (float32)
}


# =============================================================================
# 特征管理器
# =============================================================================

class FeatureManager:
    """特征管理器 - 根据方法需求统一管理特征提取。
    
    重构后的设计：
    1. 从方法配置读取 base_features 需求
    2. 计算所有方法的基础特征并集
    3. 不包含任何方法特定的硬编码逻辑
    
    Usage:
        manager = FeatureManager(methods=["lapeigvals", "haloscope"])
        
        # 获取基础特征需求
        base_features = manager.get_required_base_features()
        
        # 转换为提取器配置
        config = manager.to_features_config()
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
            config_path: 可选的配置文件路径
            allow_full_attention: 是否允许提取完整注意力矩阵
        """
        self.methods = methods or []
        self.config_path = config_path
        self.allow_full_attention = allow_full_attention
        
        # 从配置文件加载的额外需求
        self._method_configs: Dict[str, Dict[str, Any]] = {}
        
        if config_path:
            self._load_method_configs(config_path)
    
    def _load_method_configs(self, config_dir: Path):
        """从配置目录加载方法配置。"""
        method_dir = config_dir / "method"
        if not method_dir.exists():
            return
        
        for config_file in method_dir.glob("*.yaml"):
            try:
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                method_name = config.get("name", config_file.stem)
                self._method_configs[method_name] = config
            except Exception as e:
                logger.warning(f"Failed to load {config_file}: {e}")
    
    def get_required_base_features(self) -> Set[str]:
        """获取所有方法需要的基础特征并集。
        
        Returns:
            基础特征名称集合
        """
        combined = set()
        
        for method in self.methods:
            # 优先从配置文件读取
            if method in self._method_configs:
                config = self._method_configs[method]
                base_features = config.get("base_features", [])
                combined.update(base_features)
            else:
                # 回退到注册表
                combined.update(get_method_base_features(method))
        
        # 安全检查：如果需要 full_attention 但未允许，记录警告
        if "full_attention" in combined and not self.allow_full_attention:
            logger.warning(
                "⚠️ Method(s) request full_attention but allow_full_attention=False. "
                "Full attention will NOT be extracted to prevent OOM."
            )
            combined.discard("full_attention")
        
        return combined
    
    def get_combined_requirements(self) -> FeatureRequirements:
        """获取所有方法的特征需求并集（向后兼容）。
        
        ⚠️ 此方法将在 v7.0 中废弃，请使用 get_required_base_features
        """
        if not self.methods:
            return FeatureRequirements(attention_diags=True)
        
        return compute_union_requirements(self.methods)
    
    def has_high_memory_features(self) -> bool:
        """检查是否包含高内存消耗的特征需求。"""
        base_features = self.get_required_base_features()
        return "full_attention" in base_features
    
    def estimate_memory_per_sample(
        self,
        seq_len: int = 2048,
        n_layers: int = 32,
        n_heads: int = 32,
        hidden_size: int = 4096,
    ) -> Dict[str, float]:
        """估算每个样本的内存需求。"""
        base_features = self.get_required_base_features()
        estimates = {}
        
        if "attention_diags" in base_features:
            bytes_needed = n_layers * seq_len * n_heads * MEMORY_ESTIMATES["attention_diags"]
            estimates["attention_diags_mb"] = bytes_needed / 1024**2
        
        if "attention_row_sums" in base_features:
            bytes_needed = n_layers * seq_len * n_heads * MEMORY_ESTIMATES["attention_row_sums"]
            estimates["attention_row_sums_mb"] = bytes_needed / 1024**2
        
        if "full_attention" in base_features:
            bytes_needed = n_layers * n_heads * seq_len * seq_len * MEMORY_ESTIMATES["full_attention"]
            estimates["full_attention_mb"] = bytes_needed / 1024**2
        
        if "hidden_states" in base_features:
            bytes_needed = n_layers * seq_len * hidden_size * MEMORY_ESTIMATES["hidden_states"]
            estimates["hidden_states_mb"] = bytes_needed / 1024**2
        
        if "token_probs" in base_features:
            bytes_needed = seq_len * MEMORY_ESTIMATES["token_probs"]
            estimates["token_probs_mb"] = bytes_needed / 1024**2
        
        estimates["total_mb"] = sum(estimates.values())
        estimates["total_gb"] = estimates["total_mb"] / 1024
        
        return estimates
    
    def to_features_config(self) -> Dict[str, Any]:
        """转换为 FeaturesConfig 兼容的字典格式。
        
        重构后的设计：
        - 基础特征提取不做预处理
        - hidden_states 使用完整序列（不 pooling）
        - 层选择由各方法自己在衍生特征计算时处理
        
        Returns:
            可用于配置 FeatureExtractor 的字典
        """
        base_features = self.get_required_base_features()
        
        # 确定需要提取哪些特征
        needs_attention = any(f in base_features for f in 
            ["attention_diags", "attention_row_sums", "full_attention"])
        needs_hidden = "hidden_states" in base_features
        needs_token_probs = "token_probs" in base_features
        store_full = "full_attention" in base_features
        
        return {
            # 注意力相关
            "attention_enabled": needs_attention,
            "attention_layers": "all",
            "store_full_attention": store_full,
            "extract_attention_row_sums": "attention_row_sums" in base_features,
            
            # 隐藏状态 - 基础特征提取不做预处理
            "hidden_states_enabled": needs_hidden,
            "hidden_states_layers": "all",  # 提取所有层，方法自己选择
            "hidden_states_pooling": "none",  # 不做 pooling，方法自己处理
            
            # Token 概率
            "token_probs_enabled": needs_token_probs,
        }
    
    def describe(self) -> str:
        """获取人类可读的需求描述。"""
        lines = ["=" * 50]
        lines.append("Feature Requirements Summary (v6.2)")
        lines.append("=" * 50)
        lines.append(f"Methods: {', '.join(self.methods)}")
        lines.append(f"Allow full attention: {self.allow_full_attention}")
        
        base_features = self.get_required_base_features()
        lines.append(f"\nRequired base features: {base_features}")
        
        lines.append("\nFeatures to extract:")
        for feat in sorted(base_features):
            desc = BASE_FEATURES.get(feat, {}).get("description", "")
            high_mem = BASE_FEATURES.get(feat, {}).get("high_memory", False)
            warning = " (⚠️ high memory)" if high_mem else ""
            lines.append(f"  ✓ {feat}: {desc}{warning}")
        
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
# 从配置加载方法需求（向后兼容）
# =============================================================================

def load_method_requirements_from_config(
    config_dir: Path
) -> Dict[str, FeatureRequirements]:
    """从方法配置目录加载特征需求。
    
    ⚠️ 此函数将在 v7.0 中废弃
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
        except Exception as e:
            logger.warning(f"Failed to load {config_file}: {e}")
    
    return requirements


def validate_features_for_method(
    features: Dict[str, Any],
    method_name: str
) -> tuple[bool, List[str]]:
    """验证提取的特征是否满足方法需求。"""
    base_features = get_method_base_features(method_name)
    
    # 映射基础特征名称到实际存储名称
    name_mapping = {
        "attention_diags": "attn_diags",
        "attention_row_sums": "attn_row_sums",
        "full_attention": "full_attention",
        "hidden_states": "hidden_states",
        "token_probs": "token_probs",
    }
    
    missing = []
    for bf in base_features:
        stored_name = name_mapping.get(bf, bf)
        if features.get(stored_name) is None:
            missing.append(bf)
    
    return len(missing) == 0, missing
