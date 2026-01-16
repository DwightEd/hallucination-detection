"""Method Configuration Manager - 统一的方法配置管理。

v6.3 架构简化:
=============
每个方法只需要一个配置文件: config/method/{method}.yaml

配置文件结构:
```yaml
name: method_name
cls_path: src.methods.xxx.XxxMethod
level: sample                   # sample / token / both
classifier: logistic            # logistic / mlp / rf / svm
random_seed: 42

features:
  base:                         # 需要的基础特征
    - hidden_states
    - attention_diags
  
  derived:                      # 衍生特征定义
    feature_name:
      fn: compute_function_name
      from: base_feature_name
      # ... 其他参数

params:                         # 方法特定参数
  key: value
```

Usage:
    from src.core.method_config import MethodConfigManager
    
    # 加载单个方法配置
    config = MethodConfigManager.load("haloscope")
    
    # 获取所有方法的基础特征并集
    base_features = MethodConfigManager.get_combined_base_features(["haloscope", "lapeigvals"])
    
    # 获取方法的衍生特征计算器
    computer = config.get_derived_computer()
"""
from __future__ import annotations
from typing import Dict, List, Set, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# 配置数据类
# =============================================================================

@dataclass
class DerivedFeatureSpec:
    """衍生特征规格。"""
    name: str
    fn: str                          # 计算函数名
    from_feature: str                # 输入的基础特征
    token: str = "last"              # last / first / mean / all / specific
    token_indices: List[int] = field(default_factory=list)  # 当 token=specific
    layers: str = "all"              # all / last_n / first_n / middle / specific
    n_layers: Optional[int] = None   # 当 layers=last_n/first_n/middle
    layer_indices: List[int] = field(default_factory=list)  # 当 layers=specific
    scope: str = "full"              # full / response_only / prompt_only
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "DerivedFeatureSpec":
        """从字典创建。"""
        return cls(
            name=name,
            fn=data.get("fn", name),
            from_feature=data.get("from", ""),
            token=data.get("token", "last"),
            token_indices=data.get("token_indices", []),
            layers=data.get("layers", "all"),
            n_layers=data.get("n_layers"),
            layer_indices=data.get("layer_indices", []),
            scope=data.get("scope", "full"),
            extra_params={k: v for k, v in data.items() 
                         if k not in {"fn", "from", "token", "token_indices", 
                                      "layers", "n_layers", "layer_indices", "scope"}},
        )


@dataclass
class MethodConfig:
    """方法配置。"""
    name: str
    cls_path: str
    level: str = "sample"            # sample / token / both
    classifier: str = "logistic"     # logistic / mlp / rf / svm
    random_seed: Optional[int] = None
    
    # 特征配置
    base_features: Set[str] = field(default_factory=set)
    derived_features: Dict[str, DerivedFeatureSpec] = field(default_factory=dict)
    
    # 方法参数
    params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MethodConfig":
        """从字典创建。"""
        # 解析特征配置
        features_data = data.get("features", {})
        base_features = set(features_data.get("base", []))
        
        # 向后兼容：也支持旧的 base_features 字段
        if not base_features and "base_features" in data:
            base_features = set(data["base_features"])
        
        # 解析衍生特征
        derived_features = {}
        derived_data = features_data.get("derived", {})
        # 向后兼容
        if not derived_data and "derived_features" in data:
            derived_data = data["derived_features"]
        
        for name, spec in derived_data.items():
            if isinstance(spec, dict):
                derived_features[name] = DerivedFeatureSpec.from_dict(name, spec)
        
        return cls(
            name=data.get("name", ""),
            cls_path=data.get("cls_path", ""),
            level=data.get("level", "sample"),
            classifier=data.get("classifier", "logistic"),
            random_seed=data.get("random_seed"),
            base_features=base_features,
            derived_features=derived_features,
            params=data.get("params", {}),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "name": self.name,
            "cls_path": self.cls_path,
            "level": self.level,
            "classifier": self.classifier,
            "random_seed": self.random_seed,
            "features": {
                "base": list(self.base_features),
                "derived": {name: {
                    "fn": spec.fn,
                    "from": spec.from_feature,
                    "token": spec.token,
                    "layers": spec.layers,
                    "n_layers": spec.n_layers,
                    "scope": spec.scope,
                    **spec.extra_params,
                } for name, spec in self.derived_features.items()},
            },
            "params": self.params,
        }
    
    def get_derived_computer(self):
        """获取衍生特征计算器。"""
        from src.features.derived_computer import DerivedFeatureComputer, DerivedFeatureConfig
        
        configs = []
        for name, spec in self.derived_features.items():
            configs.append(DerivedFeatureConfig(
                name=name,
                compute_fn=spec.fn,
                inputs=[spec.from_feature],
                params={
                    "token_selection": spec.token,
                    "token_indices": spec.token_indices,
                    "layer_selection": spec.layers,
                    "n_layers_to_use": spec.n_layers,
                    "specific_layers": spec.layer_indices,
                    "scope": spec.scope,
                    **spec.extra_params,
                },
            ))
        
        return DerivedFeatureComputer(configs)


# =============================================================================
# 配置管理器
# =============================================================================

class MethodConfigManager:
    """方法配置管理器。"""
    
    _config_dir: Path = Path("config/method")
    _cache: Dict[str, MethodConfig] = {}
    
    # 默认基础特征映射（当配置文件未指定时使用）
    DEFAULT_BASE_FEATURES: Dict[str, Set[str]] = {
        "lapeigvals": {"attention_diags", "attention_row_sums"},
        "lookback_lens": {"attention_diags"},
        "haloscope": {"hidden_states"},
        "hsdmvaf": {"full_attention"},
        "hypergraph": {"full_attention"},
        "token_entropy": {"token_probs"},
        "act": {"attention_diags", "hidden_states"},
        "semantic_entropy_probes": {"hidden_states"},
    }
    
    @classmethod
    def set_config_dir(cls, path: Union[str, Path]):
        """设置配置目录。"""
        cls._config_dir = Path(path)
        cls._cache.clear()
    
    @classmethod
    def load(cls, method_name: str) -> MethodConfig:
        """加载方法配置。
        
        Args:
            method_name: 方法名称
            
        Returns:
            MethodConfig 实例
        """
        if method_name in cls._cache:
            return cls._cache[method_name]
        
        config_file = cls._config_dir / f"{method_name}.yaml"
        
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_file}, using defaults")
            config = cls._create_default_config(method_name)
        else:
            with open(config_file) as f:
                data = yaml.safe_load(f)
            config = MethodConfig.from_dict(data)
            
            # 如果没有指定基础特征，使用默认值
            if not config.base_features:
                config.base_features = cls.DEFAULT_BASE_FEATURES.get(method_name, set())
        
        cls._cache[method_name] = config
        return config
    
    @classmethod
    def _create_default_config(cls, method_name: str) -> MethodConfig:
        """创建默认配置。"""
        return MethodConfig(
            name=method_name,
            cls_path=f"src.methods.{method_name}.{method_name.title().replace('_', '')}Method",
            base_features=cls.DEFAULT_BASE_FEATURES.get(method_name, set()),
        )
    
    @classmethod
    def load_all(cls, method_names: List[str]) -> Dict[str, MethodConfig]:
        """加载多个方法配置。"""
        return {name: cls.load(name) for name in method_names}
    
    @classmethod
    def get_base_features(cls, method_name: str) -> Set[str]:
        """获取方法的基础特征需求。"""
        config = cls.load(method_name)
        return config.base_features
    
    @classmethod
    def get_combined_base_features(cls, method_names: List[str]) -> Set[str]:
        """获取多个方法的基础特征并集。"""
        combined = set()
        for name in method_names:
            combined |= cls.get_base_features(name)
        return combined
    
    @classmethod
    def list_available_methods(cls) -> List[str]:
        """列出所有可用的方法。"""
        methods = []
        for config_file in cls._config_dir.glob("*.yaml"):
            methods.append(config_file.stem)
        return sorted(methods)
    
    @classmethod
    def validate_config(cls, method_name: str) -> tuple[bool, List[str]]:
        """验证方法配置。
        
        Returns:
            (是否有效, 警告/错误列表)
        """
        issues = []
        config = cls.load(method_name)
        
        # 检查必需字段
        if not config.cls_path:
            issues.append(f"Missing cls_path for {method_name}")
        
        if not config.base_features:
            issues.append(f"No base_features defined for {method_name}")
        
        # 检查衍生特征的输入是否在基础特征中
        for name, spec in config.derived_features.items():
            if spec.from_feature and spec.from_feature not in config.base_features:
                issues.append(
                    f"Derived feature '{name}' requires '{spec.from_feature}' "
                    f"which is not in base_features"
                )
        
        return len(issues) == 0, issues
    
    @classmethod
    def clear_cache(cls):
        """清除缓存。"""
        cls._cache.clear()


# =============================================================================
# 特征提取配置生成
# =============================================================================

def generate_extraction_config(
    methods: List[str],
    max_length: int = 8192,
    batch_size: int = 2,
    allow_full_attention: bool = False,
) -> Dict[str, Any]:
    """从方法列表生成特征提取配置。
    
    Args:
        methods: 方法名称列表
        max_length: 最大序列长度
        batch_size: 批大小
        allow_full_attention: 是否允许完整注意力
        
    Returns:
        特征提取配置字典
    """
    base_features = MethodConfigManager.get_combined_base_features(methods)
    
    # 检查是否需要各种特征
    needs_attention = any(f in base_features for f in 
        ["attention_diags", "attention_row_sums", "full_attention"])
    needs_hidden = "hidden_states" in base_features
    needs_token_probs = "token_probs" in base_features
    needs_full_attention = "full_attention" in base_features
    
    # 安全检查
    if needs_full_attention and not allow_full_attention:
        logger.warning(
            "⚠️ Methods require full_attention but allow_full_attention=False. "
            "Full attention will NOT be extracted."
        )
        needs_full_attention = False
    
    return {
        "max_length": max_length,
        "batch_size": batch_size,
        
        # 注意力
        "attention_enabled": needs_attention,
        "attention_layers": "all",
        "store_full_attention": needs_full_attention,
        "extract_attention_row_sums": "attention_row_sums" in base_features,
        
        # 隐藏状态 - 始终提取完整数据，方法自己处理
        "hidden_states_enabled": needs_hidden,
        "hidden_states_layers": "all",
        "hidden_states_pooling": "none",
        
        # Token 概率
        "token_probs_enabled": needs_token_probs,
    }


# =============================================================================
# 配置模板生成
# =============================================================================

def generate_config_template(
    method_name: str,
    base_features: List[str],
    derived_features: Optional[Dict[str, Dict]] = None,
) -> str:
    """生成方法配置模板。
    
    Args:
        method_name: 方法名称
        base_features: 基础特征列表
        derived_features: 衍生特征定义
        
    Returns:
        YAML 格式的配置模板
    """
    template = f"""# =============================================================================
# {method_name.upper()} - Method Configuration
# =============================================================================

name: {method_name}
cls_path: src.methods.{method_name}.{method_name.title().replace('_', '')}Method

# 基础信息
level: sample                   # sample / token / both
classifier: logistic            # logistic / mlp / rf / svm
random_seed: 42

# =============================================================================
# 特征配置
# =============================================================================
features:
  # 需要的基础特征（从模型提取）
  base:
"""
    for bf in base_features:
        template += f"    - {bf}\n"
    
    if derived_features:
        template += """
  # 衍生特征定义（从基础特征计算）
  derived:
"""
        for name, spec in derived_features.items():
            template += f"    {name}:\n"
            template += f"      fn: {spec.get('fn', name)}\n"
            template += f"      from: {spec.get('from', base_features[0] if base_features else 'hidden_states')}\n"
            template += f"      token: {spec.get('token', 'last')}        # last / first / mean / all / specific\n"
            template += f"      layers: {spec.get('layers', 'all')}       # all / last_n / first_n / middle / specific\n"
            if spec.get('n_layers'):
                template += f"      n_layers: {spec['n_layers']}\n"
            template += f"      scope: {spec.get('scope', 'full')}        # full / response_only / prompt_only\n"
    
    template += """
# =============================================================================
# 方法参数
# =============================================================================
params:
  # 在此添加方法特定参数
  example_param: value
"""
    return template


# =============================================================================
# 便捷函数
# =============================================================================

def load_method_config(method_name: str) -> MethodConfig:
    """加载方法配置的便捷函数。"""
    return MethodConfigManager.load(method_name)


def get_all_base_features(methods: List[str]) -> Set[str]:
    """获取所有方法的基础特征并集。"""
    return MethodConfigManager.get_combined_base_features(methods)
