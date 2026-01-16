"""Feature extraction and loading for hallucination detection.

v6.2 架构:
- 基础特征: 直接从模型提取的原始特征
- 衍生特征: 从基础特征计算得到的方法特定特征

Modules:
- extractor: Feature extraction from model (requires model inference)
- loader: Feature loading from disk (no model required)
- registry: Unified feature requirements definitions
- derived_computer: Derived feature computation
- hallucination_spans: Token-level label processing

Usage:
    # Feature extraction (requires model)
    from src.features import FeatureExtractor, create_extractor
    extractor = create_extractor(model, config)
    features = extractor.extract(sample)
    
    # Feature loading (from disk)
    from src.features import FeatureLoader, load_features_for_method
    features_list, samples = load_features_for_method(
        features_dir,
        method_name="lapeigvals"
    )
    
    # Derived feature computation (v6.2)
    from src.features import DerivedFeatureComputer
    computer = DerivedFeatureComputer.from_method_config("haloscope")
    derived = computer.compute(base_features, prompt_len, response_len)
    
    # Feature requirements
    from src.features import get_method_base_features
    base_features = get_method_base_features("lapeigvals")
"""

from .extractor import (
    FeatureExtractor,
    create_extractor,
    create_extractor_from_requirements,
)

from .loader import (
    FeatureLoader,
    load_features,
    load_features_for_method,
    split_features_by_split,
    METHOD_REQUIRED_FEATURES,
    # 衍生特征辅助函数
    compute_laplacian_from_diags,
    compute_entropy_from_attention,
)

from .registry import (
    # 新 API (v6.2)
    BASE_FEATURES,
    DERIVED_FEATURES,
    METHOD_BASE_FEATURES,
    get_method_base_features,
    get_combined_base_features,
    needs_full_attention,
    needs_hidden_states,
    # 向后兼容
    FeatureRequirements,
    METHOD_FEATURE_REQUIREMENTS,
    DERIVED_FEATURE_DEPS,
    get_method_requirements,
    get_method_feature_set,
    get_combined_requirements,
    is_feature_derived,
    get_feature_dependencies,
)

from .derived_computer import (
    DerivedFeatureComputer,
    DerivedFeatureConfig,
    ComputeFunctionRegistry,
    compute_derived_feature,
    list_compute_functions,
)

from .hallucination_spans import (
    extract_hallucination_info_from_sample,
    calculate_hallucination_token_spans,
    get_token_hallucination_labels,
)

__all__ = [
    # Extractor
    "FeatureExtractor",
    "create_extractor",
    "create_extractor_from_requirements",
    
    # Loader
    "FeatureLoader",
    "load_features",
    "load_features_for_method",
    "split_features_by_split",
    "METHOD_REQUIRED_FEATURES",
    
    # Registry - New API (v6.2)
    "BASE_FEATURES",
    "DERIVED_FEATURES",
    "METHOD_BASE_FEATURES",
    "get_method_base_features",
    "get_combined_base_features",
    "needs_full_attention",
    "needs_hidden_states",
    
    # Registry - Legacy (for backward compatibility)
    "FeatureRequirements",
    "METHOD_FEATURE_REQUIREMENTS",
    "DERIVED_FEATURE_DEPS",
    "get_method_requirements",
    "get_method_feature_set",
    "get_combined_requirements",
    "is_feature_derived",
    "get_feature_dependencies",
    
    # Derived Feature Computer (v6.2)
    "DerivedFeatureComputer",
    "DerivedFeatureConfig",
    "ComputeFunctionRegistry",
    "compute_derived_feature",
    "list_compute_functions",
    
    # Derived feature helpers
    "compute_laplacian_from_diags",
    "compute_entropy_from_attention",
    
    # Hallucination Spans
    "extract_hallucination_info_from_sample",
    "calculate_hallucination_token_spans",
    "get_token_hallucination_labels",
]
