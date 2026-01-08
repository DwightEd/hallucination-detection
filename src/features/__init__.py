"""Feature extraction module for hallucination detection.

模块结构：
- feature_registry: 特征需求注册表，定义各方法的特征需求
- base/: 基础特征提取器 (full_attention, hidden_states, token_probs)
- derived/: 派生特征提取器 (attention_diags, laplacian_diags, etc.)
- pipeline: 两阶段特征提取管线
- extractor: 原始特征提取器（向后兼容）

Usage:
    # 新的模块化 API
    from src.features.feature_registry import get_method_requirements
    from src.features.pipeline import FeatureExtractionPipeline
    
    # 分析方法需求
    req = get_method_requirements("lapeigvals")
    
    # 创建提取管线
    pipeline = FeatureExtractionPipeline(config)
    pipeline.extract_base_features(model, samples)
    pipeline.compute_derived_features()
    
    # 旧的 API（向后兼容）
    from src.features import FeatureExtractor, create_extractor
    extractor = FeatureExtractor(model, config)
    features = extractor.extract(sample)
"""

# =====================================================================
# 旧 API（向后兼容）
# =====================================================================

# Import main extractor class
from .extractor import (
    FeatureExtractor,
    create_extractor,
    create_extractor_from_requirements,
)

# Import tensor operations from src.utils (the canonical location)
# These are re-exported here for convenience
from src.utils import (
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

# Import hallucination span utilities
from .hallucination_spans import (
    calculate_hallucination_token_spans,
    get_token_hallucination_labels,
    extract_hallucination_info_from_sample,
    calculate_hallucination_labels_for_input,
    needs_llm_judge_for_spans,
    HallucinationSpanInfo,
)

# =====================================================================
# 新 API：模块化特征系统
# =====================================================================

from .feature_registry import (
    # Core types
    FeatureScope,
    LayerSelection,
    HeadSelection,
    BaseFeatureType,
    DerivedFeatureType,
    
    # Configuration classes
    DerivedFeatureConfig,
    BaseFeatureConfig,
    MethodFeatureRequirements,
    
    # Functions
    get_method_requirements,
    compute_union_requirements,
    get_base_features_needed,
    should_store_full_attention,
    describe_requirements,
    
    # Registry
    METHOD_REQUIREMENTS,
    DERIVED_TO_BASE_DEPS,
)

from .pipeline import (
    FeatureExtractionPipeline,
    PipelineConfig,
    create_pipeline,
    analyze_method_requirements,
)

# =====================================================================
# 导出列表
# =====================================================================

__all__ = [
    # ===============================
    # Legacy API (backward compatible)
    # ===============================
    
    # Main extractor
    "FeatureExtractor",
    "create_extractor",
    "create_extractor_from_requirements",
    
    # Attention utilities (from src.utils)
    "extract_attention_diagonal",
    "compute_laplacian_diagonal",
    "compute_attention_entropy",
    "stack_layer_attentions",
    
    # Hidden state utilities (from src.utils)
    "pool_hidden_states",
    "stack_layer_hidden_states",
    
    # Token probability utilities (from src.utils)
    "compute_token_probs",
    "compute_token_entropy",
    "compute_top_k_probs",
    
    # Hallucination span utilities
    "calculate_hallucination_token_spans",
    "get_token_hallucination_labels",
    "extract_hallucination_info_from_sample",
    "calculate_hallucination_labels_for_input",
    "needs_llm_judge_for_spans",
    "HallucinationSpanInfo",
    
    # ===============================
    # New Modular API
    # ===============================
    
    # Feature Registry - Types
    "FeatureScope",
    "LayerSelection",
    "HeadSelection",
    "BaseFeatureType",
    "DerivedFeatureType",
    
    # Feature Registry - Configuration
    "DerivedFeatureConfig",
    "BaseFeatureConfig",
    "MethodFeatureRequirements",
    
    # Feature Registry - Functions
    "get_method_requirements",
    "compute_union_requirements",
    "get_base_features_needed",
    "should_store_full_attention",
    "describe_requirements",
    "METHOD_REQUIREMENTS",
    "DERIVED_TO_BASE_DEPS",
    
    # Pipeline
    "FeatureExtractionPipeline",
    "PipelineConfig",
    "create_pipeline",
    "analyze_method_requirements",
]
