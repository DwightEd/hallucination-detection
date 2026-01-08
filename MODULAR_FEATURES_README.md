# 模块化特征提取系统 - 更新说明

## 概述

本次更新实现了模块化的两阶段特征提取系统，解决了以下问题：
1. 各方法特征需求不明确
2. 特征提取代码重复
3. 内存管理困难
4. 接口不统一

## 新增文件结构

```
src/features/
├── feature_registry.py     # 特征需求注册表（核心）
├── pipeline.py             # 两阶段提取管线
├── base/                   # 基础特征提取器
│   ├── __init__.py
│   ├── attention.py        # 注意力提取
│   ├── hidden_states.py    # 隐藏状态提取
│   └── token_probs.py      # Token概率提取
└── derived/                # 派生特征提取器（按基础特征分组）
    ├── __init__.py
    ├── attention_derived.py    # 基于attention的派生特征
    ├── token_probs_derived.py  # 基于token_probs的派生特征
    └── hidden_states_derived.py # 基于hidden_states的派生特征

scripts/features/
├── extract_base_features.py    # Stage 1: 基础特征提取
├── compute_derived_features.py # Stage 2: 派生特征计算
└── analyze_requirements.py     # 方法需求分析工具
```

## 特征分类

### 基础特征（需要模型推理）
- `full_attention`: 完整注意力矩阵 [n_layers, n_heads, seq_len, seq_len]
- `hidden_states`: 隐藏状态 [n_layers, seq_len, hidden_dim]
- `token_probs`: Token概率 [seq_len]

### 派生特征（从基础特征计算）

**基于 attention 的派生特征** (`attention_derived.py`):
- `attention_diags`: 注意力对角线
- `laplacian_diags`: Laplacian对角线
- `attention_entropy`: 注意力熵
- `lookback_ratio`: Lookback比率（需要prompt attention）
- `mva_features`: Multi-View Attention特征

**基于 token_probs 的派生特征** (`token_probs_derived.py`):
- `token_entropy`: Token熵
- `token_confidence`: Token置信度
- `perplexity`: 困惑度

**基于 hidden_states 的派生特征** (`hidden_states_derived.py`):
- `pooled_states`: 池化后的隐藏状态
- `layer_similarity`: 层间相似度
- `svd_features`: SVD特征（用于HaloScope）

## 方法需求定义

每个方法在 `feature_registry.py` 中定义了明确的特征需求：

| 方法 | 基础特征 | 派生特征 | 需要prompt attention |
|------|---------|---------|---------------------|
| lapeigvals | - | attention_diags, laplacian_diags | 否 |
| lookback_lens | - | attention_diags, lookback_ratio | 是 |
| haloscope | hidden_states | - | 否 |
| hsdmvaf | full_attention | mva_features | 是 |
| hypergraph | full_attention | - | 是 |
| token_entropy | token_probs | attention_entropy | 否 |

## 使用方法

### 1. 分析方法需求
```bash
python scripts/features/analyze_requirements.py --methods lapeigvals lookback_lens haloscope
```

### 2. Stage 1: 提取基础特征
```bash
python scripts/features/extract_base_features.py \
    methods=[lapeigvals,lookback_lens] \
    dataset.name=ragtruth \
    model=mistral_7b
```

### 3. Stage 2: 计算派生特征
```bash
python scripts/features/compute_derived_features.py \
    methods=[lapeigvals,lookback_lens] \
    base_features_dir=outputs/features/ragtruth/mistral_7b/seed_42/QA/base
```

### 4. Python API
```python
from src.features import (
    get_method_requirements,
    compute_union_requirements,
    FeatureExtractionPipeline,
    PipelineConfig,
)

# 查看方法需求
req = get_method_requirements("lapeigvals")
print(req.needs_full_attention())  # False
print(req.needs_hidden_states())   # False

# 创建管线
config = PipelineConfig(
    methods=["lapeigvals", "lookback_lens"],
    output_dir="outputs/features/ragtruth/mistral_7b/seed_42/QA",
)
pipeline = FeatureExtractionPipeline(config)

# Stage 1
pipeline.extract_base_features(model, samples)

# Stage 2
pipeline.compute_derived_features()
```

## 向后兼容性

原有的 `FeatureExtractor` 和 `ExtractedFeatures` 保持不变，新旧系统可以并存：

```python
# 旧API（仍然可用）
from src.features import FeatureExtractor, create_extractor
extractor = create_extractor(model, config)
features = extractor.extract(sample)

# 新API
from src.features import get_method_requirements, FeatureExtractionPipeline
```

## 关键设计决策

1. **派生特征按基础特征分组**：便于维护，减少代码重复
2. **两阶段提取**：支持特征复用，减少重复计算
3. **方法需求注册表**：明确声明每个方法的依赖
4. **内存优化**：支持直接计算派生特征（不存储full_attention）
