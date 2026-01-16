# 特征提取架构重设计

## 问题分析

当前架构存在的问题：

1. **基础特征与衍生特征混淆**
   - `laplacian_diags` 实际上是从 `attention_diags` 计算的，却被当作独立需求
   - `hidden_states` 的 pooling 策略被硬编码，而不是作为方法配置

2. **方法特殊处理硬编码在主代码中**
   - HaloScope 需要 `hidden_states_pooling: none` 被硬编码在 `feature_manager.py`
   - 违反了"主脚本保持简洁"的原则

3. **配置分散**
   - 特征需求定义在 `registry.py`
   - 提取配置在 `features/*.yaml`
   - 方法配置在 `method/*.yaml`
   - 缺乏统一管理

---

## 新架构设计

### 核心原则

```
基础特征（统一提取）→ 衍生特征（按需计算）
     ↓                      ↓
  最大化存储            方法自己处理
  不做预处理            配置驱动计算
```

### 1. 基础特征 (Base Features)

**只提取最原始的、不可分解的特征**：

| 基础特征 | 形状 | 说明 |
|---------|------|------|
| `attention_diags` | `[n_layers, n_heads, seq_len]` | 注意力对角线 |
| `attention_row_sums` | `[n_layers, n_heads, seq_len]` | 每行的和（用于计算 degree） |
| `full_attention` | `[n_layers, n_heads, seq_len, seq_len]` | 完整注意力矩阵（可选） |
| `hidden_states` | `[n_layers, seq_len, hidden_dim]` | 完整隐藏状态（不做 pooling） |
| `token_probs` | `[seq_len]` | token 概率 |

**关键**：基础特征提取时**不做任何预处理**（如 pooling、层选择等），保留完整数据。

### 2. 衍生特征配置 (Derived Feature Config)

每个方法在自己的配置文件中定义需要的衍生特征：

```yaml
# config/method/lapeigvals.yaml
name: lapeigvals
level: sample

# 基础特征需求
base_features:
  - attention_diags
  - attention_row_sums  # 用于计算 laplacian

# 衍生特征定义
derived_features:
  laplacian_diags:
    compute_fn: compute_laplacian_from_diags
    inputs: [attention_diags, attention_row_sums]
    params:
      scope: response_only
```

```yaml
# config/method/haloscope.yaml
name: haloscope
level: sample

# 基础特征需求
base_features:
  - hidden_states

# 衍生特征定义
derived_features:
  last_token_embedding:
    compute_fn: extract_last_token_embedding
    inputs: [hidden_states]
    params:
      layer_selection: middle  # 使用中间层
      token_selection: last    # 选择最后一个 token
```

```yaml
# config/method/hsdmvaf.yaml
name: hsdmvaf
level: sample

# 基础特征需求
base_features:
  - full_attention  # 需要完整矩阵

# 衍生特征定义
derived_features:
  mva_features:
    compute_fn: compute_mva_features
    inputs: [full_attention]
    params:
      layers: [-4, -3, -2, -1]  # 最后4层
      head_aggregation: mean
```

### 3. 特征提取流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 1: 基础特征提取                          │
│                  (generate_activations.py)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 读取 params.yaml 中的 methods 列表                            │
│  2. 收集所有方法的 base_features 需求（并集）                       │
│  3. 提取基础特征，保存到 features_individual/                      │
│                                                                  │
│  输出:                                                           │
│    - features_individual/{sample_id}.pt                         │
│      {                                                          │
│        "attention_diags": [n_layers, n_heads, seq_len],         │
│        "attention_row_sums": [n_layers, n_heads, seq_len],      │
│        "hidden_states": [n_layers, seq_len, hidden_dim],        │
│        "token_probs": [seq_len],                                │
│        "prompt_len": int,                                       │
│        "response_len": int,                                     │
│      }                                                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 2: 训练/预测                            │
│                     (train_probe.py)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 加载方法配置 (config/method/{method}.yaml)                    │
│  2. 加载基础特征 from features_individual/                        │
│  3. 根据 derived_features 配置计算衍生特征                         │
│  4. 训练/预测                                                    │
│                                                                  │
│  衍生特征计算由 DerivedFeatureComputer 统一处理                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4. 衍生特征计算器 (DerivedFeatureComputer)

```python
# src/features/derived_computer.py

class DerivedFeatureComputer:
    """统一的衍生特征计算器。
    
    根据方法配置中的 derived_features 定义，
    从基础特征计算方法需要的衍生特征。
    """
    
    # 注册所有可用的计算函数
    COMPUTE_FUNCTIONS = {
        "compute_laplacian_from_diags": compute_laplacian_from_diags,
        "extract_last_token_embedding": extract_last_token_embedding,
        "compute_mva_features": compute_mva_features,
        "compute_attention_entropy": compute_attention_entropy,
        # ...
    }
    
    def __init__(self, method_config: MethodConfig):
        self.method_config = method_config
        self.derived_config = method_config.derived_features
    
    def compute(self, base_features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """从基础特征计算所有需要的衍生特征。"""
        derived = {}
        
        for name, config in self.derived_config.items():
            fn = self.COMPUTE_FUNCTIONS[config.compute_fn]
            inputs = [base_features[k] for k in config.inputs]
            derived[name] = fn(*inputs, **config.params)
        
        return derived
```

### 5. 更新后的文件结构

```
config/
├── method/
│   ├── lapeigvals.yaml      # 包含 base_features + derived_features
│   ├── haloscope.yaml
│   ├── hsdmvaf.yaml
│   └── ...
├── features/
│   └── default.yaml         # 只包含提取参数 (max_length, batch_size)
└── config.yaml

src/features/
├── extractor.py             # 只提取基础特征，不做预处理
├── derived/
│   ├── attention_derived.py
│   ├── hidden_states_derived.py
│   └── token_probs_derived.py
├── derived_computer.py      # 新增：统一的衍生特征计算器
├── registry.py              # 重构：只定义基础特征类型
└── loader.py                # 加载基础特征
```

---

## 具体改动计划

### Phase 1: 重构配置文件

1. 更新 `config/method/*.yaml`，添加 `base_features` 和 `derived_features` 定义
2. 从 `FeatureRequirements` 中移除衍生特征（如 `laplacian_diags`）
3. 更新 `registry.py`，只保留基础特征定义

### Phase 2: 重构特征提取

1. 更新 `extractor.py`，提取完整基础特征（不做 pooling）
2. 更新 `feature_manager.py`，从方法配置读取 `base_features` 需求
3. 移除 HaloScope 等方法的硬编码特殊处理

### Phase 3: 添加衍生特征计算器

1. 创建 `src/features/derived_computer.py`
2. 整合现有的 `derived/*.py` 函数
3. 更新 `train_probe.py`，使用 `DerivedFeatureComputer`

### Phase 4: 更新方法实现

1. 更新各方法的 `extract_method_features()` 方法
2. 使用 `DerivedFeatureComputer` 替代内部计算

---

## 预期收益

1. **主脚本保持简洁**：特殊处理逻辑移到配置文件
2. **配置驱动**：添加新方法只需添加配置文件
3. **统一管理**：所有特征计算在 `derived_computer.py` 中注册
4. **灵活性**：方法可以自定义层选择、pooling 等参数
5. **可复用**：基础特征提取一次，多方法共享
