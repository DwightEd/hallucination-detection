# 配置系统指南 (v6.3)

## 回答你的问题

### Q1: 取特定 token 的特征现在能实现吗？

**能！** 使用 `extract_token_embedding` 函数：

```yaml
# config/method/your_method.yaml
features:
  derived:
    specific_tokens:
      fn: extract_token_embedding
      from: hidden_states
      token: specific           # 指定模式
      token_indices: [10, 20, 30]  # 要提取的 token 索引
      layers: last_n
      n_layers: 4
```

**支持的 token 选择方式**：

| 值 | 说明 | 输出形状 |
|---|------|---------|
| `last` | 最后一个 token | `[n_layers, hidden_dim]` |
| `first` | 第一个 token | `[n_layers, hidden_dim]` |
| `mean` | 平均池化 | `[n_layers, hidden_dim]` |
| `max` | 最大池化 | `[n_layers, hidden_dim]` |
| `all` | 所有 token | `[n_layers, seq_len, hidden_dim]` |
| `specific` | 指定索引 | `[n_layers, n_tokens, hidden_dim]` |
| `range` | 范围 `[start, end]` | `[n_layers, range_len, hidden_dim]` |

---

### Q2: 现在需要配置哪些文件？

**只需要一个文件：`config/method/{method}.yaml`**

```
config/
├── method/
│   ├── haloscope.yaml      # ← 只需要这个！
│   ├── lapeigvals.yaml
│   └── ...
└── (features/*.yaml 已废弃，可删除)
```

---

### Q3: 配置文件有哪些字段？

```yaml
# =============================================================================
# 完整配置模板 (v6.3)
# =============================================================================

name: method_name               # 必需：方法名称
cls_path: src.methods.xxx.Xxx   # 必需：类路径

# ============ 基础信息 ============
level: sample                   # sample / token / both
classifier: logistic            # logistic / mlp / rf / svm
random_seed: 42                 # 随机种子（可选）

# ============ 特征配置 ============
features:
  base:                         # 需要的基础特征
    - hidden_states             # 可选: hidden_states, attention_diags, 
    - attention_diags           #       attention_row_sums, full_attention, token_probs
  
  derived:                      # 衍生特征定义（可选）
    feature_name:
      fn: compute_function      # 计算函数名
      from: base_feature        # 输入的基础特征
      token: last               # token 选择方式
      token_indices: []         # 当 token=specific 时
      layers: all               # 层选择方式
      n_layers: 4               # 当 layers=last_n/first_n/middle 时
      scope: full               # full / response_only / prompt_only

# ============ 方法参数 ============
params:
  # 方法特定的参数，任意结构
  key: value
```

**字段说明**：

| 字段 | 是否必需 | 说明 |
|------|---------|------|
| `name` | ✓ | 方法名称 |
| `cls_path` | ✓ | 实现类路径 |
| `level` | | 检测级别，默认 `sample` |
| `classifier` | | 分类器类型，默认 `logistic` |
| `features.base` | ✓ | 需要的基础特征列表 |
| `features.derived` | | 衍生特征定义 |
| `params` | | 方法特定参数 |

---

### Q4: 基础特征有哪些？

| 基础特征 | 形状 | 说明 |
|---------|------|------|
| `hidden_states` | `[n_layers, seq_len, hidden_dim]` | 完整隐藏状态 |
| `attention_diags` | `[n_layers, n_heads, seq_len]` | 注意力对角线 |
| `attention_row_sums` | `[n_layers, n_heads, seq_len]` | 注意力行和 |
| `full_attention` | `[n_layers, n_heads, seq_len, seq_len]` | 完整注意力矩阵 ⚠️ |
| `token_probs` | `[seq_len]` | token 生成概率 |

---

### Q5: 衍生特征计算函数有哪些？

| 函数名 | 输入 | 说明 |
|-------|------|------|
| `extract_token_embedding` | hidden_states | 灵活的 token/层 选择 |
| `compute_laplacian_from_diags` | attention_diags, attention_row_sums | Laplacian 对角线 |
| `compute_attention_entropy` | full_attention | 注意力熵 |
| `compute_token_entropy` | token_probs | token 熵 |
| `compute_mva_features` | full_attention | Multi-View 特征 |
| `compute_lookback_ratio` | full_attention | Lookback 比率 |

---

### Q6: 如何新增字段？

**1. 添加新的基础特征**

编辑 `src/features/registry.py`：

```python
BASE_FEATURES["your_feature"] = {
    "description": "你的特征描述",
    "shape": "[n_layers, seq_len]",
}

METHOD_BASE_FEATURES["your_method"] = {"your_feature", "hidden_states"}
```

**2. 添加新的衍生特征计算函数**

编辑 `src/features/derived_computer.py`：

```python
@ComputeFunctionRegistry.register("your_compute_function")
def your_compute_function(
    base_feature: torch.Tensor,
    your_param: int = 10,
    **kwargs  # 自动接收 prompt_len, response_len
) -> torch.Tensor:
    """你的计算逻辑。"""
    return result
```

**3. 添加新的方法参数**

直接在 `params:` 下添加，无需修改代码：

```yaml
params:
  your_new_param: value
```

方法实现中通过 `self.params["your_new_param"]` 访问。

---

### Q7: 配置会有冲突吗？

**v6.3 设计避免了冲突**：

1. **单一来源**：每个方法只有一个配置文件
2. **层级清晰**：
   - 全局配置 (`params.yaml`)：`max_length`, `batch_size`
   - 方法配置 (`config/method/*.yaml`)：方法特定设置
3. **自动推断**：基础特征需求从方法配置自动计算并集

**冲突检测**：

```python
from src.core.method_config import MethodConfigManager

# 验证配置
valid, issues = MethodConfigManager.validate_config("haloscope")
if not valid:
    print("Issues:", issues)
```

---

## 完整示例

### 示例 1：取第 10-20 个 token 的中间层 embedding

```yaml
# config/method/my_method.yaml
name: my_method
cls_path: src.methods.my_method.MyMethod
level: sample

features:
  base:
    - hidden_states
  
  derived:
    selected_tokens:
      fn: extract_token_embedding
      from: hidden_states
      token: range
      token_indices: [10, 20]   # 第 10-20 个 token
      layers: middle
      n_layers: 8
      scope: response_only      # 只看 response 部分
```

### 示例 2：同时需要注意力和隐藏状态

```yaml
name: combined_method
cls_path: src.methods.combined.CombinedMethod
level: sample

features:
  base:
    - hidden_states
    - attention_diags
    - attention_row_sums
  
  derived:
    embedding:
      fn: extract_token_embedding
      from: hidden_states
      token: last
      layers: last_n
      n_layers: 4
    
    laplacian:
      fn: compute_laplacian_from_diags
      from: attention_diags
      scope: response_only
```

---

## 配置管理 API

```python
from src.core.method_config import (
    MethodConfigManager,
    load_method_config,
    get_all_base_features,
    generate_extraction_config,
)

# 加载单个方法配置
config = load_method_config("haloscope")
print(config.base_features)  # {'hidden_states'}

# 获取多个方法的基础特征并集
features = get_all_base_features(["haloscope", "lapeigvals"])
# {'hidden_states', 'attention_diags', 'attention_row_sums'}

# 生成特征提取配置
extraction_config = generate_extraction_config(
    methods=["haloscope", "lapeigvals"],
    max_length=8192,
    batch_size=2,
)

# 获取衍生特征计算器
computer = config.get_derived_computer()
derived = computer.compute(base_features, prompt_len=100, response_len=200)
```

---

## 迁移指南

### 从 v6.2 迁移

1. **删除** `config/features/*.yaml`（已废弃）
2. **更新** `config/method/*.yaml` 为新格式
3. **移除** `required_features` 和 `base_features` 字段，使用 `features.base`
4. **移除** `derived_features` 字段，使用 `features.derived`

### 新格式对比

```yaml
# 旧格式 (v6.2)
base_features:
  - hidden_states
derived_features:
  last_token_embedding:
    compute_fn: extract_last_token_embedding
    inputs: [hidden_states]
    params:
      layer_selection: middle

# 新格式 (v6.3)
features:
  base:
    - hidden_states
  derived:
    last_token_embedding:
      fn: extract_token_embedding
      from: hidden_states
      token: last
      layers: middle
```
