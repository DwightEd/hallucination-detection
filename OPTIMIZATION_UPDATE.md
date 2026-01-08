# Hallucination Detection Framework - 优化更新

## 更新概述

本次更新针对以下问题进行了优化：

1. **数据过滤** - `models` 参数现在正确传递到数据加载
2. **内存优化** - `store_full_attention` 默认禁用，防止OOM
3. **多GPU支持** - 完整的多GPU加载和分布策略
4. **缺失配置** - 新增 haloscope 和 hsdmvaf 方法配置
5. **特征管理** - 统一的多方法特征需求计算

## 修改文件清单

### 基础层
| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `src/utils/device.py` | 重写 | 多GPU设备管理、内存查询、设备同步 |
| `src/utils/__init__.py` | 更新 | 导出新的多GPU函数 |

### 模型层
| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `src/models/loader.py` | 重写 | 多GPU加载、设备映射策略、LoadedModel增强 |

### 特征层
| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `src/features/extractor.py` | 重写 | 内存安全控制、逐层清理、内存预估 |
| `src/features/__init__.py` | 更新 | 导出 `create_extractor_from_requirements` |
| `utils/feature_manager.py` | 重写 | 方法需求定义、内存估算、安全控制 |

### 方法层
| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `src/methods/haloscope.py` | 重命名 | 原 `haloscope`（无扩展名） |
| `src/methods/hsdmvaf.py` | 重命名 | 原 `hsdmvaf`（无扩展名） |
| `src/methods/__init__.py` | 更新 | 导出 HaloScopeDetector, HSDMVAFDetector |

### 配置层
| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `config/method/haloscope.yaml` | 新建 | HaloScope方法配置（NeurIPS'24） |
| `config/method/hsdmvaf.yaml` | 新建 | HSDMVAF方法配置 |
| `config/dataset/ragtruth.yaml` | 重写 | 详细的模型过滤说明 |
| `config/features/default.yaml` | 重写 | 内存优化默认配置 |
| `config/features/with_full_attentions.yaml` | 重写 | 高内存配置带警告 |
| `config/model/default.yaml` | 重写 | 多GPU配置模板 |
| `config/config.yaml` | 重写 | 主配置增强，使用示例 |

### 脚本层
| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `scripts/generate_activations.py` | 重写 | 系统信息、模型过滤日志、多GPU支持 |

## 使用示例

### 1. 基本使用
```bash
python scripts/generate_activations.py dataset.name=ragtruth model=mistral_7b
```

### 2. 按源模型过滤数据
```bash
# 仅使用GPT-4生成的响应
python scripts/generate_activations.py dataset.models=[gpt-4]

# 使用GPT-4和Llama-2
python scripts/generate_activations.py dataset.models=[gpt-4,llama-2-7b-chat]
```

### 3. 多方法特征提取
```bash
# 自动计算lapeigvals和entropy的特征需求并集
python scripts/generate_activations.py methods=[lapeigvals,entropy]

# 使用HaloScope（无监督方法）
python scripts/generate_activations.py methods=[haloscope]
```

### 4. 启用完整注意力（hypergraph方法）
```bash
python scripts/generate_activations.py \
    methods=[hypergraph] \
    features=with_full_attentions \
    allow_full_attention=true
```

### 5. 多GPU加载
```bash
# 使用自动分布
python scripts/generate_activations.py \
    model.multi_gpu.enabled=true \
    model.multi_gpu.strategy=auto

# 指定每张卡的显存限制
python scripts/generate_activations.py \
    model.multi_gpu.enabled=true \
    model.multi_gpu.max_memory.0=20GB \
    model.multi_gpu.max_memory.1=20GB
```

### 6. 量化加载（节省显存）
```bash
python scripts/generate_activations.py model=qwen2.5_7b_4bit
```

## 关键安全机制

### 1. Full Attention 保护
- `allow_full_attention` 默认为 `false`
- 即使方法需要完整注意力，也必须显式启用
- 防止意外OOM

### 2. 内存预估
- 提取前显示内存预估
- 警告高内存配置

### 3. 逐层清理
- 每层处理完立即释放GPU张量
- 减少峰值内存使用

## 新增方法

### HaloScope (NeurIPS'24)
- 基于SVD的无监督幻觉检测
- 仅需 hidden_states，不需要 full_attention
- 内存友好

### HSDMVAF
- 多视角注意力特征
- Transformer编码器 + CRF
- 支持token级别检测
- 使用对角线特征代替完整注意力

## 注意事项

1. **首次使用**：确保所有依赖已安装
2. **多GPU**：需要 `accelerate` 库支持
3. **4bit量化**：需要 `bitsandbytes` 库
4. **检查点恢复**：默认启用，中断后可继续

## 故障排除

### OOM错误
1. 减小 `features.max_length`
2. 启用量化 `model.load_in_4bit=true`
3. 使用多GPU分布

### 无样本加载
1. 检查 `dataset.models` 过滤器
2. 查看原始数据中的可用模型
3. 设置 `dataset.models=null` 使用所有模型

### 导入错误
1. 确保在项目根目录运行
2. 检查 PYTHONPATH 设置
