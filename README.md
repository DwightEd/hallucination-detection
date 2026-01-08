# Hallucination Detection Framework (Optimized)

基于LLM内部状态的幻觉检测框架，已优化内存使用和处理效率。

## 核心优化

### 1. 内存与存储优化 (Priority: 紧急)
- **解决CUDA OOM问题**：batch_size=1处理，处理完立即释放
- **增量保存**：每处理1个样本立即保存为单独的.pt文件
- **异步IO**：使用ThreadPoolExecutor异步保存，不阻塞GPU处理
- **断点续传**：支持从中断位置继续处理，自动跳过已处理样本

### 2. 特征数据统筹管理
- **一次前向传播**：提取所有方法所需的基础特征
- **需求并集计算**：根据配置的方法自动计算所需特征并集
- **方法级需求定义**：每个方法在`config/method/`下定义自己的`required_features`

### 3. 架构系统化
- **utils/目录**：新增工具函数模块
  - `checkpoint.py`: 断点续传管理
  - `async_saver.py`: 异步保存器
  - `feature_manager.py`: 特征需求管理
- **方法独立运行**：支持直接运行方法文件，不依赖完整DVC流程

### 4. 方法集成
- **原生hypergraph支持**：完整集成超图方法
- **统一接口**：所有方法继承BaseMethod
- **标准化probe训练**：训练/非训练类方法统一管理

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 配置

1. 编辑 `params.yaml` 设置数据集、模型和方法
2. 更新模型路径指向你的本地模型

### 运行

```bash
# 完整流水线
dvc repro

# 仅特征提取（支持断点续传）
dvc repro generate_activations

# 强制重新运行
dvc repro -f generate_activations

# 查看进度
cat outputs/features/.../checkpoint.json
```

### 独立运行方法

```bash
# 直接测试单个方法
python -m src.methods.lapeigvals --features-dir outputs/features/... --output-dir results/

python -m src.methods.hypergraph --features-dir outputs/features/... --output-dir results/
```

## 目录结构

```
.
├── config/
│   ├── dataset/          # 数据集配置
│   ├── method/           # 方法配置（含required_features）
│   ├── model/            # 模型配置
│   └── config.yaml       # 主配置
├── scripts/
│   ├── generate_activations.py  # 特征提取（已优化）
│   ├── train_probe.py           # 探测器训练
│   └── evaluate.py              # 评估
├── src/
│   ├── core/             # 核心数据结构
│   ├── data/             # 数据集加载
│   ├── features/         # 特征提取器
│   ├── methods/          # 检测方法
│   └── utils/            # 工具函数
├── utils/                # 新增优化工具
│   ├── checkpoint.py     # 断点管理
│   ├── async_saver.py    # 异步保存
│   └── feature_manager.py # 特征需求管理
├── dvc.yaml              # DVC流水线
├── params.yaml           # 参数配置
└── README.md
```

## 方法特征需求

每个方法在配置文件中定义所需特征：

```yaml
# config/method/hypergraph.yaml
required_features:
  attention_diags: true
  full_attention: true  # GNN需要完整attention
  hidden_states: false
```

## 内存优化使用示例

```python
from utils.checkpoint import CheckpointManager
from utils.async_saver import MemoryEfficientSaver

# 初始化
checkpoint = CheckpointManager(output_dir)
checkpoint.initialize(total_samples=1000, config=cfg)

saver = MemoryEfficientSaver(output_dir, max_workers=2)

for sample in samples:
    if checkpoint.is_processed(sample.id):
        continue  # 跳过已处理
    
    # 提取特征
    features = extractor.extract(sample)
    
    # 异步保存，立即释放内存
    saver.save_and_release(sample.id, features_dict, metadata)
    
    # 标记完成
    checkpoint.mark_completed(sample.id)

# 等待所有保存完成
saver.finalize()

# 合并为lapeigvals格式
checkpoint.consolidate_features(output_dir / "features.pt")
```

## 支持的检测方法

| 方法 | 类型 | 需要训练 | 特征需求 |
|------|------|----------|----------|
| lapeigvals | 谱方法 | ✓ | attn_diags, laplacian_diags |
| entropy | 统计 | ✓ | attn_entropy, token_probs |
| lookback_lens | 注意力分析 | ✓ | attn_diags |
| hypergraph | GNN | ✓ | attn_diags, full_attention |
| ensemble | 集成 | ✓ | 多种特征 |

## 断点续传

系统会自动检测checkpoint并继续处理：

```bash
# 运行被中断后，重新运行会自动继续
python scripts/generate_activations.py dataset.name=ragtruth model=mistral_7b

# 查看进度
cat outputs/features/ragtruth_all/mistral_7b/seed_42/checkpoint.json
```

## License

MIT License
