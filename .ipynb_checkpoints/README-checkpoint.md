# 特征管理重构说明

## 重构目标

将特征提取流程模块化、系统化：
- **主脚本足够简洁**：`generate_activations.py` 只做流程编排
- **具体功能由模块实现**：创建 `scripts/extraction/` 模块
- **统一特征管理**：保留 `utils/feature_manager.py`，删除重复实现

## 文件变更

### 新增文件

```
scripts/extraction/
├── __init__.py              # 模块入口，导出所有公共接口
├── data_loader.py           # 样本加载和过滤
├── output_manager.py        # 输出路径构建和结果整合
├── progress.py              # 进度跟踪
└── extraction_loop.py       # 主提取循环
```

### 修改文件

1. **`scripts/generate_activations.py`**
   - 从 632 行简化到约 150 行
   - 只保留流程编排逻辑
   - 具体实现委托给 `scripts.extraction` 模块

2. **`src/features/__init__.py`**
   - 移除对 `feature_registry.py` 和 `pipeline.py` 的引用
   - 保留核心 `extractor` 相关导出

### 删除文件

见 `DELETE_THESE_FILES.txt`

## 使用方法

### 步骤 1: 备份原项目

```bash
cp -r hallucination-detection-master hallucination-detection-master.bak
```

### 步骤 2: 解压重构文件

```bash
unzip refactored_feature_management.zip -d hallucination-detection-master/
```

### 步骤 3: 删除冗余文件

```bash
cd hallucination-detection-master

# 删除备用特征系统
rm -f src/features/feature_registry.py
rm -f src/features/pipeline.py

# 删除未使用的脚本目录
rm -rf scripts/features/

# 删除其他未使用文件
rm -f src/core/feature_accessor.py
rm -f test_abc.py
```

### 步骤 4: 验证

```bash
# 测试导入
python -c "from scripts.extraction import load_samples_from_splits, run_extraction_loop"
python -c "from utils.feature_manager import create_feature_manager"

# 运行提取（dry-run）
python scripts/generate_activations.py --help
```

## 架构说明

### 重构前

```
generate_activations.py (632 lines)
├── 解析配置逻辑
├── 样本加载逻辑
├── 输出目录构建
├── 进度跟踪
├── 特征提取循环
└── 结果整合

utils/feature_manager.py  ← 实际使用
src/features/feature_registry.py  ← 未使用（重复）
src/features/pipeline.py  ← 未使用（重复）
scripts/features/*  ← 未使用（重复）
```

### 重构后

```
generate_activations.py (~150 lines)  # 只做流程编排
├── scripts/extraction/
│   ├── data_loader.py      # 样本加载
│   ├── output_manager.py   # 输出管理
│   ├── progress.py         # 进度跟踪
│   └── extraction_loop.py  # 提取循环
└── utils/feature_manager.py  # 唯一的特征需求管理
```

## 接口说明

### 主脚本调用方式（不变）

```bash
# 训练集
python scripts/generate_activations.py dataset.name=ragtruth dataset.split_name=train

# 测试集  
python scripts/generate_activations.py dataset.name=ragtruth dataset.split_name=test

# 指定方法
python scripts/generate_activations.py methods=[lapeigvals,haloscope]
```

### 特征管理器使用

```python
from utils.feature_manager import create_feature_manager

# 创建管理器
manager = create_feature_manager(
    methods=["lapeigvals", "haloscope"],
    allow_full_attention=False,
)

# 获取合并需求
requirements = manager.get_combined_requirements()

# 转换为配置
config = manager.to_features_config()

# 内存预估
mem = manager.estimate_memory_per_sample(seq_len=2048)
print(f"Estimated memory: {mem['total_mb']:.1f} MB/sample")
```

### 提取模块使用

```python
from scripts.extraction import (
    load_samples_from_splits,
    build_output_dir,
    run_extraction_loop,
)

# 加载样本
samples = load_samples_from_splits(
    dataset_name="ragtruth",
    task_types_filter=["qa"],
    split="train"
)

# 构建输出目录
output_dir = build_output_dir(cfg, split_name="train")
```

## 兼容性

- **输出格式不变**：`features/`, `metadata.json`, `labels.pt`, `answers.json`
- **配置格式不变**：Hydra 配置完全兼容
- **下游脚本不变**：`train_probe.py`, `evaluate.py` 无需修改
