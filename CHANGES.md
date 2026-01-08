# 变更说明 (2025-01-09)

## 主要更新

### 1. 目录结构优化
**变更前:**
```
outputs/features/{dataset}_{task_type}/{model}/seed_{seed}/
outputs/models/{dataset}_{task_type}/{model}/{method}/seed_{seed}/
```

**变更后:**
```
outputs/features/{dataset}/{model}/seed_{seed}/{task_type}/
outputs/models/{dataset}/{model}/seed_{seed}/{task_type}/{method}/
```

这个变更使得相同数据集和模型的不同任务类型结果更容易对比。

### 2. 特征需求并集计算
- DVC 流水线现在自动计算所有方法的特征需求并集
- 一次特征提取可以满足所有方法的需要
- 通过 `params.yaml` 中的 `methods` 列表配置
- 通过 `methods_str` 传递给 DVC 命令行

### 3. 数据分割后删除原始文件
- `params.yaml` 中新增 `split.delete_originals: true` 配置
- 分割后的数据保留所有原始信息（prompt, id, response, label, task_type等）
- 分割后自动删除原始数据文件以节省空间

### 4. 四方法完整支持
支持完整运行以下方法：
- `lapeigvals` - Laplacian特征值方法
- `lookback_lens` - Lookback Lens方法
- `haloscope` - SVD-based无监督方法
- `hsdmvaf` - Multi-View Attention特征方法

### 5. 命名一致性修复
- 修复 `full_attention` vs `full_attentions` 的命名不一致
- `ExtractedFeatures` 类使用 `full_attention`（单数）
- 存储文件使用 `full_attentions.pt`（复数）

## 文件修改列表

### 配置文件
- `dvc.yaml` - 更新目录结构和流水线配置
- `params.yaml` - 添加 `methods_str`, `delete_originals`, `allow_full_attention`
- `config/config.yaml` - 更新默认方法列表

### 脚本文件
- `scripts/generate_activations.py` - 修复特征名映射
- `scripts/train_probe.py` - 更新目录查找和特征加载逻辑
- `scripts/evaluate.py` - 更新目录查找和特征加载逻辑
- `scripts/aggregate_results.py` - 支持新旧两种目录结构

## 使用方法

### 运行完整流水线
```bash
# 使用 DVC 运行
dvc repro

# 或单独运行各阶段
dvc repro split_dataset
dvc repro generate_activations
dvc repro train_probe
dvc repro evaluate
dvc repro aggregate
```

### 查看特征需求
```bash
python scripts/features/analyze_requirements.py --methods lapeigvals lookback_lens haloscope hsdmvaf
```

### 手动运行特定任务
```bash
# 只运行 QA 任务
python scripts/generate_activations.py dataset.task_type=QA

# 运行所有任务类型
dvc repro generate_activations
```

## 注意事项

1. **显存要求**: `hsdmvaf` 方法需要 `allow_full_attention=true`，会增加显存使用
2. **磁盘空间**: `delete_originals=true` 会在分割后删除原始数据文件
3. **向后兼容**: `aggregate_results.py` 支持解析新旧两种目录结构
