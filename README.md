# Hallucination Detection Framework

基于 lapeigvals (EMNLP 2025) 的可扩展幻觉检测框架。

## 🚀 快速开始

### 安装

```bash
# 解压所有部分
unzip hallucination-detection-part1.zip -d hallucination-detection
unzip -o hallucination-detection-part2.zip -d hallucination-detection
unzip -o hallucination-detection-part3.zip -d hallucination-detection
unzip -o hallucination-detection-part4.zip -d hallucination-detection
unzip -o hallucination-detection-part5.zip -d hallucination-detection

cd hallucination-detection

# 安装依赖
pip install -r requirements.txt

# 或使用 pip install
pip install -e .
```

### 下载数据

```bash
# 下载 TruthfulQA
python scripts/download_data.py --dataset truthfulqa --output ./data

# 下载 HaluEval
python scripts/download_data.py --dataset halueval --output ./data

# RAGTruth 需要手动下载
python scripts/download_data.py --dataset ragtruth --output ./data
```

### 运行实验

```bash
# 默认配置
python scripts/run_experiment.py

# 快速测试
python scripts/run_experiment.py experiment=quick_test

# 完整基准测试
python scripts/run_experiment.py experiment=full_benchmark

# 自定义参数
python scripts/run_experiment.py \
    dataset=truthfulqa \
    method=ensemble \
    model=qwen2.5_7b_4bit
```

## 📁 项目结构

```
hallucination-detection/
├── config/                     # Hydra 配置
│   ├── dataset/               # 数据集配置
│   ├── model/                 # 模型配置
│   ├── method/                # 方法配置
│   ├── llm_api/               # LLM API 配置
│   └── experiment/            # 实验配置
├── src/
│   ├── core/                  # 核心模块
│   ├── data/                  # 数据加载
│   ├── models/                # 模型加载
│   ├── features/              # 特征提取
│   ├── methods/               # 检测方法
│   └── evaluation/            # 评估模块
├── scripts/                   # 运行脚本
├── dvc.yaml                   # DVC 流水线
├── params.yaml                # DVC 参数
└── requirements.txt           # 依赖
```

## 🔧 核心功能

### 1. 统一层选择

```python
from src.core import parse_layers

parse_layers("all", 32)        # [0, 1, ..., 31]
parse_layers("last_n:4", 32)   # [28, 29, 30, 31]
parse_layers("first_n:2", 32)  # [0, 1]
parse_layers([24, 28, 31], 32) # [24, 28, 31]
```

### 2. 特征提取

```python
from src.models import load_model
from src.features import FeatureExtractor
from src.core import ModelConfig, FeaturesConfig

model = load_model(ModelConfig(
    name="Qwen/Qwen2.5-7B-Instruct",
    attn_implementation="eager",  # 必须！
))

extractor = FeatureExtractor(model, FeaturesConfig(
    mode="teacher_forcing",
    attention_layers="last_n:4",
))

features = extractor.extract(sample)
# features.attn_diags: [n_layers, n_heads, seq]
# features.laplacian_diags: [n_layers, n_heads, seq]
# features.token_probs: [response_len]
```

### 3. 检测方法

```python
from src.methods import create_method

# 单一方法
method = create_method("lapeigvals")
method.fit(features_list)
prediction = method.predict(features)

# 集成方法
method = create_method("auto_ensemble")
method.fit(features_list)  # 自动调权
```

### 4. LLM-as-Judge

```python
from src.evaluation import create_judge
from src.core import LLMAPIConfig

judge = create_judge(
    config=LLMAPIConfig(
        provider="qwen",
        model="qwen-turbo",
        api_key="your-key",
    ),
    mode="binary",
)

result = judge.judge(sample)
```

## 📊 支持的数据集

| 数据集 | 任务类型 | 说明 |
|--------|---------|------|
| RAGTruth | QA, Summary, Data2Text | RAG幻觉检测 |
| TruthfulQA | QA | 真实性评估 |
| HaluEval | QA, Summary, Dialogue | 综合幻觉评估 |

## 🧪 检测方法

| 方法 | 说明 | 关键特征 |
|------|------|----------|
| lapeigvals | Laplacian特征值 | laplacian_diags, attn_diags |
| lookback_lens | 注意力比率 | attn_diags, attn_entropy |
| entropy | 熵统计 | token_entropy, attn_entropy |
| perplexity | 困惑度 | token_probs |
| ensemble | 组合方法 | voting/stacking/concat |
| auto_ensemble | 自动加权集成 | 基于CV分数 |

## 🔄 DVC 流水线

```bash
# 初始化 DVC
dvc init

# 运行完整流水线
dvc repro

# 修改参数后重新运行
dvc repro -f

# 查看指标
dvc metrics show
```

## ⚙️ 配置示例

### 模型配置 (config/model/qwen2.5_7b.yaml)

```yaml
name: Qwen/Qwen2.5-7B-Instruct
attn_implementation: eager  # 必须！
dtype: bfloat16
load_in_4bit: false
device_map: auto
```

### 实验配置 (config/experiment/default.yaml)

```yaml
defaults:
  - /dataset: ragtruth
  - /model: qwen2.5_7b
  - /method: lapeigvals

features:
  attention_layers: "last_n:4"
  hidden_states_layers: "last_n:4"

evaluation:
  use_llm_judge: false
```

## 📈 API 支持

- **Qwen API** (DashScope): `provider: qwen`
- **OpenAI API**: `provider: openai`
- **本地 Ollama**: `provider: openai_compatible`

```bash
# 设置环境变量
export DASHSCOPE_API_KEY=your-key
export OPENAI_API_KEY=your-key
```

## 📝 License

MIT License
