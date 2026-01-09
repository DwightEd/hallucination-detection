# Hallucination Detection Framework

A unified framework for detecting hallucinations in Large Language Model outputs using internal model states.

## Features

- **Multiple Detection Methods**: LapEigvals, Lookback Lens, HaloScope, HSDMVAF, Hypergraph, and more
- **Memory-Optimized**: Batch-size-1 processing with async saving, checkpoint/resume support
- **DVC Pipeline**: Reproducible experiments with automatic matrix expansion
- **Performance Tracking**: Training time, peak memory, model size metrics
- **Unified Evaluation**: Train/test split metrics with comprehensive comparison reports

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/hallucination-detection.git
cd hallucination-detection

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Edit `params.yaml` to configure your experiments:

```yaml
# Datasets
datasets:
  - name: ragtruth

# Task types (RAGTruth has 3)
task_types:
  - QA
  - Summary
  - Data2txt

# Models
models:
  - name: mistral_7b
    short_name: Mistral-7B-Instruct-v0.3

# Detection methods
methods:
  - lapeigvals
  - lookback_lens
  - haloscope
  - hsdmvaf
```

### Run the Pipeline

```bash
# Full pipeline
dvc repro

# Individual stages
dvc repro split_dataset
dvc repro generate_activations
dvc repro train_probe
dvc repro evaluate
dvc repro aggregate
```

## Project Structure

```
.
├── config/                    # Hydra configuration files
│   ├── config.yaml           # Main config
│   ├── dataset/              # Dataset configs
│   ├── method/               # Method configs
│   ├── model/                # Model configs
│   └── features/             # Feature extraction configs
├── scripts/                   # Entry point scripts
│   ├── split_dataset.py      # Data splitting
│   ├── generate_activations.py  # Feature extraction
│   ├── train_probe.py        # Probe training
│   ├── evaluate.py           # Evaluation
│   └── aggregate_results.py  # Results aggregation
├── src/                       # Source code
│   ├── core/                 # Core types and utilities
│   ├── data/                 # Dataset loaders
│   ├── features/             # Feature extractors
│   ├── methods/              # Detection methods
│   ├── models/               # Model loaders
│   ├── evaluation/           # Evaluation metrics
│   └── utils/                # Utilities (memory, device, metrics)
├── utils/                     # Pipeline utilities
│   ├── checkpoint.py         # Checkpoint management
│   ├── async_saver.py        # Async feature saving
│   └── feature_manager.py    # Feature requirement management
├── dvc.yaml                   # DVC pipeline definition
├── params.yaml                # Experiment parameters
└── requirements.txt           # Python dependencies
```

## Output Structure

```
outputs/
├── splits/{dataset}/          # Data splits
│   ├── train.json
│   └── test.json
├── features/{dataset}/{model}/seed_{seed}/{task_type}/
│   ├── features/             # Extracted features
│   ├── answers.json          # Sample metadata
│   └── metadata.json
├── models/{dataset}/{model}/seed_{seed}/{task_type}/{method}/
│   ├── probe/                # Trained model
│   │   ├── model.pkl
│   │   └── train_metrics.json  # Training metrics + performance
│   └── eval_results.json     # Evaluation results
└── results/
    ├── summary.json          # Complete results
    ├── comparison.csv        # Method comparison table
    └── performance.csv       # Performance metrics
```

## Detection Methods

| Method | Type | Description | Required Features |
|--------|------|-------------|-------------------|
| **lapeigvals** | Spectral | Laplacian eigenvalue analysis | attn_diags, laplacian_diags |
| **lookback_lens** | Attention | Context vs generated ratio | attn_diags |
| **haloscope** | SVD | Unsupervised SVD on hidden states | hidden_states |
| **hsdmvaf** | Multi-view | Multi-view attention features | attn_diags, attn_entropy, full_attention |
| **hypergraph** | GNN | Hypergraph neural network | attn_diags, full_attention |
| **token_entropy** | Statistical | Token probability entropy | token_probs, token_entropy |

## Performance Metrics

The framework tracks the following metrics for each method:

- **Training Time**: Wall-clock time for probe training
- **Peak CPU Memory**: Maximum CPU memory usage (MB)
- **Peak GPU Memory**: Maximum GPU memory usage (MB)
- **Model Size**: Serialized model file size (MB)

View in results:
```bash
cat outputs/results/performance.csv
```

## Memory Optimization

The framework includes several memory optimizations:

1. **Batch-size-1 Processing**: Process one sample at a time to avoid OOM
2. **Async Saving**: Save features asynchronously to not block GPU
3. **Checkpoint/Resume**: Automatically resume from interruptions
4. **Feature Union**: Calculate feature requirements union to minimize forward passes

Enable full attention (high memory) only when needed:
```yaml
# params.yaml
allow_full_attention: true  # Required for hsdmvaf, hypergraph
```

## Results Comparison

After running the pipeline, view method comparison:

```bash
# View summary
cat outputs/results/summary.json | python -m json.tool

# View comparison table
cat outputs/results/comparison.csv
```

Example output:
```
════════════════════════════════════════════════════════════════════════════════════════
METHOD COMPARISON (Train vs Test)
════════════════════════════════════════════════════════════════════════════════════════
Method                  N │ Train AUROC  Train AUPR │  Test AUROC   Test AUPR │  Time(s)  Size(MB)
────────────────────────────────────────────────────────────────────────────────────────
haloscope               3 │      0.8234      0.7856 │      0.7923      0.7541 │     2.34      0.12
hsdmvaf                 3 │      0.8567      0.8123 │      0.8234      0.7890 │     3.21      0.18
lapeigvals              3 │      0.8123      0.7654 │      0.7856      0.7432 │     1.89      0.09
lookback_lens           3 │      0.7890      0.7456 │      0.7654      0.7234 │     1.45      0.07
────────────────────────────────────────────────────────────────────────────────────────
```

## API Usage

### Standalone Method Evaluation

```python
from src.methods import create_method
from src.core import MethodConfig

# Create method
config = MethodConfig(name="lapeigvals", classifier="logistic")
method = create_method("lapeigvals", config=config)

# Train
method.fit(train_features, train_labels)

# Predict
predictions = method.predict_batch(test_features)
```

### Performance Tracking

```python
from src.utils.metrics_tracker import MetricsTracker

tracker = MetricsTracker(method_name="my_method")
tracker.start()

# ... training code ...

tracker.stop()
tracker.set_model_path("model.pkl")
tracker.set_sample_info(n_samples=1000)

metrics = tracker.get_metrics()
print(metrics.summary())
```

## Configuration Reference

### Method Configuration

```yaml
# config/method/lapeigvals.yaml
name: lapeigvals
classifier: logistic
cv_folds: 5
params:
  top_k_eigenvalues: 10
  use_diagonal_only: true
required_features:
  attention_diags: true
  laplacian_diags: true
```

### Feature Configuration

```yaml
# config/features/default.yaml
mode: teacher_forcing
attention_layers: all
store_full_attention: false
hidden_states_enabled: true
hidden_states_layers: last_4
```

## Troubleshooting

### CUDA Out of Memory

1. Set `allow_full_attention: false` in params.yaml
2. Reduce `max_length` in features config
3. Use checkpoint/resume: the pipeline automatically continues from last checkpoint

### Missing Features

Check that your method's required features are being extracted:
```bash
ls outputs/features/{dataset}/{model}/seed_{seed}/{task_type}/features/
```

### DVC Errors

```bash
# Force re-run a stage
dvc repro -f generate_activations

# Check dependencies
dvc dag
```

## Citation

If you use this framework, please cite:

```bibtex
@software{hallucination_detection,
  title = {Hallucination Detection Framework},
  year = {2024},
  url = {https://github.com/your-repo/hallucination-detection}
}
```

## License

MIT License
