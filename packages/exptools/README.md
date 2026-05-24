# exptools

Experiment analysis tools for GPT-2 training runs.

## Features

- **Metrics parsing**: Load and analyze `metrics.jsonl` files
- **Visualization**: Generate training curves and diagnostic plots
- **Summaries**: Automated summary reports with key statistics

## Installation

```bash
uv pip install -e packages/exptools
```

## Usage

### Python API

```python
from exptools import load_metrics, get_final_metrics

# Load training metrics
df = load_metrics("log/my_run")

# Get summary statistics
stats = get_final_metrics(df)
print(f"Final validation loss: {stats['final_val_loss']:.3f}")
```

### CLI Tools

```bash
# Visualize training run
exptools-visualize --log-dir log/my_run --output-dir analysis/

# Generate summary report
exptools-summarize --log-dir log/my_run
```

## Metrics Supported

- Training and validation loss
- HellaSwag accuracy
- Learning rate schedule
- Gradient norms (global and per-layer)
- Weight norms and update ratios
- Throughput (tokens/sec)
- GPU memory usage

## Analysis Functions

- `load_metrics()` - Load metrics.jsonl into pandas DataFrame
- `get_final_metrics()` - Extract final performance statistics
- `get_convergence_stats()` - Analyze training convergence
- `get_training_efficiency_stats()` - Compute throughput and timing
- `get_layer_analysis_stats()` - Per-layer gradient analysis
