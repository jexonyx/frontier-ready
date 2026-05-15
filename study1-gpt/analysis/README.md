# Analysis Scripts

This directory contains Python scripts for analyzing and visualizing GPT-2 training runs.

## Overview

The analysis infrastructure consists of three main components:

1. **parse_metrics.py** - Data loading and parsing utilities
2. **visualize_run.py** - Publication-ready plot generation
3. **summarize_run.py** - Markdown summary table generation

## Installation

Install the required dependencies:

```bash
uv pip install -e ".[analysis]"
```

This installs: pandas, matplotlib, seaborn, and jupyter.

## Usage

### Parse and Load Metrics

```python
from analysis.parse_metrics import load_metrics, get_final_metrics

# Load metrics from a training run
df = load_metrics("log/baseline")

# Get final metrics
final = get_final_metrics(df)
print(f"Final validation loss: {final['final_val_loss']:.4f}")
print(f"Max HellaSwag accuracy: {final['max_hella_acc']:.4f}")
```

### Generate Plots

Generate all four publication-ready plots for a training run:

```bash
uv run python analysis/visualize_run.py \
    --log-dir log/baseline \
    --output-dir analysis/figures
```

This creates:
- `loss_curves.png` - Training and validation loss over time
- `hellaswag_accuracy.png` - HellaSwag accuracy with OpenAI baselines
- `training_dynamics.png` - Learning rate, gradient norm, throughput, GPU memory
- `layer_analysis.png` - Per-layer gradient norms, weight norms, update ratios, activation norms

Options:
- `--model-size` - Model size for baseline comparison (default: "124M")

### Generate Summary Table

Generate a markdown summary table:

```bash
uv run python analysis/summarize_run.py \
    --log-dir log/baseline \
    --output analysis/baseline_summary.md
```

The summary includes:
- Final metrics with OpenAI baselines
- Convergence statistics
- Training efficiency metrics
- Layer analysis statistics

Options:
- `--model-size` - Model size for baseline comparison (default: "124M")
- `--tokens-per-step` - Tokens per training step (default: 524288)

## Data Format

The scripts expect a `metrics.jsonl` file in the log directory with the following fields:

**Required fields (every step):**
- `step` - Training step number
- `train_loss` - Training loss
- `lr` - Learning rate
- `grad_norm` - Gradient norm
- `dt_ms` - Step time in milliseconds
- `tok_sec` - Throughput in tokens/second
- `gpu_mem_gb` - GPU memory usage in GB
- `layer_grad_norms` - List of per-layer gradient norms
- `layer_weight_norms` - List of per-layer weight norms

**Optional fields (periodic evaluation):**
- `val_loss` - Validation loss (every 250 steps)
- `hella_acc` - HellaSwag accuracy (every 250 steps)
- `layer_update_ratios` - List of per-layer update ratios
- `layer_act_norms` - List of per-layer activation norms

## Testing

Run the test script to verify everything works:

```bash
uv run python analysis/test_parse.py
```

This creates synthetic test data and validates all parsing functions.

## Example Output

### Plots

All plots are saved as 300 DPI PNG files suitable for publication. They use:
- Seaborn whitegrid styling
- Clear axis labels and legends
- OpenAI GPT-2/GPT-3 baselines (where applicable)
- Color gradients for per-layer metrics

### Summary Table

```markdown
# Training Run Summary

**Run name:** baseline
**Model:** GPT-2 124M
**Training steps:** 19,072
**Total tokens:** ~10B

## Final Metrics

| Metric | Value | OpenAI GPT-2 Baseline | Delta |
|--------|-------|----------------------|-------|
| Validation Loss | 3.289 | 3.292 | -0.003 |
| HellaSwag Accuracy | 29.82% | 29.45% | +0.37% |
...
```

## Utility Functions

The `parse_metrics.py` module provides several utility functions:

- `load_metrics(log_dir)` - Load and parse metrics.jsonl
- `get_final_metrics(df)` - Extract final metrics and statistics
- `get_convergence_stats(df)` - Analyze convergence behavior
- `get_training_efficiency_stats(df)` - Compute efficiency metrics
- `get_layer_analysis_stats(df)` - Analyze per-layer statistics

These can be imported and used in custom analysis scripts or Jupyter notebooks.

## Future Extensions

These scripts are designed as the foundation for comparison analysis. When comparing multiple runs:

1. Import `load_metrics()` to load data from multiple directories
2. Create overlay plots with multiple lines per experiment
3. Generate comparison tables with delta columns between variants

Example:
```python
from analysis.parse_metrics import load_metrics

# Load multiple runs
baseline_df = load_metrics("log/baseline")
rope_df = load_metrics("log/rope")

# Create comparison plots
# ... (future comparison script)
```
