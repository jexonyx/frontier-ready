# Analytics Quick Start

## Installation

```bash
uv pip install -e ".[analysis]"
```

## Basic Usage

### Analyze a Training Run

```bash
# Quick statistics summary
uv run python analysis/example_usage.py --log-dir log/baseline

# Generate all plots (saved to output-dir)
uv run python analysis/visualize_run.py \
    --log-dir log/baseline \
    --output-dir analysis/figures

# Generate markdown summary table
uv run python analysis/summarize_run.py \
    --log-dir log/baseline \
    --output analysis/summary.md
```

### Programmatic Usage

```python
from analysis import load_metrics, get_final_metrics

# Load metrics
df = load_metrics("log/baseline")

# Get final statistics
final = get_final_metrics(df)
print(f"Final val loss: {final['final_val_loss']:.4f}")
print(f"Max HellaSwag: {final['max_hella_acc']:.4f}")
```

## Test the Scripts

```bash
# Run comprehensive test with synthetic data
./analysis/test_workflow.sh
```

## What Gets Generated

### Plots (300 DPI PNG)
1. `loss_curves.png` - Training/validation loss
2. `hellaswag_accuracy.png` - HellaSwag accuracy with baselines
3. `training_dynamics.png` - LR, grad norm, throughput, memory
4. `layer_analysis.png` - Per-layer gradient/weight norms

### Summary Table (Markdown)
- Final metrics vs OpenAI GPT-2 baselines
- Convergence milestones
- Training efficiency stats
- Layer analysis

## Common Commands

```bash
# For baseline run
uv run python analysis/visualize_run.py --log-dir log/baseline --output-dir writeup/figures
uv run python analysis/summarize_run.py --log-dir log/baseline --output writeup/baseline.md

# For different model sizes
uv run python analysis/visualize_run.py --log-dir log/350M --output-dir analysis/figures --model-size 350M
```

## Directory Structure

```
analysis/
├── parse_metrics.py      # Data loading utilities
├── visualize_run.py      # Plot generation
├── summarize_run.py      # Summary table generation
├── example_usage.py      # Usage examples
├── test_workflow.sh      # End-to-end test
└── README.md            # Full documentation
```

See `README.md` for complete documentation.
