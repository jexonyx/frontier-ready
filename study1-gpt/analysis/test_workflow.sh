#!/bin/bash
# Test the complete analysis workflow with synthetic data

set -e

echo "=================================="
echo "Testing Analysis Workflow"
echo "=================================="

# Create test directory
TEST_DIR="analysis/test_run"
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

echo ""
echo "1. Creating synthetic training data..."

# Generate synthetic metrics.jsonl
python3 << 'EOF'
import json
from pathlib import Path

metrics_file = Path("analysis/test_run/metrics.jsonl")

with open(metrics_file, 'w') as f:
    for step in range(0, 10000, 10):
        # Simulate realistic training dynamics
        progress = step / 10000
        train_loss = 4.0 - progress * 0.8

        # Validation only every 250 steps
        val_loss = None
        hella_acc = None
        if step % 250 == 0 and step > 0:
            val_loss = 3.5 - progress * 0.25
            hella_acc = 0.25 + progress * 0.05

        metrics = {
            "step": step,
            "train_loss": train_loss,
            "lr": 6e-4 * (1 - progress),  # Cosine decay
            "grad_norm": min(1.0, 0.7 + 0.3 * (step % 100) / 100),
            "dt_ms": 2500 + (step % 50) * 10,
            "tok_sec": 2100 + (step % 100) * 5,
            "gpu_mem_gb": 18.5 + (step % 10) * 0.05,
            "layer_grad_norms": [0.1 + i * 0.02 + (step % 10) * 0.001 for i in range(12)],
            "layer_weight_norms": [5.0 + i * 0.5 + progress * 0.2 for i in range(12)],
        }

        if val_loss is not None:
            metrics["val_loss"] = val_loss

        if hella_acc is not None:
            metrics["hella_acc"] = hella_acc

        # Add update ratios and activation norms every 100 steps
        if step % 100 == 0:
            metrics["layer_update_ratios"] = [0.002 + i * 0.0001 for i in range(12)]
            metrics["layer_act_norms"] = [1.0 + i * 0.1 for i in range(12)]

        f.write(json.dumps(metrics) + "\n")

print(f"✓ Created {metrics_file}")
EOF

echo "   ✓ Created synthetic metrics.jsonl with 1000 steps"

echo ""
echo "2. Testing parse_metrics.py..."
uv run python -c "
from analysis.parse_metrics import load_metrics
df = load_metrics('$TEST_DIR')
print(f'   ✓ Loaded {len(df)} steps')
print(f'   ✓ Found {len(df.columns)} columns')
"

echo ""
echo "3. Testing example_usage.py..."
uv run python analysis/example_usage.py --log-dir "$TEST_DIR"

echo ""
echo "4. Testing visualize_run.py..."
uv run python analysis/visualize_run.py \
    --log-dir "$TEST_DIR" \
    --output-dir "${TEST_DIR}/figures"

echo ""
echo "5. Verifying generated plots..."
for plot in loss_curves.png hellaswag_accuracy.png training_dynamics.png layer_analysis.png; do
    if [ -f "${TEST_DIR}/figures/${plot}" ]; then
        size=$(du -h "${TEST_DIR}/figures/${plot}" | cut -f1)
        echo "   ✓ ${plot} (${size})"
    else
        echo "   ✗ Missing: ${plot}"
        exit 1
    fi
done

echo ""
echo "6. Testing summarize_run.py..."
uv run python analysis/summarize_run.py \
    --log-dir "$TEST_DIR" \
    --output "${TEST_DIR}/summary.md"

echo ""
echo "7. Preview of generated summary:"
echo "-----------------------------------"
head -30 "${TEST_DIR}/summary.md"
echo "-----------------------------------"

echo ""
echo "=================================="
echo "✓ All tests passed!"
echo "=================================="
echo ""
echo "Test artifacts saved to: $TEST_DIR"
echo "  - metrics.jsonl"
echo "  - figures/*.png (4 plots)"
echo "  - summary.md"
echo ""
echo "To clean up:"
echo "  rm -rf $TEST_DIR"
