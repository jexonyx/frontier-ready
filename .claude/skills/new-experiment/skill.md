# New Experiment Skill

Creates a new experiment in the workspace with all required files and configuration.

## Usage

```
/new-experiment
```

## What This Skill Does

1. Prompts for experiment details (type, name, hypothesis, architecture/variant, resources)
2. Creates complete experiment directory structure
3. Generates all required files with proper configuration (variant-aware)
4. Updates workspace configuration
5. Runs validation tests
6. Stages files with git (user handles commit)

## Execution Steps

### Step 1: Gather Information

Use the `AskUserQuestion` tool to collect experiment details:

**Questions to ask:**

1. **Experiment Type** (header: "Type")
   - Question: "What type of experiment is this?"
   - Options:
     - label: "Architecture variant", description: "Testing a new architecture modification (attention, MLP, embeddings)"
     - label: "Hyperparameter tuning", description: "Testing different hyperparameters on baseline model"
     - label: "Data experiment", description: "Testing different data, preprocessing, or augmentation"
     - label: "Baseline reproduction", description: "Reproducing baseline results for comparison"

2. **Experiment Name** (header: "Name")
   - Question: "What is the experiment name? (lowercase, no spaces, e.g., 'rope', 'swiglu', 'parallel')"
   - Options:
     - label: "rope", description: "Rotary position embeddings"
     - label: "swiglu", description: "SwiGLU activation"
     - label: "diffattn", description: "Differential attention"
     - label: "parallel", description: "Parallel attention+MLP"
     - (User can enter custom name via "Other")

3. **Brief Description** (header: "Description")
   - Question: "What does this experiment test? (one sentence)"
   - Options:
     - label: "New attention mechanism", description: "Testing a novel attention architecture"
     - label: "New activation function", description: "Testing a different activation in MLP"
     - label: "Architectural modification", description: "Structural change to transformer blocks"

4. **Architecture/Variant** (header: "Architecture") [Only if type is "Architecture variant"]
   - Question: "Which architecture variant are you using?"
   - Options:
     - (Auto-detect from llmkit/variants/ and list available variants)
     - label: "rope", description: "RoPE position embeddings (create_rope_gpt)"
     - label: "Custom", description: "Custom variant not yet in llmkit/variants/"

5. **Hypothesis** (header: "Hypothesis")
   - Question: "What do you expect to happen and why?"
   - Options:
     - label: "Lower loss than baseline", description: "Expect better performance"
     - label: "Faster convergence", description: "Expect to reach target loss sooner"
     - label: "Similar performance", description: "Null hypothesis - checking if change is neutral"

6. **GPU Resources** (header: "Resources")
   - Question: "What GPU resources do you need?"
   - Options:
     - label: "1x A100 40GB", description: "$1.35/hr - Same as baseline (fair comparison)"
     - label: "1x A100 80GB", description: "$1.60/hr - More memory for larger batches"
     - label: "2x A100 40GB", description: "$2.70/hr - Distributed training"

### Step 2: Detect Available Variants

If experiment type is "Architecture variant", detect available variants:

```bash
# List available variants
echo "Available architecture variants:"
ls -1 packages/llmkit/llmkit/variants/*.py | grep -v "__init__" | xargs -I{} basename {} .py

# Parse each variant to extract create_* function
for variant_file in packages/llmkit/llmkit/variants/*.py; do
    if [ "$(basename ${variant_file})" != "__init__.py" ]; then
        variant_name=$(basename ${variant_file} .py)
        function_name=$(grep "^def create_" ${variant_file} | sed 's/def \(create_[^(]*\).*/\1/')
        echo "  - ${variant_name}: ${function_name}"
    fi
done
```

### Step 3: Determine Experiment Number

Read the `experiments/` directory to find the next available number:

```bash
ls -d experiments/*/ | sort | tail -1
```

Extract the number from the last experiment and increment by 1.

### Step 4: Create Directory Structure

Create the experiment directory:

```bash
EXPERIMENT_NUM="XX"  # From step 2
EXPERIMENT_NAME="xxx"  # From step 1
EXPERIMENT_DIR="experiments/${EXPERIMENT_NUM}-${EXPERIMENT_NAME}"

mkdir -p "${EXPERIMENT_DIR}/writeup/figures"
```

### Step 5: Create Files

**Important:** Determine the correct imports and model creation based on experiment type.

**If experiment type is "Architecture variant" and architecture is specified:**

```python
# Add to imports
{VARIANT_IMPORT} = f"from llmkit.variants.{architecture} import create_{architecture}_gpt"

# Model creation
{MODEL_CREATION} = f"""    # Create model using variant
    print(f"Creating model with {{cfg['experiment']['architecture']}} architecture...")
    model = create_{architecture}_gpt(model_config)"""
```

**Otherwise (baseline, hyperparameter, data experiments):**

```python
# No variant import needed
{VARIANT_IMPORT} = ""

# Standard model creation
{MODEL_CREATION} = """    # Create standard GPT model
    model = GPT(model_config)"""
```

### Step 5a: Create Files

Use the `Write` tool to create each file with proper substitutions.

#### 4.1 Create `config.yaml`

```yaml
experiment:
  name: {EXPERIMENT_NAME}
  type: {EXPERIMENT_TYPE}  # "architecture_variant", "hyperparameter", "data", "baseline"
  description: "{DESCRIPTION}"
  hypothesis: "{HYPOTHESIS}"
  architecture: {ARCHITECTURE}  # e.g., "rope", "swiglu", "baseline" (null if not architecture variant)

model:
  n_layer: 12
  n_head: 12
  n_embd: 768
  block_size: 1024
  vocab_size: 50257

training:
  max_steps: 19072
  batch_size: 64
  sequence_length: 1024
  gradient_accumulation_steps: 16
  learning_rate: 6e-4
  warmup_steps: 715
  max_lr: 6e-4
  min_lr: 6e-5
  weight_decay: 0.1
  grad_clip: 1.0

data:
  dataset: "edu_fineweb10B"
  data_dir: "../../data/edu_fineweb10B"

evaluation:
  eval_interval: 250
  eval_iters: 20
  eval_hellaswag: true

output:
  output_dir: "../../outputs/{EXPERIMENT_NAME}"
  checkpoint_interval: 5000
  log_interval: 10

hardware:
  device: "cuda"
  compile: true
  use_ddp: false
```

#### 4.2 Create `run.py`

```python
#!/usr/bin/env python3
"""
Experiment: {EXPERIMENT_NAME}

Type: {EXPERIMENT_TYPE}
Description: {DESCRIPTION}
Hypothesis: {HYPOTHESIS}
Architecture: {ARCHITECTURE}
"""

import sys
from pathlib import Path
import yaml

from llmkit import GPT, ModelConfig, Trainer, TrainingConfig
{VARIANT_IMPORT}


def load_config():
    """Load experiment configuration from config.yaml."""
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Run experiment."""
    cfg = load_config()

    model_config = ModelConfig(
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        n_embd=cfg["model"]["n_embd"],
        block_size=cfg["model"]["block_size"],
        vocab_size=cfg["model"]["vocab_size"],
    )

{MODEL_CREATION}

    training_config = TrainingConfig(
        model_config=model_config,
        data_dir=Path(cfg["data"]["data_dir"]),
        output_dir=Path(cfg["output"]["output_dir"]),
        max_steps=cfg["training"]["max_steps"],
        batch_size=cfg["training"]["batch_size"],
        sequence_length=cfg["training"]["sequence_length"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        warmup_steps=cfg["training"]["warmup_steps"],
        max_lr=cfg["training"]["max_lr"],
        min_lr=cfg["training"]["min_lr"],
        weight_decay=cfg["training"]["weight_decay"],
        grad_clip=cfg["training"]["grad_clip"],
        eval_interval=cfg["evaluation"]["eval_interval"],
        eval_iters=cfg["evaluation"]["eval_iters"],
        checkpoint_interval=cfg["output"]["checkpoint_interval"],
        device=cfg["hardware"]["device"],
        compile=cfg["hardware"]["compile"],
        use_ddp=cfg["hardware"]["use_ddp"],
    )

    print(f"Starting experiment: {cfg['experiment']['name']}")
    print(f"Description: {cfg['experiment']['description']}")
    print(f"Output directory: {training_config.output_dir}")

    trainer = Trainer(training_config)
    trainer.train()

    print(f"✓ Training complete! Results in {training_config.output_dir}")


if __name__ == "__main__":
    main()
```

#### 4.3 Create `pyproject.toml`

```toml
[project]
name = "experiment-{EXPERIMENT_NAME}"
version = "0.1.0"
description = "{DESCRIPTION}"
requires-python = ">=3.11"
dependencies = [
    "llmkit",
    "exptools",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
analysis = [
    "jupyter>=1.0.0",
    "ipykernel>=6.0.0",
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
py-modules = []
```

#### 4.4 Create `README.md`

```markdown
# Experiment: {EXPERIMENT_NAME}

## Overview

**Hypothesis:** {HYPOTHESIS}

**Description:** {DESCRIPTION}

**Expected Outcome:** [To be determined]

## Configuration

- **Model:** GPT-2 124M
- **Data:** FineWeb-Edu 10B tokens
- **Training:** 19,072 steps (~10B tokens)
- **Hardware:** {GPU_RESOURCES}

## Running

### Deploy Infrastructure
```bash
./deploy.sh
```

### SSH into VM
```bash
../../infra/connect.sh --experiment {EXPERIMENT_NAME}
```

### On VM: Run Training
```bash
cd /data
git clone <your-repo>
cd frontier-ready && uv sync
cd experiments/{EXPERIMENT_NUM}-{EXPERIMENT_NAME}
uv run python run.py
```

### Monitor
```bash
tail -f ../../outputs/{EXPERIMENT_NAME}/log.txt
```

## Analysis

```bash
# Generate plots
uv run exptools-visualize \
    --log-dir ../../outputs/{EXPERIMENT_NAME} \
    --output-dir writeup/figures

# Generate summary
uv run exptools-summarize \
    --log-dir ../../outputs/{EXPERIMENT_NAME} \
    --output writeup/summary.md
```

## Results

[To be filled after experiment completes]

## Outputs

- **Checkpoints:** `../../outputs/{EXPERIMENT_NAME}/checkpoints/`
- **Metrics:** `../../outputs/{EXPERIMENT_NAME}/metrics.jsonl`
- **Logs:** `../../outputs/{EXPERIMENT_NAME}/log.txt`
```

#### 4.5 Create `infra.yaml`

Parse the GPU resources choice and set appropriate values:

- "1x A100 40GB" → `a2-highgpu-1g`, `nvidia-tesla-a100`, count=1
- "1x A100 80GB" → `a2-ultragpu-1g`, `nvidia-tesla-a100`, count=1
- "2x A100 40GB" → `a2-highgpu-2g`, `nvidia-tesla-a100`, count=2

```yaml
experiment:
  name: {EXPERIMENT_NAME}
  description: "{DESCRIPTION}"

resources:
  machine_type: {MACHINE_TYPE}
  gpu_type: {GPU_TYPE}
  gpu_count: {GPU_COUNT}
  boot_disk_size: 200
  data_disk_size: 500
  preemptible: true

runtime:
  zone: us-central1-a
  image_family: pytorch-latest-gpu

estimated_cost:
  hourly: "{HOURLY_COST}"
  total_training: "{TOTAL_COST}"
```

#### 4.6 Create `deploy.sh`

```bash
#!/bin/bash
set -e

EXPERIMENT_NAME="{EXPERIMENT_NAME}"
INFRA_DIR="../../infra"

echo "================================================"
echo "Deploying infrastructure for: ${EXPERIMENT_NAME}"
echo "================================================"
echo ""
cat infra.yaml | grep -A 10 "resources:"
echo ""
read -p "Continue with deployment? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled"
    exit 1
fi

cd "${INFRA_DIR}"
./scripts/deploy-experiment.sh "${EXPERIMENT_NAME}"

echo ""
echo "✅ VM ready!"
echo "Connect: ../../infra/connect.sh --experiment ${EXPERIMENT_NAME}"
```

Make executable with: `chmod +x deploy.sh`

#### 4.7 Create `destroy.sh`

```bash
#!/bin/bash
set -e

EXPERIMENT_NAME="{EXPERIMENT_NAME}"
INFRA_DIR="../../infra"

echo "WARNING: Destroying infrastructure for: ${EXPERIMENT_NAME}"
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

cd "${INFRA_DIR}"
pulumi destroy --stack "${EXPERIMENT_NAME}" --yes

echo "✅ Infrastructure destroyed"
```

Make executable with: `chmod +x destroy.sh`

#### 4.8 Create `analysis.ipynb`

Create a Jupyter notebook with starter analysis code (load metrics, plot losses, compare with baseline).

### Step 5: Create Output Directory

```bash
mkdir -p outputs/{EXPERIMENT_NAME}
touch outputs/{EXPERIMENT_NAME}/.gitkeep
```

### Step 6: Update Workspace Configuration

Edit `pyproject.toml` at the workspace root:

Add to the `[tool.uv.workspace]` members list:
```toml
"experiments/{EXPERIMENT_NUM}-{EXPERIMENT_NAME}",
```

### Step 7: Validation

Run tests to ensure everything is set up correctly:

```bash
cd experiments/{EXPERIMENT_NUM}-{EXPERIMENT_NAME}

# Test imports (including variant if applicable)
if [ "{ARCHITECTURE}" != "null" ] && [ -n "{ARCHITECTURE}" ]; then
    uv run python -c "
import yaml
from llmkit import GPT, ModelConfig
from llmkit.variants.{ARCHITECTURE} import create_{ARCHITECTURE}_gpt
from exptools import load_metrics
print('✓ Imports work (including {ARCHITECTURE} variant)')
"
else
    uv run python -c "
import yaml
from llmkit import GPT, ModelConfig
from exptools import load_metrics
print('✓ Imports work')
"
fi

# Test config loading
uv run python -c "
import yaml
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
print(f'✓ Config loaded: {cfg[\"experiment\"][\"name\"]}')"

# Verify files exist
ls -la
```

### Step 8: Stage Files with Git

Stage the created files (don't commit - user handles commit):

```bash
git add experiments/{EXPERIMENT_NUM}-{EXPERIMENT_NAME}
git add outputs/{EXPERIMENT_NAME}/.gitkeep
git add pyproject.toml

# Show what's staged
git status
```

### Step 9: Summary Output

Print a summary for the user:

```
✅ Experiment {EXPERIMENT_NUM}-{EXPERIMENT_NAME} created successfully!

📁 Location: experiments/{EXPERIMENT_NUM}-{EXPERIMENT_NAME}/
📊 Outputs: outputs/{EXPERIMENT_NAME}/

✓ Validations passed:
  ✓ Imports (including {ARCHITECTURE} variant)
  ✓ Config loading
  ✓ Files exist

📝 Suggested commit message:

Add experiment {EXPERIMENT_NUM}-{EXPERIMENT_NAME}

- Type: {EXPERIMENT_TYPE}
- Architecture: {ARCHITECTURE}
- Description: {DESCRIPTION}
- Hypothesis: {HYPOTHESIS}
- Resources: {GPU_RESOURCES}

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>

📦 Next steps:
  1. Review configuration: cd experiments/{EXPERIMENT_NUM}-{EXPERIMENT_NAME} && cat config.yaml
  2. Commit changes: git commit -m "Add experiment {EXPERIMENT_NUM}-{EXPERIMENT_NAME}"
  3. Deploy VM: ./deploy.sh
  4. Run training: (see README.md for detailed instructions)

📁 Files created:
  ✓ config.yaml (with architecture: {ARCHITECTURE})
  ✓ run.py (variant-aware model creation)
  ✓ pyproject.toml
  ✓ README.md
  ✓ infra.yaml
  ✓ deploy.sh
  ✓ destroy.sh
  ✓ analysis.ipynb

💡 Architecture details:
  {ARCHITECTURE_DETAILS}
```

## Error Handling

- If experiment name already exists, ask if they want to overwrite or choose a different name
- If workspace sync fails, show error and suggest running `uv sync` manually
- If git commit fails, show the error and suggest manual commit

## Notes

- All placeholders in templates should be replaced with actual values
- File permissions: make shell scripts executable (`chmod +x`)
- Validate that experiment name is lowercase, no spaces, valid identifier
- Ensure experiment number is sequential (find last number and increment)
