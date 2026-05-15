# Workspace Monorepo Migration - Execution Plan

## Target Structure

```
frontier-ready/
├── packages/
│   ├── nanogpt-base/              # Training code as installable package
│   │   ├── nanogpt/
│   │   │   ├── __init__.py
│   │   │   ├── model.py           # GPT class, Block, etc.
│   │   │   ├── training.py        # Training loop
│   │   │   ├── data.py            # DataLoader, fineweb prep
│   │   │   ├── eval.py            # HellaSwag evaluation
│   │   │   └── variants/
│   │   │       ├── __init__.py
│   │   │       └── rope.py        # RoPE variant
│   │   ├── scripts/
│   │   │   ├── train.py           # CLI wrapper
│   │   │   ├── prepare_data.py
│   │   │   └── evaluate.py
│   │   ├── pyproject.toml
│   │   └── README.md
│   │
│   └── experiment-tools/          # Analysis/visualization package
│       ├── exptools/
│       │   ├── __init__.py
│       │   ├── metrics.py         # parse_metrics functions
│       │   ├── visualization.py   # plotting functions
│       │   └── summary.py         # summary generation
│       ├── scripts/
│       │   ├── visualize_run.py
│       │   └── summarize_run.py
│       ├── pyproject.toml
│       └── README.md
│
├── experiments/
│   ├── 01-baseline/
│   │   ├── config.yaml            # Experiment configuration
│   │   ├── run.py                 # Entry point (imports nanogpt)
│   │   ├── analysis.ipynb         # Jupyter notebook for analysis
│   │   ├── writeup/
│   │   │   ├── README.md
│   │   │   └── figures/
│   │   ├── pyproject.toml         # Experiment dependencies
│   │   └── README.md
│   │
│   ├── 02-rope/
│   │   ├── config.yaml
│   │   ├── run.py
│   │   ├── analysis.ipynb
│   │   ├── writeup/
│   │   ├── pyproject.toml
│   │   └── README.md
│   │
│   └── README.md                  # Experiments overview
│
├── data/                          # Shared datasets
│   ├── edu_fineweb10B/
│   ├── hellaswag/
│   └── README.md
│
├── outputs/                       # Training outputs & checkpoints
│   ├── baseline/
│   │   ├── checkpoints/
│   │   └── metrics.jsonl
│   ├── rope/
│   │   ├── checkpoints/
│   │   └── metrics.jsonl
│   └── README.md
│
├── infra/                         # Infrastructure (unchanged)
│
├── pyproject.toml                 # Workspace root configuration
├── uv.lock                        # Unified lockfile
├── .gitignore
├── .gitmodules                    # Remove or update
└── README.md                      # Updated root README
```

---

## Pre-Migration Decisions

### 1. Git Submodule Strategy
**Decision needed:** How to handle `study1-gpt/build-nanogpt` submodule?

**Options:**
- **A) Remove submodule, convert to monorepo code** ✓ Recommended
  - Pros: Simpler, atomic commits, unified history
  - Cons: Loses upstream tracking (but you've already forked)

- **B) Keep as submodule, package it differently**
  - Pros: Can pull upstream changes
  - Cons: Complicates workspace setup

**Recommendation:** Remove submodule, absorb code into monorepo. Your fork is already diverging with variant_rope.py.

### 2. Package Naming
- `nanogpt-base` or `nanogpt` or `frontier-nanogpt`?
- `experiment-tools` or `exptools` or `frontier-experiments`?

**Recommendation:**
- `nanogpt` (simple, matches imports)
- `exptools` (concise, clear purpose)

### 3. Workspace Tool
**Decision:** Use `uv` workspaces (native support since uv 0.2.0)

**Configuration:**
```toml
# pyproject.toml (root)
[tool.uv.workspace]
members = [
    "packages/nanogpt",
    "packages/exptools",
    "experiments/01-baseline",
    "experiments/02-rope",
]
```

### 4. Data & Outputs Location
- **Data:** `data/` at root (shared, large, not in packages)
- **Outputs:** `outputs/` at root (experiment results, symlinked if needed)

---

## Phase 1: Preparation & Backup

### 1.1 Document Current State
```bash
cd /Users/jex/frontier-ready

# Capture current structure
tree -L 3 -a -I '.venv|__pycache__|.git' > MIGRATION_BEFORE.txt

# List all tracked files
git ls-files > MIGRATION_FILES_BEFORE.txt

# Check for uncommitted changes
git status --short
```

**Action:** Commit any pending changes before migration.

### 1.2 Create Migration Branch
```bash
git checkout -b migrate-to-workspace
```

### 1.3 Backup Current State
```bash
# Create backup tag
git tag -a pre-workspace-migration -m "Backup before workspace migration"

# Optional: Create backup copy
cp -r /Users/jex/frontier-ready /Users/jex/frontier-ready-backup
```

### 1.4 Check Submodule Status
```bash
cd study1-gpt/build-nanogpt
git status
git log --oneline -5
cd ../..

# Document submodule state
git submodule status > MIGRATION_SUBMODULE_STATE.txt
```

---

## Phase 2: Create New Structure Skeleton

### 2.1 Create Top-Level Directories
```bash
cd /Users/jex/frontier-ready

mkdir -p packages/nanogpt/nanogpt/variants
mkdir -p packages/nanogpt/scripts
mkdir -p packages/exptools/exptools/scripts
mkdir -p experiments/01-baseline/writeup/figures
mkdir -p experiments/02-rope/writeup/figures
mkdir -p data
mkdir -p outputs/baseline
mkdir -p outputs/rope
```

### 2.2 Create Workspace Root Configuration
```bash
cat > pyproject.toml << 'EOF'
[project]
name = "frontier-ready"
version = "0.1.0"
description = "ML fundamentals research: GPT-2 reproduction and architectural experiments"
requires-python = ">=3.11"

[tool.uv.workspace]
members = [
    "packages/nanogpt",
    "packages/exptools",
    "experiments/01-baseline",
    "experiments/02-rope",
]

[tool.uv]
dev-dependencies = [
    "ruff>=0.4.0",
    "ipython>=8.0.0",
    "jupyter>=1.0.0",
]
EOF
```

### 2.3 Update .gitignore
```bash
cat >> .gitignore << 'EOF'

# Workspace outputs
outputs/*/checkpoints/*
outputs/*/metrics.jsonl
!outputs/*/.gitkeep

# Data (large files)
data/edu_fineweb10B/*
!data/edu_fineweb10B/.gitkeep
data/hellaswag/*
!data/hellaswag/.gitkeep

# Experiment artifacts
experiments/*/logs/*
experiments/*/*.egg-info/
experiments/*/.venv/

# Package builds
packages/*/*.egg-info/
packages/*/dist/
packages/*/build/
EOF
```

### 2.4 Create .gitkeep Files
```bash
touch data/.gitkeep
touch outputs/baseline/.gitkeep
touch outputs/rope/.gitkeep
```

---

## Phase 3: Reorganize Code into nanogpt Package (1-2 hours)

**Simplified Approach:**

We're not maintaining Karpathy's code as a separate submodule or trying to preserve its structure. Instead, we'll:
1. Copy his code as a starting point
2. Reorganize it into our desired package structure
3. Check it in as our code (with acknowledgment in README)

This is **much simpler** than careful extraction - just copy and reorganize however makes sense for us!

### 3.1 Copy Source Code from build-nanogpt

```bash
# Create temp directory for reorganization
mkdir -p temp/nanogpt-source

# Copy all source files we need
cp study1-gpt/build-nanogpt/train_gpt2.py temp/nanogpt-source/
cp study1-gpt/build-nanogpt/fineweb.py temp/nanogpt-source/
cp study1-gpt/build-nanogpt/hellaswag.py temp/nanogpt-source/
cp study1-gpt/build-nanogpt/variant_rope.py temp/nanogpt-source/

# Note: We're not preserving git history or structure - this is our code now
```

### 3.2 Create Package Structure

**Create packages/nanogpt/ directory:**
```bash
mkdir -p packages/nanogpt/nanogpt/variants
mkdir -p packages/nanogpt/scripts
```

### 3.3 Reorganize Code into Modules

**No need for "careful extraction"** - just move classes and functions to logical places:

**Step 1: Create `packages/nanogpt/nanogpt/__init__.py`**
```bash
cat > packages/nanogpt/nanogpt/__init__.py << 'EOF'
"""
NanoGPT: Minimal GPT-2 implementation for ML research.

Based on Andrej Karpathy's build-nanogpt.
Reorganized and extended for architectural experimentation.
"""

__version__ = "0.1.0"

from .model import GPT, GPTConfig
from .training import Trainer, TrainingConfig

__all__ = [
    "GPT",
    "GPTConfig",
    "Trainer",
    "TrainingConfig",
]
EOF
```

**Step 2: Create `packages/nanogpt/nanogpt/model.py`**

Copy model classes from `train_gpt2.py`:
- `GPTConfig` dataclass
- `CausalSelfAttention` class
- `MLP` class
- `Block` class
- `GPT` class

Organize however makes sense - no need to preserve original structure.

**Step 3: Create `packages/nanogpt/nanogpt/training.py`**

Copy training-related code from `train_gpt2.py`:
- Training loop logic
- Optimizer setup
- Create a `Trainer` class to encapsulate it

Again, reorganize as makes sense for us.

**Step 4: Create `packages/nanogpt/nanogpt/data.py`**

Move data loading code:
- `DataLoaderLite` from `train_gpt2.py`
- Data preparation logic from `fineweb.py`

**Step 5: Create `packages/nanogpt/nanogpt/eval.py`**

Move evaluation code:
- HellaSwag evaluation from `hellaswag.py`
- Any other eval functions

**Step 6: Create `packages/nanogpt/nanogpt/variants/rope.py`**

Copy RoPE variant implementation from `variant_rope.py`

**Step 7: Create CLI scripts in `packages/nanogpt/scripts/`**

Create thin wrappers that use the package:
- `train.py` - CLI for training
- `prepare_data.py` - CLI for data prep
- `evaluate.py` - CLI for evaluation

### 3.4 Create nanogpt pyproject.toml

```bash
cat > packages/nanogpt/pyproject.toml << 'EOF'
[project]
name = "nanogpt"
version = "0.1.0"
description = "Minimal GPT-2 implementation for architectural experiments"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "numpy>=1.24",
    "tiktoken>=0.5.0",
    "transformers>=4.36.0",
    "datasets>=2.14.0",
    "tqdm>=4.65.0",
    "requests>=2.31.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "ruff>=0.4.0",
]

[project.scripts]
nanogpt-train = "nanogpt.scripts.train:main"
nanogpt-prepare-data = "nanogpt.scripts.prepare_data:main"
nanogpt-evaluate = "nanogpt.scripts.evaluate:main"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["nanogpt*"]
EOF
```

### 3.5 Add Attribution to README

Create `packages/nanogpt/README.md`:

```markdown
# NanoGPT

Minimal GPT-2 implementation for architectural experimentation.

## Acknowledgments

This package is based on [Andrej Karpathy's build-nanogpt](https://github.com/karpathy/build-nanogpt),
which provides an excellent minimal, educational implementation of GPT-2.

We've reorganized and extended this codebase for our architectural experiments.

**Original implementation:** Copyright (c) Andrej Karpathy
**License:** MIT (original)
**Our modifications:** [Specify your license]

If you found this useful, please check out Karpathy's original implementation.
```

### 3.6 Install and Test

```bash
cd /Users/jex/frontier-ready

# Install package in development mode
uv pip install -e packages/nanogpt

# Test imports
python -c "from nanogpt import GPT, Trainer; print('✓ nanogpt package works')"

# Test CLI
nanogpt-train --help
```

---

## Phase 4: Convert Analysis to exptools Package

### 4.1 Create exptools Package Structure
```bash
cat > packages/exptools/exptools/__init__.py << 'EOF'
"""
Experiment tools for analyzing GPT training runs.

Provides utilities for parsing metrics, visualization, and summarization.
"""

__version__ = "0.1.0"

from .metrics import (
    load_metrics,
    get_final_metrics,
    get_convergence_stats,
    get_training_efficiency_stats,
    get_layer_analysis_stats,
)

__all__ = [
    "load_metrics",
    "get_final_metrics",
    "get_convergence_stats",
    "get_training_efficiency_stats",
    "get_layer_analysis_stats",
]
EOF
```

### 4.2 Move Analysis Code
```bash
# Copy analysis scripts with new structure
cp study1-gpt/analysis/parse_metrics.py packages/exptools/exptools/metrics.py
cp study1-gpt/analysis/visualize_run.py packages/exptools/exptools/visualization.py
cp study1-gpt/analysis/summarize_run.py packages/exptools/exptools/summary.py

# Copy CLI scripts
cp study1-gpt/analysis/visualize_run.py packages/exptools/scripts/visualize_run.py
cp study1-gpt/analysis/summarize_run.py packages/exptools/scripts/summarize_run.py
```

**Note:** Will need to update imports in copied files.

### 4.3 Create exptools pyproject.toml
```bash
cat > packages/exptools/pyproject.toml << 'EOF'
[project]
name = "exptools"
version = "0.1.0"
description = "Analysis and visualization tools for GPT training experiments"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "numpy>=1.24",
]

[project.optional-dependencies]
dev = [
    "jupyter>=1.0.0",
    "ipython>=8.0.0",
]

[project.scripts]
exptools-visualize = "exptools.scripts.visualize_run:main"
exptools-summarize = "exptools.scripts.summarize_run:main"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["exptools*"]
EOF
```

---

## Phase 5: Create Experiment Directories

### 5.1 Create 01-baseline Experiment
```bash
# Create run script
cat > experiments/01-baseline/run.py << 'EOF'
#!/usr/bin/env python3
"""
Baseline GPT-2 124M training run.

This experiment reproduces the standard GPT-2 124M architecture
on the FineWeb-Edu 10B dataset.
"""

import sys
from pathlib import Path

# Import from workspace packages
from nanogpt import GPT, GPTConfig, Trainer, TrainingConfig

def main():
    """Run baseline training."""
    # Configuration
    config = TrainingConfig(
        model_config=GPTConfig(),
        data_dir=Path("../../data/edu_fineweb10B"),
        output_dir=Path("../../outputs/baseline"),
        max_steps=19072,
        batch_size=64,
        learning_rate=6e-4,
    )

    # Create trainer
    trainer = Trainer(config)

    # Run training
    trainer.train()

if __name__ == "__main__":
    main()
EOF

chmod +x experiments/01-baseline/run.py
```

### 5.2 Create config.yaml
```bash
cat > experiments/01-baseline/config.yaml << 'EOF'
experiment:
  name: "baseline"
  description: "Standard GPT-2 124M architecture"

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

data:
  dataset: "edu_fineweb10B"
  num_tokens: 10_000_000_000

evaluation:
  eval_interval: 250
  eval_iters: 20

output:
  log_dir: "../../outputs/baseline"
  checkpoint_interval: 5000
EOF
```

### 5.3 Create pyproject.toml for Experiment
```bash
cat > experiments/01-baseline/pyproject.toml << 'EOF'
[project]
name = "experiment-baseline"
version = "0.1.0"
description = "Baseline GPT-2 124M experiment"
requires-python = ">=3.11"
dependencies = [
    "nanogpt",
    "exptools",
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
EOF
```

### 5.4 Create README
```bash
cat > experiments/01-baseline/README.md << 'EOF'
# Experiment 01: Baseline GPT-2 124M

## Overview

Standard GPT-2 124M architecture trained on FineWeb-Edu 10B tokens.
This serves as the baseline for comparison with architectural variants.

## Configuration

- **Model:** GPT-2 124M (12 layers, 12 heads, 768 dim)
- **Data:** FineWeb-Edu 10B tokens
- **Training:** 19,072 steps (~10B tokens)
- **Hardware:** 8x A100 GPUs (DDP)

## Running

```bash
# From experiment directory
uv run python run.py

# Or from workspace root
cd experiments/01-baseline && uv run python run.py
```

## Outputs

- **Checkpoints:** `../../outputs/baseline/checkpoints/`
- **Metrics:** `../../outputs/baseline/metrics.jsonl`
- **Logs:** `../../outputs/baseline/log.txt`

## Analysis

See `writeup/` for results and analysis notebooks.
EOF
```

### 5.5 Create Analysis Notebook
```bash
cat > experiments/01-baseline/analysis.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Baseline GPT-2 124M Analysis\n",
    "\n",
    "Analysis of the baseline training run."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "from pathlib import Path\n",
    "\n",
    "# Import from workspace packages\n",
    "from exptools import (\n",
    "    load_metrics,\n",
    "    get_final_metrics,\n",
    "    get_convergence_stats,\n",
    ")\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "sns.set_style(\"whitegrid\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load metrics\n",
    "df = load_metrics(\"../../outputs/baseline\")\n",
    "print(f\"Loaded {len(df)} training steps\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get final metrics\n",
    "final = get_final_metrics(df)\n",
    "for key, value in final.items():\n",
    "    print(f\"{key}: {value}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot loss curves\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))\n",
    "\n",
    "# Training loss\n",
    "ax1.plot(df['step'], df['train_loss'])\n",
    "ax1.set_xlabel('Step')\n",
    "ax1.set_ylabel('Loss')\n",
    "ax1.set_title('Training Loss')\n",
    "ax1.set_yscale('log')\n",
    "\n",
    "# Validation loss\n",
    "val_data = df[['step', 'val_loss']].dropna()\n",
    "ax2.plot(val_data['step'], val_data['val_loss'])\n",
    "ax2.axhline(y=3.2924, color='r', linestyle='--', label='OpenAI GPT-2')\n",
    "ax2.set_xlabel('Step')\n",
    "ax2.set_ylabel('Loss')\n",
    "ax2.set_title('Validation Loss')\n",
    "ax2.legend()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('writeup/figures/loss_curves.png', dpi=300, bbox_inches='tight')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF
```

### 5.6 Repeat for 02-rope
```bash
# Copy structure from baseline
cp -r experiments/01-baseline/* experiments/02-rope/

# Update config and README for RoPE variant
# (Manual edits needed)
```

---

## Phase 5.5: Infrastructure Integration

### 5.5.1 Update Infrastructure Directory

**Create directories for experiment-specific configs:**
```bash
cd infra
mkdir -p stacks templates scripts
```

**Create template for common configurations:**
```bash
cat > templates/single-gpu-a100.yaml << 'EOF'
# Template for single A100 GPU experiments
config:
  machineType: a2-highgpu-1g
  gpuType: nvidia-tesla-a100
  gpuCount: 1
  bootDiskSize: 200
  dataDiskSize: 500
  preemptible: true
  zone: us-central1-a
EOF
```

**Add deployment helper scripts:**
```bash
cat > scripts/deploy-experiment.sh << 'EOF'
#!/bin/bash
# Deploy infrastructure for a specific experiment
EXPERIMENT_NAME=$1
pulumi stack select "${EXPERIMENT_NAME}" 2>/dev/null || pulumi stack init "${EXPERIMENT_NAME}"
pulumi config set experimentName "${EXPERIMENT_NAME}"
pulumi up --stack "${EXPERIMENT_NAME}"
pulumi stack output --stack "${EXPERIMENT_NAME}" --json > "../experiments/*-${EXPERIMENT_NAME}/infra-outputs.json"
EOF

chmod +x scripts/deploy-experiment.sh
```

### 5.5.2 Add Experiment Infrastructure Configs

**For each experiment, create infrastructure declaration:**

```bash
# Baseline experiment
cat > experiments/01-baseline/infra.yaml << 'EOF'
experiment:
  name: baseline
  description: Standard GPT-2 124M training

resources:
  machine_type: a2-highgpu-1g
  gpu_type: nvidia-tesla-a100
  gpu_count: 1
  boot_disk_size: 200
  data_disk_size: 500
  preemptible: true

estimated_cost:
  hourly: "$1.35"
  total: "$20"
EOF

# Deployment scripts
cat > experiments/01-baseline/deploy.sh << 'EOF'
#!/bin/bash
cd ../../infra
./scripts/deploy-experiment.sh baseline
EOF

cat > experiments/01-baseline/destroy.sh << 'EOF'
#!/bin/bash
cd ../../infra
pulumi destroy --stack baseline --yes
EOF

chmod +x experiments/01-baseline/{deploy,destroy}.sh
```

**Repeat for rope experiment:**
```bash
cp experiments/01-baseline/infra.yaml experiments/02-rope/
sed -i.bak 's/baseline/rope/g' experiments/02-rope/infra.yaml
rm experiments/02-rope/infra.yaml.bak

cp experiments/01-baseline/deploy.sh experiments/02-rope/
cp experiments/01-baseline/destroy.sh experiments/02-rope/
sed -i.bak 's/baseline/rope/g' experiments/02-rope/{deploy,destroy}.sh
rm experiments/02-rope/*.bak
```

### 5.5.3 Update .gitignore

```bash
cat >> .gitignore << 'EOF'

# Infrastructure outputs (per-experiment)
experiments/*/infra-outputs.json
EOF
```

**See `INFRASTRUCTURE_INTEGRATION.md` for complete details on:**
- Full file structure
- Pulumi stack management
- VM deployment workflow
- Data synchronization strategies

---

## Phase 6: Move Data and Outputs

### 6.1 Move Data Directory
```bash
# Move FineWeb data
mv study1-gpt/build-nanogpt/edu_fineweb10B/* data/edu_fineweb10B/ 2>/dev/null || true

# Move HellaSwag data
mv study1-gpt/build-nanogpt/hellaswag/* data/hellaswag/ 2>/dev/null || true
```

### 6.2 Move Existing Logs (If Any)
```bash
# Find and move existing log directories
# (Adjust based on actual log locations)
# mv study1-gpt/build-nanogpt/log* outputs/ 2>/dev/null || true
```

---

## Phase 7: Install and Test Workspace

### 7.1 Install Workspace
```bash
cd /Users/jex/frontier-ready

# This should install all packages in editable mode
uv sync
```

### 7.2 Verify Package Installation
```bash
# Test nanogpt import
uv run python -c "from nanogpt import GPT, GPTConfig; print('✓ nanogpt package works')"

# Test exptools import
uv run python -c "from exptools import load_metrics; print('✓ exptools package works')"
```

### 7.3 Test Experiment
```bash
# Try running baseline (dry run or quick test)
cd experiments/01-baseline
uv run python -c "import run; print('✓ Experiment imports work')"
```

---

## Phase 8: Clean Up Old Structure

### 8.1 Remove Old study1-gpt Directory
```bash
cd /Users/jex/frontier-ready

# Verify everything is migrated
# Then remove
git rm -rf study1-gpt

git commit -m "Remove old study1-gpt structure after workspace migration"
```

### 8.2 Update Root README
Create new comprehensive README explaining workspace structure.

### 8.3 Create Migration Documentation
```bash
cat > MIGRATION_COMPLETE.md << 'EOF'
# Migration to Workspace Monorepo - Complete

Migrated from nested structure to workspace monorepo on [DATE].

## Changes

- Converted build-nanogpt to `packages/nanogpt` package
- Converted analysis scripts to `packages/exptools` package
- Created experiments structure
- Set up uv workspace
- Moved data to shared `data/` directory
- Centralized outputs in `outputs/` directory

## Verification

All tests passing:
- [ ] nanogpt package imports work
- [ ] exptools package imports work
- [ ] Experiments can import packages
- [ ] Data is accessible
- [ ] Can run training (dry run)

## Rollback

If needed, rollback with:
```bash
git checkout pre-workspace-migration
```
EOF
```

---

## Phase 9: Final Verification Checklist

- [ ] All packages install without errors
- [ ] Import tests pass
- [ ] No broken symlinks
- [ ] Data accessible from all experiments
- [ ] Old submodule fully removed
- [ ] .gitignore updated correctly
- [ ] Documentation updated
- [ ] Can run a quick training test
- [ ] Can run analysis on test data
- [ ] All experiments have proper structure

---

## Critical Files to Create

### Priority 1: Must Have Before Testing

1. **`packages/nanogpt/nanogpt/model.py`**
   - Copy from train_gpt2.py: GPT, Block, CausalSelfAttention, MLP classes

2. **`packages/nanogpt/nanogpt/training.py`**
   - Copy training loop, reorganize into Trainer class

3. **`packages/nanogpt/scripts/train.py`**
   - CLI wrapper that uses above modules

4. **`packages/exptools/exptools/metrics.py`**
   - Rename from parse_metrics.py, update imports

### Priority 2: Nice to Have

5. **`packages/nanogpt/nanogpt/data.py`**
   - Copy data loading from train_gpt2.py and fineweb.py

6. **`packages/nanogpt/nanogpt/eval.py`**
   - Move hellaswag.py contents here

7. **Package READMEs**
   - Document each package's API

---

## Rollback Plan

If migration fails:

```bash
# Return to pre-migration state
git checkout pre-workspace-migration

# Or if committed:
git revert <migration-commits>

# Or nuclear option:
cp -r /Users/jex/frontier-ready-backup/* /Users/jex/frontier-ready/
```

---

## Timeline Estimate

- Phase 1 (Prep): 30 minutes
- Phase 2 (Structure): 30 minutes
- Phase 3 (nanogpt package): 1-2 hours (**simplified** - just copy and reorganize)
- Phase 4 (exptools package): 1 hour
- Phase 5 (Experiments): 1 hour
- Phase 5.5 (Infrastructure integration): 45 minutes
- Phase 6 (Data move): 15 minutes
- Phase 7 (Testing): 1-2 hours
- Phase 8 (Cleanup): 30 minutes
- **Total: 6.5-9 hours**

Phase 3 is now much faster since we're just reorganizing code, not carefully extracting it.

---

## Next Steps After This Document

1. Review this plan
2. Make any necessary adjustments
3. Execute phases in order
4. Test thoroughly after each phase
5. Commit frequently with clear messages

Ready to proceed?
