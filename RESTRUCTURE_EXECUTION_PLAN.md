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

## Phase 10: SFT/RL Extensions (Optional, Future Work)

**Status:** Planning phase - implement after base workspace migration is complete and validated

**Purpose:** Extend the workspace to support the full modern LLM training pipeline: pre-training → SFT → RL

**Timeline estimate:** 4-6 hours

**Prerequisites:**
- Phases 1-9 complete and validated
- At least one successful pre-training experiment run
- Understanding of SFT and RL fundamentals

---

### 10.1 Package Structure Extensions

**Create new packages for SFT and RL:**

```bash
# Create SFT package structure
mkdir -p packages/nanogpt-sft/nanogpt_sft/{datasets,scripts}
mkdir -p packages/nanogpt-rl/nanogpt_rl/{algorithms,scripts}
```

**Package: `nanogpt-sft/`** (Supervised Fine-Tuning)

```
packages/nanogpt-sft/
├── nanogpt_sft/
│   ├── __init__.py
│   ├── data.py              # Instruction dataset loaders
│   ├── formatting.py        # Prompt templates, chat formatting
│   ├── trainer.py           # SFT-specific training loop
│   ├── eval.py              # Instruction-following evals
│   └── datasets/
│       ├── __init__.py
│       ├── alpaca.py        # Alpaca format loader
│       ├── sharegpt.py      # ShareGPT/ChatML format
│       └── custom.py        # Custom dataset adapters
├── scripts/
│   ├── prepare_sft_data.py
│   ├── train_sft.py
│   └── evaluate_sft.py
├── pyproject.toml
└── README.md
```

**Create `packages/nanogpt-sft/pyproject.toml`:**

```bash
cat > packages/nanogpt-sft/pyproject.toml << 'EOF'
[project]
name = "nanogpt-sft"
version = "0.1.0"
description = "Supervised fine-tuning for nanogpt models"
requires-python = ">=3.11"
dependencies = [
    "nanogpt",                    # Reuse base model
    "datasets>=2.14.0",           # HuggingFace datasets
    "transformers>=4.36.0",       # For tokenization/evaluation
    "jinja2>=3.1.0",             # Prompt templating
    "scikit-learn>=1.3.0",       # For metrics
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "ruff>=0.4.0",
]

[project.scripts]
nanogpt-sft-train = "nanogpt_sft.scripts.train_sft:main"
nanogpt-sft-prepare = "nanogpt_sft.scripts.prepare_sft_data:main"
nanogpt-sft-eval = "nanogpt_sft.scripts.evaluate_sft:main"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["nanogpt_sft*"]
EOF
```

**Package: `nanogpt-rl/`** (Reinforcement Learning)

```
packages/nanogpt-rl/
├── nanogpt_rl/
│   ├── __init__.py
│   ├── reward_model.py      # Reward model training
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── ppo.py           # PPO implementation
│   │   ├── dpo.py           # Direct Preference Optimization
│   │   └── reinforce.py     # Simple REINFORCE baseline
│   ├── rollouts.py          # Generation and sampling
│   ├── trainer.py           # RL training loop
│   └── eval.py              # RL-specific evaluation
├── scripts/
│   ├── train_reward_model.py
│   ├── train_ppo.py
│   ├── train_dpo.py
│   └── evaluate_rl.py
├── pyproject.toml
└── README.md
```

**Create `packages/nanogpt-rl/pyproject.toml`:**

```bash
cat > packages/nanogpt-rl/pyproject.toml << 'EOF'
[project]
name = "nanogpt-rl"
version = "0.1.0"
description = "Reinforcement learning for nanogpt models"
requires-python = ">=3.11"
dependencies = [
    "nanogpt",
    "nanogpt-sft",               # RL typically starts from SFT checkpoint
    "torch>=2.2",
    "einops>=0.7.0",             # Tensor operations
    "accelerate>=0.24.0",        # For distributed training
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "ruff>=0.4.0",
]

[project.scripts]
nanogpt-rl-reward = "nanogpt_rl.scripts.train_reward_model:main"
nanogpt-rl-ppo = "nanogpt_rl.scripts.train_ppo:main"
nanogpt-rl-dpo = "nanogpt_rl.scripts.train_dpo:main"
nanogpt-rl-eval = "nanogpt_rl.scripts.evaluate_rl:main"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["nanogpt_rl*"]
EOF
```

---

### 10.2 Experiment Structure Extensions

**Reorganize existing experiments and add new phases:**

```bash
# Reorganize existing pre-training experiments
mkdir -p experiments/00-pretrain
mv experiments/01-baseline experiments/00-pretrain/
mv experiments/02-rope experiments/00-pretrain/

# Create SFT experiment directory
mkdir -p experiments/01-sft

# Create RL experiment directory
mkdir -p experiments/02-rl
```

**Create SFT experiment template:**

```bash
mkdir -p experiments/01-sft/01-baseline-sft/{writeup/figures,}

cat > experiments/01-sft/01-baseline-sft/config.yaml << 'EOF'
experiment:
  name: "baseline-sft"
  description: "SFT on Alpaca dataset using baseline pre-trained checkpoint"
  base_checkpoint: "../../outputs/pretrain/baseline/checkpoints/step_19072.pt"

model:
  # Inherits from pre-trained checkpoint
  # Can override specific parameters if needed

training:
  max_steps: 5000
  batch_size: 32
  sequence_length: 512         # Shorter for instruction data
  gradient_accumulation_steps: 4
  learning_rate: 2e-5          # Lower LR for fine-tuning
  warmup_steps: 100
  weight_decay: 0.01

data:
  dataset: "alpaca"
  format: "instruction"        # Instruction-following format
  train_split: "train"
  val_split: "validation"
  max_samples: null            # Use all data

evaluation:
  eval_interval: 100
  eval_iters: 20
  metrics:
    - "loss"
    - "perplexity"
    - "instruction_following"  # Custom metric

output:
  log_dir: "../../outputs/sft/baseline-sft"
  checkpoint_interval: 1000
  save_samples: true           # Save generated samples for qualitative analysis
EOF

cat > experiments/01-sft/01-baseline-sft/run.py << 'EOF'
#!/usr/bin/env python3
"""
Baseline SFT experiment.

Fine-tunes pre-trained baseline model on Alpaca instruction dataset.
"""

from pathlib import Path
import yaml

from nanogpt import GPT, GPTConfig
from nanogpt_sft import SFTTrainer, SFTConfig, load_pretrained_for_sft
from nanogpt_sft.datasets import load_alpaca

def main():
    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Load pre-trained checkpoint
    checkpoint_path = Path(config["experiment"]["base_checkpoint"])
    model = load_pretrained_for_sft(checkpoint_path)

    # Load SFT dataset
    train_dataset = load_alpaca(split="train")
    val_dataset = load_alpaca(split="validation")

    # Create SFT config
    sft_config = SFTConfig(
        max_steps=config["training"]["max_steps"],
        batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        output_dir=Path(config["output"]["log_dir"]),
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        config=sft_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    # Run training
    trainer.train()

if __name__ == "__main__":
    main()
EOF

chmod +x experiments/01-sft/01-baseline-sft/run.py

cat > experiments/01-sft/01-baseline-sft/pyproject.toml << 'EOF'
[project]
name = "experiment-baseline-sft"
version = "0.1.0"
description = "Baseline SFT experiment"
requires-python = ">=3.11"
dependencies = [
    "nanogpt",
    "nanogpt-sft",
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
EOF

cat > experiments/01-sft/01-baseline-sft/README.md << 'EOF'
# Experiment 01-SFT-01: Baseline SFT

## Overview

Fine-tunes the baseline pre-trained GPT-2 124M model on instruction-following data.

## Configuration

- **Base model:** Pre-trained baseline from `00-pretrain/01-baseline`
- **Dataset:** Alpaca (52k instruction-following examples)
- **Training:** 5,000 steps, lower learning rate (2e-5)
- **Format:** Instruction-input-output format

## Running

```bash
# From experiment directory
uv run python run.py

# Or from workspace root
cd experiments/01-sft/01-baseline-sft && uv run python run.py
```

## Outputs

- **Checkpoints:** `../../outputs/sft/baseline-sft/checkpoints/`
- **Metrics:** `../../outputs/sft/baseline-sft/metrics.jsonl`
- **Samples:** `../../outputs/sft/baseline-sft/samples.jsonl`

## Analysis

See `writeup/` for results and analysis notebooks.
EOF
```

**Create RL experiment templates:**

```bash
# Reward model experiment
mkdir -p experiments/02-rl/01-reward-model/{writeup/figures,}

cat > experiments/02-rl/01-reward-model/config.yaml << 'EOF'
experiment:
  name: "reward-model"
  description: "Train reward model on preference data"
  base_checkpoint: "../../outputs/sft/baseline-sft/checkpoints/step_5000.pt"

model:
  # Binary classifier head on top of GPT
  use_pooling: "last"          # Pool last token
  hidden_dim: 768

training:
  max_steps: 3000
  batch_size: 16
  learning_rate: 1e-5
  warmup_steps: 100

data:
  dataset: "hh-rlhf"           # Anthropic helpful/harmless
  subset: "helpful-base"
  max_pairs: 10000             # Limit for initial experiments

evaluation:
  eval_interval: 100
  metrics:
    - "accuracy"               # Preference prediction accuracy
    - "calibration"

output:
  log_dir: "../../outputs/rl/reward-model"
  checkpoint_interval: 500
EOF

# PPO experiment
mkdir -p experiments/02-rl/02-ppo-baseline/{writeup/figures,}

cat > experiments/02-rl/02-ppo-baseline/config.yaml << 'EOF'
experiment:
  name: "ppo-baseline"
  description: "PPO training using trained reward model"
  policy_checkpoint: "../../outputs/sft/baseline-sft/checkpoints/step_5000.pt"
  reward_model: "../../outputs/rl/reward-model/checkpoints/best.pt"

ppo:
  num_rollouts: 10000
  rollout_batch_size: 64
  ppo_epochs: 4
  learning_rate: 1e-6
  clip_epsilon: 0.2
  kl_penalty: 0.1              # KL divergence penalty weight
  value_coef: 0.5
  entropy_coef: 0.01

generation:
  max_length: 256
  temperature: 0.7
  top_p: 0.9

evaluation:
  eval_interval: 100
  metrics:
    - "reward"
    - "kl_divergence"
    - "perplexity"

output:
  log_dir: "../../outputs/rl/ppo-baseline"
  checkpoint_interval: 500
  save_rollouts: true          # Save generated rollouts
EOF

# DPO experiment (simpler alternative to PPO)
mkdir -p experiments/02-rl/03-dpo-baseline/{writeup/figures,}

cat > experiments/02-rl/03-dpo-baseline/config.yaml << 'EOF'
experiment:
  name: "dpo-baseline"
  description: "DPO training (simpler than PPO, no reward model needed)"
  base_checkpoint: "../../outputs/sft/baseline-sft/checkpoints/step_5000.pt"

dpo:
  max_steps: 3000
  batch_size: 16
  learning_rate: 5e-7
  beta: 0.1                    # DPO temperature parameter

data:
  dataset: "hh-rlhf"
  subset: "helpful-base"
  max_pairs: 10000

evaluation:
  eval_interval: 100
  metrics:
    - "loss"
    - "accuracy"               # Preference prediction
    - "implicit_reward"

output:
  log_dir: "../../outputs/rl/dpo-baseline"
  checkpoint_interval: 500
EOF
```

**Create experiment overview READMEs:**

```bash
cat > experiments/00-pretrain/README.md << 'EOF'
# Pre-training Experiments

Base GPT-2 training from scratch with architectural modifications.

## Experiments

- **01-baseline:** Standard GPT-2 124M
- **02-rope:** RoPE positional embeddings variant

## Pipeline Position

This is **Phase 0** of the training pipeline:

```
[Pre-train] → SFT → RL
```

Experiments here produce checkpoints used as starting points for SFT experiments.
EOF

cat > experiments/01-sft/README.md << 'EOF'
# SFT (Supervised Fine-Tuning) Experiments

Instruction fine-tuning on pre-trained models.

## Experiments

- **01-baseline-sft:** SFT on Alpaca using baseline pre-trained model
- **02-rope-sft:** SFT on Alpaca using RoPE pre-trained model

## Pipeline Position

This is **Phase 1** of the training pipeline:

```
Pre-train → [SFT] → RL
```

Experiments here:
- Load checkpoints from `00-pretrain/`
- Fine-tune on instruction-following datasets
- Produce SFT'd checkpoints for RL experiments

## Datasets

- **Alpaca:** 52k instruction-following examples
- **Dolly:** 15k open-source instructions
- **ShareGPT:** Conversational format

See `../../data/sft/` for dataset details.
EOF

cat > experiments/02-rl/README.md << 'EOF'
# RL (Reinforcement Learning) Experiments

RLHF and preference-based training.

## Experiments

- **01-reward-model:** Train preference/reward model
- **02-ppo-baseline:** PPO training with reward model
- **03-dpo-baseline:** DPO training (simpler, no reward model)

## Pipeline Position

This is **Phase 2** of the training pipeline:

```
Pre-train → SFT → [RL]
```

Experiments here:
- Load checkpoints from `01-sft/`
- Train on preference data
- Optimize for human preferences

## Approaches

### PPO (Proximal Policy Optimization)
- More complex, requires reward model
- Two-stage: train reward model, then policy
- Industry standard (ChatGPT, Claude)

### DPO (Direct Preference Optimization)
- Simpler, no reward model needed
- Single-stage training
- Good baseline for comparison

See individual experiment READMEs for details.
EOF
```

---

### 10.3 Data Organization Extensions

```bash
# Reorganize existing data
mkdir -p data/pretrain
mv data/edu_fineweb10B data/pretrain/
mv data/hellaswag data/pretrain/

# Create SFT data directories
mkdir -p data/sft/{alpaca,dolly,sharegpt,custom}

# Create RL data directories
mkdir -p data/rl/{hh-rlhf,summarization,custom-preferences}

# Create data READMEs
cat > data/sft/README.md << 'EOF'
# SFT Datasets

Instruction-following and chat datasets for supervised fine-tuning.

## Datasets

### Alpaca
- **Size:** 52k instruction-following examples
- **Format:** Instruction-input-output triples
- **Source:** Stanford Alpaca
- **License:** CC BY-NC 4.0

### Dolly
- **Size:** 15k instruction-following examples
- **Format:** Similar to Alpaca
- **Source:** Databricks Dolly
- **License:** CC BY-SA 3.0

### ShareGPT
- **Size:** Variable (community-contributed)
- **Format:** Multi-turn conversations
- **Source:** ShareGPT community
- **License:** Varies

## Format

All datasets are converted to a standard format:

```json
{
  "instruction": "What is the capital of France?",
  "input": "",
  "output": "The capital of France is Paris."
}
```

For multi-turn conversations:

```json
{
  "messages": [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help you?"},
    ...
  ]
}
```

## Preparation

See `packages/nanogpt-sft/scripts/prepare_sft_data.py` for data preparation scripts.
EOF

cat > data/rl/README.md << 'EOF'
# RL Datasets

Preference and reward modeling datasets for reinforcement learning.

## Datasets

### HH-RLHF (Helpful & Harmless)
- **Size:** ~170k preference pairs
- **Format:** Chosen vs rejected responses
- **Source:** Anthropic
- **License:** MIT

### Summarization Preferences
- **Size:** Variable
- **Format:** Summary quality preferences
- **Source:** OpenAI summarization dataset
- **License:** MIT

## Format

Preference pairs:

```json
{
  "prompt": "How do I bake a cake?",
  "chosen": "Here's a simple recipe: ...",
  "rejected": "I don't know."
}
```

## Preparation

See `packages/nanogpt-rl/scripts/train_reward_model.py` for usage.
EOF
```

---

### 10.4 Output Organization Extensions

```bash
# Reorganize existing outputs
mkdir -p outputs/pretrain
mv outputs/baseline outputs/pretrain/ 2>/dev/null || true
mv outputs/rope outputs/pretrain/ 2>/dev/null || true

# Create SFT output directories
mkdir -p outputs/sft

# Create RL output directories
mkdir -p outputs/rl

# Create output structure READMEs
cat > outputs/pretrain/README.md << 'EOF'
# Pre-training Outputs

Checkpoints and metrics from pre-training experiments.

## Structure

Each experiment produces:
- `checkpoints/` - Model checkpoints at intervals
- `metrics.jsonl` - Training metrics (loss, lr, etc.)
- `log.txt` - Console output

## Usage

Load checkpoints for SFT:

```python
from nanogpt_sft import load_pretrained_for_sft

model = load_pretrained_for_sft("outputs/pretrain/baseline/checkpoints/step_19072.pt")
```
EOF

cat > outputs/sft/README.md << 'EOF'
# SFT Outputs

Checkpoints and metrics from SFT experiments.

## Structure

Each experiment produces:
- `checkpoints/` - Fine-tuned model checkpoints
- `metrics.jsonl` - Training metrics
- `eval_results.json` - Instruction-following evaluation scores
- `samples.jsonl` - Generated samples for qualitative analysis

## Usage

Load checkpoints for RL:

```python
from nanogpt_rl import load_policy

policy = load_policy("outputs/sft/baseline-sft/checkpoints/step_5000.pt")
```
EOF

cat > outputs/rl/README.md << 'EOF'
# RL Outputs

Checkpoints and metrics from RL experiments.

## Structure

### Reward Model
- `checkpoints/` - Reward model checkpoints
- `validation_acc.jsonl` - Preference prediction accuracy

### PPO/DPO Training
- `checkpoints/` - Policy checkpoints
- `metrics.jsonl` - Training metrics
- `reward_stats.jsonl` - Reward distribution over time
- `kl_divergence.jsonl` - KL divergence from reference model
- `rollouts/` - Sample generations for qualitative analysis
EOF
```

---

### 10.5 Update Workspace Configuration

**Update root `pyproject.toml`:**

```bash
cat > pyproject.toml << 'EOF'
[project]
name = "frontier-ready"
version = "0.2.0"
description = "ML research: GPT-2 pre-training, SFT, and RL experimentation"
requires-python = ">=3.11"

[tool.uv.workspace]
members = [
    # Core packages
    "packages/nanogpt",
    "packages/exptools",

    # SFT & RL packages
    "packages/nanogpt-sft",
    "packages/nanogpt-rl",

    # Pre-training experiments
    "experiments/00-pretrain/*",

    # SFT experiments
    "experiments/01-sft/*",

    # RL experiments
    "experiments/02-rl/*",
]

[tool.uv]
dev-dependencies = [
    "ruff>=0.4.0",
    "ipython>=8.0.0",
    "jupyter>=1.0.0",
    "pytest>=7.4.0",
    "tensorboard>=2.14.0",    # For monitoring all phases
    "pyyaml>=6.0",            # Config file support
]
EOF
```

**Update `.gitignore`:**

```bash
cat >> .gitignore << 'EOF'

# SFT outputs
outputs/sft/*/checkpoints/*
outputs/sft/*/samples.jsonl
!outputs/sft/*/.gitkeep

# RL outputs
outputs/rl/*/checkpoints/*
outputs/rl/*/rollouts/*
!outputs/rl/*/.gitkeep

# SFT data (potentially large)
data/sft/alpaca/*
data/sft/dolly/*
data/sft/sharegpt/*
!data/sft/*/.gitkeep

# RL data
data/rl/hh-rlhf/*
!data/rl/*/.gitkeep
EOF
```

---

### 10.6 Create Minimal Implementation Stubs

**SFT package stub - `packages/nanogpt-sft/nanogpt_sft/__init__.py`:**

```bash
cat > packages/nanogpt-sft/nanogpt_sft/__init__.py << 'EOF'
"""
NanoGPT SFT: Supervised fine-tuning for instruction following.

Provides tools for:
- Loading and formatting instruction datasets
- Fine-tuning pre-trained models
- Evaluating instruction-following capability
"""

__version__ = "0.1.0"

from .trainer import SFTTrainer, SFTConfig
from .data import load_pretrained_for_sft

__all__ = [
    "SFTTrainer",
    "SFTConfig",
    "load_pretrained_for_sft",
]
EOF

cat > packages/nanogpt-sft/nanogpt_sft/data.py << 'EOF'
"""Data loading utilities for SFT."""

import torch
from pathlib import Path
from nanogpt import GPT

def load_pretrained_for_sft(checkpoint_path: Path) -> GPT:
    """
    Load a pre-trained model checkpoint for SFT.

    Args:
        checkpoint_path: Path to pre-trained checkpoint

    Returns:
        GPT model with loaded weights
    """
    # TODO: Implement checkpoint loading
    # For now, return stub
    raise NotImplementedError("SFT checkpoint loading not yet implemented")
EOF

cat > packages/nanogpt-sft/nanogpt_sft/trainer.py << 'EOF'
"""SFT training loop."""

from dataclasses import dataclass
from pathlib import Path

@dataclass
class SFTConfig:
    """Configuration for SFT training."""
    max_steps: int = 5000
    batch_size: int = 32
    learning_rate: float = 2e-5
    output_dir: Path = Path("outputs/sft")
    # Add more as needed

class SFTTrainer:
    """Trainer for supervised fine-tuning."""

    def __init__(self, model, config, train_dataset, val_dataset):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train(self):
        """Run SFT training."""
        # TODO: Implement SFT training loop
        raise NotImplementedError("SFT training not yet implemented")
EOF

cat > packages/nanogpt-sft/nanogpt_sft/README.md << 'EOF'
# NanoGPT SFT

Supervised fine-tuning package for instruction following.

## Status

⚠️ **Stub implementation** - Core functionality to be implemented.

## Installation

```bash
cd packages/nanogpt-sft
uv pip install -e .
```

## Usage

```python
from nanogpt_sft import SFTTrainer, load_pretrained_for_sft

# Load pre-trained model
model = load_pretrained_for_sft("path/to/checkpoint.pt")

# Configure and train
trainer = SFTTrainer(model, config, train_data, val_data)
trainer.train()
```

## Implementation TODO

- [ ] Checkpoint loading from pre-training
- [ ] Instruction dataset loaders (Alpaca, Dolly, ShareGPT)
- [ ] Prompt formatting and templating
- [ ] SFT training loop
- [ ] Instruction-following evaluation metrics
- [ ] Sample generation for qualitative analysis
EOF
```

**RL package stub - `packages/nanogpt-rl/nanogpt_rl/__init__.py`:**

```bash
cat > packages/nanogpt-rl/nanogpt_rl/__init__.py << 'EOF'
"""
NanoGPT RL: Reinforcement learning from human feedback.

Provides tools for:
- Training reward models from preferences
- PPO training with reward models
- DPO training (simpler alternative)
"""

__version__ = "0.1.0"

# Stub - to be implemented
__all__ = []
EOF

cat > packages/nanogpt-rl/nanogpt_rl/README.md << 'EOF'
# NanoGPT RL

Reinforcement learning package for RLHF.

## Status

⚠️ **Stub implementation** - Core functionality to be implemented.

## Approaches

### PPO (Proximal Policy Optimization)
- Two-stage: reward model → policy optimization
- Industry standard

### DPO (Direct Preference Optimization)
- Single-stage, no reward model
- Simpler implementation

## Implementation TODO

### Reward Model
- [ ] Preference dataset loaders
- [ ] Reward model architecture (binary classifier head)
- [ ] Reward model training loop
- [ ] Preference prediction evaluation

### PPO
- [ ] Rollout generation
- [ ] PPO training loop
- [ ] KL divergence tracking
- [ ] Reward curve analysis

### DPO
- [ ] DPO loss implementation
- [ ] Training loop
- [ ] Implicit reward extraction
EOF
```

---

### 10.7 Update Root README

**Update `README.md` to reflect full pipeline:**

```bash
cat > README.md << 'EOF'
# frontier-ready

A series of research & build tasks to build expertise working with frontier language models & language model systems.

## Project Structure

This is a workspace monorepo supporting the full modern LLM training pipeline:

```
Pre-training → SFT → RL
```

### Packages

Reusable components:

- **`nanogpt`** - Core GPT-2 training from scratch
- **`exptools`** - Metrics, visualization, analysis
- **`nanogpt-sft`** - Supervised fine-tuning (instruction following)
- **`nanogpt-rl`** - Reinforcement learning from human feedback (PPO, DPO)

### Experiments

Research experiments organized by training phase:

- **`experiments/00-pretrain/`** - Pre-training experiments (baseline, RoPE, etc.)
- **`experiments/01-sft/`** - Instruction fine-tuning experiments
- **`experiments/02-rl/`** - RLHF experiments (reward models, PPO, DPO)

### Data & Outputs

- **`data/`** - Shared datasets (pre-training, SFT, RL)
- **`outputs/`** - Training outputs and checkpoints
- **`infra/`** - Infrastructure as code (Pulumi/GCP)

## Quick Start

```bash
# Clone and setup
git clone <repo>
cd frontier-ready
uv sync

# Run pre-training experiment
cd experiments/00-pretrain/01-baseline
uv run python run.py

# Run SFT experiment (after pre-training)
cd experiments/01-sft/01-baseline-sft
uv run python run.py

# Run RL experiment (after SFT)
cd experiments/02-rl/02-ppo-baseline
uv run python run.py
```

## Pipeline Workflow

1. **Pre-train** a base model from scratch
2. **SFT** on instruction-following data
3. **RL** optimization with human preferences

Each phase builds on the previous, but can also be run independently for experimentation.

## Current Status

- ✅ Pre-training infrastructure complete
- ✅ Workspace structure established
- 🚧 SFT packages (stub implementation)
- 🚧 RL packages (stub implementation)

See `PLAN.md` for detailed research roadmap.
See `RESTRUCTURE_EXECUTION_PLAN.md` for technical implementation details.
EOF
```

---

### 10.8 Testing and Validation

**Verify package installations:**

```bash
cd /Users/jex/frontier-ready

# Sync workspace with new packages
uv sync

# Test imports
uv run python -c "from nanogpt import GPT; print('✓ nanogpt')"
uv run python -c "from exptools import load_metrics; print('✓ exptools')"
uv run python -c "import nanogpt_sft; print('✓ nanogpt-sft')"
uv run python -c "import nanogpt_rl; print('✓ nanogpt-rl')"

# Verify experiment structure
ls -la experiments/00-pretrain/
ls -la experiments/01-sft/
ls -la experiments/02-rl/

# Verify data structure
ls -la data/pretrain/
ls -la data/sft/
ls -la data/rl/
```

**Create verification checklist:**

```bash
cat > PHASE10_CHECKLIST.md << 'EOF'
# Phase 10: SFT/RL Extensions - Verification Checklist

## Package Creation
- [ ] `packages/nanogpt-sft/` created with proper structure
- [ ] `packages/nanogpt-rl/` created with proper structure
- [ ] Both packages have pyproject.toml with correct dependencies
- [ ] Both packages have stub implementations
- [ ] READMEs document current status and TODOs

## Experiment Structure
- [ ] Pre-training experiments moved to `experiments/00-pretrain/`
- [ ] SFT experiment template created in `experiments/01-sft/01-baseline-sft/`
- [ ] RL experiment templates created in `experiments/02-rl/`
- [ ] All experiment directories have proper READMEs
- [ ] Config files are well-documented

## Data Organization
- [ ] Pre-training data moved to `data/pretrain/`
- [ ] SFT data directories created in `data/sft/`
- [ ] RL data directories created in `data/rl/`
- [ ] Data READMEs document formats and sources

## Output Organization
- [ ] Pre-training outputs moved to `outputs/pretrain/`
- [ ] SFT output directories created
- [ ] RL output directories created
- [ ] Output READMEs document structure

## Workspace Configuration
- [ ] Root pyproject.toml updated with new packages
- [ ] .gitignore updated for new directories
- [ ] All workspace members can be synced with `uv sync`

## Package Installation
- [ ] `uv sync` completes without errors
- [ ] Can import nanogpt-sft
- [ ] Can import nanogpt-rl
- [ ] No broken dependencies

## Documentation
- [ ] Root README updated to reflect full pipeline
- [ ] RESTRUCTURE_EXECUTION_PLAN.md includes Phase 10
- [ ] All new directories have READMEs
- [ ] Implementation TODOs are clearly marked

## Next Steps
- [ ] Implement SFT checkpoint loading
- [ ] Implement SFT dataset loaders
- [ ] Implement SFT training loop
- [ ] Implement reward model training
- [ ] Implement PPO algorithm
- [ ] Implement DPO algorithm
EOF
```

---

### 10.9 Phase 10 Completion Criteria

**Phase 10 is complete when:**

1. ✅ Package structure created for SFT and RL
2. ✅ Experiment templates created for all pipeline phases
3. ✅ Data and output directories reorganized
4. ✅ Workspace configuration updated
5. ✅ All packages install without errors
6. ✅ Documentation reflects full pipeline
7. ✅ Stub implementations provide clear API contracts
8. ✅ Implementation TODOs documented for future work

**What Phase 10 does NOT include:**

- ❌ Full SFT implementation (stub only)
- ❌ Full RL implementation (stub only)
- ❌ Dataset downloads or preparation
- ❌ Actual training runs

**These are left as future work** to be implemented incrementally as needed for specific experiments.

---

### 10.10 Timeline Estimate for Phase 10

- Package structure creation: 1 hour
- Experiment template creation: 1.5 hours
- Data/output reorganization: 45 minutes
- Documentation updates: 1 hour
- Testing and validation: 45 minutes
- **Total: 5 hours**

---

## Integration with Existing Plan

**Phase 10 enables:**

- **PLAN.md Phase 1a (SFT fundamentals)** - Infrastructure ready, just implement training loop
- **PLAN.md Phase 1b (RL fundamentals)** - Infrastructure ready, just implement algorithms
- **Future agent experiments** - Can use RL-trained models as agent backbones

**Phase 10 maintains:**

- ✅ Additive approach - doesn't modify existing pre-training code
- ✅ Modular design - each package has clear boundaries
- ✅ Consistent structure - same patterns across all phases
- ✅ Professional standards - industry-standard pipeline organization

---

## Next Steps After This Document

1. Review this plan
2. Make any necessary adjustments
3. Execute phases in order
4. Test thoroughly after each phase
5. Commit frequently with clear messages

Ready to proceed?
