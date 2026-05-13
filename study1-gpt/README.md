# Study 1: GPT-2 124M Reproduction & Architectural Modification

Reproduce GPT-2 124M training using Karpathy's build-nanogpt, then introduce one architectural modification and compare loss curves.

## Quick Start

### On your GPU VM:

```bash
# Clone the repo
cd /data
git clone --recurse-submodules https://github.com/your-username/frontier-ready.git
cd frontier-ready/study1-gpt

# Setup (one command!)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
uv venv && source .venv/bin/activate
uv pip install -e .
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# Run FineWeb data preparation
cd build-nanogpt
python fineweb.py

# Start training
python train_gpt2.py
```

That's it! No complicated package installations.

## What This Does

1. **FineWeb preparation** (`fineweb.py`): Downloads and tokenizes 10B tokens from FineWeb-Edu dataset
   - Takes ~1-2 hours
   - Creates `edu_fineweb10B/` with ~100 shards
   - Each shard is 100M tokens

2. **Training** (`train_gpt2.py`): Trains GPT-2 124M on the prepared data
   - Uses all available GPU memory
   - Saves checkpoints to `log/`
   - Logs training metrics

## Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA
- [uv](https://docs.astral.sh/uv/) (installed in setup commands above)

## Project Structure

```
study1-gpt/
  build-nanogpt/           # Karpathy's nanogpt scripts (submodule, not installed as package)
    ├── fineweb.py         # Data preparation
    ├── train_gpt2.py      # Training script
    └── hellaswag.py       # Evaluation
  pyproject.toml           # Just lists dependencies - simple!
  README.md                # This file
```

## Why It's Simple Now

- `build-nanogpt/` is just a directory of scripts (no package installation needed)
- All dependencies are in one place: `pyproject.toml`
- One install command: `uv pip install -e .`
- PyTorch installed separately to choose CUDA version

## Plan

1. Instrument `train_gpt2.py` with comprehensive logging
2. Pick architectural modification
3. Run baseline GPT-2 124M training
4. Implement the modification
5. Run modified training
6. Analysis - plot loss curves, run HellaSwag comparison
7. Writeup - 1,500 words: modification, hypothesis, result
8. Clean up for public access

## Experiments

### Baseline: GPT-2 124M

Reproduce the standard GPT-2 124M training run following build-nanogpt.

### Modification

_TBD — architectural change to compare against baseline._

## Results

_Loss curves and analysis will be added here._
