# Study 1: GPT-2 124M Reproduction & Architectural Modification

Reproduce GPT-2 124M training using Karpathy's build-nanogpt, then introduce one architectural modification and compare loss curves.

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- For training: NVIDIA GPU with CUDA (H100 recommended)

### Install

```bash
git clone --recurse-submodules <repo-url>
cd frontier-ready/study1-gpt
uv venv && uv pip install -e ".[dev,analysis]"
```

For remote GPU boxes with CUDA:
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Data Preparation

```bash
# Tokenize FineWeb-Edu (produces edu_fineweb10B/)
python build-nanogpt/fineweb.py
```

## Experiments

### Baseline: GPT-2 124M

Reproduce the standard GPT-2 124M training run following build-nanogpt.

### Modification

_TBD — architectural change to compare against baseline._

## Results

_Loss curves and analysis will be added here._

## Reproducing

_Commands and expected compute cost will be documented here._

## Project Structure

```
study1-gpt/
  build-nanogpt/           # Karpathy's nanogpt repo (submodule)
  configs/                 # Experiment configuration
  analysis/                # Plotting and analysis scripts
  writeup/                 # 1,500-word writeup and figures
  nanogpt_replication/     # Python package (for editable install)
  pyproject.toml           # Dependencies and project config
```
