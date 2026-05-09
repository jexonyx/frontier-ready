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
uv venv && uv pip install -r pyproject.toml --all-extras
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

## Plan

1. Instrument `train_gpt2.py` with comprehensive logging — capture all valuable metrics so the baseline never needs re-running
2. Pick architectural modification (see candidates in `../IDEAS.md`)
3. Run baseline GPT-2 124M training — rent H100, prepare FineWeb-Edu data, run `train_gpt2.py`, verify loss curve reproduces Karpathy's result
4. Implement the modification — fork `train_gpt2.py`, make the single-variable change
5. Run modified training — same data, same hyperparameters, only the architectural change differs
6. Analysis — plot loss curves side by side, run HellaSwag comparison
7. Writeup — 1,500 words: modification, hypothesis, result, what comes next
8. Clean up project and README for long-term public access

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
  analysis/                # Plotting and analysis scripts
  writeup/                 # 1,500-word writeup and figures
  pyproject.toml           # Dependencies
```
