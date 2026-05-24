# frontier-ready

A series of research & build tasks to build expertise working with frontier language models & language model systems.

## Project Structure

This is a workspace monorepo supporting the full modern LLM training pipeline:

```
Pre-training → SFT → RL → Agents
```

### Packages

Reusable components:

- **`nanogpt`** - Core GPT-2 training from scratch (ported from build-nanogpt)
- **`exptools`** - Metrics, visualization, and analysis tools

### Experiments

Research experiments organized by training phase:

- **`experiments/01-baseline/`** - Standard GPT-2 124M training
- **`experiments/02-rope/`** - RoPE positional embeddings variant

### Data & Outputs

- **`data/`** - Shared datasets (FineWeb-Edu, HellaSwag)
- **`outputs/`** - Training outputs and checkpoints
- **`infra/`** - Infrastructure as code (Pulumi/GCP)

## Quick Start

```bash
# Clone and setup
git clone <repo>
cd frontier-ready
uv sync

# Run baseline experiment
cd experiments/01-baseline
uv run python run.py

# Run RoPE variant experiment
cd experiments/02-rope
uv run python run.py
```

## Current Status

- ✅ Workspace structure established
- ✅ Phase 1 (Pre-training) infrastructure complete
- 📋 Phase 1a (SFT) - planned
- 📋 Phase 1b (RL) - planned
- 📋 Phase 2 (Agent benchmarks) - planned

See `PLAN.md` for detailed research roadmap.

## Acknowledgments

This project builds on [Andrej Karpathy's build-nanogpt](https://github.com/karpathy/build-nanogpt), which provides a minimal, educational implementation of GPT-2. The code has been restructured into a workspace monorepo for experimentation.
