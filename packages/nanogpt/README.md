# NanoGPT

A minimal, educational implementation of GPT (Generative Pre-trained Transformer) for language modeling.

## Acknowledgments

This package is based on **Andrej Karpathy's excellent [build-nanogpt](https://github.com/karpathy/build-nanogpt)** video lecture series. The original code demonstrates how to build and train a GPT model from scratch with clear, educational implementations.

This reorganized version preserves all the original functionality while providing a clean package structure for easier reuse and experimentation.

## Installation

Install the package in editable mode:

```bash
uv pip install -e packages/nanogpt
```

Or with pip:

```bash
pip install -e packages/nanogpt
```

## Usage

### Data Preparation

First, download and tokenize the FineWeb-Edu dataset:

```bash
python scripts/prepare_data.py
```

This will create data shards in the `edu_fineweb10B` directory.

### Training

Train a GPT model from scratch:

```bash
# Single GPU
python scripts/train.py --run-name my_run

# Multi-GPU with DDP (e.g., 8 GPUs)
torchrun --standalone --nproc_per_node=8 scripts/train.py --run-name my_run

# Resume from checkpoint
python scripts/train.py --run-name my_run --resume log/my_run/model_10000.pt
```

Training options:
- `--run-name`: Name for this run (determines log subdirectory)
- `--resume`: Checkpoint path to resume from
- `--data-root`: Root directory containing data shards (default: edu_fineweb10B)
- `--batch-size`: Micro batch size (default: 64)
- `--seq-length`: Sequence length (default: 1024)
- `--total-batch-size`: Total batch size in tokens (default: 524288)
- `--max-lr`: Maximum learning rate (default: 6e-4)
- `--max-steps`: Maximum training steps (default: 19073)
- `--eval-interval`: Steps between evaluation (default: 250)
- `--checkpoint-interval`: Steps between checkpoints (default: 1000)

### Evaluation

Evaluate a trained model on HellaSwag:

```bash
python scripts/evaluate.py --checkpoint log/my_run/model_19072.pt
```

### Text Generation

Generate text from a trained model:

```bash
python scripts/generate.py \
    --checkpoint log/my_run/model_19072.pt \
    --prompt "Hello, I'm a language model," \
    --num-samples 5 \
    --max-length 100 \
    --temperature 0.9
```

## Package Structure

```
nanogpt/
├── __init__.py         # Package exports
├── config/             # Configuration classes
│   ├── model_config.py # ModelConfig dataclass
│   └── component_config.py # Component configs (Attention, FeedForward, Initialization)
├── core/               # Core model components
│   ├── attention.py    # CausalSelfAttention
│   ├── feedforward.py  # MLP
│   ├── block.py        # TransformerBlock
│   ├── embedding.py    # Position embeddings (Learned, RoPE, None)
│   └── linear.py       # ScaledLinear
├── models/             # Full model implementations
│   └── gpt.py          # GPT model
├── training.py         # Trainer class with training loop
├── data.py             # DataLoaderLite and data utilities
├── eval.py             # HellaSwag evaluation
└── generation.py       # Text generation utilities
```

## Python API

You can also use the package programmatically:

```python
import torch
from nanogpt import GPT, ModelConfig, DataLoaderLite, Trainer, generate

# Create a model with default settings
config = ModelConfig(vocab_size=50304, n_layer=12, n_head=12, n_embd=768)
model = GPT(config)

# Or customize components
from nanogpt import AttentionConfig, FeedForwardConfig

config = ModelConfig(
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embd=768,
    attention=AttentionConfig(dropout=0.1),
    feedforward=FeedForwardConfig(expansion_factor=4, activation='gelu'),
    position_embedding_type='rope'  # Use RoPE instead of learned embeddings
)
model = GPT(config)

# Or load pretrained GPT-2 weights
model = GPT.from_pretrained('gpt2')

# Generate text
results = generate(
    model=model,
    prompt="Once upon a time",
    max_length=100,
    num_samples=3,
    device='cuda'
)

# Training
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device='cuda',
    device_type='cuda',
    log_dir='log/my_run'
)
trainer.train()
```

## Position Embedding Variants

The package supports multiple position embedding strategies via the modular API:

### Learned Position Embeddings (Default)

```python
from nanogpt import GPT, ModelConfig

config = ModelConfig(position_embedding_type='learned')  # Default
model = GPT(config)
```

### RoPE (Rotary Position Embeddings)

```python
from nanogpt import GPT, ModelConfig

config = ModelConfig(position_embedding_type='rope')
model = GPT(config)
```

### No Position Embeddings

```python
from nanogpt import GPT, ModelConfig

config = ModelConfig(position_embedding_type='none')
model = GPT(config)
```

## Features

- **Clean architecture**: Modular design with separate components for model, training, data, and evaluation
- **DDP support**: Multi-GPU training with DistributedDataParallel
- **Comprehensive logging**: Training metrics, validation loss, HellaSwag accuracy, per-layer statistics
- **Checkpointing**: Save and resume training from checkpoints
- **Text generation**: Top-k sampling with temperature control
- **Evaluation**: HellaSwag benchmark for common-sense reasoning
- **Learning rate scheduling**: Cosine decay with warmup

## Dataset

The default training uses the [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset (10B token sample), a high-quality educational text dataset from Hugging Face.

## License

MIT License (following the original build-nanogpt)

## Credits

- **Andrej Karpathy** for the original [build-nanogpt](https://github.com/karpathy/build-nanogpt) implementation and excellent educational content
- The GPT-2 architecture from OpenAI
- FineWeb-Edu dataset from Hugging Face
- HellaSwag benchmark from Zellers et al.
