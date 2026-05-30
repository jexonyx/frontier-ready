"""LLMKit - A modular LLM toolkit for educational purposes.

Based on Andrej Karpathy's build-nanogpt video lecture series:
https://github.com/karpathy/build-nanogpt
"""

# New modular API
from .config import ModelConfig, AttentionConfig, FeedForwardConfig, InitializationConfig
from .models import GPT
from .core import (
    TransformerBlock,
    CausalSelfAttention,
    MLP,
    ScaledLinear,
    PositionEmbedding,
    LearnedPositionEmbedding,
    RoPEPositionEmbedding,
    NoPositionEmbedding,
)

# Data, training, and evaluation
from .data import DataLoaderLite, load_tokens
from .training import Trainer, get_cosine_lr_schedule
from .generation import generate
from .eval import (
    evaluate_hellaswag,
    get_most_likely_row,
    iterate_examples,
    render_example,
    download as download_hellaswag
)

__version__ = "0.1.0"

__all__ = [
    # New modular API - Config
    'ModelConfig',
    'AttentionConfig',
    'FeedForwardConfig',
    'InitializationConfig',
    # New modular API - Model
    'GPT',
    'TransformerBlock',
    'CausalSelfAttention',
    'MLP',
    'ScaledLinear',
    # New modular API - Position embeddings
    'PositionEmbedding',
    'LearnedPositionEmbedding',
    'RoPEPositionEmbedding',
    'NoPositionEmbedding',
    # Data
    'DataLoaderLite',
    'load_tokens',
    # Training
    'Trainer',
    'get_cosine_lr_schedule',
    # Generation
    'generate',
    # Evaluation
    'evaluate_hellaswag',
    'get_most_likely_row',
    'iterate_examples',
    'render_example',
    'download_hellaswag',
]
