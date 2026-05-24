"""NanoGPT - A minimal implementation of GPT for educational purposes.

Based on Andrej Karpathy's build-nanogpt video lecture series:
https://github.com/karpathy/build-nanogpt
"""

from .config import GPTConfig
from .model import GPT, Block, CausalSelfAttention, MLP
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
    # Config
    'GPTConfig',
    # Model
    'GPT',
    'Block',
    'CausalSelfAttention',
    'MLP',
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
