"""Model variants with different architectures."""
from .rope import RoPEGPT, create_rope_gpt

__all__ = ['RoPEGPT', 'create_rope_gpt']
