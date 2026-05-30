"""Component-specific configuration classes.

These focused configuration classes replace the monolithic GPTConfig,
allowing each component to depend only on the parameters it needs.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms.

    Args:
        n_embd: Embedding dimension
        n_head: Number of attention heads
        dropout: Dropout probability (default: 0.0)
        bias: Whether to use bias in projections (default: True for GPT-2 compatibility)
        flash_attention: Whether to use flash attention (default: True)
    """
    n_embd: int
    n_head: int
    dropout: float = 0.0
    bias: bool = True
    flash_attention: bool = True

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")


@dataclass
class FeedForwardConfig:
    """Configuration for feed-forward networks.

    Args:
        n_embd: Embedding dimension
        expansion_factor: Hidden dimension expansion factor (default: 4)
        activation: Activation function name (default: "gelu")
        activation_kwargs: Additional kwargs for activation (default: None)
        dropout: Dropout probability (default: 0.0)
        bias: Whether to use bias in projections (default: True)
    """
    n_embd: int
    expansion_factor: int = 4
    activation: str = "gelu"
    activation_kwargs: Optional[Dict[str, Any]] = None
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self):
        if self.activation_kwargs is None:
            # Default GELU kwargs for GPT-2 compatibility
            if self.activation == "gelu":
                self.activation_kwargs = {"approximate": "tanh"}
            else:
                self.activation_kwargs = {}


@dataclass
class InitializationConfig:
    """Configuration for weight initialization.

    Args:
        std: Standard deviation for weight initialization (default: 0.02)
        residual_scale_factor: Factor to scale residual projections (default: None)
            If None, computed as (2 * n_layer) ** -0.5
    """
    std: float = 0.02
    residual_scale_factor: Optional[float] = None
