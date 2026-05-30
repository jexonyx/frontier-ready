"""Transformer block implementation."""
import torch.nn as nn
from ..config.model_config import ModelConfig
from .attention import CausalSelfAttention
from .feedforward import MLP


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture.

    A transformer block consists of:
    1. Layer norm + causal self-attention + residual
    2. Layer norm + feed-forward network + residual

    This follows the GPT-2 pre-normalization architecture.
    """

    # Class attributes for customization (can be overridden in subclasses)
    attn_cls = CausalSelfAttention
    mlp_cls = MLP
    norm_cls = nn.LayerNorm

    def __init__(self, config: ModelConfig, layer_idx: int):
        """Initialize transformer block.

        Args:
            config: Model configuration
            layer_idx: Index of this block in the model
        """
        super().__init__()

        # Layer normalizations
        self.ln_1 = self.norm_cls(config.n_embd)
        self.ln_2 = self.norm_cls(config.n_embd)

        # Attention with component-specific config
        self.attn = self.attn_cls(
            config.attention_config,
            layer_idx=layer_idx,
            n_layer=config.n_layer,
        )

        # Feed-forward network with component-specific config
        self.mlp = self.mlp_cls(
            config.feedforward_config,
            layer_idx=layer_idx,
            n_layer=config.n_layer,
        )

    def forward(self, x):
        """Apply transformer block.

        Args:
            x: Input tensor of shape (B, T, n_embd)

        Returns:
            Output tensor of shape (B, T, n_embd)
        """
        # Pre-norm architecture with residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
