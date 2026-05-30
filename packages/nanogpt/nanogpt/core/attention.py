"""Attention mechanisms."""
import torch
import torch.nn as nn
from torch.nn import functional as F
from ..config.component_config import AttentionConfig
from .initialization import ScaledLinear


class CausalSelfAttention(nn.Module):
    """Causal self-attention with optional flash attention.

    Multi-head causal self-attention using scaled dot-product attention.
    Supports flash attention for improved performance.
    """

    def __init__(self, config: AttentionConfig, layer_idx: int = 0, n_layer: int = 1):
        """Initialize causal self-attention.

        Args:
            config: Attention configuration
            layer_idx: Index of this layer (for residual scaling)
            n_layer: Total number of layers (for residual scaling)
        """
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash_attention = config.flash_attention

        # Key, query, value projections for all heads, in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # Output projection with residual scaling
        scale_factor = (2 * n_layer) ** -0.5
        self.c_proj = ScaledLinear(
            config.n_embd,
            config.n_embd,
            scale_factor=scale_factor,
            bias=config.bias,
        )

        # Dropout (if needed)
        if self.dropout > 0:
            self.attn_dropout = nn.Dropout(config.dropout)
            self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """Apply causal self-attention.

        Args:
            x: Input tensor of shape (B, T, n_embd)

        Returns:
            Output tensor of shape (B, T, n_embd)
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash_attention:
            # Efficient attention using Flash Attention CUDA kernels
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # Manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
            att = att.masked_fill(
                torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1),
                float('-inf')
            )
            att = F.softmax(att, dim=-1)
            if self.dropout > 0:
                att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.c_proj(y)
        if self.dropout > 0:
            y = self.resid_dropout(y)

        return y
