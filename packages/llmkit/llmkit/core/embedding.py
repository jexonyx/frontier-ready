"""Position embedding implementations."""
from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor


class PositionEmbedding(ABC, nn.Module):
    """Protocol for position embedding strategies.

    Different position embedding approaches (learned, RoPE, ALiBi, etc.)
    can be implemented by subclassing this interface.
    """

    @abstractmethod
    def forward(self, token_embeddings: Tensor, positions: Optional[Tensor] = None) -> Tensor:
        """Apply position information to token embeddings.

        Args:
            token_embeddings: Token embeddings of shape (B, T, n_embd)
            positions: Optional position indices of shape (T,). If None, uses range(T).

        Returns:
            Embeddings with position information applied, shape (B, T, n_embd)
        """
        pass


class LearnedPositionEmbedding(PositionEmbedding):
    """Standard GPT-2 style learned position embeddings.

    Position embeddings are learned parameters that are added to token embeddings.
    """

    def __init__(self, block_size: int, n_embd: int):
        """Initialize learned position embeddings.

        Args:
            block_size: Maximum sequence length
            n_embd: Embedding dimension
        """
        super().__init__()
        self.block_size = block_size
        self.wpe = nn.Embedding(block_size, n_embd)

    def forward(self, token_embeddings: Tensor, positions: Optional[Tensor] = None) -> Tensor:
        """Add learned position embeddings to token embeddings.

        Args:
            token_embeddings: Token embeddings of shape (B, T, n_embd)
            positions: Optional position indices of shape (T,)

        Returns:
            token_embeddings + position_embeddings
        """
        B, T, C = token_embeddings.size()
        if positions is None:
            positions = torch.arange(0, T, dtype=torch.long, device=token_embeddings.device)
        pos_emb = self.wpe(positions)  # (T, n_embd)
        return token_embeddings + pos_emb


class RoPEPositionEmbedding(PositionEmbedding):
    """Rotary Position Embeddings (RoPE).

    Instead of adding position embeddings, RoPE rotates the query and key
    vectors in attention based on their positions. This is applied in the
    attention mechanism itself, not here.

    For now, this is a placeholder that returns token embeddings unchanged.
    The actual rotation happens in the attention layer.
    """

    def __init__(self, block_size: int, n_embd: int, n_head: int, base: float = 10000.0):
        """Initialize RoPE embeddings.

        Args:
            block_size: Maximum sequence length
            n_embd: Embedding dimension
            n_head: Number of attention heads
            base: Base for frequency computation (default: 10000.0)
        """
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.base = base

        # Compute head dimension
        self.head_dim = n_embd // n_head

        # Precompute frequency tensor
        # inv_freq shape: (head_dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute position encodings for max sequence length
        self._precompute_freqs(block_size)

    def _precompute_freqs(self, seq_len: int):
        """Precompute frequency matrix for given sequence length."""
        # t shape: (seq_len,)
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        # freqs shape: (seq_len, head_dim // 2)
        freqs = torch.outer(t, self.inv_freq)
        # emb shape: (seq_len, head_dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        # Store cos and sin
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, token_embeddings: Tensor, positions: Optional[Tensor] = None) -> Tensor:
        """Return token embeddings unchanged (RoPE is applied in attention).

        Args:
            token_embeddings: Token embeddings of shape (B, T, n_embd)
            positions: Unused for RoPE

        Returns:
            token_embeddings unchanged
        """
        return token_embeddings

    def get_cos_sin(self, seq_len: int, device: torch.device):
        """Get precomputed cos and sin values for given sequence length.

        Args:
            seq_len: Sequence length
            device: Device to place tensors on

        Returns:
            Tuple of (cos, sin) tensors, each of shape (seq_len, head_dim)
        """
        if seq_len > self.block_size:
            # Recompute if sequence exceeds precomputed length
            self._precompute_freqs(seq_len)
        return self.cos_cached[:seq_len].to(device), self.sin_cached[:seq_len].to(device)


class NoPositionEmbedding(PositionEmbedding):
    """No position embedding.

    Useful for models that use alternative position encoding methods
    like ALiBi (Attention with Linear Biases) that modify attention directly.
    """

    def forward(self, token_embeddings: Tensor, positions: Optional[Tensor] = None) -> Tensor:
        """Return token embeddings unchanged.

        Args:
            token_embeddings: Token embeddings of shape (B, T, n_embd)
            positions: Unused

        Returns:
            token_embeddings unchanged
        """
        return token_embeddings
