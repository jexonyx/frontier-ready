"""Tests for position embedding implementations."""
import pytest
import torch
from nanogpt.core.embedding import (
    LearnedPositionEmbedding,
    RoPEPositionEmbedding,
    NoPositionEmbedding,
)


class TestLearnedPositionEmbedding:
    """Tests for learned position embeddings."""

    def test_learned_embedding_shape(self):
        """Test learned position embeddings preserve shape."""
        block_size = 32
        n_embd = 128
        embedding = LearnedPositionEmbedding(block_size, n_embd)

        batch_size = 2
        seq_len = 16
        token_emb = torch.randn(batch_size, seq_len, n_embd)

        output = embedding(token_emb)
        assert output.shape == (batch_size, seq_len, n_embd)

    def test_learned_embedding_adds_position_info(self):
        """Test that learned embeddings add position information."""
        embedding = LearnedPositionEmbedding(block_size=32, n_embd=128)

        token_emb = torch.randn(2, 16, 128)
        output = embedding(token_emb)

        # Output should be different from input
        assert not torch.allclose(output, token_emb)

    def test_learned_embedding_with_positions(self):
        """Test learned embeddings with explicit positions."""
        embedding = LearnedPositionEmbedding(block_size=32, n_embd=128)

        token_emb = torch.randn(2, 16, 128)
        positions = torch.arange(0, 16, dtype=torch.long)

        output = embedding(token_emb, positions)
        assert output.shape == token_emb.shape

    def test_learned_embedding_parameters(self):
        """Test that learned embeddings have trainable parameters."""
        embedding = LearnedPositionEmbedding(block_size=32, n_embd=128)

        params = list(embedding.parameters())
        assert len(params) == 1  # Only wpe.weight
        assert params[0].shape == (32, 128)
        assert params[0].requires_grad


class TestRoPEPositionEmbedding:
    """Tests for RoPE position embeddings."""

    def test_rope_embedding_shape(self):
        """Test RoPE embedding preserves shape (returns input unchanged)."""
        embedding = RoPEPositionEmbedding(
            block_size=32,
            n_embd=128,
            n_head=4,
        )

        batch_size = 2
        seq_len = 16
        token_emb = torch.randn(batch_size, seq_len, 128)

        output = embedding(token_emb)
        assert output.shape == (batch_size, seq_len, 128)
        # RoPE returns input unchanged (rotation happens in attention)
        assert torch.allclose(output, token_emb)

    def test_rope_precomputes_frequencies(self):
        """Test that RoPE precomputes frequencies."""
        embedding = RoPEPositionEmbedding(
            block_size=32,
            n_embd=128,
            n_head=4,
        )

        # Check that frequencies are precomputed
        assert hasattr(embedding, "inv_freq")
        assert hasattr(embedding, "cos_cached")
        assert hasattr(embedding, "sin_cached")

        # Check shapes
        head_dim = 128 // 4  # 32
        assert embedding.inv_freq.shape == (head_dim // 2,)
        assert embedding.cos_cached.shape == (32, head_dim)
        assert embedding.sin_cached.shape == (32, head_dim)

    def test_rope_get_cos_sin(self):
        """Test getting cos and sin values."""
        embedding = RoPEPositionEmbedding(
            block_size=32,
            n_embd=128,
            n_head=4,
        )

        seq_len = 16
        cos, sin = embedding.get_cos_sin(seq_len, torch.device("cpu"))

        head_dim = 128 // 4
        assert cos.shape == (seq_len, head_dim)
        assert sin.shape == (seq_len, head_dim)

    def test_rope_no_parameters(self):
        """Test that RoPE has no trainable parameters (only buffers)."""
        embedding = RoPEPositionEmbedding(
            block_size=32,
            n_embd=128,
            n_head=4,
        )

        # Buffers should exist but not be parameters
        params = list(embedding.parameters())
        assert len(params) == 0

        # Buffers should exist
        buffers = dict(embedding.named_buffers())
        assert "inv_freq" in buffers
        assert "cos_cached" in buffers
        assert "sin_cached" in buffers


class TestNoPositionEmbedding:
    """Tests for no position embedding."""

    def test_no_embedding_returns_input(self):
        """Test that NoPositionEmbedding returns input unchanged."""
        embedding = NoPositionEmbedding()

        token_emb = torch.randn(2, 16, 128)
        output = embedding(token_emb)

        assert torch.allclose(output, token_emb)

    def test_no_embedding_shape(self):
        """Test that NoPositionEmbedding preserves shape."""
        embedding = NoPositionEmbedding()

        batch_size = 2
        seq_len = 16
        n_embd = 128
        token_emb = torch.randn(batch_size, seq_len, n_embd)

        output = embedding(token_emb)
        assert output.shape == (batch_size, seq_len, n_embd)

    def test_no_embedding_no_parameters(self):
        """Test that NoPositionEmbedding has no parameters."""
        embedding = NoPositionEmbedding()

        params = list(embedding.parameters())
        assert len(params) == 0
