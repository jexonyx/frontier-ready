"""Tests for GPT model."""
import pytest
import torch
from nanogpt import GPT, ModelConfig
from nanogpt.core.embedding import RoPEPositionEmbedding, LearnedPositionEmbedding


class TestGPT:
    """Tests for GPT model."""

    def test_model_creation(self, small_config):
        """Test model creation."""
        model = GPT(small_config)
        assert model is not None
        assert isinstance(model, GPT)

    def test_model_forward(self, small_model, dummy_batch):
        """Test forward pass."""
        x, y = dummy_batch
        logits, loss = small_model(x, y)

        # Check shapes
        batch_size, seq_len = x.shape
        vocab_size = small_model.config.vocab_size
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert loss is not None
        assert loss.item() > 0

    def test_model_forward_no_targets(self, small_model, dummy_batch):
        """Test forward pass without targets."""
        x, _ = dummy_batch
        logits, loss = small_model(x)

        # Check shapes
        batch_size, seq_len = x.shape
        vocab_size = small_model.config.vocab_size
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert loss is None

    def test_parameter_count(self):
        """Test parameter count matches expected value."""
        # GPT-2 124M configuration
        config = ModelConfig(n_layer=12, n_head=12, n_embd=768)
        model = GPT(config)
        params = sum(p.numel() for p in model.parameters())

        # Expected parameter count for GPT-2 124M
        assert params == 124_439_808

    def test_learned_position_embedding(self):
        """Test model with learned position embeddings."""
        config = ModelConfig(
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_embd=128,
            block_size=32,
            position_embedding_type="learned",
        )
        model = GPT(config)

        assert isinstance(model.position_embedding, LearnedPositionEmbedding)

    def test_rope_position_embedding(self):
        """Test model with RoPE position embeddings."""
        config = ModelConfig(
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_embd=128,
            block_size=32,
            position_embedding_type="rope",
        )
        model = GPT(config)

        assert isinstance(model.position_embedding, RoPEPositionEmbedding)

    def test_custom_position_embedding(self):
        """Test model with custom position embedding."""
        config = ModelConfig(
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_embd=128,
            block_size=32,
        )
        custom_embedding = RoPEPositionEmbedding(32, 128, 4)
        model = GPT(config, position_embedding=custom_embedding)

        assert model.position_embedding is custom_embedding

    def test_weight_tying(self):
        """Test that input and output embeddings are tied."""
        config = ModelConfig(
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_embd=128,
            block_size=32,
            tie_word_embeddings=True,
        )
        model = GPT(config)

        # Check that weights are shared
        assert model.transformer.wte.weight is model.lm_head.weight

    def test_no_weight_tying(self):
        """Test model without weight tying."""
        config = ModelConfig(
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_embd=128,
            block_size=32,
            tie_word_embeddings=False,
        )
        model = GPT(config)

        # Check that weights are not shared
        assert model.transformer.wte.weight is not model.lm_head.weight

    def test_configure_optimizers(self, small_model):
        """Test optimizer configuration."""
        optimizer = small_model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=3e-4,
            device_type="cpu",
            verbose=False,
        )

        assert optimizer is not None
        assert len(optimizer.param_groups) == 2  # decay and no-decay groups

    def test_sequence_length_assertion(self, small_model):
        """Test that forward raises error for sequences longer than block_size."""
        # Create input longer than block_size
        x = torch.randint(0, 100, (2, small_model.config.block_size + 1))

        with pytest.raises(AssertionError):
            small_model(x)

    def test_backward_pass(self, small_model, dummy_batch):
        """Test that backward pass works."""
        x, y = dummy_batch
        logits, loss = small_model(x, y)

        # Backward pass
        loss.backward()

        # Check that gradients exist
        for param in small_model.parameters():
            if param.requires_grad:
                assert param.grad is not None
