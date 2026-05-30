"""Tests for feed-forward network components."""
import pytest
import torch
from llmkit.config.component_config import FeedForwardConfig
from llmkit.core.feedforward import MLP, get_activation


class TestGetActivation:
    """Tests for get_activation function."""

    def test_gelu_activation(self):
        """Test GELU activation function."""
        activation = get_activation("gelu", approximate="tanh")
        assert activation is not None
        x = torch.randn(10, 20)
        y = activation(x)
        assert y.shape == x.shape

    def test_relu_activation(self):
        """Test ReLU activation function."""
        activation = get_activation("relu")
        x = torch.randn(10, 20)
        y = activation(x)
        assert y.shape == x.shape

    def test_silu_activation(self):
        """Test SiLU activation function."""
        activation = get_activation("silu")
        x = torch.randn(10, 20)
        y = activation(x)
        assert y.shape == x.shape

    def test_unknown_activation_raises(self):
        """Test that unknown activation raises error."""
        with pytest.raises(ValueError, match="Unknown activation"):
            get_activation("unknown_activation")


class TestMLP:
    """Tests for MLP component."""

    def test_mlp_expansion_factor(self):
        """Test MLP respects configurable expansion factor."""
        # Test 4x expansion (default)
        config_4x = FeedForwardConfig(n_embd=768, expansion_factor=4)
        mlp_4x = MLP(config_4x)
        assert mlp_4x.c_fc.out_features == 768 * 4

        # Test 8x expansion
        config_8x = FeedForwardConfig(n_embd=768, expansion_factor=8)
        mlp_8x = MLP(config_8x)
        assert mlp_8x.c_fc.out_features == 768 * 8

    def test_mlp_activation(self):
        """Test MLP uses different activations correctly."""
        config_gelu = FeedForwardConfig(
            n_embd=128,
            activation="gelu",
            activation_kwargs={"approximate": "tanh"}
        )
        mlp_gelu = MLP(config_gelu)

        config_relu = FeedForwardConfig(
            n_embd=128,
            activation="relu",
            activation_kwargs={}
        )
        mlp_relu = MLP(config_relu)

        # Test that different activations produce different outputs
        x = torch.randn(2, 16, 128)
        out_gelu = mlp_gelu(x)
        out_relu = mlp_relu(x)

        assert out_gelu.shape == out_relu.shape
        # Outputs should be different (with high probability)
        assert not torch.allclose(out_gelu, out_relu)

    def test_mlp_forward_shape(self):
        """Test MLP forward pass preserves shape."""
        config = FeedForwardConfig(n_embd=128, expansion_factor=4)
        mlp = MLP(config)

        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, 128)
        y = mlp(x)

        assert y.shape == (batch_size, seq_len, 128)

    def test_mlp_dropout(self):
        """Test MLP with dropout."""
        config = FeedForwardConfig(n_embd=128, dropout=0.5)
        mlp = MLP(config)
        assert mlp.dropout is not None

        # In training mode, dropout should be active
        mlp.train()
        x = torch.randn(2, 16, 128)
        y1 = mlp(x)
        y2 = mlp(x)
        # With 50% dropout, outputs should differ
        assert not torch.allclose(y1, y2)

        # In eval mode, dropout should be inactive
        mlp.eval()
        y3 = mlp(x)
        y4 = mlp(x)
        assert torch.allclose(y3, y4)

    def test_mlp_no_dropout(self):
        """Test MLP without dropout."""
        config = FeedForwardConfig(n_embd=128, dropout=0.0)
        mlp = MLP(config)
        assert mlp.dropout is None

        x = torch.randn(2, 16, 128)
        y = mlp(x)
        assert y.shape == x.shape

    def test_mlp_bias(self):
        """Test MLP with and without bias."""
        config_with_bias = FeedForwardConfig(n_embd=128, bias=True)
        mlp_with = MLP(config_with_bias)
        assert mlp_with.c_fc.bias is not None
        assert mlp_with.c_proj.bias is not None

        config_no_bias = FeedForwardConfig(n_embd=128, bias=False)
        mlp_no = MLP(config_no_bias)
        assert mlp_no.c_fc.bias is None
        assert mlp_no.c_proj.bias is None
