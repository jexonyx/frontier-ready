"""Feed-forward network components."""
import torch
import torch.nn as nn
from ..config.component_config import FeedForwardConfig
from .initialization import ScaledLinear


def get_activation(name: str, **kwargs):
    """Get activation function by name.

    Args:
        name: Activation function name (e.g., "gelu", "relu", "silu")
        **kwargs: Additional arguments for the activation function

    Returns:
        Activation function module

    Raises:
        ValueError: If activation name is not recognized
    """
    activations = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "swish": nn.SiLU,  # Swish is same as SiLU
    }

    if name.lower() not in activations:
        raise ValueError(
            f"Unknown activation: {name}. "
            f"Available activations: {', '.join(activations.keys())}"
        )

    return activations[name.lower()](**kwargs)


class MLP(nn.Module):
    """Multi-layer perceptron (feed-forward network).

    A two-layer feed-forward network with configurable expansion factor
    and activation function. Used in transformer blocks.
    """

    def __init__(self, config: FeedForwardConfig, layer_idx: int = 0, n_layer: int = 1):
        """Initialize MLP.

        Args:
            config: Feed-forward configuration
            layer_idx: Index of this layer (for residual scaling)
            n_layer: Total number of layers (for residual scaling)
        """
        super().__init__()

        # Compute hidden dimension based on expansion factor
        hidden_dim = config.expansion_factor * config.n_embd

        # Input projection
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)

        # Activation function
        self.activation = get_activation(config.activation, **config.activation_kwargs)

        # Output projection with residual scaling
        scale_factor = (2 * n_layer) ** -0.5
        self.c_proj = ScaledLinear(
            hidden_dim,
            config.n_embd,
            scale_factor=scale_factor,
            bias=config.bias,
        )

        # Dropout (if needed)
        if config.dropout > 0:
            self.dropout = nn.Dropout(config.dropout)
        else:
            self.dropout = None

    def forward(self, x):
        """Apply feed-forward network.

        Args:
            x: Input tensor of shape (B, T, n_embd)

        Returns:
            Output tensor of shape (B, T, n_embd)
        """
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
