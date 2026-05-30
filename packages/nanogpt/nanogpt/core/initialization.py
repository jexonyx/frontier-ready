"""Weight initialization utilities."""
import torch.nn as nn


class ScaledLinear(nn.Linear):
    """Linear layer with residual scaling initialization.

    Used for output projections in attention and MLP blocks to implement
    the residual layer scaling from the GPT-2 paper. Weights are initialized
    with std scaled by the provided scale_factor to prevent gradient explosion.

    Args:
        in_features: Size of input features
        out_features: Size of output features
        scale_factor: Factor to scale initialization std (default: 1.0)
        bias: Whether to include bias (default: True)
        init_std: Base standard deviation for initialization (default: 0.02)
    """

    def __init__(self, in_features, out_features, scale_factor=1.0, bias=True, init_std=0.02):
        self.scale_factor = scale_factor
        self.init_std = init_std
        super().__init__(in_features, out_features, bias=bias)

    def reset_parameters(self):
        """Initialize parameters with scaled std."""
        std = self.init_std * self.scale_factor
        nn.init.normal_(self.weight, mean=0.0, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
