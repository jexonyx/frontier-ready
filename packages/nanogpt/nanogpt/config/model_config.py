"""Model-level configuration for GPT architecture."""
from dataclasses import dataclass, field
from typing import Optional
from .component_config import AttentionConfig, FeedForwardConfig, InitializationConfig


@dataclass
class ModelConfig:
    """Complete GPT model configuration.

    This configuration uses composition to combine component-specific configs,
    allowing fine-grained control over each part of the model.

    Args:
        vocab_size: Size of vocabulary (default: 50257 for GPT-2)
        block_size: Maximum sequence length (default: 1024)
        n_layer: Number of transformer blocks (default: 12)
        n_head: Number of attention heads (default: 12)
        n_embd: Embedding dimension (default: 768)
        attention_config: Custom attention configuration (default: None)
        feedforward_config: Custom feedforward configuration (default: None)
        initialization_config: Custom initialization configuration (default: None)
        position_embedding_type: Type of position embedding (default: "learned")
            Options: "learned", "rope", "none"
        tie_word_embeddings: Whether to tie input and output embeddings (default: True)
    """
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

    # Component configurations (optional overrides)
    attention_config: Optional[AttentionConfig] = None
    feedforward_config: Optional[FeedForwardConfig] = None
    initialization_config: Optional[InitializationConfig] = None

    # Position embedding strategy
    position_embedding_type: str = "learned"

    # Weight tying
    tie_word_embeddings: bool = True

    def __post_init__(self):
        """Initialize component configs with defaults if not provided."""
        if self.attention_config is None:
            self.attention_config = AttentionConfig(
                n_embd=self.n_embd,
                n_head=self.n_head,
            )

        if self.feedforward_config is None:
            self.feedforward_config = FeedForwardConfig(
                n_embd=self.n_embd,
            )

        if self.initialization_config is None:
            self.initialization_config = InitializationConfig(
                residual_scale_factor=(2 * self.n_layer) ** -0.5
            )
