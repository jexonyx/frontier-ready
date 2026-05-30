"""RoPE (Rotary Position Embeddings) variant.

Replaces learned positional embeddings (wpe) with rotary embeddings
applied to Q and K vectors in attention.

Usage example:
    from llmkit.variants.rope import create_rope_gpt
    from llmkit import ModelConfig

    config = ModelConfig()
    model = create_rope_gpt(config)

    # Or use directly:
    from llmkit import GPT, RoPEPositionEmbedding, ModelConfig
    config = ModelConfig(position_embedding_type="rope")
    model = GPT(config)
"""
from ..models import GPT
from ..config import ModelConfig
from ..core.embedding import RoPEPositionEmbedding


def create_rope_gpt(config: ModelConfig) -> GPT:
    """Create a GPT model with RoPE position embeddings.

    Args:
        config: Model configuration

    Returns:
        GPT model with RoPE position embeddings
    """
    # Set position embedding type to RoPE
    config.position_embedding_type = "rope"
    return GPT(config)
