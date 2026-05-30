"""Pytest fixtures for nanogpt tests."""
import pytest
import torch
from nanogpt import ModelConfig, GPT


@pytest.fixture
def small_config():
    """Small configuration for fast testing."""
    return ModelConfig(
        vocab_size=100,
        n_layer=2,
        n_head=4,
        n_embd=128,
        block_size=32,
    )


@pytest.fixture
def small_model(small_config):
    """Small model for fast testing."""
    return GPT(small_config)


@pytest.fixture
def dummy_batch():
    """Dummy batch of data for testing."""
    batch_size = 2
    seq_len = 16
    vocab_size = 100
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    y = torch.randint(0, vocab_size, (batch_size, seq_len))
    return x, y


@pytest.fixture
def device():
    """Device for testing."""
    return torch.device("cpu")
