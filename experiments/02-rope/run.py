#!/usr/bin/env python3
"""
RoPE variant GPT-2 124M training experiment.

This script trains a GPT-2 124M model with Rotary Position Embeddings (RoPE)
instead of learned positional embeddings.
"""

from pathlib import Path
import yaml
import torch

from nanogpt import GPTConfig, Trainer
from nanogpt.variants.rope import RoPEGPT

def main():
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create model config
    model_config = GPTConfig(**config["model"])

    # Create RoPE variant model
    model = RoPEGPT(model_config)

    # Create trainer
    trainer = Trainer(
        model=model,
        **config["training"],
        **config["data"],
        **config["output"],
        **config["evaluation"],
    )

    # Run training
    trainer.train()

if __name__ == "__main__":
    main()
