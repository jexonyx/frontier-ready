#!/usr/bin/env python3
"""
RoPE variant GPT-2 124M training experiment.

This script trains a GPT-2 124M model with Rotary Position Embeddings (RoPE)
instead of learned positional embeddings.
"""

from pathlib import Path
import sys
import yaml

# Ensure local repository root is on sys.path so imports from llmkit resolve.
repo_root = Path(__file__).resolve()
for parent in repo_root.parents:
    if (parent / "llmkit").exists():
        sys.path.insert(0, str(parent))
        break

from llmkit import GPT, ModelConfig, Trainer

def main():
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create model config using new modular API
    model_config = ModelConfig(**config["model"])

    # Use RoPE position embeddings instead of learned
    model_config.position_embedding_type = "rope"

    # Create model with RoPE
    model = GPT(model_config)

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
