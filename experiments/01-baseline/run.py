#!/usr/bin/env python3
"""
Baseline GPT-2 124M training experiment.

This script trains a standard GPT-2 124M model from scratch on FineWeb-Edu data.
"""

from pathlib import Path
import yaml
import torch

from llmkit import GPT, ModelConfig, Trainer

def main():
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create model config using new modular API
    model_config = ModelConfig(**config["model"])

    # Create model
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
