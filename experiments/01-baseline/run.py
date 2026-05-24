#!/usr/bin/env python3
"""
Baseline GPT-2 124M training experiment.

This script trains a standard GPT-2 124M model from scratch on FineWeb-Edu data.
"""

from pathlib import Path
import yaml
import torch

from nanogpt import GPT, GPTConfig, Trainer

def main():
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create model config
    model_config = GPTConfig(
        n_layer=config["model"]["n_layer"],
        n_head=config["model"]["n_head"],
        n_embd=config["model"]["n_embd"],
        block_size=config["model"]["block_size"],
        vocab_size=config["model"]["vocab_size"],
    )

    # Create model
    model = GPT(model_config)

    # Create trainer
    trainer = Trainer(
        model=model,
        max_steps=config["training"]["max_steps"],
        batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        warmup_steps=config["training"]["warmup_steps"],
        weight_decay=config["training"]["weight_decay"],
        train_data_dir=config["data"]["train_data_dir"],
        val_data_dir=config["data"]["val_data_dir"],
        log_dir=config["output"]["log_dir"],
        eval_interval=config["evaluation"]["eval_interval"],
        checkpoint_interval=config["output"]["checkpoint_interval"],
    )

    # Run training
    trainer.train()

if __name__ == "__main__":
    main()
