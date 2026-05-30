#!/usr/bin/env python3
"""
Generate text using a trained model checkpoint.

Usage:
    python scripts/generate.py --checkpoint log/default/model_19072.pt --prompt "Hello, I'm a language model"
"""
import argparse
import torch

from llmkit import GPT, ModelConfig, generate


def main():
    parser = argparse.ArgumentParser(description='Generate text from a trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (e.g., log/default/model_19072.pt)')
    parser.add_argument('--prompt', type=str, default="Hello, I'm a language model,",
                       help='Text prompt to start generation')
    parser.add_argument('--max-length', type=int, default=100,
                       help='Maximum length of generated text (in tokens)')
    parser.add_argument('--num-samples', type=int, default=3,
                       help='Number of samples to generate')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top-k for sampling (default: 50)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature (higher = more random)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    args = parser.parse_args()

    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Try to get config from checkpoint, or infer from model weights
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        # Config might be a dict, convert to ModelConfig
        if isinstance(config_dict, dict):
            config = ModelConfig(**config_dict)
        else:
            config = config_dict
    else:
        # Infer vocab_size from the checkpoint weights
        model_state = checkpoint.get('model', checkpoint)
        vocab_size = model_state['transformer.wte.weight'].shape[0]

        config = ModelConfig(
            block_size=1024,
            vocab_size=vocab_size,
            n_layer=12,
            n_head=12,
            n_embd=768
        )

    model = GPT(config)

    # Load state dict
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        if 'step' in checkpoint:
            print(f"Checkpoint from step: {checkpoint['step']}")
        if 'val_loss' in checkpoint:
            print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    else:
        # Checkpoint might be just the state dict
        model.load_state_dict(checkpoint)

    print(f"Model config: vocab_size={config.vocab_size}, n_layer={config.n_layer}, n_embd={config.n_embd}")

    model.to(args.device)
    model.eval()

    print(f"\nPrompt: {args.prompt}")
    print(f"Generating {args.num_samples} sample(s)...\n")
    print("=" * 80)

    # Generate
    results = generate(
        model=model,
        prompt=args.prompt,
        max_length=args.max_length,
        num_samples=args.num_samples,
        top_k=args.top_k,
        temperature=args.temperature,
        device=args.device,
        seed=args.seed
    )

    # Print results
    for i, text in enumerate(results, 1):
        print(f"\n[Sample {i}]")
        print(text)
        print("-" * 80)


if __name__ == '__main__':
    main()
