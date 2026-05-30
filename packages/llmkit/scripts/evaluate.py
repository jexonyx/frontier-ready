#!/usr/bin/env python3
"""
Evaluate a trained GPT model on HellaSwag.

Usage:
    python scripts/evaluate.py --checkpoint log/default/model_19072.pt
"""
import argparse
import torch

from llmkit import GPT, ModelConfig, evaluate_hellaswag


def main():
    parser = argparse.ArgumentParser(description='Evaluate a GPT model on HellaSwag')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--data-cache-dir', type=str, default='hellaswag',
                       help='Directory to cache HellaSwag data')
    args = parser.parse_args()

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Get config from checkpoint
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        if isinstance(config_dict, dict):
            config = ModelConfig(**config_dict)
        else:
            config = config_dict
    else:
        # Infer from model weights
        model_state = checkpoint.get('model', checkpoint)
        vocab_size = model_state['transformer.wte.weight'].shape[0]
        config = ModelConfig(
            block_size=1024,
            vocab_size=vocab_size,
            n_layer=12,
            n_head=12,
            n_embd=768
        )

    # Create and load model
    model = GPT(config)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        if 'step' in checkpoint:
            print(f"Checkpoint from step: {checkpoint['step']}")
        if 'val_loss' in checkpoint:
            print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    else:
        model.load_state_dict(checkpoint)

    print(f"Model config: vocab_size={config.vocab_size}, n_layer={config.n_layer}, n_embd={config.n_embd}")

    model.to(args.device)
    model.eval()

    print("\nEvaluating on HellaSwag validation set...")
    results = evaluate_hellaswag(model, args.device, args.data_cache_dir)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Accuracy: {results['accuracy']:.4f} ({results['num_correct']}/{results['num_total']})")
    print(f"Accuracy (normalized): {results['accuracy_norm']:.4f} ({results['num_correct_norm']}/{results['num_total']})")


if __name__ == '__main__':
    main()
