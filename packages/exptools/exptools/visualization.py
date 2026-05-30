#!/usr/bin/env python3
"""
Generate publication-ready plots for a single training run.

Usage:
    python analysis/visualize_run.py --log-dir log/baseline --output-dir analysis/figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .metrics import load_metrics

# OpenAI GPT-2 and GPT-3 baselines
BASELINES = {
    "val_loss": {
        "124M": 3.2924,
    },
    "hellaswag_gpt2": {
        "124M": 0.294463,
        "350M": 0.375224,
        "774M": 0.431986,
        "1558M": 0.488946,
    },
    "hellaswag_gpt3": {
        "124M": 0.337,
        "350M": 0.436,
        "774M": 0.510,
        "1558M": 0.547,
    }
}


def plot_loss_curves(df, output_path: Path, model_size: str = "124M"):
    """
    Generate Figure 1: Loss curves (train and validation).

    Args:
        df: DataFrame from load_metrics()
        output_path: Path to save the plot
        model_size: Model size for baseline comparison (default: "124M")
    """
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Training loss
    ax1.plot(df['step'], df['train_loss'], label='llmkit train loss', linewidth=1.5)
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_ylim(top=4.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')

    # Panel 2: Validation loss
    val_loss_data = df[['step', 'val_loss']].dropna()
    if not val_loss_data.empty:
        ax2.plot(val_loss_data['step'], val_loss_data['val_loss'],
                label='llmkit val loss', linewidth=1.5, marker='o', markersize=3)

        # Add OpenAI baseline
        if model_size in BASELINES["val_loss"]:
            baseline = BASELINES["val_loss"][model_size]
            ax2.axhline(y=baseline, color='r', linestyle='--', linewidth=2,
                       label=f'OpenAI GPT-2 ({model_size})')

    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved loss curves to {output_path}")


def plot_hellaswag_accuracy(df, output_path: Path, model_size: str = "124M"):
    """
    Generate Figure 2: HellaSwag accuracy.

    Args:
        df: DataFrame from load_metrics()
        output_path: Path to save the plot
        model_size: Model size for baseline comparison (default: "124M")
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    hella_data = df[['step', 'hella_acc']].dropna()
    if not hella_data.empty:
        ax.plot(hella_data['step'], hella_data['hella_acc'],
               label='llmkit', linewidth=1.5, marker='o', markersize=4)

        # Add OpenAI GPT-2 baseline
        if model_size in BASELINES["hellaswag_gpt2"]:
            baseline_gpt2 = BASELINES["hellaswag_gpt2"][model_size]
            ax.axhline(y=baseline_gpt2, color='r', linestyle='--', linewidth=2,
                      label=f'OpenAI GPT-2 ({model_size})')

        # Add OpenAI GPT-3 baseline
        if model_size in BASELINES["hellaswag_gpt3"]:
            baseline_gpt3 = BASELINES["hellaswag_gpt3"][model_size]
            ax.axhline(y=baseline_gpt3, color='g', linestyle='--', linewidth=2,
                      label=f'OpenAI GPT-3 ({model_size})')

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(0.0, 0.6)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_title('HellaSwag Accuracy', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved HellaSwag accuracy to {output_path}")


def plot_training_dynamics(df, output_path: Path):
    """
    Generate Figure 3: Training dynamics (lr, grad norm, throughput, GPU memory).

    Args:
        df: DataFrame from load_metrics()
        output_path: Path to save the plot
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Learning rate schedule
    if 'lr' in df.columns:
        axes[0, 0].plot(df['step'], df['lr'], linewidth=1.5, color='tab:blue')
        axes[0, 0].set_xlabel('Training Steps', fontsize=10)
        axes[0, 0].set_ylabel('Learning Rate', fontsize=10)
        axes[0, 0].set_title('Learning Rate Schedule (Cosine Decay)', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

    # Top-right: Gradient norm
    if 'grad_norm' in df.columns:
        axes[0, 1].plot(df['step'], df['grad_norm'], linewidth=1.5, color='tab:orange', alpha=0.7)
        axes[0, 1].axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Clip threshold')
        axes[0, 1].set_xlabel('Training Steps', fontsize=10)
        axes[0, 1].set_ylabel('Gradient Norm', fontsize=10)
        axes[0, 1].set_title('Gradient Norm (clipped at 1.0)', fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: Throughput
    if 'tok_sec' in df.columns:
        axes[1, 0].plot(df['step'], df['tok_sec'], linewidth=1.5, color='tab:green', alpha=0.7)
        axes[1, 0].set_xlabel('Training Steps', fontsize=10)
        axes[1, 0].set_ylabel('Tokens/Second', fontsize=10)
        axes[1, 0].set_title('Training Throughput', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: GPU memory
    if 'gpu_mem_gb' in df.columns:
        axes[1, 1].plot(df['step'], df['gpu_mem_gb'], linewidth=1.5, color='tab:purple', alpha=0.7)
        axes[1, 1].set_xlabel('Training Steps', fontsize=10)
        axes[1, 1].set_ylabel('GPU Memory (GB)', fontsize=10)
        axes[1, 1].set_title('GPU Memory Usage', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training dynamics to {output_path}")


def plot_layer_analysis(df, output_path: Path, num_layers: int = 12):
    """
    Generate Figure 4: Layer analysis (per-layer metrics).

    Args:
        df: DataFrame from load_metrics()
        output_path: Path to save the plot
        num_layers: Number of transformer layers (default: 12)
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Use a color palette to distinguish layers
    colors = plt.cm.viridis(np.linspace(0, 1, num_layers))

    # Top-left: Layer gradient norms
    grad_norm_cols = [f'layer_grad_norms_layer{i}' for i in range(num_layers)]
    available_grad_cols = [col for col in grad_norm_cols if col in df.columns]

    if available_grad_cols:
        for i, col in enumerate(available_grad_cols):
            layer_num = int(col.split('layer')[-1])
            axes[0, 0].plot(df['step'], df[col], linewidth=1.2, alpha=0.7,
                          color=colors[layer_num], label=f'Layer {layer_num}')

        axes[0, 0].set_xlabel('Training Steps', fontsize=10)
        axes[0, 0].set_ylabel('Gradient Norm', fontsize=10)
        axes[0, 0].set_title('Layer Gradient Norms', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        # Only show legend if not too many layers
        if num_layers <= 12:
            axes[0, 0].legend(fontsize=7, ncol=2)

    # Top-right: Layer weight norms
    weight_norm_cols = [f'layer_weight_norms_layer{i}' for i in range(num_layers)]
    available_weight_cols = [col for col in weight_norm_cols if col in df.columns]

    if available_weight_cols:
        for i, col in enumerate(available_weight_cols):
            layer_num = int(col.split('layer')[-1])
            axes[0, 1].plot(df['step'], df[col], linewidth=1.2, alpha=0.7,
                          color=colors[layer_num], label=f'Layer {layer_num}')

        axes[0, 1].set_xlabel('Training Steps', fontsize=10)
        axes[0, 1].set_ylabel('Weight Norm', fontsize=10)
        axes[0, 1].set_title('Layer Weight Norms', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: Layer update ratios
    update_ratio_cols = [f'layer_update_ratios_layer{i}' for i in range(num_layers)]
    available_update_cols = [col for col in update_ratio_cols if col in df.columns]

    if available_update_cols:
        for i, col in enumerate(available_update_cols):
            layer_num = int(col.split('layer')[-1])
            axes[1, 0].plot(df['step'], df[col], linewidth=1.2, alpha=0.7,
                          color=colors[layer_num], label=f'Layer {layer_num}')

        axes[1, 0].set_xlabel('Training Steps', fontsize=10)
        axes[1, 0].set_ylabel('Update Ratio', fontsize=10)
        axes[1, 0].set_title('Layer Update Ratios', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No update ratio data available',
                       ha='center', va='center', fontsize=12)
        axes[1, 0].set_title('Layer Update Ratios', fontsize=12, fontweight='bold')

    # Bottom-right: Layer activation norms
    act_norm_cols = [f'layer_act_norms_layer{i}' for i in range(num_layers)]
    available_act_cols = [col for col in act_norm_cols if col in df.columns]

    if available_act_cols:
        for i, col in enumerate(available_act_cols):
            layer_num = int(col.split('layer')[-1])
            axes[1, 1].plot(df['step'], df[col], linewidth=1.2, alpha=0.7,
                          color=colors[layer_num], label=f'Layer {layer_num}')

        axes[1, 1].set_xlabel('Training Steps', fontsize=10)
        axes[1, 1].set_ylabel('Activation Norm', fontsize=10)
        axes[1, 1].set_title('Layer Activation Norms', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No activation norm data available',
                       ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Layer Activation Norms', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved layer analysis to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready plots for a single training run"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Path to log directory (e.g., log/baseline)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save plots (e.g., analysis/figures)"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="124M",
        choices=["124M", "350M", "774M", "1558M"],
        help="Model size for baseline comparison (default: 124M)"
    )

    args = parser.parse_args()

    # Load metrics
    print(f"Loading metrics from {args.log_dir}...")
    df = load_metrics(args.log_dir)
    print(f"Loaded {len(df)} training steps")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\nGenerating plots...")
    plot_loss_curves(df, output_dir / "loss_curves.png", args.model_size)
    plot_hellaswag_accuracy(df, output_dir / "hellaswag_accuracy.png", args.model_size)
    plot_training_dynamics(df, output_dir / "training_dynamics.png")
    plot_layer_analysis(df, output_dir / "layer_analysis.png")

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
