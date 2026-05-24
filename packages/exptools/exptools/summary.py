#!/usr/bin/env python3
"""
Generate markdown-formatted summary table for a training run.

Usage:
    python analysis/summarize_run.py --log-dir log/baseline --output analysis/summary.md
"""

import argparse
from pathlib import Path

from .metrics import (
    load_metrics,
    get_final_metrics,
    get_convergence_stats,
    get_training_efficiency_stats,
    get_layer_analysis_stats,
)

# OpenAI GPT-2 baselines
BASELINES = {
    "124M": {
        "val_loss": 3.2924,
        "hella_acc": 0.294463,
    },
    "350M": {
        "val_loss": None,  # Not documented
        "hella_acc": 0.375224,
    },
    "774M": {
        "val_loss": None,
        "hella_acc": 0.431986,
    },
    "1558M": {
        "val_loss": None,
        "hella_acc": 0.488946,
    },
}


def format_number(value, decimals=3):
    """Format a number with commas and specified decimal places."""
    if value is None:
        return "N/A"
    if isinstance(value, int) or value == int(value):
        return f"{int(value):,}"
    return f"{value:,.{decimals}f}"


def format_tokens(tokens):
    """Format token count in human-readable form (e.g., ~10B)."""
    if tokens is None:
        return "N/A"

    billions = tokens / 1e9
    if billions >= 1:
        return f"~{billions:.1f}B"

    millions = tokens / 1e6
    return f"~{millions:.0f}M"


def generate_summary(log_dir: str, model_size: str = "124M", tokens_per_step: int = 524288) -> str:
    """
    Generate markdown summary for a training run.

    Args:
        log_dir: Path to log directory
        model_size: Model size (default: "124M")
        tokens_per_step: Tokens per training step (default: 524288)

    Returns:
        Markdown-formatted summary string
    """
    # Load data
    df = load_metrics(log_dir)
    final_metrics = get_final_metrics(df, tokens_per_step)
    convergence_stats = get_convergence_stats(df, tokens_per_step)
    efficiency_stats = get_training_efficiency_stats(df)
    layer_stats = get_layer_analysis_stats(df)

    # Get run name from log_dir
    run_name = Path(log_dir).name

    # Get baselines
    baseline_val_loss = BASELINES.get(model_size, {}).get("val_loss")
    baseline_hella_acc = BASELINES.get(model_size, {}).get("hella_acc")

    # Build markdown
    lines = []
    lines.append("# Training Run Summary")
    lines.append("")
    lines.append(f"**Run name:** {run_name}")
    lines.append(f"**Model:** GPT-2 {model_size}")
    lines.append(f"**Training steps:** {format_number(final_metrics['final_step'], 0)}")
    lines.append(f"**Total tokens:** {format_tokens(final_metrics['total_tokens'])}")
    lines.append("")

    # Final Metrics table
    lines.append("## Final Metrics")
    lines.append("")
    lines.append("| Metric | Value | OpenAI GPT-2 Baseline | Delta |")
    lines.append("|--------|-------|----------------------|-------|")

    # Training loss
    lines.append(f"| Training Loss | {format_number(final_metrics['final_train_loss'])} | - | - |")

    # Validation loss
    val_loss_str = format_number(final_metrics['final_val_loss'])
    if baseline_val_loss is not None and final_metrics['final_val_loss'] is not None:
        delta = final_metrics['final_val_loss'] - baseline_val_loss
        delta_str = f"{delta:+.3f}"
        baseline_str = format_number(baseline_val_loss)
    else:
        delta_str = "-"
        baseline_str = format_number(baseline_val_loss) if baseline_val_loss else "-"
    lines.append(f"| Validation Loss | {val_loss_str} | {baseline_str} | {delta_str} |")

    # HellaSwag accuracy
    if final_metrics['final_hella_acc'] is not None:
        hella_pct = final_metrics['final_hella_acc'] * 100
        hella_str = f"{hella_pct:.2f}%"
        if baseline_hella_acc is not None:
            baseline_pct = baseline_hella_acc * 100
            delta_pct = hella_pct - baseline_pct
            delta_str = f"{delta_pct:+.2f}%"
            baseline_str = f"{baseline_pct:.2f}%"
        else:
            delta_str = "-"
            baseline_str = "-"
    else:
        hella_str = "N/A"
        baseline_str = "-"
        delta_str = "-"
    lines.append(f"| HellaSwag Accuracy | {hella_str} | {baseline_str} | {delta_str} |")

    # Throughput
    if efficiency_stats['mean_throughput'] is not None:
        throughput_str = format_number(efficiency_stats['mean_throughput'], 0)
    else:
        throughput_str = "N/A"
    lines.append(f"| Tokens/sec (mean) | {throughput_str} | - | - |")

    # GPU Memory
    if 'gpu_mem_gb' in df.columns:
        peak_mem = df['gpu_mem_gb'].max()
        mem_str = f"{peak_mem:.1f} GB"
    else:
        mem_str = "N/A"
    lines.append(f"| GPU Memory (peak) | {mem_str} | - | - |")
    lines.append("")

    # Convergence Statistics
    lines.append("## Convergence Statistics")
    lines.append("")
    lines.append("| Milestone | Step | Tokens |")
    lines.append("|-----------|------|--------|")

    if convergence_stats['steps_to_val_loss_3.5'] is not None:
        step = convergence_stats['steps_to_val_loss_3.5']
        tokens = step * tokens_per_step
        lines.append(f"| Val loss < 3.5 | {format_number(step, 0)} | {format_tokens(tokens)} |")

    if convergence_stats['steps_to_val_loss_3.3'] is not None:
        step = convergence_stats['steps_to_val_loss_3.3']
        tokens = step * tokens_per_step
        lines.append(f"| Val loss < 3.3 | {format_number(step, 0)} | {format_tokens(tokens)} |")

    if final_metrics['max_hella_acc'] is not None:
        # Find step where max HellaSwag was achieved
        hella_data = df[['step', 'hella_acc']].dropna()
        if not hella_data.empty:
            max_idx = hella_data['hella_acc'].idxmax()
            max_step = int(hella_data.loc[max_idx, 'step'])
            max_tokens = max_step * tokens_per_step
            lines.append(f"| Max HellaSwag | {format_number(max_step, 0)} | {format_tokens(max_tokens)} |")

    if convergence_stats['plateau_detected'] and convergence_stats['plateau_step'] is not None:
        step = convergence_stats['plateau_step']
        tokens = step * tokens_per_step
        lines.append(f"| Plateau detected | {format_number(step, 0)} | {format_tokens(tokens)} |")

    lines.append("")

    # Training Efficiency
    lines.append("## Training Efficiency")
    lines.append("")

    if efficiency_stats['total_time_hours'] is not None:
        hours = efficiency_stats['total_time_hours']
        lines.append(f"- **Total training time:** {hours:.1f} hours")

    if efficiency_stats['mean_step_time_ms'] is not None:
        step_time = efficiency_stats['mean_step_time_ms']
        lines.append(f"- **Mean step time:** {step_time:,.0f} ms")

    if efficiency_stats['mean_throughput'] is not None:
        mean_tput = efficiency_stats['mean_throughput']
        lines.append(f"- **Throughput (mean):** {mean_tput:,.0f} tokens/sec")

    if efficiency_stats['std_throughput'] is not None:
        std_tput = efficiency_stats['std_throughput']
        lines.append(f"- **Throughput (std):** {std_tput:,.0f} tokens/sec")

    lines.append("")

    # Layer Analysis
    lines.append("## Layer Analysis")
    lines.append("")

    if layer_stats['highest_grad_norm_layer'] is not None:
        layer = layer_stats['highest_grad_norm_layer']
        mean_norm = layer_stats['highest_grad_norm_mean']
        lines.append(f"- **Layer with highest gradient norm:** Layer {layer} (mean: {mean_norm:.3f})")

    if layer_stats['lowest_grad_norm_layer'] is not None:
        layer = layer_stats['lowest_grad_norm_layer']
        mean_norm = layer_stats['lowest_grad_norm_mean']
        lines.append(f"- **Layer with lowest gradient norm:** Layer {layer} (mean: {mean_norm:.3f})")

    if layer_stats['update_ratio_min'] is not None and layer_stats['update_ratio_max'] is not None:
        min_ratio = layer_stats['update_ratio_min']
        max_ratio = layer_stats['update_ratio_max']
        lines.append(f"- **Layer update ratio range:** {min_ratio:.5f} - {max_ratio:.5f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate markdown summary for a training run"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Path to log directory (e.g., log/baseline)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save markdown summary (e.g., analysis/summary.md)"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="124M",
        choices=["124M", "350M", "774M", "1558M"],
        help="Model size for baseline comparison (default: 124M)"
    )
    parser.add_argument(
        "--tokens-per-step",
        type=int,
        default=524288,
        help="Tokens per training step (default: 524288)"
    )

    args = parser.parse_args()

    # Generate summary
    print(f"Loading metrics from {args.log_dir}...")
    summary = generate_summary(args.log_dir, args.model_size, args.tokens_per_step)

    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(summary)

    print(f"\nSummary saved to {output_path}")
    print("\n" + "=" * 80)
    print(summary)
    print("=" * 80)


if __name__ == "__main__":
    main()
