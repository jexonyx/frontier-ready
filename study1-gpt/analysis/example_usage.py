#!/usr/bin/env python3
"""
Example usage of the analysis scripts.

This script demonstrates how to:
1. Load and parse training metrics
2. Extract key statistics
3. Generate plots programmatically
4. Create summary tables

Usage:
    uv run python analysis/example_usage.py --log-dir log/baseline
"""

import argparse
from pathlib import Path

from parse_metrics import (
    load_metrics,
    get_final_metrics,
    get_convergence_stats,
    get_training_efficiency_stats,
    get_layer_analysis_stats,
)


def main():
    parser = argparse.ArgumentParser(description="Example usage of analysis utilities")
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Path to log directory (e.g., log/baseline)"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Analysis Example: Loading and Analyzing Training Metrics")
    print("=" * 80)

    # 1. Load metrics
    print(f"\n1. Loading metrics from {args.log_dir}...")
    df = load_metrics(args.log_dir)
    print(f"   ✓ Loaded {len(df)} training steps")
    print(f"   ✓ Columns: {len(df.columns)} total")
    print(f"   ✓ Step range: {df['step'].min()} to {df['step'].max()}")

    # 2. Get final metrics
    print("\n2. Final Metrics:")
    final = get_final_metrics(df)
    print(f"   • Final step: {final['final_step']:,}")
    print(f"   • Final train loss: {final['final_train_loss']:.4f}")
    print(f"   • Final val loss: {final['final_val_loss']:.4f}")
    print(f"   • Final HellaSwag accuracy: {final['final_hella_acc']:.4f} ({final['final_hella_acc']*100:.2f}%)")
    print(f"   • Min train loss: {final['min_train_loss']:.4f}")
    print(f"   • Min val loss: {final['min_val_loss']:.4f}")
    print(f"   • Max HellaSwag accuracy: {final['max_hella_acc']:.4f} ({final['max_hella_acc']*100:.2f}%)")
    print(f"   • Total tokens: {final['total_tokens']:,} (~{final['total_tokens']/1e9:.1f}B)")

    # 3. Convergence statistics
    print("\n3. Convergence Statistics:")
    conv = get_convergence_stats(df)
    if conv['steps_to_val_loss_3.5']:
        print(f"   • Steps to val_loss < 3.5: {conv['steps_to_val_loss_3.5']:,}")
    if conv['steps_to_val_loss_3.3']:
        print(f"   • Steps to val_loss < 3.3: {conv['steps_to_val_loss_3.3']:,}")
    print(f"   • Plateau detected: {conv['plateau_detected']}")
    if conv['plateau_step']:
        print(f"   • Plateau step: {conv['plateau_step']:,}")

    # 4. Training efficiency
    print("\n4. Training Efficiency:")
    eff = get_training_efficiency_stats(df)
    if eff['total_time_hours']:
        print(f"   • Total training time: {eff['total_time_hours']:.2f} hours")
    if eff['mean_step_time_ms']:
        print(f"   • Mean step time: {eff['mean_step_time_ms']:.0f} ms")
    if eff['mean_throughput']:
        print(f"   • Mean throughput: {eff['mean_throughput']:.0f} tokens/sec")
        print(f"   • Std throughput: {eff['std_throughput']:.0f} tokens/sec")

    # 5. Layer analysis
    print("\n5. Layer Analysis:")
    layer = get_layer_analysis_stats(df)
    if layer['highest_grad_norm_layer'] is not None:
        print(f"   • Highest gradient norm: Layer {layer['highest_grad_norm_layer']} (mean: {layer['highest_grad_norm_mean']:.4f})")
    if layer['lowest_grad_norm_layer'] is not None:
        print(f"   • Lowest gradient norm: Layer {layer['lowest_grad_norm_layer']} (mean: {layer['lowest_grad_norm_mean']:.4f})")
    if layer['update_ratio_min'] is not None:
        print(f"   • Update ratio range: {layer['update_ratio_min']:.5f} - {layer['update_ratio_max']:.5f}")

    # 6. Generate plots
    print("\n6. To generate plots, run:")
    print(f"   uv run python analysis/visualize_run.py --log-dir {args.log_dir} --output-dir analysis/figures")

    # 7. Generate summary
    print("\n7. To generate summary table, run:")
    print(f"   uv run python analysis/summarize_run.py --log-dir {args.log_dir} --output analysis/summary.md")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
