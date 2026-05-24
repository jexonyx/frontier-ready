"""
Data parsing utilities for training metrics.

This module provides functions to load and process metrics.jsonl files
generated during GPT-2 training runs.
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


def load_metrics(log_dir: str) -> pd.DataFrame:
    """
    Load metrics.jsonl from log_dir and return as pandas DataFrame.

    Args:
        log_dir: Path to log directory (e.g., "log/baseline")

    Returns:
        DataFrame with columns: step, train_loss, val_loss, hella_acc, lr,
        grad_norm, tok_sec, gpu_mem_gb, dt_ms, plus flattened layer metrics

    Notes:
        - Handles missing columns (val_loss, hella_acc only exist every 250 steps)
        - Flattens list-valued metrics (layer_grad_norms, etc.) into separate columns
        - Validates data integrity (monotonic steps, no duplicates)
    """
    metrics_file = Path(log_dir) / "metrics.jsonl"

    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    # Load all metrics into a list
    records = []
    with open(metrics_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                continue

    if not records:
        raise ValueError(f"No valid records found in {metrics_file}")

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Validate step numbers
    if 'step' not in df.columns:
        raise ValueError("Missing required 'step' column in metrics")

    # Check for monotonic increasing steps
    if not df['step'].is_monotonic_increasing:
        print("Warning: Steps are not monotonic increasing")

    # Check for duplicates
    duplicates = df['step'].duplicated()
    if duplicates.any():
        print(f"Warning: Found {duplicates.sum()} duplicate steps")
        df = df.drop_duplicates(subset='step', keep='last')

    # Sort by step to ensure order
    df = df.sort_values('step').reset_index(drop=True)

    # Flatten layer metrics
    layer_metrics = ['layer_grad_norms', 'layer_weight_norms', 'layer_update_ratios', 'layer_act_norms']
    for metric in layer_metrics:
        if metric in df.columns:
            # Extract list values and create separate columns
            layer_data = df[metric].apply(lambda x: x if isinstance(x, list) else [])
            max_layers = max(len(x) for x in layer_data) if len(layer_data) > 0 else 0

            for i in range(max_layers):
                df[f'{metric}_layer{i}'] = layer_data.apply(lambda x: x[i] if i < len(x) else np.nan)

    # Forward-fill missing val_loss and hella_acc for plotting continuity
    if 'val_loss' in df.columns:
        df['val_loss_filled'] = df['val_loss'].ffill()
    if 'hella_acc' in df.columns:
        df['hella_acc_filled'] = df['hella_acc'].ffill()

    return df


def get_final_metrics(df: pd.DataFrame, tokens_per_step: int = 524288) -> dict:
    """
    Extract final metrics from the last row(s) of training.

    Args:
        df: DataFrame from load_metrics()
        tokens_per_step: Number of tokens processed per step (default: 524288)

    Returns:
        {
            'final_step': int,
            'final_train_loss': float,
            'final_val_loss': float,
            'final_hella_acc': float,
            'min_train_loss': float,
            'min_val_loss': float,
            'max_hella_acc': float,
            'total_tokens': int,
        }
    """
    final_row = df.iloc[-1]

    # Get final validation metrics (last non-null value)
    final_val_loss = df['val_loss'].dropna().iloc[-1] if 'val_loss' in df.columns and not df['val_loss'].dropna().empty else None
    final_hella_acc = df['hella_acc'].dropna().iloc[-1] if 'hella_acc' in df.columns and not df['hella_acc'].dropna().empty else None

    return {
        'final_step': int(final_row['step']),
        'final_train_loss': float(final_row['train_loss']),
        'final_val_loss': float(final_val_loss) if final_val_loss is not None else None,
        'final_hella_acc': float(final_hella_acc) if final_hella_acc is not None else None,
        'min_train_loss': float(df['train_loss'].min()),
        'min_val_loss': float(df['val_loss'].min()) if 'val_loss' in df.columns else None,
        'max_hella_acc': float(df['hella_acc'].max()) if 'hella_acc' in df.columns else None,
        'total_tokens': int(final_row['step']) * tokens_per_step,
    }


def get_convergence_stats(df: pd.DataFrame, tokens_per_step: int = 524288) -> dict:
    """
    Analyze convergence behavior.

    Args:
        df: DataFrame from load_metrics()
        tokens_per_step: Number of tokens processed per step (default: 524288)

    Returns:
        {
            'steps_to_val_loss_3.5': int,
            'steps_to_val_loss_3.3': int,
            'plateau_detected': bool,
            'plateau_step': int,
        }
    """
    result = {
        'steps_to_val_loss_3.5': None,
        'steps_to_val_loss_3.3': None,
        'plateau_detected': False,
        'plateau_step': None,
    }

    if 'val_loss' not in df.columns:
        return result

    # Find first step where val_loss drops below thresholds
    val_loss_data = df[['step', 'val_loss']].dropna()

    if not val_loss_data.empty:
        below_3_5 = val_loss_data[val_loss_data['val_loss'] < 3.5]
        if not below_3_5.empty:
            result['steps_to_val_loss_3.5'] = int(below_3_5.iloc[0]['step'])

        below_3_3 = val_loss_data[val_loss_data['val_loss'] < 3.3]
        if not below_3_3.empty:
            result['steps_to_val_loss_3.3'] = int(below_3_3.iloc[0]['step'])

    # Detect plateau: check if val_loss hasn't improved in last 20% of training
    if len(val_loss_data) > 10:
        split_point = int(len(val_loss_data) * 0.8)
        early_min = val_loss_data.iloc[:split_point]['val_loss'].min()
        late_min = val_loss_data.iloc[split_point:]['val_loss'].min()

        # Plateau detected if late phase doesn't improve by more than 0.01
        if late_min >= early_min - 0.01:
            result['plateau_detected'] = True
            # Find the step where minimum was reached
            min_idx = val_loss_data['val_loss'].idxmin()
            result['plateau_step'] = int(val_loss_data.loc[min_idx, 'step'])

    return result


def get_training_efficiency_stats(df: pd.DataFrame) -> dict:
    """
    Compute training efficiency statistics.

    Args:
        df: DataFrame from load_metrics()

    Returns:
        {
            'total_time_hours': float,
            'mean_step_time_ms': float,
            'mean_throughput': float,
            'std_throughput': float,
        }
    """
    result = {
        'total_time_hours': None,
        'mean_step_time_ms': None,
        'mean_throughput': None,
        'std_throughput': None,
    }

    if 'dt_ms' in df.columns:
        total_time_ms = df['dt_ms'].sum()
        result['total_time_hours'] = total_time_ms / (1000 * 60 * 60)
        result['mean_step_time_ms'] = df['dt_ms'].mean()

    if 'tok_sec' in df.columns:
        result['mean_throughput'] = df['tok_sec'].mean()
        result['std_throughput'] = df['tok_sec'].std()

    return result


def get_layer_analysis_stats(df: pd.DataFrame, num_layers: int = 12) -> dict:
    """
    Analyze per-layer statistics.

    Args:
        df: DataFrame from load_metrics()
        num_layers: Number of transformer layers (default: 12 for GPT-2 124M)

    Returns:
        {
            'highest_grad_norm_layer': int,
            'highest_grad_norm_mean': float,
            'lowest_grad_norm_layer': int,
            'lowest_grad_norm_mean': float,
            'update_ratio_min': float,
            'update_ratio_max': float,
        }
    """
    result = {
        'highest_grad_norm_layer': None,
        'highest_grad_norm_mean': None,
        'lowest_grad_norm_layer': None,
        'lowest_grad_norm_mean': None,
        'update_ratio_min': None,
        'update_ratio_max': None,
    }

    # Analyze gradient norms
    grad_norm_cols = [f'layer_grad_norms_layer{i}' for i in range(num_layers)]
    available_grad_cols = [col for col in grad_norm_cols if col in df.columns]

    if available_grad_cols:
        layer_means = df[available_grad_cols].mean()
        max_layer_idx = layer_means.idxmax()
        min_layer_idx = layer_means.idxmin()

        result['highest_grad_norm_layer'] = int(max_layer_idx.split('layer')[-1])
        result['highest_grad_norm_mean'] = float(layer_means[max_layer_idx])
        result['lowest_grad_norm_layer'] = int(min_layer_idx.split('layer')[-1])
        result['lowest_grad_norm_mean'] = float(layer_means[min_layer_idx])

    # Analyze update ratios
    update_ratio_cols = [f'layer_update_ratios_layer{i}' for i in range(num_layers)]
    available_update_cols = [col for col in update_ratio_cols if col in df.columns]

    if available_update_cols:
        update_data = df[available_update_cols]
        result['update_ratio_min'] = float(update_data.min().min())
        result['update_ratio_max'] = float(update_data.max().max())

    return result
