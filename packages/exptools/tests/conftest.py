import json
import math

import matplotlib
matplotlib.use('Agg')  # must be set before pyplot is imported anywhere

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def minimal_df():
    """DataFrame with only the minimum required columns: step and train_loss."""
    n = 20
    return pd.DataFrame({
        'step': list(range(n)),
        'train_loss': [4.0 - i * 0.05 for i in range(n)],
    })


@pytest.fixture
def full_df():
    """DataFrame with all standard scalar columns; val_loss and hella_acc every 10 steps."""
    n = 100
    val_loss = [None] * n
    hella_acc = [None] * n
    for i in range(0, n, 10):
        val_loss[i] = round(4.0 - i * 0.02, 4)
        hella_acc[i] = round(0.20 + i * 0.001, 4)

    df = pd.DataFrame({
        'step': list(range(n)),
        'train_loss': [round(4.5 - i * 0.015, 4) for i in range(n)],
        'val_loss': val_loss,
        'hella_acc': hella_acc,
        'lr': [round(6e-4 * math.cos(math.pi / 2 * i / n), 8) for i in range(n)],
        'grad_norm': [round(1.0 + 0.05 * math.sin(i * 0.2), 4) for i in range(n)],
        'tok_sec': [50_000 + i * 5 for i in range(n)],
        'gpu_mem_gb': [round(14.5 + 0.005 * i, 4) for i in range(n)],
        'dt_ms': [300.0] * n,
    })
    df['val_loss_filled'] = df['val_loss'].ffill()
    df['hella_acc_filled'] = df['hella_acc'].ffill()
    return df


@pytest.fixture
def layer_df(full_df):
    """full_df extended with 12-layer grad/weight/update/activation norm columns.

    Layer i has base grad norm ≈ 0.1*(i+1), so layer 11 is highest, layer 0 is lowest.
    Layer i has base update ratio ≈ 0.001*(i+1).
    """
    df = full_df.copy()
    n = len(df)
    rng = np.random.default_rng(42)
    for i in range(12):
        df[f'layer_grad_norms_layer{i}'] = 0.1 * (i + 1) + rng.uniform(-0.005, 0.005, n)
        df[f'layer_weight_norms_layer{i}'] = 5.0 + i * 0.1 + rng.uniform(-0.05, 0.05, n)
        df[f'layer_update_ratios_layer{i}'] = 1e-3 * (i + 1) + rng.uniform(-1e-4, 1e-4, n)
        df[f'layer_act_norms_layer{i}'] = 1.0 + i * 0.05 + rng.uniform(-0.01, 0.01, n)
    return df


@pytest.fixture
def write_jsonl(tmp_path):
    """Return a helper that writes records to tmp_path/<subdir>/metrics.jsonl and returns the dir path."""
    def _write(records, subdir='run'):
        log_dir = tmp_path / subdir
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / 'metrics.jsonl', 'w') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
        return str(log_dir)
    return _write
