"""
Tests for exptools.visualization: matplotlib plot generation.

All tests use the Agg backend (set in conftest.py before any pyplot import)
and write to tmp_path rather than display. Tests verify file creation and
graceful handling of missing columns — not pixel-level plot content.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from exptools.visualization import (
    plot_hellaswag_accuracy,
    plot_layer_analysis,
    plot_loss_curves,
    plot_training_dynamics,
)


@pytest.fixture
def out(tmp_path):
    """Return a helper that gives a unique output PNG path."""
    counter = [0]
    def _path(name=None):
        counter[0] += 1
        return tmp_path / (name or f'plot_{counter[0]}.png')
    return _path


# ─── plot_loss_curves ─────────────────────────────────────────────────────────


class TestPlotLossCurves:
    def test_creates_png_file(self, full_df, out):
        path = out('loss.png')
        plot_loss_curves(full_df, path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_no_val_loss_column_no_error(self, minimal_df, out):
        path = out()
        plot_loss_curves(minimal_df, path)
        assert path.exists()

    def test_unknown_model_size_no_error(self, full_df, out):
        path = out()
        plot_loss_curves(full_df, path, model_size='999M')
        assert path.exists()

    def test_known_model_size_124m(self, full_df, out):
        path = out()
        plot_loss_curves(full_df, path, model_size='124M')
        assert path.exists()

    def test_accepts_pathlib_path(self, full_df, out):
        path = out()
        plot_loss_curves(full_df, Path(path))
        assert Path(path).exists()

    def test_single_row_df_no_val_loss(self, out):
        df = pd.DataFrame({'step': [0], 'train_loss': [4.0]})
        path = out()
        plot_loss_curves(df, path)
        assert path.exists()


# ─── plot_hellaswag_accuracy ──────────────────────────────────────────────────


class TestPlotHellaswagAccuracy:
    def test_creates_png_file(self, full_df, out):
        path = out('hella.png')
        plot_hellaswag_accuracy(full_df, path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_no_hella_acc_column_no_error(self, minimal_df, out):
        path = out()
        plot_hellaswag_accuracy(minimal_df, path)
        assert path.exists()

    def test_unknown_model_size_no_baselines(self, full_df, out):
        path = out()
        plot_hellaswag_accuracy(full_df, path, model_size='999M')
        assert path.exists()

    def test_124m_draws_gpt2_and_gpt3_baselines(self, full_df, out):
        # Both GPT-2 and GPT-3 baselines are in BASELINES dict for 124M
        path = out()
        plot_hellaswag_accuracy(full_df, path, model_size='124M')
        assert path.exists()

    def test_350m_draws_baselines(self, full_df, out):
        path = out()
        plot_hellaswag_accuracy(full_df, path, model_size='350M')
        assert path.exists()


# ─── plot_training_dynamics ───────────────────────────────────────────────────


class TestPlotTrainingDynamics:
    def test_creates_png_file(self, full_df, out):
        path = out('dynamics.png')
        plot_training_dynamics(full_df, path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_all_four_panels(self, full_df, out):
        path = out()
        plot_training_dynamics(full_df, path)
        assert path.exists()

    def test_partial_columns_no_error(self, out):
        # Only lr present — other panels should be silently skipped
        df = pd.DataFrame({
            'step': list(range(10)),
            'train_loss': [4.0 - i * 0.1 for i in range(10)],
            'lr': [6e-4 - i * 1e-5 for i in range(10)],
        })
        path = out()
        plot_training_dynamics(df, path)
        assert path.exists()

    def test_no_dynamic_columns_no_error(self, minimal_df, out):
        # None of lr, grad_norm, tok_sec, gpu_mem_gb present — empty subplots
        path = out()
        plot_training_dynamics(minimal_df, path)
        assert path.exists()

    def test_only_grad_norm_no_error(self, out):
        df = pd.DataFrame({
            'step': list(range(10)),
            'train_loss': [4.0 - i * 0.1 for i in range(10)],
            'grad_norm': [1.0] * 10,
        })
        path = out()
        plot_training_dynamics(df, path)
        assert path.exists()


# ─── plot_layer_analysis ──────────────────────────────────────────────────────


class TestPlotLayerAnalysis:
    def test_creates_png_file(self, layer_df, out):
        path = out('layers.png')
        plot_layer_analysis(layer_df, path, num_layers=12)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_no_layer_columns_no_error(self, minimal_df, out):
        # No layer columns present — should show "No data available" panels
        path = out()
        plot_layer_analysis(minimal_df, path, num_layers=12)
        assert path.exists()

    def test_all_four_layer_metric_types(self, layer_df, out):
        # layer_df has all 4 types: grad_norms, weight_norms, update_ratios, act_norms
        path = out()
        plot_layer_analysis(layer_df, path, num_layers=12)
        assert path.exists()

    def test_partial_layer_types_no_error(self, full_df, out):
        df = full_df.copy()
        rng = np.random.default_rng(0)
        for i in range(4):
            df[f'layer_grad_norms_layer{i}'] = 0.1 * (i + 1) + rng.uniform(-0.01, 0.01, len(df))
        path = out()
        plot_layer_analysis(df, path, num_layers=4)
        assert path.exists()

    def test_fewer_layers_than_requested(self, full_df, out):
        # DataFrame has only 4 layers; num_layers=12 — should silently use available
        df = full_df.copy()
        rng = np.random.default_rng(0)
        for i in range(4):
            df[f'layer_grad_norms_layer{i}'] = 0.1 + rng.uniform(-0.01, 0.01, len(df))
        path = out()
        plot_layer_analysis(df, path, num_layers=12)
        assert path.exists()

    def test_more_layers_in_df_than_num_layers_no_error(self, out):
        # DataFrame has 13 layers but num_layers=12.
        # The function only requests layers 0-11 via range(num_layers),
        # so layer 12 is ignored and colors[layer_num] stays in bounds.
        n = 20
        rng = np.random.default_rng(5)
        data = {'step': list(range(n)), 'train_loss': [4.0 - i * 0.1 for i in range(n)]}
        for i in range(13):
            data[f'layer_grad_norms_layer{i}'] = 0.1 * (i + 1) + rng.uniform(-0.01, 0.01, n)
        df = pd.DataFrame(data)
        path = out()
        plot_layer_analysis(df, path, num_layers=12)
        assert path.exists()

    def test_custom_num_layers_2(self, out):
        n = 10
        rng = np.random.default_rng(3)
        df = pd.DataFrame({
            'step': list(range(n)),
            'train_loss': [4.0 - i * 0.1 for i in range(n)],
            'layer_grad_norms_layer0': rng.uniform(0.1, 0.2, n),
            'layer_grad_norms_layer1': rng.uniform(0.2, 0.3, n),
        })
        path = out()
        plot_layer_analysis(df, path, num_layers=2)
        assert path.exists()
