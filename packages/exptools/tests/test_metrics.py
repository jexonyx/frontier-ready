"""Tests for exptools.metrics: data loading and statistical analysis functions."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from exptools.metrics import (
    get_convergence_stats,
    get_final_metrics,
    get_layer_analysis_stats,
    get_training_efficiency_stats,
    load_metrics,
)


def _rec(step, **kwargs):
    """Build a minimal valid metrics record."""
    return {"step": step, "train_loss": round(4.0 - step * 0.01, 4), **kwargs}


# ─── load_metrics ─────────────────────────────────────────────────────────────


class TestLoadMetrics:
    def test_happy_path(self, write_jsonl):
        records = [_rec(i) for i in range(5)]
        df = load_metrics(write_jsonl(records))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert list(df['step']) == [0, 1, 2, 3, 4]
        assert 'train_loss' in df.columns

    def test_sparse_val_loss_forward_filled(self, write_jsonl):
        records = [_rec(i) for i in range(10)]
        records[0]['val_loss'] = 4.0
        records[5]['val_loss'] = 3.5
        df = load_metrics(write_jsonl(records))

        assert 'val_loss_filled' in df.columns
        # Steps 1-4 should inherit val_loss=4.0 from step 0
        assert df.loc[df['step'] == 1, 'val_loss_filled'].iloc[0] == pytest.approx(4.0)
        assert df.loc[df['step'] == 4, 'val_loss_filled'].iloc[0] == pytest.approx(4.0)
        # Steps 5-9 should inherit val_loss=3.5 from step 5
        assert df.loc[df['step'] == 5, 'val_loss_filled'].iloc[0] == pytest.approx(3.5)
        assert df.loc[df['step'] == 9, 'val_loss_filled'].iloc[0] == pytest.approx(3.5)

    def test_hella_acc_forward_filled(self, write_jsonl):
        records = [_rec(i) for i in range(6)]
        records[0]['hella_acc'] = 0.25
        df = load_metrics(write_jsonl(records))
        assert 'hella_acc_filled' in df.columns
        assert df['hella_acc_filled'].iloc[5] == pytest.approx(0.25)

    def test_layer_flattening(self, write_jsonl):
        records = [
            {**_rec(0), 'layer_grad_norms': [0.1, 0.2, 0.3]},
            {**_rec(1), 'layer_grad_norms': [0.4, 0.5, 0.6]},
        ]
        df = load_metrics(write_jsonl(records))
        assert 'layer_grad_norms_layer0' in df.columns
        assert 'layer_grad_norms_layer1' in df.columns
        assert 'layer_grad_norms_layer2' in df.columns
        assert df['layer_grad_norms_layer0'].iloc[0] == pytest.approx(0.1)
        assert df['layer_grad_norms_layer2'].iloc[1] == pytest.approx(0.6)

    def test_ragged_layer_lists_padded_with_nan(self, write_jsonl):
        records = [
            {**_rec(0), 'layer_grad_norms': [0.1, 0.2, 0.3]},
            {**_rec(1), 'layer_grad_norms': [0.4, 0.5]},
        ]
        df = load_metrics(write_jsonl(records))
        assert 'layer_grad_norms_layer2' in df.columns
        assert math.isnan(df.loc[df['step'] == 1, 'layer_grad_norms_layer2'].iloc[0])

    def test_all_four_layer_metrics_flattened(self, write_jsonl):
        records = [
            {
                **_rec(0),
                'layer_grad_norms': [0.1, 0.2],
                'layer_weight_norms': [5.0, 5.1],
                'layer_update_ratios': [0.001, 0.002],
                'layer_act_norms': [1.0, 1.1],
            }
        ]
        df = load_metrics(write_jsonl(records))
        assert 'layer_grad_norms_layer0' in df.columns
        assert 'layer_weight_norms_layer0' in df.columns
        assert 'layer_update_ratios_layer0' in df.columns
        assert 'layer_act_norms_layer0' in df.columns

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_metrics('/nonexistent/path/to/nowhere')

    def test_empty_file_raises(self, write_jsonl):
        log_dir = write_jsonl([])
        with pytest.raises(ValueError, match='No valid records'):
            load_metrics(log_dir)

    def test_all_invalid_json_raises(self, tmp_path):
        log_dir = tmp_path / 'bad'
        log_dir.mkdir()
        (log_dir / 'metrics.jsonl').write_text('not json\nalso not json\n')
        with pytest.raises(ValueError, match='No valid records'):
            load_metrics(str(log_dir))

    def test_partial_invalid_json_skipped(self, write_jsonl, capsys):
        log_dir = write_jsonl([_rec(0), _rec(1)])
        with open(f'{log_dir}/metrics.jsonl', 'a') as f:
            f.write('not json at all\n')
        df = load_metrics(log_dir)
        assert len(df) == 2
        assert 'Warning' in capsys.readouterr().out

    def test_missing_step_column_raises(self, write_jsonl):
        records = [{'train_loss': 4.0}, {'train_loss': 3.9}]
        with pytest.raises(ValueError, match="Missing required 'step' column"):
            load_metrics(write_jsonl(records))

    def test_duplicate_steps_keeps_last(self, write_jsonl, capsys):
        records = [
            _rec(0),
            {'step': 1, 'train_loss': 3.9},
            {'step': 1, 'train_loss': 2.5},  # duplicate — last should win
            _rec(2),
        ]
        df = load_metrics(write_jsonl(records))
        assert len(df) == 3
        assert df.loc[df['step'] == 1, 'train_loss'].iloc[0] == pytest.approx(2.5)
        assert 'duplicate' in capsys.readouterr().out.lower()

    def test_nonmonotonic_steps_sorted(self, write_jsonl, capsys):
        records = [_rec(0), _rec(3), _rec(1), _rec(2)]
        df = load_metrics(write_jsonl(records))
        assert list(df['step']) == [0, 1, 2, 3]
        assert 'monotonic' in capsys.readouterr().out.lower()

    def test_single_row_loads(self, write_jsonl):
        df = load_metrics(write_jsonl([_rec(0)]))
        assert len(df) == 1
        assert df['step'].iloc[0] == 0

    def test_blank_lines_ignored(self, tmp_path):
        log_dir = tmp_path / 'blanks'
        log_dir.mkdir()
        with open(log_dir / 'metrics.jsonl', 'w') as f:
            f.write('\n')
            f.write(json.dumps(_rec(0)) + '\n')
            f.write('\n')
            f.write(json.dumps(_rec(1)) + '\n')
            f.write('\n')
        df = load_metrics(str(log_dir))
        assert len(df) == 2


# ─── get_final_metrics ────────────────────────────────────────────────────────


class TestGetFinalMetrics:
    def test_standard_fields_present(self, minimal_df):
        result = get_final_metrics(minimal_df)
        expected_keys = {
            'final_step', 'final_train_loss', 'final_val_loss', 'final_hella_acc',
            'min_train_loss', 'min_val_loss', 'max_hella_acc', 'total_tokens',
        }
        assert set(result.keys()) == expected_keys

    def test_final_step_is_last(self, minimal_df):
        result = get_final_metrics(minimal_df)
        assert result['final_step'] == 19

    def test_final_train_loss_matches_last_row(self, minimal_df):
        result = get_final_metrics(minimal_df)
        assert result['final_train_loss'] == pytest.approx(minimal_df['train_loss'].iloc[-1])

    def test_min_train_loss_is_global_min(self, minimal_df):
        result = get_final_metrics(minimal_df)
        assert result['min_train_loss'] == pytest.approx(minimal_df['train_loss'].min())

    def test_no_val_loss_returns_none(self, minimal_df):
        result = get_final_metrics(minimal_df)
        assert result['final_val_loss'] is None
        assert result['min_val_loss'] is None

    def test_no_hella_acc_returns_none(self, minimal_df):
        result = get_final_metrics(minimal_df)
        assert result['final_hella_acc'] is None
        assert result['max_hella_acc'] is None

    def test_val_loss_extracted_correctly(self, full_df):
        result = get_final_metrics(full_df)
        assert result['final_val_loss'] is not None
        assert result['min_val_loss'] is not None
        assert result['min_val_loss'] <= result['final_val_loss']

    def test_total_tokens_equals_final_step_times_rate(self, minimal_df):
        tokens_per_step = 1000
        result = get_final_metrics(minimal_df, tokens_per_step=tokens_per_step)
        assert result['total_tokens'] == 19 * tokens_per_step

    def test_custom_tokens_per_step(self, minimal_df):
        result_a = get_final_metrics(minimal_df, tokens_per_step=524288)
        result_b = get_final_metrics(minimal_df, tokens_per_step=262144)
        assert result_a['total_tokens'] == 2 * result_b['total_tokens']

    def test_all_val_nan_min_should_be_none(self, minimal_df):
        df = minimal_df.copy()
        df['val_loss'] = float('nan')
        result = get_final_metrics(df)
        assert result['min_val_loss'] is None


# ─── get_convergence_stats ────────────────────────────────────────────────────


class TestGetConvergenceStats:
    def test_no_val_loss_column(self, minimal_df):
        result = get_convergence_stats(minimal_df)
        assert result['steps_to_val_loss_3.5'] is None
        assert result['steps_to_val_loss_3.3'] is None
        assert result['plateau_detected'] is False
        assert result['plateau_step'] is None

    def test_threshold_3_5_first_crossing(self):
        df = pd.DataFrame({
            'step': [0, 250, 500, 750, 1000],
            'train_loss': [4.0, 3.8, 3.6, 3.4, 3.2],
            'val_loss': [4.0, 3.7, 3.4, 3.2, 3.1],  # drops below 3.5 at step 500
        })
        result = get_convergence_stats(df)
        assert result['steps_to_val_loss_3.5'] == 500

    def test_threshold_3_3_first_crossing(self):
        df = pd.DataFrame({
            'step': [0, 250, 500, 750, 1000],
            'train_loss': [4.0, 3.8, 3.6, 3.4, 3.2],
            'val_loss': [4.0, 3.7, 3.4, 3.2, 3.1],  # below 3.3 at step 750
        })
        result = get_convergence_stats(df)
        assert result['steps_to_val_loss_3.3'] == 750

    def test_threshold_never_reached_returns_none(self):
        df = pd.DataFrame({
            'step': [0, 250, 500],
            'train_loss': [4.5, 4.3, 4.1],
            'val_loss': [4.5, 4.3, 4.1],
        })
        result = get_convergence_stats(df)
        assert result['steps_to_val_loss_3.5'] is None
        assert result['steps_to_val_loss_3.3'] is None

    def test_plateau_detected_when_loss_stalls(self):
        # Loss drops strongly in first 80%, then flatlines in last 20%
        steps = list(range(20))
        val_loss = [4.0 - i * 0.1 for i in range(11)] + [2.9] * 9
        df = pd.DataFrame({
            'step': steps,
            'train_loss': [4.5 - i * 0.05 for i in range(20)],
            'val_loss': val_loss,
        })
        result = get_convergence_stats(df)
        assert result['plateau_detected'] is True
        assert result['plateau_step'] is not None

    def test_no_plateau_with_steady_improvement(self):
        # Loss drops substantially throughout including the last 20%
        # early_min = 1.75 (at index 15), late_min = 1.15 (at index 19)
        # 1.15 >= 1.75 - 0.01 = 1.74 → False → no plateau
        val_loss = [4.0 - i * 0.15 for i in range(20)]
        df = pd.DataFrame({
            'step': list(range(20)),
            'train_loss': [4.5 - i * 0.1 for i in range(20)],
            'val_loss': val_loss,
        })
        result = get_convergence_stats(df)
        assert result['plateau_detected'] is False

    def test_fewer_than_10_points_skips_plateau_check(self):
        # With exactly 10 val_loss points, len > 10 is False — plateau check skipped
        df = pd.DataFrame({
            'step': list(range(10)),
            'train_loss': [4.0] * 10,
            'val_loss': [4.0] * 10,  # flat — would trigger plateau if check ran
        })
        result = get_convergence_stats(df)
        assert result['plateau_detected'] is False


# ─── get_training_efficiency_stats ───────────────────────────────────────────


class TestGetTrainingEfficiencyStats:
    def test_standard_all_fields_populated(self, full_df):
        result = get_training_efficiency_stats(full_df)
        assert result['total_time_hours'] is not None
        assert result['mean_step_time_ms'] is not None
        assert result['mean_throughput'] is not None
        assert result['std_throughput'] is not None

    def test_time_conversion_math(self):
        # Two steps each taking exactly 1 hour (3,600,000 ms) → total 2 hours
        df = pd.DataFrame({
            'step': [0, 1],
            'train_loss': [4.0, 3.9],
            'dt_ms': [3_600_000.0, 3_600_000.0],
        })
        result = get_training_efficiency_stats(df)
        assert result['total_time_hours'] == pytest.approx(2.0)
        assert result['mean_step_time_ms'] == pytest.approx(3_600_000.0)

    def test_no_dt_ms_returns_none(self, minimal_df):
        result = get_training_efficiency_stats(minimal_df)
        assert result['total_time_hours'] is None
        assert result['mean_step_time_ms'] is None

    def test_no_tok_sec_returns_none(self, minimal_df):
        result = get_training_efficiency_stats(minimal_df)
        assert result['mean_throughput'] is None
        assert result['std_throughput'] is None

    def test_single_row_std_is_nan(self):
        # pandas returns NaN for std of a single-element series (ddof=1)
        df = pd.DataFrame({
            'step': [0],
            'train_loss': [4.0],
            'dt_ms': [300.0],
            'tok_sec': [50_000.0],
        })
        result = get_training_efficiency_stats(df)
        assert math.isnan(result['std_throughput'])

    def test_mean_throughput_is_arithmetic_mean(self):
        df = pd.DataFrame({
            'step': [0, 1, 2],
            'train_loss': [4.0, 3.9, 3.8],
            'tok_sec': [50_000.0, 60_000.0, 40_000.0],
        })
        result = get_training_efficiency_stats(df)
        assert result['mean_throughput'] == pytest.approx(50_000.0)


# ─── get_layer_analysis_stats ─────────────────────────────────────────────────


class TestGetLayerAnalysisStats:
    def test_no_layer_columns_all_none(self, minimal_df):
        result = get_layer_analysis_stats(minimal_df)
        assert all(v is None for v in result.values())

    def test_highest_grad_norm_layer_identified(self, layer_df):
        # layer_df: layer i has base grad norm ≈ 0.1*(i+1), so layer 11 is highest
        result = get_layer_analysis_stats(layer_df, num_layers=12)
        assert result['highest_grad_norm_layer'] == 11
        assert result['highest_grad_norm_mean'] > 0

    def test_lowest_grad_norm_layer_identified(self, layer_df):
        # Layer 0 has base grad norm ≈ 0.1, the lowest
        result = get_layer_analysis_stats(layer_df, num_layers=12)
        assert result['lowest_grad_norm_layer'] == 0
        assert result['lowest_grad_norm_mean'] < result['highest_grad_norm_mean']

    def test_update_ratio_range_is_global_min_max(self, layer_df):
        result = get_layer_analysis_stats(layer_df, num_layers=12)
        assert result['update_ratio_min'] is not None
        assert result['update_ratio_max'] is not None
        assert result['update_ratio_max'] > result['update_ratio_min']

        # Verify against manual computation
        update_cols = [f'layer_update_ratios_layer{i}' for i in range(12)]
        expected_min = float(layer_df[update_cols].min().min())
        expected_max = float(layer_df[update_cols].max().max())
        assert result['update_ratio_min'] == pytest.approx(expected_min)
        assert result['update_ratio_max'] == pytest.approx(expected_max)

    def test_fewer_layers_than_num_layers_uses_available(self, full_df):
        df = full_df.copy()
        rng = np.random.default_rng(1)
        for i in range(6):
            df[f'layer_grad_norms_layer{i}'] = 0.1 * (i + 1) + rng.uniform(-0.005, 0.005, len(df))
        result = get_layer_analysis_stats(df, num_layers=12)
        # Only layers 0-5 present: layer 5 should be highest, layer 0 lowest
        assert result['highest_grad_norm_layer'] == 5
        assert result['lowest_grad_norm_layer'] == 0

    def test_column_name_parsing_high_index(self):
        # Verifies that 'layer_grad_norms_layer11'.split('layer')[-1] → '11'
        df = pd.DataFrame({
            'step': [0, 1],
            'train_loss': [4.0, 3.9],
            'layer_grad_norms_layer0': [0.1, 0.15],
            'layer_grad_norms_layer11': [1.1, 1.15],
        })
        result = get_layer_analysis_stats(df, num_layers=12)
        assert result['highest_grad_norm_layer'] == 11
        assert result['lowest_grad_norm_layer'] == 0

    def test_only_update_ratios_no_grad_norms(self, full_df):
        df = full_df.copy()
        rng = np.random.default_rng(2)
        for i in range(4):
            df[f'layer_update_ratios_layer{i}'] = 1e-3 * (i + 1) + rng.uniform(-1e-4, 1e-4, len(df))
        result = get_layer_analysis_stats(df, num_layers=4)
        assert result['highest_grad_norm_layer'] is None  # no grad norm cols
        assert result['update_ratio_min'] is not None
        assert result['update_ratio_max'] is not None

    def test_all_nan_layer_values_returns_none(self):
        df = pd.DataFrame({
            'step': [0, 1],
            'train_loss': [4.0, 3.9],
            'layer_grad_norms_layer0': [float('nan'), float('nan')],
            'layer_grad_norms_layer1': [float('nan'), float('nan')],
        })
        result = get_layer_analysis_stats(df, num_layers=2)
        assert result['highest_grad_norm_layer'] is None
        assert result['highest_grad_norm_mean'] is None
        assert result['lowest_grad_norm_layer'] is None
