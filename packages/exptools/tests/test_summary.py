"""Tests for exptools.summary: number formatting and markdown report generation."""

import pytest

from exptools.summary import format_number, format_tokens, generate_summary


# ─── format_number ────────────────────────────────────────────────────────────


class TestFormatNumber:
    def test_integer_with_comma(self):
        assert format_number(1000) == '1,000'

    def test_large_integer(self):
        assert format_number(1_000_000) == '1,000,000'

    def test_float_equal_to_int(self):
        assert format_number(1000.0) == '1,000'

    def test_float_with_default_decimals(self):
        assert format_number(3.14159) == '3.142'

    def test_float_with_custom_decimals(self):
        assert format_number(3.14159, decimals=1) == '3.1'

    def test_zero_decimals(self):
        assert format_number(42.7, decimals=0) == '43'

    def test_none_returns_na(self):
        assert format_number(None) == 'N/A'

    def test_small_float(self):
        assert format_number(0.001, decimals=4) == '0.0010'

    def test_nan_should_return_na(self):
        assert format_number(float('nan')) == 'N/A'

    def test_inf_should_return_na(self):
        assert format_number(float('inf')) == 'N/A'


# ─── format_tokens ────────────────────────────────────────────────────────────


class TestFormatTokens:
    def test_none_returns_na(self):
        assert format_tokens(None) == 'N/A'

    def test_billions(self):
        assert format_tokens(10_000_000_000) == '~10.0B'

    def test_exactly_one_billion(self):
        assert format_tokens(1_000_000_000) == '~1.0B'

    def test_hundreds_of_millions(self):
        assert format_tokens(500_000_000) == '~500M'

    def test_single_millions(self):
        assert format_tokens(10_000_000) == '~10M'

    def test_zero_tokens(self):
        # Edge case: 0 / 1e9 = 0 < 1 → millions path → "~0M"
        result = format_tokens(0)
        assert result == '~0M'

    def test_just_under_one_billion(self):
        assert format_tokens(999_000_000) == '~999M'


# ─── generate_summary ─────────────────────────────────────────────────────────


def _standard_records(n=50, include_val_loss=True, include_hella=True, val_loss_level=3.0):
    """Build a list of JSONL records for summary generation tests."""
    records = []
    for i in range(n):
        rec = {
            'step': i,
            'train_loss': round(4.5 - i * 0.03, 4),
            'lr': round(6e-4 * (1 - i / n), 8),
            'grad_norm': 0.8,
            'tok_sec': 50_000,
            'gpu_mem_gb': 15.0,
            'dt_ms': 300.0,
        }
        if include_val_loss and i % 10 == 0:
            rec['val_loss'] = round(val_loss_level + (n - i) * 0.01, 4)
        if include_hella and i % 10 == 0:
            rec['hella_acc'] = round(0.20 + i * 0.001, 4)
        records.append(rec)
    return records


class TestGenerateSummary:
    def test_all_sections_present(self, write_jsonl):
        log_dir = write_jsonl(_standard_records())
        summary = generate_summary(log_dir)
        assert '## Final Metrics' in summary
        assert '## Convergence Statistics' in summary
        assert '## Training Efficiency' in summary
        assert '## Layer Analysis' in summary

    def test_run_name_from_log_dir_basename(self, write_jsonl):
        log_dir = write_jsonl(_standard_records(), subdir='my_experiment')
        summary = generate_summary(log_dir)
        assert 'my_experiment' in summary

    def test_model_size_in_header(self, write_jsonl):
        log_dir = write_jsonl(_standard_records())
        summary = generate_summary(log_dir, model_size='124M')
        assert '124M' in summary

    def test_baseline_delta_computed_for_124m(self, write_jsonl):
        # 124M has a documented val_loss baseline of 3.2924; delta should appear
        records = _standard_records(val_loss_level=3.0)
        log_dir = write_jsonl(records)
        summary = generate_summary(log_dir, model_size='124M')
        # Delta sign is present (+/-)
        assert '+' in summary or any(line.count('-') >= 2 for line in summary.splitlines())
        assert '3.292' in summary  # baseline value

    def test_no_val_loss_baseline_for_350m(self, write_jsonl):
        records = _standard_records()
        log_dir = write_jsonl(records)
        summary = generate_summary(log_dir, model_size='350M')
        # 350M has no val_loss baseline so the delta column shows "-"
        lines = [l for l in summary.splitlines() if 'Validation Loss' in l]
        assert len(lines) == 1
        # The baseline and delta cells should both be "-"
        assert lines[0].count('- |') >= 1

    def test_missing_val_loss_shows_na(self, write_jsonl):
        records = _standard_records(include_val_loss=False)
        log_dir = write_jsonl(records)
        summary = generate_summary(log_dir)
        val_line = next(l for l in summary.splitlines() if 'Validation Loss' in l)
        assert 'N/A' in val_line

    def test_missing_hella_acc_shows_na(self, write_jsonl):
        records = _standard_records(include_hella=False)
        log_dir = write_jsonl(records)
        summary = generate_summary(log_dir)
        hella_line = next(l for l in summary.splitlines() if 'HellaSwag' in l)
        assert 'N/A' in hella_line

    def test_convergence_table_empty_when_no_thresholds_crossed(self, write_jsonl):
        # val_loss stays at ~4.0, well above 3.5 and 3.3 thresholds; no hella
        records = _standard_records(include_hella=False, val_loss_level=4.0)
        log_dir = write_jsonl(records, subdir='high_loss')
        summary = generate_summary(log_dir)
        assert '## Convergence Statistics' in summary
        assert 'Val loss < 3.5' not in summary
        assert 'Val loss < 3.3' not in summary

    def test_throughput_in_efficiency_section(self, write_jsonl):
        records = _standard_records()
        log_dir = write_jsonl(records)
        summary = generate_summary(log_dir)
        assert 'tokens/sec' in summary.lower() or 'throughput' in summary.lower()

    def test_gpu_memory_in_final_metrics(self, write_jsonl):
        records = _standard_records()
        log_dir = write_jsonl(records)
        summary = generate_summary(log_dir)
        assert 'GPU Memory' in summary
        assert 'GB' in summary

    def test_total_tokens_formatted(self, write_jsonl):
        records = _standard_records(n=20)
        log_dir = write_jsonl(records)
        summary = generate_summary(log_dir, tokens_per_step=524288)
        # 19 * 524288 ≈ 9.97M tokens — should appear as ~10M or ~0.0B etc.
        assert 'Total tokens' in summary

    def test_layer_analysis_section_empty_without_layer_data(self, write_jsonl):
        records = _standard_records()
        log_dir = write_jsonl(records)
        summary = generate_summary(log_dir)
        # Without layer columns, layer analysis section is empty (no bullets)
        layer_section_idx = summary.find('## Layer Analysis')
        assert layer_section_idx >= 0
        layer_section_content = summary[layer_section_idx:]
        assert 'Layer with highest' not in layer_section_content
