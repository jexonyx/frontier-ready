"""
Analysis utilities for GPT-2 training runs.

This package provides tools for parsing, visualizing, and summarizing
training metrics from nanoGPT training runs.
"""

from .parse_metrics import (
    load_metrics,
    get_final_metrics,
    get_convergence_stats,
    get_training_efficiency_stats,
    get_layer_analysis_stats,
)

__all__ = [
    'load_metrics',
    'get_final_metrics',
    'get_convergence_stats',
    'get_training_efficiency_stats',
    'get_layer_analysis_stats',
]
