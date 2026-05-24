"""
Experiment analysis tools for GPT training runs.

This package provides utilities for:
- Loading and parsing training metrics
- Visualizing training curves
- Generating summary reports
"""

__version__ = "0.1.0"

from .metrics import (
    load_metrics,
    get_final_metrics,
    get_convergence_stats,
    get_training_efficiency_stats,
    get_layer_analysis_stats,
)

__all__ = [
    "load_metrics",
    "get_final_metrics",
    "get_convergence_stats",
    "get_training_efficiency_stats",
    "get_layer_analysis_stats",
]
