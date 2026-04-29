"""Provides the cross-session tuning-drift analysis pipeline and per-animal report container."""

from .plotting import (
    plot_drift_vs_bleaching,
    plot_recurrence_heatmap,
    plot_classification_raster,
    plot_peak_shift_distribution,
    plot_population_vector_correlation_vs_lag,
)
from .drift_report import (
    DriftReport,
    DriftSummary,
    DriftCellColumn,
    DriftPairColumn,
    run_drift_analysis,
    compute_drift_report,
)
from .drift_protocol import (
    CellDriftMetrics,
    PairwiseDriftMetrics,
    PopulationDriftMetrics,
    DriftDetectionConfiguration,
    fit_population_decay,
    compute_pairwise_drift_metrics,
    compute_per_cell_drift_metrics,
)

__all__ = [
    "CellDriftMetrics",
    "DriftCellColumn",
    "DriftDetectionConfiguration",
    "DriftPairColumn",
    "DriftReport",
    "DriftSummary",
    "PairwiseDriftMetrics",
    "PopulationDriftMetrics",
    "compute_drift_report",
    "compute_pairwise_drift_metrics",
    "compute_per_cell_drift_metrics",
    "fit_population_decay",
    "plot_classification_raster",
    "plot_drift_vs_bleaching",
    "plot_peak_shift_distribution",
    "plot_population_vector_correlation_vs_lag",
    "plot_recurrence_heatmap",
    "run_drift_analysis",
]
