"""Provides the cross-session tuning-drift analysis pipeline and per-animal report container."""

from .plotting import (
    classify_drift_profile,
    plot_drift_vs_bleaching,
    plot_recurrence_heatmap,
    plot_drift_profile_categories,
    plot_recurrence_probability_vs_lag,
    plot_sessions_active_distribution,
    plot_reference_day_sorted_rate_maps,
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
    "classify_drift_profile",
    "compute_drift_report",
    "compute_pairwise_drift_metrics",
    "compute_per_cell_drift_metrics",
    "fit_population_decay",
    "plot_drift_profile_categories",
    "plot_drift_vs_bleaching",
    "plot_population_vector_correlation_vs_lag",
    "plot_recurrence_heatmap",
    "plot_recurrence_probability_vs_lag",
    "plot_reference_day_sorted_rate_maps",
    "plot_sessions_active_distribution",
    "run_drift_analysis",
]
