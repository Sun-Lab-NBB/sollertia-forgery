"""Provides the bleaching analysis pipeline for assessing fluorescence baseline drift."""

from .plotting import (
    plot_baseline_trend,
    plot_within_session,
    plot_snr_distributions,
    plot_dataset_baseline_trend,
    plot_within_session_average,
)
from .bleaching_analysis import (
    BleachingColumn,
    BleachingReport,
    BleachingSummary,
    ExponentialDecayFit,
    run_bleaching_analysis,
)
from .bleaching_protocol import (
    BleachingConfiguration,
    BleachingSessionResult,
)

__all__ = [
    "BleachingColumn",
    "BleachingConfiguration",
    "BleachingReport",
    "BleachingSessionResult",
    "BleachingSummary",
    "ExponentialDecayFit",
    "plot_baseline_trend",
    "plot_dataset_baseline_trend",
    "plot_snr_distributions",
    "plot_within_session",
    "plot_within_session_average",
    "run_bleaching_analysis",
]
