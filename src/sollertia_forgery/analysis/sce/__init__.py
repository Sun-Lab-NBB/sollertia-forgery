"""Provides the synchronous calcium event (SCE) detection pipeline and per-session report container."""

from .plotting import plot_sce_assemblies, plot_sce_cells_across_periods
from .sce_report import (
    SCEReport,
    SCESummary,
    SCECellColumn,
    SCEPeriodColumn,
    run_sce_analysis,
)
from .sce_protocol import SCEResult, SCEDetector, SCEDetectionConfiguration

__all__ = [
    "SCECellColumn",
    "SCEDetectionConfiguration",
    "SCEDetector",
    "SCEPeriodColumn",
    "SCEReport",
    "SCEResult",
    "SCESummary",
    "plot_sce_assemblies",
    "plot_sce_cells_across_periods",
    "run_sce_analysis",
]
