"""Provides the synchronous calcium event (SCE) detection pipeline and per-session report container."""

from .plotting import plot_sce_assemblies, plot_sce_cells_across_periods
from .sce_report import (
    SCEReport,
    SCESummary,
    SCECellColumn,
    SCEPeriodColumn,
    evaluate_and_save_sce_report,
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
    "evaluate_and_save_sce_report",
    "plot_sce_assemblies",
    "plot_sce_cells_across_periods",
]
