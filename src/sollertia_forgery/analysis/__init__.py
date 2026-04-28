"""Provides cell analysis pipelines for place field, reward cell, and SCE detection."""

from .utilities import (
    RunSessionData,
    resolve_display_units,
    assemble_run_session_data,
    bin_fluorescence_by_position,
    compute_within_trial_position,
)
from .sce_protocol import SCEResult, PeriodType, SCEDetector, SCEDetectionConfiguration
from .cell_analysis import (
    SCEPeriodColumn,
    CellAnalysisColumn,
    CellAnalysisReport,
    CellAnalysisSummary,
    CellAnalysisConfiguration,
    plot_dataset_sce_rate,
    plot_dataset_cell_count,
    evaluate_and_save_cell_analysis,
    plot_dataset_place_cell_fraction,
    plot_dataset_reward_cell_fraction,
)
from .bleaching_analysis import (
    BleachingColumn,
    BleachingReport,
    BleachingSummary,
    ExponentialDecayFit,
    BleachingConfiguration,
    run_bleaching_analysis,
    plot_dataset_baseline_trend,
)
from .place_cell_protocol import PlaceFields, PlaceFieldDetector, PlaceFieldDetectionConfiguration
from .reward_cell_protocol import RewardCellResults, RewardCellDetector, RewardCellConfiguration
from .reward_relative_analysis import (
    RewardRelativeColumn,
    RewardRelativeReport,
    RewardRelativeSummary,
    RewardRelativeConfiguration,
)
from .longitudinal_reward_shift import (
    LongitudinalRewardShiftColumn,
    LongitudinalRewardShiftReport,
    LongitudinalRewardShiftSummary,
    LongitudinalRewardShiftConfiguration,
)

__all__ = [
    "BleachingColumn",
    "BleachingConfiguration",
    "BleachingReport",
    "BleachingSummary",
    "CellAnalysisColumn",
    "CellAnalysisConfiguration",
    "CellAnalysisReport",
    "CellAnalysisSummary",
    "ExponentialDecayFit",
    "LongitudinalRewardShiftColumn",
    "LongitudinalRewardShiftConfiguration",
    "LongitudinalRewardShiftReport",
    "LongitudinalRewardShiftSummary",
    "PeriodType",
    "PlaceFieldDetectionConfiguration",
    "PlaceFieldDetector",
    "PlaceFields",
    "RewardCellConfiguration",
    "RewardCellDetector",
    "RewardCellResults",
    "RewardRelativeColumn",
    "RewardRelativeConfiguration",
    "RewardRelativeReport",
    "RewardRelativeSummary",
    "RunSessionData",
    "SCEDetectionConfiguration",
    "SCEDetector",
    "SCEPeriodColumn",
    "SCEResult",
    "assemble_run_session_data",
    "bin_fluorescence_by_position",
    "compute_within_trial_position",
    "evaluate_and_save_cell_analysis",
    "plot_dataset_baseline_trend",
    "plot_dataset_cell_count",
    "plot_dataset_place_cell_fraction",
    "plot_dataset_reward_cell_fraction",
    "plot_dataset_sce_rate",
    "resolve_display_units",
    "run_bleaching_analysis",
]
