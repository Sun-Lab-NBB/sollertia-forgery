"""Provides cell-tuning pipelines for place fields, reward cells, SCEs, and longitudinal shifts."""

from .sce_protocol import SCEResult, SCEDetector, SCEDetectionConfiguration
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
    "CellAnalysisColumn",
    "CellAnalysisConfiguration",
    "CellAnalysisReport",
    "CellAnalysisSummary",
    "LongitudinalRewardShiftColumn",
    "LongitudinalRewardShiftConfiguration",
    "LongitudinalRewardShiftReport",
    "LongitudinalRewardShiftSummary",
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
    "SCEDetectionConfiguration",
    "SCEDetector",
    "SCEPeriodColumn",
    "SCEResult",
    "evaluate_and_save_cell_analysis",
    "plot_dataset_cell_count",
    "plot_dataset_place_cell_fraction",
    "plot_dataset_reward_cell_fraction",
    "plot_dataset_sce_rate",
]
