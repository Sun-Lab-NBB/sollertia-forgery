"""Provides the cell-tuning pipeline for place fields and reward cells across every trial type in a session."""

from .plotting import (
    plot_rate_map_heatmap,
    plot_per_trial_activity,
    plot_place_cell_heatmap,
    plot_reward_com_histogram,
    plot_speed_and_activity_by_position,
    plot_population_activity_by_position,
)
from .tuning_report import (
    TuningColumn,
    TuningReport,
    TuningSummary,
    TuningTrialSummary,
    TuningConfiguration,
    run_tuning_analysis,
)
from .place_tuning_protocol import PlaceFields, PlaceFieldDetector, PlaceFieldDetectionConfiguration
from .reward_tuning_protocol import RewardCellResults, RewardCellDetector, RewardCellConfiguration

__all__ = [
    "PlaceFieldDetectionConfiguration",
    "PlaceFieldDetector",
    "PlaceFields",
    "RewardCellConfiguration",
    "RewardCellDetector",
    "RewardCellResults",
    "TuningColumn",
    "TuningConfiguration",
    "TuningReport",
    "TuningSummary",
    "TuningTrialSummary",
    "plot_per_trial_activity",
    "plot_place_cell_heatmap",
    "plot_population_activity_by_position",
    "plot_rate_map_heatmap",
    "plot_reward_com_histogram",
    "plot_speed_and_activity_by_position",
    "run_tuning_analysis",
]
