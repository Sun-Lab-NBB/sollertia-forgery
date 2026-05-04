"""Provides the cell-tuning pipeline for place fields and reward cells across every trial type in a session."""

from .plotting import (
    plot_sorted_heatmap,
    plot_classified_heatmap,
    plot_per_trial_activity,
    plot_reward_com_histogram,
    plot_cue_pair_place_counts,
    plot_speed_and_activity_by_position,
    plot_population_activity_by_position,
    plot_place_cell_peak_distribution_per_session,
    plot_place_cell_peak_distribution_around_shift,
    plot_place_cell_peak_distribution_across_animals,
    plot_post_shift_reward_zone_cells_in_session,
    plot_pre_shift_reward_zone_cells_in_session,
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
    "plot_classified_heatmap",
    "plot_cue_pair_place_counts",
    "plot_per_trial_activity",
    "plot_place_cell_peak_distribution_across_animals",
    "plot_place_cell_peak_distribution_around_shift",
    "plot_place_cell_peak_distribution_per_session",
    "plot_population_activity_by_position",
    "plot_post_shift_reward_zone_cells_in_session",
    "plot_pre_shift_reward_zone_cells_in_session",
    "plot_reward_com_histogram",
    "plot_sorted_heatmap",
    "plot_speed_and_activity_by_position",
    "run_tuning_analysis",
]
