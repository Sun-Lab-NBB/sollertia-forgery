"""Provides cell analysis pipelines for place field, reward cell, and SCE detection."""

from .utilities import (
    RunSessionData,
    assemble_run_session_data,
    bin_fluorescence_by_position,
    compute_within_trial_position,
)
from .sce_analysis import SCEResult, SCEDetector, SCEDetectionConfiguration
from .analysis_dataset import (
    append_sce_columns,
    append_reward_cell_columns,
    generate_analysis_dataframe,
    generate_place_field_dataframe,
)
from .place_cell_analysis import PlaceFields, PlaceFieldDetector, PlaceFieldDetectionConfiguration
from .bleaching_evaluation import (
    BleachingColumn,
    BleachingReport,
    BleachingSummary,
    ExponentialDecayFit,
    BleachingConfiguration,
    evaluate_and_save_bleaching,
    plot_dataset_baseline_trend,
)
from .reward_cell_analysis import RewardCellResults, RewardCellDetector, RewardCellConfiguration

__all__ = [
    "BleachingColumn",
    "BleachingConfiguration",
    "BleachingReport",
    "BleachingSummary",
    "ExponentialDecayFit",
    "PlaceFieldDetectionConfiguration",
    "PlaceFieldDetector",
    "PlaceFields",
    "RewardCellConfiguration",
    "RewardCellDetector",
    "RewardCellResults",
    "RunSessionData",
    "SCEDetectionConfiguration",
    "SCEDetector",
    "SCEResult",
    "append_reward_cell_columns",
    "append_sce_columns",
    "assemble_run_session_data",
    "bin_fluorescence_by_position",
    "compute_within_trial_position",
    "evaluate_and_save_bleaching",
    "generate_analysis_dataframe",
    "generate_place_field_dataframe",
    "plot_dataset_baseline_trend",
]
