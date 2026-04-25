"""Provides cell analysis pipelines for place field, reward cell, and SCE detection."""

from .utilities import compute_track_length, compute_reward_position
from .sce_analysis import SCEResult, SCEDetector, SCEDetectionConfiguration
from .analysis_dataset import (
    append_sce_columns,
    append_reward_cell_columns,
    generate_analysis_dataframe,
    generate_place_field_dataframe,
)
from .place_cell_analysis import PlaceFields, PlaceFieldDetector, PlaceFieldDetectionConfiguration
from .reward_cell_analysis import RewardCellResults, RewardCellDetector, RewardCellConfiguration

__all__ = [
    "PlaceFieldDetectionConfiguration",
    "PlaceFieldDetector",
    "PlaceFields",
    "RewardCellConfiguration",
    "RewardCellDetector",
    "RewardCellResults",
    "SCEDetectionConfiguration",
    "SCEDetector",
    "SCEResult",
    "append_reward_cell_columns",
    "append_sce_columns",
    "compute_reward_position",
    "compute_track_length",
    "generate_analysis_dataframe",
    "generate_place_field_dataframe",
]
