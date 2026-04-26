"""Provides utility functions for computing track geometry and session metadata from feather files."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from ataraxis_base_utilities import console

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def compute_track_length(session_path: Path, trial_type: str) -> float:
    """Computes track length for a given trial type in a session feather file.

    Args:
        session_path: Path to the session feather file.
        trial_type: Trial type to compute track length for (e.g., "ABC", "ABCD").

    Returns:
        Estimated track length in centimeters (ceiling of the mean distance traveled across trials of that type).
    """
    dataframe = pl.read_ipc(source=session_path, columns=["system_state", "trial", "trial_type", "distance_cm"])
    running_dataframe = dataframe.filter((pl.col("system_state") == "run") & (pl.col("trial_type") == trial_type))

    # Computes the total distance traveled within each trial as the difference between the maximum and minimum
    # cumulative distance values.
    trial_stats = running_dataframe.group_by("trial").agg(
        pl.col("distance_cm").min().alias("minimum_distance"),
        pl.col("distance_cm").max().alias("maximum_distance"),
    )
    trial_stats = trial_stats.with_columns(
        (pl.col("maximum_distance") - pl.col("minimum_distance")).alias("distance_traveled")
    )

    mean_distance = trial_stats["distance_traveled"].mean()
    return float(math.ceil(mean_distance))


def compute_within_trial_position(
    distance: NDArray[np.float32],
    trial_ids: NDArray[np.int32],
) -> NDArray[np.float32]:
    """Computes within-trial position by subtracting each trial's starting distance.

    Notes:
        Global modulo (distance % track_length) drifts on circular tracks since lap lengths vary per trial. Computing
        distance relative to each trial's start position eliminates inter-trial drift.

    Args:
        distance: Cumulative distance in centimeters, with one value per frame.
        trial_ids: Trial identity for each frame, with one value per frame.

    Returns:
        Within-trial position in centimeters, with one value per frame.
    """
    position = np.empty(len(distance), dtype=np.float32)
    unique_trials = np.unique(trial_ids)

    for trial_id in unique_trials:
        trial_mask = trial_ids == trial_id
        trial_distance = distance[trial_mask]
        position[trial_mask] = trial_distance - trial_distance[0]

    return position


def compute_reward_position(session_path: Path, trial_type: str | None = None) -> float:
    """Computes the mean reward zone center position across trials.

    Notes:
        Estimates each trial's reward zone center as the midpoint between the first and last frame whose
        in_reward_zone flag is set, using within-trial distances, then averages the centers across trials.

    Args:
        session_path: Path to the session feather file.
        trial_type: Trial type to filter by. If None, includes all trial types.

    Returns:
        The mean reward zone center position in centimeters from trial start.
    """
    dataframe = pl.read_ipc(
        source=session_path,
        columns=["system_state", "trial_type", "distance_cm", "trial", "in_reward_zone"],
    )
    dataframe = dataframe.filter(pl.col("system_state") == "run")

    if trial_type is not None:
        dataframe = dataframe.filter(pl.col("trial_type") == trial_type)

    distance = dataframe["distance_cm"].to_numpy().astype(np.float32)
    trial_ids = dataframe["trial"].to_numpy().astype(np.int32)
    in_reward_zone = dataframe["in_reward_zone"].to_numpy().astype(np.uint8)

    position = compute_within_trial_position(distance=distance, trial_ids=trial_ids)
    unique_trials = np.unique(trial_ids)

    centers: list[float] = []
    for trial_id in unique_trials:
        trial_mask = trial_ids == trial_id
        trial_reward_zone = in_reward_zone[trial_mask]
        trial_position = position[trial_mask]

        in_zone_mask = trial_reward_zone == 1
        if np.any(in_zone_mask):
            # Estimates the center as the midpoint between the first and last reward zone frame positions.
            reward_zone_positions = trial_position[in_zone_mask]
            centers.append(float((reward_zone_positions[0] + reward_zone_positions[-1]) / 2.0))

    if not centers:
        message = (
            f"Unable to compute the reward zone center position from session feather file {session_path}. None of "
            f"the trials matching trial_type {trial_type} contain frames inside the reward zone."
        )
        console.error(message=message, error=ValueError)

    return float(np.mean(centers))
