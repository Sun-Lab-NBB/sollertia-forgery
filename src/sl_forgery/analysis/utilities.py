"""Provides utility functions for computing track geometry and session metadata from feather files."""

import math
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray


def compute_track_length(session_path: Path, trial_type: str) -> float:
    """Estimates track length for a given trial type in a session feather file.

    Args:
        session_path: Path to the session feather file.
        trial_type: Trial type to compute track length for (e.g. "ABC", "ABCD").

    Returns:
        Estimated track length in centimeters (ceiling of the mean distance traveled across trials of that type).
    """
    df = pl.read_ipc(session_path, columns=["system_state", "trial", "trial_type", "distance_cm"])
    run_df = df.filter((pl.col("system_state") == "run") & (pl.col("trial_type") == trial_type))

    # Computes the total distance traveled within each trial by taking the difference between the maximum and minimum
    # cumulative distance values.
    trial_stats = run_df.group_by("trial").agg(
        [
            pl.col("distance_cm").min().alias("min_dist"),
            pl.col("distance_cm").max().alias("max_dist"),
        ]
    )
    trial_stats = trial_stats.with_columns((pl.col("max_dist") - pl.col("min_dist")).alias("distance_traveled"))

    mean_distance = trial_stats["distance_traveled"].mean()
    return float(math.ceil(mean_distance))


def compute_within_trial_position(
    distance: NDArray[np.float64],
    trial_ids: NDArray[np.int32],
) -> NDArray[np.float32]:
    """Computes within-trial position by subtracting each trial's starting distance.

    Notes:
        Global modulo (distance % track_length) drifts on circular tracks since lap lengths vary per trial. Computing
        distance relative to each trial's start position eliminates inter-trial drift.

    Args:
        distance: Cumulative distance in centimeters with length frame_count.
        trial_ids: Trial identity for each frame with length frame_count.

    Returns:
        Within-trial position in centimeters with length frame_count.
    """
    position = np.empty(len(distance), dtype=np.float32)
    unique_trials = np.unique(trial_ids)

    for trial_id in unique_trials:
        trial_mask = trial_ids == trial_id
        trial_distance = distance[trial_mask]
        position[trial_mask] = (trial_distance - trial_distance[0]).astype(np.float32)

    return position


def compute_reward_position(session_path: Path, track_length: float, trial_type: str | None = None) -> float:
    """Computes the mean reward zone center position from the in_reward_zone column using within-trial distances.

    Args:
        session_path: Path to the session feather file.
        track_length: Length of the track in centimeters.
        trial_type: Trial type to filter by. If None, includes all trial types.

    Returns:
        The mean reward zone center position in centimeters from trial start.
    """
    df = pl.read_ipc(session_path, columns=["system_state", "trial_type", "distance_cm", "trial", "in_reward_zone"])
    df = df.filter(pl.col("system_state") == "run")

    if trial_type is not None:
        df = df.filter(pl.col("trial_type") == trial_type)

    distance = df["distance_cm"].to_numpy().astype(np.float64)
    trial_ids = df["trial"].to_numpy().astype(np.int32)
    in_reward_zone = df["in_reward_zone"].to_numpy().astype(np.uint8)

    position = compute_within_trial_position(distance=distance, trial_ids=trial_ids)

    # Computes the mean center of the reward zone across all trials.
    unique_trials = np.unique(trial_ids)

    centers = []
    for trial_id in unique_trials:
        trial_mask = trial_ids == trial_id
        trial_reward_zone = in_reward_zone[trial_mask]
        trial_position = position[trial_mask]

        if np.any(trial_reward_zone == 1):
            # Estimates the center as the midpoint between the first and last reward zone frame positions.
            reward_zone_positions = trial_position[trial_reward_zone == 1]
            centers.append(float((reward_zone_positions[0] + reward_zone_positions[-1]) / 2.0))

    return float(np.mean(centers))
