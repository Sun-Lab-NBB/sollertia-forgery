"""Utility functions for loading and preprocessing session data for analysis."""

import math
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray


def _load_run_df(session_path: Path, trial_type: str | None = None) -> pl.DataFrame:
    """Loads a session feather file and filters to 'run' state frames.

    Args:
        session_path: Path to the session feather file.
        trial_type: Trial type to filter by (e.g. "ABC", "ABCD"). If None, includes all trial types.

    Returns:
        A polars DataFrame containing only 'run' state frames, optionally filtered to a single trial type.
    """
    df = pl.read_ipc(session_path)

    if trial_type is not None:
        df = df.filter(pl.col("trial_type") == trial_type)

    return df.filter(pl.col("system_state") == "run")


def compute_track_lengths(session_path: Path) -> dict[str, float]:
    """Estimates track length for each trial type in a session feather file.

    Args:
        session_path: Path to the session feather file.

    Returns:
        A dictionary mapping each trial type string to its estimated track length in centimeters (ceiling of the
        mean distance traveled across trials of that type).
    """
    run_df = _load_run_df(session_path=session_path)

    # Computes the total distance traveled within each trial by taking the difference between the maximum and minimum
    # cumulative distance values. Groups by both 'trial' and 'trial_type' so the trial type label is preserved.
    trial_stats = run_df.group_by("trial", "trial_type").agg(
        [
            pl.col("distance_cm").min().alias("min_dist"),
            pl.col("distance_cm").max().alias("max_dist"),
        ]
    )
    trial_stats = trial_stats.with_columns((pl.col("max_dist") - pl.col("min_dist")).alias("distance_traveled"))

    # Groups by trial type and takes the average of the per-trial distances, then applies the ceiling function to
    # round up to a whole-number track length.
    type_stats = trial_stats.group_by("trial_type").agg(pl.col("distance_traveled").mean().alias("mean_distance"))
    type_stats = type_stats.sort("trial_type")

    track_lengths: dict[str, float] = {}
    for row in type_stats.iter_rows(named=True):
        trial_type: str = row["trial_type"]
        mean_distance: float = row["mean_distance"]
        track_lengths[trial_type] = float(math.ceil(mean_distance))

    return track_lengths


def load_place_field_data(
    session_path: Path,
    track_length: float,
    fluorescence_column: str = "single_day_dff",
    trial_type: str | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Loads fluorescence, position, and speed data from a session feather file for place field analysis.

    Args:
        session_path: Path to the session feather file.
        track_length: Length of the track in centimeters. Used to convert cumulative distance to track position via
            modulo.
        fluorescence_column: Name of the fluorescence column to use.
        trial_type: Trial type to filter by (e.g. "ABC", "ABCD"). If None, includes all trial types.

    Returns:
        A tuple of (fluorescence, position, speed) where fluorescence has shape (cell_count, frame_count),
        and position and speed each have shape (frame_count,).
    """
    df = _load_run_df(session_path=session_path, trial_type=trial_type)

    # Extracts fluorescence data and transposes from (frame, cell) to (cell, frame) for the detector.
    fluorescence = np.vstack(df[fluorescence_column].to_list()).T.astype(np.float32)

    # Converts the cumulative distance to track position using modulus to ensure the position wraps within a single lap.
    position = df["distance_cm"].to_numpy().astype(np.float32) % track_length
    speed = df["speed_cm_s"].to_numpy().astype(np.float32)

    return fluorescence, position, speed
