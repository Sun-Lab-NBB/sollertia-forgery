from typing import Any
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray


def behavior_to_numpy(
    source_file: Path,
) -> tuple[
    NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]
]:
    """Unpacks and returns the behavior data stored inside the target 'behavior_at_frame.feather' file as NumPy
    arrays.

    This temporary function is used to unpack .feather files into NumPy arrays to help developers with writing
    processing functions.

    Args:
        source_file: The absolute path to the source behavior_at_frame.feather file to parse into NumPy arrays.

    Returns:
        A tuple of 8 NumPy arrays: frame_index, timestamp, traveled_distance, trial, lick, reward, experiment_stage,
        system_state. All data is time-aligned to each mesoscope frame.
    """
    df = pl.read_ipc(source_file, use_pyarrow=True)

    frame_index = df["frame"].to_numpy()
    timestamps = df["frame_time_us"].to_numpy()
    traveled_distance = df["traveled_distance_cm"].to_numpy()
    trial = df["trial"].to_numpy()
    lick = df["lick_state"].to_numpy()
    reward = df["reward_state"].to_numpy()
    experiment_stage = df["experiment_stage"].to_numpy()
    system_state = df["system_state"].to_numpy()

    return frame_index, timestamps, traveled_distance, trial, lick, reward, experiment_stage, system_state


data = behavior_to_numpy(
    source_file=Path(
        "/Users/InfamousOne/Desktop/TM_06_pilot/6/2025-06-23-13-32-06-980761/behavior/behavior_at_frame.feather"
    )
)

print(data)