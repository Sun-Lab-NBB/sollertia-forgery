from typing import Any
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
from matplotlib import pyplot as plt
from numpy.typing import NDArray

matplotlib.use("QtAgg")

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


def test_plot(f_cells, f_neuropils, spks, roi_index) -> None:
    plt.figure(figsize=(20.0, 20.0), dpi=100)
    plt.suptitle(f"Fluorescence and Deconvolved Traces for ROI {roi_index} Across Sessions", y=0.92)

    # Assigns distinct color to visualized traces
    colors = ["#1f77b4", "#2ca02c", "#d62728"]  # Blue, Green, Red

    frame_in = 6400
    frame_out = 7500
    cell_start = 500
    cell_stop = 510

    # Extracts data for the specific ROI from this session
    # f_neu = f_neuropils[roi_index][frame_in:frame_out]
    # sp = spks[roi_index][frame_in:frame_out]

    # # Adjust range to match fluorescence traces
    # fmax = np.maximum(f.max(), f_neu.max())
    # fmin = np.minimum(f.min(), f_neu.min())
    # frange = fmax - fmin

    # # Normalizes spikes
    # if sp.max() > 0:
    #     sp = sp / sp.max() * frange + fmin
    # else:
    #     sp = np.zeros_like(sp) + fmin

    for ind in range(500, 510, 1):
        f = f_cells[ind][frame_in:frame_out]
        plt.plot(f, label=f"cell_{ind}")

    # plt.plot(f_neu, color=colors[1], label="Neuropil Fluorescence")
    # plt.plot(sp, color=colors[2], label="Deconvolved")

    plt.xticks(np.arange(0, f.shape[0], f.shape[0] // 10))

    # Add y-axis label for fluorescence/pixel intensity
    plt.ylabel("fluorescence")

    plt.xlabel("frame")
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


# Path to the target session
session_root = Path("/Users/InfamousOne/Desktop/TM_06_pilot/6/2025-06-23-13-32-06-980761/")

# Parses behavior data as one-dimensional NumPy arrays
frame_index, timestamps, traveled_distance, trial, lick, reward, experiment_stage, system_state = behavior_to_numpy(
    source_file=Path(session_root.joinpath("behavior", "behavior_at_frame.feather"))
)

# Loads either single-day or multi-day data for the target session
target_group = "multi_day"  # Supported values: single_day multi_day
f_path = session_root.joinpath(target_group, "F.npy")
f_neu_path = session_root.joinpath(target_group, "Fneu.npy")
spks_path = session_root.joinpath(target_group, "spks.npy")
fluorescence = np.load(file=f_path, mmap_mode="r")
neuropil = np.load(file=f_neu_path, mmap_mode="r")
spks = np.load(file=spks_path, mmap_mode="r")

test_plot(f_cells=fluorescence, f_neuropils=neuropil, spks=spks, roi_index=500)
