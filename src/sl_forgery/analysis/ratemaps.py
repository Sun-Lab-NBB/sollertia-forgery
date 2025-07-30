from pathlib import Path

import numpy as np
import polars as pl

from numpy.typing import NDArray
from typing import Any

from plotly.subplots import make_subplots
from plotly import express as px
from plotly import graph_objects as go

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


path = "data/TM_06_pilot/6/2025-06-23-13-32-06-980761/"

<<<<<<< HEAD

=======
>>>>>>> origin/rate_maps
project_root = Path(__file__).resolve().parents[3]
session_root = project_root / path


frame_index, timestamps, traveled_distance, trial, lick, reward, experiment_stage, system_state = behavior_to_numpy(
    source_file=Path(session_root.joinpath("behavior", "behavior_at_frame.feather"))
)

target_group = "single_day"
f_path = session_root.joinpath(target_group, "F.npy")
f_neu_path = session_root.joinpath(target_group, "Fneu.npy")
spks_path = session_root.joinpath(target_group, "spks.npy")


fluorescence = np.load(file=f_path, mmap_mode="r")
neuropil = np.load(file=f_neu_path, mmap_mode="r")
spks = np.load(file=spks_path, mmap_mode="r")


track_length = 240
bin_size = 5
n_bins = track_length // bin_size



gray =[27]

cue1 = [30, 31, 38]
gray1 = []
cue2 = [14, 17]
gray2 = [40]
cue3 = [12, 45]
gray3 = [7, 39] 
cue4 = [8, 16, 36, 37, 44]
gray4 = [4, 6]


reward_cells = [4, 6]
# 4 is reward cell


for cell in range(50):
    binned_spikes = [[] for _ in range(n_bins)]

    # Bin frames
    for i in range(len(frame_index)):
        track_pos = traveled_distance[i] % track_length
        bin = int(track_pos // bin_size)
        binned_spikes[bin].append(spks[cell][i])

    # Convert bins to arrays
    for i in range(n_bins):
        binned_spikes[i] = np.array(binned_spikes[i])

    mean_activity = np.array([bin.mean() for bin in binned_spikes])

    distances = np.array(range(0, track_length, bin_size)) + bin_size / 2

    cue_length = 30
    cue_changes = np.array(range(0, track_length, cue_length))

    fig = go.Figure()
    for x in cue_changes:
        fig.add_shape(
            type="line",
            x0=x, x1=x,
            y0=min(mean_activity), y1=max(mean_activity),
            line=dict(color="black", width=1,),
        )
    # Plot the initial session
    fig.add_trace(go.Scatter(
        x=distances,
        y=mean_activity,
        mode='lines',
        line=dict(width=3),
    ))

    fig.update_layout(title=f"Cell {cell}")

    fig.show()