from typing import Any
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
from numpy.typing import NDArray
from matplotlib import pyplot as plt
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


def extract_data(filepath, day=None):
    '''
    Loads either single-day or multi-day data for the mouse

    Args:
        mouse: int, mouse ID
        date: format year-month-day ex: 2025-06-23
        day: "multi" or "single"

    Returns:
    processed data: fluorescence, neuropil, spikes, iscell
    '''

    #session_root = Path("/Users/cs963/Desktop/TM_06_pilot/{}/{}-13-32-06-980761/".format(mouse, date))
    # "/Users/cs963/Desktop/sun_lab_projects/26_explore/2025-09-16-18-44-32-476061"
    session_root = Path(filepath.joinpath("mesoscope_data/suite2p/combined/"))


    #Loads either single-day or multi-day data for the target session
    target_group = "single_day" # Supported values: single_day multi_day
    f_path = session_root.joinpath(target_group, "F.npy")
    f_neu_path = session_root.joinpath(target_group, "Fneu.npy")
    spks_path = session_root.joinpath(target_group, "spks.npy")
    iscell_path = session_root.joinpath(target_group, "iscell.npy")
    fluorescence = np.load(file=f_path, mmap_mode="r")
    neuropil = np.load(file=f_neu_path, mmap_mode="r")
    spks = np.load(file=spks_path, mmap_mode="r")
    iscell = np.load(file=iscell_path, mmap_mode="r")

    return fluorescence, neuropil, spks, iscell





####################




def behavior_from_feather(source_dir):
q
    """Temp function for testing new data with old plotting code"
    """
    source_dir = Path(source_dir.joinpath("behavior_data"))

    print(pl.read_ipc(source_dir.joinpath("mesoscope_frame_data.feather")))

    #frame_index = pl.read_ipc(source_dir.joinpath("mesoscope_frame_data.feather")).to_numpy()
    traveled_distance = pl.read_ipc(source_dir.joinpath("trial_data.feather"))["traveled_distance_cm"].to_numpy()
    frame_index = np.arange(traveled_distance.shape[0])
    timestamps = np.zeros(len(frame_index))
    trial = pl.read_ipc(source_dir.joinpath("trial_data.feather"))["trial_type_index"].to_numpy()
    lick = pl.read_ipc(source_dir.joinpath("lick_data.feather"))["lick_state"].to_numpy()
    reward = pl.read_ipc(source_dir.joinpath("valve_data.feather"))["dispensed_water_volume_uL"].to_numpy()
    experiment_state = pl.read_ipc(source_dir.joinpath("experiment_state_data.feather"))["experiment_state"].to_numpy()
    system_state = pl.read_ipc(source_dir.joinpath("system_state_data.feather"))["system_state"].to_numpy()

    return frame_index, timestamps, traveled_distance, trial, lick, reward, experiment_state, system_state


def extract_data_new(filepath):
    """again temporary function for testing new data with old plotting code"""
    session_root = Path(filepath)
    session_root = session_root.joinpath("mesoscope_data/suite2p/combined")

    #Loads either single-day or multi-day data for the target session

    f_path = session_root.joinpath("F.npy")
    f_neu_path = session_root.joinpath( "Fneu.npy")
    spks_path = session_root.joinpath( "spks.npy")
    iscell_path = session_root.joinpath("iscell.npy")
    fluorescence = np.load(file=f_path, mmap_mode="r")
    neuropil = np.load(file=f_neu_path, mmap_mode="r")
    spks = np.load(file=spks_path, mmap_mode="r")
    iscell = np.load(file=iscell_path, mmap_mode="r")

    return fluorescence, neuropil, spks, iscell