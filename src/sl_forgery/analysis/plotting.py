from numba.cpython.unsafe.numbers import trailing_zeros
from sl_forgery.analysis.io import behavior_to_numpy, extract_data
from pathlib import Path

import numpy as np
import polars as pl
from matplotlib import pyplot as plt


# TODO - fix date variable and folder search

def plotting(mouse, kind):
    session_root = Path("/Users/cs963/Desktop/TM_06_pilot/6/2025-06-23-13-32-06-980761/")

    date = 1  # fix this

    #meso data is structured by cell# --> data; so shape is (cells, frames) - 2D array
    fluorescence, neuropil, spikes, iscell = extract_data(mouse, date, "single_day")


    #beh data is structured by frame --> so shape is (frames, ) 1D array
    frame_index, timestamps, traveled_distance, trial, lick, reward, experiment_stage, system_state = behavior_to_numpy(
        source_file=Path(session_root.joinpath("behavior", "behavior_at_frame.feather"))
    )

    # create polars dataframe indexed by frame with cells as columns
    fluorescence_df = pl.DataFrame(fluorescence.T, schema=[f"cell_{i}" for i in range(fluorescence.shape[0])])
    neuropil_df = pl.DataFrame(neuropil.T, schema=[f"cell_{i}" for i in range(neuropil.shape[0])])
    spikes_df = pl.DataFrame(spikes.T, schema=[f"cell_{i}" for i in range(spikes.shape[0])])

    iscell_df = pl.DataFrame(iscell)  # this has cells as indices and 2 columns, where 1st column is boolean value for
    # cell/not cell and 2nd is likelihood of being a cell
    # print(iscell_df.shape)
    iscell_df = iscell_df.with_row_index("cell_idx")  # add cell id index to cell df, 0-indexed to match F_df

    # polars dataframe indexed by frame with data as columns
    # 1 indexed
    behavior_df = pl.DataFrame(
        {"frame": frame_index,
         "timestamp": timestamps,
         "distance": traveled_distance,
         "trial": trial,
         "lick": lick,
         "reward": reward,
         "stage": experiment_stage,
         "state": system_state}
    )

    behavior_df = behavior_df.with_row_index("row_idx")  # add row index

    # 1st, choose only identified cells (currently suite2P is using 50% cutoff)
    # use column 1 i.e. boolean values
    cell_mask = iscell_df[:, 1].to_numpy()

    # only keep columns (cell data) for positive id cells  ->  cell_mask=True
    cell_fluorescence_df = fluorescence_df.select([fluorescence_df.columns[i] for i in np.where(cell_mask)[0]])
    # with np.printoptions(threshold=np.inf, suppress=True):
    #     print(cell_fluorescence_df)

    # then choose only frames where the system was in the active state i.e. mouse running
    active_state_mask = behavior_df["state"] == 2  # 2 is the active state (0 is idle, 1 is rest)

    # filter the dataframes by this active state
    active_behavior_df = behavior_df.filter(active_state_mask)
    active_fluorescence_df = fluorescence_df.filter(active_state_mask)
    print(active_fluorescence_df.shape)
    # TODO: 1. Check that the distance keeps increasing, otherwise there will be cell activity that is being compressed
    #  on the plot

    trial_start = active_behavior_df["trial"][0]
    # Find indices where the column value changes i.e. a new trial starts
    trial_start = active_behavior_df.filter(
        pl.col("trial") != pl.col("trial").shift(1)
    )
