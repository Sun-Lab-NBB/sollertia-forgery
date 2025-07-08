from numba.cpython.unsafe.numbers import trailing_zeros
from sl_forgery.analysis.io import behavior_to_numpy, extract_data
from pathlib import Path

import numpy as np
import polars as pl
from matplotlib import pyplot as plt


# TODO: Type args and write doc string
#  next step is to continue binning the data
#   and create a column with cue identity
#   *this potentially doesnt need to be a separate function; ask ivan
def create_grouped_dataframe_vectorized(distance_df, signal_df, start_indices):
    '''

    Args:
        distance_df: behavior dataframe distance column
        signal_df: signal (F, neuropil, etc) dataframe
        start_indices: trial start indices/frames

    Returns:
    A new dataframe that is grouped by trial identity, where the 1st col is trial,
    the 2nd column are arrays of the distance covered during that trial,
    and the following columns are arrays of the recorded signals for each cell during that trial.
    The distance and signal arrays in each row are of the same length. These can be used for plotting signal over
    entire trials.

    '''
    row_indices = distance_df["frame"]
    group_ids = np.searchsorted(start_indices, row_indices, side='right')
    print(row_indices, group_ids)

    # Add group_id column to distance dataframe
    distance_with_groups = distance_df.with_columns(
        pl.Series("group_id", group_ids)
    )

    # Add group_id column to signal dataframe
    signal_with_groups = signal_df.with_columns(
        pl.Series("group_id", group_ids)
    )

    # Create the grouped result for distances
    distance_grouped = distance_with_groups.group_by("group_id").agg([
        pl.col("frame").first().alias("start_index"),
        pl.col("distance").alias("distance_array")]
    ).sort("group_id")

    # Get the actual cell column names from the signal dataframe
    cell_columns = [col for col in signal_df.columns if col.startswith("cell_")]

    # Create the grouped result for signals
    signal_grouped = signal_with_groups.group_by("group_id").agg([
        *[pl.col(f"{col}").alias(f"{col}_signal") for col in cell_columns]
    ]).sort("group_id")

    # Combine the results
    result = pl.concat([distance_grouped, signal_grouped.drop("group_id")], how="horizontal")

    return result


    # TODO - fix date variable and folder search
    #   add arguments for cue length, bin size, day type for meso (single, multi)
    #   what are the other "kind" options

def plotting(mouse, kind):
    session_root = Path("/Users/cs963/Desktop/TM_06_pilot/6/2025-06-23-13-32-06-980761/")

    date = 1  # fix this

    #meso data is structured by cell# --> data; so shape is (cells, frames) - 2D array
    fluorescence, neuropil, spikes, iscell = extract_data(mouse, date, "single_day")


    #beh data is structured by frame --> so shape is (frames, ) 1D array
    frame_index, timestamps, traveled_distance, trial, lick, reward, experiment_stage, system_state = behavior_to_numpy(
        source_file=Path(session_root.joinpath("behavior", "behavior_at_frame.feather"))
    )

#TODO working on this as an outer function with df, optional filtering w keywords

    # create polars dataframe indexed by frame with cells as columns
    fluorescence_df = pl.DataFrame(fluorescence.T, schema=[f"cell_{i}" for i in range(fluorescence.shape[0])])
    neuropil_df = pl.DataFrame(neuropil.T, schema=[f"cell_{i}" for i in range(neuropil.shape[0])])
    spikes_df = pl.DataFrame(spikes.T, schema=[f"cell_{i}" for i in range(spikes.shape[0])])

    iscell_df = pl.DataFrame(iscell)  # this has cells as indices and 2 columns, where 1st column is boolean value for
    # cell/not cell and 2nd is likelihood of being a cell

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
    # np.where returns a tuple containing a numpy array with the indices ([idx], )
    cell_fluorescence_df = fluorescence_df.select([fluorescence_df.columns[i] for i in np.where(cell_mask)[0]])

    # then choose only frames where the system was in the active state i.e. mouse running
    active_state_mask = behavior_df["state"] == 2  # 2 is the active state (0 is idle, 1 is rest)

    # filter the dataframes by this active state
    active_behavior_df = behavior_df.filter(active_state_mask)
    active_fluorescence_df = cell_fluorescence_df.filter(active_state_mask)


    # TODO: 1. Check that the distance keeps increasing, otherwise there will be cell activity that is being compressed
    #  on the plot

    trial_start = active_behavior_df["trial"][0]
    # Find indices where the column value changes i.e. a new trial starts
    trial_start = active_behavior_df.filter(
        pl.col("trial") != pl.col("trial").shift(1)
    )

    #create 5 cm bins
    #TODO:  need to soft code bin size and cue length later

    track_length = np.mean(np.diff(trial_start["distance"]))
    cue_length = 30  # cm
    bin_size = 5  # cm
    n_bins = int(track_length / bin_size)  # here, 48 bins of 5 cm each


    #TODO this only works with set lengths
    # this wont work w my task, with variable track lengths
    track_length = np.mean(np.diff(trial_start["distance"]))
    print("trial start", trial_start)
    cue_length = 30  # cm
    bin_size = 5  # cm
    n_bins = int(track_length / bin_size)  # here, 48 bins of 5 cm each

    trial_indices = trial_start["frame"].to_numpy()


    result = create_grouped_dataframe_vectorized(active_behavior_df.select(active_behavior_df["frame", "distance"]),
                                                active_fluorescence_df,
                                                trial_indices)

