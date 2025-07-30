<<<<<<< HEAD
from pathlib import Path

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from numba.cpython.unsafe.numbers import trailing_zeros

from sl_forgery.analysis.io import extract_data, behavior_to_numpy


# TODO: Type args and write doc string
#  next step is to continue binning the data
#   and create a column with cue identity
#   *this potentially doesnt need to be a separate function; ask ivan
#   actually it might be better if the binning function was outside the plotting function, maybe as separate modules
def create_grouped_dataframe_vectorized(distance_df, signal_df, start_indices):
    """
=======
from numba.cpython.unsafe.numbers import trailing_zeros
from sl_forgery.analysis.io import behavior_to_numpy
from pathlib import Path
from scipy import stats
from matplotlib import pyplot as plt

import numpy as np
import polars as pl


# TODO: Type args and write doc string
#  next step is to create a column with cue identity
#   *this potentially doesnt need to be a separate function; ask ivan
#   actually it might be better if the binning function was outside the plotting function, maybe as separate modules
def create_grouped_df(distance_df, signal_df, start_indices):
    '''
>>>>>>> origin/rate_maps

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

<<<<<<< HEAD
    """
    row_indices = distance_df["frame"]
    group_ids = np.searchsorted(start_indices, row_indices, side="right")
    print(row_indices, group_ids)

    # Add group_id column to distance dataframe
    distance_with_groups = distance_df.with_columns(pl.Series("group_id", group_ids))

    # Add group_id column to signal dataframe
    signal_with_groups = signal_df.with_columns(pl.Series("group_id", group_ids))

    # Create the grouped result for distances
    distance_grouped = (
        distance_with_groups.group_by("group_id")
        .agg([pl.col("frame").first().alias("start_index"), pl.col("distance").alias("distance_array")])
        .sort("group_id")
    )
=======
    '''
    row_indices = distance_df["frame"]
    group_ids = np.searchsorted(start_indices, row_indices, side='right')

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
>>>>>>> origin/rate_maps

    # Get the actual cell column names from the signal dataframe
    cell_columns = [col for col in signal_df.columns if col.startswith("cell_")]

    # Create the grouped result for signals
<<<<<<< HEAD
    signal_grouped = (
        signal_with_groups.group_by("group_id")
        .agg([*[pl.col(f"{col}").alias(f"{col}_signal") for col in cell_columns]])
        .sort("group_id")
    )
=======
    signal_grouped = signal_with_groups.group_by("group_id").agg([
        *[pl.col(f"{col}").alias(f"{col}_signal") for col in cell_columns]
    ]).sort("group_id")
>>>>>>> origin/rate_maps

    # Combine the results
    result = pl.concat([distance_grouped, signal_grouped.drop("group_id")], how="horizontal")

    return result

<<<<<<< HEAD
=======

>>>>>>> origin/rate_maps
    # TODO - fix date variable and folder search
    #   add arguments for cue length, bin size, day type for meso (single, multi)
    #   what are the other "kind" options

<<<<<<< HEAD

def plotting(mouse, kind):
    session_root = Path("/Users/cs963/Desktop/TM_06_pilot/6/2025-06-23-13-32-06-980761/")

    date = 1  # fix this

    # meso data is structured by cell# --> data; so shape is (cells, frames) - 2D array
    fluorescence, neuropil, spikes, iscell = extract_data(mouse, date, "single_day")

    # beh data is structured by frame --> so shape is (frames, ) 1D array
=======
def plotting(mouse, kind):
    path = "data/TM_06_pilot/6/2025-06-23-13-32-06-980761/"
    project_root = Path(__file__).resolve().parents[3]
    session_root = project_root / path

    target_group = "single_day"

    #beh data is structured by frame --> so shape is (frames, ) 1D array
>>>>>>> origin/rate_maps
    frame_index, timestamps, traveled_distance, trial, lick, reward, experiment_stage, system_state = behavior_to_numpy(
        source_file=Path(session_root.joinpath("behavior", "behavior_at_frame.feather"))
    )

<<<<<<< HEAD
    # TODO working on this as an outer function with df, optional filtering w keywords

    # create polars dataframe indexed by frame with cells as columns
=======
    #TODO working on this as an outer function with df, optional filtering w keywords

    # create polars dataframe indexed by frame with cells as columns
    f_path = session_root.joinpath(target_group, "F.npy")
    f_neu_path = session_root.joinpath(target_group, "Fneu.npy")
    spks_path = session_root.joinpath(target_group, "spks.npy")
    fluorescence = np.load(file=f_path, mmap_mode="r")
    neuropil = np.load(file=f_neu_path, mmap_mode="r")
    spikes = np.load(file=spks_path, mmap_mode="r")


>>>>>>> origin/rate_maps
    fluorescence_df = pl.DataFrame(fluorescence.T, schema=[f"cell_{i}" for i in range(fluorescence.shape[0])])
    neuropil_df = pl.DataFrame(neuropil.T, schema=[f"cell_{i}" for i in range(neuropil.shape[0])])
    spikes_df = pl.DataFrame(spikes.T, schema=[f"cell_{i}" for i in range(spikes.shape[0])])

<<<<<<< HEAD
=======

    iscell_path = session_root.joinpath(target_group, "iscell.npy")
    iscell = np.load(file=iscell_path, mmap_mode="r")

>>>>>>> origin/rate_maps
    iscell_df = pl.DataFrame(iscell)  # this has cells as indices and 2 columns, where 1st column is boolean value for
    # cell/not cell and 2nd is likelihood of being a cell

    iscell_df = iscell_df.with_row_index("cell_idx")  # add cell id index to cell df, 0-indexed to match F_df

    # polars dataframe indexed by frame with data as columns
    # 1 indexed
    behavior_df = pl.DataFrame(
<<<<<<< HEAD
        {
            "frame": frame_index,
            "timestamp": timestamps,
            "distance": traveled_distance,
            "trial": trial,
            "lick": lick,
            "reward": reward,
            "stage": experiment_stage,
            "state": system_state,
        }
    )

    behavior_df = behavior_df.with_row_index("row_idx")  # add row index

=======
        {"frame": frame_index,
         "timestamp": timestamps,
         "distance": traveled_distance,
         "trial": trial,
         "lick": lick,
         "reward": reward,
         "stage": experiment_stage,
         "state": system_state}
    )

>>>>>>> origin/rate_maps
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
<<<<<<< HEAD

    # TODO: 1. Check that the distance keeps increasing, otherwise there will be cell activity that is being compressed
    #  on the plot

    trial_start = active_behavior_df["trial"][0]
    # Find indices where the column value changes i.e. a new trial starts
    trial_start = active_behavior_df.filter(pl.col("trial") != pl.col("trial").shift(1))
    # TODO ^^^could also just "group_by" the trial value column; easier?

    # create 5 cm bins
    # TODO:  need to soft code bin size and cue length later

    track_length = np.mean(np.diff(trial_start["distance"]))
    cue_length = 30  # cm
    bin_size = 5  # cm
    n_bins = int(track_length / bin_size)  # here, 48 bins of 5 cm each

    # TODO this only works with set lengths
    # this wont work w my task, with variable track lengths
    track_length = np.mean(np.diff(trial_start["distance"]))
    print("trial start", trial_start)
    cue_length = 30  # cm
    bin_size = 5  # cm
    n_bins = int(track_length / bin_size)  # here, 48 bins of 5 cm each

    trial_indices = trial_start["frame"].to_numpy()

    result = create_grouped_dataframe_vectorized(
        active_behavior_df.select(active_behavior_df["frame", "distance"]), active_fluorescence_df, trial_indices
    )

    # TODO: use result df (trial bins), normalize the distances and separate into 5 cm bins
    #  take the average of each smaller bin, then concat them and take the average of the averages
    #   and plot
=======
    # print("active F df", active_fluorescence_df)


    # TODO: 1. Check that the distance keeps increasing, otherwise there will be cell activity that is being compressed
    #  on the plot.  change this to group by trial

    # Find indices where the column value changes i.e. a new trial starts
    trial_start = active_behavior_df.filter(
        pl.col("trial") != pl.col("trial").shift(1)
    )
    #TODO ^^^could also just "group_by" the trial value column; easier?

    trial_indices = trial_start["frame"].to_numpy()

    result = create_grouped_df(active_behavior_df.select(active_behavior_df["frame", "distance"]),
                                                active_fluorescence_df,
                                                trial_indices)

    # create 5 cm bins
    # TODO:  need to soft code bin size and cue length late
    #   this only works with set lengths
    # this wont work w my task, with variable track lengths
    track_length = np.mean(np.diff(trial_start["distance"]))
    cue_length = 30  # cm
    bin_size = 5  # cm
    n_bins = int(track_length / bin_size)  # here, 48 bins of 5 cm each


>>>>>>> origin/rate_maps

    # normalize arrays
    normalized_arrays = []
    for dist in result["distance_array"]:
        arr = np.array(dist)

        # Normalize
        min_val = arr.min()
        max_val = arr.max()

        if max_val - min_val == 0:
            normalized = np.zeros_like(arr)
        else:
            normalized = 240 * (arr - min_val) / (max_val - min_val)

        normalized_arrays.append(np.floor(normalized))

<<<<<<< HEAD
    # TODO bin these new arrays
=======
>>>>>>> origin/rate_maps

    # bin the normalized arrays

    bin_edges = np.arange(0, 245, 5)  # [0, 5, 10, ..., 240]  --> again soft code for track_length + bin_size
    num_trials = len(normalized_arrays)
<<<<<<< HEAD
    binned_arrays = np.empty((num_trials, 48), dtype=object)  # arrays of binned distance arrays for each trial (i.e. N
    # trial arrays, each with 48 bins of
    # 5cm distances); make the 48 softcoded
    bin_assignments = np.empty(num_trials, dtype=object)  # indexes of bins to use for cell activity
=======
    binned_arrays = np.empty((num_trials, 48), dtype=object)   #arrays of binned distance arrays for each trial (i.e. N
    # trial arrays, each with 48 bins of
    # 5cm distances); make the 48 softcoded
    bin_assignments = np.empty(num_trials, dtype=object)   # indexes of bins to use for cell activity
>>>>>>> origin/rate_maps

    for e, arr in enumerate(normalized_arrays):
        # get the indices of the bins to which each value belongs in an array; use np.digitize
        bin_indices = np.digitize(arr, bin_edges, right=False) - 1
        # Handle values exactly equal to 240 (put in last bin)
        bin_indices = np.where(arr == 240, 47, bin_indices)
<<<<<<< HEAD
        print(bin_indices)
        bin_assignments[e] = bin_indices  # use these in future df to split up cell activity

        # Create the 5 cm arrays for each bin
=======
        bin_assignments[e] = bin_indices #use these in future df to split up cell activity


    # Create the 5 cm arrays for each bin
>>>>>>> origin/rate_maps
        for i in range(48):
            mask = bin_indices == i
            bin_values = arr[mask]
            binned_arrays[e, i] = bin_values

<<<<<<< HEAD
    # TODO -- not sure if binned_df is necessary; make reduced df from start?  OR skip all together and just use as a
    # series
    #
    # create a df with the bin idx; not sure if this is actually necessary
    binned_df = result.with_columns(pl.Series("bin_assignments", bin_assignments))
    print(binned_df)

    reduced_df = binned_df.drop("group_id", "start_index", "distance_array")
    print(reduced_df)

    # for x in binned_arrays[:]:
    #     for i in x:
    #         print(np.mean(i))

    # CREATE NEW DF
    binned_df = result.with_columns(pl.Series("bin_assignments", bin_assignments))

    reduced_df = binned_df.drop("group_id", "start_index", "distance_array")
    index_col = "bin assignments"
=======

    #TODO -- not sure if binned_df is necessary; make reduced df from start?  OR skip all together and just use as a
    # series

    # CREATE NEW DF - bin the trials into 5 cm bins, and average each bin for signal along position
    binned_df = result.with_columns(pl.Series("bin_assignments", bin_assignments))

    reduced_df = binned_df.drop("group_id", "start_index", "distance_array")
    index_col = "bin_assignments"
>>>>>>> origin/rate_maps

    signal_columns = [col for col in reduced_df.columns if col != index_col]

    # convert entire dataframe to numpy
    data_dict = reduced_df.to_dict(as_series=False)

    # create dict for trial avgs
    trial_avgs = {}

    max_bins = n_bins  # this was calculated earlier
    for col in signal_columns:  # for each cell
<<<<<<< HEAD
        col_results = []

        # process all rows for this column
        for row_idx in range(len(df)):
=======

        col_results = []

        # process all rows for this column
        for row_idx in range(len(reduced_df)):
>>>>>>> origin/rate_maps
            signal_array = np.array(data_dict[col][row_idx], dtype=np.float64)  # signal for that col/row
            index_array = np.array(data_dict[index_col][row_idx], dtype=np.int32)  # index for that trial (bins)

            # use numpy binning
            # np.bin_count counts all the values in the bin and sums them
            bin_sums = np.bincount(index_array, weights=signal_array, minlength=max_bins)  # sum the signals
            bin_counts = np.bincount(index_array, minlength=max_bins)  # find the length of the bin

            # calculate averages by dividing signal sum by bin length
<<<<<<< HEAD
            bin_averages = np.divide(bin_sums, bin_counts, out=np.full_like(bin_sums, np.nan), where=bin_counts != 0)

            col_results.append(bin_averages)

        trial_avgs[f"{col}_binned"] = col_results
=======
            bin_averages = np.divide(bin_sums, bin_counts,
                                     out=np.full_like(bin_sums, np.nan),
                                     where=bin_counts != 0)

            col_results.append(bin_averages)

        trial_avgs[f'{col}_binned'] = col_results
>>>>>>> origin/rate_maps

    trial_avg_df = pl.DataFrame(trial_avgs)

    # now create dict for the average signal for each cell in the session
    avg_data = {}
<<<<<<< HEAD
    sess_sem = []  # had to make list bc I couldnt get both arrays into a single cell, there was some issue with
=======
    sess_sem = []       #had to make list bc I couldnt get both arrays into a single cell, there was some issue with
>>>>>>> origin/rate_maps
    # polars.  Should try to use polars arrays instead of numpy arrays, or just use arrays outside df

    for col in trial_avg_df.columns:
        # stack all arrays and compute mean for each index
        stacked_arrays = np.array(trial_avg_df[col].to_list())
        avg_array = np.mean(stacked_arrays, axis=0)
        session_sem = np.array(stats.sem(stacked_arrays, axis=0))  # find the standard error
        avg_data[col] = [avg_array]  # , sessoin_sem  # save as a 2 element array which can be
        # accessed later by indexing
        sess_sem.append(session_sem)

    # create new row and add it to the bottom of the df
    session_avg_row = pl.DataFrame(avg_data)

    # again - i was having an issue getting these to stay as arrays when I put htem in the df
    # session_avg_row = session_avg_row.with_columns([
    #     pl.col(col).cast(pl.Array(pl.Float64, 48)) for col in session_avg_row.columns
    # ])
    session_avg_df = pl.concat([trial_avg_df, session_avg_row])

    # %%%%%%%%%%%%%%%%%%
    # TODO normalize F --> F - .7Fneu for y axis OR z-score;  extract cue;  add option for single day or multi day
    #  plotting;  integrate with plotly when jacob is done;  plot cue regions under the graph; basically thick little
    #  vlines of different colors

    cells = range(5)

    for cell in cells:
<<<<<<< HEAD
        xaxis = np.arange(2.5, 240, 5)  # include 240 in the plot
        print(xaxis.shape)
        fig, ax = plt.subplots()

        cell_val = session_avg_df["cell_{}_signal_binned".format(cell)][-1]  # selects the last row of the col,
=======
        xaxis = np.arange(2.5, 240, 5)  # # 5 cm bins, plot the avg signal in center of bin
        fig, ax = plt.subplots()

        cell_val = session_avg_df['cell_{}_signal_binned'.format(cell)][-1]  # selects the last row of the col,
>>>>>>> origin/rate_maps
        # which has the avg session data

        mean = cell_val.to_numpy()  # , cell_val[1].to_numpy()    #extract mean array and sem array; again issue with
        # pulling ndarrays from polars df
        sem = sess_sem[0]
<<<<<<< HEAD
        print(mean, sem)
        ax.plot(xaxis, mean)
        plt.fill_between(xaxis, mean - sem, mean + sem, color="blue", alpha=0.2, label="Mean +/- SEM")
=======
        ax.plot(xaxis, mean)
        plt.fill_between(xaxis, mean - sem, mean + sem,
                         color='blue', alpha=0.2, label='Mean +/- SEM')
>>>>>>> origin/rate_maps

        plt.title("cell {} session avg day5".format(cell))
        plt.xlabel("distance in cm")
        plt.ylabel("Fluorescent signal")

        # plot trial avgs
        fig, ax = plt.subplots()

        for i in range(result.shape[0]):
            ax.plot(xaxis, trial_avg_df[i, cell], label=f"{i}")
        # if kind == "place":
        #     for i in range(result.shape[0]):
        #         ax.plot(normalized_arrays[i], result["cell_1_signal"][i])

        # ax.vlines(trial_start["distance"][:10], ymin=-150, ymax=0, color='r', lw=2)

        # TODO make this cycle through a random subset of cells when calling the function and pull the column names for
        # cell ID

        # Hide the x-tick labels
        plt.title("cell {} trial avgs day5".format(cell))
        plt.xlabel("distance in cm")
        plt.ylabel("Fluorescent signal")
        plt.show()


plotting(mouse="6", kind="place")
