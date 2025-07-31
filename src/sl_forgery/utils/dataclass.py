import polars as pl
import numpy as np
from scipy import stats

from typing import Any
from numpy.typing import NDArray

from pathlib import Path


class Data:
    def __init__(self, root):
        self.root = Path(root)
        

    @staticmethod
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
    
    def find_mouse(self, mouse):
        """
        Return the path to a mouse directory.

        Args:
            mouse (int | str | pathlib.Path): Mouse number or path
        Raises:
            Exception: If the directory does not exist.
        """
        if isinstance(mouse, int):
            path = self.root / f"TM_{mouse:02d}_pilot/{mouse}"
        else:
            path = self.root / mouse
        if path.exists():
            return path
        else:
            raise Exception(f"Can't find mouse at {path}")   

    def find_session(self, mouse, session):
        """
        Return the path to a session for a given mouse.

        Args:
            mouse (str): Mouse identifier.
            session (int | str): Session number (0-indexed) or session name.
        Raises:
            Exception: If the given session name does not exist.
        """        
        mouse_root = self.find_mouse(mouse)
        if isinstance(session, int):
            path = sorted([f for f in mouse_root.iterdir() if f.is_dir()])[session]
        else:
            path = mouse_root / session
            if path.exists():
                return path
            else:
                raise Exception(f"Can't find session at {path}")

    def get_num_sessions(self, mouse):
        """
        Return the number of session directories for a given mouse.

        Args:
            mouse (int | str): Mouse index or name.

        Returns:
            int: Number of session directories.
        """
        mouse_root = self.find_mouse(mouse)
        return len([f for f in mouse_root.iterdir() if f.is_dir()])
        

    def get_all_data(self, mouse, session, target_group):
        """
        Return all session data as polars dataframes.
        
        Args:
            mouse (str): Mouse identifier.
            session (int | str): Session number (0-indexed) or session name.
            target_group (str): "single_day" or "multi_day"
        Returns:
            A tuple of 5 Polars dataframes: behavior_df, fluorescence_df, neuropil_df, spikes_df, iscell_df.     
        """

        session_root = self.find_session(mouse, session)

        target_group = "single_day"

        behavior_df = pl.read_ipc(session_root.joinpath("behavior", "behavior_at_frame.feather"), use_pyarrow=True)


        #TODO working on this as an outer function with df, optional filtering w keywords

        # create polars dataframe indexed by frame with cells as columns
        f_path = session_root.joinpath(target_group, "F.npy")
        f_neu_path = session_root.joinpath(target_group, "Fneu.npy")
        spks_path = session_root.joinpath(target_group, "spks.npy")
        fluorescence = np.load(file=f_path, mmap_mode="r")
        neuropil = np.load(file=f_neu_path, mmap_mode="r")
        spikes = np.load(file=spks_path, mmap_mode="r")


        fluorescence_df = pl.DataFrame(fluorescence.T, schema=[f"cell_{i}" for i in range(fluorescence.shape[0])])
        neuropil_df = pl.DataFrame(neuropil.T, schema=[f"cell_{i}" for i in range(neuropil.shape[0])])
        spikes_df = pl.DataFrame(spikes.T, schema=[f"cell_{i}" for i in range(spikes.shape[0])])


        iscell_path = session_root.joinpath(target_group, "iscell.npy")
        iscell = np.load(file=iscell_path, mmap_mode="r")

        iscell_df = pl.DataFrame(iscell)  # this has cells as indices and 2 columns, where 1st column is boolean value for
        # cell/not cell and 2nd is likelihood of being a cell

        iscell_df = iscell_df.with_row_index("cell_idx")  # add cell id index to cell df, 0-indexed to match F_df
        
        return behavior_df, fluorescence_df, neuropil_df, spikes_df, iscell_df

        # TODO: Type args and write doc string
    #  next step is to create a column with cue identity
    #   *this potentially doesnt need to be a separate function; ask ivan
    #   actually it might be better if the binning function was outside the plotting function, maybe as separate modules
    @staticmethod
    def create_grouped_df(distance_df, signal_df, start_indices):
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
            pl.col("traveled_distance_cm").alias("distance_array")]
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


    def process_data(self, mouse, session, target_group):
        """
        Return 
        
        Args:
            mouse (str): Mouse identifier.
            session (int | str): Session number (0-indexed) or session name.
            target_group (str): "single_day" or "multi_day"
        Returns:

        Notes:
            This function should ultimately be split into many smaller functions. It contains the brunt of Chelsea's 
            original code for plotting place fields.
        """
        print("getting data")
        behavior_df, fluorescence_df, neuropil_df, spikes_df, iscell_df = self.get_all_data(mouse, session, target_group)
        print("got data")


        print("grouping data")
        # 1st, choose only identified cells (currently suite2P is using 50% cutoff)
        # use column 1 i.e. boolean values
        cell_mask = iscell_df[:, 1].to_numpy()

        # only keep columns (cell data) for positive id cells  ->  cell_mask=True
        # np.where returns a tuple containing a numpy array with the indices ([idx], )
        cell_fluorescence_df = fluorescence_df.select([fluorescence_df.columns[i] for i in np.where(cell_mask)[0]])

        # then choose only frames where the system was in the active state i.e. mouse running
        active_state_mask = behavior_df["system_state"] == 2  # 2 is the active state (0 is idle, 1 is rest)

        # filter the dataframes by this active state
        active_behavior_df = behavior_df.filter(active_state_mask)
        active_fluorescence_df = cell_fluorescence_df.filter(active_state_mask)
        # print("active F df", active_fluorescence_df)


        # TODO: 1. Check that the distance keeps increasing, otherwise there will be cell activity that is being compressed
        #  on the plot.  change this to group by trial

        # Find indices where the column value changes i.e. a new trial starts
        trial_start = active_behavior_df.filter(
            pl.col("trial") != pl.col("trial").shift(1)
        )
        #TODO ^^^could also just "group_by" the trial value column; easier?

        trial_indices = trial_start["frame"].to_numpy()

        result = Data.create_grouped_df(active_behavior_df.select(active_behavior_df["frame", "traveled_distance_cm"]),
                                                    active_fluorescence_df,
                                                    trial_indices)

        print("grouped data")
        print("normalizing data")

        # create 5 cm bins
        # TODO:  need to soft code bin size and cue length late
        #   this only works with set lengths
        # this wont work w my task, with variable track lengths
        track_length = np.mean(np.diff(trial_start["traveled_distance_cm"]))
        cue_length = 30  # cm
        bin_size = 5  # cm
        n_bins = int(track_length / bin_size)  # here, 48 bins of 5 cm each



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

        print("normalized data")
        print("binning data")

        # bin the normalized arrays

        bin_edges = np.arange(0, 245, 5)  # [0, 5, 10, ..., 240]  --> again soft code for track_length + bin_size
        num_trials = len(normalized_arrays)
        binned_arrays = np.empty((num_trials, 48), dtype=object)   #arrays of binned distance arrays for each trial (i.e. N
        # trial arrays, each with 48 bins of
        # 5cm distances); make the 48 softcoded
        bin_assignments = np.empty(num_trials, dtype=object)   # indexes of bins to use for cell activity

        for e, arr in enumerate(normalized_arrays):
            # get the indices of the bins to which each value belongs in an array; use np.digitize
            bin_indices = np.digitize(arr, bin_edges, right=False) - 1
            # Handle values exactly equal to 240 (put in last bin)
            bin_indices = np.where(arr == 240, 47, bin_indices)
            bin_assignments[e] = bin_indices #use these in future df to split up cell activity


        # Create the 5 cm arrays for each bin
            for i in range(48):
                mask = bin_indices == i
                bin_values = arr[mask]
                binned_arrays[e, i] = bin_values


        #TODO -- not sure if binned_df is necessary; make reduced df from start?  OR skip all together and just use as a
        # series

        # CREATE NEW DF - bin the trials into 5 cm bins, and average each bin for signal along position
        binned_df = result.with_columns(pl.Series("bin_assignments", bin_assignments))

        reduced_df = binned_df.drop("group_id", "start_index", "distance_array")
        index_col = "bin_assignments"

        signal_columns = [col for col in reduced_df.columns if col != index_col]

        # convert entire dataframe to numpy
        data_dict = reduced_df.to_dict(as_series=False)

        print("binned data")
        print("averaging over trials")
        
        # create dict for trial avgs
        trial_avgs = {}

        max_bins = n_bins  # this was calculated earlier
        for col in signal_columns:  # for each cell

            col_results = []

            # process all rows for this column
            for row_idx in range(len(reduced_df)):
                signal_array = np.array(data_dict[col][row_idx], dtype=np.float64)  # signal for that col/row
                index_array = np.array(data_dict[index_col][row_idx], dtype=np.int32)  # index for that trial (bins)

                # use numpy binning
                # np.bin_count counts all the values in the bin and sums them
                bin_sums = np.bincount(index_array, weights=signal_array, minlength=max_bins)  # sum the signals
                bin_counts = np.bincount(index_array, minlength=max_bins)  # find the length of the bin

                # calculate averages by dividing signal sum by bin length
                bin_averages = np.divide(bin_sums, bin_counts,
                                        out=np.full_like(bin_sums, np.nan),
                                        where=bin_counts != 0)

                col_results.append(bin_averages)

            trial_avgs[f'{col}_binned'] = col_results

        trial_avg_df = pl.DataFrame(trial_avgs)


        # now create dict for the average signal for each cell in the session
        avg_data = {}
        sess_sem = []       #had to make list bc I couldnt get both arrays into a single cell, there was some issue with
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

        print("averaged over trials")

        return session_avg_df, sess_sem, result, trial_avg_df