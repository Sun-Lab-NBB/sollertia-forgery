from pathlib import Path

import umap
import numpy as np
from scipy import stats
import polars as pl

from sl_forgery.utils.dataclass import TargetGroup, ProcessedSessionData

track_length = 240
cue_length = 30
bin_size = 5


class Processing:
    @staticmethod
    def _create_grouped_df(distance_df, signal_df, start_indices):
        """Helper function to bin_data

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

        """
        row_indices = distance_df["frame"]
        group_ids = np.searchsorted(start_indices, row_indices, side="right")

        # Add group_id column to distance dataframe
        distance_with_groups = distance_df.with_columns(pl.Series("group_id", group_ids))

        # Add group_id column to signal dataframe
        signal_with_groups = signal_df.with_columns(pl.Series("group_id", group_ids))

        # Create the grouped result for distances
        distance_grouped = (
            distance_with_groups.group_by("group_id")
            .agg([pl.col("frame").first().alias("start_index"), pl.col("traveled_distance_cm").alias("distance_array")])
            .sort("group_id")
        )

        # Get the actual cell column names from the signal dataframe
        cell_columns = [col for col in signal_df.columns if col.startswith("cell_")]

        # Create the grouped result for signals
        signal_grouped = (
            signal_with_groups.group_by("group_id")
            .agg([*[pl.col(f"{col}").alias(f"{col}_signal") for col in cell_columns]])
            .sort("group_id")
        )

        # Combine the results
        result = pl.concat([distance_grouped, signal_grouped.drop("group_id")], how="horizontal")

        return result

    # TODO: delete cache, it is only for developement and memory intensive
    @staticmethod
    def bin_data(target_group: str | TargetGroup, session_data: ProcessedSessionData):
        """Bins calcium imaging and behavioral data into fixed spatial bins along the track.
        This method identifies active periods, segments trials, normalizes distance traveled,
        and averages cell activity within position bins to produce both trial-level and
        session-level statistics.

        Args:
            target_group (str | TargetGroup): Which data grouping to use. Accepts either:
                - TargetGroup.SINGLE_DAY (or "single_day"):
                    Uses the single-day data loader and filters for identified cells
                    based on the Suite2p `iscell` mask.
                - TargetGroup.MULTI_DAY (or "multi_day"):
                    Uses the multi-day data loader without additional cell filtering.
                Passing any other string will raise a ValueError.
            session_data (ProcessedSessionData): Object containing references to
                behavior and fluorescence data loaders, including file paths.

        Returns:
            tuple[pl.DataFrame, list[np.ndarray], pl.DataFrame, pl.DataFrame]:
                - session_avg_df (pl.DataFrame): Session-level averaged activity across bins
                  for each cell, appended to trial averages.
                - sess_sem (list[np.ndarray]): Standard error of the mean (SEM) per cell across trials.
                - result (pl.DataFrame): Intermediate grouped DataFrame containing trial-level
                  distance arrays and corresponding fluorescence activity.
                - trial_avg_df (pl.DataFrame): Trial-level averages of binned cell activity.

        Notes:
            - Currently assumes fixed track length and bin size (hardcoded to 48 bins of 5 cm).
            - Only includes frames where the system is in the active running state
              (system_state == 2).
            - This function consolidates much of the original place field plotting code and
              should ideally be refactored into smaller, modular components.
        """
        # I left the commmented print statements in this function because they give explanation of what each part is
        # doing which will be useful when extracting this code into smaller helper functions
        # print("getting data")

        behavior_df = session_data.behavior_data.load(session_data.behavior_data.behavior_path)

        if isinstance(target_group, str):
            target_group = TargetGroup(target_group)

        match target_group:
            case TargetGroup.SINGLE_DAY:
                data_loader = session_data.single_day_data
                iscell = data_loader.load(data_loader.iscell_path)
                iscell_df = pl.DataFrame(iscell).with_row_index("cell_idx")
            case TargetGroup.MULTI_DAY:
                data_loader = session_data.multi_day_data

        fluorescence = data_loader.load(data_loader.F_path)
        fluorescence_df = pl.DataFrame(fluorescence.T, schema=[f"cell_{i}" for i in range(fluorescence.shape[0])])

        # print("got data")

        # print("grouping data")
        # 1st, choose only identified cells (currently suite2P is using 50% cutoff)
        # use column 1 i.e. boolean values

        if target_group == TargetGroup.SINGLE_DAY:
            cell_mask = iscell_df[:, 1].to_numpy()

            # only keep columns (cell data) for positive id cells  ->  cell_mask=True
            # np.where returns a tuple containing a numpy array with the indices ([idx], )
            cell_fluorescence_df = fluorescence_df.select([fluorescence_df.columns[i] for i in np.where(cell_mask)[0]])
        else:  # target_group = "multi_day"
            cell_fluorescence_df = fluorescence_df

        # then choose only frames where the system was in the active state i.e. mouse running
        active_state_mask = behavior_df["system_state"] == 2  # 2 is the active state (0 is idle, 1 is rest)

        # filter the dataframes by this active state
        active_behavior_df = behavior_df.filter(active_state_mask)
        active_fluorescence_df = cell_fluorescence_df.filter(active_state_mask)

        # TODO: 1. Check that the distance keeps increasing, otherwise there will be cell activity that is being compressed
        #  on the plot.  change this to group by trial

        # Find indices where the column value changes i.e. a new trial starts
        trial_start = active_behavior_df.filter(pl.col("trial") != pl.col("trial").shift(1))
        # TODO ^^^could also just "group_by" the trial value column; easier?

        trial_indices = trial_start["frame"].to_numpy()

        result = Processing._create_grouped_df(
            active_behavior_df.select(active_behavior_df["frame", "traveled_distance_cm"]),
            active_fluorescence_df,
            trial_indices,
        )

        # print("grouped data")
        # print("normalizing data")

        # create 5 cm bins
        # TODO:  need to soft code bin size and cue length late
        #   this only works with set lengths
        # this wont work w my task, with variable track lengths
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
                normalized = track_length * (arr - min_val) / (max_val - min_val)

            normalized_arrays.append(np.floor(normalized))

        # print("normalized data")
        # print("binning data")

        # bin the normalized arrays

        bin_edges = np.arange(
            0, track_length + bin_size, bin_size
        )  # [0, 5, 10, ..., 240]  --> again soft code for track_length + bin_size
        num_trials = len(normalized_arrays)
        binned_arrays = np.empty(
            (num_trials, 48), dtype=object
        )  # arrays of binned distance arrays for each trial (i.e. N
        # trial arrays, each with 48 bins of
        # 5cm distances); make the 48 softcoded
        bin_assignments = np.empty(num_trials, dtype=object)  # indexes of bins to use for cell activity

        for e, arr in enumerate(normalized_arrays):
            # get the indices of the bins to which each value belongs in an array; use np.digitize
            bin_indices = np.digitize(arr, bin_edges, right=False) - 1
            # Handle values exactly equal to 240 (put in last bin)
            bin_indices = np.where(arr == track_length, 47, bin_indices)
            bin_assignments[e] = bin_indices  # use these in future df to split up cell activity

            # Create the 5 cm arrays for each bin
            for i in range(48):
                mask = bin_indices == i
                bin_values = arr[mask]
                binned_arrays[e, i] = bin_values

        # TODO -- not sure if binned_df is necessary; make reduced df from start?  OR skip all together and just use as a
        # series

        # CREATE NEW DF - bin the trials into 5 cm bins, and average each bin for signal along position
        binned_df = result.with_columns(pl.Series("bin_assignments", bin_assignments))

        reduced_df = binned_df.drop("group_id", "start_index", "distance_array")
        index_col = "bin_assignments"

        signal_columns = [col for col in reduced_df.columns if col != index_col]

        # convert entire dataframe to numpy
        data_dict = reduced_df.to_dict(as_series=False)

        # print("binned data")
        # print("averaging over trials")

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
                bin_averages = np.divide(
                    bin_sums, bin_counts, out=np.full_like(bin_sums, np.nan), where=bin_counts != 0
                )

                col_results.append(bin_averages)

            trial_avgs[f"{col}_binned"] = col_results

        trial_avg_df = pl.DataFrame(trial_avgs)

        # now create dict for the average signal for each cell in the session
        avg_data = {}
        sess_sem = []  # had to make list bc I couldnt get both arrays into a single cell, there was some issue with
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

        # print("averaged over trials")

        return session_avg_df, sess_sem, result, trial_avg_df

    # TODO: delete or reencaspulate compute_single_session_umap, currently nothing calls this function.
    @staticmethod
    def _filter_for_umap(target_group: str | TargetGroup, session_data: ProcessedSessionData):
        """Filters neural and behavioral data before inputting into UMAP. Specifically,
        selects frames where the system is active (system_state == 2) and the mouse
        is in experiment stage 2 or 4. The method also ensures the correct data source
        is chosen depending on whether the analysis is single-day or multi-day.

        Args:
            target_group (str | TargetGroup): Which data grouping to use. Accepts either:
                - TargetGroup.SINGLE_DAY (or "single_day"):
                    Uses the single-day data loader and filters for identified cells
                    based on the Suite2p `iscell` mask.
                - TargetGroup.MULTI_DAY (or "multi_day"):
                    Uses the multi-day data loader without additional cell filtering.
                Passing any other string will raise a ValueError.
            session_data (SessionData): Object containing references to
                behavior and spike data loaders and their associated file paths.

        Returns:
            tuple[pl.DataFrame, pl.DataFrame]:
                - spikes_filtered (pl.DataFrame): Filtered spike activity,
                  with each column representing a cell.
                - behavior_filtered (pl.DataFrame): Filtered behavioral data
                  corresponding to the same frames.
        """
        behavior_df = session_data.behavior_data.load(session_data.behavior_data.behavior_path)

        if isinstance(target_group, str):
            target_group = TargetGroup(target_group)

        match target_group:
            case TargetGroup.SINGLE_DAY:
                data_loader = session_data.single_day_data
                iscell = data_loader.load(data_loader.iscell_path)
                iscell_df = pl.DataFrame(iscell).with_row_index("cell_idx")
            case TargetGroup.MULTI_DAY:
                data_loader = session_data.multi_day_data

        spikes = data_loader.load(data_loader.spks_path)
        spikes_df = pl.DataFrame(spikes.T, schema=[f"cell_{i}" for i in range(spikes.shape[0])])

        # Filter data

        active_state_mask = (behavior_df["experiment_stage"].is_in([2, 4])) & (behavior_df["system_state"] == 2)
        behavior_filtered = behavior_df.filter(active_state_mask)
        spikes_filtered = spikes_df.filter(active_state_mask)

        return spikes_filtered, behavior_filtered

    @staticmethod
    def compute_single_session_umap(
        target_group: str | TargetGroup,
        session_data: ProcessedSessionData,
        use_saved: bool = True,
        save: bool = True,
        alternate_path: Path | None = None,
    ):
        """Compute or load a UMAP embedding for a single session.

        This function filters neural and behavioral data for active states,
        computes a UMAP embedding if needed, and returns both the embedding
        and the filtered behavior data. By default, it will reuse a saved
        embedding if one exists, and only save to disk when a new embedding
        is generated.

        Args:
            target_group (str | TargetGroup): Which data grouping to use. Accepts either:
                - TargetGroup.SINGLE_DAY (or "single_day"):
                    Uses the single-day data loader and filters for identified cells
                    based on the Suite2p `iscell` mask.
                - TargetGroup.MULTI_DAY (or "multi_day"):
                    Uses the multi-day data loader without additional cell filtering.
                Passing any other string will raise a ValueError.
            session_data (ProcessedSessionData): Container for the session's
                spike and behavioral data, and associated DataLoader objects.
            use_saved (bool, default=True):
                If True and an embedding file already exists, load the saved
                embedding instead of recomputing.
            save (bool, default=True):
                If True, save the newly computed embedding to disk. Ignored
                when loading an existing embedding (no re-save).
            alternate_path (Path, optional):
                If provided, use this path instead of the default embedding
                path defined in the DataLoader.

        Returns:
            tuple:
                - embedding (numpy.ndarray): Array of shape (frames, 3) containing
                the UMAP embedding. Computed or loaded from disk.
                - behavior_filtered (pl.DataFrame): Filtered behavioral dataframe
                aligned with the embedding rows. Each row corresponds to the
                same frame as the embedding.
        """
        behavior_df = session_data.behavior_data.load(session_data.behavior_data.behavior_path)

        if isinstance(target_group, str):
            target_group = TargetGroup(target_group)

        match target_group:
            case TargetGroup.SINGLE_DAY:
                data_loader = session_data.single_day_data
                iscell = data_loader.load(data_loader.iscell_path)
                iscell_df = pl.DataFrame(iscell).with_row_index("cell_idx")
            case TargetGroup.MULTI_DAY:
                data_loader = session_data.multi_day_data

        spikes = data_loader.load(data_loader.spks_path)
        spikes_df = pl.DataFrame(spikes.T, schema=[f"cell_{i}" for i in range(spikes.shape[0])])

        # Filter data
        active_state_mask = (behavior_df["experiment_stage"].is_in([2, 4])) & (behavior_df["system_state"] == 2)
        behavior_filtered = behavior_df.filter(active_state_mask)

        if alternate_path is not None:
            embedding_path = alternate_path
        else:
            embedding_path = data_loader.umap_embedding_path

        if use_saved and embedding_path.exists():
            embedding = data_loader.load(embedding_path)
        else:
            spikes_filtered = spikes_df.filter(active_state_mask)
            spikes = spikes_filtered.to_numpy()  # umap needs cells x frames

            umap_data = umap.UMAP(n_neighbors=100, n_components=3, min_dist=0.1, n_jobs=-1, metric="correlation").fit(
                spikes
            )

            embedding = umap_data.embedding_

            if save:
                data_loader.save(embedding_path, embedding)

        return embedding, behavior_filtered
