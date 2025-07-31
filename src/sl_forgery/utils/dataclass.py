import polars as pl
import numpy as np
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
