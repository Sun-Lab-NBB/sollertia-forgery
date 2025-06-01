from typing import Any, Union
from pathlib import Path

import numpy as np
import pandas as pd
from suite2p.io import BinaryFileCombined, compute_dydx
from numpy.typing import NDArray


def memory_usage(dataframe: Union[pd.DataFrame, pd.Series], verbose: bool = True) -> float:
    """Calculate memory usage for a pandas DataFrame or Series.

    Args:
        dataframe (Union[pd.DataFrame, pd.Series]): Pandas DataFrame or Series to analyze.
        verbose (bool, optional): If True, prints the memory usage. Defaults to True.

    Returns:
        float: Memory usage in megabytes, rounded to 2 decimal places.
    """
    # Calculates the total memory usage of the DataFrame or Series (if one-dimensional). Sums up the
    # memory used by each data column for Dataframes, then converts the memory to Megabytes and
    # rounds to 2 SFs.
    memory_mb = dataframe.memory_usage(deep=True)
    if dataframe.ndim > 1:
        memory_mb = memory_mb.sum()
    memory_mb = round(memory_mb / 1024**2, 2)

    if verbose:
        print(f"Memory used: {memory_mb} MB")

    return memory_mb


def read_raw_frames(bin_folder: Union[Path, str], frames: NDArray[Any]) -> NDArray[Any]:
    """Read and combine raw imaging frames from multiple binary files.

    This function reads the requested number of frames from Suite2p binary files across multiple
    imaging planes and combines them into a single array. It aligns the planes into a virtual stack
    based on their spatial offsets.

    Args:
        bin_folder (Union[Path, str]): Path to the Suite2p folder containing imaging plane data.
        frames (NDArray[Any]): NumPy array specifying which frames to read (e.g., np.arange(0, 100)).

    Returns:
        NDArray[Any]: Combined imaging frame data packaged into a NumPy array with shape
            (n_frames, FOV_width, FOV_height).

    Raises:
        NameError: If no plane folders are found in the specified directory.
        FileNotFoundError: If `ops.npy` or `data.bin` files are missing.
    """
    bin_folder = Path(bin_folder)  # Ensure input is a Path object

    # Find all plane folders sorted numerically
    plane_folders = sorted(bin_folder.glob("plane*/"))
    if not plane_folders:
        raise NameError(f"Unable to find plane folders in {bin_folder}.")

    # Load 'ops.npy' files for all planes
    try:
        ops_list = [np.load(plane / "ops.npy", allow_pickle=True).item() for plane in plane_folders]
    except FileNotFoundError:
        raise FileNotFoundError("Unable to find one of the 'ops.npy' files.")

    # Get registered binary file locations
    registration_files = [plane / "data.bin" for plane in plane_folders]
    if not all(file.exists() for file in registration_files):
        raise FileNotFoundError("Unable to find one of the 'data.bin' files.")

    # Compute x and y anchor coordinates for each frame
    y_anchors, x_anchors = compute_dydx(ops_list)

    # Extract frame dimensions for each plane
    frame_heights = np.array([ops["Ly"] for ops in ops_list])
    frame_widths = np.array([ops["Lx"] for ops in ops_list])

    # Determine the overall field of view (FOV) dimensions. Since the resultant stack includes
    # multiple frames, they may not be 100% aligned, resulting in the overall FOV being larger than
    # any given frame.
    fov_height = int(np.amax(y_anchors + frame_heights))
    fov_width = int(np.amax(x_anchors + frame_widths))

    # Combines individual planes into a virtual stack using Suite2P and packages the results into a numpy array
    # noinspection PyTypeChecker
    with BinaryFileCombined(
        fov_height, fov_width, frame_heights, frame_widths, y_anchors, x_anchors, registration_files
    ) as binary_file:
        return binary_file[frames]
