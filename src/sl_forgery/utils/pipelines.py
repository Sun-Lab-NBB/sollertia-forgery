"""This module stores assets used by other library modules to streamline constructing, submitting, and monitoring data
processing pipelines that run on remote compute servers."""

from pathlib import Path

from sl_shared_assets import Server
from ataraxis_time.time_helpers import get_timestamp
from typing import Any
from numpy.typing import NDArray
import numpy as np


def get_remote_job_work_directory(server: Server, job_name: str) -> Path:
    """Generates the working directory for the input job intended to be executed on the compute server managed by the
    input Server class.

    This worker function generates the current UTC timestamp, clips it down to minutes, and concatenates it to the
    job_name to construct the working directory name. It then resolves the path to that directory relative to the user
    working root on the remote server, creates the directory on the server, and returns the resolved path.
    """

    # Resolves working directory name using timestamp (accurate to minutes) and the job_name.
    timestamp = "-".join(get_timestamp().split("-")[:5])  # type: ignore
    working_directory = Path(server.user_working_root).joinpath("job_logs", f"{job_name}_{timestamp}")

    # Creates the working directory on the remote server.
    server.create_directory(remote_path=working_directory, parents=True)

    return working_directory


# noinspection PyTypeHints
def interpolate_data(
    timestamps: NDArray[np.uint64],
    data: NDArray[np.integer[Any] | np.floating[Any]],
    seed_timestamps: NDArray[np.uint64],
    is_discrete: bool,
) -> NDArray[np.signedinteger[Any] | np.unsignedinteger[Any] | np.floating[Any]]:
    """Interpolates data values for the provided seed timestamps.

    Notes:
        This function expects seed_timestamps and timestamps arrays to be monotonically increasing.

        Discrete interpolated data is returned as an array with the same datatype as the input data. Continuous
        interpolated data is returned as a float_64 datatype array.

    Args:
        timestamps: The one-dimensional NumPy array that stores the timestamps for the source data.
        data: The one-dimensional NumPy array that stores the source datapoints.
        seed_timestamps: The one-dimensional NumPy array that stores the timestamps for which to interpolate the data
            values.
        is_discrete: A boolean flag that determines whether the data is discrete or continuous.

    Returns:
        A one-dimensional NumPy array with the same length as the seed_timestamps array that stores the interpolated
        data values.
    """
    # Discrete data
    if is_discrete:
        # Preallocates the output array
        interpolated_data = np.empty(seed_timestamps.shape, dtype=data.dtype)

        # Handles boundary conditions in bulk using boolean masks. All seed timestamps below the minimum source
        # timestamp are statically set to data[0], and all seed timestamps above the maximum source timestamp are set
        # to data[-1].
        below_min = seed_timestamps < timestamps[0]
        above_max = seed_timestamps > timestamps[-1]
        within_bounds = ~(below_min | above_max)  # The portion of the seed that is within the source timestamp boundary

        # Assigns out-of-bounds values in-bulk
        interpolated_data[below_min] = data[0]
        interpolated_data[above_max] = data[-1]

        # Processes within-boundary timestamps by finding the last known certain value to the left of each seed
        # timestamp and setting each seed timestamp to that value.
        if np.any(within_bounds):
            indices = np.searchsorted(timestamps, seed_timestamps[within_bounds], side="right") - 1
            interpolated_data[within_bounds] = data[indices]

        return interpolated_data

    # Continuous data. Note, due to interpolation, continuous data is always returned using float_64 datatype.
    else:
        return np.interp(seed_timestamps, timestamps, data)
