from typing import Any
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sl_shared_assets import SessionData
from ataraxis_base_utilities import ensure_directory_exists


def _interpolate_data(
    timestamps: NDArray[np.uint64],
    data: NDArray[np.integer[Any] | np.floating[Any]],
    seed_timestamps: NDArray[np.uint64],
    is_discrete: bool,
) -> NDArray[np.signedinteger[Any] | np.unsignedinteger[Any] | np.floating[Any]]:
    """Interpolates data values for the provided seed timestamps.

    Primarily, this service function is used to time-align different datastreams from the same source. For example, the
    Valve module generates both the solenoid valve data and the auditory tone data, which is generated at non-matching
    rates. This function is used to equalize the data sampling rate between the two data streams, allowing to output
    the data as .feather file.

    Notes:
        This function expects seed_timestamps and timestamps arrays to be monotonically increasing.

        Discrete interpolated data will be returned as an array with the same datatype as the input data. Continuous
        interpolated data will always use float_64 datatype.

    Args:
        timestamps: The one-dimensional numpy array that stores the timestamps for the source data.
        data: The one-dimensional numpy array that stores the source datapoints.
        seed_timestamps: The one-dimensional numpy array that stores the timestamps for which to interpolate the data
            values.
        is_discrete: A boolean flag that determines whether the data is discrete or continuous.

    Returns:
        A numpy NDArray with the same dimension as the seed_timestamps array that stores the interpolated data values.
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
        return np.interp(seed_timestamps, timestamps, data)  # type: ignore


def _assemble_mesoscope_data(df: pl.DataFrame, single_day_path: Path) -> pl.DataFrame:
    """ """

    # Queries the number of frames processed by suite2p. This is used to handle rare cases where the log has more
    # frame stamps than recorded frames, which usually happens if the user manually triggers mesoscope scanning (of any
    # kind) outside the expected time slot. Specifically, if the log has more frames than the suite2p data, the log is
    # clipped at the front (since aberrant frames have to come from a period before the main experiment runtime)
    _, frames = np.load(single_day_path.joinpath("F.npy"), mmap_mode="r").shape

    # Loads the single-day 'ops' file and extracts the scanning frequency ('fs') from the dictionary. Uses it to
    # determine the minimum and maximum scan pulse duration in milliseconds. This is used to filter the logged scan
    # pulses to match expected scan durations.
    sd_ops = np.load(single_day_path.joinpath("ops.npy"), allow_pickle=True).item()
    scanning_frequency = sd_ops["fs"]
    min_duration = 1000 / scanning_frequency - 20
    max_duration = 1000 / scanning_frequency + 20

    # Sort by time to ensure the correct order
    df = df.sort("time_us")

    # Create pulse groups and find edges in one pass
    df = df.with_columns(
        [
            # Create pulse ID by counting rising edges
            (pl.col("ttl_state").diff() == 1).fill_null(False).cum_sum().alias("pulse_id"),
            # Mark if this row is a rising edge
            (pl.col("ttl_state").diff() == 1).fill_null(False).alias("is_rising_edge"),
            # Mark if this row is a falling edge
            (pl.col("ttl_state").diff() == -1).fill_null(False).alias("is_falling_edge"),
        ]
    )

    # For each pulse, find the rising and falling-edge times using window functions
    df = df.with_columns(
        [
            # Get the time of the rising edge for this pulse
            pl.when(pl.col("is_rising_edge"))
            .then(pl.col("time_us"))
            .forward_fill()
            .over("pulse_id")
            .alias("pulse_start"),
            # Get the time of the falling edge for this pulse
            pl.when(pl.col("is_falling_edge"))
            .then(pl.col("time_us"))
            .backward_fill()
            .over("pulse_id")
            .alias("pulse_end"),
        ]
    )

    # Calculate duration and filter
    df = df.with_columns([((pl.col("pulse_end") - pl.col("pulse_start")) / 1000).alias("duration_ms")])

    # Filter for valid duration pulses and get unique frames
    frame_df: pl.DataFrame = (
        df.filter(
            (pl.col("duration_ms") >= min_duration)
            & (pl.col("duration_ms") <= max_duration)
            & pl.col("duration_ms").is_not_null()
        )
        .group_by("pulse_id")
        .first()
        .select(
            [
                pl.col("pulse_id").alias("frame"),
                pl.col("pulse_start").alias("start_time_us"),
                pl.col("pulse_end").alias("stop_time_us"),
            ]
        )
        .sort("frame")
    )

    # If we have more frames than expected, keep only the last expected_frames
    original_frame_count = len(frame_df)
    if original_frame_count > frames:
        frame_df = frame_df.tail(frames)  # Keeps the last 'frames' number of pulses

    # Re-assign frame IDs starting from 1
    frame_df = frame_df.with_columns([pl.int_range(1, len(frame_df) + 1).alias("frame")])

    return frame_df


def generate_behavior_dataset(session_data: SessionData, dataset_path: Path, track_size_cm: int = 240) -> None:
    # Mesoscope Frame Data
    single_day_path = dataset_path.joinpath(session_data.animal_id, session_data.session_name, "single_day")
    frame_file_path = session_data.processed_data.behavior_data_path.joinpath(f"mesoscope_frame_data.feather")
    raw_frame_pulses = pl.read_ipc(frame_file_path, use_pyarrow=True)
    mesoscope_frames = _assemble_mesoscope_data(df=raw_frame_pulses, single_day_path=single_day_path)
    frame_timestamps = mesoscope_frames["start_time_us"].to_numpy()

    # Encoder Data
    encoder_file_path = session_data.processed_data.behavior_data_path.joinpath(f"encoder_data.feather")
    encoder_data = pl.read_ipc(encoder_file_path, use_pyarrow=True)
    timestamps = encoder_data["time_us"].to_numpy()
    distance = encoder_data["traveled_distance_cm"].to_numpy()
    distance_at_frame = _interpolate_data(
        timestamps=timestamps, data=distance, seed_timestamps=frame_timestamps, is_discrete=False
    )
    trial_at_frame = (distance_at_frame // track_size_cm + 1).astype(int)

    # Lick Data
    lick_file_path = session_data.processed_data.behavior_data_path.joinpath(f"lick_data.feather")
    lick_data = pl.read_ipc(lick_file_path, use_pyarrow=True)
    timestamps = lick_data["time_us"].to_numpy()
    licks = lick_data["lick_state"].to_numpy()
    lick_at_frame = _interpolate_data(
        timestamps=timestamps, data=licks, seed_timestamps=frame_timestamps, is_discrete=True
    )

    # Valve Data
    valve_file_path = session_data.processed_data.behavior_data_path.joinpath(f"valve_data.feather")
    valve_data = pl.read_ipc(valve_file_path, use_pyarrow=True)
    timestamps = valve_data["time_us"].to_numpy()
    reward = valve_data["tone_state"].to_numpy()
    reward_at_frame = _interpolate_data(
        timestamps=timestamps, data=reward, seed_timestamps=frame_timestamps, is_discrete=True
    )

    # Experiment States
    experiment_state_path = session_data.processed_data.behavior_data_path.joinpath(f"experiment_state_data.feather")
    experiment_state = pl.read_ipc(experiment_state_path, use_pyarrow=True)
    timestamps = experiment_state["time_us"].to_numpy()
    experiment = experiment_state["experiment_state"].to_numpy()
    experiment_stage_at_frame = _interpolate_data(
        timestamps=timestamps, data=experiment, seed_timestamps=frame_timestamps, is_discrete=True
    )

    # System States
    system_state_path = session_data.processed_data.behavior_data_path.joinpath(f"system_state_data.feather")
    system_state = pl.read_ipc(system_state_path, use_pyarrow=True)
    timestamps = system_state["time_us"].to_numpy()
    experiment = system_state["system_state"].to_numpy()
    system_state_at_frame = _interpolate_data(
        timestamps=timestamps, data=experiment, seed_timestamps=frame_timestamps, is_discrete=True
    )

    # Assembles the final behavior dataset
    behavior_dataset = pl.DataFrame(
        {
            "frame": mesoscope_frames["frame"],
            "frame_time_us": frame_timestamps,
            "traveled_distance_cm": distance_at_frame,
            "trial": trial_at_frame,
            "lick_state": lick_at_frame,
            "reward_state": reward_at_frame,
            "experiment_stage": experiment_stage_at_frame,
            "system_state": system_state_at_frame,
        }
    )

    behavior_path = dataset_path.joinpath(session_data.animal_id, session_data.session_name, "behavior")
    ensure_directory_exists(behavior_path)
    behavior_dataset.write_ipc(file=behavior_path.joinpath("behavior_at_frame.feather"), compression="lz4")

def assemble_dataset()

session = SessionData.load(session_path=Path("/media/Data/TestMice/6/2025-06-27-12-44-58-770644"))
dataset = Path("/media/Data/TestMice/TM_06_pilot")
generate_behavior_dataset(session, dataset)
