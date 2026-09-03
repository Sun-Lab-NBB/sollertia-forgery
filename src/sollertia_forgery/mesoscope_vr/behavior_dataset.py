"""Provides assets for assembling behavior datasets from processed session data, including system-state, encoder,
lick, valve, brake, torque, and screen data sources.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit
import numpy as np
import polars as pl
from ataraxis_base_utilities import console
from sollertia_shared_assets import RawDataFiles, MesoscopeHardwareState
from ataraxis_data_structures import interpolate_data

from .metadata import BehaviorDataFiles

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


_MICROSECONDS_PER_SECOND: int = 1_000_000
"""The number of microseconds in one second."""

_MICROSECONDS_PER_MINUTE: int = 60 * 1_000_000
"""The number of microseconds in one minute."""

_RUNNING_SPEED_WINDOW_US: int = 100_000
"""The default sliding window size, in microseconds, used when computing running speed from encoder data."""

_TIME_COLUMNS: frozenset[str] = frozenset({"time_us", "elapsed_minutes"})
"""The dataset columns holding the session clock, dropped when a different sub-dataset supplies them."""


def assemble_behavior_dataset(
    microcontroller_data_path: Path,
    runtime_data_path: Path,
    raw_data_path: Path,
    reference_time: NDArray[np.uint64],
    *,
    drop_time_columns: bool = False,
) -> pl.DataFrame:
    """Assembles the target session's behavior dataset and aligns it to the reference time vector.

    Reads the module-parsed microcontroller feathers and the runtime system-state feather, interpolates each present
    source onto the reference time vector, and classifies every sample's reward outcome from the valve tone and water
    volume. The encoder, screen, brake, and torque sources are optional, so a session assembles with whichever
    feathers its session type produced.

    Args:
        microcontroller_data_path: The path to the processed microcontroller-data directory holding the module-parsed
            feathers (valve, lick, encoder, screen, brake, torque).
        runtime_data_path: The path to the processed runtime-data directory holding the runtime-parsed feathers
            (system state).
        raw_data_path: The path to the session's raw data directory containing the hardware state configuration.
        reference_time: The reference time vector to which to align the assembled dataset.
        drop_time_columns: Determines whether to drop the ``time_us`` and ``elapsed_minutes`` columns from the
            assembled dataset before returning it to the caller.

    Returns:
        A DataFrame aligned to the reference time vector with the columns ``system_state``, ``lick``, ``water_uL``,
        and ``reward``, plus the ``time_us`` and ``elapsed_minutes`` columns when ``drop_time_columns`` is False, and
        the optional ``brake``, ``screens``, ``torque_N_cm``, ``distance_cm``, and ``speed_cm_s`` columns when their
        source feathers are present.

    Raises:
        FileNotFoundError: If the session's hardware state YAML, or its valve, lick, or system state feather, is
            missing.
        ValueError: If the hardware state configuration is missing the required ``system_state_codes`` mapping, or
            (when brake data is present) the required ``minimum_brake_strength`` value.
        InvalidOperationError: If the session's system state feather carries a state code absent from the hardware
            state configuration's ``system_state_codes`` mapping.
    """
    # The hardware state stores the mapping name-first, so it is inverted into the code-keyed lookup the Enum cast
    # needs.
    hardware_state_data = MesoscopeHardwareState.from_yaml(
        file_path=raw_data_path.joinpath(RawDataFiles.HARDWARE_STATE)
    )
    state_mapping = hardware_state_data.system_state_codes
    if state_mapping is None:
        message = (
            "Unable to assemble the behavior dataset for the target session. The hardware state configuration "
            "is missing the required 'system_state_codes' mapping."
        )
        console.error(message=message, error=ValueError)
    inverted_mapping = {value: key for key, value in state_mapping.items()}
    state_enum = pl.Enum(list(state_mapping.keys()))

    valve_data_frame = pl.read_ipc(source=microcontroller_data_path.joinpath(BehaviorDataFiles.VALVE), memory_map=True)
    system_state_data_frame = pl.read_ipc(
        source=runtime_data_path.joinpath(BehaviorDataFiles.SYSTEM_STATE), memory_map=True
    )
    lick_data_frame = pl.read_ipc(source=microcontroller_data_path.joinpath(BehaviorDataFiles.LICK), memory_map=True)
    valve_time = valve_data_frame["time_us"].to_numpy()

    # Creates the aligned data dictionary using the reference time vector and interpolating all other data sources to
    # the reference time vector.
    aligned_data: dict[str, NDArray[np.number]] = {
        "time_us": reference_time,
        "system_state": interpolate_data(
            source_coordinates=system_state_data_frame["time_us"].to_numpy(),
            source_values=system_state_data_frame["system_state"].to_numpy(),
            target_coordinates=reference_time,
            is_discrete=True,
        ),
        "lick": interpolate_data(
            source_coordinates=lick_data_frame["time_us"].to_numpy(),
            source_values=lick_data_frame["lick_state"].to_numpy(),
            target_coordinates=reference_time,
            is_discrete=True,
        ),
        "water_uL": interpolate_data(
            source_coordinates=valve_time,
            source_values=valve_data_frame["dispensed_water_volume_uL"].to_numpy(),
            target_coordinates=reference_time,
            # Uses discrete interpolation because the underlying power-law dispensing function makes linear
            # interpolation equally inaccurate.
            is_discrete=True,
        ).astype(np.float32),
        # Temporary column used for reward event classification.
        "_tone_state": interpolate_data(
            source_coordinates=valve_time,
            source_values=valve_data_frame["tone_state"].to_numpy(),
            target_coordinates=reference_time,
            is_discrete=True,
        ),
    }

    # Encoder data is not present for lick training.
    encoder_file = microcontroller_data_path.joinpath(BehaviorDataFiles.ENCODER)
    if encoder_file.exists():
        encoder_data_frame = pl.read_ipc(source=encoder_file, memory_map=True)
        encoder_time = encoder_data_frame["time_us"].to_numpy()
        encoder_distance = encoder_data_frame["traveled_distance_cm"].to_numpy()

        running_speed = _calculate_running_speed(sample_time=encoder_time, distance=encoder_distance)

        aligned_data["distance_cm"] = interpolate_data(
            source_coordinates=encoder_time,
            source_values=encoder_distance,
            target_coordinates=reference_time,
            is_discrete=False,
        )
        aligned_data["speed_cm_s"] = interpolate_data(
            source_coordinates=encoder_time,
            source_values=running_speed,
            target_coordinates=reference_time,
            is_discrete=False,
        ).astype(np.float32)

    # Screen data is only present for mesoscope experiments.
    screen_file = microcontroller_data_path.joinpath(BehaviorDataFiles.SCREEN)
    if screen_file.exists():
        screen_data_frame = pl.read_ipc(source=screen_file, memory_map=True)
        aligned_data["screens"] = interpolate_data(
            source_coordinates=screen_data_frame["time_us"].to_numpy(),
            source_values=screen_data_frame["screen_state"].to_numpy(),
            target_coordinates=reference_time,
            is_discrete=True,
        )

    # Brake data is only present for mesoscope experiments.
    brake_file = microcontroller_data_path.joinpath(BehaviorDataFiles.BRAKE)
    if brake_file.exists():
        minimum_brake_strength = hardware_state_data.minimum_brake_strength
        if minimum_brake_strength is None:
            message = (
                "Unable to assemble the behavior dataset for the target session. The hardware state configuration "
                "is missing the required 'minimum_brake_strength' value."
            )
            console.error(message=message, error=ValueError)
        brake_data_frame = pl.read_ipc(source=brake_file, memory_map=True)
        brake_torque = interpolate_data(
            source_coordinates=brake_data_frame["time_us"].to_numpy(),
            source_values=brake_data_frame["brake_torque_N_cm"].to_numpy(),
            target_coordinates=reference_time,
            is_discrete=True,
        )
        aligned_data["brake"] = np.asarray(brake_torque > minimum_brake_strength, dtype=np.uint8)

    # Torque data is not present for run training.
    torque_file = microcontroller_data_path.joinpath(BehaviorDataFiles.TORQUE)
    if torque_file.exists():
        torque_data_frame = pl.read_ipc(source=torque_file, memory_map=True)
        aligned_data["torque_N_cm"] = interpolate_data(
            source_coordinates=torque_data_frame["time_us"].to_numpy(),
            source_values=torque_data_frame["torque_N_cm"].to_numpy(),
            target_coordinates=reference_time,
            is_discrete=False,
        ).astype(np.float32)

    behavior_data = pl.DataFrame(aligned_data)

    # Reward classification needs a stable event identifier, so consecutive samples sharing a tone state are grouped
    # before the water delivered inside each group is summed.
    behavior_data = (
        behavior_data.with_columns(
            pl.col("system_state").replace_strict(inverted_mapping, return_dtype=pl.Utf8).cast(state_enum),
            ((pl.col("time_us") - pl.col("time_us").min()) / _MICROSECONDS_PER_MINUTE)
            .round(2)
            .cast(pl.Float32)
            .alias("elapsed_minutes"),
            (pl.col("_tone_state") > 0).alias("_tone_active"),
        )
        .with_columns(
            (pl.col("_tone_active") != pl.col("_tone_active").shift(1))
            .fill_null(value=False)
            .cum_sum()
            .alias("_reward_event_id"),
            # water_uL is a cumulative running total, so the volume delivered at a sample is its increment over the
            # preceding sample.
            pl.col("water_uL").diff().fill_null(value=0.0).alias("_water_delta"),
        )
        .with_columns(
            pl.col("_water_delta").sum().over("_reward_event_id").alias("_reward_event_water_uL"),
        )
        # Classifies each sample as belonging to one of three categories: 'no' (no tone playing), 'yes' (tone playing
        # over a span that delivered water), or 'tone' (tone playing with no water delivered).
        .with_columns(
            pl.when(~pl.col("_tone_active"))
            .then(pl.lit("no"))
            .when(pl.col("_reward_event_water_uL") > 0)
            .then(pl.lit("yes"))
            .otherwise(pl.lit("tone"))
            .cast(pl.Enum(["no", "tone", "yes"]))
            .alias("reward")
        )
        .drop("_tone_state", "_tone_active", "_reward_event_id", "_water_delta", "_reward_event_water_uL")
    )

    # The torque sensor is disabled while the system is in the run state.
    if "torque_N_cm" in behavior_data.columns:
        behavior_data = behavior_data.with_columns(
            pl.when(pl.col("system_state") == "run").then(0.0).otherwise(pl.col("torque_N_cm")).alias("torque_N_cm")
        )

    if "distance_cm" in behavior_data.columns:
        behavior_data = (
            behavior_data
            # The encoder is disabled unless the system is in the run state, so its readout is held forward across
            # every other state and anchored at 0 before the first non-idle sample.
            .with_columns((pl.col("system_state") != "idle").cum_sum().alias("_past_idle"))
            .with_columns(
                pl.when((pl.col("system_state") == "idle") & (pl.col("_past_idle") == 0))
                .then(0.0)
                .when(pl.col("system_state") == "run")
                .then(pl.col("distance_cm"))
                .otherwise(None)
                .forward_fill()
                .fill_null(0.0)
                .alias("distance_cm"),
                pl.when(pl.col("system_state") == "run").then(pl.col("speed_cm_s")).otherwise(0.0).alias("speed_cm_s"),
            )
            .drop("_past_idle")
        )

    final_columns = [
        "time_us",
        "elapsed_minutes",
        "brake",
        "screens",
        "torque_N_cm",
        "distance_cm",
        "speed_cm_s",
        "lick",
        "water_uL",
        "reward",
        "system_state",
    ]

    columns_to_select = [column for column in final_columns if column in behavior_data.columns]

    if drop_time_columns:
        columns_to_select = [column for column in columns_to_select if column not in _TIME_COLUMNS]

    return behavior_data.select(columns_to_select)


@njit(cache=True, nogil=True)
def _calculate_running_speed(
    sample_time: NDArray[np.uint64],
    distance: NDArray[np.float64],
    window_size_us: int = _RUNNING_SPEED_WINDOW_US,
) -> NDArray[np.float32]:
    """Calculates the animal's running speed over the requested sliding window.

    Args:
        sample_time: The sampling time, in microseconds elapsed since UTC epoch onset, for each cumulative traveled
            distance value.
        distance: The cumulative distance, in centimeters, traveled by the animal at each sample.
        window_size_us: The size of the sliding window, in microseconds.

    Returns:
        The calculated running speed in centimeters per second for each sample. Backward displacement is clamped to
        zero, and a sample whose window holds no earlier sample reports zero.
    """
    value_count = len(sample_time)
    running_speed: NDArray[np.float32] = np.zeros(value_count, dtype=np.float32)

    if not value_count:
        return running_speed

    microseconds_to_seconds = np.float64(1.0 / _MICROSECONDS_PER_SECOND)

    # The sliding window start index advances monotonically, which keeps the scan linear in the number of samples.
    window_start_index = 0

    for sample_index in range(value_count):
        window_start_time = sample_time[sample_index] - window_size_us

        while window_start_index < sample_index and sample_time[window_start_index] < window_start_time:
            window_start_index += 1

        if window_start_index < sample_index:
            time_delta = sample_time[sample_index] - sample_time[window_start_index]

            if time_delta > 0:
                distance_delta = distance[sample_index] - distance[window_start_index]
                speed = distance_delta / (time_delta * microseconds_to_seconds)
                running_speed[sample_index] = max(0.0, speed)

    return running_speed
