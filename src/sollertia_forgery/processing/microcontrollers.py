"""Provides assets for discovering, reading, and processing pre-extracted microcontroller module feather files
produced by the ataraxis-communication-interface library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
import polars as pl
from ataraxis_base_utilities import LogLevel, console

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from sollertia_shared_assets import MesoscopeHardwareState

from ataraxis_data_structures import interpolate_data

if TYPE_CHECKING:
    from numpy.typing import NDArray


_MODULE_FEATHER_PATTERN: str = "controller_*_module_*.feather"
"""The glob pattern used to discover microcontroller module feather files produced by ataraxis-communication-interface.
"""


def find_module_feather(data_directory: Path, controller_id: int, module_type: int, module_id: int) -> Path:
    """Searches for a single module feather file matching the specified controller, module type, and module ID.

    Recursively searches the data_directory and all subdirectories for a feather file matching the
    ``controller_{controller_id}_module_{module_type}_{module_id}.feather`` naming convention used by
    ataraxis-communication-interface. Expects exactly one match within the directory tree.

    Args:
        data_directory: The path to the root directory to search. The directory is searched recursively, so feather
            files may be nested at any depth below this path.
        controller_id: The source ID of the microcontroller.
        module_type: The type code of the hardware module.
        module_id: The instance ID of the hardware module.

    Returns:
        The path to the discovered module feather file.

    Raises:
        FileNotFoundError: If the data_directory does not exist, is not a directory, or no feather file matching the
            specified parameters is found.
        ValueError: If multiple feather files matching the parameters are found.
    """
    if not data_directory.exists() or not data_directory.is_dir():
        message = (
            f"Unable to find module feather for controller {controller_id}, module ({module_type}, {module_id}) in "
            f"'{data_directory}'. The path does not exist or is not a directory."
        )
        console.error(message=message, error=FileNotFoundError)

    pattern = f"controller_{controller_id}_module_{module_type}_{module_id}.feather"
    matches = sorted(data_directory.rglob(pattern))

    if not matches:
        message = (
            f"Unable to find module feather for controller {controller_id}, module ({module_type}, {module_id}) in "
            f"'{data_directory}'. No file matching '{pattern}' was found."
        )
        console.error(message=message, error=FileNotFoundError)

    if len(matches) > 1:
        message = (
            f"Unable to find module feather for controller {controller_id}, module ({module_type}, {module_id}) in "
            f"'{data_directory}'. Multiple files matching '{pattern}' were found: "
            f"{[str(match) for match in matches]}. Expected exactly one match."
        )
        console.error(message=message, error=ValueError)

    return matches[0]


def find_all_module_feathers(data_directory: Path) -> list[Path]:
    """Discovers all microcontroller module feather files under the data directory.

    Recursively searches the data_directory for feather files matching the
    ``controller_*_module_*.feather`` naming convention used by ataraxis-communication-interface.

    Args:
        data_directory: The path to the root directory to search. The directory is searched recursively.

    Returns:
        A sorted list of paths to all discovered module feather files. Returns an empty list if no files are found.
    """
    if not data_directory.exists() or not data_directory.is_dir():
        return []

    return sorted(data_directory.rglob(_MODULE_FEATHER_PATTERN))


def parse_module_feather_name(feather_path: Path) -> tuple[int, int, int]:
    """Extracts the controller ID, module type, and module ID from a module feather filename.

    Args:
        feather_path: The path to the module feather file. The filename must follow the
            ``controller_{controller_id}_module_{module_type}_{module_id}.feather`` naming convention.

    Returns:
        A tuple of three integers: (controller_id, module_type, module_id).

    Raises:
        ValueError: If the filename does not follow the expected naming convention.
    """
    stem = feather_path.stem  # e.g., "controller_101_module_3_1"
    parts = stem.split("_")

    _expected_part_count = 5
    if len(parts) != _expected_part_count or parts[0] != "controller" or parts[2] != "module":
        message = (
            f"Unable to parse module feather filename '{feather_path.name}'. The filename does not follow the "
            f"expected 'controller_{{id}}_module_{{type}}_{{id}}.feather' naming convention."
        )
        console.error(message=message, error=ValueError)

    return int(parts[1]), int(parts[3]), int(parts[4])


def process_microcontroller_data(
    feather_path: Path,
    output_directory: Path,
    module_type: int,
    module_id: int,
    hardware_state: MesoscopeHardwareState,
) -> None:
    """Reads a pre-extracted microcontroller module feather file and applies domain-specific data processing.

    Notes:
        Dispatches to the appropriate parse function based on the (module_type, module_id) pair. The parse function
        transforms the raw event data from the axci feather format into a domain-specific feather file with
        physically meaningful columns.

    Args:
        feather_path: The path to the input module feather file produced by ataraxis-communication-interface.
        output_directory: The path to the output directory where the processed feather file will be written.
        module_type: The type code of the hardware module.
        module_id: The instance ID of the hardware module.
        hardware_state: The MesoscopeHardwareState instance that stores the hardware configuration parameters
            needed to convert raw sensor data into physical units.

    Raises:
        ValueError: If the (module_type, module_id) pair does not match any registered module specification.
    """
    module_key = (module_type, module_id)
    if module_key not in _MODULE_REGISTRY:
        message = (
            f"Unable to process microcontroller module data. The module type-ID pair ({module_type}, {module_id}) "
            f"does not match any registered module specification."
        )
        console.error(message=message, error=ValueError)

    specification = _MODULE_REGISTRY[module_key]

    console.echo(message=f"Processing module ({module_type}, {module_id}) from '{feather_path.name}'...")

    # Reads the pre-extracted module data from the axci feather file.
    module_dataframe = pl.read_ipc(source=feather_path)

    # Ensures the output directory exists.
    output_directory.mkdir(parents=True, exist_ok=True)

    # Resolves hardware parameters and calls the appropriate parse function.
    output_file = output_directory / specification.output_filename
    specification.parse_function(
        module_dataframe=module_dataframe,
        output_file=output_file,
        hardware_state=hardware_state,
    )

    console.echo(message=f"Module ({module_type}, {module_id}) processing: Complete.", level=LogLevel.SUCCESS)


def is_module_eligible(module_type: int, module_id: int, hardware_state: MesoscopeHardwareState) -> bool:
    """Determines whether a module is eligible for processing based on the hardware state configuration.

    Notes:
        A module is eligible if all of its required hardware state fields are configured (not None). Modules
        whose hardware parameters were not set during the acquisition session are skipped during processing.

    Args:
        module_type: The type code of the hardware module.
        module_id: The instance ID of the hardware module.
        hardware_state: The MesoscopeHardwareState instance to check against.

    Returns:
        True if the module is eligible for processing, False otherwise.
    """
    module_key = (module_type, module_id)
    if module_key not in _MODULE_REGISTRY:
        return False

    specification = _MODULE_REGISTRY[module_key]
    return specification.check_eligibility(hardware_state=hardware_state)


@dataclass(frozen=True, slots=True)
class _ModuleSpecification:
    """Defines the processing specification for a single hardware module type.

    Args:
        parse_function: The function that transforms raw axci event data into domain-specific feather output.
        output_filename: The name of the output feather file.
        required_fields: The MesoscopeHardwareState field names that must be configured for this module to be
            eligible for processing.
    """

    parse_function: Callable[..., None]
    """The function that transforms raw axci event data into domain-specific feather output."""
    output_filename: str
    """The name of the output feather file."""
    required_fields: tuple[str, ...]
    """The MesoscopeHardwareState field names that must be configured for processing eligibility."""

    def check_eligibility(self, hardware_state: MesoscopeHardwareState) -> bool:
        """Determines whether the hardware state has all required fields configured for this module.

        Args:
            hardware_state: The MesoscopeHardwareState instance to validate against.

        Returns:
            True if all required fields are configured (not None and not False for boolean fields).
        """
        for field_name in self.required_fields:
            value = getattr(hardware_state, field_name, None)
            if value is None:
                return False
            # Handles boolean fields like delivered_gas_puffs and recorded_mesoscope_ttl.
            if isinstance(value, bool) and not value:
                return False
        return True


def _extract_event_timestamps(module_dataframe: pl.DataFrame, event_code: int) -> NDArray[np.uint64]:
    """Filters a module DataFrame by event code and returns the timestamps.

    Notes:
        Designed for state-only events that do not carry data payloads. Returns only the timestamps for rows
        matching the specified event code.

    Args:
        module_dataframe: The Polars DataFrame read from an axci module feather file with the standard 5-column
            schema (timestamp_us, command, event, dtype, data).
        event_code: The event code to filter by.

    Returns:
        A NumPy array of uint64 timestamps for all messages matching the event code. Returns an empty array if no
        messages match.
    """
    filtered = module_dataframe.filter(pl.col("event") == np.uint8(event_code))
    if filtered.is_empty():
        return np.array([], dtype=np.uint64)
    return filtered["timestamp_us"].to_numpy().astype(np.uint64)


def _extract_event_data(
    module_dataframe: pl.DataFrame, event_code: int
) -> tuple[NDArray[np.uint64], NDArray[np.float64]]:
    """Filters a module DataFrame by event code and returns timestamps with reconstructed data values.

    Notes:
        Designed for events that carry numeric data payloads serialized as binary. Reconstructs each payload using
        the dtype string stored alongside it. All values are cast to float64 for uniform downstream processing.

    Args:
        module_dataframe: The Polars DataFrame read from an axci module feather file with the standard 5-column
            schema (timestamp_us, command, event, dtype, data).
        event_code: The event code to filter by.

    Returns:
        A tuple of two arrays. The first is a uint64 timestamp array. The second is a float64 array of
        reconstructed data values. Both arrays are empty if no messages match the event code.
    """
    filtered = module_dataframe.filter(pl.col("event") == np.uint8(event_code))
    if filtered.is_empty():
        return np.array([], dtype=np.uint64), np.array([], dtype=np.float64)

    timestamps = filtered["timestamp_us"].to_numpy().astype(np.uint64)

    # Reconstructs numeric values from binary payloads using the per-row dtype metadata.
    data_list = filtered["data"].to_list()
    dtype_list = filtered["dtype"].to_list()
    values = np.array(
        [
            np.frombuffer(data_bytes, dtype=dtype_string).item()
            for data_bytes, dtype_string in zip(data_list, dtype_list, strict=True)
        ],
        dtype=np.float64,
    )

    return timestamps, values


def _extract_event_data_uint16(
    module_dataframe: pl.DataFrame, event_code: int
) -> tuple[NDArray[np.uint64], NDArray[np.uint16]]:
    """Filters a module DataFrame by event code and returns timestamps with reconstructed uint16 data values.

    Notes:
        Specialized variant of _extract_event_data() that preserves the original uint16 resolution of sensor
        readings such as ADC voltage values from the lick sensor.

    Args:
        module_dataframe: The Polars DataFrame read from an axci module feather file with the standard 5-column
            schema (timestamp_us, command, event, dtype, data).
        event_code: The event code to filter by.

    Returns:
        A tuple of two arrays. The first is a uint64 timestamp array. The second is a uint16 array of
        reconstructed data values.
    """
    filtered = module_dataframe.filter(pl.col("event") == np.uint8(event_code))
    if filtered.is_empty():
        return np.array([], dtype=np.uint64), np.array([], dtype=np.uint16)

    timestamps = filtered["timestamp_us"].to_numpy().astype(np.uint64)

    data_list = filtered["data"].to_list()
    dtype_list = filtered["dtype"].to_list()
    values_uint16 = np.array(
        [
            np.frombuffer(data_bytes, dtype=dtype_string).item()
            for data_bytes, dtype_string in zip(data_list, dtype_list, strict=True)
        ],
        dtype=np.uint16,
    )

    return timestamps, values_uint16


def _parse_encoder_data(
    module_dataframe: pl.DataFrame, output_file: Path, hardware_state: MesoscopeHardwareState
) -> None:
    """Extracts and saves encoder module data as a .feather file.

    Notes:
        Converts raw encoder pulse events (CCW and CW rotations) into cumulative traveled distance in centimeters.
        CCW rotation is interpreted as positive displacement, CW as negative.

    Args:
        module_dataframe: The Polars DataFrame containing the raw encoder module event data.
        output_file: The path to the output .feather file.
        hardware_state: The hardware configuration providing the cm_per_pulse conversion factor.
    """
    cm_per_pulse = np.float64(hardware_state.cm_per_pulse)

    # Extracts CCW (event 51) and CW (event 52) rotation data with displacement values.
    ccw_timestamps, ccw_values = _extract_event_data(module_dataframe=module_dataframe, event_code=51)
    cw_timestamps, cw_values = _extract_event_data(module_dataframe=module_dataframe, event_code=52)

    # Synthesizes an artificial zero-code entry if one direction is completely missing.
    if len(ccw_timestamps) == 0:
        ccw_timestamps = np.array([cw_timestamps[0] + 1], dtype=np.uint64)
        ccw_values = np.array([0.0], dtype=np.float64)
    elif len(cw_timestamps) == 0:
        cw_timestamps = np.array([ccw_timestamps[0] + 1], dtype=np.uint64)
        cw_values = np.array([0.0], dtype=np.float64)

    # Combines both directions into unified arrays.
    total_length = len(ccw_timestamps) + len(cw_timestamps)
    timestamps: NDArray[np.uint64] = np.empty(total_length, dtype=np.uint64)
    displacements: NDArray[np.float64] = np.empty(total_length, dtype=np.float64)

    timestamps[: len(ccw_timestamps)] = ccw_timestamps
    displacements[: len(ccw_timestamps)] = ccw_values

    timestamps[len(ccw_timestamps) :] = cw_timestamps
    displacements[len(ccw_timestamps) :] = -cw_values

    # Sorts by timestamp and integrates to cumulative distance.
    sort_indices = np.argsort(timestamps)
    timestamps = timestamps[sort_indices]
    displacements = displacements[sort_indices]

    # noinspection PyTypeChecker
    positions = np.cumsum(displacements * cm_per_pulse)
    positions = np.round(positions, decimals=8)
    positions[np.isclose(positions, -0.0) & np.signbit(positions)] = 0.0

    result_dataframe = pl.DataFrame({"time_us": timestamps, "traveled_distance_cm": positions})
    result_dataframe.write_ipc(file=output_file, compression="uncompressed")


def _parse_ttl_data(
    module_dataframe: pl.DataFrame,
    output_file: Path,
    hardware_state: MesoscopeHardwareState,  # noqa: ARG001
) -> None:
    """Extracts and saves TTL module data as a .feather file.

    Notes:
        Processes TTL input signals by detecting ON (event 51) and OFF (event 52) transitions. Ensures the final
        value is 0 to properly mark the end of the monitoring sequence.

    Args:
        module_dataframe: The Polars DataFrame containing the raw TTL module event data.
        output_file: The path to the output .feather file.
        hardware_state: The hardware configuration (unused for TTL processing but required for uniform dispatch).
    """
    on_timestamps = _extract_event_timestamps(module_dataframe=module_dataframe, event_code=51)
    off_timestamps = _extract_event_timestamps(module_dataframe=module_dataframe, event_code=52)

    # Aborts early if either ON or OFF signals are missing, as rising edges cannot be detected.
    if len(on_timestamps) == 0 or len(off_timestamps) == 0:
        return

    # Combines and sorts ON/OFF signals chronologically.
    total_length = len(on_timestamps) + len(off_timestamps)
    timestamps: NDArray[np.uint64] = np.empty(total_length, dtype=np.uint64)
    triggers: NDArray[np.uint8] = np.empty(total_length, dtype=np.uint8)

    timestamps[: len(on_timestamps)] = on_timestamps
    triggers[: len(on_timestamps)] = 1

    timestamps[len(on_timestamps) :] = off_timestamps
    triggers[len(on_timestamps) :] = 0

    sort_indices = np.argsort(timestamps)
    timestamps = timestamps[sort_indices]
    triggers = triggers[sort_indices]

    # Appends a terminal OFF state if the last recorded value is not 0.
    if triggers[-1] != 0:
        timestamps = np.append(timestamps, timestamps[-1] + 1)
        triggers = np.append(triggers, 0)

    result_dataframe = pl.DataFrame({"time_us": timestamps, "ttl_state": triggers})
    result_dataframe.write_ipc(file=output_file, compression="uncompressed")


def _parse_brake_data(
    module_dataframe: pl.DataFrame, output_file: Path, hardware_state: MesoscopeHardwareState
) -> None:
    """Extracts and saves brake module data as a .feather file.

    Notes:
        Converts brake engagement events into torque values in Newton centimeters. When engaged (event 51), the brake
        applies maximum torque. When disengaged (event 52), it applies minimum torque due to mechanical coupling.

    Args:
        module_dataframe: The Polars DataFrame containing the raw brake module event data.
        output_file: The path to the output .feather file.
        hardware_state: The hardware configuration providing brake strength parameters.
    """
    maximum_brake_strength = np.float64(hardware_state.maximum_brake_strength)

    # Handles legacy field naming (brake vs break) for backward compatibility.
    minimum_brake_strength_value = hardware_state.minimum_brake_strength
    if minimum_brake_strength_value is None:
        minimum_brake_strength_value = getattr(hardware_state, "minimum_break_strength", None)
    minimum_brake_strength = np.float64(minimum_brake_strength_value)

    engaged_timestamps = _extract_event_timestamps(module_dataframe=module_dataframe, event_code=51)
    disengaged_timestamps = _extract_event_timestamps(module_dataframe=module_dataframe, event_code=52)

    # Combines engaged and disengaged events with their corresponding torque values.
    total_length = len(engaged_timestamps) + len(disengaged_timestamps)
    timestamps: NDArray[np.uint64] = np.empty(total_length, dtype=np.uint64)
    torques: NDArray[np.float64] = np.empty(total_length, dtype=np.float64)

    timestamps[: len(engaged_timestamps)] = engaged_timestamps
    torques[: len(engaged_timestamps)] = maximum_brake_strength

    timestamps[len(engaged_timestamps) :] = disengaged_timestamps
    torques[len(engaged_timestamps) :] = minimum_brake_strength

    sort_indices = np.argsort(timestamps)
    timestamps = timestamps[sort_indices]
    torques = torques[sort_indices]

    result_dataframe = pl.DataFrame({"time_us": timestamps, "brake_torque_N_cm": torques})
    result_dataframe.write_ipc(file=output_file, compression="uncompressed")


def _parse_valve_data(
    module_dataframe: pl.DataFrame, output_file: Path, hardware_state: MesoscopeHardwareState
) -> None:
    """Extracts and saves water valve module data as a .feather file.

    Notes:
        Converts valve open/close timing into cumulative dispensed water volume using a calibrated power law
        equation. Also extracts tone buzzer state signals and temporally interpolates both data streams onto a
        shared timestamp grid.

    Args:
        module_dataframe: The Polars DataFrame containing the raw valve module event data.
        output_file: The path to the output .feather file.
        hardware_state: The hardware configuration providing valve calibration parameters.
    """
    scale_coefficient = np.float64(hardware_state.valve_scale_coefficient)
    nonlinearity_exponent = np.float64(hardware_state.valve_nonlinearity_exponent)

    open_timestamps = _extract_event_timestamps(module_dataframe=module_dataframe, event_code=51)
    closed_timestamps = _extract_event_timestamps(module_dataframe=module_dataframe, event_code=52)

    # Handles the edge case where the valve was never opened (no water dispensed).
    if len(open_timestamps) == 0:
        result_dataframe = pl.DataFrame(
            {
                "time_us": np.array([closed_timestamps[0]], dtype=np.uint64),
                "dispensed_water_volume_uL": np.array([0], dtype=np.float64),
                "tone_state": np.array([0], dtype=np.uint8),
            }
        )
        result_dataframe.write_ipc(file=output_file, compression="uncompressed")
        return

    # Combines open and closed events to compute valve pulse durations.
    total_length = len(open_timestamps) + len(closed_timestamps)
    timestamps: NDArray[np.uint64] = np.empty(total_length, dtype=np.uint64)
    volume: NDArray[np.float64] = np.empty(total_length, dtype=np.float64)

    timestamps[: len(open_timestamps)] = open_timestamps
    volume[: len(open_timestamps)] = 1  # Open state

    timestamps[len(open_timestamps) :] = closed_timestamps
    volume[len(open_timestamps) :] = 0  # Closed state

    sort_indices = np.argsort(timestamps)
    timestamps = timestamps[sort_indices]
    volume = volume[sort_indices]

    # Detects valve open/close cycles using edge detection.
    edges = np.diff(volume, prepend=volume[0])
    rising_edges = np.where(edges == 1)[0]
    falling_edges = np.where(edges == -1)[0]

    reward_timestamps = timestamps[falling_edges]
    pulse_durations: NDArray[np.float64] = (timestamps[falling_edges] - timestamps[rising_edges]).astype(np.float64)

    # Converts pulse durations to dispensed water volume using calibrated power law.
    # noinspection PyTypeChecker
    volumes = np.cumsum(scale_coefficient * np.power(pulse_durations, nonlinearity_exponent))
    volumes = np.round(volumes, decimals=8)

    # Re-adds the initial zero volume at the first timestamp.
    reward_timestamps = np.insert(reward_timestamps, 0, timestamps[0])
    volumes = np.insert(volumes, 0, 0.0)

    # Extracts tone buzzer signals (event 54 = ON, event 55 = OFF).
    tone_on_timestamps = _extract_event_timestamps(module_dataframe=module_dataframe, event_code=54)
    tone_off_timestamps = _extract_event_timestamps(module_dataframe=module_dataframe, event_code=55)

    tone_length = len(tone_on_timestamps) + len(tone_off_timestamps)
    tone_timestamps: NDArray[np.uint64] = np.empty(tone_length, dtype=np.uint64)
    tone_states: NDArray[np.uint8] = np.empty(tone_length, dtype=np.uint8)

    tone_timestamps[: len(tone_on_timestamps)] = tone_on_timestamps
    tone_states[: len(tone_on_timestamps)] = 1

    tone_timestamps[len(tone_on_timestamps) :] = tone_off_timestamps
    tone_states[len(tone_on_timestamps) :] = 0

    sort_indices = np.argsort(tone_timestamps)
    tone_timestamps = tone_timestamps[sort_indices]
    tone_states = tone_states[sort_indices]

    if tone_states[-1] != 0:
        tone_timestamps = np.append(tone_timestamps, tone_timestamps[-1] + 1)
        tone_states = np.append(tone_states, 0)

    # Interpolates valve and tone data onto a shared timestamp grid.
    shared_stamps = np.unique(np.concatenate([tone_timestamps, reward_timestamps]))

    out_reward = interpolate_data(
        source_coordinates=reward_timestamps,
        source_values=volumes,
        target_coordinates=shared_stamps,
        is_discrete=True,
    )
    out_tones = interpolate_data(
        source_coordinates=tone_timestamps,
        source_values=tone_states,
        target_coordinates=shared_stamps,
        is_discrete=True,
    )

    result_dataframe = pl.DataFrame(
        {"time_us": shared_stamps, "dispensed_water_volume_uL": out_reward, "tone_state": out_tones}
    )
    result_dataframe.write_ipc(file=output_file, compression="uncompressed")


def _parse_gas_puff_data(
    module_dataframe: pl.DataFrame,
    output_file: Path,
    hardware_state: MesoscopeHardwareState,  # noqa: ARG001
) -> None:
    """Extracts and saves gas puff valve module data as a .feather file.

    Notes:
        Tracks valve open/closed states and computes cumulative puff count by counting falling edges
        (open-to-closed transitions).

    Args:
        module_dataframe: The Polars DataFrame containing the raw gas puff module event data.
        output_file: The path to the output .feather file.
        hardware_state: The hardware configuration (unused for gas puff processing but required for uniform dispatch).
    """
    open_timestamps = _extract_event_timestamps(module_dataframe=module_dataframe, event_code=51)
    closed_timestamps = _extract_event_timestamps(module_dataframe=module_dataframe, event_code=52)

    # Handles the edge case where no gas puffs were delivered.
    if len(open_timestamps) == 0:
        result_dataframe = pl.DataFrame(
            {
                "time_us": np.array([closed_timestamps[0]], dtype=np.uint64),
                "puff_state": np.array([0], dtype=np.uint8),
                "cumulative_puff_count": np.array([0], dtype=np.uint32),
            }
        )
        result_dataframe.write_ipc(file=output_file, compression="uncompressed")
        return

    # Combines and sorts open/closed events.
    total_length = len(open_timestamps) + len(closed_timestamps)
    timestamps: NDArray[np.uint64] = np.empty(total_length, dtype=np.uint64)
    states: NDArray[np.uint8] = np.empty(total_length, dtype=np.uint8)

    timestamps[: len(open_timestamps)] = open_timestamps
    states[: len(open_timestamps)] = 1

    timestamps[len(open_timestamps) :] = closed_timestamps
    states[len(open_timestamps) :] = 0

    sort_indices = np.argsort(timestamps)
    timestamps = timestamps[sort_indices]
    states = states[sort_indices]

    # Computes cumulative puff count from falling edges (1 -> 0 transitions).
    edges = np.diff(states, prepend=states[0])
    falling_edges = edges == -1
    cumulative_puffs: NDArray[np.uint32] = np.cumsum(falling_edges.astype(np.uint32))

    result_dataframe = pl.DataFrame(
        {"time_us": timestamps, "puff_state": states, "cumulative_puff_count": cumulative_puffs}
    )
    result_dataframe.write_ipc(file=output_file, compression="uncompressed")


def _parse_lick_data(module_dataframe: pl.DataFrame, output_file: Path, hardware_state: MesoscopeHardwareState) -> None:
    """Extracts and saves lick sensor module data as a .feather file.

    Notes:
        Preserves the raw 12-bit ADC voltage readings and applies threshold-based binary classification to detect
        lick events. The contact duration between consecutive ON and OFF edges corresponds to the tongue contact
        time with the lick tube.

    Args:
        module_dataframe: The Polars DataFrame containing the raw lick sensor module event data.
        output_file: The path to the output .feather file.
        hardware_state: The hardware configuration providing the lick detection threshold.
    """
    lick_threshold = np.uint16(hardware_state.lick_threshold)

    # Extracts voltage change events (event 51 only, preserving uint16 resolution).
    timestamps, voltages = _extract_event_data_uint16(module_dataframe=module_dataframe, event_code=51)

    # Sorts by timestamp for additional safety.
    sort_indices = np.argsort(timestamps)
    timestamps = timestamps[sort_indices]
    voltages = voltages[sort_indices]

    # Applies threshold-based binary lick classification.
    licks = (voltages >= lick_threshold).astype(np.uint8)

    result_dataframe = pl.DataFrame({"time_us": timestamps, "voltage_12_bit_adc": voltages, "lick_state": licks})
    result_dataframe.write_ipc(file=output_file, compression="uncompressed")


def _parse_torque_data(
    module_dataframe: pl.DataFrame, output_file: Path, hardware_state: MesoscopeHardwareState
) -> None:
    """Extracts and saves torque sensor module data as a .feather file.

    Notes:
        Converts raw ADC readings from CCW (event 51, positive) and CW (event 52, negative) torque events into
        physical torque values in Newton centimeters.

    Args:
        module_dataframe: The Polars DataFrame containing the raw torque sensor module event data.
        output_file: The path to the output .feather file.
        hardware_state: The hardware configuration providing the torque conversion factor.
    """
    torque_per_adc_unit = np.float64(hardware_state.torque_per_adc_unit)

    ccw_timestamps, ccw_values = _extract_event_data(module_dataframe=module_dataframe, event_code=51)
    cw_timestamps, cw_values = _extract_event_data(module_dataframe=module_dataframe, event_code=52)

    # Synthesizes missing direction data to handle edge cases.
    if len(ccw_timestamps) == 0:
        ccw_timestamps = np.array([cw_timestamps[0] + 1], dtype=np.uint64)
        ccw_values = np.array([0.0], dtype=np.float64)
    elif len(cw_timestamps) == 0:
        cw_timestamps = np.array([ccw_timestamps[0] + 1], dtype=np.uint64)
        cw_values = np.array([0.0], dtype=np.float64)

    # Combines both directions into unified arrays with physical unit conversion.
    total_length = len(ccw_timestamps) + len(cw_timestamps)
    timestamps: NDArray[np.uint64] = np.empty(total_length, dtype=np.uint64)

    timestamps[: len(ccw_timestamps)] = ccw_timestamps
    ccw_torques = ccw_values * torque_per_adc_unit

    timestamps[len(ccw_timestamps) :] = cw_timestamps
    cw_torques = -cw_values * torque_per_adc_unit

    torques = np.concatenate([ccw_torques, cw_torques])
    torques = np.round(torques, decimals=8)

    sort_indices = np.argsort(timestamps)
    timestamps = timestamps[sort_indices]
    torques = torques[sort_indices]

    # Appends a terminal zero torque if the last value is not 0.
    if torques[-1] != 0:
        timestamps = np.append(timestamps, timestamps[-1] + 1)
        torques = np.append(torques, 0)

    torques[np.isclose(torques, -0.0) & np.signbit(torques)] = 0.0

    result_dataframe = pl.DataFrame({"time_us": timestamps, "torque_N_cm": torques})
    result_dataframe.write_ipc(file=output_file, compression="uncompressed")


def _parse_screen_data(
    module_dataframe: pl.DataFrame, output_file: Path, hardware_state: MesoscopeHardwareState
) -> None:
    """Extracts and saves screen module data as a .feather file.

    Notes:
        Tracks the state of LED screens by detecting toggle pulse rising edges and tracking the screen state
        from its initial configuration value through each toggle event.

    Args:
        module_dataframe: The Polars DataFrame containing the raw screen module event data.
        output_file: The path to the output .feather file.
        hardware_state: The hardware configuration providing the initial screen state.
    """
    initially_on = hardware_state.screens_initially_on

    on_timestamps = _extract_event_timestamps(module_dataframe=module_dataframe, event_code=51)
    off_timestamps = _extract_event_timestamps(module_dataframe=module_dataframe, event_code=52)

    # Handles the case where screens never changed state.
    if len(on_timestamps) == 0:
        result_dataframe = pl.DataFrame(
            {
                "time_us": np.array([off_timestamps[0]], dtype=np.uint64),
                "screen_state": np.array([initially_on], dtype=np.uint8),
            }
        )
        result_dataframe.write_ipc(file=output_file, compression="uncompressed")
        return

    # Combines and sorts ON/OFF signals.
    total_length = len(on_timestamps) + len(off_timestamps)
    timestamps: NDArray[np.uint64] = np.empty(total_length, dtype=np.uint64)
    triggers: NDArray[np.uint8] = np.empty(total_length, dtype=np.uint8)

    timestamps[: len(on_timestamps)] = on_timestamps
    triggers[: len(on_timestamps)] = 1

    timestamps[len(on_timestamps) :] = off_timestamps
    triggers[len(on_timestamps) :] = 0

    sort_indices = np.argsort(timestamps)
    timestamps = timestamps[sort_indices]
    triggers = triggers[sort_indices]

    # Detects rising edges to identify toggle events.
    edges = np.diff(triggers, prepend=0)
    rising_edges = np.where(edges == 1)[0]
    screen_timestamps = timestamps[rising_edges]

    # Prepends the initial state using the first recorded timestamp.
    screen_timestamps = np.concatenate(([timestamps[0]], screen_timestamps))

    # Builds the screen state array starting from the initial state and flipping at each toggle.
    state_count = len(screen_timestamps)
    screen_states: NDArray[np.uint8] = np.empty(state_count, dtype=np.uint8)
    screen_states[0] = initially_on
    if state_count > 1:
        screen_states[1:] = (initially_on + np.arange(1, state_count)) % 2

    result_dataframe = pl.DataFrame({"time_us": screen_timestamps, "screen_state": screen_states})
    result_dataframe.write_ipc(file=output_file, compression="uncompressed")


# Maps (module_type, module_id) pairs to their processing specifications.
_MODULE_REGISTRY: dict[tuple[int, int], _ModuleSpecification] = {
    (2, 1): _ModuleSpecification(
        parse_function=_parse_encoder_data,
        output_filename="encoder_data.feather",
        required_fields=("cm_per_pulse",),
    ),
    (1, 1): _ModuleSpecification(
        parse_function=_parse_ttl_data,
        output_filename="mesoscope_frame_data.feather",
        required_fields=("recorded_mesoscope_ttl",),
    ),
    (3, 1): _ModuleSpecification(
        parse_function=_parse_brake_data,
        output_filename="brake_data.feather",
        required_fields=("maximum_brake_strength", "minimum_brake_strength"),
    ),
    (5, 1): _ModuleSpecification(
        parse_function=_parse_valve_data,
        output_filename="valve_data.feather",
        required_fields=("valve_scale_coefficient", "valve_nonlinearity_exponent"),
    ),
    (5, 2): _ModuleSpecification(
        parse_function=_parse_gas_puff_data,
        output_filename="gas_puff_data.feather",
        required_fields=("delivered_gas_puffs",),
    ),
    (4, 1): _ModuleSpecification(
        parse_function=_parse_lick_data,
        output_filename="lick_data.feather",
        required_fields=("lick_threshold",),
    ),
    (6, 1): _ModuleSpecification(
        parse_function=_parse_torque_data,
        output_filename="torque_data.feather",
        required_fields=("torque_per_adc_unit",),
    ),
    (7, 1): _ModuleSpecification(
        parse_function=_parse_screen_data,
        output_filename="screen_data.feather",
        required_fields=("screens_initially_on",),
    ),
}
"""Maps (module_type, module_id) pairs to their processing specifications. Each specification defines the parse
function, output filename, and required hardware state fields for a specific hardware module."""
