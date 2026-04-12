"""Provides assets for discovering, reading, and processing pre-extracted microcontroller module feather files
produced by the ataraxis-communication-interface library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
import polars as pl
from ataraxis_base_utilities import console
from ataraxis_data_structures import interpolate_data

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from numpy.typing import NDArray
    from sollertia_shared_assets import MesoscopeHardwareState

_MODULE_FEATHER_PATTERN: str = "controller_*_module_*.feather"
"""The glob pattern used to discover microcontroller module feather files produced by ataraxis-communication-interface.
"""


@dataclass(frozen=True, slots=True)
class _ModuleSpecification:
    """Defines the processing specification for a single hardware module type."""

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


def find_module_feathers(data_directory: Path) -> list[Path]:
    """Discovers microcontroller module feather files under the data directory.

    Recursively searches the data_directory for feather files matching the ``controller_*_module_*.feather`` naming
    convention used by ataraxis-communication-interface.

    Args:
        data_directory: The path to the root directory to search. The directory is searched recursively, so feather
            files may be nested at any depth below this path.

    Returns:
        A sorted list of paths to the discovered module feather files. Returns an empty list if the directory does
        not exist or if no matching files are found.
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
    hardware_state: MesoscopeHardwareState,
) -> None:
    """Reads a pre-extracted microcontroller module feather file and applies domain-specific data processing.

    Notes:
        Recovers the module type and ID by parsing the input feather filename, which already encodes them per the
        ``controller_{id}_module_{type}_{id}.feather`` convention, then dispatches to the appropriate parse
        function. The parse function transforms the raw event data from the axci feather format into a
        domain-specific feather file with physically meaningful columns.

    Args:
        feather_path: The path to the input module feather file produced by ataraxis-communication-interface.
        output_directory: The path to the output directory where the processed feather file will be written.
        hardware_state: The MesoscopeHardwareState instance that stores the hardware configuration parameters
            needed to convert raw sensor data into physical units.

    Raises:
        ValueError: If the (module_type, module_id) pair encoded in the feather filename does not match any
            registered module specification.
    """
    _, module_type, module_id = parse_module_feather_name(feather_path=feather_path)
    module_key = (module_type, module_id)
    if module_key not in _MODULE_REGISTRY:
        message = (
            f"Unable to process microcontroller module data. The module type-ID pair ({module_type}, {module_id}) "
            f"does not match any registered module specification."
        )
        console.error(message=message, error=ValueError)

    specification = _MODULE_REGISTRY[module_key]

    # Reads the pre-extracted module data via memory mapping (supported because all module feather writes use
    # uncompressed IPC), so the file is backed by the OS page cache rather than a full copy in private RAM.
    # Then partitions it by event code in a single pass, so parse functions can resolve their per-event lookups
    # in O(1) without re-scanning the full DataFrame.
    module_dataframe = pl.read_ipc(source=feather_path, memory_map=True)
    event_partition = _partition_events(module_dataframe=module_dataframe)

    # Ensures the output directory exists.
    output_directory.mkdir(parents=True, exist_ok=True)

    # Resolves hardware parameters and calls the appropriate parse function.
    output_file = output_directory / specification.output_filename
    specification.parse_function(
        event_partition=event_partition,
        output_file=output_file,
        hardware_state=hardware_state,
    )


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


def _partition_events(module_dataframe: pl.DataFrame) -> dict[int, pl.DataFrame]:
    """Partitions a module DataFrame into per-event sub-DataFrames in a single pass.

    Notes:
        Replaces the pattern of calling ``module_dataframe.filter(pl.col("event") == code)`` once per event
        code, which scans the entire DataFrame each time. Polars' partition_by traverses the DataFrame once
        and returns the groups keyed by event code, allowing subsequent lookups to be O(1).

    Args:
        module_dataframe: The Polars DataFrame read from an axci module feather file with the standard 5-column
            schema (timestamp_us, command, event, dtype, data).

    Returns:
        A dictionary mapping integer event codes to their corresponding sub-DataFrames.
    """
    # polars 1.x partition_by(as_dict=True) returns single-element tuples as keys, even when partitioning on
    # a single column, so the event code is always at index 0.
    raw_partition = module_dataframe.partition_by("event", as_dict=True)
    return {int(key[0]): value for key, value in raw_partition.items()}


def _get_event_timestamps(partition: dict[int, pl.DataFrame], event_code: int) -> NDArray[np.uint64]:
    """Returns the timestamp array for a given event code from a partitioned DataFrame.

    Notes:
        Designed for state-only events that do not carry data payloads.

    Args:
        partition: The event-code-keyed partition dictionary produced by _partition_events().
        event_code: The event code to look up.

    Returns:
        A NumPy uint64 array of timestamps, or an empty array if the event code is not present.
    """
    event_dataframe = partition.get(event_code)
    if event_dataframe is None:
        return np.array([], dtype=np.uint64)
    return event_dataframe["timestamp_us"].to_numpy().astype(np.uint64)


def _get_event_data[ScalarT: np.generic](
    partition: dict[int, pl.DataFrame],
    event_code: int,
    values_dtype: type[ScalarT],
) -> tuple[NDArray[np.uint64], NDArray[ScalarT]]:
    """Returns timestamps and vectorized-reconstructed data values for a given event code.

    Notes:
        Relies on the axci protocol guarantee that all messages sharing an event code also share a payload
        dtype, so binary payloads can be concatenated and decoded with a single np.frombuffer() call instead
        of a per-row Python loop. The reconstructed values are then cast to the requested output dtype for
        uniform downstream handling.

    Args:
        partition: The event-code-keyed partition dictionary produced by _partition_events().
        event_code: The event code to look up.
        values_dtype: The NumPy scalar type to cast the reconstructed values to.

    Returns:
        A tuple of (uint64 timestamp array, values array cast to values_dtype). Both arrays are empty if the
        event code is not present in the partition.
    """
    event_dataframe = partition.get(event_code)
    if event_dataframe is None:
        return np.array([], dtype=np.uint64), np.array([], dtype=values_dtype)

    timestamps: NDArray[np.uint64] = event_dataframe["timestamp_us"].to_numpy().astype(np.uint64)

    data_list = event_dataframe["data"].to_list()
    dtype_list = event_dataframe["dtype"].to_list()
    payload_dtype = dtype_list[0]
    values: NDArray[ScalarT] = np.frombuffer(b"".join(data_list), dtype=payload_dtype).astype(values_dtype)

    return timestamps, values


def _merge_event_streams[ScalarT: np.generic](
    timestamps_a: NDArray[np.uint64],
    values_a: NDArray[ScalarT],
    timestamps_b: NDArray[np.uint64],
    values_b: NDArray[ScalarT],
) -> tuple[NDArray[np.uint64], NDArray[ScalarT]]:
    """Merges two chronologically-sorted event streams into a single timestamp-sorted stream.

    Notes:
        Consolidates the allocate-empty-arrays / fill-halves / argsort pattern that was previously duplicated
        across every parse function. Uses NumPy's stable mergesort, which is near-linear on the already-sorted
        runs produced by the axci log format.

    Args:
        timestamps_a: The uint64 timestamp array for the first event stream.
        values_a: The value array for the first event stream.
        timestamps_b: The uint64 timestamp array for the second event stream.
        values_b: The value array for the second event stream.

    Returns:
        A tuple of (merged timestamps, reordered values) sorted chronologically.
    """
    timestamps = np.concatenate([timestamps_a, timestamps_b])
    values = np.concatenate([values_a, values_b])
    order = np.argsort(timestamps, kind="stable")
    return timestamps[order], values[order]


def _parse_encoder_data(
    event_partition: dict[int, pl.DataFrame], output_file: Path, hardware_state: MesoscopeHardwareState
) -> None:
    """Extracts and saves encoder module data as a .feather file.

    Notes:
        Converts raw encoder pulse events (CCW and CW rotations) into cumulative traveled distance in centimeters.
        CCW rotation is interpreted as positive displacement, CW as negative.

    Args:
        event_partition: The event-code-keyed partition dictionary containing the encoder module event data.
        output_file: The path to the output .feather file.
        hardware_state: The hardware configuration providing the cm_per_pulse conversion factor.
    """
    cm_per_pulse = np.float64(hardware_state.cm_per_pulse)

    # Pre-declares variable types so the fallback-branch reassignments below are checked against a fixed
    # declared type rather than widening into a union, which avoids a false-positive type mismatch when
    # passing these arrays into _merge_event_streams().
    ccw_timestamps: NDArray[np.uint64]
    ccw_values: NDArray[np.float64]
    cw_timestamps: NDArray[np.uint64]
    cw_values: NDArray[np.float64]

    # Extracts CCW (event 51) and CW (event 52) rotation data with displacement values.
    ccw_timestamps, ccw_values = _get_event_data(partition=event_partition, event_code=51, values_dtype=np.float64)
    cw_timestamps, cw_values = _get_event_data(partition=event_partition, event_code=52, values_dtype=np.float64)

    # Synthesizes an artificial zero-code entry if one direction is completely missing.
    if len(ccw_timestamps) == 0:
        # noinspection PyTypeChecker
        ccw_timestamps = np.array([cw_timestamps[0] + 1], dtype=np.uint64)
        # noinspection PyTypeChecker
        ccw_values = np.array([0.0], dtype=np.float64)
    elif len(cw_timestamps) == 0:
        # noinspection PyTypeChecker
        cw_timestamps = np.array([ccw_timestamps[0] + 1], dtype=np.uint64)
        # noinspection PyTypeChecker
        cw_values = np.array([0.0], dtype=np.float64)

    timestamps, displacements = _merge_event_streams(
        timestamps_a=ccw_timestamps, values_a=ccw_values, timestamps_b=cw_timestamps, values_b=-cw_values
    )

    # Integrates to cumulative distance and normalizes negative-zero entries.
    # noinspection PyTypeChecker
    positions = np.cumsum(displacements * cm_per_pulse)
    positions = np.round(positions, decimals=8)
    positions[np.isclose(positions, -0.0) & np.signbit(positions)] = 0.0

    result_dataframe = pl.DataFrame({"time_us": timestamps, "traveled_distance_cm": positions})
    result_dataframe.write_ipc(file=output_file, compression="uncompressed")


# noinspection PyUnusedLocal
def _parse_ttl_data(
    event_partition: dict[int, pl.DataFrame],
    output_file: Path,
    hardware_state: MesoscopeHardwareState,  # noqa: ARG001
) -> None:
    """Extracts and saves TTL module data as a .feather file.

    Notes:
        Processes TTL input signals by detecting ON (event 51) and OFF (event 52) transitions. Ensures the final
        value is 0 to properly mark the end of the monitoring sequence.

    Args:
        event_partition: The event-code-keyed partition dictionary containing the raw TTL module event data.
        output_file: The path to the output .feather file.
        hardware_state: The hardware configuration (unused for TTL processing but required for uniform dispatch).
    """
    on_timestamps = _get_event_timestamps(partition=event_partition, event_code=51)
    off_timestamps = _get_event_timestamps(partition=event_partition, event_code=52)

    # Aborts early if either ON or OFF signals are missing, as rising edges cannot be detected.
    if len(on_timestamps) == 0 or len(off_timestamps) == 0:
        return

    timestamps, triggers = _merge_event_streams(
        timestamps_a=on_timestamps,
        values_a=np.ones(len(on_timestamps), dtype=np.uint8),
        timestamps_b=off_timestamps,
        values_b=np.zeros(len(off_timestamps), dtype=np.uint8),
    )

    # Appends a terminal OFF state if the last recorded value is not 0.
    if triggers[-1] != 0:
        timestamps = np.append(timestamps, timestamps[-1] + 1)
        triggers = np.append(triggers, 0)

    result_dataframe = pl.DataFrame({"time_us": timestamps, "ttl_state": triggers})
    result_dataframe.write_ipc(file=output_file, compression="uncompressed")


def _parse_brake_data(
    event_partition: dict[int, pl.DataFrame], output_file: Path, hardware_state: MesoscopeHardwareState
) -> None:
    """Extracts and saves brake module data as a .feather file.

    Notes:
        Converts brake engagement events into torque values in Newton centimeters. When engaged (event 51), the brake
        applies maximum torque. When disengaged (event 52), it applies minimum torque due to mechanical coupling.

    Args:
        event_partition: The event-code-keyed partition dictionary containing the raw brake module event data.
        output_file: The path to the output .feather file.
        hardware_state: The hardware configuration providing brake strength parameters.
    """
    maximum_brake_strength = np.float64(hardware_state.maximum_brake_strength)

    # Handles legacy field naming (brake vs break) for backward compatibility.
    minimum_brake_strength_value = hardware_state.minimum_brake_strength
    if minimum_brake_strength_value is None:
        minimum_brake_strength_value = getattr(hardware_state, "minimum_break_strength", None)
    minimum_brake_strength = np.float64(minimum_brake_strength_value)

    engaged_timestamps = _get_event_timestamps(partition=event_partition, event_code=51)
    disengaged_timestamps = _get_event_timestamps(partition=event_partition, event_code=52)

    timestamps, torques = _merge_event_streams(
        timestamps_a=engaged_timestamps,
        values_a=np.full(len(engaged_timestamps), maximum_brake_strength, dtype=np.float64),
        timestamps_b=disengaged_timestamps,
        values_b=np.full(len(disengaged_timestamps), minimum_brake_strength, dtype=np.float64),
    )

    result_dataframe = pl.DataFrame({"time_us": timestamps, "brake_torque_N_cm": torques})
    result_dataframe.write_ipc(file=output_file, compression="uncompressed")


def _parse_valve_data(
    event_partition: dict[int, pl.DataFrame], output_file: Path, hardware_state: MesoscopeHardwareState
) -> None:
    """Extracts and saves water valve module data as a .feather file.

    Notes:
        Converts valve open/close timing into cumulative dispensed water volume using a calibrated power law
        equation. Also extracts tone buzzer state signals and temporally interpolates both data streams onto a
        shared timestamp grid.

    Args:
        event_partition: The event-code-keyed partition dictionary containing the raw valve module event data.
        output_file: The path to the output .feather file.
        hardware_state: The hardware configuration providing valve calibration parameters.
    """
    scale_coefficient = np.float64(hardware_state.valve_scale_coefficient)
    nonlinearity_exponent = np.float64(hardware_state.valve_nonlinearity_exponent)

    open_timestamps = _get_event_timestamps(partition=event_partition, event_code=51)
    closed_timestamps = _get_event_timestamps(partition=event_partition, event_code=52)

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

    timestamps, volume = _merge_event_streams(
        timestamps_a=open_timestamps,
        values_a=np.ones(len(open_timestamps), dtype=np.float64),
        timestamps_b=closed_timestamps,
        values_b=np.zeros(len(closed_timestamps), dtype=np.float64),
    )

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
    tone_on_timestamps = _get_event_timestamps(partition=event_partition, event_code=54)
    tone_off_timestamps = _get_event_timestamps(partition=event_partition, event_code=55)

    tone_timestamps, tone_states = _merge_event_streams(
        timestamps_a=tone_on_timestamps,
        values_a=np.ones(len(tone_on_timestamps), dtype=np.uint8),
        timestamps_b=tone_off_timestamps,
        values_b=np.zeros(len(tone_off_timestamps), dtype=np.uint8),
    )

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


# noinspection PyUnusedLocal
def _parse_gas_puff_data(
    event_partition: dict[int, pl.DataFrame],
    output_file: Path,
    hardware_state: MesoscopeHardwareState,  # noqa: ARG001
) -> None:
    """Extracts and saves gas puff valve module data as a .feather file.

    Notes:
        Tracks valve open/closed states and computes cumulative puff count by counting falling edges
        (open-to-closed transitions).

    Args:
        event_partition: The event-code-keyed partition dictionary containing the raw gas puff module event data.
        output_file: The path to the output .feather file.
        hardware_state: The hardware configuration (unused for gas puff processing but required for uniform dispatch).
    """
    open_timestamps = _get_event_timestamps(partition=event_partition, event_code=51)
    closed_timestamps = _get_event_timestamps(partition=event_partition, event_code=52)

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

    timestamps, states = _merge_event_streams(
        timestamps_a=open_timestamps,
        values_a=np.ones(len(open_timestamps), dtype=np.uint8),
        timestamps_b=closed_timestamps,
        values_b=np.zeros(len(closed_timestamps), dtype=np.uint8),
    )

    # Computes cumulative puff count from falling edges (1 -> 0 transitions).
    edges = np.diff(states, prepend=states[0])
    falling_edges = edges == -1
    cumulative_puffs: NDArray[np.uint32] = np.cumsum(falling_edges.astype(np.uint32))

    result_dataframe = pl.DataFrame(
        {"time_us": timestamps, "puff_state": states, "cumulative_puff_count": cumulative_puffs}
    )
    result_dataframe.write_ipc(file=output_file, compression="uncompressed")


def _parse_lick_data(
    event_partition: dict[int, pl.DataFrame], output_file: Path, hardware_state: MesoscopeHardwareState
) -> None:
    """Extracts and saves lick sensor module data as a .feather file.

    Notes:
        Preserves the raw 12-bit ADC voltage readings and applies threshold-based binary classification to detect
        lick events. The contact duration between consecutive ON and OFF edges corresponds to the tongue contact
        time with the lick tube.

    Args:
        event_partition: The event-code-keyed partition dictionary containing the raw lick sensor module event data.
        output_file: The path to the output .feather file.
        hardware_state: The hardware configuration providing the lick detection threshold.
    """
    if hardware_state.lick_threshold is None:
        message = (
            "Unable to parse lick sensor module data. The 'lick_threshold' field is not configured on the "
            "hardware state, but _parse_lick_data was invoked. This indicates a mismatch between the module "
            "eligibility filter and the parser registry."
        )
        console.error(message=message, error=ValueError)

    lick_threshold = np.uint16(hardware_state.lick_threshold)

    # Extracts voltage change events (event 51 only, preserving uint16 resolution).
    timestamps, voltages = _get_event_data(partition=event_partition, event_code=51, values_dtype=np.uint16)

    # Sorts by timestamp for additional safety.
    sort_indices = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[sort_indices]
    voltages = voltages[sort_indices]

    # Applies threshold-based binary lick classification.
    licks = (voltages >= lick_threshold).astype(np.uint8)

    result_dataframe = pl.DataFrame({"time_us": timestamps, "voltage_12_bit_adc": voltages, "lick_state": licks})
    result_dataframe.write_ipc(file=output_file, compression="uncompressed")


def _parse_torque_data(
    event_partition: dict[int, pl.DataFrame], output_file: Path, hardware_state: MesoscopeHardwareState
) -> None:
    """Extracts and saves torque sensor module data as a .feather file.

    Notes:
        Converts raw ADC readings from CCW (event 51, positive) and CW (event 52, negative) torque events into
        physical torque values in Newton centimeters.

    Args:
        event_partition: The event-code-keyed partition dictionary containing the raw torque sensor module event data.
        output_file: The path to the output .feather file.
        hardware_state: The hardware configuration providing the torque conversion factor.
    """
    torque_per_adc_unit = np.float64(hardware_state.torque_per_adc_unit)

    # Pre-declares variable types so the fallback-branch reassignments below are checked against a fixed
    # declared type rather than widening into a union.
    ccw_timestamps: NDArray[np.uint64]
    ccw_values: NDArray[np.float64]
    cw_timestamps: NDArray[np.uint64]
    cw_values: NDArray[np.float64]

    ccw_timestamps, ccw_values = _get_event_data(partition=event_partition, event_code=51, values_dtype=np.float64)
    cw_timestamps, cw_values = _get_event_data(partition=event_partition, event_code=52, values_dtype=np.float64)

    # Synthesizes missing direction data to handle edge cases.
    if len(ccw_timestamps) == 0:
        # noinspection PyTypeChecker
        ccw_timestamps = np.array([cw_timestamps[0] + 1], dtype=np.uint64)
        # noinspection PyTypeChecker
        ccw_values = np.array([0.0], dtype=np.float64)
    elif len(cw_timestamps) == 0:
        # noinspection PyTypeChecker
        cw_timestamps = np.array([ccw_timestamps[0] + 1], dtype=np.uint64)
        # noinspection PyTypeChecker
        cw_values = np.array([0.0], dtype=np.float64)

    timestamps, torques = _merge_event_streams(
        timestamps_a=ccw_timestamps,
        values_a=ccw_values * torque_per_adc_unit,
        timestamps_b=cw_timestamps,
        values_b=-cw_values * torque_per_adc_unit,
    )

    torques = np.round(torques, decimals=8)

    # Appends a terminal zero torque if the last value is not 0.
    if torques[-1] != 0:
        timestamps = np.append(timestamps, timestamps[-1] + 1)
        torques = np.append(torques, 0)

    torques[np.isclose(torques, -0.0) & np.signbit(torques)] = 0.0

    result_dataframe = pl.DataFrame({"time_us": timestamps, "torque_N_cm": torques})
    result_dataframe.write_ipc(file=output_file, compression="uncompressed")


def _parse_screen_data(
    event_partition: dict[int, pl.DataFrame], output_file: Path, hardware_state: MesoscopeHardwareState
) -> None:
    """Extracts and saves screen module data as a .feather file.

    Notes:
        Tracks the state of LED screens by detecting toggle pulse rising edges and tracking the screen state
        from its initial configuration value through each toggle event.

    Args:
        event_partition: The event-code-keyed partition dictionary containing the raw screen module event data.
        output_file: The path to the output .feather file.
        hardware_state: The hardware configuration providing the initial screen state.
    """
    # check_eligibility() guarantees screens_initially_on is not None before this function runs, but the type
    # stub still advertises it as bool | None. Coercing to a plain int narrows the type for the downstream
    # arithmetic on line ~711 and keeps the existing uint8 semantics (False/None -> 0, True -> 1).
    initially_on: int = 1 if hardware_state.screens_initially_on else 0

    on_timestamps = _get_event_timestamps(partition=event_partition, event_code=51)
    off_timestamps = _get_event_timestamps(partition=event_partition, event_code=52)

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

    timestamps, triggers = _merge_event_streams(
        timestamps_a=on_timestamps,
        values_a=np.ones(len(on_timestamps), dtype=np.uint8),
        timestamps_b=off_timestamps,
        values_b=np.zeros(len(off_timestamps), dtype=np.uint8),
    )

    # Detects rising edges to identify toggle events.
    edges = np.diff(triggers, prepend=0)
    rising_edges = np.where(edges == 1)[0]
    screen_timestamps = timestamps[rising_edges]

    # Prepends the initial state using the first recorded timestamp.
    screen_timestamps = np.concatenate(([timestamps[0]], screen_timestamps))

    # Builds the screen state array starting from the initial state and flipping at each toggle.
    state_count = len(screen_timestamps)
    # noinspection PyTypeChecker
    screen_states: NDArray[np.uint8] = np.empty(state_count, dtype=np.uint8)
    screen_states[0] = initially_on
    if state_count > 1:
        screen_states[1:] = (initially_on + np.arange(1, state_count)) % 2

    result_dataframe = pl.DataFrame({"time_us": screen_timestamps, "screen_state": screen_states})
    result_dataframe.write_ipc(file=output_file, compression="uncompressed")


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
