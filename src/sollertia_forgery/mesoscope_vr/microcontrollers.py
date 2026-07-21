"""Provides the Mesoscope-VR module parsers that convert pre-extracted, event-code-keyed microcontroller module
event partitions into domain-specific behavior feathers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
import polars as pl
from ataraxis_base_utilities import console
from sollertia_shared_assets import MesoscopeHardwareState
from ataraxis_data_structures import interpolate_data

from .metadata import BehaviorDataFiles
from ..shared_assets import (
    get_event_data,
    merge_event_streams,
    get_event_timestamps,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from numpy.typing import NDArray
    from sollertia_shared_assets import SessionData


@dataclass(frozen=True, slots=True)
class _ModuleSpecification:
    """Defines the processing specification for a single hardware module type."""

    parse_function: Callable[..., None]
    """The function that transforms raw axci event data into domain-specific feather output."""
    output_filename: str
    """The name of the output feather file."""
    required_fields: tuple[str, ...]
    """The MesoscopeHardwareState field names that must be configured (not None) for processing eligibility."""
    usage_flags: tuple[str, ...]
    """The MesoscopeHardwareState boolean field names recording whether the module was used during acquisition. A flag
    set to False marks the module as unused, unlike a recorded state value (such as screens_initially_on), which is
    False on every session where the module was active."""
    event_codes: tuple[int, ...]
    """The axci event codes the module's parse function reads. The system-agnostic microcontroller pipeline builds the
    module's extraction filter from these codes, so a code absent here is never extracted from the log archive."""

    def check_eligibility(self, hardware_state: MesoscopeHardwareState) -> bool:
        """Determines whether the hardware state marks this module as used and fully configured.

        Notes:
            Per the MesoscopeHardwareState contract, a field set to None indicates that the corresponding module was
            not used, so None is the sole absence test for required_fields. A recorded state value is therefore never
            treated as an absence marker, even when it is False. Modules that additionally carry an explicit usage
            flag are skipped when that flag is not set.

        Args:
            hardware_state: The MesoscopeHardwareState instance to validate against.

        Returns:
            True if the module was used and all of its required fields are configured.
        """
        for field_name in self.required_fields:
            if getattr(hardware_state, field_name, None) is None:
                return False
        return all(getattr(hardware_state, flag_name, None) is True for flag_name in self.usage_flags)


# Public parser entry points wired into the MICROCONTROLLER_PARSER_REGISTRY (registries.py). Each shares the uniform
# (event_partition, output_directory, session) signature, resolves the session hardware state, skips when the module
# was not configured, and delegates to the matching private parser.
def parse_encoder(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: SessionData) -> None:
    """Parses the wheel-encoder module (type 2, id 1) into the session's encoder behavior feather."""
    hardware_state = _resolve_hardware_state(session=session)
    if not is_module_eligible(module_type=2, module_id=1, hardware_state=hardware_state):
        return
    _parse_encoder_data(
        event_partition=event_partition,
        output_file=output_directory / BehaviorDataFiles.ENCODER,
        hardware_state=hardware_state,
    )


def parse_mesoscope_frame(
    event_partition: dict[int, pl.DataFrame], output_directory: Path, session: SessionData
) -> None:
    """Parses the mesoscope-frame TTL module (type 1, id 1) into the session's mesoscope-frame behavior feather."""
    hardware_state = _resolve_hardware_state(session=session)
    if not is_module_eligible(module_type=1, module_id=1, hardware_state=hardware_state):
        return
    _parse_ttl_data(
        event_partition=event_partition,
        output_file=output_directory / BehaviorDataFiles.MESOSCOPE_FRAME,
        hardware_state=hardware_state,
    )


def parse_brake(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: SessionData) -> None:
    """Parses the brake module (type 3, id 1) into the session's brake behavior feather."""
    hardware_state = _resolve_hardware_state(session=session)
    if not is_module_eligible(module_type=3, module_id=1, hardware_state=hardware_state):
        return
    _parse_brake_data(
        event_partition=event_partition,
        output_file=output_directory / BehaviorDataFiles.BRAKE,
        hardware_state=hardware_state,
    )


def parse_valve(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: SessionData) -> None:
    """Parses the water-valve module (type 5, id 1) into the session's valve behavior feather."""
    hardware_state = _resolve_hardware_state(session=session)
    if not is_module_eligible(module_type=5, module_id=1, hardware_state=hardware_state):
        return
    _parse_valve_data(
        event_partition=event_partition,
        output_file=output_directory / BehaviorDataFiles.VALVE,
        hardware_state=hardware_state,
    )


def parse_gas_puff(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: SessionData) -> None:
    """Parses the gas-puff valve module (type 5, id 2) into the session's gas-puff behavior feather."""
    hardware_state = _resolve_hardware_state(session=session)
    if not is_module_eligible(module_type=5, module_id=2, hardware_state=hardware_state):
        return
    _parse_gas_puff_data(
        event_partition=event_partition,
        output_file=output_directory / BehaviorDataFiles.GAS_PUFF,
        hardware_state=hardware_state,
    )


def parse_lick(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: SessionData) -> None:
    """Parses the lick-sensor module (type 4, id 1) into the session's lick behavior feather."""
    hardware_state = _resolve_hardware_state(session=session)
    if not is_module_eligible(module_type=4, module_id=1, hardware_state=hardware_state):
        return
    _parse_lick_data(
        event_partition=event_partition,
        output_file=output_directory / BehaviorDataFiles.LICK,
        hardware_state=hardware_state,
    )


def parse_torque(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: SessionData) -> None:
    """Parses the torque-sensor module (type 6, id 1) into the session's torque behavior feather."""
    hardware_state = _resolve_hardware_state(session=session)
    if not is_module_eligible(module_type=6, module_id=1, hardware_state=hardware_state):
        return
    _parse_torque_data(
        event_partition=event_partition,
        output_file=output_directory / BehaviorDataFiles.TORQUE,
        hardware_state=hardware_state,
    )


def parse_screen(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: SessionData) -> None:
    """Parses the screen module (type 7, id 1) into the session's screen behavior feather."""
    hardware_state = _resolve_hardware_state(session=session)
    if not is_module_eligible(module_type=7, module_id=1, hardware_state=hardware_state):
        return
    _parse_screen_data(
        event_partition=event_partition,
        output_file=output_directory / BehaviorDataFiles.SCREEN,
        hardware_state=hardware_state,
    )


def is_module_eligible(module_type: int, module_id: int, hardware_state: MesoscopeHardwareState) -> bool:
    """Determines whether a module is eligible for processing based on the hardware state configuration.

    Notes:
        A module is eligible if all of its required hardware state fields are configured (not None) and every usage
        flag it carries is set. Modules whose hardware parameters were not set during the acquisition session, and
        modules a session explicitly records as unused, are skipped during processing.

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


def get_module_event_codes() -> dict[tuple[int, int], tuple[int, ...]]:
    """Returns the axci event codes each Mesoscope-VR hardware module's parser reads.

    Notes:
        This is the Mesoscope-VR system's donation to the MICROCONTROLLER_EVENT_CODE_REGISTRY ('registries.py'). The
        system-agnostic microcontroller pipeline builds each controller's extraction filter from this mapping, so the
        codes returned here are exactly the codes the extraction stage pulls out of the raw log archives. The mapping
        is rebuilt on every call, so callers may mutate the returned dictionary freely.

    Returns:
        A mapping from each ``(module_type, module_id)`` pair this system parses to the tuple of event codes its
        parser reads.
    """
    return {module_key: specification.event_codes for module_key, specification in _MODULE_REGISTRY.items()}


def _resolve_hardware_state(session: SessionData) -> MesoscopeHardwareState:
    """Loads the Mesoscope-VR hardware state from the session's raw data directory.

    Args:
        session: The loaded session whose microcontroller modules are being parsed.

    Returns:
        The loaded MesoscopeHardwareState instance.

    Raises:
        FileNotFoundError: If no hardware state YAML file is present at the session's canonical location.
    """
    hardware_state_path = session.raw_data.hardware_state_path
    if not hardware_state_path.is_file():
        message = (
            f"Unable to load hardware state for session '{session.session_name}'. No hardware state YAML file was "
            f"found at '{hardware_state_path}'."
        )
        console.error(message=message, error=FileNotFoundError)
    return MesoscopeHardwareState.from_yaml(file_path=hardware_state_path)


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
    # passing these arrays into merge_event_streams().
    ccw_timestamps: NDArray[np.uint64]
    ccw_values: NDArray[np.float64]
    cw_timestamps: NDArray[np.uint64]
    cw_values: NDArray[np.float64]

    # Extracts CCW (event 51) and CW (event 52) rotation data with displacement values.
    ccw_timestamps, ccw_values = get_event_data(partition=event_partition, event_code=51, values_dtype=np.float64)
    cw_timestamps, cw_values = get_event_data(partition=event_partition, event_code=52, values_dtype=np.float64)

    # Synthesizes an artificial zero-code entry if one direction is completely missing.
    if len(ccw_timestamps) == 0:
        ccw_timestamps = np.array([cw_timestamps[0] + 1], dtype=np.uint64)
        ccw_values = np.array([0.0], dtype=np.float64)
    elif len(cw_timestamps) == 0:
        cw_timestamps = np.array([ccw_timestamps[0] + 1], dtype=np.uint64)
        cw_values = np.array([0.0], dtype=np.float64)

    timestamps, displacements = merge_event_streams(
        timestamps_a=ccw_timestamps, values_a=ccw_values, timestamps_b=cw_timestamps, values_b=-cw_values
    )

    # Integrates to cumulative distance and normalizes negative-zero entries.
    positions = np.cumsum(displacements * cm_per_pulse)
    positions = np.round(positions, decimals=8)
    positions[np.isclose(positions, -0.0) & np.signbit(positions)] = 0.0

    result_dataframe = pl.DataFrame({"time_us": timestamps, "traveled_distance_cm": positions})
    result_dataframe.write_ipc(file=output_file, compression="uncompressed")


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
    on_timestamps = get_event_timestamps(partition=event_partition, event_code=51)
    off_timestamps = get_event_timestamps(partition=event_partition, event_code=52)

    # Aborts early if either ON or OFF signals are missing, as rising edges cannot be detected.
    if len(on_timestamps) == 0 or len(off_timestamps) == 0:
        return

    timestamps, triggers = merge_event_streams(
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

    engaged_timestamps = get_event_timestamps(partition=event_partition, event_code=51)
    disengaged_timestamps = get_event_timestamps(partition=event_partition, event_code=52)

    timestamps, torques = merge_event_streams(
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

    open_timestamps = get_event_timestamps(partition=event_partition, event_code=51)
    closed_timestamps = get_event_timestamps(partition=event_partition, event_code=52)

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

    timestamps, volume = merge_event_streams(
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
    volumes = np.cumsum(scale_coefficient * np.power(pulse_durations, nonlinearity_exponent))
    volumes = np.round(volumes, decimals=8)

    # Re-adds the initial zero volume at the first timestamp.
    reward_timestamps = np.insert(reward_timestamps, 0, timestamps[0])
    volumes = np.insert(volumes, 0, 0.0)

    # Extracts tone buzzer signals (event 54 = ON, event 55 = OFF).
    tone_on_timestamps = get_event_timestamps(partition=event_partition, event_code=54)
    tone_off_timestamps = get_event_timestamps(partition=event_partition, event_code=55)

    tone_timestamps, tone_states = merge_event_streams(
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

    interpolated_reward = interpolate_data(
        source_coordinates=reward_timestamps,
        source_values=volumes,
        target_coordinates=shared_stamps,
        is_discrete=True,
    )
    interpolated_tones = interpolate_data(
        source_coordinates=tone_timestamps,
        source_values=tone_states,
        target_coordinates=shared_stamps,
        is_discrete=True,
    )

    result_dataframe = pl.DataFrame(
        {"time_us": shared_stamps, "dispensed_water_volume_uL": interpolated_reward, "tone_state": interpolated_tones}
    )
    result_dataframe.write_ipc(file=output_file, compression="uncompressed")


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
    open_timestamps = get_event_timestamps(partition=event_partition, event_code=51)
    closed_timestamps = get_event_timestamps(partition=event_partition, event_code=52)

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

    timestamps, states = merge_event_streams(
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
    timestamps, voltages = get_event_data(partition=event_partition, event_code=51, values_dtype=np.uint16)

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

    ccw_timestamps, ccw_values = get_event_data(partition=event_partition, event_code=51, values_dtype=np.float64)
    cw_timestamps, cw_values = get_event_data(partition=event_partition, event_code=52, values_dtype=np.float64)

    # Synthesizes missing direction data to handle edge cases.
    if len(ccw_timestamps) == 0:
        ccw_timestamps = np.array([cw_timestamps[0] + 1], dtype=np.uint64)
        ccw_values = np.array([0.0], dtype=np.float64)
    elif len(cw_timestamps) == 0:
        cw_timestamps = np.array([ccw_timestamps[0] + 1], dtype=np.uint64)
        cw_values = np.array([0.0], dtype=np.float64)

    timestamps, torques = merge_event_streams(
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
    # screen_states arithmetic below and keeps the existing uint8 semantics (False/None -> 0, True -> 1).
    initially_on: int = 1 if hardware_state.screens_initially_on else 0

    on_timestamps = get_event_timestamps(partition=event_partition, event_code=51)
    off_timestamps = get_event_timestamps(partition=event_partition, event_code=52)

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

    timestamps, triggers = merge_event_streams(
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
    screen_states: NDArray[np.uint8] = np.empty(state_count, dtype=np.uint8)
    screen_states[0] = initially_on
    if state_count > 1:
        screen_states[1:] = (initially_on + np.arange(1, state_count)) % 2

    result_dataframe = pl.DataFrame({"time_us": screen_timestamps, "screen_state": screen_states})
    result_dataframe.write_ipc(file=output_file, compression="uncompressed")


_MODULE_REGISTRY: dict[tuple[int, int], _ModuleSpecification] = {
    (2, 1): _ModuleSpecification(
        parse_function=_parse_encoder_data,
        output_filename=BehaviorDataFiles.ENCODER,
        required_fields=("cm_per_pulse",),
        usage_flags=(),
        event_codes=(51, 52),
    ),
    (1, 1): _ModuleSpecification(
        parse_function=_parse_ttl_data,
        output_filename=BehaviorDataFiles.MESOSCOPE_FRAME,
        required_fields=(),
        usage_flags=("recorded_mesoscope_ttl",),
        event_codes=(51, 52),
    ),
    (3, 1): _ModuleSpecification(
        parse_function=_parse_brake_data,
        output_filename=BehaviorDataFiles.BRAKE,
        required_fields=("maximum_brake_strength", "minimum_brake_strength"),
        usage_flags=(),
        event_codes=(51, 52),
    ),
    (5, 1): _ModuleSpecification(
        parse_function=_parse_valve_data,
        output_filename=BehaviorDataFiles.VALVE,
        required_fields=("valve_scale_coefficient", "valve_nonlinearity_exponent"),
        usage_flags=(),
        # Code 53 (kCalibrated) is emitted only by the firmware's calibration command, never during a normal runtime.
        event_codes=(51, 52, 54, 55),
    ),
    (5, 2): _ModuleSpecification(
        parse_function=_parse_gas_puff_data,
        output_filename=BehaviorDataFiles.GAS_PUFF,
        required_fields=(),
        usage_flags=("delivered_gas_puffs",),
        # The gas-puff valve shares the water-valve firmware and therefore also emits the tone codes (54, 55), but the
        # gas-puff parser does not read them, so extracting them would be wasted work.
        event_codes=(51, 52),
    ),
    (4, 1): _ModuleSpecification(
        parse_function=_parse_lick_data,
        output_filename=BehaviorDataFiles.LICK,
        required_fields=("lick_threshold",),
        usage_flags=(),
        event_codes=(51,),
    ),
    (6, 1): _ModuleSpecification(
        parse_function=_parse_torque_data,
        output_filename=BehaviorDataFiles.TORQUE,
        required_fields=("torque_per_adc_unit",),
        usage_flags=(),
        event_codes=(51, 52),
    ),
    (7, 1): _ModuleSpecification(
        parse_function=_parse_screen_data,
        output_filename=BehaviorDataFiles.SCREEN,
        required_fields=("screens_initially_on",),
        usage_flags=(),
        event_codes=(51, 52),
    ),
}
"""Maps (module_type, module_id) pairs to their processing specifications. Each specification defines the parse
function, output filename, required hardware state fields, usage flags, and extracted event codes for a specific
hardware module."""
