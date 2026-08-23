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
from ataraxis_communication_interface import get_event_data, get_event_timestamps

from .metadata import BehaviorDataFiles
from ..shared_assets import merge_event_streams

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from sollertia_shared_assets import SessionData


_PRIMARY_EVENT_CODE: int = 51
"""The axci event code carrying each module's primary data stream."""

_SECONDARY_EVENT_CODE: int = 52
"""The axci event code carrying each module's complementary data stream."""

_TONE_ON_EVENT_CODE: int = 54
"""The axci event code marking the reward tone onset."""

_TONE_OFF_EVENT_CODE: int = 55
"""The axci event code marking the reward tone offset."""

_ENCODER_MODULE: tuple[int, int] = (2, 1)
"""The ``(module_type, module_id)`` pair identifying the wheel-encoder module."""

_MESOSCOPE_FRAME_MODULE: tuple[int, int] = (1, 1)
"""The ``(module_type, module_id)`` pair identifying the mesoscope-frame TTL module."""

_BRAKE_MODULE: tuple[int, int] = (3, 1)
"""The ``(module_type, module_id)`` pair identifying the wheel brake module."""

_VALVE_MODULE: tuple[int, int] = (5, 1)
"""The ``(module_type, module_id)`` pair identifying the water valve module."""

_GAS_PUFF_MODULE: tuple[int, int] = (5, 2)
"""The ``(module_type, module_id)`` pair identifying the gas puff valve module."""

_LICK_MODULE: tuple[int, int] = (4, 1)
"""The ``(module_type, module_id)`` pair identifying the lick sensor module."""

_TORQUE_MODULE: tuple[int, int] = (6, 1)
"""The ``(module_type, module_id)`` pair identifying the torque sensor module."""

_SCREEN_MODULE: tuple[int, int] = (7, 1)
"""The ``(module_type, module_id)`` pair identifying the VR screen module."""


@dataclass(frozen=True, slots=True)
class _ModuleSpecification:
    """Defines the processing specification for a single hardware module instance."""

    required_fields: tuple[str, ...]
    """The ``MesoscopeHardwareState`` field names that must be configured (not None) for processing eligibility."""
    usage_flags: tuple[str, ...]
    """The ``MesoscopeHardwareState`` boolean field names recording whether the module was used during acquisition. A
    flag set to False marks the module as unused. Required fields carry recorded state values such as
    ``screens_initially_on``, which stay meaningful when False, so the two groups are checked separately."""
    event_codes: tuple[int, ...]
    """The axci event codes the module's parse function reads. The system-agnostic microcontroller pipeline builds the
    module's extraction filter from these codes, so a code absent here is never extracted from the log archive."""

    def check_eligibility(self, hardware_state: MesoscopeHardwareState) -> bool:
        """Determines whether the hardware state marks this module as used and fully configured.

        Notes:
            Per the ``MesoscopeHardwareState`` contract, a field set to None indicates that the corresponding module
            was not used, so None is the sole absence test for ``required_fields``. A recorded state value is
            therefore never treated as an absence marker, even when it is False. Modules that additionally carry an
            explicit usage flag are skipped when that flag is not set.

        Args:
            hardware_state: The session hardware configuration to validate against.

        Returns:
            True if the module was used and all of its required fields are configured.
        """
        for field_name in self.required_fields:
            if getattr(hardware_state, field_name, None) is None:
                return False
        return all(getattr(hardware_state, flag_name, None) for flag_name in self.usage_flags)


_MODULE_REGISTRY: dict[tuple[int, int], _ModuleSpecification] = {
    _ENCODER_MODULE: _ModuleSpecification(
        required_fields=("cm_per_pulse",),
        usage_flags=(),
        event_codes=(_PRIMARY_EVENT_CODE, _SECONDARY_EVENT_CODE),
    ),
    _MESOSCOPE_FRAME_MODULE: _ModuleSpecification(
        required_fields=(),
        usage_flags=("recorded_mesoscope_ttl",),
        event_codes=(_PRIMARY_EVENT_CODE, _SECONDARY_EVENT_CODE),
    ),
    _BRAKE_MODULE: _ModuleSpecification(
        required_fields=("maximum_brake_strength", "minimum_brake_strength"),
        usage_flags=(),
        event_codes=(_PRIMARY_EVENT_CODE, _SECONDARY_EVENT_CODE),
    ),
    _VALVE_MODULE: _ModuleSpecification(
        required_fields=("valve_scale_coefficient", "valve_nonlinearity_exponent"),
        usage_flags=(),
        # Code 53 (kCalibrated) is emitted only by the firmware's calibration command, never during a normal runtime.
        event_codes=(_PRIMARY_EVENT_CODE, _SECONDARY_EVENT_CODE, _TONE_ON_EVENT_CODE, _TONE_OFF_EVENT_CODE),
    ),
    _GAS_PUFF_MODULE: _ModuleSpecification(
        required_fields=(),
        usage_flags=("delivered_gas_puffs",),
        # The gas-puff valve shares the water-valve firmware and therefore also emits the tone codes (54, 55), but the
        # gas-puff parser does not read them, so extracting them would be wasted work.
        event_codes=(_PRIMARY_EVENT_CODE, _SECONDARY_EVENT_CODE),
    ),
    _LICK_MODULE: _ModuleSpecification(
        required_fields=("lick_threshold",),
        usage_flags=(),
        event_codes=(_PRIMARY_EVENT_CODE,),
    ),
    _TORQUE_MODULE: _ModuleSpecification(
        required_fields=("torque_per_adc_unit",),
        usage_flags=(),
        event_codes=(_PRIMARY_EVENT_CODE, _SECONDARY_EVENT_CODE),
    ),
    _SCREEN_MODULE: _ModuleSpecification(
        required_fields=("screens_initially_on",),
        usage_flags=(),
        event_codes=(_PRIMARY_EVENT_CODE, _SECONDARY_EVENT_CODE),
    ),
}
"""Maps ``(module_type, module_id)`` pairs to their processing specifications. Each specification defines the required
hardware state fields, usage flags, and extracted event codes for a specific hardware module instance."""


# Public parser entry points wired into the microcontroller parser registry ('registries.py'). The uniform
# (event_partition, output_directory, session) signature is what the registry dispatches on.
def parse_encoder(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: SessionData) -> None:
    """Parses the wheel-encoder module (type 2, id 1) into the session's encoder behavior feather.

    Args:
        event_partition: The event-code-keyed partition dictionary holding the module's extracted event data.
        output_directory: The path to the session's processed microcontroller-data directory where the feather is
            written.
        session: The loaded session whose hardware state determines module eligibility.

    Raises:
        FileNotFoundError: If the session's hardware state YAML file is absent.
        ValueError: If either rotation event code carries no data payload, stores a null payload inside an otherwise
            decodable stream, spreads its payloads across more than one dtype, or decodes into a value count that is
            not a whole multiple of its message count.
    """
    hardware_state = _resolve_hardware_state(session=session)
    if not _is_module_eligible(module_key=_ENCODER_MODULE, hardware_state=hardware_state):
        return
    _parse_encoder_data(
        event_partition=event_partition,
        output_file=output_directory / BehaviorDataFiles.ENCODER,
        hardware_state=hardware_state,
    )


def parse_mesoscope_frame(
    event_partition: dict[int, pl.DataFrame], output_directory: Path, session: SessionData
) -> None:
    """Parses the mesoscope-frame TTL module (type 1, id 1) into the session's mesoscope-frame behavior feather.

    Args:
        event_partition: The event-code-keyed partition dictionary holding the module's extracted event data.
        output_directory: The path to the session's processed microcontroller-data directory where the feather is
            written.
        session: The loaded session whose hardware state determines module eligibility.

    Raises:
        FileNotFoundError: If the session's hardware state YAML file is absent.
        ValueError: If the extracted event data carries no rising or no falling TTL edge timestamps.
    """
    hardware_state = _resolve_hardware_state(session=session)
    if not _is_module_eligible(module_key=_MESOSCOPE_FRAME_MODULE, hardware_state=hardware_state):
        return
    _parse_ttl_data(
        event_partition=event_partition,
        output_file=output_directory / BehaviorDataFiles.MESOSCOPE_FRAME,
        session=session,
    )


def parse_brake(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: SessionData) -> None:
    """Parses the brake module (type 3, id 1) into the session's brake behavior feather.

    Args:
        event_partition: The event-code-keyed partition dictionary holding the module's extracted event data.
        output_directory: The path to the session's processed microcontroller-data directory where the feather is
            written.
        session: The loaded session whose hardware state determines module eligibility.

    Raises:
        FileNotFoundError: If the session's hardware state YAML file is absent.
    """
    hardware_state = _resolve_hardware_state(session=session)
    if not _is_module_eligible(module_key=_BRAKE_MODULE, hardware_state=hardware_state):
        return
    _parse_brake_data(
        event_partition=event_partition,
        output_file=output_directory / BehaviorDataFiles.BRAKE,
        hardware_state=hardware_state,
    )


def parse_valve(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: SessionData) -> None:
    """Parses the water-valve module (type 5, id 1) into the session's valve behavior feather.

    Args:
        event_partition: The event-code-keyed partition dictionary holding the module's extracted event data.
        output_directory: The path to the session's processed microcontroller-data directory where the feather is
            written.
        session: The loaded session whose hardware state determines module eligibility.

    Raises:
        FileNotFoundError: If the session's hardware state YAML file is absent.
    """
    hardware_state = _resolve_hardware_state(session=session)
    if not _is_module_eligible(module_key=_VALVE_MODULE, hardware_state=hardware_state):
        return
    _parse_valve_data(
        event_partition=event_partition,
        output_file=output_directory / BehaviorDataFiles.VALVE,
        hardware_state=hardware_state,
    )


def parse_gas_puff(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: SessionData) -> None:
    """Parses the gas-puff valve module (type 5, id 2) into the session's gas-puff behavior feather.

    Args:
        event_partition: The event-code-keyed partition dictionary holding the module's extracted event data.
        output_directory: The path to the session's processed microcontroller-data directory where the feather is
            written.
        session: The loaded session whose hardware state determines module eligibility.

    Raises:
        FileNotFoundError: If the session's hardware state YAML file is absent.
    """
    hardware_state = _resolve_hardware_state(session=session)
    if not _is_module_eligible(module_key=_GAS_PUFF_MODULE, hardware_state=hardware_state):
        return
    _parse_gas_puff_data(
        event_partition=event_partition,
        output_file=output_directory / BehaviorDataFiles.GAS_PUFF,
        hardware_state=hardware_state,
    )


def parse_lick(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: SessionData) -> None:
    """Parses the lick-sensor module (type 4, id 1) into the session's lick behavior feather.

    Args:
        event_partition: The event-code-keyed partition dictionary holding the module's extracted event data.
        output_directory: The path to the session's processed microcontroller-data directory where the feather is
            written.
        session: The loaded session whose hardware state determines module eligibility.

    Raises:
        FileNotFoundError: If the session's hardware state YAML file is absent.
        ValueError: If the hardware state carries no lick detection threshold, or if the lick event code carries no
            data payload, stores a null payload inside an otherwise decodable stream, spreads its payloads across more
            than one dtype, or decodes into a value count that is not a whole multiple of its message count.
    """
    hardware_state = _resolve_hardware_state(session=session)
    if not _is_module_eligible(module_key=_LICK_MODULE, hardware_state=hardware_state):
        return
    _parse_lick_data(
        event_partition=event_partition,
        output_file=output_directory / BehaviorDataFiles.LICK,
        hardware_state=hardware_state,
    )


def parse_torque(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: SessionData) -> None:
    """Parses the torque-sensor module (type 6, id 1) into the session's torque behavior feather.

    Args:
        event_partition: The event-code-keyed partition dictionary holding the module's extracted event data.
        output_directory: The path to the session's processed microcontroller-data directory where the feather is
            written.
        session: The loaded session whose hardware state determines module eligibility.

    Raises:
        FileNotFoundError: If the session's hardware state YAML file is absent.
        ValueError: If either rotation event code carries no data payload, stores a null payload inside an otherwise
            decodable stream, spreads its payloads across more than one dtype, or decodes into a value count that is
            not a whole multiple of its message count.
    """
    hardware_state = _resolve_hardware_state(session=session)
    if not _is_module_eligible(module_key=_TORQUE_MODULE, hardware_state=hardware_state):
        return
    _parse_torque_data(
        event_partition=event_partition,
        output_file=output_directory / BehaviorDataFiles.TORQUE,
        hardware_state=hardware_state,
    )


def parse_screen(event_partition: dict[int, pl.DataFrame], output_directory: Path, session: SessionData) -> None:
    """Parses the screen module (type 7, id 1) into the session's screen behavior feather.

    Args:
        event_partition: The event-code-keyed partition dictionary holding the module's extracted event data.
        output_directory: The path to the session's processed microcontroller-data directory where the feather is
            written.
        session: The loaded session whose hardware state determines module eligibility.

    Raises:
        FileNotFoundError: If the session's hardware state YAML file is absent.
    """
    hardware_state = _resolve_hardware_state(session=session)
    if not _is_module_eligible(module_key=_SCREEN_MODULE, hardware_state=hardware_state):
        return
    _parse_screen_data(
        event_partition=event_partition,
        output_file=output_directory / BehaviorDataFiles.SCREEN,
        hardware_state=hardware_state,
    )


def get_eligible_modules(session: SessionData) -> set[tuple[int, int]]:
    """Returns the Mesoscope-VR hardware modules the target session configured for use.

    Notes:
        This is the Mesoscope-VR system's donation to the microcontroller eligibility registry ('registries.py'). The
        system-agnostic microcontroller pipeline narrows each controller's extraction filter to the modules returned
        here, so a module the session records as unused is left out of both the extraction stage and the parse job
        universe. Every parser applies the same eligibility check before writing its feather, so the pipeline and the
        parsers agree on which modules a session carries.

    Args:
        session: The loaded session whose hardware state determines module eligibility.

    Returns:
        The ``(module_type, module_id)`` pairs the session configured for use.
    """
    hardware_state = _resolve_hardware_state(session=session)
    return {
        module_key
        for module_key in _MODULE_REGISTRY
        if _is_module_eligible(module_key=module_key, hardware_state=hardware_state)
    }


def get_module_event_codes() -> dict[tuple[int, int], tuple[int, ...]]:
    """Returns the axci event codes each Mesoscope-VR hardware module's parser reads.

    Notes:
        This is the Mesoscope-VR system's donation to the microcontroller event-code registry ('registries.py'). The
        system-agnostic microcontroller pipeline builds each controller's extraction filter from this mapping,
        narrowed to the modules the session marks eligible, so a code absent here is never extracted. The mapping is
        rebuilt on every call, so callers may mutate the returned dictionary freely.

    Returns:
        A mapping from each ``(module_type, module_id)`` pair this system parses to the tuple of event codes its
        parser reads.
    """
    return {module_key: specification.event_codes for module_key, specification in _MODULE_REGISTRY.items()}


def _is_module_eligible(module_key: tuple[int, int], hardware_state: MesoscopeHardwareState) -> bool:
    """Determines whether a module is eligible for processing based on the hardware state configuration.

    Notes:
        A module is eligible if all of its required hardware state fields are configured (not None) and every usage
        flag it carries is set. Modules whose hardware parameters were not set during the acquisition session, and
        modules a session explicitly records as unused, are skipped during processing.

    Args:
        module_key: The ``(module_type, module_id)`` pair identifying the hardware module.
        hardware_state: The session hardware configuration to check against.

    Returns:
        True if the module is eligible for processing, False otherwise.
    """
    specification = _MODULE_REGISTRY.get(module_key)
    if specification is None:
        return False
    return specification.check_eligibility(hardware_state=hardware_state)


def _resolve_hardware_state(session: SessionData) -> MesoscopeHardwareState:
    """Loads the Mesoscope-VR hardware state from the session's raw data directory.

    Args:
        session: The loaded session whose microcontroller modules are being parsed.

    Returns:
        The session's hardware configuration loaded from its raw-data YAML.

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
        hardware_state: The hardware configuration providing the ``cm_per_pulse`` conversion factor.

    Raises:
        ValueError: If either rotation event code carries no data payload, stores a null payload inside an otherwise
            decodable stream, spreads its payloads across more than one dtype, or decodes into a value count that is
            not a whole multiple of its message count.
    """
    cm_per_pulse = np.float64(hardware_state.cm_per_pulse)

    # Pre-declares the array types so mypy keeps the fallback-branch reassignments at the declared NDArray types
    # for the merge_event_streams() call.
    counterclockwise_timestamps: NDArray[np.uint64]
    counterclockwise_values: NDArray[np.float64]
    clockwise_timestamps: NDArray[np.uint64]
    clockwise_values: NDArray[np.float64]

    counterclockwise_timestamps, counterclockwise_values = get_event_data(
        partition=event_partition, event_code=_PRIMARY_EVENT_CODE, values_dtype=np.float64
    )
    clockwise_timestamps, clockwise_values = get_event_data(
        partition=event_partition, event_code=_SECONDARY_EVENT_CODE, values_dtype=np.float64
    )

    if counterclockwise_timestamps.size == 0:
        counterclockwise_timestamps = np.array([clockwise_timestamps[0] + 1], dtype=np.uint64)
        counterclockwise_values = np.array([0.0], dtype=np.float64)
    elif clockwise_timestamps.size == 0:
        clockwise_timestamps = np.array([counterclockwise_timestamps[0] + 1], dtype=np.uint64)
        clockwise_values = np.array([0.0], dtype=np.float64)

    timestamps, displacements = merge_event_streams(
        timestamps_a=counterclockwise_timestamps,
        values_a=counterclockwise_values,
        timestamps_b=clockwise_timestamps,
        values_b=-clockwise_values,
    )

    positions = np.cumsum(displacements * cm_per_pulse)
    positions = np.round(positions, decimals=8)
    positions[np.isclose(positions, -0.0) & np.signbit(positions)] = 0.0

    result_dataframe = pl.DataFrame({"time_us": timestamps, "traveled_distance_cm": positions})
    result_dataframe.write_ipc(file=output_file, compression="uncompressed")


def _parse_ttl_data(event_partition: dict[int, pl.DataFrame], output_file: Path, session: SessionData) -> None:
    """Extracts and saves TTL module data as a .feather file.

    Notes:
        Processes TTL input signals by detecting ON (event 51) and OFF (event 52) transitions. Ensures the final
        value is 0 to properly mark the end of the monitoring sequence. Both edge polarities are required, since the
        fluorescence assembly that consumes this feather pairs every rising edge with its falling edge.

    Args:
        event_partition: The event-code-keyed partition dictionary containing the raw TTL module event data.
        output_file: The path to the output .feather file.
        session: The loaded session whose TTL module data is parsed, named in the missing-edge error message.

    Raises:
        ValueError: If the extracted event data carries no rising or no falling edge timestamps.
    """
    on_timestamps = get_event_timestamps(partition=event_partition, event_code=_PRIMARY_EVENT_CODE)
    off_timestamps = get_event_timestamps(partition=event_partition, event_code=_SECONDARY_EVENT_CODE)

    missing_edges = tuple(
        polarity
        for polarity, edge_timestamps in (("rising", on_timestamps), ("falling", off_timestamps))
        if edge_timestamps.size == 0
    )
    if missing_edges:
        message = (
            f"Unable to parse the TTL module data for session '{session.session_name}'. The extracted event data "
            f"carries no {' and no '.join(missing_edges)} edge timestamps, so the TTL pulse train recorded in "
            f"'{output_file.name}' cannot be reconstructed."
        )
        console.error(message=message, error=ValueError)

    timestamps, triggers = merge_event_streams(
        timestamps_a=on_timestamps,
        values_a=np.ones(len(on_timestamps), dtype=np.uint8),
        timestamps_b=off_timestamps,
        values_b=np.zeros(len(off_timestamps), dtype=np.uint8),
    )

    if triggers[-1] != 0:
        timestamps = np.append(timestamps, values=timestamps[-1] + 1)
        triggers = np.append(triggers, values=np.uint8(0))

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
    minimum_brake_strength = np.float64(hardware_state.minimum_brake_strength)

    engaged_timestamps = get_event_timestamps(partition=event_partition, event_code=_PRIMARY_EVENT_CODE)
    disengaged_timestamps = get_event_timestamps(partition=event_partition, event_code=_SECONDARY_EVENT_CODE)

    timestamps, torques = merge_event_streams(
        timestamps_a=engaged_timestamps,
        values_a=np.full(len(engaged_timestamps), fill_value=maximum_brake_strength, dtype=np.float64),
        timestamps_b=disengaged_timestamps,
        values_b=np.full(len(disengaged_timestamps), fill_value=minimum_brake_strength, dtype=np.float64),
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
        shared timestamp grid. A session that never opened the valve produces a single zero-volume, zero-tone row
        stamped at the first close event, so the downstream assembly still finds the feather.

    Args:
        event_partition: The event-code-keyed partition dictionary containing the raw valve module event data.
        output_file: The path to the output .feather file.
        hardware_state: The hardware configuration providing valve calibration parameters.
    """
    scale_coefficient = np.float64(hardware_state.valve_scale_coefficient)
    nonlinearity_exponent = np.float64(hardware_state.valve_nonlinearity_exponent)

    open_timestamps = get_event_timestamps(partition=event_partition, event_code=_PRIMARY_EVENT_CODE)
    closed_timestamps = get_event_timestamps(partition=event_partition, event_code=_SECONDARY_EVENT_CODE)

    if open_timestamps.size == 0:
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

    edges = np.diff(volume, prepend=volume[0])
    rising_edges = np.where(edges == 1)[0]
    falling_edges = np.where(edges == -1)[0]

    reward_timestamps = timestamps[falling_edges]
    pulse_durations: NDArray[np.float64] = (timestamps[falling_edges] - timestamps[rising_edges]).astype(np.float64)

    volumes = np.cumsum(scale_coefficient * np.power(pulse_durations, nonlinearity_exponent))
    volumes = np.round(volumes, decimals=8)

    reward_timestamps = np.insert(reward_timestamps, obj=0, values=timestamps[0])
    volumes = np.insert(volumes, obj=0, values=0.0)

    tone_on_timestamps = get_event_timestamps(partition=event_partition, event_code=_TONE_ON_EVENT_CODE)
    tone_off_timestamps = get_event_timestamps(partition=event_partition, event_code=_TONE_OFF_EVENT_CODE)

    tone_timestamps, tone_states = merge_event_streams(
        timestamps_a=tone_on_timestamps,
        values_a=np.ones(len(tone_on_timestamps), dtype=np.uint8),
        timestamps_b=tone_off_timestamps,
        values_b=np.zeros(len(tone_off_timestamps), dtype=np.uint8),
    )

    if tone_states[-1] != 0:
        tone_timestamps = np.append(tone_timestamps, values=tone_timestamps[-1] + 1)
        tone_states = np.append(tone_states, values=np.uint8(0))

    shared_timestamps = np.unique(np.concatenate([tone_timestamps, reward_timestamps]))

    interpolated_reward = interpolate_data(
        source_coordinates=reward_timestamps,
        source_values=volumes,
        target_coordinates=shared_timestamps,
        is_discrete=True,
    )
    interpolated_tones = interpolate_data(
        source_coordinates=tone_timestamps,
        source_values=tone_states,
        target_coordinates=shared_timestamps,
        is_discrete=True,
    )

    result_dataframe = pl.DataFrame(
        {
            "time_us": shared_timestamps,
            "dispensed_water_volume_uL": interpolated_reward,
            "tone_state": interpolated_tones,
        }
    )
    result_dataframe.write_ipc(file=output_file, compression="uncompressed")


def _parse_gas_puff_data(
    event_partition: dict[int, pl.DataFrame],
    output_file: Path,
    hardware_state: MesoscopeHardwareState,  # noqa: ARG001
) -> None:
    """Extracts and saves gas puff valve module data as a .feather file.

    Notes:
        A session that never opened the valve produces a single closed-state, zero-count row stamped at the first
        close event, so the downstream assembly still finds the feather.

    Args:
        event_partition: The event-code-keyed partition dictionary containing the raw gas puff module event data.
        output_file: The path to the output .feather file.
        hardware_state: The hardware configuration (unused for gas puff processing but required for uniform dispatch).
    """
    open_timestamps = get_event_timestamps(partition=event_partition, event_code=_PRIMARY_EVENT_CODE)
    closed_timestamps = get_event_timestamps(partition=event_partition, event_code=_SECONDARY_EVENT_CODE)

    if open_timestamps.size == 0:
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

    # Differenced in a signed width, since an unsigned difference wraps a falling edge to 255 and never equals -1.
    edges = np.diff(states.astype(np.int16), prepend=np.int16(states[0]))
    falling_edges = edges == -1
    cumulative_puffs: NDArray[np.uint32] = np.cumsum(falling_edges, dtype=np.uint32)

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
        lick events. The interval between consecutive rising and falling transitions of the derived ``lick_state``
        corresponds to the tongue contact time with the lick tube.

    Args:
        event_partition: The event-code-keyed partition dictionary containing the raw lick sensor module event data.
        output_file: The path to the output .feather file.
        hardware_state: The hardware configuration providing the lick detection threshold.

    Raises:
        ValueError: If the hardware state carries no lick detection threshold, which means the module eligibility
            filter and the parser registry disagree. Also if the lick event code carries no data payload, stores a
            null payload inside an otherwise decodable stream, spreads its payloads across more than one dtype, or
            decodes into a value count that is not a whole multiple of its message count.
    """
    if hardware_state.lick_threshold is None:
        message = (
            "Unable to parse lick sensor module data. The 'lick_threshold' field is not configured on the "
            "hardware state, but _parse_lick_data was invoked. This indicates a mismatch between the module "
            "eligibility filter and the parser registry."
        )
        console.error(message=message, error=ValueError)

    lick_threshold = np.uint16(hardware_state.lick_threshold)

    timestamps, voltages = get_event_data(
        partition=event_partition, event_code=_PRIMARY_EVENT_CODE, values_dtype=np.uint16
    )

    sort_indices = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[sort_indices]
    voltages = voltages[sort_indices]

    licks = (voltages >= lick_threshold).astype(np.uint8)

    result_dataframe = pl.DataFrame({"time_us": timestamps, "voltage_12_bit_adc": voltages, "lick_state": licks})
    result_dataframe.write_ipc(file=output_file, compression="uncompressed")


def _parse_torque_data(
    event_partition: dict[int, pl.DataFrame], output_file: Path, hardware_state: MesoscopeHardwareState
) -> None:
    """Extracts and saves torque sensor module data as a .feather file.

    Notes:
        Converts raw ADC readings from CCW (event 51, positive) and CW (event 52, negative) torque events into
        physical torque values in Newton centimeters. Appends a trailing zero sample one microsecond after the last
        event when the sequence does not already end at zero, so the torque returns to rest at the end of the record.

    Args:
        event_partition: The event-code-keyed partition dictionary containing the raw torque sensor module event data.
        output_file: The path to the output .feather file.
        hardware_state: The hardware configuration providing the torque conversion factor.

    Raises:
        ValueError: If either rotation event code carries no data payload, stores a null payload inside an otherwise
            decodable stream, spreads its payloads across more than one dtype, or decodes into a value count that is
            not a whole multiple of its message count.
    """
    torque_per_adc_unit = np.float64(hardware_state.torque_per_adc_unit)

    # Pre-declares the array types so mypy keeps the fallback-branch reassignments at the declared NDArray types.
    counterclockwise_timestamps: NDArray[np.uint64]
    counterclockwise_values: NDArray[np.float64]
    clockwise_timestamps: NDArray[np.uint64]
    clockwise_values: NDArray[np.float64]

    counterclockwise_timestamps, counterclockwise_values = get_event_data(
        partition=event_partition, event_code=_PRIMARY_EVENT_CODE, values_dtype=np.float64
    )
    clockwise_timestamps, clockwise_values = get_event_data(
        partition=event_partition, event_code=_SECONDARY_EVENT_CODE, values_dtype=np.float64
    )

    if counterclockwise_timestamps.size == 0:
        counterclockwise_timestamps = np.array([clockwise_timestamps[0] + 1], dtype=np.uint64)
        counterclockwise_values = np.array([0.0], dtype=np.float64)
    elif clockwise_timestamps.size == 0:
        clockwise_timestamps = np.array([counterclockwise_timestamps[0] + 1], dtype=np.uint64)
        clockwise_values = np.array([0.0], dtype=np.float64)

    timestamps, torques = merge_event_streams(
        timestamps_a=counterclockwise_timestamps,
        values_a=counterclockwise_values * torque_per_adc_unit,
        timestamps_b=clockwise_timestamps,
        values_b=-clockwise_values * torque_per_adc_unit,
    )

    torques = np.round(torques, decimals=8)

    if torques[-1] != 0:
        timestamps = np.append(timestamps, values=timestamps[-1] + 1)
        torques = np.append(torques, values=0)

    torques[np.isclose(torques, -0.0) & np.signbit(torques)] = 0.0

    result_dataframe = pl.DataFrame({"time_us": timestamps, "torque_N_cm": torques})
    result_dataframe.write_ipc(file=output_file, compression="uncompressed")


def _parse_screen_data(
    event_partition: dict[int, pl.DataFrame], output_file: Path, hardware_state: MesoscopeHardwareState
) -> None:
    """Extracts and saves screen module data as a .feather file.

    Notes:
        Toggle pulses carry no absolute state, so the screen state is reconstructed by alternating from the initial
        configuration value on each rising edge. A session with no toggle pulse produces a single row carrying the
        initial screen state, stamped at the first off event.

    Args:
        event_partition: The event-code-keyed partition dictionary containing the raw screen module event data.
        output_file: The path to the output .feather file.
        hardware_state: The hardware configuration providing the initial screen state.
    """
    # check_eligibility() guarantees screens_initially_on is not None before this function runs, but the type
    # stub still advertises it as bool | None. Coercing to a plain int narrows the type for the downstream
    # screen_states arithmetic and yields the uint8 encoding it expects, where False or None maps to 0 and True to 1.
    initially_on: int = 1 if hardware_state.screens_initially_on else 0

    on_timestamps = get_event_timestamps(partition=event_partition, event_code=_PRIMARY_EVENT_CODE)
    off_timestamps = get_event_timestamps(partition=event_partition, event_code=_SECONDARY_EVENT_CODE)

    if on_timestamps.size == 0:
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

    edges = np.diff(triggers, prepend=0)
    rising_edges = np.where(edges == 1)[0]
    screen_timestamps = timestamps[rising_edges]

    screen_timestamps = np.concatenate(([timestamps[0]], screen_timestamps))

    state_count = len(screen_timestamps)
    screen_states: NDArray[np.uint8] = np.empty(state_count, dtype=np.uint8)
    screen_states[0] = initially_on
    screen_states[1:] = (initially_on + np.arange(1, state_count)) % 2

    result_dataframe = pl.DataFrame({"time_us": screen_timestamps, "screen_state": screen_states})
    result_dataframe.write_ipc(file=output_file, compression="uncompressed")
