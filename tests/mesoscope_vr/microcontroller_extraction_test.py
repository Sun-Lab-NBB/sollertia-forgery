"""Contains tests for the shared event-stream merge primitive and the Mesoscope-VR module parsers that consume it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pytest
from sollertia_shared_assets import MesoscopeHardwareState
from ataraxis_communication_interface import partition_events

from sollertia_forgery.mesoscope_vr.metadata import BehaviorDataFiles
from sollertia_forgery.mesoscope_vr.microcontrollers import (
    parse_lick,
    parse_brake,
    parse_valve,
    parse_screen,
    parse_torque,
    parse_encoder,
    parse_gas_puff,
    _parse_lick_data,
    _is_module_eligible,
    get_eligible_modules,
    parse_mesoscope_frame,
    get_module_event_codes,
)
from sollertia_forgery.shared_assets.microcontroller import merge_event_streams

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Sequence

    from sollertia_shared_assets import SessionData

_PRIMARY_EVENT_CODE: int = 51
"""The axci event code carrying each module's primary data stream."""

_SECONDARY_EVENT_CODE: int = 52
"""The axci event code carrying each module's complementary data stream."""

_TONE_ON_EVENT_CODE: int = 54
"""The axci event code marking the reward tone onset."""

_TONE_OFF_EVENT_CODE: int = 55
"""The axci event code marking the reward tone offset."""

_CM_PER_PULSE: float = 0.0057652
"""The wheel-encoder conversion factor the shared hardware state fixture records."""

_MAXIMUM_BRAKE_STRENGTH: float = 11.30234233
"""The engaged-brake torque the shared hardware state fixture records."""

_MINIMUM_BRAKE_STRENGTH: float = 0.42383811
"""The disengaged-brake torque the shared hardware state fixture records."""

_VALVE_SCALE_COEFFICIENT: float = 2.0e-07
"""The water-valve calibration scale coefficient the shared hardware state fixture records."""

_VALVE_NONLINEARITY_EXPONENT: float = 1.61941797
"""The water-valve calibration exponent the shared hardware state fixture records."""

_TORQUE_PER_ADC_UNIT: float = 0.00506377
"""The torque conversion factor the shared hardware state fixture records."""

_LICK_THRESHOLD: int = 600
"""The lick detection threshold the shared hardware state fixture records."""

type _MessageRow = tuple[int, int, bytes | None, str | None]


def _module_dataframe(rows: Sequence[_MessageRow]) -> pl.DataFrame:
    """Assembles a raw module DataFrame in the five-column schema the acquisition library writes.

    Args:
        rows: The per-message tuples of acquisition timestamp, event code, payload bytes, and payload dtype name.

    Returns:
        The assembled raw module DataFrame, ordered chronologically the way the extraction stage emits it.
    """
    ordered = sorted(rows, key=lambda row: row[0])
    return pl.DataFrame(
        {
            "timestamp_us": pl.Series([row[0] for row in ordered], dtype=pl.UInt64),
            "command": pl.Series([1] * len(ordered), dtype=pl.UInt8),
            "event": pl.Series([row[1] for row in ordered], dtype=pl.UInt8),
            "dtype": pl.Series([row[3] for row in ordered], dtype=pl.String),
            "data": pl.Series([row[2] for row in ordered], dtype=pl.Binary),
        }
    )


def _state_rows(event_code: int, timestamps: Sequence[int]) -> list[_MessageRow]:
    """Builds the message rows of a state-only event stream, which carries a timestamp and no payload.

    Args:
        event_code: The axci event code every built row records.
        timestamps: The acquisition timestamps, in microseconds, of the recorded events.

    Returns:
        The built message rows.
    """
    return [(int(timestamp), event_code, None, None) for timestamp in timestamps]


def _data_rows(
    event_code: int, timestamps: Sequence[int], values: Sequence[float], dtype_name: str
) -> list[_MessageRow]:
    """Builds the message rows of a data-carrying event stream, serializing one value per message.

    Args:
        event_code: The axci event code every built row records.
        timestamps: The acquisition timestamps, in microseconds, of the recorded events.
        values: The per-message payload values, one for each timestamp.
        dtype_name: The numpy dtype name with which the payload bytes are serialized.

    Returns:
        The built message rows.
    """
    payloads = np.asarray(values, dtype=dtype_name)
    return [
        (int(timestamp), event_code, payloads[index].tobytes(), dtype_name)
        for index, timestamp in enumerate(timestamps)
    ]


def _partition(*row_groups: Sequence[_MessageRow]) -> dict[int, pl.DataFrame]:
    """Builds the event partition the module parsers consume from the supplied message rows.

    Args:
        row_groups: The per-stream message row sequences to combine into one module DataFrame.

    Returns:
        The event-code-keyed partition the pipeline hands to a module parser.
    """
    rows = [row for group in row_groups for row in group]
    return partition_events(module_dataframe=_module_dataframe(rows=rows))


def _configure_hardware_state(session: SessionData, **overrides: Any) -> None:
    """Rewrites the session's hardware state snapshot with the supplied field overrides.

    Args:
        session: The session whose raw-data hardware state snapshot is rewritten.
        overrides: The hardware state field values to record in place of the fixture defaults.
    """
    state = MesoscopeHardwareState.from_yaml(file_path=session.raw_data.hardware_state_path)
    for name, value in overrides.items():
        setattr(state, name, value)
    state.to_yaml(file_path=session.raw_data.hardware_state_path)


def _read(output_directory: Path, data_file: BehaviorDataFiles) -> pl.DataFrame:
    """Reads back one behavior feather a Mesoscope-VR module parser wrote.

    Args:
        output_directory: The processed microcontroller-data directory into which the parser wrote.
        data_file: The canonical filename of the behavior feather to read.

    Returns:
        The parsed behavior DataFrame.
    """
    return pl.read_ipc(source=output_directory / data_file, memory_map=False)


@pytest.fixture
def output_directory(experiment_session: SessionData) -> Path:
    """Creates and returns the session's processed microcontroller-data directory.

    Args:
        experiment_session: The acquired Mesoscope-VR session against which the parsers run.

    Returns:
        The created output directory into which every module parser writes its behavior feather.
    """
    path = experiment_session.processed_data.microcontroller_data_path
    path.mkdir(parents=True, exist_ok=True)
    return path


# Event stream merging


def test_merge_event_streams_orders_values_chronologically() -> None:
    timestamps, values = merge_event_streams(
        timestamps_a=np.array([10, 30], dtype=np.uint64),
        values_a=np.array([1.0, 3.0], dtype=np.float64),
        timestamps_b=np.array([20, 40], dtype=np.uint64),
        values_b=np.array([2.0, 4.0], dtype=np.float64),
    )

    assert timestamps.tolist() == [10, 20, 30, 40]
    assert values.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_merge_event_streams_keeps_first_stream_ahead_on_ties() -> None:
    # The stable sort is what makes the merge reproducible when the two streams share a timestamp.
    timestamps, values = merge_event_streams(
        timestamps_a=np.array([10], dtype=np.uint64),
        values_a=np.array([1.0], dtype=np.float64),
        timestamps_b=np.array([10], dtype=np.uint64),
        values_b=np.array([2.0], dtype=np.float64),
    )

    assert timestamps.tolist() == [10, 10]
    assert values.tolist() == [1.0, 2.0]


def test_merge_event_streams_preserves_the_full_microsecond_key_width() -> None:
    # The merged keys are microseconds since the UTC epoch, and 32 bits span only about seventy-one minutes, so
    # narrowing them wraps every event of every session and sorts it ahead of the events that truly preceded it.
    timestamps, values = merge_event_streams(
        timestamps_a=np.array([100, 4_294_967_396], dtype=np.uint64),
        values_a=np.array([1.0, 3.0], dtype=np.float64),
        timestamps_b=np.array([200], dtype=np.uint64),
        values_b=np.array([2.0], dtype=np.float64),
    )

    assert timestamps.dtype == np.uint64
    assert timestamps.tolist() == [100, 200, 4_294_967_396]
    assert values.tolist() == [1.0, 2.0, 3.0]


# Module eligibility


def test_get_eligible_modules_returns_every_configured_module(experiment_session: SessionData) -> None:
    # The shared hardware state fixture configures and marks used every module the Mesoscope-VR system parses.
    assert get_eligible_modules(session=experiment_session) == set(get_module_event_codes())


def test_get_eligible_modules_drops_unconfigured_and_unused_modules(experiment_session: SessionData) -> None:
    _configure_hardware_state(session=experiment_session, cm_per_pulse=None, delivered_gas_puffs=False)

    eligible = get_eligible_modules(session=experiment_session)

    assert (2, 1) not in eligible  # The encoder carries no conversion factor.
    assert (5, 2) not in eligible  # The gas puff valve was not used.
    assert (4, 1) in eligible


def test_get_module_event_codes_returns_a_fresh_mapping() -> None:
    codes = get_module_event_codes()
    codes.pop((2, 1))

    assert (2, 1) in get_module_event_codes()


def test_is_module_eligible_rejects_unregistered_module(experiment_session: SessionData) -> None:
    state = MesoscopeHardwareState.from_yaml(file_path=experiment_session.raw_data.hardware_state_path)

    assert not _is_module_eligible(module_key=(99, 9), hardware_state=state)


def test_parser_requires_the_hardware_state_snapshot(experiment_session: SessionData, output_directory: Path) -> None:
    experiment_session.raw_data.hardware_state_path.unlink()

    with pytest.raises(FileNotFoundError, match=r"No hardware state YAML\s+file was found"):
        parse_encoder(event_partition={}, output_directory=output_directory, session=experiment_session)


# Encoder parser


def test_parse_encoder_writes_cumulative_traveled_distance(
    experiment_session: SessionData, output_directory: Path
) -> None:
    partition = _partition(
        _data_rows(event_code=_PRIMARY_EVENT_CODE, timestamps=[10, 30], values=[1, 2], dtype_name="uint32"),
        _data_rows(event_code=_SECONDARY_EVENT_CODE, timestamps=[20], values=[1], dtype_name="uint32"),
    )

    parse_encoder(event_partition=partition, output_directory=output_directory, session=experiment_session)

    result = _read(output_directory, BehaviorDataFiles.ENCODER)
    assert result["time_us"].to_list() == [10, 20, 30]
    assert result["traveled_distance_cm"].to_list() == pytest.approx(
        [_CM_PER_PULSE, 0.0, 2.0 * _CM_PER_PULSE], abs=1e-9
    )


def test_parse_encoder_substitutes_a_missing_counterclockwise_stream(
    experiment_session: SessionData, output_directory: Path
) -> None:
    partition = _partition(
        _data_rows(event_code=_SECONDARY_EVENT_CODE, timestamps=[20], values=[3], dtype_name="uint32")
    )

    parse_encoder(event_partition=partition, output_directory=output_directory, session=experiment_session)

    result = _read(output_directory, BehaviorDataFiles.ENCODER)
    # The substituted counterclockwise sample is stamped one microsecond after the clockwise event and adds nothing.
    assert result["time_us"].to_list() == [20, 21]
    assert result["traveled_distance_cm"].to_list() == pytest.approx([-3.0 * _CM_PER_PULSE] * 2, abs=1e-9)


def test_parse_encoder_substitutes_a_missing_clockwise_stream(
    experiment_session: SessionData, output_directory: Path
) -> None:
    partition = _partition(_data_rows(event_code=_PRIMARY_EVENT_CODE, timestamps=[10], values=[2], dtype_name="uint32"))

    parse_encoder(event_partition=partition, output_directory=output_directory, session=experiment_session)

    result = _read(output_directory, BehaviorDataFiles.ENCODER)
    assert result["time_us"].to_list() == [10, 11]
    assert result["traveled_distance_cm"].to_list() == pytest.approx([2.0 * _CM_PER_PULSE] * 2, abs=1e-9)


def test_parse_encoder_normalizes_negative_zero_distance(
    experiment_session: SessionData, output_directory: Path
) -> None:
    # Negating a zero-pulse clockwise sample yields a negative zero, which the parser rewrites to a positive zero.
    partition = _partition(
        _data_rows(event_code=_SECONDARY_EVENT_CODE, timestamps=[20], values=[0], dtype_name="uint32")
    )

    parse_encoder(event_partition=partition, output_directory=output_directory, session=experiment_session)

    distances = _read(output_directory, BehaviorDataFiles.ENCODER)["traveled_distance_cm"].to_numpy()
    assert distances.tolist() == [0.0, 0.0]
    assert not np.signbit(distances).any()


def test_parse_encoder_skips_an_unconfigured_module(experiment_session: SessionData, output_directory: Path) -> None:
    _configure_hardware_state(session=experiment_session, cm_per_pulse=None)
    partition = _partition(_data_rows(event_code=_PRIMARY_EVENT_CODE, timestamps=[10], values=[1], dtype_name="uint32"))

    parse_encoder(event_partition=partition, output_directory=output_directory, session=experiment_session)

    assert not (output_directory / BehaviorDataFiles.ENCODER).exists()


# Mesoscope frame TTL parser


def test_parse_mesoscope_frame_writes_the_pulse_train(experiment_session: SessionData, output_directory: Path) -> None:
    partition = _partition(
        _state_rows(_PRIMARY_EVENT_CODE, [10, 30]),
        _state_rows(_SECONDARY_EVENT_CODE, [20, 40]),
    )

    parse_mesoscope_frame(event_partition=partition, output_directory=output_directory, session=experiment_session)

    result = _read(output_directory, BehaviorDataFiles.MESOSCOPE_FRAME)
    assert result["time_us"].to_list() == [10, 20, 30, 40]
    assert result["ttl_state"].to_list() == [1, 0, 1, 0]


def test_parse_mesoscope_frame_appends_a_trailing_low_sample(
    experiment_session: SessionData, output_directory: Path
) -> None:
    partition = _partition(
        _state_rows(_PRIMARY_EVENT_CODE, [10, 30]),
        _state_rows(_SECONDARY_EVENT_CODE, [20]),
    )

    parse_mesoscope_frame(event_partition=partition, output_directory=output_directory, session=experiment_session)

    result = _read(output_directory, BehaviorDataFiles.MESOSCOPE_FRAME)
    assert result["time_us"].to_list() == [10, 20, 30, 31]
    assert result["ttl_state"].to_list() == [1, 0, 1, 0]
    # The appended sample has to carry the state column's own width, or the whole column widens to a signed 64-bit
    # integer and the feather no longer matches the schema the pulse train carries when it ends low.
    assert result.schema["ttl_state"] == pl.UInt8


def test_parse_mesoscope_frame_requires_falling_edges(experiment_session: SessionData, output_directory: Path) -> None:
    partition = _partition(_state_rows(_PRIMARY_EVENT_CODE, [10]))

    with pytest.raises(ValueError, match=r"carries no\s+falling\s+edge timestamps"):
        parse_mesoscope_frame(event_partition=partition, output_directory=output_directory, session=experiment_session)


def test_parse_mesoscope_frame_requires_rising_edges(experiment_session: SessionData, output_directory: Path) -> None:
    partition = _partition(_state_rows(_SECONDARY_EVENT_CODE, [10]))

    with pytest.raises(ValueError, match=r"carries no\s+rising\s+edge timestamps"):
        parse_mesoscope_frame(event_partition=partition, output_directory=output_directory, session=experiment_session)


def test_parse_mesoscope_frame_names_both_missing_polarities(
    experiment_session: SessionData, output_directory: Path
) -> None:
    with pytest.raises(ValueError, match=r"carries no\s+rising\s+and no falling\s+edge timestamps"):
        parse_mesoscope_frame(event_partition={}, output_directory=output_directory, session=experiment_session)


def test_parse_mesoscope_frame_skips_an_unused_module(experiment_session: SessionData, output_directory: Path) -> None:
    _configure_hardware_state(session=experiment_session, recorded_mesoscope_ttl=False)
    partition = _partition(_state_rows(_PRIMARY_EVENT_CODE, [10]))

    parse_mesoscope_frame(event_partition=partition, output_directory=output_directory, session=experiment_session)

    assert not (output_directory / BehaviorDataFiles.MESOSCOPE_FRAME).exists()


# Brake parser


def test_parse_brake_writes_engagement_torques(experiment_session: SessionData, output_directory: Path) -> None:
    partition = _partition(
        _state_rows(_PRIMARY_EVENT_CODE, [10, 30]),
        _state_rows(_SECONDARY_EVENT_CODE, [20]),
    )

    parse_brake(event_partition=partition, output_directory=output_directory, session=experiment_session)

    result = _read(output_directory=output_directory, data_file=BehaviorDataFiles.BRAKE)
    assert result["time_us"].to_list() == [10, 20, 30]
    assert result["brake_torque_N_cm"].to_list() == pytest.approx(
        [_MAXIMUM_BRAKE_STRENGTH, _MINIMUM_BRAKE_STRENGTH, _MAXIMUM_BRAKE_STRENGTH]
    )


def test_parse_brake_skips_an_unconfigured_module(experiment_session: SessionData, output_directory: Path) -> None:
    _configure_hardware_state(session=experiment_session, maximum_brake_strength=None)
    partition = _partition(_state_rows(_PRIMARY_EVENT_CODE, [10]))

    parse_brake(event_partition=partition, output_directory=output_directory, session=experiment_session)

    assert not (output_directory / BehaviorDataFiles.BRAKE).exists()


# Water valve parser


def test_parse_valve_writes_cumulative_volume_and_tone_state(
    experiment_session: SessionData, output_directory: Path
) -> None:
    partition = _partition(
        _state_rows(event_code=_SECONDARY_EVENT_CODE, timestamps=[5, 20, 40]),
        _state_rows(_PRIMARY_EVENT_CODE, [10, 30]),
        _state_rows(_TONE_ON_EVENT_CODE, [15]),
        _state_rows(event_code=_TONE_OFF_EVENT_CODE, timestamps=[25]),
    )

    parse_valve(event_partition=partition, output_directory=output_directory, session=experiment_session)

    result = _read(output_directory, BehaviorDataFiles.VALVE)
    # Both pulses stay open for ten microseconds, so each contributes the same calibrated volume.
    pulse_volume = round(_VALVE_SCALE_COEFFICIENT * (10.0**_VALVE_NONLINEARITY_EXPONENT), 8)
    both_pulses = round(2.0 * _VALVE_SCALE_COEFFICIENT * (10.0**_VALVE_NONLINEARITY_EXPONENT), 8)
    assert result["time_us"].to_list() == [5, 15, 20, 25, 40]
    assert result["dispensed_water_volume_uL"].to_list() == [0.0, 0.0, pulse_volume, pulse_volume, both_pulses]
    assert result["tone_state"].to_list() == [1, 1, 1, 0, 0]


def test_parse_valve_appends_a_trailing_tone_off_sample(
    experiment_session: SessionData, output_directory: Path
) -> None:
    partition = _partition(
        _state_rows(event_code=_SECONDARY_EVENT_CODE, timestamps=[5, 20]),
        _state_rows(_PRIMARY_EVENT_CODE, [10]),
        _state_rows(_TONE_ON_EVENT_CODE, [15]),
    )

    parse_valve(event_partition=partition, output_directory=output_directory, session=experiment_session)

    result = _read(output_directory, BehaviorDataFiles.VALVE)
    pulse_volume = round(_VALVE_SCALE_COEFFICIENT * (10.0**_VALVE_NONLINEARITY_EXPONENT), 8)
    # The tone never turned off, so the parser stamps a closing sample one microsecond after the last tone event.
    assert result["time_us"].to_list() == [5, 15, 16, 20]
    assert result["tone_state"].to_list() == [1, 1, 0, 0]
    assert result["dispensed_water_volume_uL"].to_list() == [0.0, 0.0, 0.0, pulse_volume]
    # The appended closing sample has to carry the tone column's own width. A wider sample widens the whole column to a
    # signed 64-bit integer, so the feather stops matching the schema of a session whose tone ended on its own.
    assert result.schema["tone_state"] == pl.UInt8


def test_parse_valve_writes_a_single_row_when_the_valve_never_opened(
    experiment_session: SessionData, output_directory: Path
) -> None:
    partition = _partition(_state_rows(_SECONDARY_EVENT_CODE, [40, 50]))

    parse_valve(event_partition=partition, output_directory=output_directory, session=experiment_session)

    result = _read(output_directory, BehaviorDataFiles.VALVE)
    assert result.to_dicts() == [{"time_us": 40, "dispensed_water_volume_uL": 0.0, "tone_state": 0}]


def test_parse_valve_skips_an_unconfigured_module(experiment_session: SessionData, output_directory: Path) -> None:
    _configure_hardware_state(session=experiment_session, valve_scale_coefficient=None)
    partition = _partition(_state_rows(_PRIMARY_EVENT_CODE, [10]))

    parse_valve(event_partition=partition, output_directory=output_directory, session=experiment_session)

    assert not (output_directory / BehaviorDataFiles.VALVE).exists()


# Gas puff parser


def test_parse_gas_puff_writes_the_puff_state_train(experiment_session: SessionData, output_directory: Path) -> None:
    partition = _partition(
        _state_rows(_PRIMARY_EVENT_CODE, [10, 30]),
        _state_rows(_SECONDARY_EVENT_CODE, [20, 40]),
    )

    parse_gas_puff(event_partition=partition, output_directory=output_directory, session=experiment_session)

    result = _read(output_directory, BehaviorDataFiles.GAS_PUFF)
    assert result["time_us"].to_list() == [10, 20, 30, 40]
    assert result["puff_state"].to_list() == [1, 0, 1, 0]
    assert result["cumulative_puff_count"].to_list() == [0, 1, 1, 2]
    assert result.schema["cumulative_puff_count"] == pl.UInt32


def test_parse_gas_puff_writes_both_edges_of_a_single_delivered_puff(
    experiment_session: SessionData, output_directory: Path
) -> None:
    # A session that delivered exactly one puff still carries a genuine open and close pair, so it must not be
    # mistaken for the never-delivered case, which would report the puff as never having happened.
    partition = _partition(_state_rows(_PRIMARY_EVENT_CODE, [10]), _state_rows(_SECONDARY_EVENT_CODE, [20]))

    parse_gas_puff(event_partition=partition, output_directory=output_directory, session=experiment_session)

    result = _read(output_directory, BehaviorDataFiles.GAS_PUFF)
    assert result["time_us"].to_list() == [10, 20]
    assert result["puff_state"].to_list() == [1, 0]
    assert result["cumulative_puff_count"].to_list() == [0, 1]


def test_parse_gas_puff_writes_a_single_row_when_no_puff_was_delivered(
    experiment_session: SessionData, output_directory: Path
) -> None:
    partition = _partition(_state_rows(_SECONDARY_EVENT_CODE, [40, 50]))

    parse_gas_puff(event_partition=partition, output_directory=output_directory, session=experiment_session)

    result = _read(output_directory, BehaviorDataFiles.GAS_PUFF)
    assert result.to_dicts() == [{"time_us": 40, "puff_state": 0, "cumulative_puff_count": 0}]


def test_parse_gas_puff_skips_an_unused_module(experiment_session: SessionData, output_directory: Path) -> None:
    _configure_hardware_state(session=experiment_session, delivered_gas_puffs=False)
    partition = _partition(_state_rows(_PRIMARY_EVENT_CODE, [10]))

    parse_gas_puff(event_partition=partition, output_directory=output_directory, session=experiment_session)

    assert not (output_directory / BehaviorDataFiles.GAS_PUFF).exists()


# Lick sensor parser


def test_parse_lick_thresholds_the_sensor_voltage(experiment_session: SessionData, output_directory: Path) -> None:
    partition = _partition(
        _data_rows(
            event_code=_PRIMARY_EVENT_CODE,
            timestamps=[30, 10, 20, 40],
            values=[100, 700, 650, _LICK_THRESHOLD],
            dtype_name="uint16",
        )
    )

    parse_lick(event_partition=partition, output_directory=output_directory, session=experiment_session)

    result = _read(output_directory=output_directory, data_file=BehaviorDataFiles.LICK)
    # The partition is written chronologically, so the voltages follow their timestamps rather than the input order.
    assert result["time_us"].to_list() == [10, 20, 30, 40]
    assert result["voltage_12_bit_adc"].to_list() == [700, 650, 100, _LICK_THRESHOLD]
    # The sensor reports integer ADC counts against an integer threshold, so a reading landing exactly on the
    # calibrated threshold is a routine contact and counts as a lick rather than as a gap in one.
    assert result["lick_state"].to_list() == [1, 1, 0, 1]


def test_parse_lick_data_rejects_an_unset_threshold(output_directory: Path) -> None:
    # The eligibility filter never routes an unconfigured module to its parser, so the guard is checked directly.
    partition = _partition(_data_rows(_PRIMARY_EVENT_CODE, [10], [700], "uint16"))

    with pytest.raises(ValueError, match=r"'lick_threshold' field is not\s+configured"):
        _parse_lick_data(
            event_partition=partition,
            output_file=output_directory / BehaviorDataFiles.LICK,
            hardware_state=MesoscopeHardwareState(lick_threshold=None),
        )


def test_parse_lick_skips_an_unconfigured_module(experiment_session: SessionData, output_directory: Path) -> None:
    _configure_hardware_state(session=experiment_session, lick_threshold=None)
    partition = _partition(_data_rows(_PRIMARY_EVENT_CODE, [10], [700], "uint16"))

    parse_lick(event_partition=partition, output_directory=output_directory, session=experiment_session)

    assert not (output_directory / BehaviorDataFiles.LICK).exists()


# Torque sensor parser


def test_parse_torque_writes_signed_torque_returning_to_rest(
    experiment_session: SessionData, output_directory: Path
) -> None:
    partition = _partition(
        _data_rows(event_code=_PRIMARY_EVENT_CODE, timestamps=[10, 30], values=[10, 0], dtype_name="uint16"),
        _data_rows(event_code=_SECONDARY_EVENT_CODE, timestamps=[20], values=[5], dtype_name="uint16"),
    )

    parse_torque(event_partition=partition, output_directory=output_directory, session=experiment_session)

    result = _read(output_directory, BehaviorDataFiles.TORQUE)
    # The record already ends at rest, so no trailing sample is appended.
    assert result["time_us"].to_list() == [10, 20, 30]
    assert result["torque_N_cm"].to_list() == pytest.approx(
        [10.0 * _TORQUE_PER_ADC_UNIT, -5.0 * _TORQUE_PER_ADC_UNIT, 0.0], abs=1e-9
    )


def test_parse_torque_appends_a_trailing_rest_sample(experiment_session: SessionData, output_directory: Path) -> None:
    partition = _partition(
        _data_rows(event_code=_PRIMARY_EVENT_CODE, timestamps=[10], values=[10], dtype_name="uint16"),
        _data_rows(event_code=_SECONDARY_EVENT_CODE, timestamps=[20, 30], values=[0, 5], dtype_name="uint16"),
    )

    parse_torque(event_partition=partition, output_directory=output_directory, session=experiment_session)

    result = _read(output_directory, BehaviorDataFiles.TORQUE)
    torques = result["torque_N_cm"].to_numpy()
    assert result["time_us"].to_list() == [10, 20, 30, 31]
    assert torques.tolist() == pytest.approx(
        [10.0 * _TORQUE_PER_ADC_UNIT, 0.0, -5.0 * _TORQUE_PER_ADC_UNIT, 0.0], abs=1e-9
    )
    # Negating the zero-valued clockwise sample yields a negative zero, which the parser rewrites to a positive zero.
    assert not np.signbit(torques[[1, 3]]).any()


def test_parse_torque_substitutes_a_missing_counterclockwise_stream(
    experiment_session: SessionData, output_directory: Path
) -> None:
    partition = _partition(
        _data_rows(event_code=_SECONDARY_EVENT_CODE, timestamps=[20], values=[4], dtype_name="uint16")
    )

    parse_torque(event_partition=partition, output_directory=output_directory, session=experiment_session)

    result = _read(output_directory, BehaviorDataFiles.TORQUE)
    assert result["time_us"].to_list() == [20, 21]
    assert result["torque_N_cm"].to_list() == pytest.approx([-4.0 * _TORQUE_PER_ADC_UNIT, 0.0], abs=1e-9)


def test_parse_torque_substitutes_a_missing_clockwise_stream(
    experiment_session: SessionData, output_directory: Path
) -> None:
    partition = _partition(_data_rows(_PRIMARY_EVENT_CODE, [10], [4], "uint16"))

    parse_torque(event_partition=partition, output_directory=output_directory, session=experiment_session)

    result = _read(output_directory, BehaviorDataFiles.TORQUE)
    # The substituted clockwise sample already rests at zero, so no further trailing sample is appended.
    assert result["time_us"].to_list() == [10, 11]
    assert result["torque_N_cm"].to_list() == pytest.approx([4.0 * _TORQUE_PER_ADC_UNIT, 0.0], abs=1e-9)


def test_parse_torque_skips_an_unconfigured_module(experiment_session: SessionData, output_directory: Path) -> None:
    _configure_hardware_state(session=experiment_session, torque_per_adc_unit=None)
    partition = _partition(_data_rows(_PRIMARY_EVENT_CODE, [10], [4], "uint16"))

    parse_torque(event_partition=partition, output_directory=output_directory, session=experiment_session)

    assert not (output_directory / BehaviorDataFiles.TORQUE).exists()


# Screen parser


def test_parse_screen_alternates_state_on_every_toggle(experiment_session: SessionData, output_directory: Path) -> None:
    partition = _partition(
        _state_rows(event_code=_PRIMARY_EVENT_CODE, timestamps=[20, 40]),
        _state_rows(event_code=_SECONDARY_EVENT_CODE, timestamps=[10, 30]),
    )

    parse_screen(event_partition=partition, output_directory=output_directory, session=experiment_session)

    result = _read(output_directory, BehaviorDataFiles.SCREEN)
    # The screens start off, so the first toggle turns them on and the second turns them back off.
    assert result["time_us"].to_list() == [10, 20, 40]
    assert result["screen_state"].to_list() == [0, 1, 0]


def test_parse_screen_starts_from_the_recorded_initial_state(
    experiment_session: SessionData, output_directory: Path
) -> None:
    _configure_hardware_state(experiment_session, screens_initially_on=True)
    partition = _partition(
        _state_rows(event_code=_PRIMARY_EVENT_CODE, timestamps=[20]),
        _state_rows(_SECONDARY_EVENT_CODE, [10]),
    )

    parse_screen(event_partition=partition, output_directory=output_directory, session=experiment_session)

    result = _read(output_directory, BehaviorDataFiles.SCREEN)
    assert result["time_us"].to_list() == [10, 20]
    assert result["screen_state"].to_list() == [1, 0]


def test_parse_screen_writes_a_single_row_without_toggles(
    experiment_session: SessionData, output_directory: Path
) -> None:
    partition = _partition(_state_rows(_SECONDARY_EVENT_CODE, [10, 20]))

    parse_screen(event_partition=partition, output_directory=output_directory, session=experiment_session)

    result = _read(output_directory, BehaviorDataFiles.SCREEN)
    assert result.to_dicts() == [{"time_us": 10, "screen_state": 0}]


def test_parse_screen_carries_the_recorded_initial_state_into_the_single_row_without_toggles(
    experiment_session: SessionData, output_directory: Path
) -> None:
    # Toggle pulses carry no absolute state, so this branch is the only route by which a session that ran with the
    # screens on and never toggled them reports them as on.
    _configure_hardware_state(experiment_session, screens_initially_on=True)
    partition = _partition(_state_rows(_SECONDARY_EVENT_CODE, [10, 20]))

    parse_screen(event_partition=partition, output_directory=output_directory, session=experiment_session)

    result = _read(output_directory, BehaviorDataFiles.SCREEN)
    assert result.to_dicts() == [{"time_us": 10, "screen_state": 1}]


def test_parse_screen_skips_an_unconfigured_module(experiment_session: SessionData, output_directory: Path) -> None:
    _configure_hardware_state(session=experiment_session, screens_initially_on=None)
    partition = _partition(_state_rows(_PRIMARY_EVENT_CODE, [10]))

    parse_screen(event_partition=partition, output_directory=output_directory, session=experiment_session)

    assert not (output_directory / BehaviorDataFiles.SCREEN).exists()
