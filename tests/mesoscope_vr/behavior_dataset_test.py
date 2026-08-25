"""Contains tests for the Mesoscope-VR behavior dataset assembler, its running-speed kernel, and the Mesoscope-VR
metadata schema.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from sollertia_shared_assets import RawDataFiles, MesoscopeHardwareState

from sollertia_forgery.mesoscope_vr.metadata import (
    _COLUMN_DESCRIPTIONS,
    MESOSCOPE_COLUMN_DESCRIPTIONS,
    DatasetColumn,
    VideoDataFiles,
    BehaviorDataFiles,
)
from sollertia_forgery.mesoscope_vr.behavior_dataset import _calculate_running_speed, assemble_behavior_dataset

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

_BASE_TIME_US = 1_700_000_000_000_000
"""The acquisition-clock origin every synthetic feather is anchored to, sized like a real UTC microsecond stamp."""

_SAMPLE_INTERVAL_US = 6_000_000
"""The spacing of the reference time vector, chosen so each sample advances the elapsed-minutes column by 0.1."""

_SAMPLE_COUNT = 10
"""The number of samples in the reference time vector."""

_ENCODER_PAIR_OFFSET_US = 50_000
"""The interval by which the first encoder sample of each pair precedes its reference sample, half the speed
window."""

_MINIMUM_BRAKE_STRENGTH = 1.0
"""The brake-engagement threshold written into the synthetic hardware state."""

_SYSTEM_STATE_CODES: dict[str, int] = {"idle": 0, "rest": 1, "run": 2}
"""The state-name to state-code mapping written into the synthetic hardware state."""


def _reference_time_vector() -> NDArray[np.uint64]:
    """Builds the reference time vector every assembly test aligns its dataset to.

    Returns:
        A ten-sample microsecond vector spaced by the shared sampling interval.
    """
    return np.array(
        [_BASE_TIME_US + index * _SAMPLE_INTERVAL_US for index in range(_SAMPLE_COUNT)],
        dtype=np.uint64,
    )


def _sample_time(index: int) -> int:
    """Resolves the acquisition timestamp of the requested reference sample.

    Args:
        index: The zero-based index of the reference sample.

    Returns:
        The sample's timestamp in microseconds elapsed since the UTC epoch onset.
    """
    return _BASE_TIME_US + index * _SAMPLE_INTERVAL_US


def _make_input_directories(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Creates the three input directories the behavior assembler reads.

    Args:
        tmp_path: The temporary directory to build the input tree under.

    Returns:
        A tuple of the microcontroller-data, runtime-data, and raw-data directories.
    """
    directories = tuple(tmp_path.joinpath(name) for name in ("microcontroller_data", "runtime_data", "raw_data"))
    for directory in directories:
        directory.mkdir()
    return directories[0], directories[1], directories[2]


def _write_hardware_state(raw_data_path: Path, **overrides: object) -> None:
    """Writes the session's hardware state snapshot into the raw data directory.

    Args:
        raw_data_path: The raw data directory the snapshot is written into.
        **overrides: Field values replacing the defaults of the synthetic hardware state.
    """
    arguments: dict[str, object] = {
        "system_state_codes": dict(_SYSTEM_STATE_CODES),
        "minimum_brake_strength": _MINIMUM_BRAKE_STRENGTH,
    }
    arguments.update(overrides)
    MesoscopeHardwareState(**arguments).to_yaml(file_path=raw_data_path.joinpath(RawDataFiles.HARDWARE_STATE))


def _write_required_feathers(microcontroller_data_path: Path, runtime_data_path: Path) -> None:
    """Writes the valve, lick, and system-state feathers every session type produces.

    The valve stream plays a wet tone spanning samples four and five and a dry tone spanning samples eight and nine.
    The system walks idle for samples zero through two, run for samples three through six, and rest afterwards.

    Args:
        microcontroller_data_path: The directory receiving the module feathers.
        runtime_data_path: The directory receiving the runtime feathers.
    """
    pl.DataFrame(
        {
            "time_us": np.array(
                [_sample_time(0), _sample_time(4), _sample_time(5), _sample_time(6), _sample_time(8)], dtype=np.uint64
            ),
            "dispensed_water_volume_uL": np.array([0.0, 0.0, 5.0, 5.0, 5.0], dtype=np.float64),
            "tone_state": np.array([0, 1, 1, 0, 1], dtype=np.uint8),
        }
    ).write_ipc(file=microcontroller_data_path.joinpath(BehaviorDataFiles.VALVE))

    pl.DataFrame(
        {
            "time_us": np.array([_sample_time(0), _sample_time(4), _sample_time(5)], dtype=np.uint64),
            "lick_state": np.array([0, 1, 0], dtype=np.uint8),
        }
    ).write_ipc(file=microcontroller_data_path.joinpath(BehaviorDataFiles.LICK))

    pl.DataFrame(
        {
            "time_us": np.array([_sample_time(0), _sample_time(3), _sample_time(7)], dtype=np.uint64),
            "system_state": np.array([0, 2, 1], dtype=np.uint8),
        }
    ).write_ipc(file=runtime_data_path.joinpath(BehaviorDataFiles.SYSTEM_STATE))


def _write_encoder_feather(microcontroller_data_path: Path) -> None:
    """Writes the encoder feather as one sample pair ending on each reference sample.

    Each pair is spaced by half the running-speed window, so the speed reported at a reference sample is the pair's
    displacement over that half window. The animal travels only across the run samples and once during rest.

    Args:
        microcontroller_data_path: The directory receiving the encoder feather.
    """
    pair_distances = [
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 1.0),
        (5.0, 6.5),
        (12.0, 14.0),
        (20.0, 21.5),
        (25.0, 26.0),
        (26.0, 26.0),
        (26.0, 26.0),
    ]

    times: list[int] = []
    distances: list[float] = []
    for index, (before, at) in enumerate(pair_distances):
        times.extend((_sample_time(index) - _ENCODER_PAIR_OFFSET_US, _sample_time(index)))
        distances.extend((before, at))

    pl.DataFrame(
        {
            "time_us": np.array(times, dtype=np.uint64),
            "traveled_distance_cm": np.array(distances, dtype=np.float64),
        }
    ).write_ipc(file=microcontroller_data_path.joinpath(BehaviorDataFiles.ENCODER))


def _write_screen_feather(microcontroller_data_path: Path) -> None:
    """Writes the screen feather, switching the displays on at the sixth reference sample.

    Args:
        microcontroller_data_path: The directory receiving the screen feather.
    """
    pl.DataFrame(
        {
            "time_us": np.array([_sample_time(0), _sample_time(5)], dtype=np.uint64),
            "screen_state": np.array([0, 1], dtype=np.uint8),
        }
    ).write_ipc(file=microcontroller_data_path.joinpath(BehaviorDataFiles.SCREEN))


def _write_brake_feather(microcontroller_data_path: Path) -> None:
    """Writes the brake feather, releasing the brake across the run samples and re-engaging it during rest.

    The released samples carry exactly the brake threshold, since the module parser fills every disengagement with
    the hardware state's minimum brake strength rather than with zero: a released brake still drags through its
    mechanical coupling.

    Args:
        microcontroller_data_path: The directory receiving the brake feather.
    """
    pl.DataFrame(
        {
            "time_us": np.array([_sample_time(0), _sample_time(3), _sample_time(7)], dtype=np.uint64),
            "brake_torque_N_cm": np.array([5.0, _MINIMUM_BRAKE_STRENGTH, 5.0], dtype=np.float64),
        }
    ).write_ipc(file=microcontroller_data_path.joinpath(BehaviorDataFiles.BRAKE))


def _write_torque_feather(microcontroller_data_path: Path) -> None:
    """Writes the torque feather as a linear ramp of one N·cm per reference sample.

    Args:
        microcontroller_data_path: The directory receiving the torque feather.
    """
    pl.DataFrame(
        {
            "time_us": np.array([_sample_time(0), _sample_time(index=9)], dtype=np.uint64),
            "torque_N_cm": np.array([0.0, 9.0], dtype=np.float64),
        }
    ).write_ipc(file=microcontroller_data_path.joinpath(BehaviorDataFiles.TORQUE))


def test_assemble_behavior_dataset_emits_only_the_mandatory_columns(tmp_path: Path) -> None:
    """Verifies that a session carrying just the valve, lick, and system-state feathers assembles the shared columns.

    The optional encoder, screen, brake, and torque sources are absent, so the assembled dataset holds the clock, the
    lick and water traces, the reward classification, and the system state alone.
    """
    microcontroller_data_path, runtime_data_path, raw_data_path = _make_input_directories(tmp_path=tmp_path)
    _write_hardware_state(raw_data_path=raw_data_path)
    _write_required_feathers(microcontroller_data_path=microcontroller_data_path, runtime_data_path=runtime_data_path)

    behavior_data = assemble_behavior_dataset(
        microcontroller_data_path=microcontroller_data_path,
        runtime_data_path=runtime_data_path,
        raw_data_path=raw_data_path,
        reference_time=_reference_time_vector(),
    )

    assert behavior_data.columns == ["time_us", "elapsed_minutes", "lick", "water_uL", "reward", "system_state"]
    assert behavior_data["time_us"].to_list() == [_sample_time(index) for index in range(_SAMPLE_COUNT)]
    assert behavior_data["elapsed_minutes"].to_list() == pytest.approx(
        [index * 0.1 for index in range(_SAMPLE_COUNT)], abs=1e-6
    )
    assert behavior_data["lick"].to_list() == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    assert behavior_data["water_uL"].to_list() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    assert behavior_data["reward"].to_list() == [
        "no",
        "no",
        "no",
        "no",
        "yes",
        "yes",
        "no",
        "no",
        "tone",
        "tone",
    ]
    assert behavior_data["system_state"].to_list() == [
        "idle",
        "idle",
        "idle",
        "run",
        "run",
        "run",
        "run",
        "rest",
        "rest",
        "rest",
    ]
    assert behavior_data.schema["system_state"] == pl.Enum(["idle", "rest", "run"])
    assert behavior_data.schema["reward"] == pl.Enum(["no", "tone", "yes"])
    assert behavior_data.schema["water_uL"] == pl.Float32
    # This assembler is the sole producer of the column for a training session, while the fluorescence assembler
    # produces it for an experiment session, so both are pinned to the same width or one dataset carries two.
    assert behavior_data.schema["elapsed_minutes"] == pl.Float32


def test_assemble_behavior_dataset_aligns_every_optional_source(tmp_path: Path) -> None:
    """Verifies that the encoder, screen, brake, and torque feathers add their columns with the documented gating.

    The brake column thresholds the interpolated brake torque, the torque column is forced to zero across the run
    samples, the encoder distance is held forward outside the run state, and the speed is zeroed outside it.
    """
    microcontroller_data_path, runtime_data_path, raw_data_path = _make_input_directories(tmp_path=tmp_path)
    _write_hardware_state(raw_data_path=raw_data_path)
    _write_required_feathers(microcontroller_data_path=microcontroller_data_path, runtime_data_path=runtime_data_path)
    _write_encoder_feather(microcontroller_data_path=microcontroller_data_path)
    _write_screen_feather(microcontroller_data_path=microcontroller_data_path)
    _write_brake_feather(microcontroller_data_path=microcontroller_data_path)
    _write_torque_feather(microcontroller_data_path=microcontroller_data_path)

    behavior_data = assemble_behavior_dataset(
        microcontroller_data_path=microcontroller_data_path,
        runtime_data_path=runtime_data_path,
        raw_data_path=raw_data_path,
        reference_time=_reference_time_vector(),
    )

    assert behavior_data.columns == [
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
    assert behavior_data["brake"].to_list() == [1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
    assert behavior_data["screens"].to_list() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    # The ramp reads one N·cm per sample, and the four run samples are overwritten with zero.
    assert behavior_data["torque_N_cm"].to_list() == pytest.approx([0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 7.0, 8.0, 9.0])
    # The encoder advances to 21.5 cm by the last run sample, and the rest samples hold that readout forward.
    assert behavior_data["distance_cm"].to_list() == pytest.approx(
        [0.0, 0.0, 0.0, 1.0, 6.5, 14.0, 21.5, 21.5, 21.5, 21.5]
    )
    # The rest-sample displacement produces a 20 cm/s readout that the run gate zeroes.
    assert behavior_data["speed_cm_s"].to_list() == pytest.approx(
        [0.0, 0.0, 0.0, 20.0, 30.0, 40.0, 30.0, 0.0, 0.0, 0.0]
    )
    assert behavior_data.schema["brake"] == pl.UInt8
    assert behavior_data.schema["torque_N_cm"] == pl.Float32
    assert behavior_data.schema["speed_cm_s"] == pl.Float32


def test_assemble_behavior_dataset_steps_the_water_total_between_deliveries(tmp_path: Path) -> None:
    """Verifies that the cumulative water total is held forward between valve events rather than ramped across them.

    A real reference clock is the mesoscope frame clock and never lands on a valve event, so nearly every sample falls
    between two deliveries. Blending them would report a fractional volume that was never dispensed, which the power-law
    dispensing function makes wrong in any case. The encoder distance beside it does blend, so this pins the two sources
    to their different interpolation modes rather than to a reference clock that hides the difference.
    """
    microcontroller_data_path, runtime_data_path, raw_data_path = _make_input_directories(tmp_path=tmp_path)
    _write_hardware_state(raw_data_path=raw_data_path)
    _write_required_feathers(microcontroller_data_path=microcontroller_data_path, runtime_data_path=runtime_data_path)
    _write_encoder_feather(microcontroller_data_path=microcontroller_data_path)
    # Shifts the reference clock half a sampling interval off every feather timestamp, so the fifth sample sits
    # exactly midway between the dry valve event at sample four and the five-microliter delivery at sample five.
    off_grid_time = _reference_time_vector() + _SAMPLE_INTERVAL_US // 2

    behavior_data = assemble_behavior_dataset(
        microcontroller_data_path=microcontroller_data_path,
        runtime_data_path=runtime_data_path,
        raw_data_path=raw_data_path,
        reference_time=off_grid_time,
    )

    assert behavior_data["water_uL"].to_list() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    # The encoder is interpolated linearly, so its off-grid samples do blend their bracketing readings: the fifth
    # sample sits three seconds past the 6.5 cm reading and 2.95 seconds short of the following 12.0 cm one.
    blend = (_SAMPLE_INTERVAL_US // 2) / (_SAMPLE_INTERVAL_US - _ENCODER_PAIR_OFFSET_US)
    assert behavior_data["distance_cm"][4] == pytest.approx(6.5 + blend * 5.5)


def test_assemble_behavior_dataset_holds_the_traveled_distance_across_a_paused_run(tmp_path: Path) -> None:
    """Verifies that the zero anchor applies to the leading idle span alone and never to a later one.

    A paused session returns to idle mid-run, and the encoder is disabled outside the run state, so the samples of
    that later idle span hold the last run readout forward. Anchoring them at zero instead would make the cumulative
    traveled distance drop back to the session start and then jump forward again on the next run sample.
    """
    microcontroller_data_path, runtime_data_path, raw_data_path = _make_input_directories(tmp_path=tmp_path)
    _write_hardware_state(raw_data_path=raw_data_path)
    _write_required_feathers(microcontroller_data_path=microcontroller_data_path, runtime_data_path=runtime_data_path)
    _write_encoder_feather(microcontroller_data_path=microcontroller_data_path)
    # Replaces the monotone idle-run-rest walk with one that pauses back into idle at the seventh sample.
    pl.DataFrame(
        {
            "time_us": np.array([_sample_time(0), _sample_time(3), _sample_time(6), _sample_time(8)], dtype=np.uint64),
            "system_state": np.array([0, 2, 0, 2], dtype=np.uint8),
        }
    ).write_ipc(file=runtime_data_path.joinpath(BehaviorDataFiles.SYSTEM_STATE))

    behavior_data = assemble_behavior_dataset(
        microcontroller_data_path=microcontroller_data_path,
        runtime_data_path=runtime_data_path,
        raw_data_path=raw_data_path,
        reference_time=_reference_time_vector(),
    )

    assert behavior_data["system_state"].to_list() == [
        "idle",
        "idle",
        "idle",
        "run",
        "run",
        "run",
        "idle",
        "idle",
        "run",
        "run",
    ]
    # The leading idle span reads zero, the paused samples six and seven hold the 14 cm reached by sample five, and
    # the cumulative trace never steps backwards.
    assert behavior_data["distance_cm"].to_list() == pytest.approx(
        [0.0, 0.0, 0.0, 1.0, 6.5, 14.0, 14.0, 14.0, 26.0, 26.0]
    )
    # The speed gate answers to the run state alone, so the paused samples report no motion.
    assert behavior_data["speed_cm_s"].to_list() == pytest.approx([0.0, 0.0, 0.0, 20.0, 30.0, 40.0, 0.0, 0.0, 0.0, 0.0])


def test_assemble_behavior_dataset_drops_the_time_columns_on_request(tmp_path: Path) -> None:
    """Verifies that enabling the drop flag removes the clock columns while keeping every data column."""
    microcontroller_data_path, runtime_data_path, raw_data_path = _make_input_directories(tmp_path=tmp_path)
    _write_hardware_state(raw_data_path=raw_data_path)
    _write_required_feathers(microcontroller_data_path=microcontroller_data_path, runtime_data_path=runtime_data_path)
    _write_torque_feather(microcontroller_data_path=microcontroller_data_path)

    behavior_data = assemble_behavior_dataset(
        microcontroller_data_path=microcontroller_data_path,
        runtime_data_path=runtime_data_path,
        raw_data_path=raw_data_path,
        reference_time=_reference_time_vector(),
        drop_time_columns=True,
    )

    assert behavior_data.columns == ["torque_N_cm", "lick", "water_uL", "reward", "system_state"]
    assert behavior_data.height == _SAMPLE_COUNT


def test_assemble_behavior_dataset_reports_a_missing_state_code_mapping(tmp_path: Path) -> None:
    """Verifies that a hardware state without the system-state code mapping aborts the assembly."""
    microcontroller_data_path, runtime_data_path, raw_data_path = _make_input_directories(tmp_path=tmp_path)
    _write_hardware_state(raw_data_path=raw_data_path, system_state_codes=None)
    _write_required_feathers(microcontroller_data_path=microcontroller_data_path, runtime_data_path=runtime_data_path)

    with pytest.raises(ValueError, match="'system_state_codes'"):
        assemble_behavior_dataset(
            microcontroller_data_path=microcontroller_data_path,
            runtime_data_path=runtime_data_path,
            raw_data_path=raw_data_path,
            reference_time=_reference_time_vector(),
        )


def test_assemble_behavior_dataset_reports_a_missing_brake_threshold(tmp_path: Path) -> None:
    """Verifies that a present brake feather without the brake threshold aborts the assembly."""
    microcontroller_data_path, runtime_data_path, raw_data_path = _make_input_directories(tmp_path=tmp_path)
    _write_hardware_state(raw_data_path=raw_data_path, minimum_brake_strength=None)
    _write_required_feathers(microcontroller_data_path=microcontroller_data_path, runtime_data_path=runtime_data_path)
    _write_brake_feather(microcontroller_data_path=microcontroller_data_path)

    with pytest.raises(ValueError, match="'minimum_brake_strength'"):
        assemble_behavior_dataset(
            microcontroller_data_path=microcontroller_data_path,
            runtime_data_path=runtime_data_path,
            raw_data_path=raw_data_path,
            reference_time=_reference_time_vector(),
        )


def test_assemble_behavior_dataset_reports_a_missing_valve_feather(tmp_path: Path) -> None:
    """Verifies that the assembly propagates the reader's error when a mandatory module feather is absent."""
    microcontroller_data_path, runtime_data_path, raw_data_path = _make_input_directories(tmp_path=tmp_path)
    _write_hardware_state(raw_data_path=raw_data_path)
    _write_required_feathers(microcontroller_data_path=microcontroller_data_path, runtime_data_path=runtime_data_path)
    microcontroller_data_path.joinpath(BehaviorDataFiles.VALVE).unlink()

    with pytest.raises(FileNotFoundError, match=r"valve_data\.feather"):
        assemble_behavior_dataset(
            microcontroller_data_path=microcontroller_data_path,
            runtime_data_path=runtime_data_path,
            raw_data_path=raw_data_path,
            reference_time=_reference_time_vector(),
        )


def test_calculate_running_speed_returns_no_speed_for_an_empty_recording() -> None:
    """Verifies that an encoder stream with no samples yields an empty single-precision speed vector."""
    empty_time = np.zeros(0, dtype=np.uint64)
    empty_distance = np.zeros(0, dtype=np.float64)

    speed = _calculate_running_speed.py_func(sample_time=empty_time, distance=empty_distance)

    assert speed.shape == (0,)
    assert speed.dtype == np.float32
    assert _calculate_running_speed(sample_time=empty_time, distance=empty_distance).shape == (0,)


def test_calculate_running_speed_scans_the_sliding_window() -> None:
    """Verifies the window scan across duplicate stamps, backward travel, and samples with no in-window predecessor.

    The first sample has no predecessor at all, the second repeats its timestamp, the third and fourth sit inside the
    window, the fifth travels backwards, and the sixth outruns the window entirely.
    """
    stamps = np.array(
        [
            _BASE_TIME_US,
            _BASE_TIME_US,
            _BASE_TIME_US + 50_000,
            _BASE_TIME_US + 100_000,
            _BASE_TIME_US + 150_000,
            _BASE_TIME_US + 300_000,
        ],
        dtype=np.uint64,
    )
    distance = np.array([0.0, 0.0, 5.0, 10.0, 4.0, 20.0], dtype=np.float64)

    speed = _calculate_running_speed.py_func(sample_time=stamps, distance=distance)

    assert speed.tolist() == pytest.approx([0.0, 0.0, 100.0, 100.0, 0.0, 0.0])
    assert speed.dtype == np.float32
    # The compiled kernel the assembler calls agrees with the interpreted implementation.
    assert _calculate_running_speed(sample_time=stamps, distance=distance).tolist() == pytest.approx(
        [0.0, 0.0, 100.0, 100.0, 0.0, 0.0]
    )


def test_calculate_running_speed_honors_a_widened_window() -> None:
    """Verifies that widening the sliding window keeps earlier samples in scope and changes the reported speeds."""
    stamps = np.array(
        [
            _BASE_TIME_US,
            _BASE_TIME_US,
            _BASE_TIME_US + 50_000,
            _BASE_TIME_US + 100_000,
            _BASE_TIME_US + 150_000,
            _BASE_TIME_US + 300_000,
        ],
        dtype=np.uint64,
    )
    distance = np.array([0.0, 0.0, 5.0, 10.0, 4.0, 20.0], dtype=np.float64)

    speed = _calculate_running_speed.py_func(sample_time=stamps, distance=distance, window_size_us=200_000)

    # The fifth sample now measures against the session start, so its backward step no longer clamps to zero.
    assert speed.tolist() == pytest.approx([0.0, 0.0, 100.0, 100.0, 26.666667, 50.0], rel=1e-6)


def test_dataset_column_descriptions_cover_every_declared_column() -> None:
    """Verifies that the donated description mapping is keyed by column name and spans every declared column."""
    assert set(MESOSCOPE_COLUMN_DESCRIPTIONS) == {column.value for column in DatasetColumn}
    assert len(MESOSCOPE_COLUMN_DESCRIPTIONS) == len(_COLUMN_DESCRIPTIONS)
    assert all(description for description in MESOSCOPE_COLUMN_DESCRIPTIONS.values())
    assert MESOSCOPE_COLUMN_DESCRIPTIONS["water_uL"] == _COLUMN_DESCRIPTIONS[DatasetColumn.WATER_UL]
    assert MESOSCOPE_COLUMN_DESCRIPTIONS["time_us"].startswith("Microsecond-precision sample timestamps")


def test_metadata_filenames_are_string_valued_feather_names() -> None:
    """Verifies that the behavior and video filename enumerations compare equal to the raw feather filenames."""
    assert BehaviorDataFiles.VALVE == "valve_data.feather"
    assert BehaviorDataFiles.SYSTEM_STATE == "system_state_data.feather"
    assert VideoDataFiles.FACE_CAMERA_PUPIL == "face_camera_pupil.feather"
    assert all(member.value.endswith(".feather") for member in BehaviorDataFiles)
    assert all(member.value.endswith(".feather") for member in VideoDataFiles)
    assert len({member.value for member in BehaviorDataFiles}) == len(BehaviorDataFiles)
    assert len({member.value for member in VideoDataFiles}) == len(VideoDataFiles)
