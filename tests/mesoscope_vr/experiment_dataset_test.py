"""Contains tests for the Mesoscope-VR experiment-session data assembler against an on-disk session built from real
outputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from cindra import CombinedData, DetectionData, ExtractionData, resolve_dataset_path
import polars as pl
import pytest

from sollertia_forgery.shared_assets import multi_recording_dataset_name
from sollertia_forgery.mesoscope_vr.forging import assemble_mesoscope_session
from sollertia_forgery.mesoscope_vr.metadata import VideoDataFiles, BehaviorDataFiles
from sollertia_forgery.mesoscope_vr.experiment_dataset import assemble_experiment_dataset

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from sollertia_shared_assets import SessionData

_DATASET_NAME: str = "Learning"
"""The unqualified dataset name the assembler resolves the multi-recording cindra directory from."""

_FRAME_COUNT: int = 24
"""The number of mesoscope frames cindra reports, which the TTL pulse alignment is reconciled against."""

_ROI_COUNT: int = 5
"""The number of regions of interest the single-recording cindra output holds before the cell filter is applied."""

_CELL_COUNT: int = 3
"""The number of regions of interest the cell classification marks as cells."""

_MULTI_ROI_COUNT: int = 2
"""The number of cells the multi-recording cindra output tracks across the animal's sessions."""

_SAMPLING_RATE_HZ: float = 10.0
"""The per-plane sampling rate the combined cindra metadata reports, which sets the expected scan pulse duration."""

_COMBINED_FRAME_EXTENT: int = 512
"""The height and the width recorded in the synthetic combined metadata archive. The assembler reads the sampling
rate alone, so the extent only has to be a shape cindra's writer accepts."""

_FIRST_PULSE_US: int = 1_000_000
"""The timestamp of the first mesoscope scan pulse rising edge."""

_PULSE_PERIOD_US: int = 120_000
"""The interval between two consecutive scan pulse rising edges."""

_PULSE_DURATION_US: int = 100_000
"""The high duration of each scan pulse, which matches the duration the reported sampling rate implies."""

_SESSION_START_US: int = 1_500_000
"""The timestamp at which the acquisition system first leaves idle, which anchors the head of the clipped dataset."""

_REST_ONSET_US: int = 2_500_000
"""The timestamp at which the acquisition system enters the rest state."""

_REST_END_US: int = 2_800_000
"""The timestamp at which the acquisition system returns to the run state."""

_RUNTIME_END_US: int = 3_500_000
"""The timestamp of the final runtime-state entry, which anchors the tail of the clipped dataset."""

_BRAKE_ONSET_US: int = 3_000_000
"""The timestamp at which the wheel brake engages for the remainder of the session."""

_CM_PER_MICROSECOND: float = 1.0 / 10_000.0
"""The wheel distance the encoder feather advances per microsecond, which makes a reference distance predictable."""

_RESTING_TORQUE_N_CM: float = 3.0
"""The torque the animal exerts on the wheel while the system is not in the run state."""

_CUE_UNDEFINED: int = 255
"""The cue sentinel the masking step writes for every sample acquired outside the run state."""

_TRIAL_UNDEFINED: int = 65535
"""The trial sentinel the masking step writes for every sample acquired outside the run state."""

_FLUORESCENCE_COLUMNS: frozenset[str] = frozenset(
    {
        "frame",
        "time_us",
        "elapsed_minutes",
        "single_day_cell_fluorescence",
        "single_day_neuropil_fluorescence",
        "single_day_subtracted_fluorescence",
        "single_day_spikes",
        "multi_day_cell_fluorescence",
        "multi_day_neuropil_fluorescence",
        "multi_day_subtracted_fluorescence",
        "multi_day_spikes",
    }
)
"""The columns the fluorescence sub-dataset contributes, including the reference clock the others align to."""

_BEHAVIOR_COLUMNS: frozenset[str] = frozenset(
    {"brake", "screens", "torque_N_cm", "distance_cm", "speed_cm_s", "lick", "water_uL", "reward", "system_state"}
)
"""The columns a mesoscope experiment session's behavior sub-dataset contributes, with its time columns dropped."""

_RUNTIME_COLUMNS: frozenset[str] = frozenset(
    {"trial", "trial_type", "cue", "in_trigger_zone", "runtime_state", "reinforcing_guided"}
)
"""The columns the runtime sub-dataset contributes when the session recorded reinforcing guidance alone."""

_VIDEO_COLUMNS: frozenset[str] = frozenset(
    {
        "face_camera_motion_energy",
        "face_camera_frame_luminance",
        "body_camera_motion_energy",
        "body_camera_frame_luminance",
    }
)
"""The columns the video sub-dataset contributes when both cameras carry a motion-energy feather."""


def _pulse_times() -> NDArray[np.uint64]:
    """Returns the rising-edge timestamp of every mesoscope scan pulse, which becomes the fluorescence clock.

    Returns:
        The per-frame pulse rising edges in microseconds since the UTC epoch.
    """
    return np.uint64(_FIRST_PULSE_US) + np.arange(_FRAME_COUNT, dtype=np.uint64) * np.uint64(_PULSE_PERIOD_US)


def _retained_frame_indices() -> list[int]:
    """Returns the zero-based frame indices the session-bounds clip keeps.

    Returns:
        The indices whose pulse timestamp sits inside the session bounds.
    """
    return [
        index
        for index, timestamp in enumerate(_pulse_times().tolist())
        if _SESSION_START_US <= timestamp <= _RUNTIME_END_US
    ]


def _write_feather(path: Path, columns: dict[str, NDArray[np.number]]) -> None:
    """Writes one uncompressed feather holding the given columns.

    Args:
        path: The path of the feather file to write.
        columns: The column name to column value mapping the feather stores.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(columns).write_ipc(file=path, compression="uncompressed")


def _write_microcontroller_sources(session: SessionData) -> None:
    """Writes the module-parsed feathers a mesoscope experiment session's microcontroller stage produces.

    The valve stream plays one reward tone that dispenses water, the encoder advances at a constant rate, the screens
    switch on during setup, the brake engages late in the session, and the torque sensor reads a constant value.

    Args: session: The loaded experiment session whose processed microcontroller directory receives the feathers.
    """
    directory = session.processed_data.microcontroller_data_path

    scan_pulses = _pulse_times()
    edge_times = np.empty(scan_pulses.size * 2 + 1, dtype=np.uint64)
    edge_states = np.zeros(scan_pulses.size * 2 + 1, dtype=np.uint8)
    edge_times[0] = 0
    edge_times[1::2] = scan_pulses
    edge_times[2::2] = scan_pulses + np.uint64(_PULSE_DURATION_US)
    edge_states[1::2] = 1
    _write_feather(
        path=directory.joinpath(BehaviorDataFiles.MESOSCOPE_FRAME),
        columns={"time_us": edge_times, "ttl_state": edge_states},
    )

    _write_feather(
        path=directory.joinpath(BehaviorDataFiles.VALVE),
        columns={
            "time_us": np.array([0, 1_700_000, 1_900_000, 2_100_000], dtype=np.uint64),
            "dispensed_water_volume_uL": np.array([0.0, 0.0, 5.0, 5.0], dtype=np.float64),
            "tone_state": np.array([0, 1, 1, 0], dtype=np.uint8),
        },
    )
    _write_feather(
        path=directory.joinpath(BehaviorDataFiles.LICK),
        columns={
            "time_us": np.array([0, 1_900_000, 2_000_000], dtype=np.uint64),
            "lick_state": np.array([0, 1, 0], dtype=np.uint8),
        },
    )
    encoder_time = np.arange(0, 4_100_000, 100_000, dtype=np.uint64)
    _write_feather(
        path=directory.joinpath(BehaviorDataFiles.ENCODER),
        columns={
            "time_us": encoder_time,
            "traveled_distance_cm": encoder_time.astype(np.float64) * _CM_PER_MICROSECOND,
        },
    )
    _write_feather(
        path=directory.joinpath(BehaviorDataFiles.SCREEN),
        columns={
            "time_us": np.array([0, 500_000], dtype=np.uint64),
            "screen_state": np.array([0, 1], dtype=np.uint8),
        },
    )
    _write_feather(
        path=directory.joinpath(BehaviorDataFiles.BRAKE),
        columns={
            "time_us": np.array([0, _BRAKE_ONSET_US], dtype=np.uint64),
            "brake_torque_N_cm": np.array([0.0, 5.0], dtype=np.float64),
        },
    )
    _write_feather(
        path=directory.joinpath(BehaviorDataFiles.TORQUE),
        columns={
            "time_us": np.array([0, 4_000_000], dtype=np.uint64),
            "torque_N_cm": np.array([_RESTING_TORQUE_N_CM, _RESTING_TORQUE_N_CM], dtype=np.float64),
        },
    )


def _write_runtime_sources(session: SessionData) -> None:
    """Writes the runtime-parsed feathers a mesoscope experiment session's runtime stage produces.

    The session records four trials of the single configured trial structure, an alternating cue sequence, one
    stimulus trigger zone per trial, and reinforcing guidance alone.

    Args:
        session: The loaded experiment session whose processed runtime directory receives the feathers.
    """
    directory = session.processed_data.runtime_data_path

    _write_feather(
        path=directory.joinpath(BehaviorDataFiles.SYSTEM_STATE),
        columns={
            "time_us": np.array([0, _SESSION_START_US, _REST_ONSET_US, _REST_END_US], dtype=np.uint64),
            "system_state": np.array([0, 2, 1, 2], dtype=np.uint8),
        },
    )
    _write_feather(
        path=directory.joinpath(BehaviorDataFiles.RUNTIME_STATE),
        columns={
            "time_us": np.array([0, _SESSION_START_US, _RUNTIME_END_US], dtype=np.uint64),
            "runtime_state": np.array([0, 1, 1], dtype=np.uint8),
        },
    )
    _write_feather(
        path=directory.joinpath(BehaviorDataFiles.TRIAL),
        columns={
            "traveled_distance_cm": np.array([0.0, 100.0, 200.0, 300.0], dtype=np.float64),
            "trial_type_index": np.zeros(4, dtype=np.uint8),
        },
    )
    _write_feather(
        path=directory.joinpath(BehaviorDataFiles.VR_CUE),
        columns={
            "traveled_distance_cm": np.arange(0.0, 400.0, 50.0, dtype=np.float64),
            "vr_cue": np.array([1, 2, 1, 2, 1, 2, 1, 2], dtype=np.uint8),
        },
    )
    _write_feather(
        path=directory.joinpath(BehaviorDataFiles.VR_TRIGGER_ZONE),
        columns={
            "trigger_zone_start_cm": np.array([30.0, 130.0, 230.0, 330.0], dtype=np.float64),
            "trigger_zone_end_cm": np.array([45.0, 145.0, 245.0, 345.0], dtype=np.float64),
        },
    )
    _write_feather(
        path=directory.joinpath(BehaviorDataFiles.REINFORCING_GUIDANCE),
        columns={
            "time_us": np.array([0, 2_000_000], dtype=np.uint64),
            "reinforcing_guidance_state": np.array([0, 1], dtype=np.uint8),
        },
    )


def _build_extraction(roi_count: int, is_cell: Sequence[int] | None = None) -> ExtractionData:
    """Builds the cindra extraction record the fluorescence assembly reads back.

    Every trace array counts up from zero in row-major order, so a value pins both the region of interest it came from
    and the frame it was sampled at, and the per-array offset keeps the four distinguishable.

    Args: roi_count: The number of region rows every trace array carries. is_cell: The per-region cell label, one entry
    per row, or None for a record carrying no classification.

    Returns: The populated extraction record, which cindra's own writer saves under its canonical array names.
    """
    base = np.arange(roi_count * _FRAME_COUNT, dtype=np.float32).reshape(roi_count, _FRAME_COUNT)
    classification: NDArray[np.float32] | None = None
    if is_cell is not None:
        classification = np.zeros((roi_count, 2), dtype=np.float32)
        classification[:, 0] = np.asarray(is_cell, dtype=np.float32)
    return ExtractionData(
        cell_fluorescence=base,
        neuropil_fluorescence=base + 1.0,
        subtracted_fluorescence=base + 2.0,
        spikes=base + 3.0,
        cell_classification=classification,
    )


def _write_cindra_outputs(session: SessionData) -> None:
    """Writes the single-recording and multi-recording cindra arrays the fluorescence assembly reads.

    Notes: Both directories are written through cindra's own writers, so the arrays and the combined metadata archive
    the assembler reads back are the ones cindra's stages publish.

    Args: session: The loaded experiment session whose cindra directories receive the arrays.
    """
    single_path = session.processed_data.cindra_data_path
    multi_path = resolve_dataset_path(
        output_root=session.processed_data_path,
        dataset_name=multi_recording_dataset_name(animal_id=str(session.animal_id), dataset_name=_DATASET_NAME),
    )
    single_path.mkdir(parents=True, exist_ok=True)
    multi_path.mkdir(parents=True, exist_ok=True)

    is_cell = [1 if index in {0, 2, 3} else 0 for index in range(_ROI_COUNT)]
    CombinedData(
        detection=DetectionData(),
        extraction=_build_extraction(roi_count=_ROI_COUNT, is_cell=is_cell),
        plane_count=1,
        frame_count=_FRAME_COUNT,
        combined_height=_COMBINED_FRAME_EXTENT,
        combined_width=_COMBINED_FRAME_EXTENT,
        sampling_rate=_SAMPLING_RATE_HZ,
    ).save(root_path=single_path)

    _build_extraction(roi_count=_MULTI_ROI_COUNT).save_arrays(output_path=multi_path)


def _write_video_sources(session: SessionData) -> None:
    """Writes both cameras' timestamp and motion-energy feathers.

    Args:
        session: The loaded experiment session whose processed video directory receives the feathers.
    """
    directory = session.processed_data.video_data_path
    for timestamps_file, energy_file, period_us, frame_count in (
        (VideoDataFiles.FACE_CAMERA_TIMESTAMPS, VideoDataFiles.FACE_CAMERA_ENERGY, 100_000, 45),
        (VideoDataFiles.BODY_CAMERA_TIMESTAMPS, VideoDataFiles.BODY_CAMERA_ENERGY, 200_000, 23),
    ):
        _write_feather(
            path=directory.joinpath(timestamps_file),
            columns={"frame_time_us": np.arange(frame_count, dtype=np.uint64) * np.uint64(period_us)},
        )
        _write_feather(
            path=directory.joinpath(energy_file),
            columns={
                "motion_energy": np.arange(frame_count, dtype=np.float32),
                "frame_luminance": np.full(frame_count, 96.0, dtype=np.float32),
            },
        )


@pytest.fixture
def prepared_experiment_session(experiment_session: SessionData) -> SessionData:
    """Builds a fully processed mesoscope experiment session carrying every input the experiment assembler reads.

    Args:
        experiment_session: The acquired experiment session the processed outputs are written under.

    Returns:
        The same session, now holding its microcontroller, runtime, cindra, and video outputs.
    """
    _write_microcontroller_sources(session=experiment_session)
    _write_runtime_sources(session=experiment_session)
    _write_cindra_outputs(session=experiment_session)
    _write_video_sources(session=experiment_session)
    return experiment_session


@pytest.fixture
def assembled_experiment(prepared_experiment_session: SessionData, tmp_path: Path) -> pl.DataFrame:
    """Runs the forging dispatcher over the prepared session and reads the feather it wrote back.

    Args:
        prepared_experiment_session: The session holding every processed input.
        tmp_path: The temporary directory the forged dataset is written under.

    Returns:
        The assembled data feather.
    """
    output_path = tmp_path.joinpath("forged", "data.feather")
    assemble_mesoscope_session(
        source_session_path=prepared_experiment_session.raw_data_path.parent,
        output_path=output_path,
        dataset_name=_DATASET_NAME,
    )
    return pl.read_ipc(source=output_path)


def test_assemble_experiment_dataset_emits_every_sub_dataset_column(assembled_experiment: pl.DataFrame) -> None:
    """Verifies the unified feather carries the fluorescence, behavior, runtime, and video columns together."""
    assert set(assembled_experiment.columns) == (
        _FLUORESCENCE_COLUMNS | _BEHAVIOR_COLUMNS | _RUNTIME_COLUMNS | _VIDEO_COLUMNS
    )


def test_assemble_experiment_dataset_rides_the_fluorescence_clock(assembled_experiment: pl.DataFrame) -> None:
    """Verifies the reference clock is the scan pulse clock, numbered from one and then clipped to the session
    bounds, so the first retained frame keeps its original one-based index.
    """
    retained = _retained_frame_indices()

    assert assembled_experiment["time_us"].to_list() == [int(_pulse_times()[index]) for index in retained]
    assert assembled_experiment["frame"].to_list() == [index + 1 for index in retained]
    # The clock starts at the first pulse, so the first retained sample sits a fixed span into the recording.
    first_elapsed = (int(_pulse_times()[retained[0]]) - _FIRST_PULSE_US) / 60_000_000
    assert assembled_experiment["elapsed_minutes"].to_list()[0] == pytest.approx(round(first_elapsed, 2))


def test_assemble_experiment_dataset_keeps_only_the_classified_cells(assembled_experiment: pl.DataFrame) -> None:
    """Verifies the single-recording traces are filtered by the cell classification and the multi-day traces are not."""
    retained = _retained_frame_indices()
    single_traces = assembled_experiment["single_day_cell_fluorescence"].to_numpy()
    multi_traces = assembled_experiment["multi_day_spikes"].to_numpy()

    assert single_traces.shape == (len(retained), _CELL_COUNT)
    assert multi_traces.shape == (len(retained), _MULTI_ROI_COUNT)
    # The traces count up in row-major order, so a value names the region of interest and the frame it came from.
    expected_single = [float(roi * _FRAME_COUNT + retained[0]) for roi in (0, 2, 3)]
    assert single_traces[0].tolist() == expected_single
    expected_multi = [float(roi * _FRAME_COUNT + retained[0] + 3) for roi in range(_MULTI_ROI_COUNT)]
    assert multi_traces[0].tolist() == expected_multi


def test_assemble_experiment_dataset_masks_the_non_run_samples(assembled_experiment: pl.DataFrame) -> None:
    """Verifies the cue, trial, and trial type columns are masked for every sample acquired outside the run state."""
    by_time = dict(
        zip(
            assembled_experiment["time_us"].to_list(),
            zip(
                assembled_experiment["cue"].to_list(),
                assembled_experiment["trial"].to_list(),
                assembled_experiment["trial_type"].to_list(),
                assembled_experiment["system_state"].to_list(),
                strict=True,
            ),
            strict=True,
        )
    )
    resting = [time for time in by_time if _REST_ONSET_US <= time < _REST_END_US]
    running = [time for time in by_time if time not in resting]

    assert resting
    assert all(by_time[time] == (_CUE_UNDEFINED, _TRIAL_UNDEFINED, "undefined", "rest") for time in resting)
    assert all(by_time[time][2] == "reward_trial" for time in running)
    assert all(by_time[time][1] != _TRIAL_UNDEFINED for time in running)


def test_assemble_experiment_dataset_reports_the_recorded_behavior(assembled_experiment: pl.DataFrame) -> None:
    """Verifies the behavior, runtime, and video columns carry the values their source feathers recorded."""
    frame = assembled_experiment.with_columns(pl.col("time_us").alias("clock"))
    resting = frame.filter(pl.col("clock").is_between(lower_bound=_REST_ONSET_US, upper_bound=_REST_END_US - 1))
    braking = frame.filter(pl.col("clock") >= _BRAKE_ONSET_US)

    # The torque sensor reads zero while the system runs and reports the recorded torque while it rests.
    assert resting["torque_N_cm"].to_list() == [pytest.approx(_RESTING_TORQUE_N_CM)] * resting.height
    assert frame.filter(pl.col("clock") < _REST_ONSET_US)["torque_N_cm"].to_list()[0] == pytest.approx(0.0)
    assert set(braking["brake"].to_list()) == {1}
    assert set(frame.filter(pl.col("clock") < _BRAKE_ONSET_US)["brake"].to_list()) == {0}
    # The screens switch on during setup and the runtime holds the single configured experiment state throughout.
    assert set(frame["screens"].to_list()) == {1}
    assert set(frame["runtime_state"].to_list()) == {"run_state"}
    # Reinforcing guidance switches on at 2_000_000 and stays on, so the column reports both of its states.
    assert set(frame.filter(pl.col("clock") < 2_000_000)["reinforcing_guided"].to_list()) == {0}
    assert set(frame.filter(pl.col("clock") >= 2_000_000)["reinforcing_guided"].to_list()) == {1}
    # The animal sits inside a stimulus trigger zone over part of the retained span.
    assert set(frame["in_trigger_zone"].to_list()) == {0, 1}
    assert frame["face_camera_frame_luminance"].to_list() == [pytest.approx(96.0)] * frame.height


def test_assemble_experiment_dataset_forges_without_video_data(
    prepared_experiment_session: SessionData, tmp_path: Path
) -> None:
    """Verifies a session processed without camera data still forges, contributing no video columns."""
    for feather in prepared_experiment_session.processed_data.video_data_path.iterdir():
        feather.unlink()
    prepared_experiment_session.processed_data.video_data_path.rmdir()
    output_path = tmp_path.joinpath("forged", "data.feather")

    assemble_experiment_dataset(
        source_session_path=prepared_experiment_session.raw_data_path.parent,
        output_path=output_path,
        dataset_name=_DATASET_NAME,
    )

    assembled = pl.read_ipc(source=output_path)
    assert set(assembled.columns) == _FLUORESCENCE_COLUMNS | _BEHAVIOR_COLUMNS | _RUNTIME_COLUMNS
    assert assembled.height == len(_retained_frame_indices())


def test_assemble_experiment_dataset_rejects_a_missing_microcontroller_directory(
    experiment_session: SessionData, tmp_path: Path
) -> None:
    """Verifies an unprocessed microcontroller stage is reported before any expensive work starts."""
    _write_runtime_sources(session=experiment_session)
    _write_cindra_outputs(session=experiment_session)
    session_name = experiment_session.session_name

    with pytest.raises(FileNotFoundError, match=rf"(?s)session '{session_name}'.*microcontroller data\s+directory"):
        assemble_experiment_dataset(
            source_session_path=experiment_session.raw_data_path.parent,
            output_path=tmp_path.joinpath("forged", "data.feather"),
            dataset_name=_DATASET_NAME,
        )


def test_assemble_experiment_dataset_rejects_a_missing_runtime_directory(
    experiment_session: SessionData, tmp_path: Path
) -> None:
    """Verifies an unprocessed runtime stage is reported even when the microcontroller stage finished."""
    _write_microcontroller_sources(session=experiment_session)
    _write_cindra_outputs(session=experiment_session)

    with pytest.raises(FileNotFoundError, match=r"(?s)processed runtime\s+data directory"):
        assemble_experiment_dataset(
            source_session_path=experiment_session.raw_data_path.parent,
            output_path=tmp_path.joinpath("forged", "data.feather"),
            dataset_name=_DATASET_NAME,
        )


def test_assemble_experiment_dataset_rejects_a_missing_cindra_directory(
    experiment_session: SessionData, tmp_path: Path
) -> None:
    """Verifies an unprocessed two-photon stage is reported, since it supplies the assembly reference clock."""
    _write_microcontroller_sources(session=experiment_session)
    _write_runtime_sources(session=experiment_session)

    with pytest.raises(FileNotFoundError, match=r"(?s)single-recording cindra output\s+directory"):
        assemble_experiment_dataset(
            source_session_path=experiment_session.raw_data_path.parent,
            output_path=tmp_path.joinpath("forged", "data.feather"),
            dataset_name=_DATASET_NAME,
        )
