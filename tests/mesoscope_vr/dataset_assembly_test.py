"""Contains tests for the Mesoscope-VR forging dispatcher, the training-session reference-clock resolver, the session-
bounds clip, and the behavior dataset's reward classifier.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sollertia_shared_assets import RawDataFiles, SessionTypes, MesoscopeHardwareState

from sollertia_forgery.shared_assets import multi_recording_dataset_name
import sollertia_forgery.mesoscope_vr.forging as dispatcher_module
from sollertia_forgery.mesoscope_vr.metadata import VideoDataFiles, BehaviorDataFiles
from sollertia_forgery.mesoscope_vr.video_dataset import resolve_slowest_camera_clock
from sollertia_forgery.mesoscope_vr.runtime_dataset import clip_to_session_bounds
from sollertia_forgery.mesoscope_vr.behavior_dataset import assemble_behavior_dataset

if TYPE_CHECKING:
    from numpy.typing import NDArray

_FRAME_TIME_COLUMN: str = "frame_time_us"
"""The single column name each camera timestamp feather carries, matching the video-dataset assembler's contract."""


def _write_timestamps(directory: Path, filename: str, timestamps: NDArray[np.uint64]) -> None:
    """Writes a camera timestamp feather holding the given per-frame acquisition timestamps."""
    pl.DataFrame({_FRAME_TIME_COLUMN: timestamps.astype(np.uint64)}).write_ipc(file=directory.joinpath(filename))


def _stub_session_loader(session_type: SessionTypes) -> SimpleNamespace:
    """Returns a stand-in for the shared ``SessionData`` whose ``load`` yields a session of the given type."""
    return SimpleNamespace(load=lambda session_path: SimpleNamespace(session_type=session_type))  # noqa: ARG005


def _patch_assemblers(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    """Replaces both sub-assemblers with recorders and returns the mapping into which the routing writes."""
    calls: dict[str, dict[str, Any]] = {}
    monkeypatch.setattr(
        target=dispatcher_module,
        name="assemble_experiment_dataset",
        value=lambda **kwargs: calls.setdefault("experiment", kwargs),
    )
    monkeypatch.setattr(
        target=dispatcher_module,
        name="assemble_training_dataset",
        value=lambda **kwargs: calls.setdefault("training", kwargs),
    )
    return calls


def _write_state_streams(directory: Path, system_states: dict[int, int], runtime_times: NDArray[np.uint64]) -> None:
    """Writes the system-state and runtime-state feathers the session-bounds clip reads.

    Args:
        directory: The processed runtime-data directory that receives both feathers.
        system_states: The system state code to record at each timestamp, keyed by timestamp.
        runtime_times: The runtime-state entry timestamps, whose last value marks the end of the runtime.
    """
    pl.DataFrame(
        {
            "time_us": np.fromiter(system_states.keys(), dtype=np.uint64),
            "system_state": np.fromiter(system_states.values(), dtype=np.uint8),
        }
    ).write_ipc(file=directory.joinpath(BehaviorDataFiles.SYSTEM_STATE))
    pl.DataFrame(
        {"time_us": runtime_times.astype(np.uint64), "runtime_state": np.ones(runtime_times.size, dtype=np.uint8)}
    ).write_ipc(file=directory.joinpath(BehaviorDataFiles.RUNTIME_STATE))


def _build_assembled_dataset(timestamps: list[int]) -> pl.DataFrame:
    """Returns a stand-in assembled dataset carrying the given reference-clock timestamps."""
    return pl.DataFrame(
        {"time_us": np.array(timestamps, dtype=np.uint64), "value": [float(value) for value in timestamps]}
    )


def _write_reward_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Writes the minimal behavior-assembly inputs carrying three tone events and two water deliveries.

    The valve stream plays a tone at 2_000 that dispenses water at 3_000, a second tone at 8_000 that dispenses
    nothing, and a third tone at 14_000 that dispenses water at 15_000. The second tone therefore falls after the
    session's first delivery, which is the case that pins the classifier to the water delivered inside the tone span.

    Args:
        tmp_path: The temporary directory that receives the three input directories.

    Returns:
        A tuple of the microcontroller-data, runtime-data, and raw-data directories.
    """
    microcontroller_data_path = tmp_path.joinpath("microcontroller_data")
    runtime_data_path = tmp_path.joinpath("runtime_data")
    raw_data_path = tmp_path.joinpath("raw_data")
    for directory in (microcontroller_data_path, runtime_data_path, raw_data_path):
        directory.mkdir()

    MesoscopeHardwareState(system_state_codes={"idle": 0, "rest": 1, "run": 2}).to_yaml(
        file_path=raw_data_path.joinpath(RawDataFiles.HARDWARE_STATE)
    )

    pl.DataFrame(
        {
            "time_us": np.array([0, 2_000, 3_000, 4_000, 8_000, 10_000, 14_000, 15_000, 16_000], dtype=np.uint64),
            "dispensed_water_volume_uL": np.array([0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 10.0, 10.0], dtype=np.float64),
            "tone_state": np.array([0, 1, 1, 0, 1, 0, 1, 1, 0], dtype=np.uint8),
        }
    ).write_ipc(file=microcontroller_data_path.joinpath(BehaviorDataFiles.VALVE))

    pl.DataFrame({"time_us": np.array([0], dtype=np.uint64), "lick_state": np.array([0], dtype=np.uint8)}).write_ipc(
        file=microcontroller_data_path.joinpath(BehaviorDataFiles.LICK)
    )

    pl.DataFrame(
        {"time_us": np.array([0, 1_000], dtype=np.uint64), "system_state": np.array([0, 2], dtype=np.uint8)}
    ).write_ipc(file=runtime_data_path.joinpath(BehaviorDataFiles.SYSTEM_STATE))

    return microcontroller_data_path, runtime_data_path, raw_data_path


def test_resolve_slowest_camera_clock_picks_lowest_mean_rate(tmp_path: Path) -> None:
    """Verifies the resolver returns the timestamps of the camera with the lowest mean frame rate."""
    # Gives the body camera a lower frame rate than the face camera so it becomes the slowest reference clock.
    fast_clock = np.arange(100, dtype=np.uint64) * np.uint64(10_000)
    slow_clock = np.arange(30, dtype=np.uint64) * np.uint64(34_000)
    _write_timestamps(directory=tmp_path, filename=VideoDataFiles.FACE_CAMERA_TIMESTAMPS, timestamps=fast_clock)
    _write_timestamps(directory=tmp_path, filename=VideoDataFiles.BODY_CAMERA_TIMESTAMPS, timestamps=slow_clock)

    resolved = resolve_slowest_camera_clock(video_data_path=tmp_path)

    assert np.array_equal(resolved, slow_clock)


def test_resolve_slowest_camera_clock_uses_single_present_camera(tmp_path: Path) -> None:
    """Verifies that with only one camera present, its clock is the reference clock."""
    clock = np.arange(50, dtype=np.uint64) * np.uint64(20_000)
    _write_timestamps(directory=tmp_path, filename=VideoDataFiles.BODY_CAMERA_TIMESTAMPS, timestamps=clock)

    assert np.array_equal(resolve_slowest_camera_clock(video_data_path=tmp_path), clock)


def test_resolve_slowest_camera_clock_ignores_degenerate_feathers(tmp_path: Path) -> None:
    """Verifies a single-frame feather cannot define a rate, so a longer, faster camera still wins over it."""
    _write_timestamps(
        directory=tmp_path,
        filename=VideoDataFiles.FACE_CAMERA_TIMESTAMPS,
        timestamps=np.array([1_000], dtype=np.uint64),
    )
    fast_clock = np.arange(40, dtype=np.uint64) * np.uint64(10_000)
    _write_timestamps(directory=tmp_path, filename=VideoDataFiles.BODY_CAMERA_TIMESTAMPS, timestamps=fast_clock)

    assert np.array_equal(resolve_slowest_camera_clock(video_data_path=tmp_path), fast_clock)


def test_resolve_slowest_camera_clock_errors_without_cameras(tmp_path: Path) -> None:
    """Verifies a video directory with no usable camera timestamp feather cannot supply a reference clock."""
    with pytest.raises(FileNotFoundError, match=r"no\s+camera\s+clock"):
        resolve_slowest_camera_clock(video_data_path=tmp_path)


def test_multi_recording_dataset_name_is_qualified_by_animal() -> None:
    """Verifies the multi-recording dataset name carries the animal identifier. That identifier keeps one animal's
    tracked output separate from its peers when a forged dataset spans several animals.
    """
    assert multi_recording_dataset_name(animal_id="101", dataset_name="Learning") == "101_Learning"
    assert multi_recording_dataset_name(animal_id="321", dataset_name="MaalstroomicFlow") == "321_MaalstroomicFlow"


def test_dispatch_routes_experiment_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies a mesoscope experiment session is routed to the experiment assembler with the dataset name forwarded."""
    monkeypatch.setattr(
        target=dispatcher_module,
        name="SessionData",
        value=_stub_session_loader(session_type=SessionTypes.MESOSCOPE_EXPERIMENT),
    )
    calls = _patch_assemblers(monkeypatch)

    dispatcher_module.assemble_mesoscope_session(
        source_session_path=Path("/project/animal/session"),
        output_path=Path("/out/data.feather"),
        dataset_name="dataset",
    )

    assert "training" not in calls
    assert calls["experiment"]["dataset_name"] == "dataset"


@pytest.mark.parametrize("session_type", [SessionTypes.RUN_TRAINING, SessionTypes.LICK_TRAINING])
def test_dispatch_routes_training_session(monkeypatch: pytest.MonkeyPatch, session_type: SessionTypes) -> None:
    """Verifies a run or lick training session is routed to the training assembler, which takes no dataset name."""
    monkeypatch.setattr(
        target=dispatcher_module, name="SessionData", value=_stub_session_loader(session_type=session_type)
    )
    calls = _patch_assemblers(monkeypatch)

    dispatcher_module.assemble_mesoscope_session(
        source_session_path=Path("/project/animal/session"),
        output_path=Path("/out/data.feather"),
        dataset_name="dataset",
    )

    assert "experiment" not in calls
    assert "dataset_name" not in calls["training"]


def test_dispatch_rejects_window_checking_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies a window-checking session is rejected by the dispatcher."""
    monkeypatch.setattr(
        target=dispatcher_module,
        name="SessionData",
        value=_stub_session_loader(session_type=SessionTypes.WINDOW_CHECKING),
    )
    _patch_assemblers(monkeypatch)

    with pytest.raises(ValueError, match="not a supported forging"):
        dispatcher_module.assemble_mesoscope_session(
            source_session_path=Path("/project/animal/session"),
            output_path=Path("/out/data.feather"),
            dataset_name="ds",
        )


def test_clip_to_session_bounds_drops_the_setup_and_teardown_spans(tmp_path: Path) -> None:
    """Verifies that samples outside the session start and the runtime end are discarded from both ends."""
    # The system idles through setup and leaves idle at 2_000, which is where the session's data begins.
    _write_state_streams(
        directory=tmp_path,
        system_states={0: 0, 1_000: 0, 2_000: 2, 3_000: 2},
        runtime_times=np.array([0, 3_000], dtype=np.uint64),
    )

    clipped = clip_to_session_bounds(
        assembled_data=_build_assembled_dataset(timestamps=[0, 1_000, 2_000, 3_000, 4_000]), runtime_data_path=tmp_path
    )

    assert clipped["time_us"].to_list() == [2_000, 3_000]


def test_clip_to_session_bounds_anchors_the_head_on_the_first_non_idle_state(tmp_path: Path) -> None:
    """Verifies that a mid-session return to idle does not move the head anchor."""
    _write_state_streams(
        directory=tmp_path,
        system_states={0: 0, 1_000: 3, 2_000: 0, 3_000: 3},
        runtime_times=np.array([0, 4_000], dtype=np.uint64),
    )

    clipped = clip_to_session_bounds(
        assembled_data=_build_assembled_dataset(timestamps=[0, 1_000, 2_000, 3_000]), runtime_data_path=tmp_path
    )

    # The acquisition system re-enters idle whenever a running session pauses, so only the first departure from idle
    # marks the session start.
    assert clipped["time_us"].to_list() == [1_000, 2_000, 3_000]


def test_clip_to_session_bounds_keeps_a_dataset_inside_both_bounds(tmp_path: Path) -> None:
    """Verifies that a dataset already contained within the session bounds is left whole."""
    _write_state_streams(directory=tmp_path, system_states={0: 2}, runtime_times=np.array([0, 5_000], dtype=np.uint64))

    clipped = clip_to_session_bounds(
        assembled_data=_build_assembled_dataset(timestamps=[1_000, 2_000, 3_000]), runtime_data_path=tmp_path
    )

    assert clipped["time_us"].to_list() == [1_000, 2_000, 3_000]


def test_clip_to_session_bounds_keeps_the_head_when_the_session_never_leaves_idle(tmp_path: Path) -> None:
    """Verifies that a session with no non-idle state keeps its head."""
    _write_state_streams(
        directory=tmp_path, system_states={0: 0, 1_000: 0}, runtime_times=np.array([0, 3_000], dtype=np.uint64)
    )

    clipped = clip_to_session_bounds(
        assembled_data=_build_assembled_dataset([0, 1_000, 2_000]), runtime_data_path=tmp_path
    )

    # A session terminated during setup never leaves idle, so no session start anchors the head.
    assert clipped["time_us"].to_list() == [0, 1_000, 2_000]


def test_clip_to_session_bounds_keeps_the_tail_without_a_runtime_state_entry(tmp_path: Path) -> None:
    """Verifies that an empty runtime-state stream leaves the tail in place."""
    _write_state_streams(directory=tmp_path, system_states={1_000: 2}, runtime_times=np.array([], dtype=np.uint64))

    clipped = clip_to_session_bounds(
        assembled_data=_build_assembled_dataset([0, 1_000, 2_000]), runtime_data_path=tmp_path
    )

    assert clipped["time_us"].to_list() == [1_000, 2_000]


def test_assemble_behavior_dataset_classifies_a_dry_mid_session_tone_as_tone(tmp_path: Path) -> None:
    """Verifies that a tone event delivering no water classifies as 'tone' after an earlier delivery."""
    microcontroller_data_path, runtime_data_path, raw_data_path = _write_reward_inputs(tmp_path=tmp_path)

    behavior_data = assemble_behavior_dataset(
        microcontroller_data_path=microcontroller_data_path,
        runtime_data_path=runtime_data_path,
        raw_data_path=raw_data_path,
        reference_time=np.arange(0, 21_000, 1_000, dtype=np.uint64),
    )

    rewards = behavior_data["reward"].to_list()

    # The 8_000 and 9_000 samples span the dry tone, which follows the delivery at 3_000. The classifier measures the
    # water delivered inside each tone span, so a dry tone stays 'tone' no matter how much water the session dispensed
    # before it.
    assert rewards[8:10] == ["tone", "tone"]
    assert rewards[2:4] == ["yes", "yes"]
    assert rewards[14:16] == ["yes", "yes"]
    assert set(rewards[4:8]) == {"no"}
    assert set(rewards[16:]) == {"no"}
