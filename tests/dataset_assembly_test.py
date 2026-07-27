"""Tests for the Mesoscope-VR forging dispatcher and the training-session reference-clock resolver.

The dispatcher's session-type routing is exercised with a stubbed session loader and stubbed sub-assemblers, and the
slowest-camera clock resolver is exercised against written camera timestamp feathers.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sollertia_shared_assets import SessionTypes

from sollertia_forgery.shared_assets import multi_recording_dataset_directory
import sollertia_forgery.mesoscope_vr.forging as dispatcher_module
from sollertia_forgery.mesoscope_vr.metadata import VideoDataFiles, BehaviorDataFiles
from sollertia_forgery.mesoscope_vr.video_dataset import resolve_slowest_camera_clock
from sollertia_forgery.mesoscope_vr.runtime_dataset import clip_to_runtime_end

_FRAME_TIME_COLUMN: str = "frame_time_us"
"""The single column name each camera timestamp feather carries, matching the video-dataset assembler's contract."""


def _write_timestamps(directory: Path, filename: str, timestamps: np.ndarray) -> None:
    """Writes a camera timestamp feather holding the given per-frame acquisition timestamps."""
    pl.DataFrame({_FRAME_TIME_COLUMN: timestamps.astype(np.uint64)}).write_ipc(file=directory.joinpath(filename))


def _stub_session_loader(session_type: SessionTypes) -> SimpleNamespace:
    """Returns a stand-in for the shared ``SessionData`` whose ``load`` yields a session of the given type."""
    return SimpleNamespace(load=lambda session_path: SimpleNamespace(session_type=session_type))  # noqa: ARG005


def _patch_assemblers(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    """Replaces both sub-assemblers with recorders and returns the mapping the routing writes into."""
    calls: dict[str, dict[str, Any]] = {}
    monkeypatch.setattr(
        dispatcher_module, "assemble_experiment_dataset", lambda **kwargs: calls.setdefault("experiment", kwargs)
    )
    monkeypatch.setattr(
        dispatcher_module, "assemble_training_dataset", lambda **kwargs: calls.setdefault("training", kwargs)
    )
    return calls


def test_resolve_slowest_camera_clock_picks_lowest_mean_rate(tmp_path: Path) -> None:
    """Verifies the resolver returns the timestamps of the camera with the lowest mean frame rate."""
    # Give the body camera a lower frame rate than the face camera so it becomes the slowest reference clock.
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
    with pytest.raises(FileNotFoundError, match="no camera clock"):
        resolve_slowest_camera_clock(video_data_path=tmp_path)


def test_multi_recording_dataset_directory_is_lowercased() -> None:
    """Verifies the multi-recording directory name is lowercased to match cindra, so the forging writer and the
    experiment assembler resolve the same directory even when the dataset name or animal id carries uppercase.
    """
    assert multi_recording_dataset_directory(animal_id="101", dataset_name="Learning") == "101_learning"
    name = multi_recording_dataset_directory(animal_id="321", dataset_name="MaalstroomicFlow")
    assert name == name.lower()


def test_dispatch_routes_experiment_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies a mesoscope experiment session is routed to the experiment assembler with the dataset name forwarded."""
    monkeypatch.setattr(dispatcher_module, "SessionData", _stub_session_loader(SessionTypes.MESOSCOPE_EXPERIMENT))
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
    monkeypatch.setattr(dispatcher_module, "SessionData", _stub_session_loader(session_type))
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
    monkeypatch.setattr(dispatcher_module, "SessionData", _stub_session_loader(SessionTypes.WINDOW_CHECKING))
    _patch_assemblers(monkeypatch)

    with pytest.raises(ValueError, match="not a supported forging"):
        dispatcher_module.assemble_mesoscope_session(
            source_session_path=Path("/project/animal/session"),
            output_path=Path("/out/data.feather"),
            dataset_name="ds",
        )


def _write_runtime_state(directory: Path, timestamps: np.ndarray) -> None:
    """Writes a runtime-state feather carrying the given runtime-state entry timestamps."""
    pl.DataFrame(
        {"time_us": timestamps.astype(np.uint64), "runtime_state": np.ones(timestamps.size, dtype=np.uint8)}
    ).write_ipc(file=directory.joinpath(BehaviorDataFiles.RUNTIME_STATE))


def test_clip_to_runtime_end_drops_samples_past_the_last_runtime_entry(tmp_path: Path) -> None:
    """Verifies that samples acquired after the final runtime-state entry are discarded."""
    _write_runtime_state(directory=tmp_path, timestamps=np.array([0, 1_000, 2_000]))
    assembled = pl.DataFrame(
        {"time_us": np.array([0, 1_000, 2_000, 3_000, 4_000], dtype=np.uint64), "value": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )

    clipped = clip_to_runtime_end(assembled_data=assembled, runtime_data_path=tmp_path)

    assert clipped["time_us"].to_list() == [0, 1_000, 2_000]
    assert clipped["value"].to_list() == [1.0, 2.0, 3.0]


def test_clip_to_runtime_end_keeps_a_dataset_ending_with_the_runtime(tmp_path: Path) -> None:
    """Verifies that a dataset whose final sample coincides with the runtime end is left whole."""
    _write_runtime_state(directory=tmp_path, timestamps=np.array([0, 1_000, 2_000]))
    assembled = pl.DataFrame({"time_us": np.array([0, 1_000, 2_000], dtype=np.uint64), "value": [1.0, 2.0, 3.0]})

    clipped = clip_to_runtime_end(assembled_data=assembled, runtime_data_path=tmp_path)

    assert clipped.height == assembled.height


def test_clip_to_runtime_end_keeps_a_dataset_ending_before_the_runtime(tmp_path: Path) -> None:
    """Verifies that a reference clock ending before the runtime does is left whole.

    A camera clock can stop short of the final runtime-state entry, which leaves nothing to clip.
    """
    _write_runtime_state(directory=tmp_path, timestamps=np.array([0, 1_000, 5_000]))
    assembled = pl.DataFrame({"time_us": np.array([0, 1_000, 2_000], dtype=np.uint64), "value": [1.0, 2.0, 3.0]})

    clipped = clip_to_runtime_end(assembled_data=assembled, runtime_data_path=tmp_path)

    assert clipped.height == assembled.height
