"""Tests the split between priming a two-photon recording and resolving its jobs.

cindra's job model lives in the per-plane bootstrap, and cindra requires one single-threaded step to write that
bootstrap before any job reads it. These tests pin that the priming step owns every write while job resolution stays a
read, so reading a session's job list never mutates its processed data.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from pathlib import Path

import pytest
from cindra import SingleRecordingJobNames

import sollertia_forgery.two_photon.pipeline as two_photon_pipeline
from sollertia_forgery.two_photon import discover_two_photon_jobs, prime_two_photon_recording
from sollertia_forgery.orchestration import resolve_dispatch
from sollertia_forgery.shared_assets import SESSION_PIPELINES, ProcessingPipelines


def _stub_session(session_path: Path) -> SimpleNamespace:
    """Builds a stand-in session exposing the attributes the pipeline reads."""
    cindra_directory = session_path.joinpath("processed_data", "cindra")
    cindra_directory.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        session_name=session_path.name, processed_data=SimpleNamespace(cindra_data_path=cindra_directory)
    )


def _patch_session_loader(monkeypatch: pytest.MonkeyPatch, session: SimpleNamespace) -> None:
    """Replaces the session loader so the pipeline resolves the stand-in session."""
    monkeypatch.setattr(
        two_photon_pipeline.SessionData,
        "load",
        classmethod(lambda _cls, session_path: session),  # noqa: ARG005
    )


def _forbid_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fails the test if anything attempts to materialize the configuration or the bootstrap."""

    def _refuse(*_args: Any, **_kwargs: Any) -> Any:  # noqa: ANN401
        message = "the call materialized the bootstrap"
        raise AssertionError(message)

    monkeypatch.setattr(two_photon_pipeline, "_resolve_configuration", _refuse)
    monkeypatch.setattr(two_photon_pipeline, "resolve_single_recording_contexts", _refuse)


def test_resolving_jobs_without_a_bootstrap_names_the_priming_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session that has never been primed carries no plane count, so resolution reports what to run first."""
    session = _stub_session(session_path=tmp_path.joinpath("2024_11_04"))
    _patch_session_loader(monkeypatch=monkeypatch, session=session)
    monkeypatch.setattr(two_photon_pipeline, "_resolve_primed_plane_count", lambda session: None)  # noqa: ARG005

    with pytest.raises(FileNotFoundError, match="Prime the recording"):
        discover_two_photon_jobs(session_path=tmp_path.joinpath("2024_11_04"))


def test_resolving_jobs_reads_the_bootstrap_and_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolution shapes the universe from the primed plane count without materializing any part of the bootstrap."""
    session = _stub_session(session_path=tmp_path.joinpath("2024_11_04"))
    _patch_session_loader(monkeypatch=monkeypatch, session=session)
    monkeypatch.setattr(two_photon_pipeline, "_resolve_primed_plane_count", lambda session: 2)  # noqa: ARG005
    _forbid_writes(monkeypatch=monkeypatch)

    _resolved, universe, possible = discover_two_photon_jobs(session_path=tmp_path.joinpath("2024_11_04"))

    assert universe == [
        (str(SingleRecordingJobNames.BINARIZE), ""),
        (str(SingleRecordingJobNames.REGISTER), "plane_0"),
        (str(SingleRecordingJobNames.REGISTER), "plane_1"),
        (str(SingleRecordingJobNames.PROCESS), "plane_0"),
        (str(SingleRecordingJobNames.PROCESS), "plane_1"),
        (str(SingleRecordingJobNames.COMBINE), ""),
    ]
    # Every stage is possible once the bootstrap exists, so the subset covers the whole universe.
    assert possible == universe


def test_priming_an_already_primed_recording_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Priming is idempotent, so repeated preparation of one session costs a read rather than a rewrite."""
    session = _stub_session(session_path=tmp_path.joinpath("2024_11_04"))
    _patch_session_loader(monkeypatch=monkeypatch, session=session)
    monkeypatch.setattr(two_photon_pipeline, "_resolve_primed_plane_count", lambda session: 3)  # noqa: ARG005
    _forbid_writes(monkeypatch=monkeypatch)

    prime_two_photon_recording(session_path=tmp_path.joinpath("2024_11_04"))


def test_priming_an_unprimed_recording_materializes_the_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent bootstrap is written once, which is the single-threaded step cindra requires before any job runs."""
    session = _stub_session(session_path=tmp_path.joinpath("2024_11_04"))
    _patch_session_loader(monkeypatch=monkeypatch, session=session)
    monkeypatch.setattr(two_photon_pipeline, "_resolve_primed_plane_count", lambda session: None)  # noqa: ARG005

    persisted: list[bool] = []

    def _record_configuration(session: Any, *, display_progress: bool, persist: bool) -> tuple[None, Path]:  # noqa: ANN401, ARG001
        persisted.append(persist)
        return None, Path()

    def _record_contexts(configuration: Any, *, persist: bool) -> list[Any]:  # noqa: ANN401, ARG001
        persisted.append(persist)
        return []

    monkeypatch.setattr(two_photon_pipeline, "_resolve_configuration", _record_configuration)
    monkeypatch.setattr(two_photon_pipeline, "resolve_single_recording_contexts", _record_contexts)

    prime_two_photon_recording(session_path=tmp_path.joinpath("2024_11_04"))

    # Both halves of the bootstrap are written, which is what an unprimed recording needs before its jobs dispatch.
    assert persisted == [True, True]


def test_only_the_two_photon_pipeline_declares_a_priming_step() -> None:
    """Priming exists for the one pipeline whose job model lives in state a dependency writes."""
    priming = {
        pipeline: resolve_dispatch(pipeline=pipeline).prime is not None  # type: ignore[union-attr]
        for pipeline in (*SESSION_PIPELINES, ProcessingPipelines.FORGING)
    }
    assert priming[ProcessingPipelines.TWO_PHOTON]
    assert not any(declares for pipeline, declares in priming.items() if pipeline is not ProcessingPipelines.TWO_PHOTON)
