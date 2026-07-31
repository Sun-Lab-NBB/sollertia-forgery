"""Tests the system-agnostic substrate: the per-session tracker locations, the tracker reporters, the tracked-job
envelope, the shared utilities, and the terminal configuration the distribution applies when it is imported.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING
import importlib

import pytest
from ataraxis_base_utilities import console
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

import sollertia_forgery
from sollertia_forgery.shared_assets import (
    SESSION_PIPELINES,
    ProcessingPipelines,
    tracked_job,
    delay_terminal,
    summarize_tracker,
    derive_tracker_status,
    pinned_worker_threads,
    resolve_session_tracker_path,
    multi_recording_dataset_directory,
)
from sollertia_forgery.shared_assets.utilities import _WORKER_THREAD_VARIABLES

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from sollertia_shared_assets import SessionData

TRACKED_JOBS: list[tuple[str, str]] = [
    ("succeeded_stage", "one"),
    ("failed_stage", "two"),
    ("running_stage", "three"),
    ("scheduled_stage", "four"),
]
"""The four jobs the reporter fixture places in one of each recordable state."""


def job_identifier(job: tuple[str, str]) -> str:
    """Resolves the tracker identifier of one named job.

    Args:
        job: The job name and specifier pair the tracker records the job under.

    Returns:
        The hexadecimal identifier the tracker keys the job by.
    """
    return ProcessingTracker.generate_job_id(job_name=job[0], specifier=job[1])


@pytest.fixture
def mixed_tracker(tmp_path: Path, write_tracker: Callable[..., ProcessingTracker]) -> ProcessingTracker:
    """Writes one tracker holding a succeeded, a failed, a running, and a scheduled job.

    Args:
        tmp_path: The directory the tracker is written under.
        write_tracker: The builder that drives each job into the requested state.

    Returns:
        The written tracker.
    """
    return write_tracker(
        tmp_path.joinpath("mixed_tracker.yaml"),
        TRACKED_JOBS,
        succeeded=[TRACKED_JOBS[0]],
        failed={TRACKED_JOBS[1]: "the decoder rejected the archive"},
        running=[TRACKED_JOBS[2]],
        executor_id="slurm:4242",
    )


# Per-session tracker locations


def test_every_per_session_pipeline_resolves_the_location_its_own_session_declares(
    experiment_session: SessionData,
) -> None:
    """The reporting layer and the dispatch table both read this mapping, so it must name the session's own paths."""
    resolved = {
        pipeline: resolve_session_tracker_path(session=experiment_session, pipeline=pipeline)
        for pipeline in SESSION_PIPELINES
    }

    assert resolved == {
        ProcessingPipelines.CHECKSUM: experiment_session.raw_data.checksum_tracker_path,
        ProcessingPipelines.RUNTIME: experiment_session.processed_data.runtime_tracker_path,
        ProcessingPipelines.MICROCONTROLLER: experiment_session.processed_data.microcontroller_tracker_path,
        ProcessingPipelines.VIDEO: experiment_session.processed_data.video_tracker_path,
        ProcessingPipelines.TWO_PHOTON: experiment_session.processed_data.two_photon_tracker_path,
    }


def test_the_checksum_tracker_sits_beside_the_data_it_verifies(experiment_session: SessionData) -> None:
    """That pipeline verifies the acquired data in place, so its record belongs under the acquired data."""
    resolved = resolve_session_tracker_path(session=experiment_session, pipeline=ProcessingPipelines.CHECKSUM)

    assert resolved.parent == experiment_session.raw_data_path


@pytest.mark.parametrize("pipeline", [ProcessingPipelines.MANIFEST, ProcessingPipelines.FORGING])
def test_a_pipeline_that_processes_no_single_session_is_rejected(
    experiment_session: SessionData, pipeline: ProcessingPipelines
) -> None:
    """One pipeline operates on a project and the other on a dataset, so neither records a per-session tracker."""
    with pytest.raises(ValueError, match=f"tracker path of pipeline '{pipeline.value}'"):
        resolve_session_tracker_path(session=experiment_session, pipeline=pipeline)


# Tracker reporters


def test_a_tracker_summary_counts_every_state_it_holds(mixed_tracker: ProcessingTracker) -> None:
    """The counts are what a status label is derived from, so each state has to land in its own bucket."""
    summary = summarize_tracker(jobs=mixed_tracker.snapshot())["summary"]

    assert summary == {"total": 4, "succeeded": 1, "failed": 1, "running": 1, "scheduled": 1}


def test_each_summarized_job_carries_every_field_its_state_records(mixed_tracker: ProcessingTracker) -> None:
    """A consumer snapshotting tracker state serializes the job from this entry alone."""
    entries = {entry["job_id"]: entry for entry in summarize_tracker(jobs=mixed_tracker.snapshot())["jobs"]}
    succeeded = entries[job_identifier(TRACKED_JOBS[0])]

    assert succeeded["job_name"] == "succeeded_stage"
    assert succeeded["specifier"] == "one"
    assert succeeded["status"] == "SUCCEEDED"
    assert succeeded["executor_id"] == "slurm:4242"
    assert succeeded["started_at"] is not None
    assert succeeded["completed_at"] is not None
    assert "error_message" not in succeeded, "a job that recorded no failure carries a reason"


def test_a_failed_job_carries_the_reason_its_worker_recorded(mixed_tracker: ProcessingTracker) -> None:
    """The reason is the one field a caller cannot recover from the counts."""
    entries = {entry["job_id"]: entry for entry in summarize_tracker(jobs=mixed_tracker.snapshot())["jobs"]}

    assert entries[job_identifier(TRACKED_JOBS[1])]["error_message"] == "the decoder rejected the archive"


def test_an_empty_tracker_summarizes_to_zero_counts() -> None:
    """A tracker aligned against nothing reports totals rather than raising on an absent job registry."""
    assert summarize_tracker(jobs={}) == {
        "jobs": [],
        "summary": {"total": 0, "succeeded": 0, "failed": 0, "running": 0, "scheduled": 0},
    }


@pytest.mark.parametrize(
    "summary,expected",
    [
        ({"total": 3, "succeeded": 1, "failed": 1, "running": 1, "scheduled": 0}, "failed"),
        ({"total": 2, "succeeded": 2, "failed": 0, "running": 0, "scheduled": 0}, "completed"),
        ({"total": 2, "succeeded": 1, "failed": 0, "running": 1, "scheduled": 0}, "processing"),
        ({"total": 2, "succeeded": 0, "failed": 0, "running": 0, "scheduled": 2}, "not_started"),
        ({"total": 2, "succeeded": 1, "failed": 0, "running": 0, "scheduled": 1}, "in_progress"),
        ({"total": 0, "succeeded": 0, "failed": 0, "running": 0, "scheduled": 0}, "in_progress"),
        ({}, "in_progress"),
    ],
)
def test_the_status_label_follows_the_documented_priority(summary: dict[str, int], expected: str) -> None:
    """A failure outranks every other signal and a tracker holding no job at all is never reported as finished."""
    assert derive_tracker_status(summary=summary) == expected


def test_the_derived_label_matches_the_tracker_it_was_summarized_from(mixed_tracker: ProcessingTracker) -> None:
    """The two reporters are used as a pair, so the label has to follow the counts the summary actually produced."""
    assert derive_tracker_status(summary=summarize_tracker(jobs=mixed_tracker.snapshot())["summary"]) == "failed"


# Tracked-job envelope


def run_tracked_body(
    tracker: ProcessingTracker, job_id: str, entered: list[str], failure: BaseException | None
) -> None:
    """Runs one tracked job whose body records that it ran and then raises the requested failure.

    Args:
        tracker: The processing tracker recording the job's state transitions.
        job_id: The identifier of the job to run.
        entered: The list the body appends the job identifier to when it runs.
        failure: The exception the body raises, or None for a body that simply returns.
    """
    with tracked_job(tracker=tracker, job_id=job_id):
        entered.append(job_id)
        if failure is not None:
            raise failure


def test_a_job_that_returns_is_recorded_as_succeeded(
    tmp_path: Path, write_tracker: Callable[..., ProcessingTracker]
) -> None:
    """The envelope owns the state machine, so a body that simply returns completes the job without saying so."""
    tracker = write_tracker(tmp_path.joinpath("tracker.yaml"), [("stage", "one")])
    job_id = job_identifier(("stage", "one"))
    entered: list[str] = []

    run_tracked_body(tracker=tracker, job_id=job_id, entered=entered, failure=None)

    assert entered == [job_id]
    assert tracker.snapshot()[job_id].status is ProcessingStatus.SUCCEEDED


def test_a_failing_job_is_recorded_with_its_reason_and_the_exception_is_re_raised(
    tmp_path: Path, write_tracker: Callable[..., ProcessingTracker]
) -> None:
    """The caller still sees the original failure, and the tracker holds the message it carried."""
    tracker = write_tracker(tmp_path.joinpath("tracker.yaml"), [("stage", "one")])
    job_id = job_identifier(("stage", "one"))
    failure = RuntimeError("the archive is truncated")

    with pytest.raises(RuntimeError, match="the archive is truncated"):
        run_tracked_body(tracker=tracker, job_id=job_id, entered=[], failure=failure)

    job_state = tracker.snapshot()[job_id]
    assert job_state.status is ProcessingStatus.FAILED
    assert job_state.error_message == "the archive is truncated"


def test_an_interrupt_leaves_the_job_running(tmp_path: Path, write_tracker: Callable[..., ProcessingTracker]) -> None:
    """A run torn down by an interrupt recorded no outcome, so the record must not claim one."""
    tracker = write_tracker(tmp_path.joinpath("tracker.yaml"), [("stage", "one")])
    job_id = job_identifier(("stage", "one"))

    interrupt = KeyboardInterrupt("the operator interrupted the run")

    with pytest.raises(KeyboardInterrupt, match="the operator interrupted the run"):
        run_tracked_body(tracker=tracker, job_id=job_id, entered=[], failure=interrupt)

    assert tracker.snapshot()[job_id].status is ProcessingStatus.RUNNING


def test_an_unknown_job_identifier_is_rejected_before_the_body_runs(
    tmp_path: Path, write_tracker: Callable[..., ProcessingTracker]
) -> None:
    """The envelope starts the job first, so a tracker that holds no such job never enters the block."""
    tracker = write_tracker(tmp_path.joinpath("tracker.yaml"), [("stage", "one")])
    entered: list[str] = []

    with pytest.raises(ValueError, match="0123456789abcdef"):
        run_tracked_body(tracker=tracker, job_id="0123456789abcdef", entered=entered, failure=None)

    assert entered == []


# Shared utilities


def test_the_terminal_delay_holds_the_runtime_for_its_declared_period() -> None:
    """Consecutive printouts stay visually separated only if the delay actually elapses."""
    start = time.perf_counter()

    delay_terminal()

    assert time.perf_counter() - start >= 0.09


def test_the_multi_recording_directory_is_qualified_by_animal_and_lowercased() -> None:
    """cindra lowercases the configured dataset name, so the writer and the reader agree only when this one does."""
    assert multi_recording_dataset_directory(animal_id="305", dataset_name="PlaceCells") == "305_placecells"


def test_every_threading_layer_is_capped_inside_the_block() -> None:
    """A worker that inherits the defaults sizes a pool to the whole machine rather than to the core it was given."""
    with pinned_worker_threads():
        capped = {variable: os.environ.get(variable) for variable in _WORKER_THREAD_VARIABLES}

    assert capped == dict.fromkeys(_WORKER_THREAD_VARIABLES, "1")


def test_the_environment_is_restored_exactly_as_the_block_found_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leaving a cap in place would fail every later numba compilation, so a set value returns and an unset one goes."""
    monkeypatch.setenv("OMP_NUM_THREADS", "12")
    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)

    with pinned_worker_threads():
        pass

    assert os.environ["OMP_NUM_THREADS"] == "12"
    assert "NUMBA_NUM_THREADS" not in os.environ


# Import-time terminal configuration


def test_importing_the_library_turns_on_whichever_terminal_channel_is_off() -> None:
    """Every pipeline reports through the console and its progress bars, so importing has to switch on both."""
    console.disable()

    importlib.reload(sollertia_forgery)

    assert console.enabled, "an import found the console off and left it off"
    assert console.progress_enabled

    console.disable_progress()

    importlib.reload(sollertia_forgery)

    assert console.enabled
    assert console.progress_enabled, "an import found the progress bars off and left them off"
