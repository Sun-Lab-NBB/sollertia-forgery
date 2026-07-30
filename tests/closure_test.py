"""Tests the closure that snapshots what a finished batch's jobs recorded.

Closure is what makes a finished batch answerable, so these tests pin how each job is counted and pin the ordering that
matters most: a batch leaves the submission ledger only once its snapshot is on disk.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

import pytest
import platformdirs
from sollertia_shared_assets import set_working_directory

from sollertia_forgery.server import JobStatus
from sollertia_forgery.orchestration import (
    BatchDocument,
    read_ledger,
    close_settled_batches,
    read_batch_outcome,
    record_prepared_batch,
)
from sollertia_forgery.orchestration.closure import _resolve_outcome
from sollertia_forgery.orchestration.ledger import SubmissionBatch, RemoteSubmission, record_batch

UNIT = "/data/Project/305/2024_11_04"
"""The processing unit the closure tests place their jobs under."""


@pytest.fixture(autouse=True)
def isolated_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Points the Sollertia platform working directory at a pristine tmp-backed location."""
    working = tmp_path.joinpath("working")
    monkeypatch.setattr(platformdirs, "user_data_dir", lambda *_args, **_kwargs: str(tmp_path.joinpath("platform")))
    set_working_directory(path=working)
    return working


class StubHost:
    """Stands in for an execution host, answering fixed state rows and recording what it was asked to materialize."""

    def __init__(self, rows: list[dict[str, Any]], *, fails: bool = False) -> None:
        self._rows: list[dict[str, Any]] = rows
        self._fails: bool = fails
        self.materialized: int = 0

    @property
    def label(self) -> str:
        """Returns the name this host is reported under."""
        return "remote"

    def materialize(self, project_root: Path, unit_paths: Any, unit_kind: str, *, replan: bool) -> None:  # noqa: ANN401, ARG002
        """Records the call, or fails when the stub is configured to."""
        self.materialized += 1
        if self._fails:
            message = "the host could not regenerate its artifacts"
            raise RuntimeError(message)

    def fetch(self, path: Path, destination: Path) -> Path | None:
        """Writes a stand-in snapshot file so the outcome records a real location."""
        destination.mkdir(parents=True, exist_ok=True)
        delivered = destination.joinpath(path.name)
        delivered.write_text("snapshot")
        return delivered

    def read_rows(self, path: Path) -> list[dict[str, Any]]:  # noqa: ARG002
        """Answers the fixed state rows."""
        return self._rows


def job(job_id: str, prerequisites: tuple[str, ...] = ()) -> dict[str, Any]:
    """Builds one dispatched job descriptor."""
    return {
        "job_id": job_id,
        "job_name": "motion_energy",
        "specifier": job_id,
        "unit_path": UNIT,
        "unit_name": "2024_11_04",
        "pipeline": "video",
        "prerequisite_ids": list(prerequisites),
    }


def state(job_id: str, status: str, error_message: str | None = None) -> dict[str, Any]:
    """Builds one recorded state row."""
    return {"job_id": job_id, "status": status, "error_message": error_message, "session": "2024_11_04"}


def resolve(jobs: list[dict[str, Any]], rows: list[dict[str, Any]], blocked: list[dict[str, Any]] | None = None) -> Any:  # noqa: ANN401
    """Counts a stand-in batch against stand-in state rows."""
    document = BatchDocument(pipeline="video", host="remote", jobs=jobs, blocked_jobs=blocked or [])
    recorded = {"2024_11_04": {row["job_id"]: row for row in rows}}
    return _resolve_outcome(document=document, batch_id="batch", recorded=recorded, snapshots=[])


def test_a_batch_whose_jobs_all_succeeded_reports_complete() -> None:
    """The one signal a caller acts on is whether every job the batch held succeeded."""
    outcome = resolve(jobs=[job("a"), job("b")], rows=[state("a", "SUCCEEDED"), state("b", "SUCCEEDED")])

    assert (outcome.total, outcome.succeeded, outcome.failed, outcome.blocked, outcome.outstanding) == (2, 2, 0, 0, 0)
    assert outcome.complete


def test_a_failed_job_is_counted_and_carries_its_error_text() -> None:
    """A caller reading a failure needs the reason without opening a tracker of its own."""
    outcome = resolve(
        jobs=[job("a"), job("b")],
        rows=[state("a", "SUCCEEDED"), state("b", "FAILED", error_message="decode failed")],
    )

    assert (outcome.succeeded, outcome.failed) == (1, 1)
    assert not outcome.complete
    assert outcome.failed_jobs[0]["error_message"] == "decode failed"


def test_a_job_waiting_on_a_failed_prerequisite_counts_as_blocked() -> None:
    """No rerun of the job alone can succeed, which separates it from work the batch was merely cut short of."""
    outcome = resolve(
        jobs=[job("up"), job("down", prerequisites=("up",))],
        rows=[state("up", "FAILED"), state("down", "SCHEDULED")],
    )

    assert (outcome.failed, outcome.blocked, outcome.outstanding) == (1, 1, 0)
    assert outcome.blocked_jobs[0]["unsatisfied_prerequisite_ids"] == ["up"]


def test_a_job_that_simply_never_ran_counts_as_outstanding() -> None:
    """A batch that was canceled leaves jobs waiting on nothing that failed."""
    outcome = resolve(jobs=[job("a"), job("b")], rows=[state("a", "SUCCEEDED"), state("b", "SCHEDULED")])

    assert (outcome.succeeded, outcome.blocked, outcome.outstanding) == (1, 0, 1)
    assert not outcome.complete


def test_a_job_absent_from_the_state_artifact_counts_as_outstanding() -> None:
    """A tracker that holds no entry for a job describes one that never ran."""
    outcome = resolve(jobs=[job("a")], rows=[])

    assert (outcome.outstanding, outcome.succeeded) == (1, 0)


def test_jobs_blocked_at_preparation_reach_the_totals() -> None:
    """A batch's totals cover every job it held, including the ones it never dispatched."""
    blocked = [
        {
            "job_id": "late",
            "job_name": "motion_energy",
            "specifier": "late",
            "unit_name": "2024_11_04",
            "unsatisfied_prerequisite_ids": ["missing"],
        }
    ]
    outcome = resolve(jobs=[job("a")], rows=[state("a", "SUCCEEDED")], blocked=blocked)

    assert (outcome.total, outcome.succeeded, outcome.blocked) == (2, 1, 1)
    # Every job succeeding is not completeness while the batch still holds a job it could not dispatch.
    assert not outcome.complete


def test_a_settled_batch_is_snapshotted_before_it_leaves_the_ledger() -> None:
    """Closure precedes retirement, so a finished batch is answerable rather than forgotten."""
    batch_id = record_prepared_batch(
        document=BatchDocument(
            pipeline="video",
            host="remote",
            jobs=[job("a")],
            units=[{"unit_path": UNIT, "unit_name": "2024_11_04", "job_count": 1}],
        )
    )
    batch = SubmissionBatch(
        batch_id=batch_id, submissions=[RemoteSubmission(job_id="a", slurm_job_id="7", unit_path=UNIT)]
    )
    record_batch(batch=batch)
    host = StubHost(rows=[state("a", "SUCCEEDED")])

    closed = close_settled_batches(host=host, batches=[batch], statuses={"7": JobStatus.COMPLETED})

    assert [outcome.batch_id for outcome in closed] == [batch_id]
    assert read_batch_outcome(batch_id=batch_id)["complete"]
    assert not read_ledger().batches, "a closed batch stays outstanding"


def test_a_batch_that_cannot_be_closed_stays_outstanding() -> None:
    """Retiring a batch this host could not snapshot would lose the run's record, so the ledger keeps it."""
    batch_id = record_prepared_batch(
        document=BatchDocument(
            pipeline="video",
            host="remote",
            jobs=[job("a")],
            units=[{"unit_path": UNIT, "unit_name": "2024_11_04", "job_count": 1}],
        )
    )
    batch = SubmissionBatch(
        batch_id=batch_id, submissions=[RemoteSubmission(job_id="a", slurm_job_id="7", unit_path=UNIT)]
    )
    record_batch(batch=batch)

    closed = close_settled_batches(
        host=StubHost(rows=[], fails=True), batches=[batch], statuses={"7": JobStatus.FAILED}
    )

    assert not closed
    assert read_batch_outcome(batch_id=batch_id) is None
    assert [recorded.batch_id for recorded in read_ledger().batches] == [batch_id]


def test_a_batch_still_running_is_neither_closed_nor_retired() -> None:
    """An allocation that has yet to settle means the run is not over, so nothing is snapshotted."""
    batch_id = record_prepared_batch(document=BatchDocument(pipeline="video", host="remote", jobs=[job("a")]))
    batch = SubmissionBatch(
        batch_id=batch_id,
        submissions=[
            RemoteSubmission(job_id="a", slurm_job_id="7", unit_path=UNIT),
            RemoteSubmission(job_id="b", slurm_job_id="8", unit_path=UNIT),
        ],
    )
    record_batch(batch=batch)
    host = StubHost(rows=[state("a", "SUCCEEDED")])

    closed = close_settled_batches(
        host=host, batches=[batch], statuses={"7": JobStatus.COMPLETED, "8": JobStatus.RUNNING}
    )

    assert not closed
    assert host.materialized == 0, "an unfinished batch triggered a materialization"
    assert [recorded.batch_id for recorded in read_ledger().batches] == [batch_id]
