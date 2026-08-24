"""Contains tests for the closure that snapshots what a finished batch's jobs recorded."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path

import pytest

from sollertia_forgery.server import JobStatus
from sollertia_forgery.orchestration import (
    BatchDocument,
    close_batch,
    read_ledger,
    resolve_batches,
    read_batch_outcome,
    close_settled_batches,
    record_prepared_batch,
    retire_settled_batches,
)
from sollertia_forgery.orchestration.ledger import SubmissionBatch, RemoteSubmission, record_batch
from sollertia_forgery.orchestration.closure import _OUTCOME_FIELD_LIMIT, _resolve_outcome

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sollertia_forgery.orchestration import BatchOutcome

pytestmark = pytest.mark.usefixtures("isolated_working_directory")

_UNIT_NAME: str = "2024_11_04"
"""The name of the processing unit the closure tests place their jobs under."""

_UNIT_PATH: str = f"/data/Project/305/{_UNIT_NAME}"
"""The path to the processing unit the closure tests place their jobs under."""


class _StubHost:
    """Stands in for an execution host, answering fixed state rows and recording what it was asked to materialize.

    Args:
        rows: The state rows this host answers every read with.
        fails: Determines whether materialization raises for every batch.
        failing_units: The unit paths whose materialization raises.

    Attributes:
        _rows: Cached state rows.
        _fails: Cached failure flag.
        _failing_units: Cached unit paths that fail to materialize.
        materialized: The number of materialization requests this host received.
    """

    def __init__(self, rows: list[dict[str, Any]], *, fails: bool = False, failing_units: tuple[str, ...] = ()) -> None:
        self._rows: list[dict[str, Any]] = rows
        self._fails: bool = fails
        self._failing_units: frozenset[str] = frozenset(failing_units)
        self.materialized: int = 0

    @property
    def label(self) -> str:
        """Returns the name this host is reported under."""
        return "remote"

    def materialize(
        self,
        project_root: Path,  # noqa: ARG002
        unit_paths: Sequence[Path],
        unit_kind: str,  # noqa: ARG002
        *,
        replan: bool,  # noqa: ARG002
    ) -> None:
        """Records the call, or fails when the stub is configured to fail for this batch's units.

        Args:
            project_root: The project the artifacts are regenerated for.
            unit_paths: The processing units the artifacts are regenerated for.
            unit_kind: The kind of processing unit the paths name.
            replan: Determines whether the plan is regenerated alongside the state artifacts.

        Raises:
            RuntimeError: If this stub is configured to fail for the batch's units.
        """
        self.materialized += 1
        if self._fails or any(str(unit_path) in self._failing_units for unit_path in unit_paths):
            message = "Unable to materialize the project artifacts. This stub host is configured to fail."
            raise RuntimeError(message)

    def fetch(self, path: Path, destination: Path) -> Path | None:
        """Writes a stand-in snapshot file so the outcome records a real location.

        Args:
            path: The artifact the host was asked to deliver.
            destination: The directory the artifact is delivered into.

        Returns:
            The path to the delivered snapshot.
        """
        destination.mkdir(parents=True, exist_ok=True)
        delivered = destination.joinpath(path.name)
        delivered.write_text("snapshot")
        return delivered

    def read_rows(self, path: Path) -> list[dict[str, Any]]:  # noqa: ARG002
        """Answers the fixed state rows.

        Args:
            path: The delivered artifact the rows would be read from.

        Returns:
            The state rows this stub was built with.
        """
        return self._rows


class _DatasetStubHost:
    """Stands in for a remote host holding one state table per dataset, answering each by the directory it sits in.

    Args:
        rows_by_dataset: The state rows each dataset's table holds, keyed by the dataset directory name.

    Attributes:
        _rows_by_dataset: Cached per-dataset state rows.
        read_paths: The artifact paths this host was asked to read, in order.
        delivered: The local paths this host delivered its artifacts to, in order.
    """

    def __init__(self, rows_by_dataset: dict[str, list[dict[str, Any]]]) -> None:
        self._rows_by_dataset: dict[str, list[dict[str, Any]]] = rows_by_dataset
        self.read_paths: list[Path] = []
        self.delivered: list[Path] = []

    @property
    def label(self) -> str:
        """Returns the name this host is reported under."""
        return "remote"

    def materialize(
        self,
        project_root: Path,
        unit_paths: Sequence[Path],
        unit_kind: str,
        *,
        replan: bool,
    ) -> None:
        """Accepts the regeneration request without rewriting anything."""

    def fetch(self, path: Path, destination: Path) -> Path | None:
        """Delivers a stand-in snapshot and records where it landed.

        Args:
            path: The artifact the host was asked to deliver.
            destination: The directory the artifact is delivered into.

        Returns:
            The path to the delivered snapshot.
        """
        destination.mkdir(parents=True, exist_ok=True)
        delivered = destination.joinpath(path.name)
        delivered.write_text("snapshot")
        self.delivered.append(delivered)
        return delivered

    def read_rows(self, path: Path) -> list[dict[str, Any]]:
        """Answers the rows the named dataset's table holds, recording the path this host was given.

        Args:
            path: The artifact the rows are read from.

        Returns:
            The state rows that dataset recorded, empty for a path naming no dataset this host holds.
        """
        self.read_paths.append(path)
        return self._rows_by_dataset.get(path.parent.name, [])


def _make_dataset_job(job_id: str, unit_path: str) -> dict[str, Any]:
    """Builds one dispatched forging job descriptor.

    Args:
        job_id: The identifier the descriptor is built under.
        unit_path: The dataset root the job runs against.

    Returns:
        The job descriptor.
    """
    return {
        "job_id": job_id,
        "job_name": "session_data_assembly",
        "specifier": job_id,
        "unit_path": unit_path,
        "unit_name": Path(unit_path).name,
        "pipeline": "forging",
        "prerequisite_ids": [],
    }


def _make_job(job_id: str, prerequisites: tuple[str, ...] = ()) -> dict[str, Any]:
    """Builds one dispatched job descriptor.

    Args:
        job_id: The identifier the descriptor is built under.
        prerequisites: The identifiers of the jobs this job waits on.

    Returns:
        The job descriptor.
    """
    return {
        "job_id": job_id,
        "job_name": "motion_energy",
        "specifier": job_id,
        "unit_path": _UNIT_PATH,
        "unit_name": _UNIT_NAME,
        "pipeline": "video",
        "prerequisite_ids": list(prerequisites),
    }


def _make_state_row(job_id: str, status: str, error_message: str | None = None) -> dict[str, Any]:
    """Builds one recorded state row.

    Args:
        job_id: The identifier of the job the row records.
        status: The status the job recorded.
        error_message: The failure text the job recorded.

    Returns:
        The state row.
    """
    return {"job_id": job_id, "status": status, "error_message": error_message, "session": _UNIT_NAME}


def _resolve_batch_outcome(
    jobs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    blocked: list[dict[str, Any]] | None = None,
) -> BatchOutcome:
    """Counts a stand-in batch against stand-in state rows.

    Args:
        jobs: The job descriptors the batch dispatched.
        rows: The state rows the jobs recorded.
        blocked: The job descriptors the batch was unable to dispatch.

    Returns:
        The batch's outcome.
    """
    document = BatchDocument(pipeline="video", host="remote", jobs=jobs, blocked_jobs=blocked or [])
    recorded = {_UNIT_NAME: {row["job_id"]: row for row in rows}}
    return _resolve_outcome(document=document, batch_id="batch", recorded=recorded, snapshots=[])


def _record_settled_batch(unit_path: str = _UNIT_PATH, slurm_job_id: str = "7") -> SubmissionBatch:
    """Records one prepared batch and the ledger entry naming the single allocation it holds.

    Args:
        unit_path: The processing unit the batch's single job runs against.
        slurm_job_id: The scheduler identifier of the allocation the batch holds.

    Returns:
        The recorded submission batch.
    """
    batch_id = record_prepared_batch(
        document=BatchDocument(
            pipeline="video",
            host="remote",
            jobs=[_make_job(job_id="a")],
            units=[{"unit_path": unit_path, "unit_name": Path(unit_path).name, "job_count": 1}],
        )
    )
    batch = SubmissionBatch(
        batch_id=batch_id,
        submissions=[RemoteSubmission(job_id="a", slurm_job_id=slurm_job_id, unit_path=unit_path)],
    )
    record_batch(batch=batch)
    return batch


def test_a_forging_batch_spanning_two_datasets_counts_every_dataset_it_dispatched() -> None:
    """A dataset batch reads one same-named state table per dataset, unlike a session batch's single project table.

    Each table therefore has to be read where the host holds it rather than from a delivered copy, and delivered
    somewhere it cannot overwrite the table another dataset already delivered under the same filename.
    """
    alpha, beta = "/data/Project/alpha", "/data/Project/beta"
    batch_id = record_prepared_batch(
        document=BatchDocument(
            pipeline="forging",
            host="remote",
            jobs=[_make_dataset_job(job_id="a", unit_path=alpha), _make_dataset_job(job_id="b", unit_path=beta)],
            units=[
                {"unit_path": alpha, "unit_name": "alpha", "job_count": 1},
                {"unit_path": beta, "unit_name": "beta", "job_count": 1},
            ],
        )
    )
    host = _DatasetStubHost(
        rows_by_dataset={
            "alpha": [{"job_id": "a", "status": "SUCCEEDED", "error_message": None, "dataset": "alpha"}],
            "beta": [{"job_id": "b", "status": "SUCCEEDED", "error_message": None, "dataset": "beta"}],
        }
    )

    outcome = close_batch(host=host, batch_id=batch_id)

    assert outcome is not None
    assert (outcome.succeeded, outcome.outstanding, outcome.failed) == (2, 0, 0)
    assert outcome.complete
    # Every table is read from the dataset directory the host holds it in, so a remote host resolves a path it owns.
    assert [path.parent.name for path in host.read_paths] == ["alpha", "beta"]
    # The two tables share a filename, so one destination for both would leave only the second on this machine.
    assert len(set(host.delivered)) == 2
    assert len(set(outcome.snapshot_paths)) == 2


def test_a_batch_whose_jobs_all_succeeded_reports_complete() -> None:
    """Verifies that a batch whose jobs all succeeded reports as complete."""
    outcome = _resolve_batch_outcome(
        jobs=[_make_job(job_id="a"), _make_job(job_id="b")],
        rows=[_make_state_row(job_id="a", status="SUCCEEDED"), _make_state_row(job_id="b", status="SUCCEEDED")],
    )

    assert (outcome.total, outcome.succeeded, outcome.failed, outcome.blocked, outcome.outstanding) == (2, 2, 0, 0, 0)
    assert outcome.complete


def test_a_failed_job_is_counted_and_carries_its_error_text() -> None:
    """Verifies that a failed job is counted and carries the error text it recorded."""
    outcome = _resolve_batch_outcome(
        jobs=[_make_job(job_id="a"), _make_job(job_id="b")],
        rows=[
            _make_state_row(job_id="a", status="SUCCEEDED"),
            _make_state_row(job_id="b", status="FAILED", error_message="decode failed"),
        ],
    )

    assert (outcome.succeeded, outcome.failed) == (1, 1)
    assert not outcome.complete
    assert outcome.failed_jobs[0]["error_message"] == "decode failed"


def test_a_job_waiting_on_a_failed_prerequisite_counts_as_blocked() -> None:
    """Verifies that a job waiting on a failed prerequisite is counted as blocked."""
    outcome = _resolve_batch_outcome(
        jobs=[_make_job(job_id="up"), _make_job(job_id="down", prerequisites=("up",))],
        rows=[_make_state_row(job_id="up", status="FAILED"), _make_state_row(job_id="down", status="SCHEDULED")],
    )

    # A blocked job is separated from work the batch was merely cut short of, because no rerun of it alone succeeds.
    assert (outcome.failed, outcome.blocked, outcome.outstanding) == (1, 1, 0)
    assert outcome.blocked_jobs[0]["unsatisfied_prerequisite_ids"] == ["up"]


def test_a_job_that_simply_never_ran_counts_as_outstanding() -> None:
    """Verifies that a job that never ran and waits on nothing failed is counted as outstanding."""
    outcome = _resolve_batch_outcome(
        jobs=[_make_job(job_id="a"), _make_job(job_id="b")],
        rows=[_make_state_row(job_id="a", status="SUCCEEDED"), _make_state_row(job_id="b", status="SCHEDULED")],
    )

    assert (outcome.succeeded, outcome.blocked, outcome.outstanding) == (1, 0, 1)
    assert not outcome.complete


def test_a_job_absent_from_the_state_artifact_counts_as_outstanding() -> None:
    """Verifies that a job absent from the state artifact is counted as outstanding."""
    outcome = _resolve_batch_outcome(jobs=[_make_job(job_id="a")], rows=[])

    assert (outcome.outstanding, outcome.succeeded) == (1, 0)


def test_jobs_blocked_at_preparation_reach_the_totals() -> None:
    """Verifies that jobs blocked at preparation are included in the batch totals."""
    blocked = [
        {
            "job_id": "late",
            "job_name": "motion_energy",
            "specifier": "late",
            "unit_name": _UNIT_NAME,
            "unsatisfied_prerequisite_ids": ["missing"],
        },
    ]
    outcome = _resolve_batch_outcome(
        jobs=[_make_job(job_id="a")], rows=[_make_state_row(job_id="a", status="SUCCEEDED")], blocked=blocked
    )

    assert (outcome.total, outcome.succeeded, outcome.blocked) == (2, 1, 1)
    # Completeness covers the blocked jobs too, so a batch still holding an undispatched job stays incomplete.
    assert not outcome.complete


def test_a_settled_batch_is_snapshotted_before_it_leaves_the_ledger() -> None:
    """Verifies that a settled batch is snapshotted before it leaves the ledger."""
    batch = _record_settled_batch()
    host = _StubHost(rows=[_make_state_row(job_id="a", status="SUCCEEDED")])

    closed = close_settled_batches(host=host, batches=[batch], statuses={"7": JobStatus.COMPLETED})

    assert [outcome.batch_id for outcome in closed] == [batch.batch_id]
    assert read_batch_outcome(batch_id=batch.batch_id)["complete"]
    assert not read_ledger().batches, "a closed batch stays outstanding"


def test_a_submission_dispatching_several_prepared_batches_snapshots_each_one_it_covered() -> None:
    """One submission may dispatch several prepared batches, and each carries its own document, so closing the
    submission has to snapshot every batch it covered rather than the one its ledger entry is keyed by.
    """
    covered = [
        record_prepared_batch(
            document=BatchDocument(
                pipeline="video",
                host="remote",
                jobs=[_make_job(job_id=job_id)],
                units=[{"unit_path": _UNIT_PATH, "unit_name": _UNIT_NAME, "job_count": 1}],
            )
        )
        for job_id in ("a", "b")
    ]
    batch = SubmissionBatch(
        batch_id=covered[0],
        batch_ids=covered,
        submissions=[RemoteSubmission(job_id="a", slurm_job_id="7", unit_path=_UNIT_PATH)],
    )
    record_batch(batch=batch)
    host = _StubHost(rows=[_make_state_row(job_id=job_id, status="SUCCEEDED") for job_id in ("a", "b")])

    closed = close_settled_batches(host=host, batches=[batch], statuses={"7": JobStatus.COMPLETED})

    assert [outcome.batch_id for outcome in closed] == covered
    # The second batch is the one a closure keyed by the ledger entry alone would leave open forever, since retiring
    # the entry drops the only record naming it.
    assert read_batch_outcome(batch_id=covered[1])["complete"]
    assert not read_ledger().batches


def test_a_batch_that_cannot_be_closed_stays_outstanding() -> None:
    """Verifies that a batch this host cannot snapshot stays in the ledger."""
    batch = _record_settled_batch()

    closed = close_settled_batches(
        host=_StubHost(rows=[], fails=True), batches=[batch], statuses={"7": JobStatus.FAILED}
    )

    # The ledger keeps the batch, because retiring one this host could not snapshot loses the run's record.
    assert not closed
    assert read_batch_outcome(batch_id=batch.batch_id) is None
    assert [recorded.batch_id for recorded in read_ledger().batches] == [batch.batch_id]


def test_one_batch_failing_to_close_leaves_the_others_retired() -> None:
    """Verifies that a batch failing to close leaves the siblings that were snapshotted retired."""
    failing_unit = "/data/Project/305/2024_11_05"
    healthy = _record_settled_batch()
    failing = _record_settled_batch(unit_path=failing_unit, slurm_job_id="8")
    host = _StubHost(rows=[_make_state_row(job_id="a", status="SUCCEEDED")], failing_units=(failing_unit,))

    closed = close_settled_batches(
        host=host,
        batches=[healthy, failing],
        statuses={"7": JobStatus.COMPLETED, "8": JobStatus.COMPLETED},
    )

    assert [outcome.batch_id for outcome in closed] == [healthy.batch_id]
    assert read_batch_outcome(batch_id=healthy.batch_id)["complete"]
    assert read_batch_outcome(batch_id=failing.batch_id) is None
    assert [recorded.batch_id for recorded in read_ledger().batches] == [failing.batch_id]


def test_a_batch_still_running_is_neither_closed_nor_retired() -> None:
    """Verifies that a batch holding an unsettled allocation is neither closed nor retired."""
    batch_id = record_prepared_batch(
        document=BatchDocument(pipeline="video", host="remote", jobs=[_make_job(job_id="a")])
    )
    batch = SubmissionBatch(
        batch_id=batch_id,
        submissions=[
            RemoteSubmission(job_id="a", slurm_job_id="7", unit_path=_UNIT_PATH),
            RemoteSubmission(job_id="b", slurm_job_id="8", unit_path=_UNIT_PATH),
        ],
    )
    record_batch(batch=batch)
    host = _StubHost(rows=[_make_state_row(job_id="a", status="SUCCEEDED")])

    closed = close_settled_batches(
        host=host, batches=[batch], statuses={"7": JobStatus.COMPLETED, "8": JobStatus.RUNNING}
    )

    assert not closed
    assert host.materialized == 0, "an unfinished batch triggered a materialization"
    assert [recorded.batch_id for recorded in read_ledger().batches] == [batch_id]


def test_the_enumerated_examples_are_capped_while_the_counts_stay_whole() -> None:
    """Verifies that the enumerated job examples are capped while the counts cover every job."""
    overflow = _OUTCOME_FIELD_LIMIT + 1
    jobs = [_make_job(job_id=f"up{index}") for index in range(overflow)]
    jobs.extend(_make_job(job_id=f"down{index}", prerequisites=(f"up{index}",)) for index in range(overflow))
    rows = [_make_state_row(job_id=f"up{index}", status="FAILED") for index in range(overflow)]
    rows.extend(_make_state_row(job_id=f"down{index}", status="SCHEDULED") for index in range(overflow))

    outcome = _resolve_batch_outcome(jobs=jobs, rows=rows)

    assert (outcome.failed, outcome.blocked, outcome.outstanding) == (overflow, overflow, 0)
    assert (len(outcome.failed_jobs), len(outcome.blocked_jobs)) == (_OUTCOME_FIELD_LIMIT, _OUTCOME_FIELD_LIMIT)


def test_a_settled_batch_this_host_never_prepared_is_retired_without_an_outcome() -> None:
    """Verifies that a settled batch this host holds no prepared record of is retired without an outcome."""
    batch = SubmissionBatch(
        batch_id="unprepared",
        submissions=[RemoteSubmission(job_id="a", slurm_job_id="7", unit_path=_UNIT_PATH)],
    )
    record_batch(batch=batch)
    host = _StubHost(rows=[])

    closed = close_settled_batches(host=host, batches=[batch], statuses={"7": JobStatus.COMPLETED})

    assert not closed
    assert host.materialized == 0, "a batch with no prepared record reached the host"
    assert not read_ledger().batches, "an unprepared batch stayed outstanding forever"


def test_a_batch_reports_every_pipeline_its_allocations_belong_to() -> None:
    """Verifies that a batch names each pipeline its allocations belong to once and in a stable order."""
    batch = SubmissionBatch(
        batch_id="mixed",
        submissions=[
            RemoteSubmission(job_id="a", slurm_job_id="7", pipeline="video"),
            RemoteSubmission(job_id="b", slurm_job_id="8", pipeline="checksum"),
            RemoteSubmission(job_id="c", slurm_job_id="9", pipeline="video"),
        ],
    )

    assert batch.pipelines == ["checksum", "video"]


def test_the_ledger_resolves_a_recorded_batch_by_identifier() -> None:
    """Verifies that the ledger resolves a recorded batch by identifier and reports an unknown one as absent."""
    batch = SubmissionBatch(batch_id="first", submissions=[RemoteSubmission(job_id="a", slurm_job_id="7")])
    record_batch(batch=batch)

    ledger = read_ledger()

    assert ledger.resolve_batch(batch_id="first") is not None
    assert ledger.resolve_batch(batch_id="first").batch_id == "first"
    assert ledger.resolve_batch(batch_id="absent") is None


def test_an_empty_status_query_retires_nothing() -> None:
    """Verifies that a status query observing no allocation retires no batch."""
    record_batch(batch=SubmissionBatch(batch_id="first", submissions=[RemoteSubmission(job_id="a", slurm_job_id="7")]))

    assert retire_settled_batches(statuses={}) == []
    assert [batch.batch_id for batch in read_ledger().batches] == ["first"]


def test_a_batch_with_a_live_allocation_is_left_outstanding() -> None:
    """Verifies that a batch holding one allocation still running is left outstanding."""
    record_batch(
        batch=SubmissionBatch(
            batch_id="first",
            submissions=[
                RemoteSubmission(job_id="a", slurm_job_id="7"),
                RemoteSubmission(job_id="b", slurm_job_id="8"),
            ],
        )
    )

    retired = retire_settled_batches(statuses={"7": JobStatus.COMPLETED, "8": JobStatus.RUNNING})

    assert retired == []
    assert [batch.batch_id for batch in read_ledger().batches] == ["first"]


def test_only_the_settled_batches_leave_the_ledger() -> None:
    """Verifies that a settled batch leaves the ledger while an unfinished one stays."""
    record_batch(batch=SubmissionBatch(batch_id="done", submissions=[RemoteSubmission(job_id="a", slurm_job_id="7")]))
    record_batch(batch=SubmissionBatch(batch_id="live", submissions=[RemoteSubmission(job_id="b", slurm_job_id="8")]))

    retired = retire_settled_batches(statuses={"7": JobStatus.COMPLETED, "8": JobStatus.PENDING})

    assert retired == ["done"]
    assert [batch.batch_id for batch in read_ledger().batches] == ["live"]


def test_a_batch_holding_no_allocation_is_never_reported_as_settled() -> None:
    """Verifies that a batch whose submissions were never recorded is never reported as settled."""
    record_batch(batch=SubmissionBatch(batch_id="empty"))

    assert retire_settled_batches(statuses={"7": JobStatus.COMPLETED}) == []
    assert [batch.batch_id for batch in read_ledger().batches] == ["empty"]


def test_naming_no_identifier_resolves_every_outstanding_batch() -> None:
    """Verifies that a status query naming no identifier resolves every outstanding batch."""
    record_batch(batch=SubmissionBatch(batch_id="first"))
    record_batch(batch=SubmissionBatch(batch_id="second"))
    ledger = read_ledger()

    assert [batch.batch_id for batch in resolve_batches(ledger=ledger)] == ["first", "second"]
    assert [batch.batch_id for batch in resolve_batches(ledger=ledger, batch_ids=["second"])] == ["second"]
    assert resolve_batches(ledger=ledger, batch_ids=[]) == []
