"""Contains tests for the remote batch surface: the verdict a status read resolves for every outstanding allocation,
and the cancellation that stops the allocations a batch holds. It also covers the remediation that acts on those
verdicts before the ledger entries are dropped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self
from dataclasses import field, dataclass

import pytest

from sollertia_forgery.server import JobStatus
from sollertia_forgery.interfaces import remote_tools
from sollertia_forgery.orchestration import (
    STALLED_BATCH,
    NO_REMEDIATION,
    GONE_ALLOCATION,
    DROP_REMEDIATION,
    RESET_REMEDIATION,
    CANCEL_REMEDIATION,
    RUNNING_ALLOCATION,
    STRANDED_ALLOCATION,
    AWAITING_CLOSURE_BATCH,
    read_ledger,
    forget_batches,
)
from sollertia_forgery.orchestration.ledger import (
    SubmissionBatch,
    RemoteSubmission,
    record_batch,
    current_timestamp,
)
from sollertia_forgery.orchestration.remote import (
    HELD_ALLOCATION,
    FAILED_ALLOCATION,
    PROGRESSING_BATCH,
    SETTLED_ALLOCATION,
    FINISHED_ALLOCATION,
    ABANDONED_ALLOCATION,
    TrackerClaim,
)
from sollertia_forgery.orchestration.closure import _BatchOutcome
from sollertia_forgery.interfaces.remote_tools import (
    remote_batch_cancel,
    remote_batch_retire,
    remote_batch_status,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

pytestmark = pytest.mark.usefixtures("isolated_working_directory")

_UNIT_PATH: str = "/data/Project/305/2024_11_04"
"""The path to the processing unit under which these tests place their jobs."""

_MICROSECONDS_PER_HOUR: int = 60 * 60 * 1_000_000
"""The multiplier placing a recorded submission a whole number of hours in the past."""

_SNAPSHOT_FAILURE: RuntimeError = RuntimeError("the state table could not be regenerated")
"""The failure raised by the two steps that regenerate a host's state artifacts, which are the tracker read and the
outcome snapshot."""


@dataclass
class _RemoteStub:
    """Drives what the stubbed server answers and records what the tools asked it to do."""

    statuses: dict[str, JobStatus] = field(default_factory=dict)
    """The accounting state each allocation identifier reports."""
    queued: set[str] = field(default_factory=set)
    """The allocation identifiers the scheduler's queue holds."""
    claims: dict[tuple[str, str], TrackerClaim] = field(default_factory=dict)
    """What each job's own tracker records, keyed by unit path and job identifier."""
    connection_error: Exception | None = None
    """The failure the connection raises, or None when the server is reachable."""
    accounting_error: Exception | None = None
    """The failure the accounting query raises, or None when it answers."""
    queue_error: Exception | None = None
    """The failure the queue read raises, or None when it answers."""
    tracker_error: Exception | None = None
    """The failure the tracker read raises, or None when it answers."""
    cancel_error: Exception | None = None
    """The failure the cancellation raises, or None when it succeeds."""
    reset_error: Exception | None = None
    """The failure the tracker reset raises, or None when it succeeds."""
    closure_error: Exception | None = None
    """The failure the settled-batch closure raises, or None when it succeeds."""
    closure_retires: bool = True
    """Determines whether the closure retires the batches it settles, which a closure that fails does not."""
    records_during_closure: tuple[str, ...] = ()
    """The batches another process records while this call runs, written to the ledger inside the closure step."""
    snapshot_error: Exception | None = None
    """The failure the snapshot raises, or None when the outcome is recorded."""
    failing_batches: tuple[str, ...] = ()
    """The batches whose snapshot raises, which is every batch when the failure above is set."""
    snapshotted: list[str] = field(default_factory=list)
    """The batches whose outcome the remediation snapshotted."""
    cancelled: list[str] = field(default_factory=list)
    """The allocations the cancellation named."""
    reset: list[tuple[str, str]] = field(default_factory=list)
    """The unit path and job identifier of every job whose tracker was reset."""
    actions: list[str] = field(default_factory=list)
    """Every step the remediation took, in the order it took them."""


class _StubServer:
    """Stands in for the connected compute server, answering the scheduler records the stub holds."""

    def __init__(self, stub: _RemoteStub) -> None:
        self._stub = stub

    def __enter__(self) -> Self:
        """Returns this server as the connection a batch tool opens."""
        return self

    def __exit__(self, exception_type: object, exception: object, traceback: object) -> bool:
        """Closes the connection, reporting that it swallowed no exception."""
        return False

    def get_job_statuses(self, slurm_job_ids: Sequence[str]) -> dict[str, JobStatus]:
        """Answers the accounting state of every requested allocation."""
        if self._stub.accounting_error is not None:
            raise self._stub.accounting_error
        return {job_id: self._stub.statuses.get(job_id, JobStatus.UNRESOLVED) for job_id in slurm_job_ids}

    def get_queued_job_ids(self) -> set[str]:
        """Answers the allocations the scheduler's queue holds."""
        if self._stub.queue_error is not None:
            raise self._stub.queue_error
        return set(self._stub.queued)

    def abort_jobs(self, slurm_job_ids: Sequence[str]) -> None:
        """Records one cancellation request."""
        if self._stub.cancel_error is not None:
            raise self._stub.cancel_error
        self._stub.actions.append("cancel")
        self._stub.cancelled.extend(slurm_job_ids)


@pytest.fixture
def remote(monkeypatch: pytest.MonkeyPatch) -> _RemoteStub:
    """Replaces the server connection, the tracker read, the reset, and the closure with recorders under control."""
    stub = _RemoteStub()

    def _connect() -> _StubServer:
        if stub.connection_error is not None:
            raise stub.connection_error
        return _StubServer(stub=stub)

    def _claims(host: object, submissions: Sequence[RemoteSubmission]) -> dict[tuple[str, str], TrackerClaim]:
        if stub.tracker_error is not None:
            raise stub.tracker_error
        return {
            (entry.unit_path, entry.job_id): stub.claims.get((entry.unit_path, entry.job_id), TrackerClaim())
            for entry in submissions
        }

    def _reset(host: object, resolutions: Sequence[Any]) -> set[tuple[str, str]]:
        if stub.reset_error is not None:
            raise stub.reset_error
        released = {
            (entry.submission.unit_path, entry.submission.job_id)
            for entry in resolutions
            if entry.verdict == STRANDED_ALLOCATION
        }
        if released:
            stub.actions.append("reset")
            stub.reset.extend(sorted(released))
        return released

    def _close_settled(host: object, batches: Sequence[SubmissionBatch], resolutions: Sequence[Any]) -> list[Any]:
        if stub.closure_error is not None:
            raise stub.closure_error
        if not stub.closure_retires:
            return []
        # Mirrors the real derivation: a batch closes when every entry resolved for it prescribes a plain drop.
        prescribed: dict[str, list[str]] = {batch.batch_id: [] for batch in batches}
        for resolution in resolutions:
            prescribed[resolution.batch_id].append(resolution.remediation)
        settled = [
            batch.batch_id
            for batch in batches
            if all(remediation == DROP_REMEDIATION for remediation in prescribed[batch.batch_id])
        ]
        forget_batches(batch_ids=settled)
        for batch_id in stub.records_during_closure:
            _record_batch(batch_id=batch_id, allocations=("9000",))
        return [_BatchOutcome(batch_id=batch_id, complete=True) for batch_id in settled]

    def _close_covered(host: object, batch: SubmissionBatch) -> list[_BatchOutcome]:
        if batch.batch_id in stub.failing_batches:
            raise _SNAPSHOT_FAILURE
        if stub.snapshot_error is not None:
            raise stub.snapshot_error
        stub.actions.append("snapshot")
        stub.snapshotted.append(batch.batch_id)
        return [_BatchOutcome(batch_id=batch.batch_id, complete=True)]

    monkeypatch.setattr(remote_tools, "connect_to_server", _connect)
    monkeypatch.setattr(remote_tools, "resolve_tracker_claims", _claims)
    monkeypatch.setattr(remote_tools, "reset_stranded_jobs", _reset)
    monkeypatch.setattr(remote_tools, "close_settled_batches", _close_settled)
    monkeypatch.setattr(remote_tools, "close_covered_batches", _close_covered)
    return stub


def _record_batch(batch_id: str = "batch01", allocations: tuple[str, ...] = ("1000",), hours_ago: int = 1) -> None:
    """Records one ledger batch holding an allocation per named identifier."""
    record_batch(
        batch=SubmissionBatch(
            batch_id=batch_id,
            batch_ids=[batch_id],
            submitted_at=current_timestamp() - hours_ago * _MICROSECONDS_PER_HOUR,
            submissions=[
                RemoteSubmission(
                    job_id=f"job{index}",
                    slurm_job_id=allocation,
                    unit_path=_UNIT_PATH,
                    unit_name="2024_11_04",
                    pipeline="video",
                    job_name="motion_energy",
                    specifier=str(index),
                )
                for index, allocation in enumerate(allocations)
            ],
        )
    )


# Tests for the verdict a status read resolves


def test_an_allocation_the_queue_holds_reports_as_running(remote: _RemoteStub) -> None:
    """Verifies that a submission accounting has not committed yet is resolved from the queue rather than written
    off.
    """
    _record_batch()
    remote.statuses = {"1000": JobStatus.UNRESOLVED}
    remote.queued = {"1000"}

    response = remote_batch_status()
    reported = response["batches"][0]

    assert reported["progress"] == PROGRESSING_BATCH
    assert reported["verdicts"] == {RUNNING_ALLOCATION: 1}
    assert reported["running_allocations"] == ["1000"]
    assert reported["unresolvable_allocation_count"] == 0
    assert response["active"]
    assert response["stalled_batch_ids"] == []


def test_a_blocked_allocation_settles_its_batch_however_the_queue_reads(remote: _RemoteStub) -> None:
    """Verifies that a blocked allocation settles whether or not the queue snapshot still carries it."""
    _record_batch(allocations=("1000", "1001"))
    remote.statuses = {"1000": JobStatus.BLOCKED, "1001": JobStatus.UNRESOLVED}
    # The blocked state is derived from the queue's own reason field, so every allocation carrying it was queued.
    remote.queued = {"1000"}

    response = remote_batch_status()

    # A blocked allocation never runs, so it never writes to its job's tracker and nothing it holds changes
    # again. Resolving it from the queue would leave a run that never starts reported as progressing for as
    # long as the queue held it.
    assert response["batches"] == []
    assert "closed on it" in response["message"]
    assert not read_ledger().batches


def test_a_batch_neither_record_carries_an_allocation_of_reports_as_stalled(remote: _RemoteStub) -> None:
    """Verifies that a batch both scheduler records disclaim is stalled and names the tool that remediates it."""
    _record_batch(allocations=("1000", "1001"))
    remote.statuses = {"1000": JobStatus.UNRESOLVED, "1001": JobStatus.COMPLETED}
    remote.claims = {(_UNIT_PATH, "job1"): TrackerClaim(status="RUNNING", executor_id="slurm:1001", allocation="1001")}

    response = remote_batch_status()
    reported = response["batches"][0]

    assert reported["progress"] == STALLED_BATCH
    assert reported["unresolvable_allocations"] == ["1000"]
    assert reported["verdicts"] == {ABANDONED_ALLOCATION: 1, STRANDED_ALLOCATION: 1}
    # The batch stays outstanding because one of its jobs is stranded, the entry an automatic closure may not drop.
    assert response["stalled_batch_ids"] == ["batch01"]
    assert not response["active"]
    assert "retire_remote_batches_tool" in reported["remedy"]
    assert "slf server retire-batch" in reported["remedy"]


def test_a_stranded_job_is_named_and_carries_the_remediation_that_releases_it(remote: _RemoteStub) -> None:
    """Verifies that a tracker still claiming to run a job no allocation is running resolves as stranded."""
    _record_batch()
    remote.statuses = {"1000": JobStatus.UNRESOLVED}
    remote.claims = {(_UNIT_PATH, "job0"): TrackerClaim(status="RUNNING", executor_id="slurm:1000", allocation="1000")}

    response = remote_batch_status(include_items=True)
    reported = response["batches"][0]

    assert reported["progress"] == STALLED_BATCH
    assert reported["stranded_allocations"] == ["1000"]
    assert reported["verdicts"] == {STRANDED_ALLOCATION: 1}
    assert "stranded job(s) to the scheduled state" in reported["remedy"]
    assert response["jobs"][0]["verdict"] == STRANDED_ALLOCATION
    assert response["jobs"][0]["remediation"] == RESET_REMEDIATION
    assert response["jobs"][0]["scheduler_state"] == GONE_ALLOCATION
    assert response["jobs"][0]["tracker_status"] == "RUNNING"


def test_a_finished_job_is_never_resolved_as_stranded(remote: _RemoteStub) -> None:
    """Verifies that a job whose tracker recorded success is left with a verdict that touches no tracker."""
    # The second allocation the scheduler still holds keeps the batch outstanding, since a batch resolving every
    # entry to the plain drop closes on the read that resolves it.
    _record_batch(allocations=("1000", "1001"))
    remote.statuses = {"1000": JobStatus.UNRESOLVED, "1001": JobStatus.RUNNING}
    remote.claims = {(_UNIT_PATH, "job0"): TrackerClaim(status="SUCCEEDED")}

    listed = remote_batch_status(include_items=True)["jobs"][0]

    assert listed["verdict"] == FINISHED_ALLOCATION
    assert listed["remediation"] == DROP_REMEDIATION


def test_a_failed_job_keeps_the_verdict_its_operator_has_yet_to_see(remote: _RemoteStub) -> None:
    """Verifies that a recorded failure resolves as a failure rather than as work to be released."""
    _record_batch(allocations=("1000", "1001"))
    remote.statuses = {"1000": JobStatus.UNRESOLVED, "1001": JobStatus.RUNNING}
    remote.claims = {(_UNIT_PATH, "job0"): TrackerClaim(status="FAILED")}

    listed = remote_batch_status(include_items=True)["jobs"][0]

    assert listed["verdict"] == FAILED_ALLOCATION
    assert listed["remediation"] == DROP_REMEDIATION


def test_a_batch_that_closed_on_this_read_is_not_reported_outstanding(remote: _RemoteStub) -> None:
    """Verifies that the outstanding batches are resolved after the closure rather than before it."""
    _record_batch(batch_id="settled")
    _record_batch(batch_id="running", allocations=("2000",))
    remote.statuses = {"1000": JobStatus.COMPLETED, "2000": JobStatus.RUNNING}

    response = remote_batch_status()

    assert [entry["batch_id"] for entry in response["batches"]] == ["running"]
    assert [outcome["batch_id"] for outcome in response["outcomes"]] == ["settled"]
    assert response["summary"]["total"] == 1


def test_a_read_that_closed_every_batch_reports_none_outstanding(remote: _RemoteStub) -> None:
    """Verifies that a read whose closure emptied the ledger says so instead of reporting a batch that is gone."""
    _record_batch()
    remote.statuses = {"1000": JobStatus.COMPLETED}

    response = remote_batch_status()

    assert response["batches"] == []
    assert [outcome["batch_id"] for outcome in response["outcomes"]] == ["batch01"]
    assert "closed on it" in response["message"]
    assert not response["active"]


def test_a_batch_whose_closure_failed_reports_as_awaiting_closure(remote: _RemoteStub) -> None:
    """Verifies that a settled batch still in the ledger is reported as awaiting the closure that failed on it."""
    _record_batch()
    remote.statuses = {"1000": JobStatus.COMPLETED}
    remote.claims = {(_UNIT_PATH, "job0"): TrackerClaim(status="SUCCEEDED")}
    remote.closure_retires = False

    reported = remote_batch_status()["batches"][0]

    assert reported["progress"] == AWAITING_CLOSURE_BATCH
    assert reported["verdicts"] == {FINISHED_ALLOCATION: 1}
    assert "read this status again" in reported["remedy"]


def test_a_batch_the_queue_still_carries_is_not_closed_by_the_read(remote: _RemoteStub) -> None:
    """Verifies that the read closes exactly the batches its own resolution settles."""
    _record_batch()
    remote.statuses = {"1000": JobStatus.COMPLETED}
    remote.queued = {"1000"}

    response = remote_batch_status()

    # Accounting reports this allocation as terminal while the queue still carries it, so a closure reading
    # accounting alone would drop the ledger entry of a run the same read reports as progressing.
    assert response["outcomes"] == []
    assert [entry["batch_id"] for entry in response["batches"]] == ["batch01"]
    assert response["batches"][0]["progress"] == PROGRESSING_BATCH
    assert [batch.batch_id for batch in read_ledger().batches] == ["batch01"]


def test_a_batch_holding_no_allocation_closes_on_the_read(remote: _RemoteStub) -> None:
    """Verifies that a record the scheduler holds nothing of leaves the ledger rather than awaiting closure forever."""
    record_batch(batch=SubmissionBatch(batch_id="empty"))

    response = remote_batch_status()

    assert response["batches"] == []
    assert "closed on it" in response["message"]
    assert not read_ledger().batches


def test_the_active_flag_and_the_batch_progress_are_one_reading(remote: _RemoteStub) -> None:
    """Verifies that a batch running on a tracker claim alone reports as active, the way it reports as progressing."""
    _record_batch()
    remote.statuses = {"1000": JobStatus.UNRESOLVED, "2000": JobStatus.RUNNING}
    remote.claims = {(_UNIT_PATH, "job0"): TrackerClaim(status="RUNNING", executor_id="slurm:2000", allocation="2000")}

    response = remote_batch_status()

    assert response["active"]
    assert response["batches"][0]["progress"] == PROGRESSING_BATCH


def test_a_failed_queue_read_leaves_every_allocation_held(remote: _RemoteStub) -> None:
    """Verifies that a queue that cannot answer reports as itself and never resolves an allocation as gone."""
    _record_batch()
    remote.statuses = {"1000": JobStatus.UNRESOLVED}
    remote.queue_error = RuntimeError("squeue: error: Invalid user id")

    response = remote_batch_status()

    assert response["success"]
    assert "Invalid user id" in response["scheduler_read_error"]
    assert response["batches"][0]["progress"] == PROGRESSING_BATCH
    assert response["batches"][0]["verdicts"] == {RUNNING_ALLOCATION: 1}


def test_a_read_reports_how_long_a_batch_has_been_outstanding(remote: _RemoteStub) -> None:
    """Verifies that the outstanding period is measured from the submission time the ledger already records."""
    _record_batch(hours_ago=6)
    remote.statuses = {"1000": JobStatus.RUNNING}

    assert remote_batch_status()["batches"][0]["outstanding_seconds"] == pytest.approx(6 * 60 * 60, abs=10)


def test_a_record_carrying_no_submission_time_reports_no_outstanding_period(remote: _RemoteStub) -> None:
    """Verifies that a record written before the ledger dated its submissions reports no period rather than a false
    one.
    """
    record_batch(
        batch=SubmissionBatch(
            batch_id="legacy", submissions=[RemoteSubmission(job_id="a", slurm_job_id="1000", unit_path=_UNIT_PATH)]
        )
    )
    remote.statuses = {"1000": JobStatus.UNRESOLVED}
    remote.claims = {(_UNIT_PATH, "a"): TrackerClaim(status="RUNNING")}

    reported = remote_batch_status()["batches"][0]

    assert reported["outstanding_seconds"] is None
    assert reported["progress"] == STALLED_BATCH


def test_an_empty_ledger_is_reported_rather_than_queried(remote: _RemoteStub) -> None:
    """Verifies that a host with nothing outstanding never opens a connection to say so."""
    response = remote_batch_status()

    assert not response["active"]
    assert "No remote batch is outstanding" in response["message"]


def test_each_failed_read_of_a_status_call_reports_as_itself(remote: _RemoteStub) -> None:
    """Verifies that the connection, the trackers, the accounting, and the closure each name their own failure."""
    _record_batch()
    remote.statuses = {"1000": JobStatus.UNRESOLVED}

    remote.connection_error = ConnectionError("the compute server refused the connection")
    assert "Unable to reach the remote compute server" in remote_batch_status()["error"]
    remote.connection_error = None

    remote.tracker_error = _SNAPSHOT_FAILURE
    assert "on their own processing trackers" in remote_batch_status()["error"]
    remote.tracker_error = None

    remote.accounting_error = RuntimeError("slurmdbd is not responding")
    assert "scheduler accounting" in remote_batch_status()["error"]
    remote.accounting_error = None

    remote.closure_error = TimeoutError("the ledger lock is held")
    assert "resolves to a plain drop" in remote_batch_status()["error"]


def test_an_unknown_status_filter_names_the_states_it_accepts(remote: _RemoteStub) -> None:
    """Verifies that a filter naming no accounting state is rejected before anything is queried."""
    _record_batch()

    response = remote_batch_status(status_filter="STRANDED")

    assert not response["success"]
    assert "Unknown scheduler state" in response["error"]


def test_naming_a_batch_the_ledger_does_not_hold_is_rejected(remote: _RemoteStub) -> None:
    """Verifies that a status read never silently covers a batch other than the one it was asked for."""
    _record_batch()

    response = remote_batch_status(batch_ids=["absent"])

    assert not response["success"]
    assert "batch01" in response["error"]


def test_a_settled_batch_whose_job_is_stranded_is_never_closed_by_the_read(remote: _RemoteStub) -> None:
    """Verifies that the automatic closure covers the plain drop alone, so it can never drop a claim it also reports."""
    _record_batch()
    remote.statuses = {"1000": JobStatus.COMPLETED}
    remote.claims = {(_UNIT_PATH, "job0"): TrackerClaim(status="RUNNING", executor_id="slurm:1000", allocation="1000")}

    response = remote_batch_status()

    # The scheduler has finished with this allocation, so a closure resolving settlement of its own would retire
    # the batch. The same read reports the job as stranded, whose remediation writes to a tracker, and only the
    # explicit remediation performs that write.
    assert response["outcomes"] == []
    assert response["batches"][0]["verdicts"] == {STRANDED_ALLOCATION: 1}
    assert response["batches"][0]["stranded_allocations"] == ["1000"]
    assert [batch.batch_id for batch in read_ledger().batches] == ["batch01"]


def test_a_batch_whose_job_runs_outside_the_scheduler_is_neither_closed_nor_reset(remote: _RemoteStub) -> None:
    """Verifies that a tracker claiming a process rather than an allocation is refused rather than called stranded."""
    _record_batch()
    remote.statuses = {"1000": JobStatus.COMPLETED}
    remote.claims = {(_UNIT_PATH, "job0"): TrackerClaim(status="RUNNING", executor_id="pid:4821")}

    response = remote_batch_status(include_items=True)

    assert response["outcomes"] == []
    assert response["jobs"][0]["verdict"] == RUNNING_ALLOCATION
    # Neither scheduler record answers for such an executor, so the job can be shown neither to be live nor to
    # have stopped, and resetting its tracker could clear a claim its own process still writes.
    assert response["jobs"][0]["remediation"] == NO_REMEDIATION
    assert [batch.batch_id for batch in read_ledger().batches] == ["batch01"]


def test_a_batch_recorded_while_the_read_ran_is_named_rather_than_resolved(remote: _RemoteStub) -> None:
    """Verifies that a batch this read's records never covered is reported as uncovered instead of resolved from
    them.
    """
    _record_batch()
    remote.statuses = {"1000": JobStatus.RUNNING}
    remote.queued = {"1000"}
    remote.records_during_closure = ("latecomer",)

    response = remote_batch_status()

    assert [entry["batch_id"] for entry in response["batches"]] == ["batch01"]
    # The scheduler states and tracker claims are gathered before the closure, and the ledger is read again after
    # it, so resolving whatever that second read returns would answer for a batch against records taken before it
    # existed.
    assert response["uncovered_batch_ids"] == ["latecomer"]
    assert response["summary"]["total"] == 1
    assert "Read the status again" in response["message"]


# Tests for the cancellation of an outstanding batch


def test_cancelling_closes_only_the_batches_the_scheduler_released(remote: _RemoteStub) -> None:
    """Verifies that a cancellation retires the batch the scheduler finished with and leaves the one it still holds."""
    _record_batch(batch_id="released")
    _record_batch(batch_id="queued", allocations=("2000",))
    remote.statuses = {"1000": JobStatus.CANCELLED, "2000": JobStatus.RUNNING}
    remote.queued = {"2000"}

    response = remote_batch_cancel()

    assert response["canceled"]
    assert response["canceled_jobs"] == 2
    assert sorted(remote.cancelled) == ["1000", "2000"]
    assert [batch.batch_id for batch in read_ledger().batches] == ["queued"]


def test_cancelling_a_batch_the_ledger_does_not_hold_is_rejected(remote: _RemoteStub) -> None:
    """Verifies that a cancellation answers an unknown identifier the way its two sibling tools answer it."""
    _record_batch()

    response = remote_batch_cancel(batch_ids=["absent"])

    assert not response["success"]
    assert "batch01" in response["error"]
    assert not remote.cancelled


def test_each_failed_step_of_a_cancellation_reports_as_itself(remote: _RemoteStub) -> None:
    """Verifies that the connection, the cancellation, the two reads behind it, and the closure each name their own
    cause.
    """
    _record_batch()
    remote.statuses = {"1000": JobStatus.RUNNING}
    remote.queued = {"1000"}

    remote.connection_error = ConnectionError("the compute server refused the connection")
    assert "Unable to reach the remote compute server" in remote_batch_cancel()["error"]
    remote.connection_error = None

    # A cancellation that never reached the scheduler leaves the work running, so its error says nothing was cancelled.
    remote.cancel_error = RuntimeError("scancel: error: Invalid job id")
    refused = remote_batch_cancel()
    assert "Unable to cancel the allocations the named batches hold" in refused["error"]
    assert "nothing was cancelled" in refused["error"]
    remote.cancel_error = None

    # A read that fails behind an accepted cancellation leaves that cancellation standing, so the three errors
    # that follow each say the cancellation itself was issued.
    remote.tracker_error = _SNAPSHOT_FAILURE
    trackers = remote_batch_cancel()
    assert "on their own processing trackers" in trackers["error"]
    assert "cancellation itself was issued" in trackers["error"]
    remote.tracker_error = None

    remote.accounting_error = RuntimeError("slurmdbd is not responding")
    accounting = remote_batch_cancel()
    assert "scheduler accounting" in accounting["error"]
    assert "cancellation itself was issued" in accounting["error"]
    remote.accounting_error = None

    remote.closure_error = TimeoutError("the ledger lock is held")
    closure = remote_batch_cancel()
    assert "submission ledger" in closure["error"]
    assert "cancellation itself was issued" in closure["error"]


def test_cancelling_leaves_a_batch_whose_job_its_tracker_still_claims_outstanding(remote: _RemoteStub) -> None:
    """Verifies that a cancellation closes what the resolution drops and leaves a stranded claim for remediation."""
    _record_batch()
    remote.statuses = {"1000": JobStatus.CANCELLED}
    remote.claims = {(_UNIT_PATH, "job0"): TrackerClaim(status="RUNNING", executor_id="slurm:1000", allocation="1000")}

    response = remote_batch_cancel()

    assert response["canceled"]
    assert remote.cancelled == ["1000"]
    # The scheduler has released this allocation, so the batch would settle on the reading alone. Its job's tracker
    # still claims the run the cancellation stopped, which only an explicit remediation clears.
    assert [batch.batch_id for batch in read_ledger().batches] == ["batch01"]


def test_a_cancellation_that_cannot_read_the_ledger_reports_as_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that an unreadable ledger is reported as an error rather than raised out of the tool."""

    def _unreadable() -> None:
        message = "the ledger file is malformed"
        raise RuntimeError(message)

    monkeypatch.setattr(remote_tools, "read_ledger", _unreadable)

    response = remote_batch_cancel()

    assert not response["success"]
    assert "submission ledger" in response["error"]
    assert "malformed" in response["error"]


# Tests for the remediation that acts on those verdicts


def test_remediating_a_stranded_batch_resets_snapshots_and_drops_it(remote: _RemoteStub) -> None:
    """Verifies that the one verdict whose remediation writes to a tracker releases the job before the entry goes."""
    _record_batch()
    remote.statuses = {"1000": JobStatus.UNRESOLVED}
    remote.claims = {(_UNIT_PATH, "job0"): TrackerClaim(status="RUNNING", executor_id="slurm:1000", allocation="1000")}

    response = remote_batch_retire(batch_ids=["batch01"])

    assert response["retired"]
    assert response["batch_ids"] == ["batch01"]
    assert response["reset_jobs"] == 1
    assert remote.reset == [(_UNIT_PATH, "job0")]
    assert remote.actions == ["reset", "snapshot"]
    assert read_ledger().batches == []

    allocation = response["allocations"][0]
    assert allocation["verdict"] == STRANDED_ALLOCATION
    assert allocation["remediation"] == RESET_REMEDIATION
    assert not allocation["cancelled"]
    assert allocation["tracker_reset"]
    assert allocation["snapshot_recorded"]
    assert allocation["entry_dropped"]


def test_remediating_leaves_a_recorded_outcome_untouched(remote: _RemoteStub) -> None:
    """Verifies that neither a recorded success nor a recorded failure is cleared by a remediation."""
    _record_batch(allocations=("1000", "1001"))
    remote.statuses = {"1000": JobStatus.COMPLETED, "1001": JobStatus.FAILED}
    remote.claims = {
        (_UNIT_PATH, "job0"): TrackerClaim(status="SUCCEEDED"),
        (_UNIT_PATH, "job1"): TrackerClaim(status="FAILED"),
    }

    response = remote_batch_retire(batch_ids=["batch01"])

    assert response["reset_jobs"] == 0
    assert remote.reset == []
    assert [entry["verdict"] for entry in response["allocations"]] == [FINISHED_ALLOCATION, FAILED_ALLOCATION]
    assert all(not entry["tracker_reset"] for entry in response["allocations"])
    assert all(entry["entry_dropped"] for entry in response["allocations"])


def test_remediating_refuses_a_batch_holding_a_running_allocation(remote: _RemoteStub) -> None:
    """Verifies that the default refuses to disturb live work and names both the allocation and the waiver."""
    _record_batch(allocations=("1000", "1001"))
    remote.statuses = {"1000": JobStatus.RUNNING, "1001": JobStatus.UNRESOLVED}

    response = remote_batch_retire(batch_ids=["batch01"])

    assert not response["success"]
    assert "'1000'" in response["error"]
    assert "force=True" in response["error"]
    assert not remote.snapshotted
    assert not remote.reset
    assert [batch.batch_id for batch in read_ledger().batches] == ["batch01"]


def test_a_job_a_live_allocation_claims_refuses_the_remediation(remote: _RemoteStub) -> None:
    """Verifies that a job another machine is still running is refused rather than reset out from under it."""
    _record_batch()
    remote.statuses = {"1000": JobStatus.UNRESOLVED, "2000": JobStatus.RUNNING}
    remote.claims = {(_UNIT_PATH, "job0"): TrackerClaim(status="RUNNING", executor_id="slurm:2000", allocation="2000")}

    response = remote_batch_retire(batch_ids=["batch01"])

    assert not response["success"]
    assert "force=True" in response["error"]
    assert not remote.reset


def test_forcing_cancels_a_running_allocation_before_any_tracker_is_written(remote: _RemoteStub) -> None:
    """Verifies that the override cancels first, so a live allocation cannot write into a tracker reset under it."""
    _record_batch()
    remote.statuses = {"1000": JobStatus.RUNNING}
    remote.claims = {(_UNIT_PATH, "job0"): TrackerClaim(status="RUNNING", executor_id="slurm:1000", allocation="1000")}

    response = remote_batch_retire(batch_ids=["batch01"], force=True)

    assert response["retired"]
    assert remote.actions == ["cancel", "reset", "snapshot"]
    assert response["cancelled_allocations"] == ["1000"]
    assert response["allocations"][0]["cancelled"]
    assert response["allocations"][0]["tracker_reset"]
    assert response["allocations"][0]["remediation"] == CANCEL_REMEDIATION
    assert read_ledger().batches == []


def test_forcing_cancels_the_allocation_the_tracker_claims(remote: _RemoteStub) -> None:
    """Verifies that the override cancels the allocation actually carrying the job rather than the recorded one
    alone.
    """
    _record_batch()
    remote.statuses = {"1000": JobStatus.UNRESOLVED, "2000": JobStatus.RUNNING}
    remote.claims = {(_UNIT_PATH, "job0"): TrackerClaim(status="RUNNING", executor_id="slurm:2000", allocation="2000")}

    response = remote_batch_retire(batch_ids=["batch01"], force=True)

    assert response["retired"]
    assert remote.actions == ["cancel", "reset", "snapshot"]
    # The recorded allocation is gone and the live one is the claim, which another machine may have submitted.
    # Naming the recorded allocation alone would leave that live allocation free to write into the tracker this
    # remediation then resets.
    assert response["cancelled_allocations"] == ["2000"]
    assert response["reset_jobs"] == 1
    assert response["allocations"][0]["cancelled"]
    assert response["allocations"][0]["remediation"] == CANCEL_REMEDIATION
    assert response["allocations"][0]["tracker_reset"]


def test_forcing_never_clears_the_tracker_of_a_job_that_recorded_a_result(remote: _RemoteStub) -> None:
    """Verifies that the override waives the refusal alone, and never the rule that keeps a result from being lost."""
    _record_batch()
    remote.statuses = {"1000": JobStatus.RUNNING}
    remote.claims = {(_UNIT_PATH, "job0"): TrackerClaim(status="SUCCEEDED")}

    response = remote_batch_retire(batch_ids=["batch01"], force=True)

    assert response["retired"]
    assert remote.actions == ["cancel", "snapshot"]
    assert response["reset_jobs"] == 0
    assert not response["allocations"][0]["tracker_reset"]
    assert response["allocations"][0]["cancelled"]
    # The report names what ran rather than the sequence the flag is named for, since this tracker was left alone.
    assert response["allocations"][0]["remediation"] == DROP_REMEDIATION


def test_a_cancellation_that_fails_stops_the_remediation_with_nothing_changed(remote: _RemoteStub) -> None:
    """Verifies that a failed cancellation leaves the trackers and the ledger exactly as they stood."""
    _record_batch()
    remote.statuses = {"1000": JobStatus.RUNNING}
    remote.cancel_error = RuntimeError("scancel: error: Invalid job id")

    response = remote_batch_retire(batch_ids=["batch01"], force=True)

    assert not response["success"]
    assert "Unable to cancel the allocations" in response["error"]
    assert not remote.reset
    assert [batch.batch_id for batch in read_ledger().batches] == ["batch01"]


def test_a_reset_that_fails_refuses_the_drop(remote: _RemoteStub) -> None:
    """Verifies that a stranded job whose tracker could not be cleared keeps the entry that names its run."""
    _record_batch()
    remote.statuses = {"1000": JobStatus.UNRESOLVED}
    remote.claims = {(_UNIT_PATH, "job0"): TrackerClaim(status="RUNNING", executor_id="slurm:1000", allocation="1000")}
    remote.reset_error = RuntimeError("the reset command failed")

    response = remote_batch_retire(batch_ids=["batch01"])

    assert not response["success"]
    assert "Unable to return the stranded jobs" in response["error"]
    assert not remote.snapshotted
    assert [batch.batch_id for batch in read_ledger().batches] == ["batch01"]


def test_a_failed_snapshot_refuses_the_drop_and_names_its_waiver(remote: _RemoteStub) -> None:
    """Verifies that a batch whose outcome cannot be read keeps its entry, which is the last record naming the run."""
    _record_batch()
    remote.statuses = {"1000": JobStatus.UNRESOLVED}
    remote.snapshot_error = _SNAPSHOT_FAILURE

    response = remote_batch_retire(batch_ids=["batch01"])

    assert not response["success"]
    assert "the state table could not be regenerated" in response["error"]
    assert "drop_without_outcome=True" in response["error"]
    assert [batch.batch_id for batch in read_ledger().batches] == ["batch01"]


def test_dropping_without_an_outcome_remediates_a_batch_that_cannot_be_snapshotted(remote: _RemoteStub) -> None:
    """Verifies that the caller can drop an unsnapshottable entry and is told what was lost with it."""
    _record_batch()
    remote.statuses = {"1000": JobStatus.UNRESOLVED}
    remote.snapshot_error = _SNAPSHOT_FAILURE

    response = remote_batch_retire(batch_ids=["batch01"], drop_without_outcome=True)

    assert response["retired"]
    assert response["outcomes"] == []
    assert not response["allocations"][0]["snapshot_recorded"]
    assert "the state table could not be regenerated" in response["snapshot_error"]
    assert read_ledger().batches == []


def test_one_batch_failing_to_snapshot_keeps_the_outcomes_of_the_others(remote: _RemoteStub) -> None:
    """Verifies that each named batch is snapshotted on its own, so one failure does not discard the readable ones."""
    _record_batch(batch_id="healthy")
    _record_batch(batch_id="unreadable", allocations=("2000",))
    remote.statuses = {"1000": JobStatus.UNRESOLVED, "2000": JobStatus.UNRESOLVED}
    remote.failing_batches = ("unreadable",)

    response = remote_batch_retire(batch_ids=["healthy", "unreadable"], drop_without_outcome=True)

    assert [outcome["batch_id"] for outcome in response["outcomes"]] == ["healthy"]
    assert "'unreadable'" in response["snapshot_error"]
    assert "'healthy'" not in response["snapshot_error"]
    assert sorted(response["batch_ids"]) == ["healthy", "unreadable"]
    recorded = {entry["batch_id"]: entry["snapshot_recorded"] for entry in response["allocations"]}
    assert recorded == {"healthy": True, "unreadable": False}


def test_an_unreachable_server_refuses_the_remediation_until_both_waivers_are_given(remote: _RemoteStub) -> None:
    """Verifies that an unreachable server waives nothing on its own, since it hides every record at once."""
    _record_batch()
    remote.connection_error = ConnectionError("the compute server refused the connection")

    # Both waivers are needed because the failure withholds two separate guarantees, since the allocations cannot
    # be shown to have stopped and what their jobs recorded cannot be snapshotted.
    unverified = remote_batch_retire(batch_ids=["batch01"])
    assert not unverified["success"]
    assert "force=True" in unverified["error"]
    assert "refused the connection" in unverified["error"]

    unsnapshotted = remote_batch_retire(batch_ids=["batch01"], force=True)
    assert not unsnapshotted["success"]
    assert "drop_without_outcome=True" in unsnapshotted["error"]
    assert [batch.batch_id for batch in read_ledger().batches] == ["batch01"]

    retired = remote_batch_retire(batch_ids=["batch01"], force=True, drop_without_outcome=True)
    assert retired["retired"]
    assert retired["allocations"][0]["verdict"] == RUNNING_ALLOCATION
    assert not retired["allocations"][0]["cancelled"]
    assert not retired["allocations"][0]["tracker_reset"]
    assert retired["allocations"][0]["entry_dropped"]
    assert read_ledger().batches == []


def test_a_tracker_read_that_fails_reports_as_itself(remote: _RemoteStub) -> None:
    """Verifies that a remediation that cannot read the trackers says so rather than reporting a scheduler failure."""
    _record_batch()
    remote.tracker_error = _SNAPSHOT_FAILURE

    response = remote_batch_retire(batch_ids=["batch01"])

    assert not response["success"]
    assert "on their own processing trackers" in response["error"]
    assert [batch.batch_id for batch in read_ledger().batches] == ["batch01"]


def test_an_accounting_read_that_fails_reports_as_itself(remote: _RemoteStub) -> None:
    """Verifies that an accounting outage is named as one rather than folded into the tracker read."""
    _record_batch()
    remote.accounting_error = RuntimeError("slurmdbd is not responding")

    response = remote_batch_retire(batch_ids=["batch01"])

    assert not response["success"]
    assert "scheduler accounting" in response["error"]


def test_remediating_names_the_batches_it_drops(remote: _RemoteStub) -> None:
    """Verifies that remediation never defaults to the whole ledger and never accepts an identifier it does not
    hold.
    """
    _record_batch()
    remote.statuses = {"1000": JobStatus.UNRESOLVED}

    unnamed = remote_batch_retire(batch_ids=[])
    assert not unnamed["success"]
    assert "without an identifier" in unnamed["error"]

    unknown = remote_batch_retire(batch_ids=["absent"])
    assert not unknown["success"]
    assert "batch01" in unknown["error"]
    assert [batch.batch_id for batch in read_ledger().batches] == ["batch01"]


def test_remediating_reports_an_empty_ledger_rather_than_failing(remote: _RemoteStub) -> None:
    """Verifies that a host with nothing outstanding says so instead of reporting the identifier as unknown."""
    response = remote_batch_retire(batch_ids=["batch01"])

    assert not response["success"]
    assert "No remote batch is outstanding" in response["error"]


def test_a_job_running_outside_the_scheduler_is_refused_and_then_dropped_untouched(remote: _RemoteStub) -> None:
    """Verifies that the refusal for an unqueryable executor is waivable and never writes to that job's tracker."""
    _record_batch()
    remote.statuses = {"1000": JobStatus.COMPLETED}
    remote.claims = {(_UNIT_PATH, "job0"): TrackerClaim(status="RUNNING", executor_id="pid:4821")}

    refused = remote_batch_retire(batch_ids=["batch01"])
    assert not refused["success"]
    assert "executor outside the scheduler" in refused["error"]
    assert not remote.reset

    forced = remote_batch_retire(batch_ids=["batch01"], force=True)
    assert forced["retired"]
    # Nothing here can cancel a process the scheduler does not carry, so waiving the refusal drops the ledger entry
    # and reports the drop it actually applied rather than a cancellation and a reset that never ran.
    assert forced["cancelled_allocations"] == []
    assert forced["reset_jobs"] == 0
    assert not forced["allocations"][0]["cancelled"]
    assert not forced["allocations"][0]["tracker_reset"]
    assert forced["allocations"][0]["remediation"] == DROP_REMEDIATION
    assert read_ledger().batches == []


def test_the_batch_verdict_and_the_allocation_verdicts_are_one_resolution(remote: _RemoteStub) -> None:
    """Verifies that a batch is progressing exactly when one of its allocations resolves as running."""
    _record_batch(allocations=("1000", "1001", "1002"))
    remote.statuses = {"1000": JobStatus.RUNNING, "1001": JobStatus.COMPLETED, "1002": JobStatus.UNRESOLVED}
    remote.claims = {(_UNIT_PATH, "job1"): TrackerClaim(status="SUCCEEDED")}

    response = remote_batch_status(include_items=True)
    reported = response["batches"][0]
    verdicts = [entry["verdict"] for entry in response["jobs"]]

    assert verdicts == [RUNNING_ALLOCATION, FINISHED_ALLOCATION, ABANDONED_ALLOCATION]
    assert reported["progress"] == PROGRESSING_BATCH
    assert reported["verdicts"] == {ABANDONED_ALLOCATION: 1, FINISHED_ALLOCATION: 1, RUNNING_ALLOCATION: 1}
    assert response["breakdown"]["scheduler_state"] == {
        GONE_ALLOCATION: 1,
        HELD_ALLOCATION: 1,
        SETTLED_ALLOCATION: 1,
    }
    assert reported["progress"] != AWAITING_CLOSURE_BATCH
