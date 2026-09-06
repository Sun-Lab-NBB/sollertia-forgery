"""Contains tests for job reconciliation and for the on-disk prepared-batch registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from sollertia_forgery.server import JobStatus
from sollertia_forgery.orchestration import (
    REMOTE_JOB_WALLTIME_MINUTES,
    resolve_batch_host,
    reconcile_local_jobs,
    read_prepared_batches,
    reconcile_remote_jobs,
    record_prepared_batch,
)
from sollertia_forgery.orchestration.graph import BatchDocument, build_pending_job
from sollertia_forgery.orchestration.ledger import (
    SubmissionBatch,
    RemoteSubmission,
    record_batch,
    current_timestamp,
)
from sollertia_forgery.orchestration.batches import read_prepared_batch, forget_batch_records

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from sollertia_forgery.orchestration import GenericPendingJob

pytestmark = pytest.mark.usefixtures("isolated_working_directory")

_UNIT_PATH: str = "/data/Project/305/2024_11_04"
"""The path to the processing unit under which the reconciliation tests place their jobs."""

_TRACKER_JOB: tuple[str, str] = ("motion_energy", "1")
"""The job name and specifier against which every tracker written by these tests is aligned."""

_SECOND_TRACKER_JOB: tuple[str, str] = ("motion_energy", "2")
"""The second job a multi-job tracker holds, which the ordering tests place downstream of the first."""

_THIRD_TRACKER_JOB: tuple[str, str] = ("motion_energy", "3")
"""The third job a multi-job tracker holds, which waits on nothing and is therefore dispatched on its own."""


@pytest.fixture
def running_job(
    tmp_path: Path, write_tracker: Callable[..., ProcessingTracker]
) -> Callable[[str | None], GenericPendingJob]:
    """Returns a builder that writes a tracker recording one job as running and points a pending job at it.

    Args:
        tmp_path: The temporary directory under which the tracker is written.
        write_tracker: The builder that writes the tracker.

    Returns:
        A callable taking the executor identifier the running record names, and returning the pending job.
    """

    def _build(executor_id: str | None) -> GenericPendingJob:
        tracker_path = tmp_path.joinpath("tracker.yaml")
        write_tracker(path=tracker_path, jobs=[_TRACKER_JOB], running=[_TRACKER_JOB], executor_id=executor_id)
        return _make_job(
            job_id=_tracker_job_id(),
            tracker_path=tracker_path,
            status=ProcessingStatus.RUNNING.name,
            executor_id=executor_id or "",
        )

    return _build


@pytest.fixture
def scheduled_job(tmp_path: Path, write_tracker: Callable[..., ProcessingTracker]) -> GenericPendingJob:
    """Builds a pending job whose tracker holds it in the scheduled state, so the record names no executor.

    Args:
        tmp_path: The temporary directory under which the tracker is written.
        write_tracker: The builder that writes the tracker.

    Returns:
        The pending job pointing at the written tracker.
    """
    tracker_path = tmp_path.joinpath("tracker.yaml")
    write_tracker(path=tracker_path, jobs=[_TRACKER_JOB])
    return _make_job(job_id=_tracker_job_id(), tracker_path=tracker_path, status=ProcessingStatus.SCHEDULED.name)


class _StubServer:
    """Stands in for a connected server, answering the two scheduler records a test places on it.

    Args:
        statuses: The scheduler state this server answers for each allocation identifier.
        queued: The allocation identifiers this server's queue holds.
        queue_error: The failure the queue read raises, or None when it answers.

    Attributes:
        _statuses: Cached scheduler states.
        _queued: Cached queue membership.
        _queue_error: Cached queue failure.
        queried: The allocation identifiers of the most recent accounting query.
        queue_reads: How many times the queue was read.
    """

    def __init__(
        self, statuses: dict[str, JobStatus], queued: set[str] | None = None, queue_error: Exception | None = None
    ) -> None:
        self._statuses: dict[str, JobStatus] = statuses
        self._queued: set[str] = set(queued or ())
        self._queue_error: Exception | None = queue_error
        self.queried: list[str] = []
        self.queue_reads: int = 0

    def get_job_statuses(self, slurm_job_ids: list[str]) -> dict[str, JobStatus]:
        """Records the queried allocations and answers the fixed state of each.

        Args:
            slurm_job_ids: The allocation identifiers about which the caller asked.

        Returns:
            The state of each queried allocation for which this server holds one.
        """
        self.queried = list(slurm_job_ids)
        return {allocation: self._statuses[allocation] for allocation in slurm_job_ids if allocation in self._statuses}

    def get_queued_job_ids(self) -> set[str]:
        """Records one queue read and answers the identifiers the queue holds.

        Returns:
            The queued allocation identifiers.
        """
        self.queue_reads += 1
        if self._queue_error is not None:
            raise self._queue_error
        return set(self._queued)


def _claiming_batch(job_id: str) -> SubmissionBatch:
    """Builds one recorded batch claiming the given job with a single allocation.

    Args:
        job_id: The identifier of the job the batch's allocation claims.

    Returns:
        The recorded batch.
    """
    return SubmissionBatch(
        batch_id="earlier",
        submitted_at=current_timestamp(),
        walltime_minutes=REMOTE_JOB_WALLTIME_MINUTES,
        submissions=[RemoteSubmission(job_id=job_id, slurm_job_id="777", unit_path=_UNIT_PATH, pipeline="video")],
    )


def _tracker_job_id(job: tuple[str, str] = _TRACKER_JOB) -> str:
    """Returns the identifier under which the tracker registers one job name and specifier.

    Args:
        job: The job name and specifier to identify.

    Returns:
        The job identifier.
    """
    return ProcessingTracker.generate_job_id(job_name=job[0], specifier=job[1])


def _make_job(
    job_id: str,
    tracker_path: Path,
    unit_path: str = _UNIT_PATH,
    status: str = "",
    executor_id: str = "",
    specifier: str = _TRACKER_JOB[1],
    prerequisite_ids: tuple[str, ...] = (),
) -> GenericPendingJob:
    """Builds a pending job carrying the outcome that preparation read out of the host's state artifact.

    Args:
        job_id: The identifier under which the job is registered in the tracker.
        tracker_path: The path to the tracker on which the job is recorded.
        unit_path: The processing unit against which the job runs.
        status: The status the state artifact recorded for the job.
        executor_id: The executor the same record named.
        specifier: The specifier that differentiates this job from the others of its type in the same unit.
        prerequisite_ids: The identifiers of the jobs this one waits on.

    Returns:
        The pending job.
    """
    return build_pending_job(
        job={
            "job_id": job_id,
            "job_name": _TRACKER_JOB[0],
            "specifier": specifier,
            "unit_path": unit_path,
            "unit_name": "2024_11_04",
            "pipeline": "video",
            "tracker_path": str(tracker_path),
            "cores": 4,
            "memory_mb": 1024,
            "resident_mb": 2048,
            "prerequisite_ids": list(prerequisite_ids),
            "options": {},
            "status": status,
            "executor_id": executor_id,
        }
    )


def test_a_local_batch_adopts_nothing_and_reruns_every_job(
    running_job: Callable[[str | None], GenericPendingJob],
) -> None:
    """Verifies that a local batch adopts no job and reruns every job a tracker records as running."""
    job = running_job("pid:4242")

    reconciliation = reconcile_local_jobs(jobs=[job])

    # Nothing else on this machine holds a job, so a running record describes a pool that died.
    assert not reconciliation.adopted
    assert [dispatched.job_id for dispatched in reconciliation.dispatchable] == [job.job_id]
    assert [resettable.job_id for resettable in reconciliation.resettable] == [job.job_id]


def test_a_live_allocation_named_by_the_tracker_is_adopted(
    running_job: Callable[[str | None], GenericPendingJob],
) -> None:
    """Verifies that a job whose tracker names a live scheduler allocation is adopted."""
    job = running_job("slurm:991")
    server = _StubServer(statuses={"991": JobStatus.RUNNING})

    reconciliation = reconcile_remote_jobs(server=server, jobs=[job])

    # An executor identifier travels with the data, so it covers an allocation submitted from another machine.
    assert reconciliation.adopted == {(_UNIT_PATH, job.job_id): "991"}
    assert not reconciliation.dispatchable
    # An adopted job is never reset, since that would wipe a record its allocation is still writing.
    assert not reconciliation.resettable


def test_the_allocation_the_tracker_names_outranks_the_one_the_ledger_recorded(
    running_job: Callable[[str | None], GenericPendingJob],
) -> None:
    """Verifies that a job claimed by both sources is adopted onto the allocation its tracker names, not the
    ledger's.
    """
    job = running_job("slurm:991")
    record_batch(
        batch=SubmissionBatch(
            batch_id="earlier",
            submissions=[
                RemoteSubmission(job_id=job.job_id, slurm_job_id="777", unit_path=_UNIT_PATH, pipeline="video")
            ],
        )
    )
    server = _StubServer(statuses={"777": JobStatus.FAILED, "991": JobStatus.RUNNING})

    reconciliation = reconcile_remote_jobs(server=server, jobs=[job])

    # An executor identifier appears only once an allocation starts running, so it describes a later moment than
    # this host's record of submitting one. Preferring the ledger's stale identifier would query an allocation
    # that has already finished and submit a second one over the files the live allocation still writes.
    assert reconciliation.adopted == {(_UNIT_PATH, job.job_id): "991"}
    # Dispatching the job again would run two allocations over the same tracker and the same output.
    assert not reconciliation.dispatchable
    assert not reconciliation.resettable


def test_an_executor_naming_the_scheduler_but_no_allocation_is_withheld(
    running_job: Callable[[str | None], GenericPendingJob],
) -> None:
    """Verifies that a truncated executor identifier names nothing to query and nothing to adopt, so its job is held."""
    job = running_job("slurm:")
    server = _StubServer(statuses={})

    reconciliation = reconcile_remote_jobs(server=server, jobs=[job])

    assert server.queried == [], "a job claiming no allocation sent the scheduler a query"
    assert not reconciliation.adopted
    # A record carrying the scheme without an allocation names nothing the scheduler can resolve, so the
    # resolution reports the job as running and prescribes no remediation. Dispatching it would clear a record
    # its own executor may still be writing, and it names no allocation for a dependent to await.
    assert not reconciliation.dispatchable
    assert not reconciliation.resettable
    assert [withheld.job_id for withheld in reconciliation.withheld] == [job.job_id]


def test_a_finished_allocation_is_submitted_again(running_job: Callable[[str | None], GenericPendingJob]) -> None:
    """Verifies that a job whose tracker names a finished allocation is submitted again."""
    job = running_job("slurm:991")
    server = _StubServer(statuses={"991": JobStatus.FAILED})

    reconciliation = reconcile_remote_jobs(server=server, jobs=[job])

    # A record left running by an allocation that has since finished describes work that never completed.
    assert not reconciliation.adopted
    assert [dispatched.job_id for dispatched in reconciliation.dispatchable] == [job.job_id]


def test_an_off_scheduler_executor_is_withheld_rather_than_run_a_second_time(
    running_job: Callable[[str | None], GenericPendingJob],
) -> None:
    """Verifies that a job whose tracker names a bare process identifier is neither submitted nor cleared."""
    job = running_job("pid:4242")
    server = _StubServer(statuses={})

    reconciliation = reconcile_remote_jobs(server=server, jobs=[job])

    # A bare process identifier was never a scheduler allocation, so there is nothing to adopt onto either.
    assert not reconciliation.adopted
    # Neither scheduler record answers for such an executor, so the job can be shown neither to be live nor to
    # have stopped. Submitting an allocation for it would run a second copy over the tracker and the output its
    # own process may still be writing.
    assert not reconciliation.dispatchable
    assert not reconciliation.resettable
    assert [withheld.job_id for withheld in reconciliation.withheld] == [job.job_id]


def test_the_ledger_claims_an_allocation_the_tracker_cannot_yet_name(scheduled_job: GenericPendingJob) -> None:
    """Verifies that the ledger supplies the allocation claim for a job the tracker cannot yet name."""
    record_batch(
        batch=SubmissionBatch(
            batch_id="earlier",
            submissions=[
                RemoteSubmission(
                    job_id=scheduled_job.job_id, slurm_job_id="777", unit_path=_UNIT_PATH, pipeline="video"
                )
            ],
        )
    )
    server = _StubServer(statuses={"777": JobStatus.PENDING})

    reconciliation = reconcile_remote_jobs(server=server, jobs=[scheduled_job])

    # A queued allocation has not started, so the ledger is the only source that knows it exists.
    assert reconciliation.adopted == {(_UNIT_PATH, scheduled_job.job_id): "777"}
    assert not reconciliation.dispatchable


def test_a_claim_the_queue_still_carries_stays_adopted_though_accounting_reports_no_row(
    scheduled_job: GenericPendingJob,
) -> None:
    """Verifies that a submission the controller queued but accounting has not committed keeps its job adopted."""
    record_batch(batch=_claiming_batch(job_id=scheduled_job.job_id))
    server = _StubServer(statuses={"777": JobStatus.UNRESOLVED}, queued={"777"})

    reconciliation = reconcile_remote_jobs(server=server, jobs=[scheduled_job])

    # Accounting answers the same way for a submission it has not registered yet and for one the scheduler purged.
    # Releasing the claim on that answer alone would run a second allocation over the tracker of one about to start.
    # The queue is the record separating the two, and it holds this allocation.
    assert reconciliation.adopted == {(_UNIT_PATH, scheduled_job.job_id): "777"}
    assert not reconciliation.dispatchable


def test_a_claim_neither_scheduler_record_carries_releases_its_job(scheduled_job: GenericPendingJob) -> None:
    """Verifies that a job is run again once both scheduler records disclaim the allocation the ledger recorded."""
    record_batch(batch=_claiming_batch(job_id=scheduled_job.job_id))
    server = _StubServer(statuses={"777": JobStatus.UNRESOLVED})

    reconciliation = reconcile_remote_jobs(server=server, jobs=[scheduled_job])

    assert server.queried == ["777"]
    assert server.queue_reads == 1
    # Accounting returning no row is on its own no evidence that the scheduler released an allocation, but the
    # queue disclaiming it too is. Nothing then carries the job, and its own tracker never left the scheduled
    # state.
    assert not reconciliation.adopted
    assert [dispatched.job_id for dispatched in reconciliation.dispatchable] == [scheduled_job.job_id]
    assert [resettable.job_id for resettable in reconciliation.resettable] == [scheduled_job.job_id]


def test_a_queue_that_cannot_be_read_leaves_every_claimed_job_adopted(scheduled_job: GenericPendingJob) -> None:
    """Verifies that a queue read which fails is carried rather than raised, and holds every claimed allocation."""
    record_batch(batch=_claiming_batch(job_id=scheduled_job.job_id))
    server = _StubServer(statuses={"777": JobStatus.UNRESOLVED}, queue_error=RuntimeError("squeue: error"))

    reconciliation = reconcile_remote_jobs(server=server, jobs=[scheduled_job])

    # A record that did not answer is no evidence that the scheduler finished with an allocation, so a dispatch
    # made through a failed queue read adopts rather than submits. That reading keeps a scheduler outage from
    # turning a routine dispatch into a second allocation over a live one.
    assert reconciliation.adopted == {(_UNIT_PATH, scheduled_job.job_id): "777"}
    assert not reconciliation.dispatchable


def test_an_allocation_in_a_state_this_stack_cannot_read_is_adopted(scheduled_job: GenericPendingJob) -> None:
    """Verifies that a job whose allocation reports an unmodeled state is adopted rather than submitted again."""
    record_batch(batch=_claiming_batch(job_id=scheduled_job.job_id))
    server = _StubServer(statuses={"777": JobStatus.UNKNOWN})

    reconciliation = reconcile_remote_jobs(server=server, jobs=[scheduled_job])

    # Accounting names live states this library does not model, such as suspended, requeued, or completing, and
    # each reads as unknown here. The row proves the scheduler still holds the allocation, so it is never
    # written off.
    assert reconciliation.adopted == {(_UNIT_PATH, scheduled_job.job_id): "777"}
    assert not reconciliation.dispatchable


def test_a_claim_the_ledger_does_not_hold_is_resolved_against_the_scheduler_all_the_same(
    running_job: Callable[[str | None], GenericPendingJob],
) -> None:
    """Verifies that a tracker-sourced claim the ledger never recorded is queried and resolved like any other."""
    job = running_job("slurm:991")
    server = _StubServer(statuses={"991": JobStatus.UNRESOLVED})

    reconciliation = reconcile_remote_jobs(server=server, jobs=[job])

    assert server.queried == ["991"], "a claim the ledger does not record was never queried"
    # An executor identifier read off a tracker names an allocation another machine submitted, for which this
    # host's ledger holds no entry. It reaches the query all the same, and both scheduler records disclaiming it
    # releases the job.
    assert not reconciliation.adopted
    assert [dispatched.job_id for dispatched in reconciliation.dispatchable] == [job.job_id]
    # The tracker still claims that run, which is the stranded verdict, so this dispatch clears the one claim no
    # rerun could otherwise reach.
    assert [resettable.job_id for resettable in reconciliation.resettable] == [job.job_id]


def test_a_job_withheld_from_the_batch_withholds_the_jobs_that_wait_on_it(
    tmp_path: Path, write_tracker: Callable[..., ProcessingTracker]
) -> None:
    """Verifies that a dependent of a withheld job is withheld too, rather than started against absent input."""
    tracker_path = tmp_path.joinpath("tracker.yaml")
    write_tracker(path=tracker_path, jobs=[_TRACKER_JOB, _SECOND_TRACKER_JOB, _THIRD_TRACKER_JOB])
    withheld = _make_job(
        job_id=_tracker_job_id(), tracker_path=tracker_path, status=ProcessingStatus.RUNNING.name, executor_id="pid:42"
    )
    dependent = _make_job(
        job_id=_tracker_job_id(job=_SECOND_TRACKER_JOB),
        tracker_path=tracker_path,
        specifier=_SECOND_TRACKER_JOB[1],
        prerequisite_ids=(withheld.job_id,),
    )
    unrelated = _make_job(
        job_id=_tracker_job_id(job=_THIRD_TRACKER_JOB), tracker_path=tracker_path, specifier=_THIRD_TRACKER_JOB[1]
    )

    reconciliation = reconcile_remote_jobs(server=_StubServer(statuses={}), jobs=[withheld, dependent, unrelated])

    # A withheld job names no allocation, so a dependent submitted here would wait on nothing and run while the
    # stage it reads is still being produced. An adopted job names one, so its own dependents are submitted and
    # wired to it.
    assert [entry.job_id for entry in reconciliation.withheld] == [withheld.job_id, dependent.job_id]
    assert [entry.job_id for entry in reconciliation.dispatchable] == [unrelated.job_id]
    assert [entry.job_id for entry in reconciliation.resettable] == [unrelated.job_id]


def test_a_dependent_of_an_adopted_job_is_dispatched_and_waits_on_its_allocation(
    tmp_path: Path, write_tracker: Callable[..., ProcessingTracker]
) -> None:
    """Verifies that an adopted job releases its dependents, since it names the allocation they wait on."""
    tracker_path = tmp_path.joinpath("tracker.yaml")
    write_tracker(path=tracker_path, jobs=[_TRACKER_JOB, _SECOND_TRACKER_JOB])
    adopted = _make_job(
        job_id=_tracker_job_id(),
        tracker_path=tracker_path,
        status=ProcessingStatus.RUNNING.name,
        executor_id="slurm:991",
    )
    dependent = _make_job(
        job_id=_tracker_job_id(job=_SECOND_TRACKER_JOB),
        tracker_path=tracker_path,
        specifier=_SECOND_TRACKER_JOB[1],
        prerequisite_ids=(adopted.job_id,),
    )
    server = _StubServer(statuses={"991": JobStatus.RUNNING})

    reconciliation = reconcile_remote_jobs(server=server, jobs=[adopted, dependent])

    assert reconciliation.adopted == {(_UNIT_PATH, adopted.job_id): "991"}
    assert not reconciliation.withheld
    assert [entry.job_id for entry in reconciliation.dispatchable] == [dependent.job_id]


def test_a_recorded_batch_outlives_the_process_that_prepared_it() -> None:
    """Verifies that a recorded batch is readable after the process that prepared it exits."""
    document = BatchDocument(pipeline="video", host="local", jobs=[{"job_id": "a"}], units=[{"unit_name": "one"}])

    batch_id = record_prepared_batch(document=document)
    recovered = read_prepared_batch(batch_id=batch_id)

    assert recovered is not None
    assert recovered.pipeline == "video"
    assert recovered.host == "local"
    assert recovered.jobs == [{"job_id": "a"}]

    assert forget_batch_records(batch_ids=[batch_id]) == [batch_id]
    assert read_prepared_batch(batch_id=batch_id) is None


def test_an_unknown_batch_identifier_is_reported_rather_than_guessed() -> None:
    """Verifies that reading an unknown batch identifier reports it as missing."""
    batch_id = record_prepared_batch(document=BatchDocument(pipeline="video", host="local"))

    found, missing = read_prepared_batches(batch_ids=[batch_id, "absent"])

    assert [document.pipeline for document in found] == ["video"]
    assert missing == ["absent"]


def test_batches_prepared_against_different_hosts_are_not_dispatched_together() -> None:
    """Verifies that batches prepared against different hosts are refused as one dispatch."""
    local = BatchDocument(pipeline="video", host="local")
    remote = BatchDocument(pipeline="video", host="remote")

    # A batch runs where it was prepared, because its jobs read the data that host holds.
    assert resolve_batch_host(documents=[local, local]) == "local"
    with pytest.raises(ValueError, match="prepared against the hosts"):
        resolve_batch_host(documents=[local, remote])


def test_a_job_no_allocation_ever_started_is_submitted(scheduled_job: GenericPendingJob) -> None:
    """Verifies that a job never started by any allocation is submitted without reading either scheduler record."""
    server = _StubServer(statuses={})

    reconciliation = reconcile_remote_jobs(server=server, jobs=[scheduled_job])

    assert server.queried == [], "an unclaimed job sent the scheduler a query"
    assert server.queue_reads == 0, "an unclaimed job sent the scheduler's queue a query"
    assert not reconciliation.adopted
    assert [dispatched.job_id for dispatched in reconciliation.dispatchable] == [scheduled_job.job_id]
    assert [resettable.job_id for resettable in reconciliation.resettable] == [scheduled_job.job_id]
