"""Contains tests for job reconciliation and for the on-disk prepared-batch registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from sollertia_forgery.server import JobStatus
from sollertia_forgery.orchestration import (
    BatchDocument,
    resolve_batch_host,
    read_prepared_batch,
    reconcile_local_jobs,
    read_prepared_batches,
    reconcile_remote_jobs,
    record_prepared_batch,
    forget_prepared_batches,
)
from sollertia_forgery.orchestration.graph import build_pending_job
from sollertia_forgery.orchestration.ledger import (
    SubmissionBatch,
    RemoteSubmission,
    record_batch,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from sollertia_forgery.orchestration import GenericPendingJob

pytestmark = pytest.mark.usefixtures("isolated_working_directory")

_UNIT_PATH: str = "/data/Project/305/2024_11_04"
"""The path to the processing unit the reconciliation tests place their jobs under."""

_TRACKER_JOB: tuple[str, str] = ("motion_energy", "1")
"""The job name and specifier every tracker these tests write is aligned against."""


@pytest.fixture
def running_job(
    tmp_path: Path, write_tracker: Callable[..., ProcessingTracker]
) -> Callable[[str | None], GenericPendingJob]:
    """Returns a builder that writes a tracker recording one job as running and points a pending job at it.

    Args:
        tmp_path: The temporary directory the tracker is written under.
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
        tmp_path: The temporary directory the tracker is written under.
        write_tracker: The builder that writes the tracker.

    Returns:
        The pending job pointing at the written tracker.
    """
    tracker_path = tmp_path.joinpath("tracker.yaml")
    write_tracker(path=tracker_path, jobs=[_TRACKER_JOB])
    return _make_job(job_id=_tracker_job_id(), tracker_path=tracker_path, status=ProcessingStatus.SCHEDULED.name)


class _StubServer:
    """Stands in for a connected server, answering a fixed scheduler state per allocation.

    Args:
        statuses: The scheduler state this server answers for each allocation identifier.

    Attributes:
        _statuses: Cached scheduler states.
        queried: The allocation identifiers of the most recent query.
    """

    def __init__(self, statuses: dict[str, JobStatus]) -> None:
        self._statuses: dict[str, JobStatus] = statuses
        self.queried: list[str] = []

    def get_job_statuses(self, slurm_job_ids: list[str]) -> dict[str, JobStatus]:
        """Records the queried allocations and answers the fixed state of each.

        Args:
            slurm_job_ids: The allocation identifiers the caller asked about.

        Returns:
            The state of each queried allocation this server holds one for.
        """
        self.queried = list(slurm_job_ids)
        return {allocation: self._statuses[allocation] for allocation in slurm_job_ids if allocation in self._statuses}


def _tracker_job_id() -> str:
    """Returns the identifier the tracker registers the shared job name and specifier under."""
    return ProcessingTracker.generate_job_id(job_name=_TRACKER_JOB[0], specifier=_TRACKER_JOB[1])


def _make_job(
    job_id: str,
    tracker_path: Path,
    unit_path: str = _UNIT_PATH,
    status: str = "",
    executor_id: str = "",
) -> GenericPendingJob:
    """Builds a pending job carrying the outcome preparation read out of the host's state artifact.

    Args:
        job_id: The identifier the job is registered under in the tracker.
        tracker_path: The path to the tracker the job is recorded on.
        unit_path: The processing unit the job runs against.
        status: The status the state artifact recorded for the job.
        executor_id: The executor the same record named.

    Returns:
        The pending job.
    """
    return build_pending_job(
        job={
            "job_id": job_id,
            "job_name": _TRACKER_JOB[0],
            "specifier": _TRACKER_JOB[1],
            "unit_path": unit_path,
            "unit_name": "2024_11_04",
            "pipeline": "video",
            "tracker_path": str(tracker_path),
            "cores": 4,
            "memory_mb": 1024,
            "prerequisite_ids": [],
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
    """Verifies that a job both sources claim is adopted onto the allocation its tracker names, not the ledger's.

    An executor identifier appears only once an allocation starts running, so it describes a later moment than this
    host's record of submitting one. Preferring the ledger's stale identifier would query an allocation that has
    already finished, read the job as finished with it, and submit a second allocation over the files the live one is
    still writing.
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

    assert reconciliation.adopted == {(_UNIT_PATH, job.job_id): "991"}
    # Dispatching the job again would run two allocations over the same tracker and the same output.
    assert not reconciliation.dispatchable
    assert not reconciliation.resettable


def test_an_executor_naming_the_scheduler_but_no_allocation_is_submitted_again(
    running_job: Callable[[str | None], GenericPendingJob],
) -> None:
    """Verifies that a truncated executor identifier is treated as naming no allocation at all.

    A record carrying the scheme without an allocation names nothing the scheduler can be asked about. Treating it as
    a claim would adopt the job forever, since an allocation the scheduler reports nothing for reads as one that has
    not yet reached a terminal state.
    """
    job = running_job("slurm:")
    server = _StubServer(statuses={})

    reconciliation = reconcile_remote_jobs(server=server, jobs=[job])

    assert server.queried == [], "a job claiming no allocation sent the scheduler a query"
    assert not reconciliation.adopted
    assert [dispatched.job_id for dispatched in reconciliation.dispatchable] == [job.job_id]
    assert [resettable.job_id for resettable in reconciliation.resettable] == [job.job_id]


def test_a_finished_allocation_is_submitted_again(running_job: Callable[[str | None], GenericPendingJob]) -> None:
    """Verifies that a job whose tracker names a finished allocation is submitted again."""
    job = running_job("slurm:991")
    server = _StubServer(statuses={"991": JobStatus.FAILED})

    reconciliation = reconcile_remote_jobs(server=server, jobs=[job])

    # A record left running by an allocation that has since finished describes work that never completed.
    assert not reconciliation.adopted
    assert [dispatched.job_id for dispatched in reconciliation.dispatchable] == [job.job_id]


def test_an_off_scheduler_executor_is_treated_as_dead(running_job: Callable[[str | None], GenericPendingJob]) -> None:
    """Verifies that a job whose tracker names a bare process identifier is submitted again."""
    job = running_job("pid:4242")
    server = _StubServer(statuses={})

    reconciliation = reconcile_remote_jobs(server=server, jobs=[job])

    # A bare process identifier was never a scheduler allocation, so there is nothing to adopt.
    assert not reconciliation.adopted
    assert [dispatched.job_id for dispatched in reconciliation.dispatchable] == [job.job_id]


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


def test_a_recorded_batch_outlives_the_process_that_prepared_it() -> None:
    """Verifies that a recorded batch is readable after the process that prepared it exits."""
    document = BatchDocument(pipeline="video", host="local", jobs=[{"job_id": "a"}], units=[{"unit_name": "one"}])

    batch_id = record_prepared_batch(document=document)
    recovered = read_prepared_batch(batch_id=batch_id)

    assert recovered is not None
    assert recovered.pipeline == "video"
    assert recovered.host == "local"
    assert recovered.jobs == [{"job_id": "a"}]

    assert forget_prepared_batches(batch_ids=[batch_id]) == [batch_id]
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
    """Verifies that a job no allocation ever started is submitted without a scheduler query."""
    server = _StubServer(statuses={})

    reconciliation = reconcile_remote_jobs(server=server, jobs=[scheduled_job])

    assert server.queried == [], "an unclaimed job sent the scheduler a query"
    assert not reconciliation.adopted
    assert [dispatched.job_id for dispatched in reconciliation.dispatchable] == [scheduled_job.job_id]
    assert [resettable.job_id for resettable in reconciliation.resettable] == [scheduled_job.job_id]
