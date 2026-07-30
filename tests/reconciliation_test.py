"""Tests the reconciliation of jobs the trackers already record as running, and the on-disk prepared-batch registry.

A re-submitted batch must not run one job twice, and it must not reset a record an allocation is still writing. These
tests pin which source a claim is read from, when an allocation is adopted rather than submitted again, and that a
batch identifier outlives the process that issued it.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

import pytest
import platformdirs
from sollertia_shared_assets import set_working_directory
from ataraxis_data_structures import ProcessingTracker

from sollertia_forgery.server import JobStatus
from sollertia_forgery.orchestration import (
    BatchDocument,
    read_prepared_batch,
    resolve_batch_host,
    read_prepared_batches,
    record_prepared_batch,
    reconcile_local_jobs,
    reconcile_remote_jobs,
    forget_prepared_batches,
)
from sollertia_forgery.orchestration.graph import build_pending_job
from sollertia_forgery.orchestration.ledger import (
    SubmissionBatch,
    RemoteSubmission,
    record_batch,
)

UNIT = "/data/Project/305/2024_11_04"
"""The processing unit the reconciliation tests place their jobs under."""


@pytest.fixture(autouse=True)
def isolated_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Points the Sollertia platform working directory at a pristine tmp-backed location.

    The ledger and the prepared-batch registry are both written under that directory, so every test in this module
    records into its own files rather than into whatever this host already holds.
    """
    working = tmp_path.joinpath("working")
    monkeypatch.setattr(platformdirs, "user_data_dir", lambda *_args, **_kwargs: str(tmp_path.joinpath("platform")))
    set_working_directory(path=working)
    return working


class StubServer:
    """Stands in for a connected server, answering a fixed scheduler state per allocation."""

    def __init__(self, statuses: dict[str, JobStatus]) -> None:
        self._statuses: dict[str, JobStatus] = statuses
        self.queried: list[str] = []

    def get_job_statuses(self, slurm_job_ids: list[str]) -> dict[str, JobStatus]:
        """Records the queried allocations and answers the fixed state of each."""
        self.queried = list(slurm_job_ids)
        return {allocation: self._statuses[allocation] for allocation in slurm_job_ids if allocation in self._statuses}


def make_job(job_id: str, tracker_path: Path, unit_path: str = UNIT) -> Any:  # noqa: ANN401
    """Builds a pending job pointing at a real tracker file."""
    return build_pending_job(
        job={
            "job_id": job_id,
            "job_name": "motion_energy",
            "specifier": "1",
            "unit_path": unit_path,
            "unit_name": "2024_11_04",
            "pipeline": "video",
            "tracker_path": str(tracker_path),
            "cores": 4,
            "memory_mb": 1024,
            "prerequisite_ids": [],
            "options": {},
        }
    )


def write_tracker(path: Path, job_name: str, specifier: str, executor_id: str | None) -> str:
    """Writes a tracker holding one job left in the running state, and returns its identifier."""
    tracker = ProcessingTracker(file_path=path)
    tracker.align_jobs(jobs=[(job_name, specifier)], universe=[(job_name, specifier)])
    job_id = ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier)
    tracker.start_job(job_id=job_id, executor_id=executor_id)
    return job_id


def test_a_local_batch_adopts_nothing_and_reruns_every_job(tmp_path: Path) -> None:
    """Nothing else on this machine should hold a job, so a running record describes a pool that died."""
    tracker_path = tmp_path.joinpath("tracker.yaml")
    job_id = write_tracker(path=tracker_path, job_name="motion_energy", specifier="1", executor_id="pid:4242")
    job = make_job(job_id=job_id, tracker_path=tracker_path)

    reconciliation = reconcile_local_jobs(jobs=[job])

    assert not reconciliation.adopted
    assert [dispatched.job_id for dispatched in reconciliation.dispatchable] == [job_id]
    assert [resettable.job_id for resettable in reconciliation.resettable] == [job_id]


def test_a_live_allocation_named_by_the_tracker_is_adopted(tmp_path: Path) -> None:
    """An executor identifier travels with the data, so it covers an allocation submitted from another machine."""
    tracker_path = tmp_path.joinpath("tracker.yaml")
    job_id = write_tracker(path=tracker_path, job_name="motion_energy", specifier="1", executor_id="slurm:991")
    job = make_job(job_id=job_id, tracker_path=tracker_path)
    server = StubServer(statuses={"991": JobStatus.RUNNING})

    reconciliation = reconcile_remote_jobs(server=server, jobs=[job])  # type: ignore[arg-type]

    assert reconciliation.adopted == {(UNIT, job_id): "991"}
    assert not reconciliation.dispatchable
    # An adopted job is never reset, since that would wipe a record its allocation is still writing.
    assert not reconciliation.resettable


def test_a_finished_allocation_is_submitted_again(tmp_path: Path) -> None:
    """A record left running by an allocation that has since finished describes work that never completed."""
    tracker_path = tmp_path.joinpath("tracker.yaml")
    job_id = write_tracker(path=tracker_path, job_name="motion_energy", specifier="1", executor_id="slurm:991")
    job = make_job(job_id=job_id, tracker_path=tracker_path)
    server = StubServer(statuses={"991": JobStatus.FAILED})

    reconciliation = reconcile_remote_jobs(server=server, jobs=[job])  # type: ignore[arg-type]

    assert not reconciliation.adopted
    assert [dispatched.job_id for dispatched in reconciliation.dispatchable] == [job_id]


def test_an_off_scheduler_executor_is_treated_as_dead(tmp_path: Path) -> None:
    """A record naming a bare process identifier was never a scheduler allocation, so nothing can be adopted."""
    tracker_path = tmp_path.joinpath("tracker.yaml")
    job_id = write_tracker(path=tracker_path, job_name="motion_energy", specifier="1", executor_id="pid:4242")
    job = make_job(job_id=job_id, tracker_path=tracker_path)
    server = StubServer(statuses={})

    reconciliation = reconcile_remote_jobs(server=server, jobs=[job])  # type: ignore[arg-type]

    assert not reconciliation.adopted
    assert [dispatched.job_id for dispatched in reconciliation.dispatchable] == [job_id]


def test_the_ledger_claims_an_allocation_the_tracker_cannot_yet_name(tmp_path: Path) -> None:
    """A queued allocation has not started, so only the ledger knows it exists."""
    tracker_path = tmp_path.joinpath("tracker.yaml")
    tracker = ProcessingTracker(file_path=tracker_path)
    tracker.align_jobs(jobs=[("motion_energy", "1")], universe=[("motion_energy", "1")])
    job_id = ProcessingTracker.generate_job_id(job_name="motion_energy", specifier="1")
    job = make_job(job_id=job_id, tracker_path=tracker_path)

    record_batch(
        batch=SubmissionBatch(
            batch_id="earlier",
            submissions=[RemoteSubmission(job_id=job_id, slurm_job_id="777", unit_path=UNIT, pipeline="video")],
        )
    )
    server = StubServer(statuses={"777": JobStatus.PENDING})

    reconciliation = reconcile_remote_jobs(server=server, jobs=[job])  # type: ignore[arg-type]

    assert reconciliation.adopted == {(UNIT, job_id): "777"}
    assert not reconciliation.dispatchable


def test_a_recorded_batch_outlives_the_process_that_prepared_it() -> None:
    """Batch identifiers are recorded on disk, so preparing and executing need not share one server lifetime."""
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
    """A caller naming a batch this host does not hold is told which identifiers were not found."""
    batch_id = record_prepared_batch(document=BatchDocument(pipeline="video", host="local"))

    found, missing = read_prepared_batches(batch_ids=[batch_id, "absent"])

    assert [document.pipeline for document in found] == ["video"]
    assert missing == ["absent"]


def test_batches_prepared_against_different_hosts_are_not_dispatched_together() -> None:
    """A batch runs where it was prepared, because its jobs read the data that host holds."""
    local = BatchDocument(pipeline="video", host="local")
    remote = BatchDocument(pipeline="video", host="remote")

    assert resolve_batch_host(documents=[local, local]) == "local"
    with pytest.raises(ValueError, match="prepared against the hosts"):
        resolve_batch_host(documents=[local, remote])
