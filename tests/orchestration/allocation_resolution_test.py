"""Contains tests for the deterministic resolution of a remote allocation: what the scheduler's two records place it
in, what the tracker of the job it carries holds, and the verdict and remediation the three of them carry together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path
from dataclasses import field, dataclass

import pytest

from sollertia_forgery.server import JobStatus
from sollertia_forgery.forging import DATASET_STATE_FILENAME
from sollertia_forgery.managing import project_jobs_path
from sollertia_forgery.orchestration import (
    STALLED_BATCH,
    NO_REMEDIATION,
    GONE_ALLOCATION,
    DROP_REMEDIATION,
    RESET_REMEDIATION,
    RUNNING_ALLOCATION,
    STRANDED_ALLOCATION,
    AWAITING_CLOSURE_BATCH,
    SchedulerReading,
    classify_batch,
    render_allocation,
    reset_stranded_jobs,
    resolve_allocations,
    read_scheduler_records,
    resolve_tracker_claims,
    resolve_live_allocations,
    resolve_queried_allocations,
)
from sollertia_forgery.orchestration.ledger import SubmissionBatch, RemoteSubmission
from sollertia_forgery.orchestration.remote import (
    HELD_ALLOCATION,
    FAILED_ALLOCATION,
    PROGRESSING_BATCH,
    SETTLED_ALLOCATION,
    FINISHED_ALLOCATION,
    ABANDONED_ALLOCATION,
    TrackerClaim,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_PROJECT_ROOT: Path = Path("/data/TestProject")
"""The project root under which every unit these tests resolve sits."""

_SESSION_PATH: Path = _PROJECT_ROOT.joinpath("305", "2024_11_04")
"""The session unit these tests place their per-session jobs under."""

_DATASET_PATH: Path = _PROJECT_ROOT.joinpath("Dataset")
"""The dataset unit these tests place their forging jobs under."""

_JOBS_ARTIFACT: Path = project_jobs_path(project_directory=_PROJECT_ROOT)
"""The project job artifact from which a session pipeline's recorded state is read."""

_DATASET_ARTIFACT: Path = _DATASET_PATH.joinpath(DATASET_STATE_FILENAME)
"""The dataset state artifact from which the forging pipeline's recorded state is read."""

_HOST_FAILURE: RuntimeError = RuntimeError("the state table could not be regenerated")
"""The failure the stubbed host raises for a project it cannot answer for."""


@dataclass
class _StubHost:
    """Stands in for the host that holds the units, answering the state rows a test places on it."""

    rows: dict[Path, list[dict[str, Any]]] = field(default_factory=dict)
    """The rows each state artifact holds, keyed by the artifact's path."""
    generated: list[tuple[str, str, list[str]]] = field(default_factory=list)
    """The unit kind, project root, and units of every regeneration the host was asked for."""
    reset_calls: list[tuple[str, dict[str, list[str]]]] = field(default_factory=list)
    """The pipeline and per-unit identifiers of every reset the host was asked for."""
    generate_error: Exception | None = None
    """The failure the regeneration raises, or None when it succeeds."""
    reset_error: Exception | None = None
    """The failure the reset raises, or None when it succeeds."""

    def generate_state(self, project_root: Path, unit_paths: Sequence[Path], unit_kind: str) -> None:
        """Records one regeneration request."""
        if self.generate_error is not None:
            raise self.generate_error
        self.generated.append((unit_kind, str(project_root), [str(unit) for unit in unit_paths]))

    def read_rows(self, path: Path) -> list[dict[str, Any]]:
        """Answers the rows a test placed at one artifact path."""
        return self.rows.get(path, [])

    def reset_jobs(self, pipeline: str, job_ids_by_unit: Mapping[Path, Sequence[str]]) -> None:
        """Records one reset request."""
        if self.reset_error is not None:
            raise self.reset_error
        self.reset_calls.append(
            (pipeline, {str(unit): list(identifiers) for unit, identifiers in job_ids_by_unit.items()})
        )


@dataclass
class _StubServer:
    """Stands in for the connected server, answering the two scheduler records a test places on it."""

    statuses: dict[str, JobStatus] = field(default_factory=dict)
    """The state accounting reports for each allocation identifier."""
    queued: set[str] = field(default_factory=set)
    """The allocation identifiers the queue holds."""
    queue_error: Exception | None = None
    """The failure the queue read raises, or None when the queue answers."""
    queried: list[list[str]] = field(default_factory=list)
    """The identifiers of every accounting query the server was asked for."""

    def get_job_statuses(self, slurm_job_ids: Sequence[str]) -> dict[str, JobStatus]:
        """Answers the accounting state of every requested allocation, seeding an unreported one as unresolved."""
        self.queried.append(list(slurm_job_ids))
        return {job_id: self.statuses.get(job_id, JobStatus.UNRESOLVED) for job_id in slurm_job_ids}

    def get_queued_job_ids(self) -> set[str]:
        """Answers the identifiers the queue holds."""
        if self.queue_error is not None:
            raise self.queue_error
        return set(self.queued)


def _build_submission(
    job_id: str = "job0",
    slurm_job_id: str = "1000",
    unit_path: str = str(_SESSION_PATH),
    pipeline: str = "video",
    job_name: str = "motion_energy",
    specifier: str = "1",
) -> RemoteSubmission:
    """Builds one recorded submission."""
    return RemoteSubmission(
        job_id=job_id,
        slurm_job_id=slurm_job_id,
        slurm_job_name=f"0000-{job_id}",
        pipeline=pipeline,
        job_name=job_name,
        specifier=specifier,
        unit_path=unit_path,
        unit_name=Path(unit_path).name if unit_path else "",
        cores=4,
        memory_mb=2048,
    )


def _build_batch(batch_id: str = "batch01", submissions: Sequence[RemoteSubmission] = ()) -> SubmissionBatch:
    """Builds one recorded batch holding the given submissions."""
    return SubmissionBatch(batch_id=batch_id, batch_ids=[batch_id], submissions=list(submissions))


def _build_row(
    job_id: str = "job0",
    status: str = "SCHEDULED",
    executor_id: str | None = None,
    pipeline: str | None = "video",
    unit_column: str = "session",
    unit_name: str = "2024_11_04",
) -> dict[str, Any]:
    """Builds one row of the shape a state artifact holds."""
    row: dict[str, Any] = {
        unit_column: unit_name,
        "job_id": job_id,
        "job_name": "motion_energy",
        "specifier": "1",
        "status": status,
        "executor_id": executor_id,
    }
    if pipeline is not None:
        row["pipeline"] = pipeline
    return row


# Tests for the state the scheduler's two records place an allocation in


def test_a_record_naming_no_allocation_holds_nothing() -> None:
    """Verifies that an entry carrying no allocation identifier resolves as gone whatever the records say."""
    assert SchedulerReading(unreadable_reason="the queue is down").resolve_state(allocation="") == GONE_ALLOCATION


def test_an_allocation_this_process_cancelled_is_the_schedulers_no_longer() -> None:
    """Verifies that a cancellation this process issued settles the allocation without another query."""
    reading = SchedulerReading(statuses={"1000": JobStatus.RUNNING}, queued=frozenset({"1000"}))

    assert reading.resolve_state(allocation="1000") == HELD_ALLOCATION
    assert reading.cancelling(allocations=["1000"]).resolve_state(allocation="1000") == SETTLED_ALLOCATION


def test_a_record_that_could_not_be_read_holds_every_allocation() -> None:
    """Verifies that an unread record is no evidence of absence, so nothing resolves as gone through it."""
    reading = SchedulerReading(statuses={"1000": JobStatus.UNRESOLVED}, unreadable_reason="the queue is down")

    assert reading.resolve_state(allocation="1000") == HELD_ALLOCATION


def test_an_allocation_the_queue_holds_is_held_though_accounting_reports_no_row() -> None:
    """Verifies that a submission the controller has queued but slurmdbd has not committed reads as held."""
    reading = SchedulerReading(statuses={"1000": JobStatus.UNRESOLVED}, queued=frozenset({"1000"}))

    # This is the allocation that accounting alone cannot see. Reporting it as gone would retire a batch whose work
    # is about to start, and a rerun would then submit a second allocation over the same tracker.
    assert reading.resolve_state(allocation="1000") == HELD_ALLOCATION


def test_a_blocked_allocation_settles_whichever_way_the_two_snapshots_fall() -> None:
    """Verifies that a permanently blocked allocation resolves the same way on every read of one unchanged batch."""
    carried = SchedulerReading(statuses={"1000": JobStatus.BLOCKED}, queued=frozenset({"1000"}))
    released = SchedulerReading(statuses={"1000": JobStatus.BLOCKED})

    # The state is derived from the queue's own reason field while the queue membership is a second, later read, so
    # the two disagree exactly when the scheduler releases such an allocation between them. The blocked state alone
    # therefore decides the outcome, because a dependency that can never be satisfied means the allocation never runs
    # and never changes what it holds.
    assert carried.resolve_state(allocation="1000") == SETTLED_ALLOCATION
    assert released.resolve_state(allocation="1000") == SETTLED_ALLOCATION


def test_a_blocked_allocation_settles_though_the_queue_could_not_be_read() -> None:
    """Verifies that the blocked state is positive evidence rather than the absence a failed read would leave."""
    reading = SchedulerReading(statuses={"1000": JobStatus.BLOCKED}, unreadable_reason="the queue is down")

    # An unread record holds every allocation on which it is silent, but a blocked state is something accounting's
    # own read reported, so it settles the allocation rather than leaving it unresolved.
    assert reading.resolve_state(allocation="1000") == SETTLED_ALLOCATION


def test_an_allocation_neither_record_carries_is_gone() -> None:
    """Verifies that both records have to disclaim an allocation before it resolves as gone."""
    reading = SchedulerReading(statuses={"1000": JobStatus.UNRESOLVED}, queued=frozenset({"1001"}))

    assert reading.resolve_state(allocation="1000") == GONE_ALLOCATION


def test_a_settled_allocation_the_queue_no_longer_carries_is_settled() -> None:
    """Verifies that a terminal accounting row the queue does not contradict settles the allocation."""
    reading = SchedulerReading(statuses={"1000": JobStatus.COMPLETED})

    assert reading.resolve_state(allocation="1000") == SETTLED_ALLOCATION


def test_a_state_this_library_cannot_read_still_holds_its_allocation() -> None:
    """Verifies that an unmodeled accounting state is held, since the row proves the scheduler carries it."""
    reading = SchedulerReading(statuses={"1000": JobStatus.UNKNOWN})

    assert reading.resolve_state(allocation="1000") == HELD_ALLOCATION
    assert reading.resolve_status(allocation="1000") is JobStatus.UNKNOWN
    assert reading.resolve_status(allocation="9999") is JobStatus.UNRESOLVED


# Tests for reading the two records


def test_reading_the_scheduler_records_both_of_them() -> None:
    """Verifies that one reading carries accounting's answer alongside the queue's."""
    server = _StubServer(statuses={"1000": JobStatus.RUNNING}, queued={"1000", "1001"})

    reading = read_scheduler_records(server=server, allocations=["1000"])

    assert reading.statuses == {"1000": JobStatus.RUNNING}
    assert reading.queued == frozenset({"1000", "1001"})
    assert reading.unreadable_reason == ""


def test_a_failed_queue_read_is_carried_rather_than_raised() -> None:
    """Verifies that a queue that cannot answer leaves a usable reading in which nothing resolves as gone."""
    server = _StubServer(statuses={"1000": JobStatus.UNRESOLVED}, queue_error=RuntimeError("squeue: error"))

    reading = read_scheduler_records(server=server, allocations=["1000"])

    assert "squeue: error" in reading.unreadable_reason
    assert reading.statuses == {"1000": JobStatus.UNRESOLVED}
    assert reading.resolve_state(allocation="1000") == HELD_ALLOCATION


def test_a_query_covers_the_allocation_a_tracker_claims_alongside_the_recorded_one() -> None:
    """Verifies that a job claimed by another machine's allocation is queried too, and that empties are dropped."""
    submissions = [_build_submission(slurm_job_id="1000"), _build_submission(job_id="job1", slurm_job_id="")]
    claims = {
        (str(_SESSION_PATH), "job0"): TrackerClaim(status="RUNNING", executor_id="slurm:2000", allocation="2000"),
        (str(_SESSION_PATH), "job1"): TrackerClaim(),
    }

    assert resolve_queried_allocations(submissions=submissions, claims=claims) == ["1000", "2000"]


# Tests for reading what each job's own tracker holds


def test_a_tracker_claim_is_read_out_of_freshly_regenerated_state() -> None:
    """Verifies that the artifacts are rewritten before they are read, so a finished job never reads as running."""
    host = _StubHost(rows={_JOBS_ARTIFACT: [_build_row(status="RUNNING", executor_id="slurm:1000")]})

    claims = resolve_tracker_claims(host=host, submissions=[_build_submission()])

    assert host.generated == [("session", str(_PROJECT_ROOT), [str(_SESSION_PATH)])]
    assert claims[(str(_SESSION_PATH), "job0")] == TrackerClaim(
        status="RUNNING", executor_id="slurm:1000", allocation="1000"
    )


def test_a_job_the_state_artifact_holds_no_row_for_carries_an_empty_claim() -> None:
    """Verifies that a tracker that lost an entry claims nothing rather than claiming the row of another job."""
    host = _StubHost(rows={_JOBS_ARTIFACT: [_build_row(job_id="other", status="RUNNING")]})

    claims = resolve_tracker_claims(host=host, submissions=[_build_submission()])

    assert claims[(str(_SESSION_PATH), "job0")] == TrackerClaim()


def test_a_submission_naming_no_unit_reads_no_tracker() -> None:
    """Verifies that a record that does not say which unit it ran against is skipped rather than resolved."""
    host = _StubHost()

    claims = resolve_tracker_claims(host=host, submissions=[_build_submission(unit_path="")])

    assert host.generated == []
    assert claims[("", "job0")] == TrackerClaim()


def test_an_executor_that_is_not_an_allocation_claims_none() -> None:
    """Verifies that a process identifier is read as claiming no allocation, so nothing is queried for it."""
    host = _StubHost(rows={_JOBS_ARTIFACT: [_build_row(status="RUNNING", executor_id="pid:4242")]})

    claims = resolve_tracker_claims(host=host, submissions=[_build_submission()])

    assert claims[(str(_SESSION_PATH), "job0")].allocation == ""


def test_one_project_is_regenerated_once_for_every_pipeline_it_holds() -> None:
    """Verifies that two pipelines of one project cost one regeneration and are narrowed apart afterwards."""
    host = _StubHost(
        rows={
            _JOBS_ARTIFACT: [
                _build_row(job_id="job0", status="RUNNING", pipeline="video"),
                _build_row(job_id="job1", status="FAILED", pipeline="checksum"),
            ]
        }
    )
    submissions = [
        _build_submission(job_id="job0", pipeline="video"),
        _build_submission(job_id="job1", slurm_job_id="1001", pipeline="checksum"),
    ]

    claims = resolve_tracker_claims(host=host, submissions=submissions)

    assert len(host.generated) == 1
    assert claims[(str(_SESSION_PATH), "job0")].status == "RUNNING"
    assert claims[(str(_SESSION_PATH), "job1")].status == "FAILED"


def test_a_forging_batch_reads_each_datasets_own_state_artifact() -> None:
    """Verifies that a dataset unit resolves its project one level up and reads the artifact beside the dataset."""
    host = _StubHost(
        rows={
            _DATASET_ARTIFACT: [
                _build_row(status="SUCCEEDED", pipeline=None, unit_column="dataset", unit_name="Dataset")
            ]
        }
    )
    submission = _build_submission(unit_path=str(_DATASET_PATH), pipeline="forging")

    claims = resolve_tracker_claims(host=host, submissions=[submission])

    assert host.generated == [("dataset", str(_PROJECT_ROOT), [str(_DATASET_PATH)])]
    assert claims[(str(_DATASET_PATH), "job0")].status == "SUCCEEDED"


def test_a_host_that_cannot_regenerate_its_state_raises() -> None:
    """Verifies that an unreadable tracker state is reported rather than resolved as an absent record."""
    host = _StubHost(generate_error=_HOST_FAILURE)

    with pytest.raises(RuntimeError, match=r"could not be regenerated"):
        resolve_tracker_claims(host=host, submissions=[_build_submission()])


# Tests for the verdict the two states carry together


@pytest.mark.parametrize(
    ("status", "tracker_status", "verdict", "remediation"),
    [
        (JobStatus.RUNNING, "RUNNING", RUNNING_ALLOCATION, NO_REMEDIATION),
        (JobStatus.PENDING, "SCHEDULED", RUNNING_ALLOCATION, NO_REMEDIATION),
        (JobStatus.COMPLETED, "SUCCEEDED", FINISHED_ALLOCATION, DROP_REMEDIATION),
        (JobStatus.FAILED, "FAILED", FAILED_ALLOCATION, DROP_REMEDIATION),
        (JobStatus.UNRESOLVED, "SCHEDULED", ABANDONED_ALLOCATION, DROP_REMEDIATION),
        (JobStatus.UNRESOLVED, "", ABANDONED_ALLOCATION, DROP_REMEDIATION),
        (JobStatus.UNRESOLVED, "RUNNING", STRANDED_ALLOCATION, RESET_REMEDIATION),
        (JobStatus.COMPLETED, "RUNNING", STRANDED_ALLOCATION, RESET_REMEDIATION),
    ],
)
def test_the_state_table_resolves_every_pairing(
    status: JobStatus, tracker_status: str, verdict: str, remediation: str
) -> None:
    """Verifies that each pairing of a scheduler state and a tracker status resolves to its row of the table."""
    reading = SchedulerReading(statuses={"1000": status})
    claims = {(str(_SESSION_PATH), "job0"): TrackerClaim(status=tracker_status)}

    resolved = resolve_allocations(
        batches=[_build_batch(submissions=[_build_submission()])], reading=reading, claims=claims
    )

    assert [entry.verdict for entry in resolved] == [verdict]
    assert [entry.remediation for entry in resolved] == [remediation]


def test_a_job_a_live_allocation_claims_is_running_though_its_own_allocation_is_gone() -> None:
    """Verifies that a job another machine is still running is never resolved as stranded."""
    reading = SchedulerReading(statuses={"1000": JobStatus.UNRESOLVED, "2000": JobStatus.RUNNING})
    claims = {(str(_SESSION_PATH), "job0"): TrackerClaim(status="RUNNING", executor_id="slurm:2000", allocation="2000")}

    resolved = resolve_allocations(
        batches=[_build_batch(submissions=[_build_submission()])], reading=reading, claims=claims
    )

    # Its own recorded allocation is gone and its tracker says running, which is the stranded pairing exactly. The
    # allocation the tracker claims is held, though, and resetting that job would destroy work that is still live.
    assert resolved[0].scheduler_state == GONE_ALLOCATION
    assert resolved[0].claim_state == HELD_ALLOCATION
    assert resolved[0].verdict == RUNNING_ALLOCATION


def test_a_job_whose_claiming_allocation_is_also_gone_is_stranded() -> None:
    """Verifies that a claim both records disclaim leaves the job stranded rather than protected."""
    reading = SchedulerReading(statuses={"1000": JobStatus.UNRESOLVED, "2000": JobStatus.UNRESOLVED})
    claims = {(str(_SESSION_PATH), "job0"): TrackerClaim(status="RUNNING", executor_id="slurm:2000", allocation="2000")}

    resolved = resolve_allocations(
        batches=[_build_batch(submissions=[_build_submission()])], reading=reading, claims=claims
    )

    assert resolved[0].claim_state == GONE_ALLOCATION
    assert resolved[0].verdict == STRANDED_ALLOCATION


def test_an_allocation_with_no_tracker_claim_resolves_without_one() -> None:
    """Verifies that a job the claims mapping does not cover is resolved as claiming nothing."""
    reading = SchedulerReading(statuses={"1000": JobStatus.UNRESOLVED})

    resolved = resolve_allocations(
        batches=[_build_batch(submissions=[_build_submission()])], reading=reading, claims={}
    )

    assert resolved[0].claim_state == ""
    assert resolved[0].verdict == ABANDONED_ALLOCATION


# Tests for the batch verdict the allocation verdicts carry


def test_a_batch_holding_a_running_allocation_is_progressing() -> None:
    """Verifies that one running allocation makes a batch progressing, whatever the others resolve to."""
    reading = SchedulerReading(statuses={"1000": JobStatus.RUNNING, "1001": JobStatus.UNRESOLVED})
    batch = _build_batch(submissions=[_build_submission(), _build_submission(job_id="job1", slurm_job_id="1001")])

    assert classify_batch(resolutions=resolve_allocations(batches=[batch], reading=reading, claims={})) == (
        PROGRESSING_BATCH
    )


def test_a_batch_holding_a_gone_allocation_is_stalled() -> None:
    """Verifies that an allocation both records disclaim stalls its batch, since no query moves it again."""
    reading = SchedulerReading(statuses={"1000": JobStatus.COMPLETED, "1001": JobStatus.UNRESOLVED})
    batch = _build_batch(submissions=[_build_submission(), _build_submission(job_id="job1", slurm_job_id="1001")])

    assert classify_batch(resolutions=resolve_allocations(batches=[batch], reading=reading, claims={})) == STALLED_BATCH


def test_a_batch_whose_allocations_have_all_settled_awaits_closure() -> None:
    """Verifies that a settled batch still in the ledger is one whose closure failed rather than a stalled one."""
    reading = SchedulerReading(statuses={"1000": JobStatus.COMPLETED})
    batch = _build_batch(submissions=[_build_submission()])

    assert classify_batch(resolutions=resolve_allocations(batches=[batch], reading=reading, claims={})) == (
        AWAITING_CLOSURE_BATCH
    )


def test_a_batch_holding_no_allocation_awaits_closure() -> None:
    """Verifies that a record with nothing left for the scheduler to advance is never reported as progressing."""
    assert classify_batch(resolutions=[]) == AWAITING_CLOSURE_BATCH


# Tests for the executor a verdict is unable to query


def test_a_job_running_under_an_executor_outside_the_scheduler_is_refused_rather_than_stranded() -> None:
    """Verifies that a tracker claiming to run under a process rather than an allocation is never called stranded."""
    reading = SchedulerReading(statuses={"1000": JobStatus.COMPLETED})
    claims = {(str(_SESSION_PATH), "job0"): TrackerClaim(status="RUNNING", executor_id="pid:4821")}
    batch = _build_batch(submissions=[_build_submission()])

    resolved = resolve_allocations(batches=[batch], reading=reading, claims=claims)[0]

    # Only the scheduler's own scheme names an allocation, so a local run's process identifier resolves to no claim
    # and neither scheduler record answers for it. Calling such a job stranded would reset a tracker whose own
    # executor may still be writing to it.
    assert resolved.scheduler_state == SETTLED_ALLOCATION
    assert resolved.claim_state == ""
    assert resolved.verdict == RUNNING_ALLOCATION
    assert resolved.remediation == NO_REMEDIATION


def test_a_job_running_under_no_recorded_executor_is_still_stranded() -> None:
    """Verifies that the refusal covers a named executor alone, so a tracker naming none stays releasable."""
    reading = SchedulerReading(statuses={"1000": JobStatus.COMPLETED})
    claims = {(str(_SESSION_PATH), "job0"): TrackerClaim(status="RUNNING")}
    batch = _build_batch(submissions=[_build_submission()])

    resolved = resolve_allocations(batches=[batch], reading=reading, claims=claims)[0]

    # A tracker that recorded no executor names nothing that could still be running, which is exactly the claim the
    # stranded verdict exists to clear.
    assert resolved.verdict == STRANDED_ALLOCATION
    assert resolved.remediation == RESET_REMEDIATION


def test_a_job_that_recorded_an_outcome_outside_the_scheduler_keeps_its_verdict() -> None:
    """Verifies that the refusal covers a running claim alone, so a recorded outcome still resolves as one."""
    reading = SchedulerReading(statuses={"1000": JobStatus.COMPLETED})
    claims = {(str(_SESSION_PATH), "job0"): TrackerClaim(status="SUCCEEDED", executor_id="pid:4821")}
    batch = _build_batch(submissions=[_build_submission()])

    resolved = resolve_allocations(batches=[batch], reading=reading, claims=claims)[0]

    assert resolved.verdict == FINISHED_ALLOCATION
    assert resolved.remediation == DROP_REMEDIATION


# Tests for the allocations a cancellation names


def test_every_allocation_a_running_resolution_leaves_held_is_named() -> None:
    """Verifies that both allocations of a resolution are named, so a cancellation reaches the one carrying the job."""
    reading = SchedulerReading(statuses={"1000": JobStatus.RUNNING, "2000": JobStatus.RUNNING})
    claims = {
        (str(_SESSION_PATH), "job0"): TrackerClaim(status="RUNNING", executor_id="slurm:2000", allocation="2000"),
        (str(_SESSION_PATH), "job1"): TrackerClaim(status="RUNNING"),
    }
    batch = _build_batch(submissions=[_build_submission(), _build_submission(job_id="job1", slurm_job_id="1001")])

    resolutions = resolve_allocations(batches=[batch], reading=reading, claims=claims)

    # The recorded allocation is this host's own submission and the claimed one may be another machine's, so naming
    # the recorded one alone would leave the allocation actually running free to write into the tracker that is then
    # reset.
    assert resolve_live_allocations(resolutions=resolutions) == ["1000", "2000"]


def test_the_allocation_a_tracker_claims_is_named_though_the_recorded_one_is_gone() -> None:
    """Verifies that a resolution running on its claim alone still names the allocation carrying its job."""
    reading = SchedulerReading(statuses={"1000": JobStatus.UNRESOLVED, "2000": JobStatus.RUNNING})
    claims = {(str(_SESSION_PATH), "job0"): TrackerClaim(status="RUNNING", executor_id="slurm:2000", allocation="2000")}

    resolutions = resolve_allocations(
        batches=[_build_batch(submissions=[_build_submission()])], reading=reading, claims=claims
    )

    assert resolve_live_allocations(resolutions=resolutions) == ["2000"]


# Tests for the one remediation that writes to a tracker


def test_only_a_stranded_job_is_returned_to_the_scheduled_state() -> None:
    """Verifies that the reset covers the stranded jobs alone, grouped by the pipeline and unit that record them."""
    reading = SchedulerReading(
        statuses={"1000": JobStatus.UNRESOLVED, "1001": JobStatus.COMPLETED, "1002": JobStatus.FAILED}
    )
    claims = {
        (str(_SESSION_PATH), "job0"): TrackerClaim(status="RUNNING"),
        (str(_SESSION_PATH), "job1"): TrackerClaim(status="SUCCEEDED"),
        (str(_SESSION_PATH), "job2"): TrackerClaim(status="FAILED"),
    }
    batch = _build_batch(
        submissions=[
            _build_submission(),
            _build_submission(job_id="job1", slurm_job_id="1001"),
            _build_submission(job_id="job2", slurm_job_id="1002"),
        ]
    )
    host = _StubHost()

    reset = reset_stranded_jobs(
        host=host, resolutions=resolve_allocations(batches=[batch], reading=reading, claims=claims)
    )

    assert reset == {(str(_SESSION_PATH), "job0")}
    assert host.reset_calls == [("video", {str(_SESSION_PATH): ["job0"]})]


def test_one_stranded_job_of_two_batches_is_named_once() -> None:
    """Verifies that a job two ledger entries both record is reset once rather than named twice in one command."""
    reading = SchedulerReading(statuses={"1000": JobStatus.UNRESOLVED, "1001": JobStatus.UNRESOLVED})
    claims = {(str(_SESSION_PATH), "job0"): TrackerClaim(status="RUNNING")}
    batches = [
        _build_batch(batch_id="first", submissions=[_build_submission()]),
        _build_batch(batch_id="second", submissions=[_build_submission(slurm_job_id="1001")]),
    ]
    host = _StubHost()

    reset_stranded_jobs(host=host, resolutions=resolve_allocations(batches=batches, reading=reading, claims=claims))

    assert host.reset_calls == [("video", {str(_SESSION_PATH): ["job0"]})]


def test_a_batch_holding_no_stranded_job_reaches_the_host_not_at_all() -> None:
    """Verifies that a remediation with nothing to release never names a unit, which would reset every job it holds."""
    reading = SchedulerReading(statuses={"1000": JobStatus.COMPLETED})
    claims = {(str(_SESSION_PATH), "job0"): TrackerClaim(status="SUCCEEDED")}
    batch = _build_batch(submissions=[_build_submission()])
    host = _StubHost()

    assert (
        reset_stranded_jobs(host=host, resolutions=resolve_allocations(batches=[batch], reading=reading, claims=claims))
        == set()
    )
    assert host.reset_calls == []


def test_a_reset_the_host_refuses_raises() -> None:
    """Verifies that a tracker that could not be cleared is reported rather than passed over."""
    reading = SchedulerReading(statuses={"1000": JobStatus.UNRESOLVED})
    claims = {(str(_SESSION_PATH), "job0"): TrackerClaim(status="RUNNING")}
    batch = _build_batch(submissions=[_build_submission()])
    host = _StubHost(reset_error=_HOST_FAILURE)

    with pytest.raises(RuntimeError, match=r"could not be regenerated"):
        reset_stranded_jobs(host=host, resolutions=resolve_allocations(batches=[batch], reading=reading, claims=claims))


# Tests for how a resolved allocation is reported


def test_a_resolved_allocation_reports_the_evidence_behind_its_verdict() -> None:
    """Verifies that the rendered row carries what each record said alongside the verdict they carry together."""
    reading = SchedulerReading(statuses={"1000": JobStatus.UNRESOLVED, "2000": JobStatus.UNRESOLVED})
    claims = {(str(_SESSION_PATH), "job0"): TrackerClaim(status="RUNNING", executor_id="slurm:2000", allocation="2000")}
    batch = _build_batch(submissions=[_build_submission()])

    resolved = resolve_allocations(batches=[batch], reading=reading, claims=claims)
    rendered = render_allocation(resolution=resolved[0], reading=reading)

    assert rendered["batch_id"] == "batch01"
    assert rendered["slurm_job_id"] == "1000"
    assert rendered["status"] == JobStatus.UNRESOLVED.value
    assert rendered["queued"] is False
    assert rendered["scheduler_state"] == GONE_ALLOCATION
    assert rendered["tracker_status"] == "RUNNING"
    assert rendered["tracker_executor_id"] == "slurm:2000"
    assert rendered["claimed_allocation"] == "2000"
    assert rendered["claim_state"] == GONE_ALLOCATION
    assert rendered["verdict"] == STRANDED_ALLOCATION
    assert rendered["remediation"] == RESET_REMEDIATION
