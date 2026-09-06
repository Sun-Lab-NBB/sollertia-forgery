"""Provides the remote execution backend that runs prepared jobs as SLURM allocations on the compute server."""

from __future__ import annotations

import re
from math import ceil
import shlex
from typing import TYPE_CHECKING, Any
from pathlib import Path
from dataclasses import field, asdict, replace, dataclass

from natsort import natsorted
from ataraxis_base_utilities import LogLevel, console
from sollertia_shared_assets import DATASET_MARKER_FILENAME, ProcessingTrackers
from ataraxis_data_structures import ProcessingStatus

from .graph import build_pending_job, index_rows_by_unit, resolve_submission_order
from .hosts import environment_command, state_artifact_paths
from .ledger import SubmissionBatch, RemoteSubmission, record_batch, current_timestamp
from ..server import TERMINAL_JOB_STATUSES, Job, Server, JobStatus, discover_project_markers, get_server_configuration
from ..forging import DATASET_STATE_FILENAME
from .dispatch import resolve_unit_kind, resolve_job_command
from .planning import project_plan_path
from ..managing import project_jobs_path, project_manifest_path
from .preparation import resolve_project_root

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .graph import GenericPendingJob
    from .hosts import ExecutionHost


REMOTE_JOB_WALLTIME_MINUTES: int = 480
"""The wall-time a remote allocation requests when a caller names none, in minutes.

Notes:
    One figure covers every job type, because this bound exists to stop a run that has stopped making progress rather
    than to describe how long a stage takes.
"""

_BATCH_DIRECTORY_NAME: str = "processing_batches"
"""The directory under the server's data root that holds one subdirectory per submitted batch."""

_MEGABYTES_PER_GIGABYTE: int = 1024
"""The divisor converting an estimate in megabytes into the gigabyte figure a SLURM memory request takes."""

_SLURM_NAME_SANITIZER: re.Pattern[str] = re.compile(r"[^A-Za-z0-9._-]+")
"""Matches every character excluded from a SLURM job name and its script filename."""

_SLURM_EXECUTOR_SCHEME: str = "slurm"
"""The scheme an executor identifier carries when the job ran as a SLURM allocation."""

HELD_ALLOCATION: str = "held"
"""The scheduler state of an allocation the scheduler still holds, which is one its queue carries or one accounting
reports in a state it has yet to leave.

Notes:
    A queue row and a non-terminal accounting row each prove the allocation is still the scheduler's, and neither is
    contradicted by the other record saying nothing about it. A record that could not be read at all resolves here
    too, because a source that did not answer is no evidence of absence. A queue row is not evidence of this state for
    an allocation the queue itself reports as permanently blocked, since such an allocation never runs however long
    the queue carries it.
"""

SETTLED_ALLOCATION: str = "settled"
"""The scheduler state of an allocation that accounting reports in a state it never leaves and that the queue no longer
carries, of one the queue reports as permanently blocked, and of one this reading has since cancelled. The scheduler
has finished with it, or will never start it, so nothing it holds can change again."""

GONE_ALLOCATION: str = "gone"
"""The scheduler state of an allocation accounting returns no row for and the queue does not carry. Both records
disclaim it, and only both of them together are evidence that the scheduler no longer holds it."""

RUNNING_ALLOCATION: str = "running"
"""The verdict on an allocation the scheduler still holds, on one whose job a still-held allocation claims on its
tracker, and on one whose job's tracker claims to be running under an executor outside the scheduler. Nothing is
remediated for it, because every remediation this module applies would disturb work that may still be live."""

FINISHED_ALLOCATION: str = "finished"
"""The verdict on an allocation the scheduler no longer holds whose job recorded success. Its tracker is left exactly
as it stands, since resetting it would discard a result the run actually produced."""

FAILED_ALLOCATION: str = "failed"
"""The verdict on an allocation the scheduler no longer holds whose job recorded failure. Its tracker is left exactly
as it stands, because a failure is a real verdict the operator must see and clear deliberately."""

ABANDONED_ALLOCATION: str = "abandoned"
"""The verdict on an allocation the scheduler no longer holds, whose job either never left the scheduled state or
carries no record on its tracker at all. Nothing claims that job, so it is already runnable once the ledger entry is
gone."""

STRANDED_ALLOCATION: str = "stranded"
"""The verdict on an allocation the scheduler no longer holds whose job's tracker still claims to be running it under
an executor for which the scheduler can answer. This is the one verdict whose remediation writes to a tracker, because
that claim is what no rerun can otherwise clear."""

NO_REMEDIATION: str = "none"
"""The remediation prescribed for a running allocation, which is to leave it alone."""

DROP_REMEDIATION: str = "drop"
"""The remediation prescribed for a finished, failed, or abandoned allocation, which is to snapshot what its batch's
jobs recorded and drop the ledger entry, leaving every tracker as it stands."""

RESET_REMEDIATION: str = "reset_and_drop"
"""The remediation prescribed for a stranded allocation, which returns its job to the scheduled state on its own
tracker before the snapshot and the drop."""

CANCEL_REMEDIATION: str = "cancel_reset_and_drop"
"""The remediation applied to an allocation a caller overriding the refusal cancelled and whose job the cancellation
left stranded on its own tracker. The cancellation runs before any tracker is written, so no tracker is reset underneath
an allocation the scheduler was never told to stop."""

PROGRESSING_BATCH: str = "progressing"
"""The verdict a batch carries while at least one of its allocations resolves as running. Holding an allocation is the
only evidence that the work can still advance, and it is evidence no elapsed period can contradict, so a batch that
carries this verdict is left alone however long it has been queued."""

STALLED_BATCH: str = "stalled"
"""The verdict a batch carries when none of its allocations resolves as running and at least one of them is gone from
both of the scheduler's records. Neither record answers for such an allocation again, so nothing the scheduler does
advances the batch. A stalled batch stays outstanding past a read only when it holds an entry the automatic closure may
not drop, which is a stranded job, or when that closure failed, and an explicit remediation is what releases it."""

AWAITING_CLOSURE_BATCH: str = "awaiting_closure"
"""The verdict a batch carries when every allocation it holds has settled. The read that observes a batch resolving
entirely to the plain drop also closes it. A batch outstanding under this verdict is therefore one holding an entry
that closure may not drop or one whose closure failed, and the next read retries it."""

_VERDICT_REMEDIATIONS: dict[str, str] = {
    RUNNING_ALLOCATION: NO_REMEDIATION,
    FINISHED_ALLOCATION: DROP_REMEDIATION,
    FAILED_ALLOCATION: DROP_REMEDIATION,
    ABANDONED_ALLOCATION: DROP_REMEDIATION,
    STRANDED_ALLOCATION: RESET_REMEDIATION,
}
"""The remediation each verdict prescribes.

Notes:
    This mapping is the state table itself, so the remediation a caller is shown and the remediation that runs are one
    fact rather than two that could drift. Only the stranded verdict prescribes a tracker write, which is what keeps a
    recorded success from being discarded and a recorded failure from being silently cleared.
"""


@dataclass(frozen=True, slots=True)
class TrackerClaim:
    """Records what one job's own processing tracker holds for the run that a submission covers."""

    status: str = ""
    """The name of the ``ProcessingStatus`` member the tracker recorded, or empty when the state artifact carries no
    row for the job at all."""
    executor_id: str = ""
    """The executor the same record named, which is a scheme-tagged identifier such as a scheduler allocation."""
    allocation: str = ""
    """The scheduler allocation that executor claims, or empty when it claims none. A claim is what survives the
    ledger entry, so it is resolved against the scheduler alongside the allocation the ledger recorded.

    Notes:
        Only the scheduler's own scheme names an allocation, so an executor recorded under any other scheme, such as
        the process identifier a local run writes, resolves to nothing here and is read against neither of the
        scheduler's records. Such an executor is therefore invisible to this resolution rather than absent from it,
        which is why a tracker claiming to be running under one is refused rather than treated as stranded."""

    @property
    def unqueryable_executor(self) -> bool:
        """Returns True when the tracker recorded an executor that resolves to no scheduler allocation."""
        return bool(self.executor_id) and not self.allocation


@dataclass(frozen=True, slots=True)
class SchedulerReading:
    """Records what the scheduler's two records reported about one set of allocations.

    Notes:
        Accounting says what the scheduler has committed and the queue says what it holds right now. Neither answers
        for the other, so an allocation is resolved against both and a source that did not answer withholds nothing
        but certainty.
    """

    statuses: dict[str, JobStatus] = field(default_factory=dict)
    """The state accounting reported for each queried allocation, keyed by its scheduler identifier."""
    queued: frozenset[str] = frozenset()
    """The allocations the queue currently holds, which is empty for a user whose queue holds none of them."""
    cancelled: frozenset[str] = frozenset()
    """The allocations this process has since cancelled. The scheduler applies a cancellation to the queued and
    running allocations alike, so a cancelled allocation is the scheduler's no longer whatever the reading that
    preceded the cancellation said."""
    unreadable_reason: str = ""
    """What stopped one of the scheduler's records from being read, or empty when both answered. Every allocation this
    reading has neither cancelled nor found permanently blocked resolves as held while this is set, because a record
    that did not answer is no evidence that the scheduler has finished with an allocation."""

    def resolve_state(self, allocation: str) -> str:
        """Resolves one allocation to the state in which the scheduler's two records place it.

        Notes:
            The invariant this holds is that an allocation's state follows from what each record says about it and
            never from the order in which the two were read. Accounting and the queue are read one after the other
            rather than as one snapshot, so they answer for different moments and can disagree about the same
            allocation. A resolution that let the earlier read win would answer differently on two reads of one
            unchanged batch.

            The ``BLOCKED`` state is where that disagreement bites, and it is why the accounting map is consulted for
            it ahead of the queue. Accounting calls such an allocation pending, so the state is derived from the
            queue's own reason field. Every allocation carrying it was therefore in the queue when accounting was
            read, while the later queue snapshot may already have released it. A dependency that can never be satisfied
            is permanent. The allocation never runs, so it never writes to its job's tracker and nothing it holds can
            change again. That is the settled state, and resolving it here answers the same way on whichever side of
            the queue's own release of the allocation the two reads fall. Deciding it from the queue instead would hold
            the allocation on one read and settle it on the next, leaving its batch reported as progressing for as
            long as the queue kept a job that will never start.

        Args:
            allocation: The scheduler identifier to resolve, or an empty string for a record naming no allocation.

        Returns:
            One of ``held``, ``settled``, or ``gone``.
        """
        # A record naming no allocation holds nothing, whatever either source says about the allocations it does name.
        if not allocation:
            return GONE_ALLOCATION
        if allocation in self.cancelled:
            return SETTLED_ALLOCATION
        status = self.statuses.get(allocation, JobStatus.UNRESOLVED)
        if status is JobStatus.BLOCKED:
            return SETTLED_ALLOCATION
        if self.unreadable_reason or allocation in self.queued:
            return HELD_ALLOCATION
        if status is JobStatus.UNRESOLVED:
            return GONE_ALLOCATION
        if status in TERMINAL_JOB_STATUSES:
            return SETTLED_ALLOCATION
        return HELD_ALLOCATION

    def resolve_status(self, allocation: str) -> JobStatus:
        """Returns the state accounting reported for one allocation, which is ``UNRESOLVED`` when it reported none.

        Args:
            allocation: The scheduler identifier whose accounting state to report.

        Returns:
            The reported state.
        """
        return self.statuses.get(allocation, JobStatus.UNRESOLVED)

    def cancelling(self, allocations: Sequence[str]) -> SchedulerReading:
        """Returns this reading with the named allocations recorded as cancelled.

        Notes:
            A cancellation is issued rather than observed, since the scheduler applies it asynchronously and an
            accounting query issued straight afterwards may still report the allocation as running. Recording the
            cancellation here is what lets the verdicts be resolved again from what this process actually did.

        Args:
            allocations: The allocations the cancellation named.

        Returns:
            The reading, which resolves every named allocation as settled.
        """
        return replace(self, cancelled=self.cancelled | set(allocations))


@dataclass(frozen=True, slots=True)
class AllocationResolution:
    """Resolves one recorded allocation to the verdict its scheduler state and its tracker state together carry."""

    batch_id: str
    """The identifier of the ledger batch that recorded this allocation."""
    submission: RemoteSubmission
    """The recorded submission, naming both the job and the allocation that carries it."""
    scheduler_state: str
    """The state in which the scheduler's records place the recorded allocation."""
    claim_state: str
    """The state in which those same records place the allocation this job's tracker claims, or empty when it claims
    none."""
    tracker: TrackerClaim
    """What the job's own tracker recorded, which is the record that gates rerunning it."""
    verdict: str
    """The verdict to which the two states resolve."""
    remediation: str
    """The remediation that verdict prescribes."""


def remote_batch_directory(server: Server, batch_id: str) -> Path:
    """Resolves the server-side directory holding one batch's job scripts and logs.

    Args:
        server: The server that runs the batch.
        batch_id: The identifier of the batch.

    Returns:
        The path to the batch's directory on the server.
    """
    return server.root.joinpath(_BATCH_DIRECTORY_NAME, batch_id)


def submit_batch(
    server: Server,
    jobs: Sequence[dict[str, Any]],
    batch_id: str,
    adopted: dict[tuple[str, str], str] | None = None,
    covered_batch_ids: Sequence[str] = (),
    *,
    walltime_minutes: int = REMOTE_JOB_WALLTIME_MINUTES,
    verbose: bool = False,
) -> list[RemoteSubmission]:
    """Submits a prepared batch to the scheduler as a dependency graph.

    Notes:
        The scheduler sequences the graph itself, so this process may exit as soon as the last job is queued.

        The batch is recorded before the first allocation is queued, so a submission that does not return still
        leaves a batch the remote tools name and resolve. A submission the scheduler refuses partway through records
        the allocations it had accepted, because the record is written on the way out of the submission either way. A
        submission the host kills outright records none of them, and the batch it leaves resolves through the jobs'
        own trackers and the scheduler queue rather than from allocations the ledger names.

        Every accepted allocation is recorded in the submission ledger, including when the scheduler rejects a later
        job of the same batch, since the allocations it already accepted stay queued.

        The record is merged into whatever the ledger already holds for this batch, so re-running a batch the scheduler
        only partly accepted keeps the allocations the first attempt queued. An entry this call re-submitted is
        replaced rather than duplicated. The merge is left to the recording call, which reads the entries it carries
        forward under the same lock that writes them. An entry a concurrent writer committed against the same batch is
        therefore never dropped by a list this call read before that lock was taken.

        An adopted job's allocation seeds the dependency map before anything is submitted, so a dependent of a job that
        is already running waits on the allocation running it rather than on a second one.

        The concurrency ceilings that the local engine applies do not reach the scheduler. Expressing one natively
        needs a job array, whose tasks share a single memory request, so the scheduler is left to sequence the whole
        batch.

    Args:
        server: The connected server that receives the submission.
        jobs: The job descriptors to submit.
        batch_id: The identifier of the batch, which names the directory that holds the scripts and logs.
        adopted: The allocation already running each adopted job, keyed by dispatch key. These jobs are not submitted,
            and their dependents wait on those allocations.
        covered_batch_ids: Every prepared batch this submission dispatches. Closure snapshots an outcome for each.
            Leave empty for a submission covering the batch ``batch_id`` names alone.
        walltime_minutes: The wall-time every allocation requests.
        verbose: Determines whether to report each submission as it is accepted.

    Returns:
        The submissions, in the order they were accepted.

    Raises:
        RuntimeError: If the scheduler rejects a submission.
        Timeout: If the submission ledger's lock cannot be acquired within the timeout period.
    """
    batch_directory = remote_batch_directory(server=server, batch_id=batch_id)
    server.create(remote_path=batch_directory, is_dir=True, parents=True)

    pending = [build_pending_job(job=descriptor) for descriptor in jobs]
    ordered = resolve_submission_order(jobs=pending)

    submissions: list[RemoteSubmission] = []
    allocation_of_job: dict[tuple[str, str], str] = dict(adopted or {})
    covered = list(covered_batch_ids) if covered_batch_ids else [batch_id]
    submitted_at = current_timestamp()

    # The batch reaches the ledger before the first allocation is queued, so a submission that never returns still
    # leaves a batch the remote tools name. The record carries no allocation yet, because the allocations are written
    # on the way out, so this names the batch rather than its contents. The empty resubmission list takes the merge
    # path, which carries forward every allocation an earlier attempt recorded.
    record_batch(
        batch=SubmissionBatch(
            batch_id=batch_id,
            batch_ids=covered,
            batch_directory=str(batch_directory),
            submitted_at=submitted_at,
            walltime_minutes=walltime_minutes,
            submissions=[],
        ),
        resubmitted=[],
    )

    try:
        _submit_ordered_jobs(
            server=server,
            ordered=ordered,
            batch_directory=batch_directory,
            walltime_minutes=walltime_minutes,
            submissions=submissions,
            allocation_of_job=allocation_of_job,
            verbose=verbose,
        )
    finally:
        if submissions:
            record_batch(
                batch=SubmissionBatch(
                    batch_id=batch_id,
                    batch_ids=covered,
                    batch_directory=str(batch_directory),
                    submitted_at=submitted_at,
                    walltime_minutes=walltime_minutes,
                    submissions=submissions,
                ),
                resubmitted=[(entry.unit_path, entry.job_id) for entry in submissions],
            )

    return submissions


def cancel_submissions(server: Server, submissions: Sequence[RemoteSubmission]) -> list[str]:
    """Cancels every allocation the given submissions hold in one call, which the scheduler applies to the queued and
    running ones alone.

    Args:
        server: The connected server that runs the batch.
        submissions: The submissions to cancel.

    Returns:
        The allocation identifiers the cancellation named.
    """
    return cancel_allocations(server=server, allocations=[submission.slurm_job_id for submission in submissions])


def cancel_allocations(server: Server, allocations: Sequence[str]) -> list[str]:
    """Cancels every named allocation in one call, which the scheduler applies to the queued and running ones alone.

    Notes:
        The allocations are named rather than resolved from the ledger, because the allocation running a job is not
        always the one this host recorded for it. A job whose tracker claims another submitter's allocation is
        reached through this call, since the ledger holds no record naming that allocation.

    Args:
        server: The connected server that runs the allocations.
        allocations: The scheduler identifiers to cancel.

    Returns:
        The allocation identifiers the cancellation named.
    """
    named = list(allocations)
    server.abort_jobs(slurm_job_ids=named)
    return named


def resolve_slurm_allocation(executor_id: str) -> str:
    """Resolves the scheduler allocation that one recorded executor identifier claims.

    Notes:
        An executor identifier is scheme-tagged, and only the scheduler's own scheme names an allocation. A job that
        recorded a process identifier or nothing at all therefore claims no allocation, and neither of the scheduler's
        records answers for it.

    Args:
        executor_id: The executor identifier a job's tracker recorded.

    Returns:
        The allocation identifier the executor claims, or an empty string when it claims none.
    """
    scheme, _, allocation = executor_id.partition(":")
    return allocation if scheme == _SLURM_EXECUTOR_SCHEME else ""


def resolve_tracker_claims(
    host: ExecutionHost, submissions: Sequence[RemoteSubmission]
) -> dict[tuple[str, str], TrackerClaim]:
    """Reads what each submitted job's own processing tracker currently records.

    Notes:
        The trackers are rewritten into the host's state artifacts before they are read, because a job records its
        outcome on its tracker and nothing regenerates those artifacts while a batch runs. Reading them as they stand
        would report the state against which the batch was prepared, which is what would let a finished job read as one
        still running and be reset.

        The regeneration is issued once per project and unit kind, since a session artifact covers a whole project
        while a dataset artifact covers one dataset. The rows it produces are then narrowed to each pipeline the
        submissions name, the way a batch is resolved out of the same artifacts.

        A submission naming no unit is skipped rather than resolved, since a record that does not say which unit it
        ran against names no tracker to read.

    Args:
        host: The host holding the units whose trackers to read.
        submissions: The submissions whose jobs to read.

    Returns:
        What each job's tracker records, keyed by the unit path and job identifier that name the job. A job the
        artifacts hold no row for carries an empty claim.

    Raises:
        RuntimeError: If a step fails on the host.
    """
    units: dict[tuple[str, str], set[str]] = {}
    pipelines: dict[tuple[str, str], set[str]] = {}
    for submission in submissions:
        if not submission.unit_path:
            continue
        unit_kind = resolve_unit_kind(pipeline=submission.pipeline)
        project_root = resolve_project_root(unit_paths=[Path(submission.unit_path)], unit_kind=unit_kind)
        group = (unit_kind, str(project_root))
        units.setdefault(group, set()).add(submission.unit_path)
        pipelines.setdefault(group, set()).add(submission.pipeline)

    recorded: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for group, unit_paths in units.items():
        rows = _regenerate_recorded_state(host=host, group=group, unit_paths=unit_paths)
        unit_kind = group[0]
        for pipeline in natsorted(pipelines[group]):
            indexed = index_rows_by_unit(
                rows=[row for row in rows if row.get("pipeline", pipeline) == pipeline], key=unit_kind
            )
            for unit_name, jobs in indexed.items():
                recorded.setdefault(pipeline, {}).setdefault(unit_name, {}).update(jobs)

    claims: dict[tuple[str, str], TrackerClaim] = {}
    for submission in submissions:
        row = recorded.get(submission.pipeline, {}).get(Path(submission.unit_path).name, {}).get(submission.job_id, {})
        executor_id = str(row.get("executor_id") or "")
        claims[(submission.unit_path, submission.job_id)] = TrackerClaim(
            status=str(row.get("status") or ""),
            executor_id=executor_id,
            allocation=resolve_slurm_allocation(executor_id=executor_id),
        )
    return claims


def resolve_queried_allocations(
    submissions: Sequence[RemoteSubmission], claims: Mapping[tuple[str, str], TrackerClaim]
) -> list[str]:
    """Resolves every allocation a verdict reads, which is the one each submission recorded and the one each job's
    tracker claims.

    Notes:
        A tracker claim is read alongside the ledger's own record because the two cover different submitters. A job
        another machine submitted is claimed on the tracker alone, and resetting that job while its allocation still
        runs is exactly the destruction this resolution exists to refuse.

    Args:
        submissions: The submissions whose recorded allocations to read.
        claims: What each job's tracker recorded, keyed by unit path and job identifier.

    Returns:
        The allocation identifiers to query, in natural order and holding no duplicates.
    """
    allocations = {submission.slurm_job_id for submission in submissions}
    allocations.update(claim.allocation for claim in claims.values())
    return natsorted(allocation for allocation in allocations if allocation)


def read_scheduler_records(server: Server, allocations: Sequence[str]) -> SchedulerReading:
    """Reads what accounting and the queue each report about the named allocations.

    Notes:
        Accounting is read first and a failure there is raised, because a failed query writes nothing to standard
        output and reporting that as 'no row for anything' would resolve every allocation as gone at once.

        The queue is read second and a failure there is carried rather than raised, because the reading it leaves is
        still usable. Every allocation the reading has neither cancelled nor found permanently blocked resolves as
        held, which is the reading that disturbs nothing.

    Args:
        server: The connected server that runs the allocations.
        allocations: The scheduler identifiers to read.

    Returns:
        The reading, carrying accounting's answer, the queue's answer, and what stopped the queue when it could not be
        read.

    Raises:
        RuntimeError: If the accounting query fails.
    """
    statuses = server.get_job_statuses(slurm_job_ids=allocations)
    try:
        queued = frozenset(server.get_queued_job_ids())
    except Exception as exception:
        return SchedulerReading(
            statuses=statuses,
            unreadable_reason=f"The remote compute server's scheduler queue could not be read. {exception}",
        )
    return SchedulerReading(statuses=statuses, queued=queued)


def resolve_allocations(
    batches: Sequence[SubmissionBatch], reading: SchedulerReading, claims: Mapping[tuple[str, str], TrackerClaim]
) -> list[AllocationResolution]:
    """Resolves every allocation of the given batches to one verdict and the remediation it prescribes.

    Notes:
        Every allocation is resolved from the batch's own submissions rather than from the answer a query returned,
        so an allocation for which neither record answered is still resolved rather than passed over.

        An allocation resolves as running whenever the scheduler holds it, holds the allocation its job's tracker
        claims, or that tracker claims to be running under an executor for which neither of the scheduler's records
        answers. Everything else is resolved by what that tracker recorded, because the tracker is what gates rerunning
        the job.

    Args:
        batches: The recorded batches whose allocations to resolve.
        reading: What the scheduler's records reported.
        claims: What each job's tracker recorded, keyed by unit path and job identifier.

    Returns:
        One resolution per recorded allocation, in the order the batches hold them.
    """
    resolutions: list[AllocationResolution] = []
    for batch in batches:
        for submission in batch.submissions:
            claim = claims.get((submission.unit_path, submission.job_id), TrackerClaim())
            claim_state = reading.resolve_state(allocation=claim.allocation) if claim.allocation else ""
            scheduler_state = reading.resolve_state(allocation=submission.slurm_job_id)
            verdict = _resolve_verdict(scheduler_state=scheduler_state, claim_state=claim_state, tracker=claim)
            resolutions.append(
                AllocationResolution(
                    batch_id=batch.batch_id,
                    submission=submission,
                    scheduler_state=scheduler_state,
                    claim_state=claim_state,
                    tracker=claim,
                    verdict=verdict,
                    remediation=_VERDICT_REMEDIATIONS[verdict],
                )
            )
    return resolutions


def resolve_live_allocations(resolutions: Sequence[AllocationResolution]) -> list[str]:
    """Resolves every allocation of the given resolutions that the scheduler still holds.

    Notes:
        A resolution names two allocations, and either of them being held is what makes its verdict running. The
        recorded one is this host's own submission, while the one the job's tracker claims may be another submitter's,
        so both are named here. A cancellation covering the recorded allocation alone would leave the allocation
        actually running the job free to write into the tracker that same remediation then resets.

    Args:
        resolutions: The resolutions whose held allocations to name.

    Returns:
        The allocation identifiers, in natural order and holding no duplicates.
    """
    allocations = {
        resolution.submission.slurm_job_id
        for resolution in resolutions
        if resolution.scheduler_state == HELD_ALLOCATION
    }
    allocations.update(
        resolution.tracker.allocation for resolution in resolutions if resolution.claim_state == HELD_ALLOCATION
    )
    return natsorted(allocations)


def classify_batch(resolutions: Sequence[AllocationResolution]) -> str:
    """Resolves one batch's verdict from the verdicts of the allocations it holds.

    Notes:
        The verdict rests on the states the batch's allocations hold rather than on how long the batch has been
        outstanding, so nothing has to be tuned and a slow run is never mistaken for a stopped one.

        A batch every allocation of which has settled is awaiting closure rather than progressing or stalled, since
        the read that observes a batch resolving entirely to the plain drop also closes it. Such a batch is
        outstanding only because it holds an entry that closure may not drop or because that closure failed. A batch
        holding no allocation at all resolves the same way, since there is nothing left for the scheduler to advance.

    Args:
        resolutions: The resolutions of the allocations the batch holds.

    Returns:
        One of ``progressing``, ``stalled``, or ``awaiting_closure``.
    """
    if any(resolution.verdict == RUNNING_ALLOCATION for resolution in resolutions):
        return PROGRESSING_BATCH
    if any(resolution.scheduler_state == GONE_ALLOCATION for resolution in resolutions):
        return STALLED_BATCH
    return AWAITING_CLOSURE_BATCH


def reset_stranded_jobs(host: ExecutionHost, resolutions: Sequence[AllocationResolution]) -> set[tuple[str, str]]:
    """Returns every stranded job of the given resolutions to the scheduled state on its own tracker.

    Notes:
        Only a stranded job is reset. A job that recorded success would lose that result, a job that recorded failure
        would lose the verdict its operator has yet to see, and a job that never left the scheduled state is already
        runnable. A write to any of those trackers would destroy or invent a record rather than release one.

        The identifiers are grouped by pipeline and by unit, because a job identifier names a stage rather than a
        unit and two units of one project share the identifier of the same stage. A unit named with no identifier at
        all has every job it tracks reset, so a unit contributing no stranded job is never named.

    Args:
        host: The host holding the units whose trackers to write.
        resolutions: The resolutions whose stranded jobs to reset.

    Returns:
        The unit path and job identifier of each job that was reset.

    Raises:
        RuntimeError: If a step fails on the host.
    """
    grouped: dict[str, dict[Path, list[str]]] = {}
    reset: set[tuple[str, str]] = set()
    for resolution in resolutions:
        if resolution.verdict != STRANDED_ALLOCATION:
            continue
        submission = resolution.submission
        job_ids = grouped.setdefault(submission.pipeline, {}).setdefault(Path(submission.unit_path), [])
        if submission.job_id not in job_ids:
            job_ids.append(submission.job_id)
        reset.add((submission.unit_path, submission.job_id))

    for pipeline, job_ids_by_unit in grouped.items():
        host.reset_jobs(pipeline=pipeline, job_ids_by_unit=job_ids_by_unit)
    return reset


def render_allocation(resolution: AllocationResolution, reading: SchedulerReading) -> dict[str, Any]:
    """Renders one resolved allocation as a response payload.

    Args:
        resolution: The resolution to render.
        reading: The reading against which the resolution was resolved, which carries the state accounting reported.

    Returns:
        The submission's own fields alongside the batch that recorded it, the state in which each scheduler record
        placed it, what its job's tracker holds, the verdict, and the remediation that verdict prescribes.
    """
    allocation = resolution.submission.slurm_job_id
    return {
        **render_submission(submission=resolution.submission),
        "batch_id": resolution.batch_id,
        "status": reading.resolve_status(allocation=allocation).value,
        "queued": allocation in reading.queued,
        "scheduler_state": resolution.scheduler_state,
        "tracker_status": resolution.tracker.status,
        "tracker_executor_id": resolution.tracker.executor_id,
        "claimed_allocation": resolution.tracker.allocation,
        "claim_state": resolution.claim_state,
        "verdict": resolution.verdict,
        "remediation": resolution.remediation,
    }


def sync_project_state(server: Server, project: str, local_directory: Path, *, regenerate: bool = True) -> list[Path]:
    """Regenerates a remote project's state artifacts and mirrors them onto this host.

    Notes:
        Regeneration precedes the pull, so the mirrored state tables describe the state after the runs rather than
        before them. The plan projection is mirrored as the server last wrote it, since replanning is a preparation
        step rather than a mirroring one.

    Args:
        server: The connected server holding the project.
        project: The name of the project whose state to mirror.
        local_directory: The local directory that receives the mirrored artifacts.
        regenerate: Determines whether to regenerate the artifacts on the server before pulling them.

    Returns:
        Where the artifacts were written, holding one entry per artifact the server carried.

    Raises:
        FileNotFoundError: If the server holds no directory for the named project.
        RuntimeError: If the server-side search for the project's datasets reached only part of its tree.
    """
    project_path = server.root.joinpath(project)
    if not server.is_directory(remote_path=project_path):
        message = (
            f"Unable to mirror the state of project '{project}'. The remote compute server holds no directory at "
            f"'{project_path}'."
        )
        console.error(message=message, error=FileNotFoundError)

    # Only the datasets are mirrored, so the search is held to the depth at which their markers sit rather than
    # reading every session directory the project holds.
    datasets = list(discover_project_markers(project_path=project_path, server=server, include_sessions=False).datasets)
    if regenerate:
        _regenerate_remote_state(server=server, project_path=project_path, datasets=datasets)

    # The markers and the manifest tracker travel alongside the tables, because the read tools resolve a dataset from
    # its marker and read the manifest's progress from its tracker. Mirroring the tables alone would leave those tools
    # reporting a project with no datasets and a manifest that had never been generated.
    remote_artifacts = [
        project_manifest_path(project_directory=project_path),
        project_path.joinpath(ProcessingTrackers.MANIFEST),
        project_jobs_path(project_directory=project_path),
        project_plan_path(project_directory=project_path),
        *[dataset.joinpath(DATASET_MARKER_FILENAME) for dataset in datasets],
        *[dataset.joinpath(DATASET_STATE_FILENAME) for dataset in datasets],
    ]

    mirrored: list[Path] = []
    for remote_artifact in remote_artifacts:
        if not server.exists(remote_path=remote_artifact):
            continue
        local_artifact = local_directory.joinpath(remote_artifact.relative_to(project_path))
        local_artifact.parent.mkdir(parents=True, exist_ok=True)
        server.pull(local_path=local_artifact, remote_path=remote_artifact)
        mirrored.append(local_artifact)

    return mirrored


def connect_to_server() -> Server:
    """Opens a connection to the configured remote compute server.

    Returns:
        The connected server, which the caller closes or uses as a context manager.
    """
    return Server(configuration=get_server_configuration())


def render_submission(submission: RemoteSubmission) -> dict[str, Any]:
    """Renders one submission as a response payload.

    Args:
        submission: The submission to render.

    Returns:
        The submission's fields as a plain dictionary.
    """
    return asdict(submission)


def _regenerate_recorded_state(
    host: ExecutionHost, group: tuple[str, str], unit_paths: set[str]
) -> list[dict[str, Any]]:
    """Rewrites one project's state artifacts from the trackers under it and reads back every row they hold.

    Args:
        host: The host holding the units.
        group: The unit kind and project root for which the artifacts are written.
        unit_paths: The unit directories the artifacts cover.

    Returns:
        The rows the artifacts hold.

    Raises:
        RuntimeError: If a step fails on the host.
    """
    unit_kind, project_root = group
    units = [Path(unit_path) for unit_path in natsorted(unit_paths)]
    root = Path(project_root)
    host.generate_state(project_root=root, unit_paths=units, unit_kind=unit_kind)
    return [
        row
        for artifact in state_artifact_paths(project_root=root, unit_paths=units, unit_kind=unit_kind)
        for row in host.read_rows(path=artifact)
    ]


def _resolve_verdict(scheduler_state: str, claim_state: str, tracker: TrackerClaim) -> str:
    """Resolves one allocation's verdict from its scheduler state and what its job's tracker recorded.

    Notes:
        A held allocation resolves as running whatever its tracker holds. The tracker of a job the scheduler is still
        carrying may be written at any moment, so a verdict read off it would be a verdict about the past.

        A tracker claiming to be running under an executor that names no scheduler allocation resolves as running too,
        rather than as stranded. Neither of the scheduler's records answers for such an executor, so calling that job
        stranded would reset a tracker whose own executor may still be writing to it. The refusal leaves the batch
        outstanding until a caller remediates it explicitly, and that remediation drops the ledger entry while leaving
        the tracker exactly as it stands.

    Args:
        scheduler_state: The state in which the scheduler's records place the recorded allocation.
        claim_state: The state in which they place the allocation the job's tracker claims, or empty when it claims
            none.
        tracker: What the job's own tracker recorded, which names both the status and the executor holding it.

    Returns:
        One of ``running``, ``finished``, ``failed``, ``abandoned``, or ``stranded``.
    """
    if HELD_ALLOCATION in (scheduler_state, claim_state):
        return RUNNING_ALLOCATION
    if tracker.status == ProcessingStatus.SUCCEEDED.name:
        return FINISHED_ALLOCATION
    if tracker.status == ProcessingStatus.FAILED.name:
        return FAILED_ALLOCATION
    if tracker.status == ProcessingStatus.RUNNING.name:
        return RUNNING_ALLOCATION if tracker.unqueryable_executor else STRANDED_ALLOCATION
    return ABANDONED_ALLOCATION


def _submit_ordered_jobs(
    server: Server,
    ordered: Sequence[GenericPendingJob],
    batch_directory: Path,
    walltime_minutes: int,
    submissions: list[RemoteSubmission],
    allocation_of_job: dict[tuple[str, str], str],
    *,
    verbose: bool,
) -> None:
    """Submits each ordered job, appending its record as the scheduler accepts it.

    Notes:
        Accumulates into the caller's list rather than returning one, so the caller still holds every accepted
        allocation when the scheduler rejects a later job.

    Args:
        server: The connected server that receives the submission.
        ordered: The jobs to submit, in dependency order.
        batch_directory: The server-side directory that holds the scripts and logs.
        walltime_minutes: The wall-time every allocation requests.
        submissions: The list that receives each accepted allocation's record.
        allocation_of_job: The mapping from each submitted job's dispatch key to its allocation identifier, which is
            what resolves a dependent job's dependency directive.
        verbose: Determines whether to report each submission as it is accepted.

    Raises:
        RuntimeError: If the scheduler rejects a submission.
    """
    for index, job in enumerate(ordered):
        slurm_job_name = _resolve_slurm_job_name(job=job, index=index)
        output_log = batch_directory.joinpath(f"{slurm_job_name}.out")
        error_log = batch_directory.joinpath(f"{slurm_job_name}.err")
        dependencies = [
            allocation_of_job[prerequisite]
            for prerequisite in job.prerequisite_keys
            if prerequisite in allocation_of_job
        ]

        allocation = Job(
            job_name=slurm_job_name,
            output_log=output_log,
            error_log=error_log,
            working_directory=batch_directory,
            conda_environment=server.environment,
            cpu_threads=job.core_weight,
            ram=max(1, ceil(job.resident_mb / _MEGABYTES_PER_GIGABYTE)),
            time=walltime_minutes,
            dependencies=dependencies,
        )
        allocation.add_command(command=shlex.join(resolve_job_command(job=job)))
        allocation = server.submit_job(job=allocation, verbose=verbose)

        # submit_job() raises rather than returning an unidentified job, so the identifier is always present here.
        slurm_job_id = str(allocation.job_id)
        allocation_of_job[job.dispatch_key] = slurm_job_id
        submissions.append(
            RemoteSubmission(
                job_id=job.job_id,
                slurm_job_id=slurm_job_id,
                slurm_job_name=slurm_job_name,
                pipeline=job.pipeline,
                job_name=job.job_name,
                specifier=job.specifier,
                unit_path=str(job.unit_path),
                unit_name=job.name,
                cores=job.core_weight,
                memory_mb=job.resident_mb,
                output_log=str(output_log),
                error_log=str(error_log),
            )
        )


def _resolve_slurm_job_name(job: GenericPendingJob, index: int) -> str:
    """Builds the name one allocation carries in the scheduler's queue and on its script and log files.

    Notes:
        Leads with the batch position, so two jobs whose names sanitize to the same text still write to separate
        files.

    Args:
        job: The pending job to name.
        index: The job's position in the submission order.

    Returns:
        The allocation name.
    """
    readable = "-".join(part for part in (job.name or job.unit_path.name, job.job_name, job.specifier) if part)
    return f"{index:04d}-{_SLURM_NAME_SANITIZER.sub(repl='_', string=readable)}"


def _regenerate_remote_state(server: Server, project_path: Path, datasets: Sequence[Path]) -> None:
    """Rewrites a remote project's state artifacts so a pull answers from a current snapshot.

    Notes:
        A generation that fails is reported as a warning rather than raised, since the artifacts it would have
        refreshed may still be worth pulling.

    Args:
        server: The connected server holding the project.
        project_path: The path to the project's root directory on the server.
        datasets: The project's dataset directories.
    """
    commands = [["slf", "manifest", "-pp", str(project_path), "create"]]
    if datasets:
        commands.append(
            ["slf", "dataset-state", *[argument for dataset in datasets for argument in ("-dp", str(dataset))]]
        )

    for command in commands:
        result = server.execute_command(command=environment_command(environment=server.environment, command=command))
        if result.return_code != 0:
            console.echo(
                message=(
                    f"Unable to regenerate the remote state of project '{project_path.name}' with "
                    f"'{shlex.join(command)}'. The mirrored artifacts may be out of date. {result.stderr.strip()}"
                ),
                level=LogLevel.WARNING,
            )
