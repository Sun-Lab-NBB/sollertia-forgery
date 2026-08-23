"""Provides the job descriptors both execution backends dispatch, alongside the graph algorithms that order them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path
from dataclasses import field, dataclass

from ataraxis_data_structures import ProcessingStatus

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(slots=True)
class PendingJob:
    """Describes a single batch processing job tracked by a ``ProcessingTracker`` file.

    Notes:
        Subclasses extend this dataclass with the additional fields their worker callables need. The base fields carry
        everything the shared graph and the shared execution manager need. That is the unit and job a record names, the
        tracker it is recorded on, the cores and memory it occupies, and the jobs it waits for.
    """

    tracker_path: Path
    """The path to the ``ProcessingTracker`` YAML file that tracks this job."""
    job_id: str
    """The unique hexadecimal identifier for this job in the tracker."""
    unit_path: Path = field(default_factory=Path)
    """The path to the processing unit this job operates on, which is the session root for a session job and the
    dataset root for a dataset job. This is what scopes a job identifier to one unit."""
    job_name: str = ""
    """The pipeline job type name registered in the ``ProcessingTracker``, which is what groups this job with the
    others of its type for the concurrency terms admission enforces."""
    core_weight: int = 1
    """The cores this job occupies while it runs, carried over from the plan artifact's per-job cores figure. Sizing
    is per job rather than per type, so two jobs of one type legitimately differ here, and dispatch only caps this
    width at what the executing host can supply."""
    memory_mb: int = 0
    """The memory this job occupies while it runs, estimated from the data it will process."""
    prerequisite_ids: tuple[str, ...] = ()
    """The identifiers of the jobs that must succeed before this job may be dispatched. Resolved from the pipeline's
    own job ordering, and empty for a job that depends on nothing."""

    @property
    def dispatch_key(self) -> tuple[str, str]:
        """Returns the composite key that uniquely identifies this job across the entire batch, combining the unit it
        operates on with the job identifier.
        """
        return str(self.unit_path), self.job_id

    @property
    def prerequisite_keys(self) -> tuple[tuple[str, str], ...]:
        """Returns the dispatch keys of this job's upstream jobs.

        Notes:
            A job identifier is derived from the job name and specifier alone, so the same stage of two different
            units shares one identifier. Pairing each identifier with this job's unit keeps a batch spanning many
            units from treating one unit's completed stage as every unit's.
        """
        return tuple((str(self.unit_path), prerequisite) for prerequisite in self.prerequisite_ids)


@dataclass(slots=True)
class GenericPendingJob(PendingJob):
    """Describes a single batch processing job for the system-agnostic processing tools.

    Notes:
        Extends the shared ``PendingJob`` base with the descriptor set every registered pipeline worker needs, so one
        descriptor serves every pipeline. The shared worker routes on ``pipeline`` and each session pipeline's worker
        reads ``unit_path`` and ``job_id``. Fields a pipeline does not use stay at their defaults, and a descriptor
        missing a field the engine requires is rejected before dispatch.
    """

    pipeline: str = ""
    """The pipeline this job belongs to, which the shared worker routes on so one pool serves every pipeline."""
    name: str = ""
    """The human-readable unit name used for logging and status reporting."""
    specifier: str = ""
    """The specifier that differentiates jobs of the same type within one unit, such as a camera or controller source
    identifier, a controller-module triple, or a plane index."""
    project_root: Path | None = None
    """The project root directory, which stays unset while every worker resolves its output location from
    ``unit_path``. A pipeline whose worker needs the root above that path reads it from here."""
    status: str = ""
    """The status this job's tracker recorded when preparation last regenerated the host's state artifact, named by its
    ``ProcessingStatus`` member. Empty for a descriptor built without one."""
    executor_id: str = ""
    """The executor the same record named, which is a scheme-tagged identifier such as a scheduler allocation. Carried
    so reconciliation resolves a running job's allocation from the batch rather than from a tracker it would have to
    open, which on a remote host would mean reading one across the transport mid-run."""
    options: dict[str, Any] = field(default_factory=dict)
    """The pipeline-specific parameters the caller chose for this job, such as the mode a multi-mode pipeline runs in.
    The mapping is carried through to the pipeline's own worker, which interprets whichever keys it declares. A
    pipeline that takes no parameters leaves it empty."""


@dataclass(frozen=True, slots=True)
class BatchDocument:
    """Describes one prepared batch, which is what both the local and the remote backend dispatch.

    Notes:
        Preparation produces this from a project's own plan and state artifacts, so a batch is the same document
        whichever host holds the data. The local pool admits ``jobs`` against its budgets and the remote backend
        submits them as a dependency graph, and neither resolves the batch's membership for itself.
    """

    pipeline: str = ""
    """The pipeline this batch dispatches."""
    host: str = ""
    """The host this batch was prepared against, which is the one holding the data its jobs read. Execution reads the
    host from here, so a batch runs where it was prepared."""
    options: dict[str, Any] = field(default_factory=dict)
    """The pipeline-specific parameters every job of this batch runs with."""
    units: list[dict[str, Any]] = field(default_factory=list)
    """One entry per named unit, carrying its path, its name, and its job counts, or the reason it contributed none."""
    jobs: list[dict[str, Any]] = field(default_factory=list)
    """The dispatchable job descriptors, holding every job this run can carry out."""
    blocked_jobs: list[dict[str, Any]] = field(default_factory=list)
    """The jobs this run can neither dispatch nor find already succeeded, each naming what it waits on."""


def build_batch_document(
    pipeline: str,
    host: str,
    unit_column: str,
    plan_rows: list[dict[str, Any]],
    state_rows: list[dict[str, Any]],
    unit_paths: Sequence[Path],
    options: dict[str, Any],
    tracker_paths: dict[str, str] | None = None,
) -> BatchDocument:
    """Joins a project's plan and state rows into the batch one run dispatches.

    Notes:
        The state rows are the pipelines' own tracker registries, so a job absent from them is one the unit cannot
        produce and is never dispatched. A job the state records as succeeded is left out, which is what makes
        preparing a unit twice queue only the work still outstanding.

        A recorded prerequisite naming a job with no state row is dropped rather than treated as unsatisfied, since
        the unit can never produce it. A prerequisite this run neither dispatches nor finds already succeeded blocks
        its dependents instead, and blocking propagates.

    Args:
        pipeline: The pipeline being prepared.
        host: The name of the host the artifacts were materialized and read from.
        unit_column: The state and plan column naming each row's unit, which is ``dataset`` for a dataset pipeline and
            ``session`` for a session pipeline.
        plan_rows: The project's plan rows, carrying each job's cores, memory, and recorded ordering.
        state_rows: The project's state rows, carrying each tracked job's recorded status.
        unit_paths: The processing unit directories the batch covers.
        options: The pipeline-specific parameters to stamp onto every descriptor.
        tracker_paths: Where each unit's tracker sits, keyed by unit path, for a backend that opens it directly.
            Passing nothing leaves every descriptor's tracker location empty, which is correct for a backend whose
            jobs record their own outcomes on another host.

    Returns:
        The prepared batch document.
    """
    trackers = tracker_paths if tracker_paths is not None else {}
    planned = index_rows_by_unit(rows=[row for row in plan_rows if row.get("pipeline") == pipeline], key=unit_column)
    recorded = index_rows_by_unit(
        rows=[row for row in state_rows if row.get("pipeline", pipeline) == pipeline], key=unit_column
    )

    units: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for unit_path in unit_paths:
        unit_name = unit_path.name
        unit_state = recorded.get(unit_name, {})
        unit_plan = planned.get(unit_name, {})
        if not unit_state:
            units.append(_unresolved_unit(unit_path=unit_path, reason=_no_state_reason(pipeline=pipeline)))
            continue

        # A state table records each job's status by the name of its ``ProcessingStatus`` member, so the comparison
        # reads that name rather than a literal restating it.
        succeeded = {job_id for job_id, row in unit_state.items() if row["status"] == ProcessingStatus.SUCCEEDED.name}
        unplanned = sorted(job_id for job_id in unit_state if job_id not in unit_plan and job_id not in succeeded)
        if unplanned:
            units.append(_unresolved_unit(unit_path=unit_path, reason=_unplanned_reason(job_ids=unplanned)))
            continue

        outstanding = [
            build_job_descriptor(
                state_row=row,
                plan_row=unit_plan[job_id],
                unit_path=unit_path,
                unit_name=unit_name,
                pipeline=pipeline,
                options=options,
                tracker_path=trackers.get(str(unit_path), ""),
                # Scopes each recorded edge to the jobs this unit actually tracks, which drops the edges naming a
                # stage the unit can never produce.
                trackable_ids=set(unit_state),
            )
            for job_id, row in unit_state.items()
            if job_id not in succeeded
        ]
        submittable, unit_blocked = partition_blocked_jobs(jobs=outstanding, succeeded=succeeded)
        jobs.extend(submittable)
        blocked.extend(unit_blocked)
        units.append(
            {
                "unit_path": str(unit_path),
                "unit_name": unit_name,
                "job_count": len(submittable),
                "blocked_count": len(unit_blocked),
            }
        )

    return BatchDocument(
        pipeline=pipeline, host=host, options=dict(options), units=units, jobs=jobs, blocked_jobs=blocked
    )


def build_job_descriptor(
    state_row: dict[str, Any],
    plan_row: dict[str, Any],
    unit_path: Path,
    unit_name: str,
    pipeline: str,
    options: dict[str, Any],
    trackable_ids: set[str],
    tracker_path: str = "",
) -> dict[str, Any]:
    """Renders one job as the descriptor both backends dispatch.

    Args:
        state_row: The job's row in the state table, carrying the status and the executor its tracker recorded.
        plan_row: The job's row in the plan table.
        unit_path: The path to the unit the job operates on.
        unit_name: The name of that unit.
        pipeline: The pipeline the job belongs to.
        options: The pipeline-specific parameters to run the job with.
        trackable_ids: The job identifiers the unit tracks, which the recorded ordering is narrowed to.
        tracker_path: Where the unit's tracker sits, or empty when this backend never opens it.

    Returns:
        The job descriptor.
    """
    return {
        "job_id": state_row["job_id"],
        "job_name": state_row["job_name"],
        "specifier": state_row["specifier"] or "",
        # Carried from the state artifact preparation regenerated on the host, so reconciliation reads what a job's
        # tracker recorded without opening that tracker while the batch runs.
        "status": state_row["status"],
        "executor_id": state_row.get("executor_id") or "",
        "unit_path": str(unit_path),
        "unit_name": unit_name,
        "pipeline": pipeline,
        "tracker_path": tracker_path,
        "cores": int(plan_row["cores"]),
        "memory_mb": int(plan_row["memory_mb"]),
        "prerequisite_ids": [
            prerequisite for prerequisite in (plan_row["prerequisite_ids"] or []) if prerequisite in trackable_ids
        ],
        "options": dict(options),
    }


def index_rows_by_unit(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, dict[str, Any]]]:
    """Indexes table rows by their unit and then by their job identifier.

    Args:
        rows: The rows to index.
        key: The column naming each row's unit.

    Returns:
        The rows, keyed by unit name and then by job identifier.
    """
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        unit = row.get(key)
        if unit is None:
            continue
        indexed.setdefault(str(unit), {})[row["job_id"]] = row
    return indexed


def partition_blocked_jobs(
    jobs: list[dict[str, Any]], succeeded: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Splits a unit's outstanding jobs into the ones this run may dispatch and the ones it may not.

    Notes:
        A prerequisite that neither runs in this batch nor already succeeded is one this run cannot produce. Blocking
        propagates, so a stage waiting on a blocked stage is blocked in turn.

    Args:
        jobs: The unit's outstanding job descriptors.
        succeeded: The identifiers of the unit's jobs already recorded as succeeded.

    Returns:
        A tuple of the dispatchable descriptors and the blocked entries, each naming the prerequisites it waits on.
    """
    runnable: dict[str, dict[str, Any]] = {descriptor["job_id"]: descriptor for descriptor in jobs}
    blocked: dict[str, list[str]] = {}
    while True:
        newly_blocked = {
            job_id: unsatisfied
            for job_id, descriptor in runnable.items()
            if job_id not in blocked
            and (
                unsatisfied := [
                    prerequisite
                    for prerequisite in descriptor.get("prerequisite_ids", ())
                    if prerequisite not in succeeded and (prerequisite not in runnable or prerequisite in blocked)
                ]
            )
        }
        if not newly_blocked:
            break
        blocked.update(newly_blocked)

    dispatchable = [descriptor for descriptor in jobs if descriptor["job_id"] not in blocked]
    blocked_entries = [
        {
            "job_id": descriptor["job_id"],
            "job_name": descriptor["job_name"],
            "specifier": descriptor["specifier"],
            "pipeline": descriptor["pipeline"],
            "unit_path": descriptor["unit_path"],
            "unit_name": descriptor["unit_name"],
            "unsatisfied_prerequisite_ids": blocked[descriptor["job_id"]],
        }
        for descriptor in jobs
        if descriptor["job_id"] in blocked
    ]
    return dispatchable, blocked_entries


def resolve_dispatch_priorities[PendingJobT: PendingJob](
    jobs: dict[tuple[str, str], PendingJobT],
) -> dict[tuple[str, str], int]:
    """Resolves how much queued work waits on each job, which is the weight admission uses to order candidates.

    Notes:
        A job's priority is the cores committed by every job that cannot run until it succeeds, summed over its
        transitive dependents. Weighing the dependents by their cores rather than counting them separates a job
        holding back three wide stages from one holding back a single narrow stage. A job nothing waits on weighs
        zero, whatever its own size.

        Ordering by this weight is what keeps a batch working on its critical path. Admitting by size alone lets a
        crowd of leaf jobs hold the budget while the root of a long chain waits, which idles the host once those
        leaves finish and the chain has yet to start. The dependents are collected as a set, so a stage reachable
        along several paths at once is counted a single time.

        A prerequisite naming a job outside this batch is skipped, since a job the batch does not hold cannot be
        ordered against the ones it does. Cyclic prerequisites resolve to a finite weight rather than recursing
        without end, which leaves a malformed pipeline ordering poorly instead of stalling the batch.

    Args:
        jobs: Every job the batch holds, keyed by dispatch key.

    Returns:
        A dictionary mapping each job's dispatch key to the cores its transitive dependents commit.
    """
    dependents: dict[tuple[str, str], list[tuple[str, str]]] = {key: [] for key in jobs}
    for key, job in jobs.items():
        for prerequisite in job.prerequisite_keys:
            if prerequisite in dependents:
                dependents[prerequisite].append(key)

    resolved: dict[tuple[str, str], frozenset[tuple[str, str]]] = {}

    def _collect(key: tuple[str, str], visiting: set[tuple[str, str]]) -> frozenset[tuple[str, str]]:
        """Gathers every job reachable downstream of the given dispatch key, memoizing each resolved set."""
        cached = resolved.get(key)
        if cached is not None:
            return cached
        if key in visiting:
            return frozenset()
        visiting.add(key)
        reachable: set[tuple[str, str]] = set()
        for dependent in dependents[key]:
            reachable.add(dependent)
            reachable |= _collect(key=dependent, visiting=visiting)
        visiting.discard(key)
        resolved[key] = frozenset(reachable)
        return resolved[key]

    return {key: sum(jobs[dependent].core_weight for dependent in _collect(key=key, visiting=set())) for key in jobs}


def resolve_submission_order[PendingJobT: PendingJob](jobs: Sequence[PendingJobT]) -> list[PendingJobT]:
    """Orders a batch's jobs so every job follows the jobs it waits on.

    Notes:
        A scheduler names a dependency by the identifier it assigned to the upstream allocation, so an upstream job
        must already be submitted before its dependents are.

        A prerequisite outside this batch contributes no depth, and a cyclic ordering resolves to a finite depth, so a
        malformed pipeline is ordered poorly rather than stalling the submission.

    Args:
        jobs: The batch's pending jobs.

    Returns:
        The jobs, ordered by dependency depth and then by their position in the input.
    """
    by_key = {job.dispatch_key: job for job in jobs}
    depths: dict[tuple[str, str], int] = {}

    def _depth(key: tuple[str, str], visiting: set[tuple[str, str]]) -> int:
        """Resolves how many in-batch prerequisites the given dispatch key sits behind, memoizing each depth."""
        cached = depths.get(key)
        if cached is not None:
            return cached
        if key in visiting:
            return 0
        visiting.add(key)
        resolved = max(
            (
                _depth(key=prerequisite, visiting=visiting) + 1
                for prerequisite in by_key[key].prerequisite_keys
                if prerequisite in by_key
            ),
            default=0,
        )
        visiting.discard(key)
        depths[key] = resolved
        return resolved

    order = sorted(
        range(len(jobs)),
        key=lambda position: (_depth(key=jobs[position].dispatch_key, visiting=set()), position),
    )
    return [jobs[position] for position in order]


def build_pending_job(job: dict[str, Any]) -> GenericPendingJob:
    """Builds one job descriptor into the pending job both backends dispatch.

    Args:
        job: A job descriptor carrying ``job_id``, ``unit_path``, ``cores``, and ``memory_mb``, and optionally
            ``tracker_path``, ``job_name``, ``unit_name``, ``specifier``, ``pipeline``, ``prerequisite_ids``,
            ``options``, ``status``, and ``executor_id``.

    Returns:
        The pending job.

    Raises:
        KeyError: If the descriptor omits a field the engine requires.
    """
    return GenericPendingJob(
        tracker_path=Path(job.get("tracker_path", "")),
        job_id=job["job_id"],
        unit_path=Path(job["unit_path"]),
        job_name=job.get("job_name", ""),
        name=job.get("unit_name", ""),
        specifier=job.get("specifier", ""),
        pipeline=job.get("pipeline", ""),
        core_weight=int(job["cores"]),
        memory_mb=int(job["memory_mb"]),
        prerequisite_ids=tuple(job.get("prerequisite_ids", ())),
        options=dict(job.get("options") or {}),
        status=job.get("status") or "",
        executor_id=job.get("executor_id") or "",
    )


def _unresolved_unit(unit_path: Path, reason: str) -> dict[str, Any]:
    """Renders a unit that contributed no job, carrying the reason it contributed none.

    Args:
        unit_path: The path to the unit.
        reason: Why the unit contributed no job.

    Returns:
        The unit entry.
    """
    return {"unit_path": str(unit_path), "unit_name": unit_path.name, "error": reason, "job_count": 0}


def _no_state_reason(pipeline: str) -> str:
    """Builds the reason reported for a unit whose state table records no job of the prepared pipeline.

    Args:
        pipeline: The pipeline that was prepared.

    Returns:
        The reason to record on the unit's entry.
    """
    return (
        f"The project's state table records no '{pipeline}' job for this unit, so the unit carries none of the data "
        f"that pipeline consumes."
    )


def _unplanned_reason(job_ids: list[str]) -> str:
    """Builds the reason reported for a unit whose outstanding jobs carry no planned figures.

    Args:
        job_ids: The identifiers of the outstanding jobs that carry no planned figures.

    Returns:
        The reason to record on the unit's entry.
    """
    return (
        f"The project's plan table carries no figures for job(s) {job_ids}. Every outstanding job must be planned "
        f"before it can be sized for a host or a scheduler."
    )
