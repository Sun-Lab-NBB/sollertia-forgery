"""Provides the remote execution backend that runs prepared jobs as SLURM allocations on the compute server."""

from __future__ import annotations

import re
from math import ceil
import shlex
from typing import TYPE_CHECKING, Any
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import asdict

import polars as pl
from ataraxis_base_utilities import LogLevel, console

from .local import GenericPendingJob
from .ledger import (
    SubmissionBatch,
    RemoteSubmission,
    record_batch,
    current_timestamp,
    retire_settled_batches,
)
from ..server import Job, Server, JobStatus, get_server_configuration
from ..forging import DATASET_STATE_FILENAME, DATASET_MARKER_FILENAME
from .dispatch import resolve_dispatch, resolve_job_command
from .planning import project_plan_path
from ..managing import project_jobs_path, project_manifest_path
from ..shared_assets import ProcessingPipelines

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .dispatch import PipelineDispatch

REMOTE_JOB_WALLTIME_MINUTES: int = 480
"""The wall-time every remote allocation requests, in minutes.

Notes:
    One figure covers every job type, because this bound exists to stop a run that has stopped making progress rather
    than to describe how long a stage takes.
"""

BATCH_DIRECTORY_NAME: str = "processing_batches"
"""The directory under the server's data root that holds one subdirectory per submitted batch."""

_MEGABYTES_PER_GIGABYTE: int = 1024
"""The divisor converting an estimate in megabytes into the gigabyte figure a SLURM memory request takes."""

_SLURM_NAME_SANITIZER: re.Pattern[str] = re.compile(r"[^A-Za-z0-9._-]+")
"""Matches the characters a SLURM job name and its script filename should not carry."""

_SUCCEEDED_STATUS: str = "SUCCEEDED"
"""The state a job's recorded status carries once it has completed successfully, as the state tables spell it."""


def remote_batch_directory(server: Server, batch_id: str) -> Path:
    """Resolves the server-side directory holding one batch's job scripts and logs.

    Args:
        server: The server the batch runs on.
        batch_id: The identifier of the batch.

    Returns:
        The path to the batch's directory on the server.
    """
    return server.root.joinpath(BATCH_DIRECTORY_NAME, batch_id)


def prepare_remote_batch(
    server: Server, pipeline: str, unit_paths: Sequence[str], options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Resolves a pipeline's submittable jobs for the named units out of the project's own artifacts.

    Notes:
        A job the unit cannot run never reaches its processing tracker, so its absence from the state table is what
        rules it out.

        A job whose upstream stage neither runs in this batch nor already succeeded is reported as blocked rather
        than submitted.

    Args:
        server: The connected server holding the units.
        pipeline: The batch pipeline to prepare.
        unit_paths: The processing unit directories on the server to prepare jobs for.
        options: The pipeline-specific parameters to run the prepared jobs with.

    Returns:
        The batch document, carrying the ``pipeline``, the per-unit ``units`` entries, the submittable ``jobs``
        descriptors, and the ``blocked_jobs`` entries naming what each blocked job waits on.

    Raises:
        ValueError: If the named pipeline is not a supported batch pipeline, or if the named units do not share one
            project.
        RuntimeError: If a server-side command fails.
    """
    dispatch = resolve_dispatch(pipeline=pipeline)
    if dispatch is None:
        message = (
            f"Unable to prepare a remote batch for pipeline '{pipeline}', which is not a supported batch pipeline."
        )
        console.error(message=message, error=ValueError)

    units = [Path(unit_path) for unit_path in unit_paths]
    project_root = _resolve_project_root(dispatch=dispatch, unit_paths=units)
    _refresh_remote_artifacts(server=server, dispatch=dispatch, project_root=project_root, unit_paths=units)

    with TemporaryDirectory() as staging_directory:
        plan, state = _pull_remote_artifacts(
            server=server,
            dispatch=dispatch,
            project_root=project_root,
            unit_paths=units,
            staging_directory=Path(staging_directory),
        )

    return _build_batch_document(
        dispatch=dispatch, plan=plan, state=state, unit_paths=units, options=dict(options or {})
    )


def _resolve_project_root(dispatch: PipelineDispatch[Any], unit_paths: Sequence[Path]) -> Path:
    """Resolves the project the named units belong to.

    Notes:
        A dataset sits directly under its project root while a session sits under its animal, which sets the two
        depths.

    Args:
        dispatch: The pipeline's dispatch entry.
        unit_paths: The processing unit directories to resolve the project of.

    Returns:
        The path to the project root on the server.

    Raises:
        ValueError: If no unit is named, or if the named units span more than one project.
    """
    if not unit_paths:
        message = "Unable to prepare a remote batch. No processing unit was named."
        console.error(message=message, error=ValueError)

    depth = 1 if dispatch.pipeline is ProcessingPipelines.FORGING else 2
    roots = {unit_path.parents[depth - 1] for unit_path in unit_paths}
    if len(roots) > 1:
        message = (
            f"Unable to prepare a remote batch spanning the projects {sorted(str(root) for root in roots)}. The plan "
            f"and state tables a batch is resolved from are written per project, so every unit of one batch must "
            f"belong to the same project."
        )
        console.error(message=message, error=ValueError)
    return roots.pop()


def _refresh_remote_artifacts(
    server: Server, dispatch: PipelineDispatch[Any], project_root: Path, unit_paths: Sequence[Path]
) -> None:
    """Rewrites the project artifacts a batch is resolved from, on the host that holds the data.

    Notes:
        Planning must run first, since it registers a unit's runnable jobs on its processing tracker and state
        generation reads those trackers.

    Args:
        server: The connected server holding the units.
        dispatch: The pipeline's dispatch entry.
        project_root: The path to the project root on the server.
        unit_paths: The processing unit directories the batch covers.

    Raises:
        RuntimeError: If a server-side command fails.
    """
    named = [str(unit_path) for unit_path in unit_paths]
    if dispatch.pipeline is ProcessingPipelines.FORGING:
        commands = [
            ["slf", "plan", "dataset", *_repeated(flag="-dp", values=named)],
            ["slf", "plan", "project", "-pp", str(project_root)],
            ["slf", "dataset-state", *_repeated(flag="-dp", values=named)],
        ]
    else:
        commands = [
            ["slf", "plan", "session", *_repeated(flag="-sp", values=named)],
            ["slf", "plan", "project", "-pp", str(project_root)],
            ["slf", "manifest", "-pp", str(project_root), "create"],
        ]

    for command in commands:
        result = server.execute_command(command=environment_command(environment=server.environment, command=command))
        if result.return_code != 0:
            message = (
                f"Unable to prepare the '{dispatch.pipeline.value}' batch. The server-side command "
                f"'{shlex.join(command)}' exited with code {result.return_code}. {result.stderr.strip()}"
            )
            console.error(message=message, error=RuntimeError)


def _pull_remote_artifacts(
    server: Server,
    dispatch: PipelineDispatch[Any],
    project_root: Path,
    unit_paths: Sequence[Path],
    staging_directory: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Copies the plan and state tables home and reads them.

    Args:
        server: The connected server holding the artifacts.
        dispatch: The pipeline's dispatch entry.
        project_root: The path to the project root on the server.
        unit_paths: The processing unit directories the batch covers.
        staging_directory: The local directory the tables are copied into.

    Returns:
        A tuple of the project plan table and the job state table.

    Raises:
        FileNotFoundError: If the server holds no plan table for the project.
    """
    plan_path = project_plan_path(project_directory=project_root)
    if not server.exists(remote_path=plan_path):
        message = (
            f"Unable to prepare the batch. The remote compute server holds no plan table at '{plan_path}', so the "
            f"planning step wrote nothing for this project."
        )
        console.error(message=message, error=FileNotFoundError)

    local_plan = staging_directory.joinpath(plan_path.name)
    server.pull(local_path=local_plan, remote_path=plan_path)

    if dispatch.pipeline is ProcessingPipelines.FORGING:
        remote_state = [unit_path.joinpath(DATASET_STATE_FILENAME) for unit_path in unit_paths]
    else:
        remote_state = [project_jobs_path(project_directory=project_root)]

    frames: list[pl.DataFrame] = []
    for index, remote_path in enumerate(remote_state):
        if not server.exists(remote_path=remote_path):
            continue
        local_path = staging_directory.joinpath(f"{index:04d}_{remote_path.name}")
        server.pull(local_path=local_path, remote_path=remote_path)
        frames.append(pl.read_ipc(source=local_path, memory_map=False))

    plan = pl.read_ipc(source=local_plan, memory_map=False)
    return plan, pl.concat(frames) if frames else pl.DataFrame()


def _build_batch_document(
    dispatch: PipelineDispatch[Any],
    plan: pl.DataFrame,
    state: pl.DataFrame,
    unit_paths: Sequence[Path],
    options: dict[str, Any],
) -> dict[str, Any]:
    """Joins the plan and state tables into the job descriptors a submission dispatches.

    Args:
        dispatch: The pipeline's dispatch entry.
        plan: The project plan table.
        state: The job state table.
        unit_paths: The processing unit directories the batch covers.
        options: The pipeline-specific parameters to stamp onto every descriptor.

    Returns:
        The batch document.
    """
    unit_column = "dataset" if dispatch.pipeline is ProcessingPipelines.FORGING else "session"
    pipeline_value = dispatch.pipeline.value

    planned = _index_by_unit(frame=plan.filter(pl.col("pipeline") == pipeline_value), unit_column=unit_column)
    recorded = _index_by_unit(
        frame=state if "pipeline" not in state.columns else state.filter(pl.col("pipeline") == pipeline_value),
        unit_column=unit_column,
    )

    units: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for unit_path in unit_paths:
        unit_name = unit_path.name
        unit_state = recorded.get(unit_name, {})
        unit_plan = planned.get(unit_name, {})
        if not unit_state:
            units.append(
                {
                    "unit_path": str(unit_path),
                    "unit_name": unit_name,
                    "error": (
                        f"The project's state table records no '{pipeline_value}' job for this unit, so the unit "
                        f"carries none of the data that pipeline consumes."
                    ),
                    "job_count": 0,
                }
            )
            continue

        succeeded = {job_id for job_id, row in unit_state.items() if row["status"] == _SUCCEEDED_STATUS}
        unplanned = sorted(job_id for job_id in unit_state if job_id not in unit_plan and job_id not in succeeded)
        if unplanned:
            units.append(
                {
                    "unit_path": str(unit_path),
                    "unit_name": unit_name,
                    "error": (
                        f"The project's plan table carries no figures for job(s) {unplanned}. Every outstanding job "
                        f"must be planned before it can be sized for a scheduler."
                    ),
                    "job_count": 0,
                }
            )
            continue

        outstanding = [
            _descriptor(
                state_row=row,
                plan_row=unit_plan[job_id],
                unit_path=unit_path,
                unit_name=unit_name,
                pipeline=pipeline_value,
                options=options,
            )
            for job_id, row in unit_state.items()
            if job_id not in succeeded
        ]
        submittable, unit_blocked = _partition_blocked_jobs(jobs=outstanding, succeeded=succeeded)
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

    return {
        "pipeline": pipeline_value,
        "options": dict(options),
        "units": units,
        "jobs": jobs,
        "blocked_jobs": blocked,
    }


def _descriptor(
    state_row: dict[str, Any],
    plan_row: dict[str, Any],
    unit_path: Path,
    unit_name: str,
    pipeline: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    """Renders one job as the descriptor a submission dispatches.

    Args:
        state_row: The job's row in the state table.
        plan_row: The job's row in the plan table.
        unit_path: The path, on the server, to the unit the job operates on.
        unit_name: The name of that unit.
        pipeline: The pipeline the job belongs to.
        options: The pipeline-specific parameters to run the job with.

    Returns:
        The job descriptor.
    """
    return {
        "job_id": state_row["job_id"],
        "job_name": state_row["job_name"],
        "specifier": state_row["specifier"] or "",
        "unit_path": str(unit_path),
        "unit_name": unit_name,
        "pipeline": pipeline,
        "cores": int(plan_row["cores"]),
        "memory_mb": int(plan_row["memory_mb"]),
        "memory_modeled": bool(plan_row["memory_modeled"]),
        "prerequisite_ids": list(plan_row["prerequisite_ids"] or []),
        "options": dict(options),
    }


def _index_by_unit(frame: pl.DataFrame, unit_column: str) -> dict[str, dict[str, dict[str, Any]]]:
    """Indexes a table's rows by their unit and then by their job identifier.

    Notes:
        A job identifier is derived from the job name and specifier alone, so the same stage of two different units
        shares one identifier.

    Args:
        frame: The table to index.
        unit_column: The column naming each row's unit.

    Returns:
        The rows, keyed by unit name and then by job identifier.
    """
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    if unit_column not in frame.columns:
        return indexed
    for row in frame.iter_rows(named=True):
        unit = row[unit_column]
        if unit is None:
            continue
        indexed.setdefault(str(unit), {})[row["job_id"]] = row
    return indexed


def _repeated(flag: str, values: Sequence[str]) -> list[str]:
    """Expands one repeated command-line option over every value it is given.

    Args:
        flag: The option flag to repeat.
        values: The values to pass.

    Returns:
        The flattened argument list.
    """
    return [argument for value in values for argument in (flag, value)]


def submit_batch(
    server: Server,
    jobs: Sequence[dict[str, Any]],
    batch_id: str,
    *,
    walltime_minutes: int = REMOTE_JOB_WALLTIME_MINUTES,
    verbose: bool = False,
) -> list[RemoteSubmission]:
    """Submits a prepared batch to the scheduler as a dependency graph.

    Notes:
        The scheduler sequences the graph itself, so this process may exit as soon as the last job is queued.

        Every accepted allocation is recorded in the submission ledger, including when the scheduler rejects a later
        job of the same batch, since the allocations it already accepted stay queued.

        The concurrency ceilings the local engine applies do not reach the scheduler. Expressing one natively needs a
        job array, whose tasks share a single memory request.

    Args:
        server: The connected server to submit to.
        jobs: The job descriptors to submit.
        batch_id: The identifier of the batch, which names the directory the scripts and logs are written into.
        walltime_minutes: The wall-time every allocation requests.
        verbose: Determines whether to report each submission as it is accepted.

    Returns:
        The submissions, in the order they were accepted.

    Raises:
        RuntimeError: If the scheduler rejects a submission.
    """
    batch_directory = remote_batch_directory(server=server, batch_id=batch_id)
    server.create(remote_path=batch_directory, is_dir=True, parents=True)

    pending = [_pending_job(descriptor=descriptor) for descriptor in jobs]
    ordered = _resolve_submission_order(jobs=pending)

    submissions: list[RemoteSubmission] = []
    allocation_of_job: dict[tuple[str, str], str] = {}
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
                    batch_directory=str(batch_directory),
                    submitted_at=current_timestamp(),
                    walltime_minutes=walltime_minutes,
                    submissions=list(submissions),
                )
            )

    return submissions


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
        server: The connected server to submit to.
        ordered: The jobs to submit, in dependency order.
        batch_directory: The server-side directory the scripts and logs are written into.
        walltime_minutes: The wall-time every allocation requests.
        submissions: The list each accepted allocation's record is appended to.
        allocation_of_job: The mapping from each submitted job's dispatch key to its allocation identifier, which is
            what a dependent job's dependency directive is resolved from.
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
            for prerequisite in _prerequisite_keys(job=job)
            if prerequisite in allocation_of_job
        ]

        allocation = Job(
            job_name=slurm_job_name,
            output_log=output_log,
            error_log=error_log,
            working_directory=batch_directory,
            conda_environment=server.environment,
            cpu_threads=job.core_weight,
            ram=max(1, ceil(job.memory_mb / _MEGABYTES_PER_GIGABYTE)),
            time=walltime_minutes,
            dependencies=dependencies,
        )
        allocation.add_command(shlex.join(resolve_job_command(job=job)))
        allocation = server.submit_job(job=allocation, verbose=verbose)

        # submit_job() raises rather than returning an unidentified job, so the identifier is always present here.
        slurm_job_id = str(allocation.job_id)
        allocation_of_job[_job_key(job=job)] = slurm_job_id
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
                memory_mb=job.memory_mb,
                output_log=str(output_log),
                error_log=str(error_log),
            )
        )


def query_submissions(server: Server, submissions: Sequence[RemoteSubmission]) -> dict[str, JobStatus]:
    """Queries the scheduler for the state of every submitted allocation and records what it observed.

    Notes:
        A batch this query observes as wholly finished is retired from the submission ledger.

    Args:
        server: The connected server the batch runs on.
        submissions: The submissions to query.

    Returns:
        A dictionary mapping each submission's allocation identifier to its scheduler state.
    """
    statuses = server.get_job_statuses(slurm_job_ids=[submission.slurm_job_id for submission in submissions])
    retire_settled_batches(statuses=statuses)
    return statuses


def cancel_submissions(server: Server, submissions: Sequence[RemoteSubmission]) -> list[str]:
    """Cancels every allocation the given submissions hold, leaving the ones that already finished untouched.

    Args:
        server: The connected server the batch runs on.
        submissions: The submissions to cancel.

    Returns:
        The allocation identifiers the cancellation named.
    """
    allocations = [submission.slurm_job_id for submission in submissions]
    server.abort_jobs(slurm_job_ids=allocations)
    return allocations


def sync_project_state(server: Server, project: str, local_directory: Path, *, regenerate: bool = True) -> list[Path]:
    """Regenerates a remote project's state artifacts and mirrors them onto this host.

    Notes:
        Regeneration precedes the pull, so the mirrored tables describe the state after the runs rather than before
        them.

    Args:
        server: The connected server holding the project.
        project: The name of the project whose state to mirror.
        local_directory: The local directory to mirror the artifacts into.
        regenerate: Determines whether to regenerate the artifacts on the server before pulling them.

    Returns:
        The local paths the artifacts were written to, holding one entry per artifact the server carried.

    Raises:
        FileNotFoundError: If the server holds no directory for the named project.
    """
    project_path = server.root.joinpath(project)
    if not server.is_directory(remote_path=project_path):
        message = (
            f"Unable to mirror the state of project '{project}'. The remote compute server holds no directory at "
            f"'{project_path}'."
        )
        console.error(message=message, error=FileNotFoundError)

    datasets = _discover_remote_datasets(server=server, project_path=project_path)
    if regenerate:
        _regenerate_remote_state(server=server, project_path=project_path, datasets=datasets)

    remote_artifacts = [
        project_manifest_path(project_directory=project_path),
        project_jobs_path(project_directory=project_path),
        project_plan_path(project_directory=project_path),
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


def environment_command(environment: str, command: Sequence[str]) -> str:
    """Wraps a command so it runs inside the server's shared processing environment.

    Args:
        environment: The name of the conda environment to activate.
        command: The command to run, as an argument vector.

    Returns:
        The shell command to issue on the server.
    """
    activation = f'eval "$(conda shell.bash hook)" && source activate {shlex.quote(environment)}'
    return f"bash -lc {shlex.quote(f'{activation} && {shlex.join(command)}')}"


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


def _partition_blocked_jobs(
    jobs: list[dict[str, Any]], succeeded: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Splits a unit's outstanding jobs into the ones that may be submitted and the ones that may not.

    Notes:
        A prerequisite that neither runs in this batch nor already succeeded is one this run cannot produce. Blocking
        propagates, so a stage waiting on a blocked stage is blocked in turn.

    Args:
        jobs: The unit's outstanding job descriptors.
        succeeded: The identifiers of the unit's jobs already recorded as succeeded.

    Returns:
        A tuple of the submittable descriptors and the blocked entries, each naming the prerequisites it waits on.
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

    submittable = [descriptor for descriptor in jobs if descriptor["job_id"] not in blocked]
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
    return submittable, blocked_entries


def _pending_job(descriptor: dict[str, Any]) -> GenericPendingJob:
    """Builds the pending job a submission renders from one remote descriptor.

    Args:
        descriptor: The job descriptor to build from.

    Returns:
        The pending job.
    """
    return GenericPendingJob(
        tracker_path=Path(),
        job_id=descriptor["job_id"],
        job_name=descriptor["job_name"],
        name=descriptor["unit_name"],
        specifier=descriptor["specifier"],
        pipeline=descriptor["pipeline"],
        unit_path=Path(descriptor["unit_path"]),
        core_weight=int(descriptor["cores"]),
        memory_mb=int(descriptor["memory_mb"]),
        prerequisite_ids=tuple(descriptor.get("prerequisite_ids", ())),
        options=dict(descriptor.get("options") or {}),
    )


def _job_key(job: GenericPendingJob) -> tuple[str, str]:
    """Returns the key uniquely identifying one job across a remote batch.

    Notes:
        A job identifier is derived from the job name and specifier alone, so the same stage of two different units
        shares one identifier.

    Args:
        job: The job to key.

    Returns:
        The unit path paired with the job identifier.
    """
    return str(job.unit_path), job.job_id


def _prerequisite_keys(job: GenericPendingJob) -> tuple[tuple[str, str], ...]:
    """Returns the keys of the jobs one job waits on, scoped to its own unit.

    Args:
        job: The job whose upstream jobs to key.

    Returns:
        The upstream jobs' keys.
    """
    return tuple((str(job.unit_path), prerequisite) for prerequisite in job.prerequisite_ids)


def _resolve_submission_order(jobs: Sequence[GenericPendingJob]) -> list[GenericPendingJob]:
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
    by_key = {_job_key(job=job): job for job in jobs}
    depths: dict[tuple[str, str], int] = {}

    def _depth(key: tuple[str, str], visiting: set[tuple[str, str]]) -> int:
        cached = depths.get(key)
        if cached is not None:
            return cached
        if key in visiting:
            return 0
        visiting.add(key)
        resolved = max(
            (
                _depth(prerequisite, visiting) + 1
                for prerequisite in _prerequisite_keys(job=by_key[key])
                if prerequisite in by_key
            ),
            default=0,
        )
        visiting.discard(key)
        depths[key] = resolved
        return resolved

    order = sorted(range(len(jobs)), key=lambda position: (_depth(_job_key(job=jobs[position]), set()), position))
    return [jobs[position] for position in order]


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
    return f"{index:04d}-{_SLURM_NAME_SANITIZER.sub('_', readable)}"


def _discover_remote_datasets(server: Server, project_path: Path) -> list[Path]:
    """Lists the forged dataset directories a remote project holds.

    Notes:
        A forged dataset is a top-level directory under the project root carrying a dataset marker.

    Args:
        server: The connected server holding the project.
        project_path: The path to the project's root directory on the server.

    Returns:
        The paths to the project's dataset directories, ordered by name.
    """
    datasets: list[Path] = []
    for entry in server.list_directory(remote_path=project_path):
        candidate = project_path.joinpath(entry)
        if server.exists(remote_path=candidate.joinpath(DATASET_MARKER_FILENAME)):
            datasets.append(candidate)
    return sorted(datasets)


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
