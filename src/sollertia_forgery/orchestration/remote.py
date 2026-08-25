"""Provides the remote execution backend that runs prepared jobs as SLURM allocations on the compute server."""

from __future__ import annotations

import re
from math import ceil
import shlex
from typing import TYPE_CHECKING, Any
from dataclasses import asdict

from ataraxis_base_utilities import LogLevel, console
from sollertia_shared_assets import DATASET_MARKER_FILENAME, ProcessingTrackers

from .graph import build_pending_job, resolve_submission_order
from .hosts import RemoteHost, environment_command
from .ledger import SubmissionBatch, RemoteSubmission, read_ledger, record_batch, current_timestamp
from ..server import Job, Server, discover_project_markers, get_server_configuration
from ..forging import DATASET_STATE_FILENAME
from .dispatch import resolve_job_command
from .planning import project_plan_path
from ..managing import project_jobs_path, project_manifest_path
from .preparation import prepare_batch

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Sequence

    from .graph import BatchDocument, GenericPendingJob
    from ..server import JobStatus


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


def remote_batch_directory(server: Server, batch_id: str) -> Path:
    """Resolves the server-side directory holding one batch's job scripts and logs.

    Args:
        server: The server that runs the batch.
        batch_id: The identifier of the batch.

    Returns:
        The path to the batch's directory on the server.
    """
    return server.root.joinpath(_BATCH_DIRECTORY_NAME, batch_id)


def _prepare_remote_batch(
    server: Server, pipeline: str, unit_paths: Sequence[str], options: dict[str, Any] | None = None
) -> BatchDocument:
    """Resolves a pipeline's submittable jobs for the named units on the remote compute server.

    Notes:
        Delegates to the shared preparation, so a remote batch is resolved by the same code that resolves a local one.
        The only difference is the host that materializes the artifacts and delivers them here.

    Args:
        server: The connected server holding the units.
        pipeline: The batch pipeline to prepare.
        unit_paths: The processing unit directories on the server whose jobs to prepare.
        options: The pipeline-specific parameters given to the prepared jobs.

    Returns:
        The prepared batch document.

    Raises:
        ValueError: If the named pipeline is not a supported batch pipeline, if no unit is named, or if the named
            units do not share one project.
        FileNotFoundError: If the server holds no plan table for the units' project.
        RuntimeError: If a server-side command fails.
    """
    return prepare_batch(host=RemoteHost(server=server), pipeline=pipeline, unit_paths=unit_paths, options=options)


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

        Every accepted allocation is recorded in the submission ledger, including when the scheduler rejects a later
        job of the same batch, since the allocations it already accepted stay queued.

        The record is merged into whatever the ledger already holds for this batch, so re-running a batch the scheduler
        only partly accepted keeps the allocations the first attempt queued. An entry this call re-submitted is
        replaced rather than duplicated.

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
    """
    batch_directory = remote_batch_directory(server=server, batch_id=batch_id)
    server.create(remote_path=batch_directory, is_dir=True, parents=True)

    pending = [build_pending_job(job=descriptor) for descriptor in jobs]
    ordered = resolve_submission_order(jobs=pending)

    submissions: list[RemoteSubmission] = []
    allocation_of_job: dict[tuple[str, str], str] = dict(adopted or {})
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
            resubmitted = {(entry.unit_path, entry.job_id) for entry in submissions}
            already_recorded = read_ledger().resolve_batch(batch_id=batch_id)
            carried = (
                [entry for entry in already_recorded.submissions if (entry.unit_path, entry.job_id) not in resubmitted]
                if already_recorded is not None
                else []
            )
            record_batch(
                batch=SubmissionBatch(
                    batch_id=batch_id,
                    batch_ids=list(covered_batch_ids) if covered_batch_ids else [batch_id],
                    batch_directory=str(batch_directory),
                    submitted_at=current_timestamp(),
                    walltime_minutes=walltime_minutes,
                    submissions=[*carried, *submissions],
                )
            )

    return submissions


def query_submissions(server: Server, submissions: Sequence[RemoteSubmission]) -> dict[str, JobStatus]:
    """Queries the scheduler for the state of every submitted allocation.

    Notes:
        Observing a state and acting on it are separate, so this retires nothing. A batch leaves the submission ledger
        only once closure has snapshotted what its jobs recorded, which keeps a finished batch answerable rather than
        forgotten the moment it settles.

    Args:
        server: The connected server that runs the batch.
        submissions: The submissions to query.

    Returns:
        A dictionary mapping each submission's allocation identifier to its scheduler state.
    """
    return server.get_job_statuses(slurm_job_ids=[submission.slurm_job_id for submission in submissions])


def cancel_submissions(server: Server, submissions: Sequence[RemoteSubmission]) -> list[str]:
    """Cancels every allocation the given submissions hold in one call, which the scheduler applies to the queued and
    running ones alone.

    Args:
        server: The connected server that runs the batch.
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
            ram=max(1, ceil(job.memory_mb / _MEGABYTES_PER_GIGABYTE)),
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
                memory_mb=job.memory_mb,
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
