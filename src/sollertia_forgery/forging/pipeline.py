"""Provides the system-agnostic, end-to-end dataset forging pipeline.

The pipeline owns dataset definition, the optional cindra multi-day stage, and per-session orchestration; the only
system-specific concern -- assembling one session's ``data.feather`` plus its data-format descriptor -- is resolved
from ``FORGING_ASSEMBLY_REGISTRY`` by the dataset's acquisition system. The dependency is strictly one-way: the
pipeline reaches the system worker through the registry, never the reverse.

See ``run_forging_pipeline`` for the stage ordering, the local/remote execution modes, and the tracker contract.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING
from contextlib import nullcontext
from concurrent.futures import Future, ProcessPoolExecutor, as_completed

from cindra import run_multi_recording_pipeline
from natsort import natsorted
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count
from sollertia_shared_assets import SessionData, RawDataFiles, ProcessingTrackers
from ataraxis_data_structures import ProcessingTracker

from .dataset import resolve_dataset
from ..registries import resolve_forging_assembly_worker
from ..shared_assets import tracked_job, prepare_tracker

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from sollertia_shared_assets import DatasetSession

# The registered, picklable per-session assembly worker resolved from FORGING_ASSEMBLY_REGISTRY. The PEP 695 alias
# is evaluated lazily, so its annotation-only operands need not exist at runtime.
type ForgingAssemblyWorker = Callable[[Path, Path, str], None]

FORGING_JOB_NAME: str = "session_data_assembly"
"""The job name identifying per-session assembly jobs in the forging processing tracker. The same string is used by
any deployment layer that submits per-session forging jobs, so the job identifiers it derives match the ones this
pipeline computes."""


def run_forging_pipeline(
    name: str,
    session_names: tuple[str, ...],
    project_root: Path,
    job_id: str | None = None,
    *,
    workers: int = -1,
    display_progress: bool = False,
    force_recreate: bool = False,
    activity_configuration: Path | None = None,
) -> None:
    """Defines the dataset hierarchy and executes the forging assembly jobs for the target sessions.

    Notes:
        This is the system-agnostic forging entry point. Stage 1 (dataset definition) and the optional Stage 2
        (cindra multi-day processing) run up front before the processing tracker is created, so the tracker only
        holds per-session assembly jobs. The per-session assembly worker is resolved from the central
        ``FORGING_ASSEMBLY_REGISTRY`` by the dataset's acquisition system, so the pipeline stays system-agnostic and
        never names a system-specific type.

        In local mode (``job_id`` is None), all assembly jobs are distributed across a parallel worker pool. In
        remote mode (``job_id`` is provided), the identifier must match an assembly job and only that single
        session's assembly runs in-process; the cindra multi-day stage never runs in remote mode because it is a
        dataset-level prerequisite executed once, not per session.

        The cindra multi-day stage runs only when ``activity_configuration`` is provided (gated on the configuration
        alone, assuming single-recording cindra has already completed). When it is omitted, Stage 2 is skipped and
        the assembly stage consumes whatever cindra outputs already exist on disk.

    Args:
        name: The unique name of the dataset.
        session_names: The session names to include in the dataset. When the dataset already exists and a non-empty
            list is provided, the set is verified against the existing definition; pass an empty tuple to work with
            an already-defined dataset without triggering verification.
        project_root: The path to the project's root directory that stores the animal and session data directories.
            The dataset hierarchy is also created under this directory.
        job_id: The hexadecimal identifier of the single assembly job to execute (remote mode). If not provided, the
            whole pipeline runs (local mode).
        workers: The number of worker processes to use. A value less than 1 uses all available CPU cores (minus
            reserved cores); 1 forces sequential processing.
        display_progress: Determines whether to display a progress bar during assembly.
        force_recreate: Determines whether to allow deletion of the existing dataset hierarchy when the provided
            session list does not match the existing definition.
        activity_configuration: The path to the cindra multi-recording configuration file. When provided (and in a
            full local run), the multi-day cell-tracking stage runs before assembly; when None, the stage is skipped.

    Raises:
        ValueError: If the dataset does not exist and no sessions were provided to create it, if the provided session
            list does not match the existing dataset and force_recreate is False, if the dataset's acquisition system
            has no registered assembly worker, or if the provided job_id does not match any assembly job.
    """
    console.echo(message=f"Initializing the forging pipeline for dataset '{name}'...", level=LogLevel.INFO)

    # Stage 1: resolves the dataset hierarchy (create, load, verify, or recreate). Any error propagates unchanged.
    dataset = resolve_dataset(
        name=name, session_names=session_names, project_root=project_root, force_recreate=force_recreate
    )

    # Resolves the per-session assembly worker for the dataset's acquisition system from the central registry. The
    # system is inferred from the data, so the pipeline never names a system-specific type.
    worker = resolve_forging_assembly_worker(dataset.acquisition_system)

    dataset_path = dataset.dataset_data_path.parent
    dataset_session_names = [entry.session for entry in dataset.sessions]
    session_lookup: dict[str, DatasetSession] = {entry.session: entry for entry in dataset.sessions}

    console.echo(message=f"Discovered {len(dataset_session_names)} assembly job(s).")

    # Stage 2: runs the cindra multi-day stage once, up front, only in a full local run with a supplied
    # configuration. It is a dataset-level prerequisite and writes its own trackers, so it is not part of the forging
    # tracker and never runs for a single remote assembly job.
    if job_id is None and activity_configuration is not None:
        _run_activity_stage(configuration_path=activity_configuration)

    # Prepares the forging tracker and registers one assembly job per session (Stages 3 and 4). For forging the job
    # universe equals the requested set: every session in the resolved dataset is always processable.
    dataset_path.mkdir(parents=True, exist_ok=True)
    tracker = ProcessingTracker(file_path=dataset_path.joinpath(ProcessingTrackers.FORGING))
    jobs = [(FORGING_JOB_NAME, session) for session in dataset_session_names]
    prepare_tracker(tracker=tracker, jobs=jobs, universe=jobs)

    job_ids = {
        session: ProcessingTracker.generate_job_id(job_name=FORGING_JOB_NAME, specifier=session)
        for session in dataset_session_names
    }

    if job_id is not None:
        # Remote mode: routes on the identifier to assemble a single session in-process.
        id_to_session = {generated_id: session for session, generated_id in job_ids.items()}
        if job_id not in id_to_session:
            message = (
                f"Unable to execute the requested job with ID '{job_id}'. The identifier does not match any assembly "
                f"job available for this dataset. Valid job IDs: {natsorted(id_to_session.keys())}."
            )
            console.error(message=message, error=ValueError)
        _execute_job(
            session_name=id_to_session[job_id],
            session_lookup=session_lookup,
            dataset_name=dataset.name,
            project_root=project_root,
            tracker=tracker,
            job_id=job_id,
            worker=worker,
        )
    else:
        # Local mode: distributes all jobs across a shared worker pool, falling back to sequential execution when a
        # single worker or a single job makes a pool pointless.
        resolved_workers = resolve_worker_count(requested_workers=workers)
        if resolved_workers > 1 and len(dataset_session_names) > 1:
            _execute_jobs_parallel(
                sessions=dataset_session_names,
                session_lookup=session_lookup,
                dataset_name=dataset.name,
                project_root=project_root,
                tracker=tracker,
                job_ids=job_ids,
                worker=worker,
                workers=resolved_workers,
                display_progress=display_progress,
            )
        else:
            _execute_jobs_sequential(
                sessions=dataset_session_names,
                session_lookup=session_lookup,
                dataset_name=dataset.name,
                project_root=project_root,
                tracker=tracker,
                job_ids=job_ids,
                worker=worker,
                display_progress=display_progress,
            )

        # Every session's data.feather now exists, so the dataset's data-description contract can be enforced against
        # the fully composed dataset: every column any session actually wrote must be described in the dataset's
        # data_descriptions.feather. A violation means the acquisition system emitted an undescribed column. Remote
        # mode skips this because the other sessions are not yet present; the batch verifier enforces it once the full
        # set has been assembled.
        console.echo(message="Verifying assembled-data column descriptions...", level=LogLevel.INFO)
        dataset.verify_data_descriptions()

    console.echo(message="All forging jobs completed successfully.", level=LogLevel.SUCCESS)


def _run_activity_stage(configuration_path: Path) -> None:
    """Runs the cindra multi-day (across-session cell tracking) stage for the dataset.

    Notes:
        Runs both the cross-recording cell-discovery and the per-recording aligned-fluorescence extraction stages in
        sequence so a single local invocation performs the full multi-day pipeline. cindra owns its own per-plane /
        per-stage job decomposition and writes its own processing trackers at the recording root resolved from the
        configuration file.

    Args:
        configuration_path: The path to the cindra multi-recording configuration file. The configuration encodes the
            recordings to process and the per-dataset processing parameters.
    """
    console.echo(
        message=f"Stage 2: running multi-day cell-activity processing for '{configuration_path}'...",
        level=LogLevel.INFO,
    )
    run_multi_recording_pipeline(
        configuration_path=configuration_path, job_id=None, discover=True, extract=True, target_recording=None
    )
    console.echo(message="Multi-day cell-activity processing completed successfully.", level=LogLevel.SUCCESS)


def _execute_jobs_sequential(
    sessions: list[str],
    session_lookup: dict[str, DatasetSession],
    dataset_name: str,
    project_root: Path,
    tracker: ProcessingTracker,
    job_ids: dict[str, str],
    worker: ForgingAssemblyWorker,
    *,
    display_progress: bool,
) -> None:
    """Runs all assembly jobs sequentially in the parent process with an optional progress bar.

    Notes:
        Selected automatically when the resolved worker count is 1 or only a single job is discovered. Each job is
        fully owned by the parent process, so the first exception aborts the remaining jobs.

    Args:
        sessions: The ordered list of session names to assemble.
        session_lookup: The mapping from session name to its DatasetSession metadata.
        dataset_name: The name of the dataset being assembled.
        project_root: The path to the project's root directory.
        tracker: The forging processing tracker.
        job_ids: The mapping from session name to job ID.
        worker: The registered per-session assembly worker.
        display_progress: Determines whether to display a per-session progress bar.
    """
    progress_context = (
        console.progress(total=len(sessions), description="Assembling dataset sessions", unit="session")
        if display_progress
        else nullcontext()
    )

    with progress_context as progress_bar:
        for session_name in sessions:
            _execute_job(
                session_name=session_name,
                session_lookup=session_lookup,
                dataset_name=dataset_name,
                project_root=project_root,
                tracker=tracker,
                job_id=job_ids[session_name],
                worker=worker,
            )
            if progress_bar is not None:
                progress_bar.update(1)


def _execute_jobs_parallel(
    sessions: list[str],
    session_lookup: dict[str, DatasetSession],
    dataset_name: str,
    project_root: Path,
    tracker: ProcessingTracker,
    job_ids: dict[str, str],
    worker: ForgingAssemblyWorker,
    workers: int,
    *,
    display_progress: bool,
) -> None:
    """Runs all assembly jobs concurrently across a shared ProcessPoolExecutor.

    Notes:
        Every dispatched job is tracked individually, and in-flight futures are allowed to finish on failure so the
        tracker stays accurate for all of them; the first captured exception is re-raised after all futures resolve.

    Args:
        sessions: The ordered list of session names to assemble.
        session_lookup: The mapping from session name to its DatasetSession metadata.
        dataset_name: The name of the dataset being assembled.
        project_root: The path to the project's root directory.
        tracker: The forging processing tracker.
        job_ids: The mapping from session name to job ID.
        worker: The registered per-session assembly worker; must be picklable for the worker processes.
        workers: The resolved worker-process count for the shared pool.
        display_progress: Determines whether to display a per-session progress bar.
    """
    first_exception: Exception | None = None

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_job_id: dict[Future[None], str] = {}
        for session_name in sessions:
            job_id = job_ids[session_name]
            session_metadata = session_lookup[session_name]
            source_session_path = project_root.joinpath(session_metadata.animal, session_name)

            console.echo(message=f"Running assembly job for session '{session_name}' (ID: {job_id})...")
            tracker.start_job(job_id=job_id)
            future = executor.submit(
                _forge_session,
                source_session_path=source_session_path,
                output_path=session_metadata.data_path,
                dataset_name=dataset_name,
                worker=worker,
            )
            future_to_job_id[future] = job_id

        progress_context = (
            console.progress(total=len(sessions), description="Assembling dataset sessions", unit="session")
            if display_progress
            else nullcontext()
        )

        with progress_context as progress_bar:
            for completed_future in as_completed(future_to_job_id):
                completed_job_id = future_to_job_id[completed_future]
                try:
                    completed_future.result()
                    tracker.complete_job(job_id=completed_job_id)
                except Exception as exception:
                    tracker.fail_job(job_id=completed_job_id, error_message=str(exception))
                    if first_exception is None:
                        first_exception = exception
                if progress_bar is not None:
                    progress_bar.update(1)

    if first_exception is not None:
        raise first_exception


def _execute_job(
    session_name: str,
    session_lookup: dict[str, DatasetSession],
    dataset_name: str,
    project_root: Path,
    tracker: ProcessingTracker,
    job_id: str,
    worker: ForgingAssemblyWorker,
) -> None:
    """Executes a single session assembly job in-process with full tracker state management.

    Notes:
        Used by the remote (single-job) execution path and by the sequential local path. The parallel path calls
        ``_forge_session`` directly from worker processes and manages tracker state separately in the parent.

    Args:
        session_name: The name of the session whose data to assemble.
        session_lookup: The mapping from session name to its DatasetSession metadata.
        dataset_name: The name of the dataset being assembled.
        project_root: The path to the project's root directory.
        tracker: The forging processing tracker.
        job_id: The unique hexadecimal identifier for this assembly job.
        worker: The registered per-session assembly worker.
    """
    console.echo(message=f"Running assembly job for session '{session_name}' (ID: {job_id})...")
    with tracked_job(tracker=tracker, job_id=job_id):
        session_metadata = session_lookup[session_name]
        source_session_path = project_root.joinpath(session_metadata.animal, session_name)
        _forge_session(
            source_session_path=source_session_path,
            output_path=session_metadata.data_path,
            dataset_name=dataset_name,
            worker=worker,
        )
    console.echo(message=f"Session '{session_name}' data assembly: Complete.", level=LogLevel.SUCCESS)


def _forge_session(
    source_session_path: Path,
    output_path: Path,
    dataset_name: str,
    worker: ForgingAssemblyWorker,
) -> None:
    """Forges and assembles a single session: runs the system worker, then re-exports the shared assets.

    Notes:
        The atomic unit dispatched to worker processes by the parallel path, so it must stay importable at module
        level and accept only picklable arguments. These shared assets are system-agnostic, so the pipeline
        hard-defines their handling rather than delegating it to the system worker. The session descriptor is written
        by every acquisition runtime, but the VR and experiment configurations are present only for the session types
        that carry them (sessions that use VR and experiment sessions, respectively). The session's own required-asset
        policy decides which of them are mandatory, so a session missing a required asset (e.g., a session that uses VR
        without its VR configuration) fails fast before any expensive work, while session types that carry neither
        configuration still forge into a self-contained session. Whichever assets the session actually holds are
        re-exported alongside the assembled feather.

    Args:
        source_session_path: The path to the source session's root directory in the project hierarchy.
        output_path: The path to the session's ``data.feather`` in the forged dataset hierarchy.
        dataset_name: The unqualified dataset name, forwarded to the assembly worker.
        worker: The registered per-session assembly worker for the dataset's acquisition system.

    Raises:
        FileNotFoundError: If a shared asset the session is required to carry is missing from the source session.
    """
    # Resolves the shared assets the forged session re-exports. The session descriptor is universal; the VR and
    # experiment configurations are present only for some session types, so each is re-exported only when present.
    session = SessionData.load(session_path=source_session_path)
    reexported_assets = {
        RawDataFiles.SESSION_DESCRIPTOR: session.raw_data.session_descriptor_path,
        RawDataFiles.VR_CONFIGURATION: session.raw_data.vr_configuration_path,
        RawDataFiles.EXPERIMENT_CONFIGURATION: session.raw_data.experiment_configuration_path,
    }

    # Validates the assets this session is required to carry before any expensive work. The session's required-asset
    # policy is the single source of truth for which re-exported assets are mandatory for its session type.
    required_filenames = {filename for filename, _ in session.required_raw_assets()}
    for filename, source_path in reexported_assets.items():
        if filename in required_filenames and not source_path.is_file():
            message = (
                f"Unable to assemble session '{source_session_path.name}'. The session's raw data directory does not "
                f"contain the required shared asset '{filename}' at '{source_path}'."
            )
            console.error(message=message, error=FileNotFoundError)

    # Stage 3: assembles the session data (data.feather + the system data-format descriptor).
    worker(source_session_path, output_path, dataset_name)

    # Stage 4: re-exports each shared asset the session actually carries alongside the assembled feather, so session
    # types that do not run an experiment or use VR still forge into a self-contained session directory.
    output_directory = output_path.parent
    output_directory.mkdir(parents=True, exist_ok=True)
    for filename, source_path in reexported_assets.items():
        if source_path.is_file():
            shutil.copy2(src=source_path, dst=output_directory.joinpath(filename))
