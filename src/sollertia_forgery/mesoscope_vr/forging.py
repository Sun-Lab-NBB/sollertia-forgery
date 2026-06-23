"""Provides the forging pipeline entry point that resolves the dataset hierarchy, registers per-session assembly
jobs, and executes them locally or via remote job routing.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING
from functools import partial
from contextlib import nullcontext
from dataclasses import dataclass
from concurrent.futures import Future, ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import polars as pl
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count, ensure_directory_exists
from sollertia_shared_assets import (
    SessionData,
    RawDataFiles,
    SessionTypes,
    ProcessingTrackers,
    MesoscopeExperimentConfiguration,
)
from ataraxis_data_structures import ProcessingTracker

from ..forging import resolve_dataset
from .metadata import TrialGeometry
from .fluorescence import assemble_cindra_dataset
from ..shared_assets import DatasetFiles, DatasetSession, prepare_tracker
from .runtime_dataset import assemble_runtime_dataset, _mask_non_run_experiment_data
from .behavior_dataset import assemble_behavior_dataset

if TYPE_CHECKING:
    from pathlib import Path


FORGING_JOB_NAME: str = "session_data_assembly"
"""The job name used to identify per-session assembly stages in forging processing trackers."""


@dataclass(frozen=True, slots=True)
class _SessionPaths:
    """Stores resolved filesystem paths for a single session assembly job."""

    behavior_data_path: Path
    """The path to the directory containing the processed behavior feather files."""
    raw_data_path: Path
    """The path to the session's raw data directory containing the hardware state and experiment configuration."""
    cindra_data_path: Path
    """The path to the single-recording cindra output directory."""
    multiday_data_path: Path
    """The path to the dataset-specific multiday output directory."""


def run_forging_pipeline(
    name: str,
    session_names: tuple[str, ...],
    project_root: Path,
    job_id: str | None = None,
    *,
    workers: int = -1,
    display_progress: bool = False,
    force_recreate: bool = False,
) -> None:
    """Defines the dataset hierarchy and executes data assembly jobs for the target sessions.

    Notes:
        Dataset definition runs up front (create-or-verify-or-recreate) before the processing tracker is created,
        so the tracker only holds per-session assembly jobs — there is no tracker-managed definition stage. In
        local mode (job_id is None), all assembly jobs are distributed across a parallel worker pool. In remote
        mode (job_id is provided), the identifier must match an assembly job and the pipeline executes only that
        single session's assembly in-process.

        Dataset creation is currently limited to mesoscope experiment sessions.

    Args:
        name: The unique name of the dataset.
        session_names: The session names to include in the dataset. The caller is responsible for any
            pre-filtering. When the dataset already exists and a non-empty list is provided, the set is verified
            against the existing definition. Pass an empty tuple to work with an already-defined dataset without
            triggering the verification step.
        project_root: The path to the project's root directory that stores the animal and session data directories.
            The dataset hierarchy is also created under this directory.
        job_id: The unique hexadecimal identifier for the assembly job to execute. If provided, only the job
            matching this ID is executed (remote mode). If not provided, all assembly jobs are executed with
            automatic tracker management (local mode).
        workers: The number of worker processes to use for parallel processing. Setting this to a value less than 1
            uses all available CPU cores (minus reserved cores). Setting this to 1 conducts processing sequentially
            without spawning worker processes.
        display_progress: Determines whether to display a progress bar during processing.
        force_recreate: Determines whether to allow deletion of the existing dataset hierarchy when the provided
            session list does not match the existing definition. Set this to True when extending, shrinking, or
            modifying the session set of an existing dataset. Any prior assembled data and the existing processing
            tracker are discarded.

    Raises:
        ValueError: If the first session's type is not MESOSCOPE_EXPERIMENT, if the dataset does not exist and no
            sessions were provided to create it, if the provided session list does not match the existing dataset
            and force_recreate is False, or if the provided job_id does not match any assembly job.
    """
    console.echo(
        message=f"Initializing the forging pipeline for dataset '{name}'...",
        level=LogLevel.INFO,
    )

    # Resolves the dataset hierarchy (create, load, verify, or recreate). Any error propagates unchanged.
    dataset = resolve_dataset(
        name=name,
        session_names=session_names,
        project_root=project_root,
        required_session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
        force_recreate=force_recreate,
    )

    # Resolves the dataset path and enumerates session assembly jobs.
    dataset_path = dataset.dataset_data_path.parent
    dataset_session_names = [entry.session for entry in dataset.sessions]

    console.echo(message=f"Discovered {len(dataset_session_names)} assembly job(s).")

    # Builds a session metadata lookup for O(1) access per session.
    session_lookup: dict[str, DatasetSession] = {entry.session: entry for entry in dataset.sessions}

    # Prepares the processing tracker and registers one assembly job per session. Dataset definition is not
    # tracker-managed: it has already run to completion by the time the tracker is created, so there is nothing
    # for the tracker to track.
    tracker = ProcessingTracker(file_path=dataset_path.joinpath(ProcessingTrackers.FORGING))
    jobs = [(FORGING_JOB_NAME, session) for session in dataset_session_names]
    prepare_tracker(tracker=tracker, jobs=jobs, universe=jobs)

    job_ids = {
        session: ProcessingTracker.generate_job_id(job_name=FORGING_JOB_NAME, specifier=session)
        for session in dataset_session_names
    }

    if job_id is not None:
        # Remote mode: routes on the identifier to execute a single session's assembly in-process.
        id_to_session: dict[str, str] = {jid: session for session, jid in job_ids.items()}

        if job_id not in id_to_session:
            message = (
                f"Unable to execute the requested job with ID '{job_id}'. The input identifier does not match "
                f"any assembly jobs available for this dataset. Valid job IDs: {sorted(id_to_session.keys())}."
            )
            console.error(message=message, error=ValueError)

        _execute_job(
            session_name=id_to_session[job_id],
            session_lookup=session_lookup,
            dataset_name=dataset.name,
            project_root=project_root,
            tracker=tracker,
            job_id=job_id,
        )
    else:
        # Local mode: resolves the worker count and distributes all discovered jobs across a shared
        # ProcessPoolExecutor. Sequential execution is used automatically when the pool would contain a single
        # worker or only a single job is available.
        resolved_workers = resolve_worker_count(requested_workers=workers)

        if resolved_workers > 1 and len(dataset_session_names) > 1:
            _execute_jobs_parallel(
                sessions=dataset_session_names,
                session_lookup=session_lookup,
                dataset_name=dataset.name,
                project_root=project_root,
                tracker=tracker,
                job_ids=job_ids,
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
                display_progress=display_progress,
            )

    console.echo(message="All forging jobs completed successfully.", level=LogLevel.SUCCESS)


def _resolve_session_paths(session_data_path: Path, dataset_name: str) -> _SessionPaths:
    """Resolves all data directory paths for a single session assembly job from the canonical session layout.

    Notes:
        Loads ``SessionData`` and reads its path-resolution properties for the canonical ``behavior_data`` and
        ``cindra`` locations under ``processed_data``. The cindra multi-recording path is derived by joining
        ``multi_recording/{animal_id}_{dataset_name}`` to the cindra output directory; the animal identifier is
        prepended to match Cindra's on-disk qualification convention for collision-free multi-animal batches.

    Args:
        session_data_path: The path to the session's root directory.
        dataset_name: The unqualified dataset name being assembled, used together with the animal identifier to
            resolve the cindra multi-recording output directory.

    Returns:
        A frozen ``_SessionPaths`` instance containing all resolved data directory paths.

    Raises:
        FileNotFoundError: If either the behavior or cindra canonical output directory is missing.
    """
    # Loads the session's metadata to obtain canonical raw and processed data root paths.
    session = SessionData.load(session_path=session_data_path)

    # Validates that the canonical behavior and cindra output directories exist under processed_data.
    if not session.processed_data.behavior_data_path.is_dir():
        message = (
            f"Unable to resolve the behavior data directory for session '{session_data_path.name}'. "
            f"Expected '{session.processed_data.behavior_data_path}' to exist and contain "
            f"'{ProcessingTrackers.BEHAVIOR}'."
        )
        console.error(message=message, error=FileNotFoundError)
    if not session.processed_data.cindra_data_path.is_dir():
        message = (
            f"Unable to resolve the cindra data directory for session '{session_data_path.name}'. "
            f"Expected '{session.processed_data.cindra_data_path}' to exist and contain "
            f"'{ProcessingTrackers.CINDRA_SINGLE_RECORDING}'."
        )
        console.error(message=message, error=FileNotFoundError)

    # Derives the cindra multi-recording output path. Cindra writes the dataset directory as
    # ``{animal_id}_{dataset_name}`` for collision avoidance when batching multiple animals under a single
    # forged dataset name, so the animal identifier is prepended here.
    multiday_data_path = session.processed_data.cindra_multi_recording_path.joinpath(
        f"{session.animal_id}_{dataset_name}"
    )

    return _SessionPaths(
        behavior_data_path=session.processed_data.behavior_data_path,
        raw_data_path=session.raw_data_path,
        cindra_data_path=session.processed_data.cindra_data_path,
        multiday_data_path=multiday_data_path,
    )


def _assemble_session_dataset(
    session_paths: _SessionPaths,
    output_path: Path,
    *,
    progress: bool = False,
) -> None:
    """Assembles the experiment forged dataset for the target session.

    Extracts, post-processes, and combines all relevant data for the processed session into a unified Polars
    DataFrame and saves it to an uncompressed .feather file at ``output_path``. Also copies the session's
    experiment descriptor YAML alongside the feather file so the forged session is self-contained for
    downstream consumers.

    Args:
        session_paths: The resolved filesystem paths for the target session's data directories.
        output_path: The path to the .feather file where to save the assembled dataset.
        progress: Determines whether to display the session's data assembly progress via the terminal progress bar.

    Raises:
        FileNotFoundError: If the session's raw data directory does not contain a ``session_descriptor.yaml``
            file.
    """
    # Ensures that the output directory exists.
    ensure_directory_exists(path=output_path)

    # Verifies the experiment descriptor exists up front so the failure surfaces before any expensive work.
    source_descriptor_path = session_paths.raw_data_path.joinpath(RawDataFiles.SESSION_DESCRIPTOR)
    if not source_descriptor_path.is_file():
        message = (
            f"Unable to assemble session '{output_path.parent.name}'. The session's raw data directory does "
            f"not contain a '{RawDataFiles.SESSION_DESCRIPTOR}' file at '{source_descriptor_path}'. The "
            f"experiment descriptor is required for every session in a forged dataset."
        )
        console.error(message=message, error=FileNotFoundError)

    # Configures progress bar visibility based on the progress parameter.
    prior_progress = console.progress_enabled
    if progress:
        console.enable_progress()
    else:
        console.disable_progress()

    # Loads the experiment configuration once so the runtime assembly and the trial geometry data file share a single
    # parsed instance instead of reading the same YAML twice.
    experiment_configuration = MesoscopeExperimentConfiguration.from_yaml(
        file_path=session_paths.raw_data_path.joinpath(RawDataFiles.EXPERIMENT_CONFIGURATION)
    )

    try:
        # Assembles the fluorescence data first, which is needed to generate the reference time vector for the
        # behavior and runtime datasets.
        with console.progress(
            total=3, description=f"Assembling session {session_paths.behavior_data_path.parent.stem} datasets"
        ) as pbar:
            fluorescence_data = assemble_cindra_dataset(
                cindra_data_path=session_paths.cindra_data_path,
                behavior_data_path=session_paths.behavior_data_path,
                multiday_data_path=session_paths.multiday_data_path,
                raw_data_path=session_paths.raw_data_path,
            )
            pbar.update(1)

            # Extracts reference time to assemble other datasets in parallel.
            reference_time = fluorescence_data["time_us"].to_numpy()

            # Defines tasks for parallel execution.
            tasks = {
                "behavior": partial(
                    assemble_behavior_dataset,
                    behavior_data_path=session_paths.behavior_data_path,
                    raw_data_path=session_paths.raw_data_path,
                    reference_time=reference_time,
                    drop_time_columns=True,
                ),
                "runtime": partial(
                    assemble_runtime_dataset,
                    behavior_data_path=session_paths.behavior_data_path,
                    experiment_configuration=experiment_configuration,
                    reference_time=reference_time,
                ),
            }

            # Executes the behavior and runtime assembly in parallel.
            results: dict[str, pl.DataFrame] = {}
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_to_name = {executor.submit(task): name for name, task in tasks.items()}

                for future in as_completed(future_to_name):
                    name = future_to_name[future]
                    results[name] = future.result()
                    pbar.update(1)

            # Extracts processing results.
            behavior_data = results["behavior"]
            runtime_data = results["runtime"]

        # Concatenates all dataframes into the unified dataset.
        result = pl.concat([fluorescence_data, behavior_data, runtime_data], how="horizontal")

        # Masks cue, trial, and trial_type with 255 (or "undefined") for non-run experiment states.
        result = _mask_non_run_experiment_data(experiment_data=result)

        # Saves the unified dataset to disk as an uncompressed .feather file (to support memory-mapping).
        result.write_ipc(file=output_path)

        # Copies the experiment descriptor next to data.feather so the forged session carries the experimenter
        # context (animal weight, water dispensed/consumed, completion status, notes) needed for downstream
        # consumers without reaching back into the raw session.
        shutil.copy2(
            src=source_descriptor_path,
            dst=output_path.parent.joinpath(RawDataFiles.SESSION_DESCRIPTOR),
        )

        # Projects the canonical trial geometry out of the experiment configuration and writes it next to data.feather
        # so downstream consumers can reconstruct per-trial position without re-reading the experiment configuration.
        trial_geometry = TrialGeometry.from_experiment_configuration(experiment_configuration=experiment_configuration)
        trial_geometry.to_yaml(file_path=output_path.parent.joinpath(DatasetFiles.TRIAL_GEOMETRY))
    finally:
        # Restores the previous progress bar visibility state.
        if prior_progress:
            console.enable_progress()
        else:
            console.disable_progress()


def _execute_jobs_sequential(
    sessions: list[str],
    session_lookup: dict[str, DatasetSession],
    dataset_name: str,
    project_root: Path,
    tracker: ProcessingTracker,
    job_ids: dict[str, str],
    *,
    display_progress: bool,
) -> None:
    """Runs all discovered session assembly jobs sequentially in the parent process with an optional progress bar.

    Notes:
        Selected automatically when the resolved worker count is 1 or only a single job is discovered. Each job
        is fully owned by the parent process (tracker transitions, computation, and failure handling), so the
        first exception aborts the remaining jobs — matching the original single-threaded semantics.

    Args:
        sessions: The ordered list of session names to assemble.
        session_lookup: The mapping from session name to DatasetSession metadata.
        dataset_name: The name of the dataset being assembled.
        project_root: The path to the project's root directory.
        tracker: The ProcessingTracker instance for recording job state transitions.
        job_ids: The mapping from session name to job ID.
        display_progress: Determines whether to display a progress bar during processing.
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
    workers: int,
    *,
    display_progress: bool,
) -> None:
    """Runs all discovered session assembly jobs concurrently across a shared ProcessPoolExecutor.

    Notes:
        Every discovered job is submitted to the pool as an atomic unit. The parent process calls
        ``tracker.start_job`` immediately before submitting each job's future, so tracker state advances in
        lockstep with dispatch and no job can be marked as running without also being dispatched. Results are
        then collected via ``as_completed`` and recorded against the tracker individually. In-flight futures are
        allowed to finish on failure rather than cancelling pending work, so the tracker state remains accurate
        for every dispatched job. After all futures resolve, the first captured exception is re-raised to
        propagate the failure to the caller.

    Args:
        sessions: The ordered list of session names to assemble.
        session_lookup: The mapping from session name to DatasetSession metadata.
        dataset_name: The name of the dataset being assembled.
        project_root: The path to the project's root directory.
        tracker: The ProcessingTracker instance for recording job state transitions.
        job_ids: The mapping from session name to job ID.
        workers: The resolved worker process count for the shared ProcessPoolExecutor.
        display_progress: Determines whether to display a progress bar during processing.
    """
    first_exception: Exception | None = None

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_job_id: dict[Future[None], str] = {}
        for session_name in sessions:
            job_id = job_ids[session_name]

            # Resolves the session's paths for the worker. Path resolution happens in the parent process before
            # submission to ensure picklability.
            session_metadata = session_lookup[session_name]
            session_data_path = project_root.joinpath(session_metadata.animal, session_name)
            session_paths = _resolve_session_paths(session_data_path=session_data_path, dataset_name=dataset_name)

            console.echo(message=f"Running assembly job for session '{session_name}' (ID: {job_id})...")
            tracker.start_job(job_id=job_id)
            future = executor.submit(
                _run_job,
                session_paths=session_paths,
                output_path=session_metadata.data_path,
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
) -> None:
    """Executes a single session assembly job in-process with full tracker state management.

    Notes:
        Used by the remote execution path and by the sequential local execution path. The parallel execution path
        calls ``_run_job`` directly from worker processes and manages tracker state separately in the parent.

    Args:
        session_name: The name of the session whose data to assemble.
        session_lookup: The mapping from session name to DatasetSession metadata.
        dataset_name: The name of the dataset being assembled.
        project_root: The path to the project's root directory.
        tracker: The ProcessingTracker instance for recording job state transitions.
        job_id: The unique hexadecimal identifier for this processing job.
    """
    console.echo(message=f"Running assembly job for session '{session_name}' (ID: {job_id})...")
    tracker.start_job(job_id=job_id)

    try:
        # Discovers all data directory paths for the session.
        session_metadata = session_lookup[session_name]
        session_data_path = project_root.joinpath(session_metadata.animal, session_name)
        session_paths = _resolve_session_paths(session_data_path=session_data_path, dataset_name=dataset_name)

        # Dispatches the assembly to the pure computation function.
        _run_job(
            session_paths=session_paths,
            output_path=session_metadata.data_path,
        )

        tracker.complete_job(job_id=job_id)
        console.echo(message=f"Session '{session_name}' data assembly: Complete.", level=LogLevel.SUCCESS)

    except Exception as exception:
        tracker.fail_job(job_id=job_id, error_message=str(exception))
        raise


def _run_job(
    session_paths: _SessionPaths,
    output_path: Path,
) -> None:
    """Dispatches a single session assembly job to the dataset assembly function.

    Notes:
        This function is the atomic unit of work submitted to worker processes by the parallel execution path. It
        must remain importable at module level and accept only picklable arguments so that ``ProcessPoolExecutor``
        can dispatch it across process boundaries. The ``_SessionPaths`` frozen dataclass satisfies this constraint.
        Tracker state transitions, progress display, and error reporting are all handled by the parent process;
        this function performs pure computation and either returns ``None`` on success or propagates any raised
        exception back through the future.

    Args:
        session_paths: The resolved filesystem paths for the target session's data directories.
        output_path: The path to the output .feather file.
    """
    _assemble_session_dataset(
        session_paths=session_paths,
        output_path=output_path,
        progress=False,
    )
