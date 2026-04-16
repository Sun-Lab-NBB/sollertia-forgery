"""Provides the forging pipeline entry point that defines the dataset hierarchy, discovers session assembly jobs,
constructs the processing graph, and executes jobs following the same pattern as the behavior processing pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from functools import partial
from contextlib import nullcontext
from dataclasses import dataclass
from concurrent.futures import Future, ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import polars as pl
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count, ensure_directory_exists
from sollertia_shared_assets import (
    DatasetData,
    SessionData,
    SessionTypes,
    DatasetSession,
)
from ataraxis_data_structures import ProcessingTracker, delete_directory

from .cindra import assemble_cindra_dataset
from .runtime import assemble_runtime_dataset, _mask_non_run_experiment_data
from .behavior import assemble_behavior_dataset
from ..processing import TRACKER_FILENAME as _BEHAVIOR_TRACKER_FILENAME
from ..shared_assets import prepare_tracker

if TYPE_CHECKING:
    from pathlib import Path


TRACKER_FILENAME: str = "forging.yaml"
"""The filename for the processing tracker placed in the dataset directory."""

DEFINITION_JOB_NAME: str = "dataset_definition"
"""The job name used to identify the dataset definition stage in forging processing trackers."""

FORGING_JOB_NAME: str = "session_data_assembly"
"""The job name used to identify per-session assembly stages in forging processing trackers."""

_CINDRA_TRACKER_FILENAME: str = "single_recording_tracker.yaml"
"""The tracker filename written by the cindra single-recording pipeline into the cindra output directory."""


@dataclass(frozen=True, slots=True)
class SessionPaths:
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
    sessions: tuple[DatasetSession, ...],
    project_root: Path,
    job_id: str | None = None,
    *,
    target_session: str | None = None,
    workers: int = -1,
    display_progress: bool = False,
    force_recreate: bool = False,
) -> None:
    """Defines the dataset hierarchy and executes data assembly jobs for the target sessions.

    Notes:
        The pipeline is a two-stage graph mirroring the cindra multi-day pipeline's discovery → extraction layering.
        Stage 1 (dataset definition) creates the dataset hierarchy or loads an existing one if a prior run has
        already populated it; stage 2 (per-session assembly) assembles analysis data for each session. Both stages
        are registered as distinct jobs in the processing tracker: a single definition job (keyed on the dataset
        name) and one assembly job per session. Dataset creation is currently limited to mesoscope experiment
        sessions. In local mode (job_id is None), definition is marked complete and every assembly job is then
        distributed across a shared ``ProcessPoolExecutor`` whose size is resolved from the ``workers`` argument.
        Sequential execution is used automatically when ``workers`` resolves to 1 or only a single assembly job is
        discovered. In remote mode (job_id is provided), the pipeline routes on the identifier: if it matches the
        definition job id, only the definition stage runs; if it matches an assembly job id, only that single
        assembly runs in-process without any worker pool.

    Args:
        name: The unique name for the dataset.
        sessions: The DatasetSession instances representing the sessions to include in the dataset. Ignored when the
            dataset already exists and ``force_recreate`` is False (the existing definition takes precedence).
        project_root: The path to the project's root directory that stores the animal and session data directories.
            The dataset hierarchy is also created under this directory.
        job_id: The unique hexadecimal identifier for the processing job to execute. If provided, only the job
            matching this ID is executed (remote mode). The identifier may target either the dataset definition
            stage or a single per-session assembly job. If not provided, all stages are executed with automatic
            tracker management (local mode).
        target_session: If provided, limits the assembly to the specified session only.
        workers: The number of worker processes to use for parallel processing. Setting this to a value less than 1
            uses all available CPU cores (minus reserved cores). Setting this to 1 conducts processing sequentially
            without spawning worker processes.
        display_progress: Determines whether to display a progress bar during processing.
        force_recreate: Determines whether to delete any existing dataset hierarchy before creating it fresh. Set
            this to True when extending, shrinking, or modifying the session set of an existing dataset. Any prior
            assembled data and the existing processing tracker are discarded.

    Raises:
        ValueError: If the first session's type is not MESOSCOPE_EXPERIMENT, if the target session is not found in
            the dataset, or if the provided job_id does not match any available jobs.
    """
    console.echo(
        message=f"Initializing the forging pipeline for dataset '{name}'...",
        level=LogLevel.INFO,
    )

    # Removes the existing dataset hierarchy if the caller explicitly requested recreation.
    dataset_directory = project_root.joinpath(name)
    if force_recreate and dataset_directory.exists():
        delete_directory(directory_path=dataset_directory)
        console.echo(
            message=f"Dataset '{name}': Removed existing hierarchy for recreation.",
            level=LogLevel.INFO,
        )

    # Creates the dataset hierarchy or loads an existing one (e.g., when invoked via the remote assembly path after
    # a prior 'define' job). Dataset creation is currently limited to mesoscope experiment sessions.
    try:
        first_session_path = project_root.joinpath(sessions[0].animal, sessions[0].session)
        first_session_data = SessionData.load(session_path=first_session_path)
        if first_session_data.session_type != SessionTypes.MESOSCOPE_EXPERIMENT:
            message = (
                f"Unable to define dataset '{name}'. Dataset creation is currently supported only for mesoscope "
                f"experiment sessions, but the first session's type resolved to "
                f"'{first_session_data.session_type}'."
            )
            console.error(message=message, error=ValueError)
        dataset = DatasetData.create(
            name=name,
            project=project_root.name,
            session_type=first_session_data.session_type,
            acquisition_system=first_session_data.acquisition_system,
            sessions=sessions,
            datasets_root=project_root,
        )
        console.echo(
            message=(
                f"Dataset '{name}' data hierarchy: Defined with {len(sessions)} sessions from "
                f"{len(dataset.animals)} animals."
            ),
            level=LogLevel.SUCCESS,
        )
    except FileExistsError:
        dataset = DatasetData.load(dataset_path=dataset_directory)

    # Resolves the dataset path and discovers session assembly jobs.
    dataset_path = dataset.dataset_data_path.parent
    session_names = _discover_jobs(dataset=dataset, target_session=target_session)

    console.echo(message=f"Discovered {len(session_names)} assembly job(s).")

    # Builds a session metadata lookup for O(1) access per session.
    session_lookup: dict[str, DatasetSession] = {smd.session: smd for smd in dataset.sessions}

    # Prepares the processing tracker and registers both pipeline stages: one dataset definition job (keyed on the
    # dataset name) followed by one assembly job per session. Mirrors the cindra multi-day pipeline's discovery →
    # extraction layering.
    tracker = ProcessingTracker(file_path=dataset_path.joinpath(TRACKER_FILENAME))
    jobs = [(DEFINITION_JOB_NAME, name)] + [(FORGING_JOB_NAME, session) for session in session_names]
    prepare_tracker(tracker=tracker, jobs=jobs)

    definition_job_id = ProcessingTracker.generate_job_id(job_name=DEFINITION_JOB_NAME, specifier=name)
    job_ids = {
        session: ProcessingTracker.generate_job_id(job_name=FORGING_JOB_NAME, specifier=session)
        for session in session_names
    }

    # Marks the definition stage complete. The dataset hierarchy is guaranteed to exist by this point (freshly
    # created or loaded above).
    tracker.start_job(job_id=definition_job_id)
    tracker.complete_job(job_id=definition_job_id)

    if job_id is not None:
        # Remote mode: routes on the identifier. The definition stage has already been marked complete above, so a
        # definition-targeting job_id simply returns. An assembly-targeting job_id executes that single session's
        # assembly in-process.
        if job_id == definition_job_id:
            console.echo(
                message=f"Dataset '{name}' definition stage: Complete.",
                level=LogLevel.SUCCESS,
            )
            return

        id_to_session: dict[str, str] = {jid: session for session, jid in job_ids.items()}

        if job_id not in id_to_session:
            valid_ids = [definition_job_id, *sorted(id_to_session.keys())]
            message = (
                f"Unable to execute the requested job with ID '{job_id}'. The input identifier does not match "
                f"any jobs available for this dataset. Valid job IDs: {valid_ids}."
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

        if resolved_workers > 1 and len(session_names) > 1:
            _execute_jobs_parallel(
                sessions=session_names,
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
                sessions=session_names,
                session_lookup=session_lookup,
                dataset_name=dataset.name,
                project_root=project_root,
                tracker=tracker,
                job_ids=job_ids,
                display_progress=display_progress,
            )

    console.echo(message="All forging jobs completed successfully.", level=LogLevel.SUCCESS)


def assemble_session_dataset(
    session_paths: SessionPaths,
    output_path: Path,
    *,
    progress: bool = False,
) -> None:
    """Assembles the experiment analysis dataset for the target session.

    This function acts as the entry-point for all dataset assembly (forging) runtimes. It extracts, post-processes, and
    combines all relevant data for the processed session into a Polars DataFrame object and saves it to an uncompressed
    .feather file at the output_path.

    Args:
        session_paths: The resolved filesystem paths for the target session's data directories.
        output_path: The path to the .feather file where to save the assembled dataset.
        progress: Determines whether to display the session's data assembly progress via the terminal progress bar.
    """
    # Ensures that the output directory exists.
    ensure_directory_exists(output_path)

    # Configures progress bar visibility based on the progress parameter.
    _prior_progress = console.progress_enabled
    if progress:
        console.enable_progress()
    else:
        console.disable_progress()

    try:
        # First assembles the fluorescence data, which is needed to generate the reference time vector for other
        # datasets.
        with console.progress(
            total=3, description=f"Assembling session {session_paths.behavior_data_path.parent.stem} datasets"
        ) as pbar:
            fluorescence_data = assemble_cindra_dataset(
                cindra_data_path=session_paths.cindra_data_path,
                behavior_data_path=session_paths.behavior_data_path,
                multiday_data_path=session_paths.multiday_data_path,
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
                    raw_data_path=session_paths.raw_data_path,
                    reference_time=reference_time,
                ),
            }

            # Executes the processing in parallel.
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

        # Post-processing: masks cue, trial, and trial_type with 255 (or "undefined") for non-run experiment states.
        result = _mask_non_run_experiment_data(result)

        # Saves the unified dataset to disk as an uncompressed .feather file (to support memory-mapping).
        result.write_ipc(file=output_path)
    finally:
        # Restores the previous progress bar visibility state.
        if _prior_progress:
            console.enable_progress()
        else:
            console.disable_progress()


def _discover_jobs(dataset: DatasetData, *, target_session: str | None = None) -> list[str]:
    """Discovers all session assembly jobs available for the target dataset.

    Args:
        dataset: The initialized DatasetData instance that stores the dataset's metadata.
        target_session: If provided, limits the discovery to the specified session only.

    Returns:
        The ordered list of session names available for assembly.

    Raises:
        ValueError: If the target session is not found in the dataset.
    """
    session_names = [s.session for s in dataset.sessions]

    if target_session is not None:
        if target_session not in session_names:
            message = (
                f"Unable to assemble the data for the session '{target_session}'. The session is not found in the "
                f"'{dataset.name}' dataset."
            )
            console.error(message=message, error=ValueError)
        return [target_session]

    return session_names


def _resolve_session_paths(session_data_path: Path, dataset_name: str) -> SessionPaths:
    """Discovers and resolves all data directory paths for a single session assembly job.

    Notes:
        Loads ``SessionData`` to obtain the canonical ``raw_data_path`` and ``processed_data_path``, then uses
        tracker-file rglob within ``processed_data_path`` to discover the behavior and cindra output directories.
        The multiday path is derived from the cindra directory's parent (``mesoscope_data/``) by joining the
        dataset name, since the multiday tracker is only stored in the first session of each dataset.

    Args:
        session_data_path: The path to the session's root directory.
        dataset_name: The name of the dataset being assembled, used to resolve the multiday output directory.

    Returns:
        A frozen ``SessionPaths`` instance containing all resolved data directory paths.

    Raises:
        FileNotFoundError: If a required tracker file is not found under the processed data directory.
        RuntimeError: If multiple instances of a tracker file are found, indicating an ambiguous directory structure.
    """
    # Loads the session's metadata to obtain canonical raw and processed data root paths.
    session = SessionData.load(session_path=session_data_path)

    # Discovers the behavior data directory by locating its processing tracker.
    behavior_candidates = sorted(session.processed_data_path.rglob(_BEHAVIOR_TRACKER_FILENAME))
    if len(behavior_candidates) != 1:
        message = (
            f"Unable to resolve the behavior data directory for session '{session_data_path.name}'. "
            f"Expected exactly one '{_BEHAVIOR_TRACKER_FILENAME}' under '{session.processed_data_path}', "
            f"but found {len(behavior_candidates)}."
        )
        console.error(message=message, error=FileNotFoundError if not behavior_candidates else RuntimeError)
    behavior_data_path = behavior_candidates[0].parent

    # Discovers the cindra single-day output directory by locating its processing tracker.
    cindra_candidates = sorted(session.processed_data_path.rglob(_CINDRA_TRACKER_FILENAME))
    if len(cindra_candidates) != 1:
        message = (
            f"Unable to resolve the cindra data directory for session '{session_data_path.name}'. "
            f"Expected exactly one '{_CINDRA_TRACKER_FILENAME}' under '{session.processed_data_path}', "
            f"but found {len(cindra_candidates)}."
        )
        console.error(message=message, error=FileNotFoundError if not cindra_candidates else RuntimeError)
    cindra_data_path = cindra_candidates[0].parent

    # Derives the multiday output path from the cindra directory's parent (mesoscope_data/) and the dataset name.
    multiday_data_path = cindra_data_path.parent.joinpath("multiday", dataset_name)

    return SessionPaths(
        behavior_data_path=behavior_data_path,
        raw_data_path=session.raw_data_path,
        cindra_data_path=cindra_data_path,
        multiday_data_path=multiday_data_path,
    )


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
            session_paths = _resolve_session_paths(
                session_data_path=session_data_path, dataset_name=dataset_name
            )
            output_path = session_metadata.session_path.joinpath("data.feather")

            console.echo(message=f"Running assembly job for session '{session_name}' (ID: {job_id})...")
            tracker.start_job(job_id=job_id)
            future = executor.submit(
                _run_job,
                session_paths=session_paths,
                output_path=output_path,
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
        session_paths = _resolve_session_paths(
            session_data_path=session_data_path, dataset_name=dataset_name
        )
        output_path = session_metadata.session_path.joinpath("data.feather")

        # Dispatches the assembly to the pure computation function.
        _run_job(
            session_paths=session_paths,
            output_path=output_path,
        )

        tracker.complete_job(job_id=job_id)
        console.echo(message=f"Session '{session_name}' data assembly: Complete.", level=LogLevel.SUCCESS)

    except Exception:
        tracker.fail_job(job_id=job_id)
        raise


def _run_job(
    session_paths: SessionPaths,
    output_path: Path,
) -> None:
    """Dispatches a single session assembly job to the dataset assembly function.

    Notes:
        This function is the atomic unit of work submitted to worker processes by the parallel execution path. It
        must remain importable at module level and accept only picklable arguments so that ``ProcessPoolExecutor``
        can dispatch it across process boundaries. The ``SessionPaths`` frozen dataclass satisfies this constraint.
        Tracker state transitions, progress display, and error reporting are all handled by the parent process;
        this function performs pure computation and either returns ``None`` on success or propagates any raised
        exception back through the future.

    Args:
        session_paths: The resolved filesystem paths for the target session's data directories.
        output_path: The path to the output .feather file.
    """
    assemble_session_dataset(
        session_paths=session_paths,
        output_path=output_path,
        progress=False,
    )
