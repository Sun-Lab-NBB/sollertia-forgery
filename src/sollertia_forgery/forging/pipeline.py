"""Provides the forging pipeline entry point that discovers session assembly jobs, validates the dataset, constructs
the processing graph, and executes jobs following the same pattern as the behavior processing pipeline.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING
from functools import partial
from contextlib import nullcontext
from concurrent.futures import Future, ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import polars as pl
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count, ensure_directory_exists
from sollertia_shared_assets import (
    DatasetData,
    SessionData,
    SessionTypes,
    DatasetTrackers,
    SessionMetadata,
    ProcessingTracker,
)

from .cindra import assemble_cindra_dataset
from .runtime import assemble_runtime_dataset, _mask_non_run_experiment_data
from .behavior import assemble_behavior_dataset
from ..shared_assets import prepare_tracker

if TYPE_CHECKING:
    from pathlib import Path


class DatasetTypes(IntEnum):
    """Stores the types of datasets currently supported by the Sollertia data processing workflow."""

    MESOSCOPE_VR_LICK_TRAINING = 1
    """Mesoscope-VR acquisition system + Lick training session type."""
    MESOSCOPE_VR_RUN_TRAINING = 2
    """Mesoscope-VR acquisition system + Run training session type."""
    MESOSCOPE_VR_EXPERIMENT = 3
    """Mesoscope-VR acquisition system + Mesoscope Experiment session type."""


TRACKER_FILENAME: str = DatasetTrackers.FORGING
"""The filename for the processing tracker placed in the dataset directory."""

FORGING_JOB_NAME: str = "session_assembly"
"""The job name used to identify session assembly jobs in forging processing trackers."""


def define_dataset(
    name: str,
    sessions: tuple[SessionMetadata, ...],
    project_root: Path,
) -> DatasetData:
    """Creates a new analysis dataset and initializes its data hierarchy.

    Notes:
        The dataset is created under the project's root directory, at the same level as animal directories. Sessions
        should be pre-filtered before being passed to this function. The project name, session type, and acquisition
        system are derived from the project root path and the first session's metadata.

    Args:
        name: The unique name for the dataset.
        sessions: The SessionMetadata instances representing the sessions to include in the dataset.
        project_root: The path to the project's root directory where the dataset hierarchy should be created.

    Returns:
        An initialized DatasetData instance that stores the structure and metadata of the created dataset.
    """
    # Derives the project name from the project's root directory path.
    project = project_root.name

    # Derives session type and acquisition system from the first session's metadata.
    first_session_path = project_root.joinpath(sessions[0].animal, sessions[0].session)
    first_session_data = SessionData.load(session_path=first_session_path)

    # Creates the dataset using the DatasetData class from sollertia-shared-assets.
    dataset = DatasetData.create(
        name=name,
        project=project,
        session_type=first_session_data.session_type,
        acquisition_system=first_session_data.acquisition_system,
        sessions=sessions,
        datasets_root=project_root,
    )

    console.echo(
        message=(
            f"Dataset's '{name}' data hierarchy: Defined with {len(sessions)} sessions from {len(dataset.animals)} "
            f"animals."
        ),
        level=LogLevel.SUCCESS,
    )

    return dataset


def run_forging_pipeline(
    dataset: DatasetData,
    project_root: Path,
    job_id: str | None = None,
    *,
    target_session: str | None = None,
    workers: int = -1,
    display_progress: bool = False,
) -> None:
    """Discovers, validates, and executes data assembly jobs for the target dataset's sessions.

    Notes:
        In local mode (job_id is None), all discovered session assembly jobs are distributed across a shared
        ``ProcessPoolExecutor`` whose size is resolved from the ``workers`` argument. Each session is an atomic unit
        dispatched to a worker process. The parent process owns tracker state transitions and aggregates worker
        outcomes as futures complete. Sequential execution is used automatically when ``workers`` resolves to 1 or
        only a single job is discovered. In remote mode (job_id is provided), only the job matching the provided ID
        is executed in-process without any worker pool.

    Args:
        dataset: The initialized DatasetData instance that stores the dataset's metadata.
        project_root: The path to the project's root directory that stores the animal and session data directories.
        job_id: The unique hexadecimal identifier for the processing job to execute. If provided, only the job
            matching this ID is executed (remote mode). If not provided, all available jobs are distributed across
            the worker pool with automatic tracker management (local mode).
        target_session: If provided, limits the assembly to the specified session only.
        workers: The number of worker processes to use for parallel processing. Setting this to a value less than 1
            uses all available CPU cores (minus reserved cores). Setting this to 1 conducts processing sequentially
            without spawning worker processes.
        display_progress: Determines whether to display a progress bar during processing.

    Raises:
        ValueError: If the target session is not found in the dataset, or if the provided job_id does not match any
            available jobs.
    """
    console.echo(
        message=f"Initializing the forging pipeline for dataset '{dataset.name}'...",
        level=LogLevel.INFO,
    )

    # Resolves the dataset path and discovers session assembly jobs.
    dataset_path = dataset.dataset_data_path.parent
    sessions = _discover_jobs(dataset=dataset, target_session=target_session)

    console.echo(message=f"Discovered {len(sessions)} assembly job(s).")

    # Prepares the processing tracker and aligns it with the discovered jobs.
    tracker = ProcessingTracker(file_path=dataset_path.joinpath(TRACKER_FILENAME))
    jobs = [(FORGING_JOB_NAME, session) for session in sessions]
    prepare_tracker(tracker=tracker, jobs=jobs)
    job_ids = {
        session: ProcessingTracker.generate_job_id(job_name=FORGING_JOB_NAME, specifier=session) for session in sessions
    }

    if job_id is not None:
        # Remote mode: resolves the session for the requested job ID and executes that single job in-process. The
        # remote path never spawns a worker pool.
        id_to_session: dict[str, str] = {jid: session for session, jid in job_ids.items()}

        if job_id not in id_to_session:
            message = (
                f"Unable to execute the requested job with ID '{job_id}'. The input identifier does not match "
                f"any jobs available for this dataset. Valid job IDs: {sorted(id_to_session.keys())}."
            )
            console.error(message=message, error=ValueError)

        _execute_job(
            dataset=dataset,
            session_name=id_to_session[job_id],
            project_root=project_root,
            tracker=tracker,
            job_id=job_id,
        )
    else:
        # Local mode: resolves the worker count and distributes all discovered jobs across a shared
        # ProcessPoolExecutor. Sequential execution is used automatically when the pool would contain a single
        # worker or only a single job is available.
        resolved_workers = resolve_worker_count(requested_workers=workers)

        if resolved_workers > 1 and len(sessions) > 1:
            _execute_jobs_parallel(
                dataset=dataset,
                sessions=sessions,
                project_root=project_root,
                tracker=tracker,
                job_ids=job_ids,
                workers=resolved_workers,
                display_progress=display_progress,
            )
        else:
            _execute_jobs_sequential(
                dataset=dataset,
                sessions=sessions,
                project_root=project_root,
                tracker=tracker,
                job_ids=job_ids,
                display_progress=display_progress,
            )

    console.echo(message="All forging jobs completed successfully.", level=LogLevel.SUCCESS)


def discover_forging_jobs(dataset: DatasetData, *, target_session: str | None = None) -> list[str]:
    """Discovers all session assembly jobs available for the target dataset.

    Factors out the discovery logic shared by ``run_forging_pipeline`` and the MCP batch-preparation tools so that
    external callers can inspect the job set without triggering execution or tracker initialization.

    Args:
        dataset: The initialized DatasetData instance that stores the dataset's metadata.
        target_session: If provided, limits the discovery to the specified session only.

    Returns:
        The ordered list of session names available for assembly.

    Raises:
        ValueError: If the target session is not found in the dataset.
    """
    return _discover_jobs(dataset=dataset, target_session=target_session)


def assemble_session_dataset(
    session_data_path: Path,
    session_multiday_path: Path,
    output_path: Path,
    dataset_type: DatasetTypes | int,
    *,
    progress: bool = False,
) -> None:
    """Assembles the requested analysis dataset for the target session.

    This function acts as the entry-point for all dataset assembly (forging) runtimes. It extracts, post-processes, and
    combines all relevant data for the processed session into a Polars DataFrame object and saves it to an uncompressed
    .feather file under the output_path directory.

    Args:
        session_data_path: The path to the session's processed data directory.
        session_multiday_path: The path to the session's multi-day data directory.
        output_path: The path to the directory where to save the assembled dataset as a .feather file.
        dataset_type: The type of the processed session. Must be one of the valid DatasetTypes enumeration members.
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
        # Experiment dataset.
        if dataset_type == DatasetTypes.MESOSCOPE_VR_EXPERIMENT:
            # First assembles the fluorescence data, which is needed to generate the reference time vector for other
            # datasets.
            with console.progress(total=3, description=f"Assembling session {session_data_path.stem} datasets") as pbar:
                fluorescence_data = assemble_cindra_dataset(
                    session_data_path=session_data_path, multiday_data_path=session_multiday_path
                )
                pbar.update(1)

                # Extracts reference time to assemble other datasets in parallel.
                reference_time = fluorescence_data["time_us"].to_numpy()

                # Defines tasks for parallel execution.
                tasks = {
                    "behavior": partial(
                        assemble_behavior_dataset,
                        session_data_path=session_data_path,
                        reference_time=reference_time,
                        drop_time_columns=True,
                    ),
                    "runtime": partial(
                        assemble_runtime_dataset,
                        session_data_path=session_data_path,
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

        # Behavior-only training dataset.
        elif dataset_type in (DatasetTypes.MESOSCOPE_VR_LICK_TRAINING, DatasetTypes.MESOSCOPE_VR_RUN_TRAINING):
            # Training session data is always aligned to the face camera frame acquisition time. Extracts the reference
            # timepoints from the face camera timestamp data.
            face_camera_path = session_data_path.joinpath(
                "processed_data", "behavior_data", "face_camera_timestamps.feather"
            )
            face_camera_df = pl.read_ipc(face_camera_path, memory_map=True)
            reference_time = face_camera_df["frame_time_us"].to_numpy()

            # Assembles and saves the behavior dataset to disk as an uncompressed .feather file (to support
            # memory-mapping).
            with console.progress(total=1, description=f"Assembling session {session_data_path.stem} datasets") as pbar:
                behavior_data = assemble_behavior_dataset(
                    session_data_path=session_data_path, reference_time=reference_time
                )
                behavior_data.write_ipc(file=output_path)
                pbar.update(1)

        # If the input dataset type is not supported, raises a ValueError.
        else:
            message = (
                f"Unsupported dataset type '{dataset_type}' encountered when assembling the dataset for the session "
                f"{session_data_path.stem}. Use one of the valid DatasetTypes enumeration members."
            )
            console.error(message=message, error=ValueError)
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


def _resolve_dataset_type(session_type: str | SessionTypes) -> DatasetTypes:
    """Maps a SessionTypes value to the corresponding DatasetTypes value.

    Args:
        session_type: The session type to map.

    Returns:
        The corresponding DatasetTypes value.

    Raises:
        ValueError: If the session type is not supported.
    """
    if session_type == SessionTypes.MESOSCOPE_EXPERIMENT:
        return DatasetTypes.MESOSCOPE_VR_EXPERIMENT
    if session_type == SessionTypes.RUN_TRAINING:
        return DatasetTypes.MESOSCOPE_VR_RUN_TRAINING
    if session_type == SessionTypes.LICK_TRAINING:
        return DatasetTypes.MESOSCOPE_VR_LICK_TRAINING
    message = (
        f"Unable to resolve the dataset type for session type '{session_type}'. The session type is not "
        f"supported by the forging pipeline."
    )
    console.error(message=message, error=ValueError)

    # Unreachable: console.error always raises when given an error class. Explicit raise satisfies the linter.
    raise ValueError(message)


def _execute_jobs_sequential(
    dataset: DatasetData,
    sessions: list[str],
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
        dataset: The initialized DatasetData instance.
        sessions: The ordered list of session names to assemble.
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
                dataset=dataset,
                session_name=session_name,
                project_root=project_root,
                tracker=tracker,
                job_id=job_ids[session_name],
            )
            if progress_bar is not None:
                progress_bar.update(1)


def _execute_jobs_parallel(
    dataset: DatasetData,
    sessions: list[str],
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
        dataset: The initialized DatasetData instance.
        sessions: The ordered list of session names to assemble.
        project_root: The path to the project's root directory.
        tracker: The ProcessingTracker instance for recording job state transitions.
        job_ids: The mapping from session name to job ID.
        workers: The resolved worker process count for the shared ProcessPoolExecutor.
        display_progress: Determines whether to display a progress bar during processing.
    """
    # Resolves the dataset type once for all sessions.
    dataset_type = _resolve_dataset_type(session_type=dataset.session_type)

    first_exception: Exception | None = None

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_job_id: dict[Future[None], str] = {}
        for session_name in sessions:
            job_id = job_ids[session_name]

            # Resolves the session's paths for the worker.
            session_metadata = next(smd for smd in dataset.sessions if smd.session == session_name)
            session_data_path = project_root.joinpath(session_metadata.animal, session_name)
            multiday_path = session_data_path.joinpath("processed_data", "mesoscope_data", "multiday", dataset.name)
            output_path = dataset.get_session_data(animal=session_metadata.animal, session=session_name).data_path

            console.echo(message=f"Running assembly job for session '{session_name}' (ID: {job_id})...")
            tracker.start_job(job_id=job_id)
            future = executor.submit(
                _run_job,
                session_data_path=session_data_path,
                session_multiday_path=multiday_path,
                output_path=output_path,
                dataset_type=dataset_type,
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
    dataset: DatasetData,
    session_name: str,
    project_root: Path,
    tracker: ProcessingTracker,
    job_id: str,
) -> None:
    """Executes a single session assembly job in-process with full tracker state management.

    Notes:
        Used by the remote execution path and by the sequential local execution path. The parallel execution path
        calls ``_run_job`` directly from worker processes and manages tracker state separately in the parent.

    Args:
        dataset: The initialized DatasetData instance.
        session_name: The name of the session whose data to assemble.
        project_root: The path to the project's root directory.
        tracker: The ProcessingTracker instance for recording job state transitions.
        job_id: The unique hexadecimal identifier for this processing job.

    Raises:
        ValueError: If the session type is not supported.
    """
    console.echo(message=f"Running assembly job for session '{session_name}' (ID: {job_id})...")
    tracker.start_job(job_id=job_id)

    try:
        # Resolves the dataset type from the session type.
        dataset_type = _resolve_dataset_type(session_type=dataset.session_type)

        # Finds the session's metadata and resolves all filesystem paths.
        session_metadata = next(smd for smd in dataset.sessions if smd.session == session_name)
        session_data_path = project_root.joinpath(session_metadata.animal, session_name)
        multiday_path = session_data_path.joinpath("processed_data", "mesoscope_data", "multiday", dataset.name)
        output_path = dataset.get_session_data(animal=session_metadata.animal, session=session_name).data_path

        # Dispatches the assembly to the pure computation function.
        _run_job(
            session_data_path=session_data_path,
            session_multiday_path=multiday_path,
            output_path=output_path,
            dataset_type=dataset_type,
        )

        tracker.complete_job(job_id=job_id)
        console.echo(message=f"Session '{session_name}' data assembly: Complete.", level=LogLevel.SUCCESS)

    except Exception:
        tracker.fail_job(job_id=job_id)
        raise


def _run_job(
    session_data_path: Path,
    session_multiday_path: Path,
    output_path: Path,
    dataset_type: DatasetTypes | int,
) -> None:
    """Dispatches a single session assembly job to the dataset assembly function.

    Notes:
        This function is the atomic unit of work submitted to worker processes by the parallel execution path. It
        must remain importable at module level and accept only picklable arguments so that ``ProcessPoolExecutor``
        can dispatch it across process boundaries. Tracker state transitions, progress display, and error
        reporting are all handled by the parent process; this function performs pure computation and either
        returns ``None`` on success or propagates any raised exception back through the future.

    Args:
        session_data_path: The path to the session's data directory.
        session_multiday_path: The path to the session's multi-day data directory.
        output_path: The path to the output .feather file.
        dataset_type: The type of the processed session.
    """
    assemble_session_dataset(
        session_data_path=session_data_path,
        session_multiday_path=session_multiday_path,
        output_path=output_path,
        dataset_type=dataset_type,
        progress=False,
    )
