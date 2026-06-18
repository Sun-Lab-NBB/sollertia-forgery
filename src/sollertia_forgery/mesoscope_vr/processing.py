"""Provides the end-to-end behavior processing pipeline entry point. Extracts the raw microcontroller log archives
in-process via the ataraxis-communication-interface binding, discovers the resulting runtime and microcontroller
module jobs, validates the session, and executes the jobs following the same pattern as the axvs, axci, and cindra
pipelines. Camera-timestamp extraction is handled separately by the video pipeline.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING
from contextlib import nullcontext
from concurrent.futures import Future, ProcessPoolExecutor, as_completed

from ataraxis_base_utilities import LogLevel, console, resolve_worker_count
from sollertia_shared_assets import (
    SessionData,
    SessionTypes,
    ProcessingTrackers,
    MesoscopeHardwareState,
    MesoscopeExperimentConfiguration,
)
from ataraxis_data_structures import ProcessingTracker

from .runtime import RUNTIME_SOURCE_ID, find_log_archive, process_runtime_data
from ..cross_system import prepare_tracker
from .microcontrollers import (
    is_module_eligible,
    find_module_feathers,
    parse_module_feather_name,
    process_microcontroller_data,
)

if TYPE_CHECKING:
    from pathlib import Path

PROCESSABLE_SESSION_TYPES: frozenset[SessionTypes] = frozenset(
    {
        SessionTypes.LICK_TRAINING,
        SessionTypes.RUN_TRAINING,
        SessionTypes.MESOSCOPE_EXPERIMENT,
    }
)
"""The set of session types that are eligible for behavior data processing. Exposed so that external tools (such as
the MCP batch preparation helpers) can filter discovered sessions without duplicating the eligibility rules."""


class BehaviorJobNames(StrEnum):
    """Defines the job type names used by the behavior processing pipeline."""

    RUNTIME = "runtime_processing"
    """Extracts acquisition system and runtime task data from system log NPZ archives."""
    MICROCONTROLLER = "microcontroller_processing"
    """Parses the microcontroller module feather files extracted in-process from the raw log archives."""


def run_behavior_processing_pipeline(
    session_path: Path,
    job_id: str | None = None,
    *,
    workers: int = -1,
    display_progress: bool = False,
) -> None:
    """Discovers, validates, and executes behavior data processing jobs for the target session.

    Notes:
        In local mode (job_id is None), all discoverable jobs are distributed across a shared
        ``ProcessPoolExecutor`` whose size is resolved from the ``workers`` argument. Each discovered job (runtime
        archive, camera feather, or microcontroller module feather) is an atomic unit dispatched to a worker
        process. The parent process owns tracker state transitions and aggregates worker outcomes as futures
        complete. Sequential execution is used automatically when ``workers`` resolves to 1 or only a single job is
        discovered. In remote mode (job_id is provided), only the job matching the provided ID is executed
        in-process without any worker pool.

        This pipeline saves all processed data under the session's /processed_data/behavior_data directory.

    Args:
        session_path: The path to the root session directory containing the session data hierarchy.
        job_id: The unique hexadecimal identifier for the processing job to execute. If provided, only the job
            matching this ID is executed (remote mode). If not provided, all available jobs are distributed across
            the worker pool with automatic tracker management (local mode).
        workers: The number of worker processes to use for parallel processing. Setting this to a value less than 1
            uses all available CPU cores (minus reserved cores). Setting this to 1 conducts processing sequentially
            without spawning worker processes.
        display_progress: Determines whether to display a progress bar during processing.

    Raises:
        ValueError: If the session type is not supported for behavior processing, if no processable jobs are
            discovered, or if the provided job_id does not match any discoverable job.
    """
    # Loads and validates the session data. The session's ``processed_data_path`` is used as the static root for
    # all behavior processing outputs, so the caller never passes an output directory.
    session = SessionData.load(session_path=session_path)

    if session.session_type not in PROCESSABLE_SESSION_TYPES:
        message = (
            f"Unable to process behavior data for session '{session.session_name}'. The session type "
            f"'{session.session_type}' is not supported for behavior processing. Supported session types: "
            f"{sorted(str(session_type) for session_type in PROCESSABLE_SESSION_TYPES)}."
        )
        console.error(message=message, error=ValueError)

    console.echo(
        message=f"Initializing behavior processing pipeline for session '{session.session_name}'...",
        level=LogLevel.INFO,
    )

    # Loads the hardware state configuration required for microcontroller module processing.
    hardware_state = _load_hardware_state(session=session)

    # Loads experiment configuration for experiment sessions (required for runtime data extraction).
    experiment_configuration = _load_experiment_configuration(session=session)

    # Extracts the raw microcontroller log archives into per-module feather files in-process via axci. This makes
    # the behavior pipeline end-to-end (raw .npz logs -> module feathers -> parsed behavior feathers) instead of
    # depending on a separate, pre-run extraction step.
    _extract_microcontroller_logs(session=session, workers=workers, display_progress=display_progress)

    # Discovers all available processing jobs based on files present in the session directory. The mapping
    # caches the feather file already resolved for each job, so the execution helpers can skip a second glob
    # pass, and iterating its keys yields the (job_name, specifier) tuples in discovery order.
    job_paths = _discover_jobs(session=session, hardware_state=hardware_state)

    if not job_paths:
        message = (
            f"Unable to process behavior data for session '{session.session_name}'. No processable files were "
            f"discovered in the session's raw or processed data directories."
        )
        console.error(message=message, error=ValueError)

    console.echo(message=f"Discovered {len(job_paths)} processing job(s).")

    # Creates the output directory structure and tracker, then aligns the tracker's job registry with the
    # discovered jobs. The same regeneration strategy is applied in both local and remote modes so that stale
    # or foreign tracker entries consistently trigger a reset rather than silently persisting across runs. The
    # ``behavior_data/`` subdirectory is always placed under the session's ``processed_data_path``, co-located
    # with the upstream ``camera_timestamps/`` and ``microcontroller_data/`` produced by axvs and axci.
    data_path = session.processed_data.behavior_data_path
    data_path.mkdir(parents=True, exist_ok=True)
    tracker = ProcessingTracker(file_path=data_path / ProcessingTrackers.BEHAVIOR)
    jobs = list(job_paths.keys())
    prepare_tracker(tracker=tracker, jobs=jobs)

    if job_id is not None:
        # Remote mode: resolves the (job_name, specifier) tuple for the requested job ID and executes that
        # single job in-process. The remote path never spawns a worker pool.
        id_to_job: dict[str, tuple[str, str]] = {
            ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier): (job_name, specifier)
            for job_name, specifier in jobs
        }

        if job_id not in id_to_job:
            message = (
                f"Unable to execute the requested job with ID '{job_id}'. The input identifier does not match "
                f"any jobs available for this session. Valid job IDs: {sorted(id_to_job.keys())}."
            )
            console.error(message=message, error=ValueError)

        job_name, specifier = id_to_job[job_id]
        _execute_job(
            job_name=job_name,
            specifier=specifier,
            input_path=job_paths[(job_name, specifier)],
            output_directory=data_path,
            tracker=tracker,
            hardware_state=hardware_state,
            experiment_configuration=experiment_configuration,
        )
    else:
        # Local mode: resolves the worker count and distributes all discovered jobs across a shared
        # ProcessPoolExecutor. Sequential execution is used automatically when the pool would contain a single
        # worker or only a single job is available.
        resolved_workers = resolve_worker_count(requested_workers=workers)

        if resolved_workers > 1 and len(job_paths) > 1:
            _execute_jobs_parallel(
                job_paths=job_paths,
                output_directory=data_path,
                tracker=tracker,
                hardware_state=hardware_state,
                experiment_configuration=experiment_configuration,
                workers=resolved_workers,
                display_progress=display_progress,
            )
        else:
            _execute_jobs_sequential(
                job_paths=job_paths,
                output_directory=data_path,
                tracker=tracker,
                hardware_state=hardware_state,
                experiment_configuration=experiment_configuration,
                display_progress=display_progress,
            )

    console.echo(message="All behavior processing jobs completed successfully.", level=LogLevel.SUCCESS)


def discover_behavior_jobs(session_path: Path) -> tuple[SessionData, list[tuple[str, str]]]:
    """Discovers all processable behavior jobs for the target session.

    Loads the session, verifies its type is eligible for behavior processing, loads the session's hardware state,
    and returns the ordered list of discovered ``(job_name, specifier)`` tuples. Factors out the discovery logic
    shared by ``run_behavior_processing_pipeline`` and the MCP batch-preparation tools so that external callers
    can inspect the job set without triggering execution or tracker initialization.

    Args:
        session_path: The path to the root session directory containing the session data hierarchy.

    Returns:
        A tuple of (session, jobs) where ``session`` is the loaded ``SessionData`` instance and ``jobs`` is
        the ordered list of ``(job_name, specifier)`` tuples yielded by discovery.

    Raises:
        ValueError: If the session type is not in ``PROCESSABLE_SESSION_TYPES``.
    """
    session = SessionData.load(session_path=session_path)

    if session.session_type not in PROCESSABLE_SESSION_TYPES:
        message = (
            f"Unable to discover behavior jobs for session '{session.session_name}'. The session type "
            f"'{session.session_type}' is not supported for behavior processing. Supported session types: "
            f"{sorted(str(session_type) for session_type in PROCESSABLE_SESSION_TYPES)}."
        )
        console.error(message=message, error=ValueError)

    hardware_state = _load_hardware_state(session=session)
    job_paths = _discover_jobs(session=session, hardware_state=hardware_state)

    return session, list(job_paths.keys())


def _load_hardware_state(session: SessionData) -> MesoscopeHardwareState:
    """Loads the MesoscopeHardwareState configuration from the session's raw data directory.

    Args:
        session: The loaded SessionData instance.

    Returns:
        The loaded MesoscopeHardwareState instance.

    Raises:
        FileNotFoundError: If no hardware state YAML file is found at the session's canonical location.
    """
    hardware_state_path = session.raw_data.hardware_state_path

    if not hardware_state_path.is_file():
        message = (
            f"Unable to load hardware state for session '{session.session_name}'. No hardware state YAML file was "
            f"found at '{hardware_state_path}'."
        )
        console.error(message=message, error=FileNotFoundError)

    return MesoscopeHardwareState.from_yaml(file_path=hardware_state_path)


def _load_experiment_configuration(session: SessionData) -> MesoscopeExperimentConfiguration | None:
    """Loads the MesoscopeExperimentConfiguration for experiment sessions or returns None otherwise.

    Args:
        session: The loaded SessionData instance.

    Returns:
        The loaded MesoscopeExperimentConfiguration instance for experiment sessions, or None for non-experiment
        sessions.
    """
    if session.session_type != SessionTypes.MESOSCOPE_EXPERIMENT:
        return None

    experiment_configuration_path = session.raw_data.experiment_configuration_path

    if not experiment_configuration_path.is_file():
        message = (
            f"Unable to load experiment configuration for session '{session.session_name}'. No experiment "
            f"configuration YAML file was found at '{experiment_configuration_path}'."
        )
        console.error(message=message, error=FileNotFoundError)

    return MesoscopeExperimentConfiguration.from_yaml(file_path=experiment_configuration_path)


def _extract_microcontroller_logs(
    session: SessionData,
    *,
    workers: int,
    display_progress: bool,
) -> None:
    """Extracts the raw microcontroller log archives into per-module feather files via ataraxis-communication-interface.

    Notes:
        Runs the axci log-processing binding in-process, reading the raw ``{controller_id}_log.npz`` archives from
        the session's raw behavior data directory and writing per-module feather files into the canonical
        ``processed_data/microcontroller_data`` directory consumed by the downstream module-parsing jobs. The
        extraction is driven by the session's ``extraction_configuration.yaml``, which is authored during
        acquisition and encodes the per-controller event codes to extract. The ataraxis-communication-interface
        import is deferred to call time so importing this module does not require the acquisition library.

        The extraction configuration is resolved from the session's raw behavior data directory (alongside the
        raw log archives). Confirm this location matches the acquisition convention on the target machine.

    Args:
        session: The loaded SessionData instance.
        workers: The number of worker processes the extraction binding may use. Set to -1 to use all available
            CPU cores (minus reserved cores).
        display_progress: Determines whether to display a progress bar during extraction.

    Raises:
        FileNotFoundError: If the extraction configuration is not present at the session's canonical location.
    """
    # Deferred import: the acquisition binding is only required when behavior processing actually runs.
    from ataraxis_communication_interface import (  # noqa: PLC0415
        EXTRACTION_CONFIGURATION_FILENAME,
        run_log_processing_pipeline,
    )

    log_directory = session.raw_data.behavior_data_path
    config_path = log_directory.joinpath(EXTRACTION_CONFIGURATION_FILENAME)
    if not config_path.is_file():
        message = (
            f"Unable to extract microcontroller logs for session '{session.session_name}'. No extraction "
            f"configuration was found at '{config_path}'. The extraction configuration is authored during "
            f"acquisition and defines the per-controller event codes the log-processing pipeline extracts."
        )
        console.error(message=message, error=FileNotFoundError)

    output_directory = session.processed_data.microcontroller_data_path
    output_directory.mkdir(parents=True, exist_ok=True)

    run_log_processing_pipeline(
        log_directory=log_directory,
        output_directory=output_directory,
        config=config_path,
        workers=workers,
        display_progress=display_progress,
    )


def _discover_jobs(
    session: SessionData,
    hardware_state: MesoscopeHardwareState,
) -> dict[tuple[str, str], Path]:
    """Discovers all available processing jobs based on files present in the session directories.

    Args:
        session: The loaded SessionData instance. Canonical subdirectories (``behavior_data_path`` for the raw
            runtime archive and ``microcontroller_data_path`` for the extracted module feathers) are queried
            directly to avoid recursive project scans.
        hardware_state: The hardware configuration used to filter microcontroller modules by eligibility.

    Returns:
        An ordered mapping from each (job_name, specifier) tuple to the absolute input file path resolved
        during discovery. Insertion order is preserved, so iterating ``.keys()`` yields the jobs in the same
        order they were discovered (runtime archive first, then microcontroller module feathers). Caching the
        resolved path on the mapping lets the execution helpers reuse it instead of running a second recursive
        glob over the session directory.
    """
    job_paths: dict[tuple[str, str], Path] = {}

    # Discovers the single runtime processing job, if a runtime log archive is present. The Mesoscope-VR
    # runtime DataLogger always writes to a fixed source ID, so there is at most one archive per session.
    archive_path = find_log_archive(data_directory=session.raw_data.behavior_data_path)
    if archive_path is not None:
        job_paths[(BehaviorJobNames.RUNTIME, RUNTIME_SOURCE_ID)] = archive_path

    # Discovers microcontroller processing jobs from the extracted module feather files, filtering out modules
    # whose hardware parameters are not configured.
    for feather_path in find_module_feathers(data_directory=session.processed_data.microcontroller_data_path):
        controller_id, module_type, module_id = parse_module_feather_name(feather_path=feather_path)
        if not is_module_eligible(module_type=module_type, module_id=module_id, hardware_state=hardware_state):
            continue
        specifier = f"{controller_id}-{module_type}-{module_id}"
        job_paths[(BehaviorJobNames.MICROCONTROLLER, specifier)] = feather_path

    return job_paths


def _execute_jobs_sequential(
    job_paths: dict[tuple[str, str], Path],
    output_directory: Path,
    tracker: ProcessingTracker,
    hardware_state: MesoscopeHardwareState,
    experiment_configuration: MesoscopeExperimentConfiguration | None,
    *,
    display_progress: bool,
) -> None:
    """Runs all discovered jobs sequentially in the parent process with an optional progress bar.

    Notes:
        Selected automatically when the resolved worker count is 1 or only a single job is discovered. Each job
        is fully owned by the parent process (tracker transitions, computation, and failure handling), so the
        first exception aborts the remaining jobs — matching the original single-threaded semantics.

    Args:
        job_paths: The mapping from (job_name, specifier) tuples to their resolved input file paths. Iteration
            order matches the discovery order produced by ``_discover_jobs``.
        output_directory: The path to the behavior data output directory.
        tracker: The ProcessingTracker instance for recording job state transitions.
        hardware_state: The hardware configuration for microcontroller module processing.
        experiment_configuration: The experiment configuration for runtime data extraction, or None for
            non-experiment sessions.
        display_progress: Determines whether to display a progress bar during processing.
    """
    progress_context = (
        console.progress(total=len(job_paths), description="Processing behavior data", unit="job")
        if display_progress
        else nullcontext()
    )

    with progress_context as progress_bar:
        for (job_name, specifier), input_path in job_paths.items():
            _execute_job(
                job_name=job_name,
                specifier=specifier,
                input_path=input_path,
                output_directory=output_directory,
                tracker=tracker,
                hardware_state=hardware_state,
                experiment_configuration=experiment_configuration,
            )
            if progress_bar is not None:
                progress_bar.update(1)


def _execute_jobs_parallel(
    job_paths: dict[tuple[str, str], Path],
    output_directory: Path,
    tracker: ProcessingTracker,
    hardware_state: MesoscopeHardwareState,
    experiment_configuration: MesoscopeExperimentConfiguration | None,
    workers: int,
    *,
    display_progress: bool,
) -> None:
    """Runs all discovered jobs concurrently across a shared ProcessPoolExecutor.

    Notes:
        Every discovered job is submitted to the pool as an atomic unit. The parent process calls
        ``tracker.start_job`` immediately before submitting each job's future, so tracker state advances in
        lockstep with dispatch and no job can be marked as running without also being dispatched. Results are
        then collected via ``as_completed`` and recorded against the tracker individually. In-flight futures are
        allowed to finish on failure rather than cancelling pending work, so the tracker state remains accurate
        for every dispatched job. After all futures resolve, the first captured exception is re-raised to
        propagate the failure to the caller. Job identifiers are derived deterministically from
        ``(job_name, specifier)`` via ``ProcessingTracker.generate_job_id`` so the parent never needs to cache
        or forward them separately.

    Args:
        job_paths: The mapping from (job_name, specifier) tuples to their resolved input file paths. Iteration
            order matches the discovery order produced by ``_discover_jobs``.
        output_directory: The path to the behavior data output directory.
        tracker: The ProcessingTracker instance for recording job state transitions.
        hardware_state: The hardware configuration for microcontroller module processing.
        experiment_configuration: The experiment configuration for runtime data extraction, or None for
            non-experiment sessions.
        workers: The resolved worker process count for the shared ProcessPoolExecutor.
        display_progress: Determines whether to display a progress bar during processing.
    """
    first_exception: Exception | None = None

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_job_id: dict[Future[None], str] = {}
        for (job_name, specifier), input_path in job_paths.items():
            job_id = ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier)
            console.echo(message=f"Running '{job_name}' job with specifier '{specifier}' (ID: {job_id})...")
            tracker.start_job(job_id=job_id)
            future = executor.submit(
                _run_job,
                job_name=job_name,
                input_path=input_path,
                output_directory=output_directory,
                hardware_state=hardware_state,
                experiment_configuration=experiment_configuration,
            )
            future_to_job_id[future] = job_id

        progress_context = (
            console.progress(total=len(job_paths), description="Processing behavior data", unit="job")
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
    job_name: str,
    specifier: str,
    input_path: Path,
    output_directory: Path,
    tracker: ProcessingTracker,
    hardware_state: MesoscopeHardwareState,
    experiment_configuration: MesoscopeExperimentConfiguration | None,
) -> None:
    """Executes a single processing job in-process with full tracker state management.

    Notes:
        Used by the remote execution path and by the sequential local execution path. The parallel execution path
        calls ``_run_job`` directly from worker processes and manages tracker state separately in the parent. The
        job identifier is derived deterministically from ``(job_name, specifier)`` via
        ``ProcessingTracker.generate_job_id``, so callers never need to materialize or forward the ID explicitly.

    Args:
        job_name: The job type name (runtime_processing, camera_processing, or microcontroller_processing).
        specifier: The job-specific specifier (system ID, camera source ID, or controller-type-id triple).
        input_path: The input file path resolved by ``_discover_jobs``. Reusing the cached path avoids a second
            recursive glob over the session directory.
        output_directory: The path to the behavior data output directory.
        tracker: The ProcessingTracker instance for recording job state transitions.
        hardware_state: The hardware configuration for microcontroller module processing.
        experiment_configuration: The experiment configuration for runtime data extraction, or None for
            non-experiment sessions.
    """
    job_id = ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier)
    console.echo(message=f"Running '{job_name}' job with specifier '{specifier}' (ID: {job_id})...")
    tracker.start_job(job_id=job_id)

    try:
        _run_job(
            job_name=job_name,
            input_path=input_path,
            output_directory=output_directory,
            hardware_state=hardware_state,
            experiment_configuration=experiment_configuration,
        )
        tracker.complete_job(job_id=job_id)

    except Exception as exception:
        tracker.fail_job(job_id=job_id, error_message=str(exception))
        raise


def _run_job(
    job_name: str,
    input_path: Path,
    output_directory: Path,
    hardware_state: MesoscopeHardwareState,
    experiment_configuration: MesoscopeExperimentConfiguration | None,
) -> None:
    """Dispatches a single processing job to the appropriate processing module.

    Notes:
        This function is the atomic unit of work submitted to worker processes by the parallel execution path. It
        must remain importable at module level and accept only picklable arguments so that ``ProcessPoolExecutor``
        can dispatch it across process boundaries. Tracker state transitions, progress display, and error
        reporting are all handled by the parent process; this function performs pure computation and either
        returns ``None`` on success or propagates any raised exception back through the future.

    Args:
        job_name: The job type name (runtime_processing, camera_processing, or microcontroller_processing).
        input_path: The input file path resolved by ``_discover_jobs`` for this job.
        output_directory: The path to the behavior data output directory.
        hardware_state: The hardware configuration used by the microcontroller processing path.
        experiment_configuration: The experiment configuration used by the runtime processing path, or None for
            non-experiment sessions.

    Raises:
        ValueError: If the ``job_name`` does not match any known behavior processing job type.
    """
    if job_name == BehaviorJobNames.RUNTIME:
        process_runtime_data(
            log_path=input_path,
            output_directory=output_directory,
            experiment_configuration=experiment_configuration,
        )

    elif job_name == BehaviorJobNames.MICROCONTROLLER:
        process_microcontroller_data(
            feather_path=input_path,
            output_directory=output_directory,
            hardware_state=hardware_state,
        )

    else:
        valid_names = sorted(str(name) for name in BehaviorJobNames)
        message = (
            f"Unable to dispatch processing job with name '{job_name}'. The input name does not match any known "
            f"behavior processing job type. Valid names: {valid_names}."
        )
        console.error(message=message, error=ValueError)
