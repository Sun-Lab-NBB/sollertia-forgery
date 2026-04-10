"""Provides the main behavior processing pipeline entry point that discovers available jobs, validates the session,
constructs the processing graph, and executes jobs following the same pattern as axvs, axci, and cindra pipelines.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from ataraxis_base_utilities import LogLevel, console
from sollertia_shared_assets import (
    SessionData,
    SessionTypes,
    MesoscopeHardwareState,
    MesoscopeExperimentConfiguration,
)
from ataraxis_data_structures import ProcessingTracker

from .camera import find_camera_feather, find_camera_feathers, extract_camera_source_id, process_camera_timestamps
from .runtime import find_log_archive, find_log_archives, process_runtime_data, extract_log_source_id
from .microcontrollers import (
    is_module_eligible,
    find_module_feather,
    find_all_module_feathers,
    parse_module_feather_name,
    process_microcontroller_data,
)

if TYPE_CHECKING:
    from pathlib import Path

BEHAVIOR_DATA_DIRECTORY: str = "behavior_data"
"""The name of the subdirectory created under the output path for behavior processing results. All tracker files and
processed feather outputs are written into this subdirectory."""

TRACKER_FILENAME: str = "behavior_processing_tracker.yaml"
"""The filename for the processing tracker placed in the behavior data output directory."""


class BehaviorJobNames(StrEnum):
    """Defines the job type names used by the behavior processing pipeline."""

    RUNTIME = "runtime_processing"
    """Extracts acquisition system and runtime task data from system log NPZ archives."""
    CAMERA = "camera_processing"
    """Processes pre-extracted camera timestamp feather files."""
    MICROCONTROLLER = "microcontroller_processing"
    """Processes pre-extracted microcontroller module feather files."""


_PROCESSABLE_SESSION_TYPES: frozenset[SessionTypes] = frozenset(
    {
        SessionTypes.LICK_TRAINING,
        SessionTypes.RUN_TRAINING,
        SessionTypes.MESOSCOPE_EXPERIMENT,
    }
)
"""The set of session types that are eligible for behavior data processing."""


def run_behavior_processing_pipeline(
    session_path: Path,
    output_directory: Path,
    job_id: str | None = None,
    *,
    workers: int = -1,  # noqa: ARG001
    display_progress: bool = True,  # noqa: ARG001
) -> None:
    """Discovers, validates, and executes behavior data processing jobs for the target session.

    Notes:
        Follows the same pipeline pattern as axvs, axci, and cindra: discover available files, validate the
        session type and acquisition system, construct the processing job graph, and execute. Supports both local
        and remote processing modes.

        In local mode (job_id is None), all discoverable jobs are executed sequentially with automatic tracker
        management. In remote mode (job_id is provided), only the job matching the provided ID is executed.

    Args:
        session_path: The path to the root session directory containing the session data hierarchy.
        output_directory: The path to the root output directory. A ``behavior_data/`` subdirectory is created
            automatically under this path, and all tracker and feather output files are written there.
        job_id: The unique hexadecimal identifier for the processing job to execute. If provided, only the job
            matching this ID is executed (remote mode). If not provided, all available jobs are run sequentially
            with automatic tracker management (local mode).
        workers: The number of worker processes to use for parallel processing. Setting this to a value less than 1
            uses all available CPU cores. Setting this to 1 conducts processing sequentially. Currently only
            affects runtime log reading performance.
        display_progress: Determines whether to display progress bars and status messages during processing.

    Raises:
        ValueError: If the session type is not supported for behavior processing, if no processable jobs are
            discovered, or if the provided job_id does not match any discoverable job.
    """
    # Loads and validates the session data.
    session = SessionData.load(session_path=session_path)

    if session.session_type not in _PROCESSABLE_SESSION_TYPES:
        message = (
            f"Unable to process behavior data for session '{session.session_name}'. The session type "
            f"'{session.session_type}' is not supported for behavior processing. Supported session types: "
            f"{sorted(str(session_type) for session_type in _PROCESSABLE_SESSION_TYPES)}."
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

    # Discovers all available processing jobs based on files present in the session directory.
    jobs = _discover_jobs(
        raw_data_path=session.raw_data_path,
        processed_data_path=session.processed_data_path,
        hardware_state=hardware_state,
    )

    if not jobs:
        message = (
            f"Unable to process behavior data for session '{session.session_name}'. No processable files were "
            f"discovered in the session's raw or processed data directories."
        )
        console.error(message=message, error=ValueError)

    console.echo(message=f"Discovered {len(jobs)} processing job(s).")

    # Creates the output directory structure and tracker.
    data_path = output_directory / BEHAVIOR_DATA_DIRECTORY
    data_path.mkdir(parents=True, exist_ok=True)
    tracker = ProcessingTracker(file_path=data_path / TRACKER_FILENAME)

    if job_id is not None:
        # Remote mode: generates all possible job IDs and executes only the matching job.
        all_job_ids = _generate_job_ids(jobs=jobs)
        id_to_job: dict[str, tuple[str, str]] = {generated_id: job for job, generated_id in all_job_ids.items()}

        if job_id not in id_to_job:
            message = (
                f"Unable to execute the requested job with ID '{job_id}'. The input identifier does not match "
                f"any jobs available for this session. Valid job IDs: {list(all_job_ids.values())}."
            )
            console.error(message=message, error=ValueError)

        job_name, specifier = id_to_job[job_id]
        _execute_job(
            job_name=job_name,
            specifier=specifier,
            job_id=job_id,
            session=session,
            output_directory=data_path,
            tracker=tracker,
            hardware_state=hardware_state,
            experiment_configuration=experiment_configuration,
        )
    else:
        # Local mode: initializes the tracker and runs all discovered jobs sequentially. The tracker generates
        # and returns job IDs from the (job_name, specifier) tuples.
        console.echo(message=f"Initializing processing tracker for {len(jobs)} job(s)...")
        returned_job_ids = tracker.initialize_jobs(jobs=jobs)

        for (job_name, specifier), current_job_id in zip(jobs, returned_job_ids, strict=True):
            _execute_job(
                job_name=job_name,
                specifier=specifier,
                job_id=current_job_id,
                session=session,
                output_directory=data_path,
                tracker=tracker,
                hardware_state=hardware_state,
                experiment_configuration=experiment_configuration,
            )

    console.echo(message="All behavior processing jobs completed successfully.", level=LogLevel.SUCCESS)


def _discover_jobs(
    raw_data_path: Path,
    processed_data_path: Path,
    hardware_state: MesoscopeHardwareState,
) -> list[tuple[str, str]]:
    """Discovers all available processing jobs based on files present in the session directories.

    Args:
        raw_data_path: The path to the session's raw data directory (searched for system log NPZ archives).
        processed_data_path: The path to the session's processed data directory (searched for pre-extracted
            camera and microcontroller feather files).
        hardware_state: The hardware configuration used to filter microcontroller modules by eligibility.

    Returns:
        A list of (job_name, specifier) tuples representing all discoverable and eligible jobs.
    """
    jobs: list[tuple[str, str]] = []

    # Discovers runtime processing jobs from system log NPZ archives.
    for archive_path in find_log_archives(data_directory=raw_data_path):
        source_id = extract_log_source_id(archive_path=archive_path)
        jobs.append((BehaviorJobNames.RUNTIME, source_id))

    # Discovers camera processing jobs from pre-extracted camera timestamp feather files.
    for feather_path in find_camera_feathers(data_directory=processed_data_path):
        source_id = str(extract_camera_source_id(feather_path=feather_path))
        jobs.append((BehaviorJobNames.CAMERA, source_id))

    # Discovers microcontroller processing jobs from pre-extracted module feather files.
    for feather_path in find_all_module_feathers(data_directory=processed_data_path):
        controller_id, module_type, module_id = parse_module_feather_name(feather_path=feather_path)

        # Filters out modules whose hardware parameters are not configured.
        if not is_module_eligible(module_type=module_type, module_id=module_id, hardware_state=hardware_state):
            continue

        specifier = f"{controller_id}-{module_type}-{module_id}"
        jobs.append((BehaviorJobNames.MICROCONTROLLER, specifier))

    return jobs


def _generate_job_ids(jobs: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """Generates unique hexadecimal job IDs for each job using ProcessingTracker.

    Args:
        jobs: The list of (job_name, specifier) tuples to generate IDs for.

    Returns:
        A dictionary mapping each (job_name, specifier) tuple to its unique hexadecimal job ID.
    """
    return {
        (job_name, specifier): ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier)
        for job_name, specifier in jobs
    }


def _execute_job(
    job_name: str,
    specifier: str,
    job_id: str,
    session: SessionData,
    output_directory: Path,
    tracker: ProcessingTracker,
    hardware_state: MesoscopeHardwareState,
    experiment_configuration: MesoscopeExperimentConfiguration | None,
) -> None:
    """Executes a single processing job with tracker state management.

    Args:
        job_name: The job type name (runtime_processing, camera_processing, or microcontroller_processing).
        specifier: The job-specific specifier (system ID, camera source ID, or controller-type-id triple).
        job_id: The unique hexadecimal job identifier.
        session: The loaded SessionData instance.
        output_directory: The path to the behavior data output directory.
        tracker: The ProcessingTracker instance for recording job state transitions.
        hardware_state: The hardware configuration for microcontroller module processing.
        experiment_configuration: The experiment configuration for runtime data extraction, or None for
            non-experiment sessions.
    """
    console.echo(message=f"Running '{job_name}' job with specifier '{specifier}' (ID: {job_id})...")
    tracker.start_job(job_id=job_id)

    try:
        if job_name == BehaviorJobNames.RUNTIME:
            log_path = find_log_archive(data_directory=session.raw_data_path, source_id=specifier)
            process_runtime_data(
                log_path=log_path,
                output_directory=output_directory,
                experiment_configuration=experiment_configuration,
            )

        elif job_name == BehaviorJobNames.CAMERA:
            camera_source_id = int(specifier)
            feather_path = find_camera_feather(data_directory=session.processed_data_path, source_id=camera_source_id)
            process_camera_timestamps(
                feather_path=feather_path, output_directory=output_directory, source_id=camera_source_id
            )

        elif job_name == BehaviorJobNames.MICROCONTROLLER:
            controller_id_str, module_type_str, module_id_str = specifier.split("-")
            controller_id = int(controller_id_str)
            module_type = int(module_type_str)
            module_id = int(module_id_str)

            feather_path = find_module_feather(
                data_directory=session.processed_data_path,
                controller_id=controller_id,
                module_type=module_type,
                module_id=module_id,
            )
            process_microcontroller_data(
                feather_path=feather_path,
                output_directory=output_directory,
                module_type=module_type,
                module_id=module_id,
                hardware_state=hardware_state,
            )

        tracker.complete_job(job_id=job_id)

    except Exception as exception:
        tracker.fail_job(job_id=job_id, error_message=str(exception))
        raise


def _load_hardware_state(session: SessionData) -> MesoscopeHardwareState:
    """Loads the MesoscopeHardwareState configuration from the session's raw data directory.

    Args:
        session: The loaded SessionData instance.

    Returns:
        The loaded MesoscopeHardwareState instance.

    Raises:
        FileNotFoundError: If no hardware state YAML file is found in the session's raw data directory.
    """
    # Searches for the hardware state YAML file in the raw data directory.
    candidates = sorted(session.raw_data_path.rglob("*hardware_state*.yaml"))

    if not candidates:
        message = (
            f"Unable to load hardware state for session '{session.session_name}'. No hardware state YAML file was "
            f"found in '{session.raw_data_path}'."
        )
        console.error(message=message, error=FileNotFoundError)

    return MesoscopeHardwareState.from_yaml(file_path=candidates[0])


def _load_experiment_configuration(session: SessionData) -> MesoscopeExperimentConfiguration | None:
    """Loads the MesoscopeExperimentConfiguration from the session's raw data directory if the session is an
    experiment session.

    Args:
        session: The loaded SessionData instance.

    Returns:
        The loaded MesoscopeExperimentConfiguration instance for experiment sessions, or None for non-experiment
        sessions.
    """
    if session.session_type != SessionTypes.MESOSCOPE_EXPERIMENT:
        return None

    # Searches for the experiment configuration YAML file in the raw data directory.
    candidates = sorted(session.raw_data_path.rglob("*experiment_configuration*.yaml"))

    if not candidates:
        message = (
            f"Unable to load experiment configuration for session '{session.session_name}'. No experiment "
            f"configuration YAML file was found in '{session.raw_data_path}'."
        )
        console.error(message=message, error=FileNotFoundError)

    return MesoscopeExperimentConfiguration.from_yaml(file_path=candidates[0])
