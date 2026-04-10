"""Provides assets for defining and assembling analysis datasets from raw and processed session data.

Notes:
    Assets from this module form the foundation for all dataset forging pipelines. The pipeline supports both local
    and remote processing modes.
"""

from typing import TYPE_CHECKING

from sollertia_shared_assets import (
    DatasetData,
    SessionData,
    SessionTypes,
    DatasetTrackers,
    SessionMetadata,
    ProcessingTracker,
    ProcessingPipelines,
)
from ataraxis_base_utilities import LogLevel, console

from .data_assembly import DatasetTypes, assemble_report_dataset, assemble_session_dataset

if TYPE_CHECKING:
    from pathlib import Path


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


def _initialize_forging_tracker(dataset_path: Path, dataset_name: str, session_names: list[str]) -> dict[str, str]:
    """Initializes the processing tracker file for the dataset forging pipeline jobs.

    Notes:
        This function is used to process the data in the 'local' processing mode. During remote data processing, the
        tracker file is pre-generated before submitting the processing jobs to the remote compute server.

    Args:
        dataset_path: The path to the dataset's root directory.
        dataset_name: The name of the dataset being processed.
        session_names: The list of session names for which to create forging jobs.

    Returns:
        A dictionary mapping job names to their generated job IDs.
    """
    # Initializes the processing tracker for this pipeline.
    tracker = ProcessingTracker(file_path=dataset_path.joinpath(DatasetTrackers.FORGING))

    # Generates job IDs for each session forging job.
    job_ids: dict[str, str] = {}
    for session in session_names:
        job_name = f"{dataset_name}_{ProcessingPipelines.FORGING}_session_{session}"
        job_ids[job_name] = ProcessingTracker.generate_job_id(session_path=dataset_path, job_name=job_name)

    # Initializes all jobs in the tracker file.
    tracker.initialize_jobs(job_ids=list(job_ids.values()))

    return job_ids


def _execute_session_data_assembly(
    dataset: DatasetData,
    session_name: str,
    project_root: Path,
    job_id: str,
    tracker: ProcessingTracker,
    *,
    progress: bool,
) -> None:
    """Assembles the target session's processed data into the session-specific dataset .feather file.

    Args:
        dataset: The initialized DatasetData instance that stores the assembled dataset's metadata.
        session_name: The name of the session whose data to assemble.
        project_root: The path to the project's root directory that stores the animal and session data directories.
        job_id: The unique hexadecimal identifier for this processing job.
        tracker: The ProcessingTracker instance used to track the pipeline's runtime status.
        progress: Determines whether to display the assembly progress via a terminal progress bar.

    Raises:
        ValueError: If the session type is not supported.
    """
    console.echo(message=f"Running the data assembly job with ID {job_id} for the session '{session_name}'...")
    tracker.start_job(job_id=job_id)

    try:
        # Finds the session's metadata.
        session_metadata = next(smd for smd in dataset.sessions if smd.session == session_name)

        # Resolves the paths to all filesystem components used in the data assembly process.
        session_data_path = project_root.joinpath(session_metadata.animal, session_name)
        multiday_path = session_data_path.joinpath("processed_data", "mesoscope_data", "multiday", dataset.name)
        output_path = dataset.get_session_data(animal=session_metadata.animal, session=session_name).data_path

        # Determines the dataset type based on the processed session type.
        if dataset.session_type == SessionTypes.MESOSCOPE_EXPERIMENT:
            dataset_type = DatasetTypes.MESOSCOPE_VR_EXPERIMENT
        elif dataset.session_type == SessionTypes.RUN_TRAINING:
            dataset_type = DatasetTypes.MESOSCOPE_VR_RUN_TRAINING
        elif dataset.session_type == SessionTypes.LICK_TRAINING:
            dataset_type = DatasetTypes.MESOSCOPE_VR_LICK_TRAINING
        else:
            message = (
                f"Unable to assemble the '{dataset.name}' dataset, as it uses an unsupported type of data acquisition "
                f"sessions '{dataset.session_type}'."
            )
            console.error(message=message, error=ValueError)

        # Runs the session's data assembly pipeline.
        assemble_session_dataset(
            session_data_path=session_data_path,
            session_multiday_path=multiday_path,
            output_path=output_path,
            dataset_type=dataset_type,
            progress=progress,
        )

        tracker.complete_job(job_id=job_id)
        console.echo(message=f"Session '{session_name}' data assembly: Complete.", level=LogLevel.SUCCESS)

    except Exception:
        tracker.fail_job(job_id=job_id)
        raise


def assemble_dataset(
    dataset: DatasetData,
    project_root: Path,
    job_id: str | None = None,
    *,
    target_session: str | None = None,
    progress: bool = False,
) -> None:
    """Assembles the forged data for the target dataset's sessions.

    This function iterates over the sessions in the dataset and assembles each session's data into a unified
    data.feather file within the dataset hierarchy.

    Args:
        dataset: The initialized DatasetData instance that stores the dataset's metadata.
        project_root: The path to the project's root directory that stores the animal and session data directories.
        job_id: The unique hexadecimal identifier for the processing job to execute. If provided, only the job
            matching this ID is executed. If not provided, all requested jobs are run sequentially with automatic
            tracker management.
        target_session: If provided, limits the assembly to the specified session only.
        progress: Determines whether to display the assembly progress via a terminal progress bar.

    Raises:
        ValueError: If the target session is not found in the dataset or if the job_id does not match any available
            jobs.
    """
    # Resolves the dataset path.
    dataset_path = dataset.dataset_data_path.parent

    # Collects session names from the dataset.
    session_names = [s.session for s in dataset.sessions]

    # Filters to target session if specified.
    if target_session is not None:
        if target_session not in session_names:
            message = (
                f"Unable to assemble the data for the session '{target_session}'. The session is not found in the "
                f"'{dataset.name}' dataset."
            )
            console.error(message=message, error=ValueError)
        session_names = [target_session]

    # Initializes the tracker for the forging pipeline.
    tracker = ProcessingTracker(file_path=dataset_path.joinpath(DatasetTrackers.FORGING))

    # Determines the execution mode based on whether job_id is provided.
    if job_id is not None:
        # REMOTE mode: Finds the job name matching the provided job_id.
        all_job_ids = _initialize_forging_tracker(
            dataset_path=dataset_path, dataset_name=dataset.name, session_names=session_names
        )
        id_to_name: dict[str, str] = {v: k for k, v in all_job_ids.items()}

        if job_id not in id_to_name:
            tracker.fail_job(job_id=job_id)
            message = (
                f"Unable to execute the requested job with ID '{job_id}'. The input identifier does not match any "
                f"jobs available for this dataset. Use one of the valid job IDs: {list(all_job_ids.values())}."
            )
            console.error(message=message, error=ValueError)

        # Extracts the session from the job name.
        job_name = id_to_name[job_id]
        session_name = job_name.split("_session_")[-1]

        # Runs the assembly for the single session.
        _execute_session_data_assembly(
            dataset=dataset,
            session_name=session_name,
            project_root=project_root,
            job_id=job_id,
            tracker=tracker,
            progress=progress,
        )
    else:
        # LOCAL mode: Runs all sessions sequentially with automatic tracker management.
        console.echo(message=f"Initializing forging tracker for {len(session_names)} session(s)...")
        job_ids = _initialize_forging_tracker(
            dataset_path=dataset_path, dataset_name=dataset.name, session_names=session_names
        )

        for session_name in session_names:
            job_name = f"{dataset.name}_{ProcessingPipelines.FORGING}_session_{session_name}"
            session_job_id = job_ids[job_name]

            _execute_session_data_assembly(
                dataset=dataset,
                session_name=session_name,
                project_root=project_root,
                job_id=session_job_id,
                tracker=tracker,
                progress=progress,
            )


def _initialize_report_tracker(session_path: Path, session_name: str) -> str:
    """Initializes the processing tracker for the report assembly job.

    Notes:
        This function is used in LOCAL processing mode. During remote data processing, the tracker file is pre-generated
        before submitting the processing jobs to the remote compute server.

    Args:
        session_path: The path to the session's data directory.
        session_name: The name of the session being processed.

    Returns:
        The generated job ID for the report assembly job.
    """
    # Resolves the tracker path within the session's tracking_data directory.
    tracker_path = session_path.joinpath("tracking_data", "report_tracker.yaml")

    # Initializes the processing tracker for this pipeline.
    tracker = ProcessingTracker(file_path=tracker_path)

    # Generates the job ID for the report assembly job.
    job_name = f"{session_name}_report_assembly"
    job_id = ProcessingTracker.generate_job_id(session_path=session_path, job_name=job_name)

    # Initializes the job in the tracker file.
    tracker.initialize_jobs(job_ids=[job_id])

    return job_id


def assemble_report_data(
    session_path: Path,
    job_id: str | None = None,
    *,
    progress: bool = False,
) -> None:
    """Assembles a report dataset for the target session.

    This function generates a behavior report dataset containing synchronized camera timestamps, behavior data, and
    experiment data for the target session. The function supports both local and remote processing modes.

    Args:
        session_path: The path to the session's data directory.
        job_id: The unique hexadecimal identifier for the processing job. If provided (REMOTE mode), uses the
            pre-generated tracker. If None (LOCAL mode), initializes the tracker internally.
        progress: Determines whether to display the assembly progress via a terminal progress bar.
    """
    session_name = session_path.name
    tracker_path = session_path.joinpath("tracking_data", "report_tracker.yaml")

    # LOCAL mode: Initializes the tracker internally.
    if job_id is None:
        console.echo(message=f"Initializing report tracker for session '{session_name}'...")
        job_id = _initialize_report_tracker(session_path=session_path, session_name=session_name)

    # Loads the tracker and starts the job.
    tracker = ProcessingTracker(file_path=tracker_path)
    console.echo(message=f"Running the report assembly job with ID {job_id} for session '{session_name}'...")
    tracker.start_job(job_id=job_id)

    try:
        # Resolves the output path for the report dataset.
        output_path = session_path.joinpath("processed_data", "report_data", "report.feather")

        # Runs the report assembly.
        assemble_report_dataset(
            session_data_path=session_path,
            output_path=output_path,
            progress=progress,
        )

        tracker.complete_job(job_id=job_id)
        console.echo(message=f"Session '{session_name}' report assembly: Complete.", level=LogLevel.SUCCESS)

    except Exception:
        tracker.fail_job(job_id=job_id)
        raise
