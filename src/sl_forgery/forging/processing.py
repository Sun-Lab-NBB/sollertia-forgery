"""This module provides the assets for defining and assembling Sun lab analysis datasets from the raw and processed
session data. Assets from this module form the foundation for all dataset forging pipelines available from this
library. The pipeline supports both local and remote processing modes.
"""

from typing import TYPE_CHECKING

from sl_shared_assets import (
    DatasetData,
    SessionTypes,
    DatasetTrackers,
    SessionMetadata,
    ProcessingTracker,
    AcquisitionSystems,
    ProcessingPipelines,
)
from ataraxis_base_utilities import LogLevel, console

from .data_assembly import DatasetTypes, assemble_session_dataset
from ..shared_assets import filter_sessions

if TYPE_CHECKING:
    from pathlib import Path


def _generate_forging_job_id(dataset_path: Path, job_name: str) -> str:
    """Generates a unique processing job identifier for the dataset forging pipeline job.

    Args:
        dataset_path: The path to the dataset's root directory.
        job_name: The name of the job for which to generate the ID.

    Returns:
        The generated job ID for the forging job.
    """
    return ProcessingTracker.generate_job_id(session_path=dataset_path, job_name=job_name)


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
        job_ids[job_name] = _generate_forging_job_id(dataset_path=dataset_path, job_name=job_name)

    # Initializes all jobs in the tracker file.
    tracker.initialize_jobs(job_ids=list(job_ids.values()))

    return job_ids


def _execute_session_assembly(
    dataset: DatasetData,
    session_name: str,
    session_data_root: Path,
    job_id: str,
    tracker: ProcessingTracker,
    *,
    progress: bool,
) -> None:
    """Executes the data assembly for a single session.

    Args:
        dataset: The initialized DatasetData instance that stores the dataset's metadata.
        session_name: The name of the session to assemble.
        session_data_root: The path to the root directory that stores the session data directories.
        job_id: The unique hexadecimal identifier for this processing job.
        tracker: The ProcessingTracker instance used to track the pipeline's runtime status.
        progress: Determines whether to display the assembly progress via a terminal progress bar.

    Raises:
        ValueError: If the session is not found in the dataset or the session type is not supported.
    """
    console.echo(message=f"Running assembly job for session '{session_name}' with ID {job_id}...")
    tracker.start_job(job_id=job_id)

    try:
        # Finds the session metadata.
        session_meta = next((s for s in dataset.sessions if s.session == session_name), None)
        if session_meta is None:
            message = f"Session '{session_name}' not found in the dataset."
            console.error(message=message, error=ValueError)

        # Resolves paths.
        dataset_path = dataset.dataset_data_path.parent
        session_data_path = session_data_root.joinpath(session_meta.animal, session_name)
        multiday_path = dataset_path.joinpath(session_meta.animal, session_name)
        output_path = dataset.get_session_data(animal=session_meta.animal, session=session_name).data_path

        # Determines the dataset type based on the session type.
        if dataset.session_type == SessionTypes.MESOSCOPE_EXPERIMENT:
            dataset_type = DatasetTypes.MESOSCOPE_VR_EXPERIMENT
        elif dataset.session_type == SessionTypes.RUN_TRAINING:
            dataset_type = DatasetTypes.MESOSCOPE_VR_RUN_TRAINING
        elif dataset.session_type == SessionTypes.LICK_TRAINING:
            dataset_type = DatasetTypes.MESOSCOPE_VR_LICK_TRAINING
        else:
            message = f"Unsupported session type '{dataset.session_type}' for dataset assembly."
            console.error(message=message, error=ValueError)

        # Runs the assembly.
        assemble_session_dataset(
            session_data_path=session_data_path,
            session_multiday_path=multiday_path,
            output_path=output_path,
            dataset_type=dataset_type,
            progress=progress,
        )

        tracker.complete_job(job_id=job_id)
        console.echo(message=f"Session '{session_name}' assembly completed.", level=LogLevel.SUCCESS)

    except Exception:
        tracker.fail_job(job_id=job_id)
        raise


def define_dataset(
    name: str,
    project: str,
    session_type: SessionTypes,
    acquisition_system: AcquisitionSystems,
    sessions: set[SessionMetadata],
    datasets_root: Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    include_sessions: set[str] | None = None,
    exclude_sessions: set[str] | None = None,
    include_animals: set[str] | None = None,
    exclude_animals: set[str] | None = None,
) -> DatasetData:
    """Creates a new analysis dataset by filtering the provided sessions and initializing the dataset hierarchy.

    This function applies the filtering rules to the provided sessions using the filter_sessions utility from
    shared_assets, then creates a new DatasetData instance with the filtered sessions. The dataset's directory
    structure is created on disk.

    Args:
        name: The unique name for the dataset.
        project: The name of the project from which the dataset's sessions originate.
        session_type: The type of data acquisition sessions included in the dataset.
        acquisition_system: The name of the data acquisition system used to acquire all sessions included in the
            dataset.
        sessions: The set of SessionMetadata instances representing the sessions to be filtered and included.
        datasets_root: The path to the root directory where the dataset hierarchy should be created.
        start_date: The start date for the date range filter. Sessions recorded on or after this date are included.
        end_date: The end date for the date range filter. Sessions recorded on or before this date are included.
        include_sessions: A set of session names to include regardless of the date range.
        exclude_sessions: A set of session names to exclude from the results.
        include_animals: A set of animal names to include. If specified, only sessions from these animals are
            considered.
        exclude_animals: A set of animal names to exclude. Sessions from these animals are removed from the results.

    Returns:
        An initialized DatasetData instance that stores the structure and metadata of the created dataset.

    Raises:
        ValueError: If no sessions pass the filtering criteria.
    """
    # Applies the filtering rules using the shared_assets filter_sessions utility.
    filtered_sessions = filter_sessions(
        sessions=sessions,
        start_date=start_date,
        end_date=end_date,
        include_sessions=include_sessions,
        exclude_sessions=exclude_sessions,
        include_animals=include_animals,
        exclude_animals=exclude_animals,
        utc_timezone=True,
    )

    # Ensures at least one session passed the filtering.
    if not filtered_sessions:
        message = f"Unable to create the '{name}' dataset. No sessions passed the filtering criteria."
        console.error(message=message, error=ValueError)

    # Creates the dataset using the DatasetData class from sl-shared-assets.
    dataset = DatasetData.create(
        name=name,
        project=project,
        session_type=session_type,
        acquisition_system=acquisition_system,
        sessions=filtered_sessions,
        datasets_root=datasets_root,
    )

    console.echo(
        message=(
            f"Created dataset '{name}' with {len(filtered_sessions)} sessions from {len(dataset.animals)} animals."
        ),
        level=LogLevel.SUCCESS,
    )

    return dataset


def assemble_dataset(
    dataset: DatasetData,
    session_data_root: Path,
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
        session_data_root: The path to the root directory that stores the session data directories.
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
        _execute_session_assembly(
            dataset=dataset,
            session_name=session_name,
            session_data_root=session_data_root,
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

            _execute_session_assembly(
                dataset=dataset,
                session_name=session_name,
                session_data_root=session_data_root,
                job_id=session_job_id,
                tracker=tracker,
                progress=progress,
            )
