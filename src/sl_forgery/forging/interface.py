"""This module provides the interface functions for the Sun lab dataset forging pipelines. The assets from this module
are designed to forge (assemble) and post-process analysis datasets from the processed data stored on the Sun lab's
remote compute server.
"""

from typing import TYPE_CHECKING

from sl_shared_assets import (
    SessionTypes,
    DatasetTrackers,
    ProcessingTracker,
    AcquisitionSystems,
    ProcessingPipelines,
    get_working_directory,
)

from ..server import Job, Server, ProcessingPipeline, get_remote_job_work_directory
from ..shared_assets import check_session_eligibility

if TYPE_CHECKING:
    from ..managing import ProjectManifest


def _construct_suite2p_multiday_pipeline(
    manifest: ProjectManifest,
    project: str,
    animal: str,
    sessions: list[str],
    dataset_name: str,
    server: Server,
    *,
    configuration_file: str = "GCaMP6f_CA1_MD.yaml",
    reprocess: bool = False,
    keep_job_logs: bool = False,
) -> ProcessingPipeline | str:
    """Generates and returns the ProcessingPipeline instance used to execute the multi-day suite2p processing pipeline
    for the target set of sessions.

    Args:
        manifest: The initialized ProjectManifest instance that stores the session's project metadata.
        project: The name of the project for which to execute the target processing pipeline.
        animal: The unique identifier of the animal that performed the sessions processed by this pipeline.
        sessions: The list of session names to process using the target processing pipeline.
        dataset_name: The name of the dataset to process.
        server: The Server class instance that manages access to the remote server that executes the pipeline and
            stores the target session's data.
        configuration_file: The name of the configuration file stored on the remote compute server that contains the
            data-specific processing parameters for the sl-suite2p multi-day pipeline.
        reprocess: Determines whether to reprocess the dataset if it has already been processed with this pipeline.
        keep_job_logs: Determines whether to keep completed job logs on the server or (default) remove them after
            runtime. If any job of the pipeline fails, the logs for all jobs are kept regardless of this argument's
            value.

    Returns:
        The configured ProcessingPipeline instance if all sessions can be processed with this pipeline.
        Otherwise, returns a string describing why the dataset was excluded from processing.
    """
    # Resolves the path to the local Sun lab working directory.
    local_working_directory = get_working_directory()

    # Resolves the configuration path on the remote server.
    configuration_path = server.suite2p_configurations_directory.joinpath(configuration_file)

    # Validates each session for eligibility with the multiday pipeline.
    for session in sessions:
        exclusion_reason = check_session_eligibility(
            manifest=manifest,
            session=session,
            pipeline=ProcessingPipelines.MULTIDAY,
            server=server,
            supported_systems={AcquisitionSystems.MESOSCOPE_VR},
            supported_sessions={SessionTypes.MESOSCOPE_EXPERIMENT},
            allow_reprocessing=reprocess,
            configuration_path=configuration_path,
        )
        if exclusion_reason is not None:
            return f"Session '{session}': {exclusion_reason}"

    # Resolves the dataset path and session data root for the processed set of sessions.
    remote_dataset_path = server.user_working_root.joinpath(project, dataset_name)
    session_data_root = server.user_working_root.joinpath(project)

    # Precreates the iterables to store stage jobs.
    stage_1 = []
    stage_2 = []

    # Stage 1: Multi-day cell tracking (discovery).
    job_name = f"{dataset_name}_ss2p_discovery"
    job_id = ProcessingTracker.generate_job_id(session_path=remote_dataset_path, job_name=job_name)
    working_directory = get_remote_job_work_directory(
        server=server, job_name=job_name, pipeline_name=ProcessingPipelines.MULTIDAY
    )
    job = Job(
        job_name=job_name,
        output_log=working_directory.joinpath("output.txt"),
        error_log=working_directory.joinpath("errors.txt"),
        working_directory=working_directory,
        conda_environment="suite2p",
        cpu_threads=30,
        ram=80,
        time=180,
    )
    job.add_command(
        f"ss2p run -i {configuration_path} -w -1 multi-day -dp {remote_dataset_path} "
        f"-sdr {session_data_root} -id {job_id} -d"
    )
    stage_1.append((job, working_directory))

    # Stage 2: Across-day-tracked cell fluorescence extraction.
    for session in sessions:
        job_name = f"{dataset_name}_ss2p_extraction_session_{session}"
        job_id = ProcessingTracker.generate_job_id(session_path=remote_dataset_path, job_name=job_name)
        working_directory = get_remote_job_work_directory(
            server=server, job_name=job_name, pipeline_name=ProcessingPipelines.MULTIDAY
        )
        server.create(remote_path=working_directory, is_dir=True)
        job = Job(
            job_name=job_name,
            output_log=working_directory.joinpath("output.txt"),
            error_log=working_directory.joinpath("errors.txt"),
            working_directory=working_directory,
            conda_environment="suite2p",
            cpu_threads=30,
            ram=80,
            time=180,
        )
        job.add_command(
            f"ss2p run -i {configuration_path} -w -1 multi-day -dp {remote_dataset_path} "
            f"-sdr {session_data_root} -id {job_id} -e -t {session}"
        )
        stage_2.append((job, working_directory))

    # Resolves the paths to the local and remote job tracker files.
    remote_tracker_path = remote_dataset_path.joinpath(DatasetTrackers.MULTIDAY)
    local_tracker_path = local_working_directory.joinpath(project, dataset_name, DatasetTrackers.MULTIDAY)

    # Packages job data into a ProcessingPipeline object and returns it to the caller.
    return ProcessingPipeline(
        pipeline=ProcessingPipelines.MULTIDAY,
        server=server,
        data_path=remote_dataset_path,
        jobs={1: tuple(stage_1), 2: tuple(stage_2)},
        remote_tracker_path=remote_tracker_path,
        local_tracker_path=local_tracker_path,
        session=dataset_name,
        animal=animal,
        project=project,
        keep_job_logs=keep_job_logs,
    )
