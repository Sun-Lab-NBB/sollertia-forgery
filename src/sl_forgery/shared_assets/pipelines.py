"""This module provides the pipeline tracker filename enumerations used by various data management, processing, and
analysis pipelines available from this library.
"""

from enum import StrEnum

from sl_shared_assets import SessionTypes, AcquisitionSystems, ProcessingPipelines
from ataraxis_base_utilities import LogLevel, console

from .manifest import ProjectManifest


def check_session_eligibility(
    manifest: ProjectManifest,
    project: str,
    session: str,
    pipeline: str,
    supported_systems: set[str | AcquisitionSystems],
    supported_sessions: set[str | SessionTypes],
    allow_reprocessing: bool = False,
    configuration_exists: bool = True,
) -> bool:
    """Checks whether the target session meets the eligibility criteria for being processed with the specified pipeline.

    This function aggregates common eligibility checks to streamline the process for all supported processing pipelines.

    Args:
        manifest: The ProjectManifest instance that stores the session's project metadata.
        project: The name of the session's project.
        session: The name (ID) of the session to be processed.
        pipeline: The processing pipeline to be used to process the session's data.
        supported_systems: A set of data acquisition systems that support this type of processing.
        supported_sessions: A set of session types that support this type of processing.
        allow_reprocessing: Determines whether to allow reprocessing already processed sessions.
        configuration_exists: Determines whether the required configuration file exists on the remote server. If the
            target pipeline requires a configuration file and this is set to False, the session is marked as ineligible.
            Callers should verify configuration file existence before calling this function.

    Returns:
        True if the session meets the eligibility criteria, False otherwise.
    """
    # Parses the target session data from the manifest file.
    session_data = manifest.get_session_info(session=session)
    session_type = session_data["type"][0]
    session_system = session_data["system"][0]
    animal = str(session_data["animal"][0])
    complete = session_data["complete"][0]

    # Determines whether the session has already been processed using the specified pipeline.
    prepared = True
    requires_configuration = False
    if pipeline == ProcessingPipelines.CHECKSUM:
        processed = bool(session_data["integrity"][0])
    elif pipeline == ProcessingPipelines.PREPARATION:
        processed = bool(session_data["prepared"][0])
    elif pipeline == ProcessingPipelines.ARCHIVING:
        prepared = bool(session_data["prepared"][0])
        processed = bool(session_data["archived"][0])
    elif pipeline == ProcessingPipelines.BEHAVIOR:
        prepared = bool(session_data["prepared"][0])
        processed = bool(session_data["behavior"][0])
    elif pipeline == ProcessingPipelines.SUITE2P:
        prepared = bool(session_data["prepared"][0])
        processed = bool(session_data["suite2p"][0])
        requires_configuration = True
    else:
        message = (
            f"Unable to construct the {pipeline} pipeline for the session '{session}' performed by the animal "
            f"'{animal}' for the '{project}' project. The pipeline '{pipeline}' is not supported. "
            f"Use one of the supported pipelines: {list(ProcessingPipelines)}. Skipping processing the session."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return False

    # Ensures that the pipeline's name is stored as a ProcessingPipelines instance.
    pipeline = ProcessingPipelines(pipeline)

    # If the session was acquired using a data acquisition system that does not support this type of processing,
    # skips processing the session.
    if session_system not in supported_systems:
        message = (
            f"Unable to construct the {pipeline} pipeline for the session '{session}' performed by the animal "
            f"'{animal}' for the '{project}' project. The session was acquired using the acquisition system "
            f"'{session_system},' which does not support this form of processing. Skipping processing the session."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return False

    # If the session type is not one of the supported types, skips processing the session.
    if session_type not in supported_sessions:
        message = (
            f"Unable to construct the {pipeline} pipeline for the session '{session}' performed by the animal "
            f"'{animal}' for the '{project}' project. The session is of type '{session_type},' which does not support "
            f"this form of processing. Skipping processing the session."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return False

    # Prevents processing incomplete sessions.
    if not complete:
        message = (
            f"Unable to construct the {pipeline} pipeline for the session '{session}' performed "
            f"by the animal '{animal}' for the '{project}' project. The session is marked as 'incomplete,' which "
            f"excludes it from all further data processing. To enable processing, manually mark it as 'complete' by "
            f"creating the 'telomere.bin' marker file in the session's 'raw_data' directory on the remote server and "
            f"setting the integrity_verification_tracker.yaml file to indicate that the verification was passed."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return False

    # If the session has already been processed and reprocessing is not allowed, skips processing the session.
    if processed and not allow_reprocessing:
        message = (
            f"Unable to construct the {pipeline} pipeline for the session '{session}' performed by the animal "
            f"'{animal}' for the '{project}' project. The session has already been processed with this pipeline "
            f"and reprocessing is disabled. To enable reprocessing, call this command with the '--reprocess (-r)' "
            f"flag."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return False

    # If the target processing pipeline requires the session data to be prepared, excludes any unprepared sessions from
    # processing.
    if not prepared:
        message = (
            f"Unable to construct the {pipeline} pipeline for the session '{session}' performed by the animal "
            f"'{animal}' for the '{project}' project. The pipeline requires the session data to be prepared for "
            f"processing before it can be executed. Call the project data processing CLI with the '--prepare (-p)' "
            f"flag to prepare the target session for processing."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return False

    # If the target processing pipeline requires a specific server-side configuration file, ensures that the file is
    # present at the expected remote server location.
    if requires_configuration and not configuration_exists:
        message = (
            f"Unable to construct the {pipeline} pipeline for the session '{session}' performed by the animal "
            f"'{animal}' for the '{project}' project. The target configuration file does not exist on the remote "
            f"server at the expected path."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        return False

    # The session is eligible for processing with this pipeline.
    return True


class ManagingTrackers(StrEnum):
    """Defines the filenames for tracker files used by data managing pipelines."""

    CHECKSUM = "checksum.yaml"
    """The tracker file used by the checksum resolution pipeline."""
    MANIFEST = "manifest.yaml"
    """The tracker file used by the project manifest generation pipeline."""


class ProcessingTrackers(StrEnum):
    """Defines the filenames for tracker files used by data processing pipelines."""

    SUITE2P = "suite2p.yaml"
    """The tracker file used by the suite2p processing pipeline."""
    BEHAVIOR = "behavior.yaml"
    """The tracker file used by the behavior extraction pipeline."""
    VIDEO = "video.yaml"
    """The tracker file used by the video (DeepLabCut) processing pipeline."""


class DatasetTrackers(StrEnum):
    """Defines the filenames for tracker files used by dataset forging and multi-day analysis pipelines."""

    FORGING = "forging.yaml"
    """The tracker file used by the dataset forging pipeline."""
    MULTIDAY = "multiday.yaml"
    """The tracker file used by the multi-day suite2p registration pipeline."""
