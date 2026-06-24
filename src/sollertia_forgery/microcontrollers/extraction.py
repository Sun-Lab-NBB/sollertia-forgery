"""Provides the extraction stage of the microcontroller processing pipeline, rebinding the
ataraxis-communication-interface log-processing assets to extract per-module data from raw controller log archives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from natsort import natsorted
from ataraxis_base_utilities import console
from ataraxis_communication_interface.microcontroller import (
    EXTRACTION_CONFIGURATION_FILENAME,
    MICROCONTROLLER_MANIFEST_FILENAME,
    ExtractionConfig,
    MicroControllerManifest,
    execute_job,
)

if TYPE_CHECKING:
    from pathlib import Path
    from concurrent.futures import ProcessPoolExecutor

    from sollertia_shared_assets import SessionData
    from ataraxis_data_structures import ProcessingTracker
    from ataraxis_communication_interface.microcontroller import ControllerExtractionConfig

_LOG_ARCHIVE_SUFFIX: str = "_log.npz"
"""The naming suffix of the raw controller log archives produced by the acquisition DataLogger. Matches the
ataraxis-communication-interface ``LOG_ARCHIVE_SUFFIX`` convention; each archive is named
``{controller_id}_log.npz``."""


def resolve_controllers(session: SessionData) -> dict[str, ControllerExtractionConfig]:
    """Resolves the per-controller extraction configurations for the target session.

    Notes:
        Loads the acquisition-time extraction configuration (the source of truth for which controllers, modules,
        and event codes to extract) from the session's raw behavior data directory, and validates every configured
        controller ID against the microcontroller manifest written alongside the log archives. The manifest check
        confirms the archives were produced by ataraxis-communication-interface, which also distinguishes the
        microcontroller controllers from the runtime DataLogger archive that shares the same directory.

    Args:
        session: The loaded session whose microcontroller logs are being processed.

    Returns:
        An ordered mapping from each configured controller ID (as a string) to its ControllerExtractionConfig.

    Raises:
        FileNotFoundError: If the extraction configuration or the microcontroller manifest is not present at the
            session's canonical raw behavior data location.
        ValueError: If a configured controller ID is not registered in the microcontroller manifest.
    """
    log_directory = session.raw_data.behavior_data_path

    config_path = log_directory.joinpath(EXTRACTION_CONFIGURATION_FILENAME)
    if not config_path.is_file():
        message = (
            f"Unable to resolve microcontroller controllers for session '{session.session_name}'. No extraction "
            f"configuration was found at '{config_path}'. The extraction configuration is authored during "
            f"acquisition and defines the per-controller event codes the extraction stage processes."
        )
        console.error(message=message, error=FileNotFoundError)

    manifest_path = log_directory.joinpath(MICROCONTROLLER_MANIFEST_FILENAME)
    if not manifest_path.is_file():
        message = (
            f"Unable to resolve microcontroller controllers for session '{session.session_name}'. No "
            f"microcontroller manifest was found at '{manifest_path}'. The manifest is required to confirm the log "
            f"archives were produced by ataraxis-communication-interface."
        )
        console.error(message=message, error=FileNotFoundError)

    config = ExtractionConfig.load(file_path=config_path)
    manifest = MicroControllerManifest.load(file_path=manifest_path)
    manifest_ids = {str(controller.id) for controller in manifest.controllers}

    controllers = {str(controller.controller_id): controller for controller in config.controllers}

    unregistered = natsorted(controller_id for controller_id in controllers if controller_id not in manifest_ids)
    if unregistered:
        message = (
            f"Unable to resolve microcontroller controllers for session '{session.session_name}'. The following "
            f"configured controller IDs are not registered in the microcontroller manifest: "
            f"{', '.join(unregistered)}. Registered IDs: {natsorted(manifest_ids)}."
        )
        console.error(message=message, error=ValueError)

    return controllers


def find_controller_archive(log_directory: Path, controller_id: str) -> Path | None:
    """Locates the raw log archive for a controller, if it is present under the log directory.

    Notes:
        Searches recursively for the ``{controller_id}_log.npz`` archive, mirroring how the
        ataraxis-communication-interface log reader resolves archives. Returns None when no archive is present so
        the pipeline can skip controllers whose logs were not staged, rather than failing the whole session.

    Args:
        log_directory: The session's raw behavior data directory holding the controller log archives.
        controller_id: The controller ID whose archive to locate.

    Returns:
        The path to the controller's log archive, or None if no matching archive exists.
    """
    if not log_directory.is_dir():
        return None
    matches = natsorted(log_directory.rglob(f"{controller_id}{_LOG_ARCHIVE_SUFFIX}"))
    return matches[0] if matches else None


def extract_controller(
    archive_path: Path,
    output_directory: Path,
    controller_id: str,
    controller_config: ControllerExtractionConfig,
    job_id: str,
    tracker: ProcessingTracker,
    *,
    workers: int,
    display_progress: bool,
    executor: ProcessPoolExecutor | None = None,
) -> None:
    """Extracts one controller's log archive into raw per-module feather files via ataraxis-communication-interface.

    Notes:
        Delegates to the acquisition library's ``execute_job`` binding, which reads the archive once, filters
        messages by the configured per-module event codes, writes a ``controller_{id}_module_{type}_{id}.feather``
        file per module that produced data, and manages this job's state on the passed-in tracker (start, complete,
        or fail). The output directory is created if it does not exist.

    Args:
        archive_path: The path to the controller's ``{controller_id}_log.npz`` archive.
        output_directory: The directory where the raw per-module feather files are written (the session's
            microcontroller data directory).
        controller_id: The controller ID whose archive is being extracted.
        controller_config: The controller's extraction configuration (its modules and per-module event codes).
        job_id: The hexadecimal identifier of this extraction job in the shared processing tracker.
        tracker: The shared processing tracker the extraction job records its state against.
        workers: The number of worker processes the extraction may use to parallelize message decoding within the
            archive. Set to a value less than 1 to use all available CPU cores (minus reserved cores).
        display_progress: Determines whether to display a progress bar during extraction.
        executor: An optional shared process pool to reuse for parallel message decoding, so a sequence of
            controller extractions does not create and tear down a pool per controller.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    execute_job(
        log_path=archive_path,
        output_directory=output_directory,
        source_id=controller_id,
        job_id=job_id,
        workers=workers,
        tracker=tracker,
        controller_config=controller_config,
        display_progress=display_progress,
        executor=executor,
    )
