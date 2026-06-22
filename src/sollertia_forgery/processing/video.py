"""Provides the system-agnostic camera-timestamp extraction pipeline that converts raw VideoSystem log archives
into canonical per-camera timestamp feathers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from contextlib import nullcontext

import numpy as np
import polars as pl
from ataraxis_base_utilities import LogLevel, console
from sollertia_shared_assets import SessionData
from ataraxis_data_structures import ProcessingTracker

from ..shared_assets import prepare_tracker

if TYPE_CHECKING:
    from pathlib import Path

VIDEO_JOB_NAME: str = "camera_timestamp_extraction"
"""The job name used to identify camera-timestamp extraction jobs in the camera processing tracker."""

_RAW_CAMERA_LOG_PATTERN: str = "*_log.npz"
"""The glob pattern used to discover raw VideoSystem camera log archives. Each archive is named
``{source_id}_log.npz``, matching the DataLogger source-ID naming convention shared across the Sollertia stack."""

_RAW_CAMERA_LOG_PART_COUNT: int = 2
"""The expected number of underscore-delimited components in a ``{source_id}_log`` archive stem."""

_FRAME_TIME_COLUMN: str = "frame_time_us"
"""The output column name holding the per-frame acquisition timestamps, in microseconds since the UTC epoch. This
matches the canonical camera-timestamp column recognized across the downstream forging pipelines."""

_CAMERA_TIMESTAMP_SUFFIX: str = "_timestamps.feather"
"""The suffix appended to each camera's manifest name to form its canonical timestamp feather filename within the
behavior data directory (e.g., the ``face_camera`` source produces ``face_camera_timestamps.feather``)."""


def find_camera_logs(data_directory: Path) -> list[Path]:
    """Discovers raw VideoSystem camera log archives inside the canonical raw camera data directory.

    Args:
        data_directory: The path to the session's raw camera data directory
            (``session.raw_data.camera_data_path``).

    Returns:
        A sorted list of paths to the discovered ``{source_id}_log.npz`` archives. Returns an empty list if the
        directory does not exist or contains no matching archives.
    """
    if not data_directory.is_dir():
        return []
    return sorted(data_directory.glob(_RAW_CAMERA_LOG_PATTERN))


def extract_camera_source_id(log_path: Path) -> int:
    """Extracts the numeric camera source ID from a raw camera log archive filename.

    Args:
        log_path: The path to the raw camera log archive. The filename must follow the ``{source_id}_log.npz``
            naming convention.

    Returns:
        The numeric source ID encoded in the filename.

    Raises:
        ValueError: If the filename does not follow the expected naming convention.
    """
    stem = log_path.stem  # e.g., "51_log"
    parts = stem.split("_")

    if len(parts) != _RAW_CAMERA_LOG_PART_COUNT or parts[1] != "log" or not parts[0].isdigit():
        message = (
            f"Unable to extract the camera source ID from '{log_path.name}'. The filename does not follow the "
            f"expected '{{source_id}}_log.npz' naming convention."
        )
        console.error(message=message, error=ValueError)

    return int(parts[0])


def resolve_camera_output_names(data_directory: Path) -> dict[int, str]:
    """Maps each camera source ID registered in the acquisition-time manifest to its canonical timestamp filename.

    Reads the camera manifest that every VideoSystem writes alongside its log archives and projects each registered
    source into its canonical ``{name}_timestamps.feather`` output filename. The manifest is the sole source of
    camera output names, so the pipeline requires no acquisition-system-specific configuration: the colloquial
    source names recorded at acquisition time (for example, ``face_camera``) directly determine the output names.

    Args:
        data_directory: The path to the session's raw camera data directory (``session.raw_data.camera_data_path``),
            which holds the camera log archives and their shared camera manifest.

    Returns:
        A dictionary mapping each registered camera source ID to its canonical timestamp feather filename.

    Raises:
        FileNotFoundError: If the camera manifest file does not exist in the data directory.
    """
    # Deferred import: the acquisition binding is only required when a video extraction job actually runs.
    from ataraxis_video_system import CAMERA_MANIFEST_FILENAME, CameraManifest  # noqa: PLC0415

    manifest_path = data_directory.joinpath(CAMERA_MANIFEST_FILENAME)
    if not manifest_path.is_file():
        message = (
            f"Unable to resolve camera-timestamp output names. No camera manifest ('{CAMERA_MANIFEST_FILENAME}') "
            f"was found in the raw camera data directory '{data_directory}'."
        )
        console.error(message=message, error=FileNotFoundError)

    # Each manifest source associates a source ID with a colloquial name (e.g., 'face_camera'); the canonical output
    # filename is that name suffixed with '_timestamps.feather'.
    manifest = CameraManifest.from_yaml(file_path=manifest_path)
    return {source.id: f"{source.name}{_CAMERA_TIMESTAMP_SUFFIX}" for source in manifest.sources}


def process_camera_log(log_path: Path, output_directory: Path, output_name: str, *, workers: int = -1) -> None:
    """Extracts camera frame acquisition timestamps from a raw VideoSystem log archive and writes them as a feather.

    Notes:
        Extracts the timestamps in-process via the ataraxis-video-system ``extract_logged_camera_timestamps``
        binding, then writes them directly to the behavior data directory under the provided canonical timestamp
        filename. This consolidates the extraction and renaming into a single pass. The ataraxis-video-system import
        is deferred to call time so that importing this module does not require the acquisition library to be
        installed.

    Args:
        log_path: The path to the raw ``{source_id}_log.npz`` camera log archive produced by a VideoSystem.
        output_directory: The path to the behavior data directory where the renamed timestamp feather is written.
        output_name: The canonical timestamp feather filename resolved from the camera manifest (for example,
            ``face_camera_timestamps.feather``).
        workers: The number of worker processes the extraction binding may use. Set to -1 to use all available
            CPU cores (minus reserved cores).
    """
    # Deferred import: the acquisition binding is only required when a video extraction job actually runs.
    from ataraxis_video_system import extract_logged_camera_timestamps  # noqa: PLC0415

    # Extracts the frame acquisition timestamps (microseconds since the UTC epoch) from the raw log archive.
    timestamps = extract_logged_camera_timestamps(log_path=log_path, n_workers=workers)

    # Writes the timestamps directly under the canonical behavior-data name as an uncompressed feather so that it
    # supports memory-mapped reads by the downstream forging pipeline.
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory.joinpath(output_name)
    frame = pl.DataFrame({_FRAME_TIME_COLUMN: np.array(timestamps, dtype=np.uint64)})
    frame.write_ipc(file=output_path, compression="uncompressed")


def run_video_processing_pipeline(
    session_path: Path,
    job_id: str | None = None,
    *,
    workers: int = -1,
    display_progress: bool = False,
) -> None:
    """Discovers, validates, and executes camera-timestamp extraction jobs for the target session.

    Notes:
        Each raw ``{source_id}_log.npz`` camera archive whose source ID is registered in the session's camera
        manifest becomes a single extraction job. In local mode (job_id is None), all jobs run sequentially in the
        parent process; the ataraxis-video-system binding parallelizes the extraction of an individual archive
        internally via the ``workers`` budget, so no additional worker pool is layered here. In remote mode (job_id
        is provided), only the job matching the identifier is executed. Camera-timestamp extraction state is tracked
        by the session's camera processing tracker, and the renamed timestamp feathers are written under the
        session's behavior data directory where the forging pipeline reads them.

    Args:
        session_path: The path to the root session directory containing the session data hierarchy.
        job_id: The unique hexadecimal identifier for the extraction job to execute. If provided, only the matching
            job is executed (remote mode). If not provided, all discovered jobs are executed (local mode).
        workers: The number of worker processes the extraction binding may use per archive. Set to -1 to use all
            available CPU cores (minus reserved cores).
        display_progress: Determines whether to display a progress bar during processing.

    Raises:
        ValueError: If no camera log archives are discovered, or if the provided job_id does not match any job.
    """
    session = SessionData.load(session_path=session_path)

    console.echo(
        message=f"Initializing camera-timestamp extraction pipeline for session '{session.session_name}'...",
        level=LogLevel.INFO,
    )

    # Resolves the canonical output name for every camera source registered in the acquisition-time manifest.
    camera_data_directory = session.raw_data.camera_data_path
    output_names = resolve_camera_output_names(data_directory=camera_data_directory)

    # Discovers one extraction job per registered raw camera log archive.
    job_paths: dict[tuple[str, str], Path] = {}
    for log_path in find_camera_logs(data_directory=camera_data_directory):
        source_id = extract_camera_source_id(log_path=log_path)
        if source_id not in output_names:
            continue
        job_paths[(VIDEO_JOB_NAME, str(source_id))] = log_path

    if not job_paths:
        message = (
            f"Unable to extract camera timestamps for session '{session.session_name}'. No registered camera log "
            f"archives were discovered in '{camera_data_directory}'."
        )
        console.error(message=message, error=ValueError)

    console.echo(message=f"Discovered {len(job_paths)} camera extraction job(s).")

    output_directory = session.processed_data.behavior_data_path
    output_directory.mkdir(parents=True, exist_ok=True)
    tracker = ProcessingTracker(file_path=session.processed_data.camera_tracker_path)
    prepare_tracker(tracker=tracker, jobs=list(job_paths.keys()))

    if job_id is not None:
        id_to_job = {
            ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier): (job_name, specifier)
            for job_name, specifier in job_paths
        }
        if job_id not in id_to_job:
            message = (
                f"Unable to execute the requested job with ID '{job_id}'. The identifier does not match any camera "
                f"extraction job available for this session. Valid job IDs: {sorted(id_to_job.keys())}."
            )
            console.error(message=message, error=ValueError)
        job_name, specifier = id_to_job[job_id]
        _execute_camera_job(
            job_name=job_name,
            specifier=specifier,
            log_path=job_paths[(job_name, specifier)],
            output_directory=output_directory,
            output_name=output_names[int(specifier)],
            tracker=tracker,
            workers=workers,
        )
    else:
        progress_context = (
            console.progress(total=len(job_paths), description="Extracting camera timestamps", unit="job")
            if display_progress
            else nullcontext()
        )
        with progress_context as progress_bar:
            for (job_name, specifier), log_path in job_paths.items():
                _execute_camera_job(
                    job_name=job_name,
                    specifier=specifier,
                    log_path=log_path,
                    output_directory=output_directory,
                    output_name=output_names[int(specifier)],
                    tracker=tracker,
                    workers=workers,
                )
                if progress_bar is not None:
                    progress_bar.update(1)

    console.echo(message="All camera-timestamp extraction jobs completed successfully.", level=LogLevel.SUCCESS)


def _execute_camera_job(
    job_name: str,
    specifier: str,
    log_path: Path,
    output_directory: Path,
    output_name: str,
    tracker: ProcessingTracker,
    *,
    workers: int,
) -> None:
    """Executes a single camera-timestamp extraction job with full tracker state management.

    Args:
        job_name: The job type name (``camera_timestamp_extraction``).
        specifier: The camera source ID specifier.
        log_path: The path to the raw camera log archive to extract.
        output_directory: The behavior data output directory where the renamed timestamp feather is written.
        output_name: The canonical timestamp feather filename resolved from the camera manifest.
        tracker: The camera ProcessingTracker instance for recording job state transitions.
        workers: The number of worker processes the extraction binding may use.
    """
    job_id = ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier)
    console.echo(message=f"Running '{job_name}' job with specifier '{specifier}' (ID: {job_id})...")
    tracker.start_job(job_id=job_id)

    try:
        process_camera_log(
            log_path=log_path, output_directory=output_directory, output_name=output_name, workers=workers
        )
        tracker.complete_job(job_id=job_id)
    except Exception as exception:
        tracker.fail_job(job_id=job_id, error_message=str(exception))
        raise
