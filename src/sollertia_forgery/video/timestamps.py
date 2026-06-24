"""Provides the two-stage camera-timestamp processing pipeline that parses raw VideoSystem log archives into the
camera timestamps directory and hardlinks them under their canonical names into the behavior data directory.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING
from concurrent.futures import ProcessPoolExecutor

from natsort import natsorted
from ataraxis_video_system import CAMERA_MANIFEST_FILENAME, CameraManifest
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count
from sollertia_shared_assets import SessionData, ProcessingTrackers
from ataraxis_data_structures import ProcessingTracker
from ataraxis_video_system.video import TIMESTAMP_JOB_NAME, execute_job

from ..orchestration import prepare_tracker

if TYPE_CHECKING:
    from pathlib import Path

RENAME_JOB_NAME: str = "camera_timestamp_rename"
"""The job name used to identify the single timestamp renaming job (stage 2) in the camera processing tracker. The
job uses an empty specifier because it publishes every camera's parsed feather in one pass."""

_RAW_CAMERA_LOG_PATTERN: str = "*_log.npz"
"""The glob pattern used to discover raw VideoSystem camera log archives. Each archive is named
``{source_id}_log.npz``, matching the DataLogger source-ID naming convention shared across the Sollertia stack."""

_RAW_CAMERA_LOG_PART_COUNT: int = 2
"""The expected number of underscore-delimited components in a ``{source_id}_log`` archive stem."""

_CAMERA_TIMESTAMP_SUFFIX: str = "_timestamps.feather"
"""The suffix appended to each camera's manifest name to form its canonical timestamp feather filename within the
behavior data directory (e.g., the ``face_camera`` source produces ``face_camera_timestamps.feather``)."""

_PARSED_TIMESTAMP_TEMPLATE: str = "camera_{source_id}_timestamps.feather"
"""The timestamp feather name written by the parsing stage (via the ataraxis-video-system extraction binding) into
the camera timestamps directory, before the renaming stage hardlinks it under its canonical name."""


def run_video_processing_pipeline(
    session_path: Path,
    job_id: str | None = None,
    *,
    parse: bool = False,
    rename: bool = False,
    target_camera: int = -1,
    workers: int = -1,
    display_progress: bool = False,
) -> None:
    """Discovers, validates, and executes the two-stage camera-timestamp processing pipeline for the target session.

    Notes:
        The pipeline has two stages tracked by the session's camera processing tracker. Stage 1 (``parse``) creates
        one job per camera registered in the session's camera manifest; each job extracts the frame acquisition
        timestamps from that camera's raw ``{source_id}_log.npz`` archive and writes a
        ``camera_{source_id}_timestamps.feather`` into the camera timestamps directory that ataraxis-video-system
        owns. Stage 2 (``rename``) is a single job that hardlinks every parsed feather under its canonical manifest
        name (for example, ``face_camera_timestamps.feather``) into the behavior data directory, the aggregate of the
        session's processed data that the downstream forging pipeline reads. When neither stage flag is requested,
        both stages run in sequence, so a single local invocation performs the full pipeline. In remote mode (job_id
        is provided), only the job matching the identifier is executed, which lets an external scheduler run each
        parse job and the rename job independently.

    Args:
        session_path: The path to the root session directory containing the session data hierarchy.
        job_id: The unique hexadecimal identifier for the job to execute. If provided, only the matching job is
            executed (remote mode). If not provided, all requested jobs are executed (local mode).
        parse: Determines whether to run the per-camera timestamp parsing stage.
        rename: Determines whether to run the timestamp renaming stage.
        target_camera: The numeric source ID of the single camera to parse when running the parsing stage. Set to -1
            to parse all discovered cameras. Ignored by the renaming stage, which always publishes every parsed feather.
        workers: The number of worker processes the extraction binding may use per archive. Set to -1 to use all
            available CPU cores (minus reserved cores).
        display_progress: Determines whether to display a progress bar during each archive's parsing.

    Raises:
        ValueError: If the camera manifest registers no cameras, if no camera log archives are discovered for the
            parsing stage, if target_camera is not a discovered camera, or if job_id does not match an available job.
        FileNotFoundError: If the camera manifest is missing, or if the job_id-selected camera has no log archive.
    """
    session = SessionData.load(session_path=session_path)

    console.echo(
        message=f"Initializing camera-timestamp processing pipeline for session '{session.session_name}'...",
        level=LogLevel.INFO,
    )

    # Resolves the canonical output name for every camera registered in the acquisition-time manifest. The manifest
    # defines the full job universe, decoupling tracker alignment from whichever archives are currently on disk.
    camera_data_directory = session.raw_data.camera_data_path
    output_names = _resolve_camera_output_names(data_directory=camera_data_directory)
    if not output_names:
        message = (
            f"Unable to process camera timestamps for session '{session.session_name}'. The camera manifest in "
            f"'{camera_data_directory}' does not register any cameras."
        )
        console.error(message=message, error=ValueError)

    # The universe is one parse job per registered camera plus the single rename job, used for tracker alignment.
    universe = [(TIMESTAMP_JOB_NAME, str(source_id)) for source_id in output_names]
    universe.append((RENAME_JOB_NAME, ""))

    # Discovers the raw log archive backing each registered camera.
    log_paths: dict[int, Path] = {}
    for log_path in _find_camera_logs(data_directory=camera_data_directory):
        source_id = _extract_camera_source_id(log_path=log_path)
        if source_id in output_names:
            log_paths[source_id] = log_path

    # Stage 1 writes into the camera timestamps directory that ataraxis-video-system owns; stage 2 hardlinks the
    # parsed feathers under their canonical names into the behavior data directory. The processing tracker lives in
    # the behavior data directory alongside the published data rather than next to the intermediate parsed feathers.
    timestamps_directory = session.processed_data.camera_timestamps_path
    behavior_directory = session.processed_data.behavior_data_path
    timestamps_directory.mkdir(parents=True, exist_ok=True)
    behavior_directory.mkdir(parents=True, exist_ok=True)
    tracker = ProcessingTracker(file_path=behavior_directory.joinpath(ProcessingTrackers.CAMERA))

    if job_id is not None:
        # Remote mode: aligns the tracker against the full universe so that the partial (single-job) invocation does
        # not wipe sibling jobs, then executes only the requested job.
        id_to_job = {
            ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier): (job_name, specifier)
            for job_name, specifier in universe
        }
        if job_id not in id_to_job:
            message = (
                f"Unable to execute the requested job with ID '{job_id}'. The identifier does not match any camera "
                f"processing job available for this session. Valid job IDs: {sorted(id_to_job.keys())}."
            )
            console.error(message=message, error=ValueError)

        prepare_tracker(tracker=tracker, jobs=universe, universe=universe)

        job_name, specifier = id_to_job[job_id]
        if job_name == TIMESTAMP_JOB_NAME and int(specifier) not in log_paths:
            message = (
                f"Unable to execute the requested timestamp parsing job with ID '{job_id}'. No raw log archive was "
                f"discovered for camera source ID {specifier} in '{camera_data_directory}'."
            )
            console.error(message=message, error=FileNotFoundError)

        _dispatch_job(
            job_name=job_name,
            specifier=specifier,
            log_paths=log_paths,
            output_names=output_names,
            timestamps_directory=timestamps_directory,
            behavior_directory=behavior_directory,
            tracker=tracker,
            workers=workers,
            display_progress=display_progress,
            executor=None,
        )
        console.echo(message="Camera-timestamp processing job completed successfully.", level=LogLevel.SUCCESS)
        return

    # Local mode: runs the requested stages in order. When no stage flag is set, both stages run.
    run_parse, run_rename = (parse, rename) if (parse or rename) else (True, True)

    jobs: list[tuple[str, str]] = []
    if run_parse:
        if not log_paths:
            message = (
                f"Unable to parse camera timestamps for session '{session.session_name}'. No registered camera log "
                f"archives were discovered in '{camera_data_directory}'."
            )
            console.error(message=message, error=ValueError)
        if target_camera == -1:
            jobs.extend((TIMESTAMP_JOB_NAME, str(source_id)) for source_id in log_paths)
        else:
            if target_camera not in log_paths:
                message = (
                    f"Unable to parse camera timestamps for the requested camera source ID {target_camera}. No "
                    f"registered camera log archive was discovered for it in '{camera_data_directory}'."
                )
                console.error(message=message, error=ValueError)
            jobs.append((TIMESTAMP_JOB_NAME, str(target_camera)))
    if run_rename:
        jobs.append((RENAME_JOB_NAME, ""))

    # Detects foreign entries against the full universe rather than the requested subset, so a partial invocation (a
    # single stage, or a partial discovery) aligns the tracker without wiping the previously completed sibling jobs.
    prepare_tracker(tracker=tracker, jobs=jobs, universe=universe)

    console.echo(message=f"Running {len(jobs)} camera-timestamp processing job(s).")

    # Resolves the worker count once and creates a single ProcessPoolExecutor shared across every parse job, mirroring
    # the ataraxis-video-system pipeline. This amortizes the cost of spawning and tearing down worker processes across
    # all cameras instead of paying it once per camera. The shared pool requires a positive, pre-resolved worker count
    # because the extraction binding sizes its batch submissions to match the pool. The renaming stage ignores it.
    resolved_workers = resolve_worker_count(requested_workers=workers)
    shared_executor = ProcessPoolExecutor(max_workers=resolved_workers) if resolved_workers > 1 else None

    try:
        for job_name, specifier in jobs:
            _dispatch_job(
                job_name=job_name,
                specifier=specifier,
                log_paths=log_paths,
                output_names=output_names,
                timestamps_directory=timestamps_directory,
                behavior_directory=behavior_directory,
                tracker=tracker,
                workers=resolved_workers,
                display_progress=display_progress,
                executor=shared_executor,
            )
    finally:
        if shared_executor is not None:
            shared_executor.shutdown(wait=True)

    console.echo(message="All camera-timestamp processing jobs completed successfully.", level=LogLevel.SUCCESS)


def _resolve_camera_output_names(data_directory: Path) -> dict[int, str]:
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


def _find_camera_logs(data_directory: Path) -> list[Path]:
    """Discovers raw VideoSystem camera log archives inside the canonical raw camera data directory.

    Args:
        data_directory: The path to the session's raw camera data directory (``session.raw_data.camera_data_path``).

    Returns:
        A naturally sorted list of paths to the discovered ``{source_id}_log.npz`` archives. Returns an empty list if
        the directory does not exist or contains no matching archives.
    """
    if not data_directory.is_dir():
        return []
    return natsorted(data_directory.glob(_RAW_CAMERA_LOG_PATTERN))


def _extract_camera_source_id(log_path: Path) -> int:
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


def _dispatch_job(
    job_name: str,
    specifier: str,
    log_paths: dict[int, Path],
    output_names: dict[int, str],
    timestamps_directory: Path,
    behavior_directory: Path,
    tracker: ProcessingTracker,
    *,
    workers: int,
    display_progress: bool,
    executor: ProcessPoolExecutor | None,
) -> None:
    """Executes a single pipeline job, routing to the parsing or renaming stage by job name.

    Args:
        job_name: The job name identifying the stage to run (``TIMESTAMP_JOB_NAME`` or ``RENAME_JOB_NAME``).
        specifier: The job specifier. For a parse job this is the camera source ID; for the rename job it is empty.
        log_paths: The mapping of discovered camera source IDs to their raw log archive paths.
        output_names: The mapping of camera source IDs to their canonical timestamp feather filenames.
        timestamps_directory: The camera timestamps directory where parsed feathers are written.
        behavior_directory: The behavior data directory where canonical hardlinks are published.
        tracker: The camera ProcessingTracker instance for recording job state transitions.
        workers: The number of worker processes the extraction binding may use.
        display_progress: Determines whether the extraction binding displays a progress bar.
        executor: An optional process pool shared across parse jobs so the extraction binding reuses it instead of
            spawning its own. The renaming stage ignores it. When None, the binding creates and tears down its own
            pool for this job.
    """
    job_id = ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier)

    if job_name == TIMESTAMP_JOB_NAME:
        # The ataraxis-video-system binding extracts the timestamps, writes the
        # 'camera_{source_id}_timestamps.feather' into the camera timestamps directory, and records this job's start,
        # completion, or failure on the tracker.
        execute_job(
            log_path=log_paths[int(specifier)],
            output_directory=timestamps_directory,
            source_id=specifier,
            job_id=job_id,
            workers=workers,
            tracker=tracker,
            display_progress=display_progress,
            executor=executor,
        )
    else:
        _link_parsed_timestamps(
            timestamps_directory=timestamps_directory,
            behavior_directory=behavior_directory,
            output_names=output_names,
            job_id=job_id,
            tracker=tracker,
        )


def _link_parsed_timestamps(
    timestamps_directory: Path,
    behavior_directory: Path,
    output_names: dict[int, str],
    job_id: str,
    tracker: ProcessingTracker,
) -> None:
    """Hardlinks every parsed feather under its canonical name into the behavior data directory as one tracked job.

    Each ``camera_{source_id}_timestamps.feather`` the parsing stage wrote into the camera timestamps directory is
    hardlinked to its canonical ``{name}_timestamps.feather`` in the behavior data directory. The hardlink shares the
    parsed feather's inode, so the canonical copy adds no extra bytes and stays in sync, while the parsed feather
    remains in the camera timestamps directory that ataraxis-video-system owns. If the two directories live on
    different filesystems, where hardlinking is impossible, the feather is copied instead. Cameras whose parsed
    feather is absent (for example, because their parse job has not run) are skipped, and any stale canonical link is
    replaced, so the job is safe to re-run.

    Args:
        timestamps_directory: The camera timestamps directory holding the parsed feathers.
        behavior_directory: The behavior data directory where the canonical hardlinks are published.
        output_names: The mapping of camera source IDs to their canonical timestamp feather filenames.
        job_id: The unique hexadecimal identifier for the rename job.
        tracker: The camera ProcessingTracker instance for recording job state transitions.
    """
    tracker.start_job(job_id=job_id)

    try:
        published = 0
        for source_id, output_name in output_names.items():
            parsed_path = timestamps_directory.joinpath(_PARSED_TIMESTAMP_TEMPLATE.format(source_id=source_id))
            if not parsed_path.is_file():
                continue
            canonical_path = behavior_directory.joinpath(output_name)
            # Replaces any stale link so a re-run re-points the canonical name at the freshly parsed feather.
            canonical_path.unlink(missing_ok=True)
            try:
                canonical_path.hardlink_to(parsed_path)
            except OSError:
                # Hardlinking fails when the directories live on different filesystems; fall back to a full copy.
                shutil.copy2(parsed_path, canonical_path)
            published += 1
        console.echo(message=f"Published {published} parsed camera timestamp feather(s) under their canonical names.")
        tracker.complete_job(job_id=job_id)
    except Exception as exception:
        tracker.fail_job(job_id=job_id, error_message=str(exception))
        raise
