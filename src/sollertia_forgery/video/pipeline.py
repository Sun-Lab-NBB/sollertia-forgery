"""Provides the camera video-processing pipeline that parses each VideoSystem log archive into per-camera frame
timestamps, publishes them under canonical manifest names, runs the acquisition system's donated video-tracking
function over pose predictions, and measures each recording's per-frame motion energy.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING
from contextlib import ExitStack
from concurrent.futures import ProcessPoolExecutor

import polars as pl
from natsort import natsorted
from ataraxis_video_system import CAMERA_MANIFEST_FILENAME, CameraManifest
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count
from sollertia_shared_assets import SessionData, ProcessingTrackers
from ataraxis_data_structures import ProcessingTracker
from ataraxis_video_system.video import TIMESTAMP_JOB_NAME, execute_job

from ..registries import resolve_video_tracking
from .motion_energy import (
    MOTION_ENERGY_SUFFIX,
    resolve_camera_video,
    compute_camera_motion_energy,
)
from ..shared_assets import LOG_ARCHIVE_SUFFIX, tracked_job, pinned_worker_threads

if TYPE_CHECKING:
    from pathlib import Path

RENAME_JOB_NAME: str = "camera_timestamp_rename"
"""The job name used to identify the single timestamp renaming job in the video processing tracker. The job uses an
empty specifier because it publishes every camera's parsed feather in one pass. It is the pipeline's only job that
depends on another: it links the feathers the per-camera parse jobs write, so it must run after them."""

TRACKING_JOB_NAME: str = "pose_tracking"
"""The job name used to identify the single video-tracking job in the video processing tracker. The job uses an empty
specifier because the acquisition system's donated function performs all of that session's tracking in one pass. It
reads only the raw pose predictions, so it is independent of the timestamp and motion-energy jobs that share its
tracker."""

ENERGY_JOB_NAME: str = "motion_energy"
"""The job name used to identify a single camera's motion-energy job in the video processing tracker. The job uses the
camera's source ID as its specifier, so each camera's recording is measured as an independently schedulable job,
mirroring the per-camera timestamp parsing job. It reads only that camera's recording, so it is independent of every
other job."""

_RAW_CAMERA_LOG_PART_COUNT: int = 2
"""The expected number of underscore-delimited components in a ``{source_id}_log`` archive stem."""

_PARSED_CAMERA_PREFIX: str = "camera_"
"""The prefix the extraction binding puts on each parsed timestamp feather it writes, ahead of the camera's numeric
source ID."""

_CAMERA_TIMESTAMP_SUFFIX: str = "_timestamps.feather"
"""The suffix appended to each camera's manifest name to form its canonical timestamp feather filename within the
video data directory (for example, the ``left_camera`` source produces ``left_camera_timestamps.feather``)."""


def run_video_processing_pipeline(
    session_path: Path,
    job_id: str | None = None,
    *,
    timestamp: bool = False,
    track: bool = False,
    energy: bool = False,
    target_camera: int = -1,
    workers: int = -1,
    display_progress: bool = False,
) -> None:
    """Discovers, validates, and executes the camera video-processing pipeline for the target session.

    Notes:
        The pipeline runs four kinds of job sharing one processing tracker, grouped under three flags. The ``timestamp``
        flag runs one parse job per camera whose ``{source_id}_log.npz`` archive is on disk, each extracting that
        camera's frame timestamps, then the single rename job that republishes every parsed feather under its canonical
        manifest name. The ``track`` flag runs the single job that applies the acquisition system's donated tracking
        function to the session's pose predictions, which no-ops when none are present. The ``energy`` flag runs one
        job per camera, measuring its recording into a motion-energy feather, and no-ops when the recording is absent.

        The camera manifest defines the full job universe, one parse and one energy job per registered camera plus the
        single rename and tracking jobs, so tracker alignment does not depend on which archives are on disk.

        Two runtimes select which jobs execute. In local mode (``job_id`` is None) the pipeline runs each requested
        stage sequentially over one shared worker pool, and runs every stage when no flag is set or all three are set,
        mirroring the cindra pipeline's resolution. The parse and energy jobs honor ``target_camera`` to narrow that
        pass to a single camera. In remote mode (a ``job_id`` is provided) only the single job matching that identifier
        runs, chosen entirely by the identifier, so the flags and ``target_camera`` are ignored. This lets an external
        scheduler drive cross-job parallelism by dispatching each job identifier concurrently. Either way each parse or
        energy job fans its per-recording decoding across the worker pool, while the rename and tracking jobs run
        single-core.

    Args:
        session_path: The path to the root session directory containing the session data hierarchy.
        job_id: The unique hexadecimal identifier for the job to execute. If provided, only the matching job is
            executed (remote mode). If not provided, all requested jobs are executed (local mode).
        timestamp: Determines whether to run the timestamp stage, which parses each camera's log archive into a
            frame-timestamp feather and republishes it under its canonical manifest name.
        track: Determines whether to run the video-tracking job (the system's donated tracking function).
        energy: Determines whether to run the per-camera motion-energy job.
        target_camera: The numeric source ID of the single camera to process when running the timestamp or
            motion-energy stages. Set to -1 to process every camera with a discovered log archive for the timestamp
            stage, and every camera registered in the manifest for the motion-energy stage. Ignored by the tracking
            job, and in remote mode (when job_id is provided), where the job to run is selected entirely by job_id.
        workers: The number of worker processes the extraction binding may use per archive, and that the motion-energy
            job decodes each recording with. Set to -1 to use all available CPU cores (minus reserved cores).
        display_progress: Determines whether to display progress during each archive's parsing and each recording's
            motion-energy measurement.

    Raises:
        ValueError: If the camera manifest registers no cameras, if no camera log archives are discovered for the
            timestamp stage, or if job_id does not match an available job. Also raised when target_camera has no
            discovered log archive while the timestamp stage runs, or is not registered in the camera manifest while
            the motion-energy stage runs.
        FileNotFoundError: If the camera manifest is missing, or if the job_id selects a timestamp-parsing job whose
            camera has no log archive.
    """
    session = SessionData.load(session_path=session_path)

    console.echo(
        message=f"Initializing camera video-processing pipeline for session '{session.session_name}'...",
        level=LogLevel.INFO,
    )

    # Resolves the colloquial name of every camera registered in the acquisition-time manifest. The manifest defines
    # the full job universe, decoupling tracker alignment from whichever archives are currently on disk. Both
    # the manifest and the archives it describes live in the raw behavior-data directory, which collects the messages
    # every DataLogger-backed source emits during acquisition. The raw camera-data directory holds the recordings
    # themselves, which this pipeline's motion-energy job reads, alongside the pose predictions its tracking job
    # reads.
    log_directory = session.raw_data.behavior_data_path
    camera_names = _resolve_camera_names(data_directory=log_directory)
    if not camera_names:
        message = (
            f"Unable to process camera timestamps for session '{session.session_name}'. The camera manifest in "
            f"'{log_directory}' does not register any cameras."
        )
        console.error(message=message, error=ValueError)

    # The universe is one parse job per registered camera, the single rename job, the single tracking job, and one
    # energy job per registered camera, used for tracker alignment. The tracking and energy jobs are always present
    # (both no-op when the inputs they read are absent), so a partial invocation never wipes them from the shared
    # video tracker.
    universe = [(TIMESTAMP_JOB_NAME, str(source_id)) for source_id in camera_names]
    universe.append((RENAME_JOB_NAME, ""))
    universe.append((TRACKING_JOB_NAME, ""))
    universe.extend((ENERGY_JOB_NAME, str(source_id)) for source_id in camera_names)

    # Discovers the raw log archive backing each registered camera.
    log_paths: dict[int, Path] = {
        source_id: log_path
        for log_path in _find_camera_logs(data_directory=log_directory)
        if (source_id := _extract_camera_source_id(log_path=log_path)) in camera_names
    }

    # All four job kinds write into the single processed video-data directory. The processing tracker lives
    # there too, matching SessionData.processed_data.video_tracker_path.
    video_data_directory = session.processed_data.video_data_path
    video_data_directory.mkdir(parents=True, exist_ok=True)
    tracker = ProcessingTracker(file_path=video_data_directory.joinpath(ProcessingTrackers.VIDEO))

    if job_id is not None:
        # Remote mode: registers the requested job alone while detecting foreign entries against the full universe, so
        # the invocation neither wipes its sibling jobs nor registers a job this session cannot run.
        jobs_by_identifier = {
            ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier): (job_name, specifier)
            for job_name, specifier in universe
        }
        if job_id not in jobs_by_identifier:
            message = (
                f"Unable to execute the requested job with ID '{job_id}'. The identifier does not match any camera "
                f"processing job available for this session. Valid job IDs: {sorted(jobs_by_identifier.keys())}."
            )
            console.error(message=message, error=ValueError)

        job_name, specifier = jobs_by_identifier[job_id]
        tracker.align_jobs(jobs=[(job_name, specifier)], universe=universe)

        if job_name == TIMESTAMP_JOB_NAME and int(specifier) not in log_paths:
            message = (
                f"Unable to execute the requested timestamp parsing job with ID '{job_id}'. No raw log archive was "
                f"discovered for camera source ID {specifier} in '{log_directory}'."
            )
            console.error(message=message, error=FileNotFoundError)

        # Caps the worker threading layers before the extraction binding starts its own pool, since in remote mode
        # the binding owns the pool and would otherwise spawn each worker with the machine's full thread budget.
        with pinned_worker_threads():
            _dispatch_job(
                job_name=job_name,
                specifier=specifier,
                session=session,
                log_paths=log_paths,
                camera_names=camera_names,
                video_data_directory=video_data_directory,
                tracker=tracker,
                workers=workers,
                display_progress=display_progress,
                executor=None,
            )
        console.echo(message="Camera video-processing job completed successfully.", level=LogLevel.SUCCESS)
        return

    # Local mode: runs the requested stages sequentially over one shared pool. Mirrors the cindra resolution, running
    # every stage when the flags share one state (all off or all on).
    if not (timestamp or track or energy):
        timestamp = track = energy = True

    jobs: list[tuple[str, str]] = []
    if timestamp:
        if not log_paths:
            message = (
                f"Unable to parse camera timestamps for session '{session.session_name}'. No registered camera log "
                f"archives were discovered in '{log_directory}'."
            )
            console.error(message=message, error=ValueError)
        if target_camera == -1:
            jobs.extend((TIMESTAMP_JOB_NAME, str(source_id)) for source_id in log_paths)
        else:
            if target_camera not in log_paths:
                message = (
                    f"Unable to parse camera timestamps for the requested camera source ID {target_camera}. No "
                    f"registered camera log archive was discovered for it in '{log_directory}'."
                )
                console.error(message=message, error=ValueError)
            jobs.append((TIMESTAMP_JOB_NAME, str(target_camera)))
        # The rename job republishes every parsed feather under its canonical name, so it belongs to the timestamp
        # stage and always follows the parse jobs that write those feathers.
        jobs.append((RENAME_JOB_NAME, ""))
    if track:
        # Reads only the raw pose predictions and does not consume the parsed timestamp feathers, so its position in
        # the local sequence is arbitrary. It is listed here only for a stable, readable order.
        jobs.append((TRACKING_JOB_NAME, ""))
    if energy:
        # Reads only each camera's recording, so it is independent of every other job. Its position in the local
        # sequence is arbitrary. Honors target_camera exactly as the timestamp stage does, and covers every registered
        # camera rather than only those with a discovered log archive, since a recording can be measured without its
        # archive.
        if target_camera == -1:
            jobs.extend((ENERGY_JOB_NAME, str(source_id)) for source_id in camera_names)
        else:
            if target_camera not in camera_names:
                message = (
                    f"Unable to measure motion energy for the requested camera source ID {target_camera}. It is not "
                    f"registered in the camera manifest in '{log_directory}'."
                )
                console.error(message=message, error=ValueError)
            jobs.append((ENERGY_JOB_NAME, str(target_camera)))

    # Detects foreign entries against the full universe rather than the requested subset, so a partial invocation (a
    # single job kind, or a partial discovery) aligns the tracker without wiping the previously completed sibling jobs.
    tracker.align_jobs(jobs=jobs, universe=universe)

    console.echo(message=f"Running {len(jobs)} camera video-processing job(s).")

    # One pool shared across every parse and energy job pays the process spawn and teardown cost once for the whole
    # session. The shared pool requires a positive, pre-resolved worker count because the extraction binding sizes
    # its batch submissions to match the pool. Jobs run one at a time, so each in turn has the whole pool to itself.
    # The renaming and tracking jobs ignore it.
    resolved_workers = resolve_worker_count(requested_workers=workers)

    # Caps the worker threading layers for the whole life of the shared pool, so a run given N workers occupies N
    # cores. The pool starts its children on demand and each child sizes its library thread pools while importing,
    # before any job code of ours runs, so the caps have to be in place here rather than inside the workers. Scoping
    # them to the pool rather than setting them at import keeps the rest of the library multithreaded.
    with pinned_worker_threads(), ExitStack() as pool_scope:
        shared_executor = (
            pool_scope.enter_context(ProcessPoolExecutor(max_workers=resolved_workers))
            if resolved_workers > 1
            else None
        )

        for job_name, specifier in jobs:
            _dispatch_job(
                job_name=job_name,
                specifier=specifier,
                session=session,
                log_paths=log_paths,
                camera_names=camera_names,
                video_data_directory=video_data_directory,
                tracker=tracker,
                workers=resolved_workers,
                display_progress=display_progress,
                executor=shared_executor,
            )

    console.echo(message="All camera video-processing jobs completed successfully.", level=LogLevel.SUCCESS)


def discover_video_jobs(session_path: Path) -> tuple[SessionData, list[tuple[str, str]], list[tuple[str, str]]]:
    """Resolves the video pipeline's job universe and possible subset for the target session.

    Notes:
        The universe is the full acquisition-time job set the camera manifest defines. It holds one timestamp job and
        one motion-energy job per registered camera, plus the single rename and tracking jobs. The possible subset is
        the job set the session's own data supports. A timestamp job is possible only when its camera's
        ``{source_id}_log.npz`` archive is on disk, and the rename job joins them when at least one is possible. The
        tracking and energy jobs are always possible because they read only their own inputs and complete with no
        output when those are absent. A camera with a recording but no log archive therefore keeps its energy job
        possible while its timestamp job stays in the universe alone. This is discovery only, reading the manifest and
        globbing for archives while decoding no data and mutating nothing.

    Args:
        session_path: The path to the root session directory containing the session data hierarchy.

    Returns:
        A tuple of the loaded session, the job universe as a list of ``(job_name, specifier)`` pairs, and the possible
        subset of that universe. Timestamp and energy specifiers are camera source IDs, and the rename and tracking
        specifiers are empty.

    Raises:
        FileNotFoundError: If the session's camera manifest is not present.
        ValueError: If the camera manifest registers no cameras.
    """
    session = SessionData.load(session_path=session_path)
    log_directory = session.raw_data.behavior_data_path
    camera_names = _resolve_camera_names(data_directory=log_directory)
    if not camera_names:
        message = (
            f"Unable to resolve video processing jobs for session '{session.session_name}'. The camera manifest in "
            f"'{log_directory}' does not register any cameras."
        )
        console.error(message=message, error=ValueError)

    universe = [(TIMESTAMP_JOB_NAME, str(source_id)) for source_id in camera_names]
    universe.append((RENAME_JOB_NAME, ""))
    universe.append((TRACKING_JOB_NAME, ""))
    universe.extend((ENERGY_JOB_NAME, str(source_id)) for source_id in camera_names)

    log_paths: dict[int, Path] = {
        source_id: log_path
        for log_path in _find_camera_logs(data_directory=log_directory)
        if (source_id := _extract_camera_source_id(log_path=log_path)) in camera_names
    }

    possible: list[tuple[str, str]] = [(TIMESTAMP_JOB_NAME, str(source_id)) for source_id in log_paths]
    if log_paths:
        possible.append((RENAME_JOB_NAME, ""))
    possible.append((TRACKING_JOB_NAME, ""))
    possible.extend((ENERGY_JOB_NAME, str(source_id)) for source_id in camera_names)

    return session, universe, possible


def video_job_prerequisites(
    session: SessionData,  # noqa: ARG001
    universe: list[tuple[str, str]],
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]:
    """Returns the intra-pipeline job ordering for the video pipeline.

    Notes:
        The rename job hardlinks the parsed timestamp feathers under their canonical names, so it must run after the
        timestamp parse jobs that write them and depends on every timestamp job present in the given set. The tracking
        and energy jobs read only their own inputs and have no upstream dependency. Passing the possible subset scopes
        the rename job to the timestamp jobs that can actually produce feathers, while passing the full universe scopes
        it to all registered cameras.

    Args:
        session: The loaded session, accepted for the shared dispatch contract and not read by this ordering.
        universe: The job set to build ordering over, as returned by ``discover_video_jobs`` (either the universe or
            its possible subset).

    Returns:
        A mapping of each job to its tuple of prerequisite jobs. The rename job maps to the timestamp jobs in the set,
        and every other job maps to an empty tuple.
    """
    timestamp_jobs = tuple(job for job in universe if job[0] == TIMESTAMP_JOB_NAME)
    return {job: (timestamp_jobs if job[0] == RENAME_JOB_NAME else ()) for job in universe}


def _resolve_camera_names(data_directory: Path) -> dict[int, str]:
    """Maps each camera source ID registered in the acquisition-time manifest to its colloquial camera name.

    Reads the camera manifest that every VideoSystem writes alongside its log archives. The manifest is the sole
    source of camera names, so the pipeline requires no acquisition-system-specific configuration. The colloquial
    source names recorded at acquisition time (for example, ``left_camera``) determine every output filename this
    pipeline writes and locate each camera's recording on disk.

    Args:
        data_directory: The path to the session's raw behavior data directory
            (``session.raw_data.behavior_data_path``), which holds the camera log archives and their shared camera
            manifest alongside every other DataLogger-backed source's archives.

    Returns:
        A dictionary mapping each registered camera source ID to its colloquial camera name.

    Raises:
        FileNotFoundError: If the camera manifest file does not exist in the data directory.
    """
    manifest_path = data_directory.joinpath(CAMERA_MANIFEST_FILENAME)
    if not manifest_path.is_file():
        message = (
            f"Unable to resolve camera names. No camera manifest ('{CAMERA_MANIFEST_FILENAME}') was found in the "
            f"raw behavior data directory '{data_directory}'."
        )
        console.error(message=message, error=FileNotFoundError)

    manifest = CameraManifest.from_yaml(file_path=manifest_path)
    return {source.id: source.name for source in manifest.sources}


def _find_camera_logs(data_directory: Path) -> list[Path]:
    """Discovers every raw DataLogger log archive inside the canonical raw behavior data directory.

    The caller narrows the discovered archives to the cameras the acquisition-time manifest registers.

    Args:
        data_directory: The path to the session's raw behavior data directory
            (``session.raw_data.behavior_data_path``).

    Returns:
        A naturally sorted list of paths to the discovered ``{source_id}_log.npz`` archives. Returns an empty list if
        the directory does not exist or contains no matching archives.
    """
    if not data_directory.is_dir():
        return []
    return natsorted(data_directory.glob(f"*{LOG_ARCHIVE_SUFFIX}"))


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
    stem = log_path.stem
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
    session: SessionData,
    log_paths: dict[int, Path],
    camera_names: dict[int, str],
    video_data_directory: Path,
    tracker: ProcessingTracker,
    *,
    workers: int,
    display_progress: bool,
    executor: ProcessPoolExecutor | None,
) -> None:
    """Executes a single pipeline job, dispatching to the parsing, renaming, tracking, or energy handler by job name.

    Args:
        job_name: The job name identifying which job to run (``TIMESTAMP_JOB_NAME``, ``RENAME_JOB_NAME``,
            ``TRACKING_JOB_NAME``, or ``ENERGY_JOB_NAME``).
        specifier: The job specifier. For a parse or energy job this is the camera source ID. For the rename and
            tracking jobs it is empty.
        session: The loaded session, used by the tracking job to resolve and run the system's donated function and
            by the energy job to locate each camera's recording.
        log_paths: The mapping of discovered camera source IDs to their raw log archive paths.
        camera_names: The mapping of camera source IDs to their colloquial manifest names.
        video_data_directory: The processed video-data directory where parsed feathers, their canonical hardlinks, the
            tracking outputs, and the motion-energy feathers are all written.
        tracker: The video ProcessingTracker instance for recording job state transitions.
        workers: The number of worker processes the extraction binding may use, and that the energy job decodes
            each recording with.
        display_progress: Determines whether the extraction binding displays a progress bar and whether the energy
            job reports per-chunk completion.
        executor: An optional process pool shared across parse and energy jobs so neither spawns its own. The renaming
            and tracking jobs ignore it. When None, the job creates and tears down its own pool. A motion-energy job
            whose recording plans a single decode chunk runs in-process either way and touches no pool.

    Raises:
        ValueError: If the job name does not identify a pipeline job.
    """
    job_id = ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier)

    # The extraction binding announces and runs the timestamp job itself. The jobs this module owns are announced here
    # in the same format, so every job kind reports its start uniformly.
    if job_name != TIMESTAMP_JOB_NAME:
        source_fragment = f" for source '{specifier}'" if specifier else ""
        console.echo(message=f"Running '{job_name}' job{source_fragment} (ID: {job_id})...", level=LogLevel.INFO)

    if job_name == TIMESTAMP_JOB_NAME:
        # The ataraxis-video-system binding extracts the timestamps, writes the
        # 'camera_{source_id}_timestamps.feather' into the video data directory, and records this job's start,
        # completion, or failure on the tracker.
        execute_job(
            log_path=log_paths[int(specifier)],
            output_directory=video_data_directory,
            source_id=specifier,
            job_id=job_id,
            workers=workers,
            tracker=tracker,
            display_progress=display_progress,
            executor=executor,
        )
        # The binding writes one timestamp per camera frame, so the feather's row count is the extracted frame count.
        frame_count = pl.read_ipc(
            source=video_data_directory.joinpath(f"{_PARSED_CAMERA_PREFIX}{specifier}{_CAMERA_TIMESTAMP_SUFFIX}"),
            memory_map=True,
        ).height
        console.echo(
            message=f"Extracted {frame_count} frame timestamp(s) for source '{specifier}'.",
            level=LogLevel.SUCCESS,
        )
    elif job_name == RENAME_JOB_NAME:
        _link_parsed_timestamps(
            video_data_directory=video_data_directory,
            camera_names=camera_names,
            job_id=job_id,
            tracker=tracker,
        )
    elif job_name == TRACKING_JOB_NAME:
        _run_pose_tracking(
            session=session,
            video_data_directory=video_data_directory,
            job_id=job_id,
            tracker=tracker,
        )
    elif job_name == ENERGY_JOB_NAME:
        _run_motion_energy(
            session=session,
            camera_name=camera_names[int(specifier)],
            video_data_directory=video_data_directory,
            job_id=job_id,
            tracker=tracker,
            workers=workers,
            display_progress=display_progress,
            executor=executor,
        )
    else:
        # Every job name is matched explicitly, so an unrecognized name surfaces here as a failure.
        message = (
            f"Unable to execute the camera video-processing job '{job_name}'. The name does not identify any "
            f"pipeline job."
        )
        console.error(message=message, error=ValueError)


def _link_parsed_timestamps(
    video_data_directory: Path,
    camera_names: dict[int, str],
    job_id: str,
    tracker: ProcessingTracker,
) -> None:
    """Hardlinks every parsed feather under its canonical name within the video data directory as one tracked job.

    Each ``camera_{source_id}_timestamps.feather`` the parsing job wrote into the video data directory is hardlinked
    to its canonical ``{name}_timestamps.feather`` in the same directory. The hardlink shares the parsed feather's
    inode, so the canonical copy adds no extra bytes and stays in sync while the original parsed feather is preserved.
    Should hardlinking be unavailable, the feather is copied instead. A camera whose manifest name already equals its
    parsed filename needs no link and is left as-is. Cameras whose parsed feather is absent (for example, because their
    parse job has not run) are skipped, and any stale canonical link is replaced, so the job is safe to re-run.

    Args:
        video_data_directory: The processed video-data directory holding the parsed feathers and receiving their
            canonical hardlinks.
        camera_names: The mapping of camera source IDs to their colloquial manifest names, from which the canonical
            timestamp feather filenames are built.
        job_id: The unique hexadecimal identifier for the rename job.
        tracker: The video ProcessingTracker instance for recording job state transitions.
    """
    with tracked_job(tracker=tracker, job_id=job_id):
        published = 0
        for source_id, camera_name in camera_names.items():
            # The parsing job (ataraxis-video-system extraction binding) writes each camera's feather under this
            # name into the video data directory. The rename job then hardlinks it under its canonical name.
            parsed_path = video_data_directory.joinpath(f"{_PARSED_CAMERA_PREFIX}{source_id}{_CAMERA_TIMESTAMP_SUFFIX}")
            if not parsed_path.is_file():
                continue
            canonical_path = video_data_directory.joinpath(f"{camera_name}{_CAMERA_TIMESTAMP_SUFFIX}")
            # A manifest name of literally 'camera_{source_id}' makes the canonical name the parsed feather itself.
            # Unlinking it here would destroy the parsed feather, so the already-published feather is left untouched.
            if canonical_path == parsed_path:
                published += 1
                continue
            # Replaces any stale link so a re-run re-points the canonical name at the freshly parsed feather.
            canonical_path.unlink(missing_ok=True)
            try:
                canonical_path.hardlink_to(target=parsed_path)
            except OSError:
                # Hardlinking can fail in some environments. Falls back to a copy so the canonical name is published.
                shutil.copy2(src=parsed_path, dst=canonical_path)
            published += 1
        console.echo(
            message=f"Renamed {published} parsed camera timestamp feather(s) to their canonical names.",
            level=LogLevel.SUCCESS,
        )


def _run_pose_tracking(
    session: SessionData,
    video_data_directory: Path,
    job_id: str,
    tracker: ProcessingTracker,
) -> None:
    """Runs the acquisition system's donated video-tracking function over the session's pose predictions, as one job.

    The system's donated tracking function locates its externally-produced pose predictions, parses them, and writes
    its tracking outputs into the processed video-data directory. The predictions are produced upstream and travel with
    the session's raw data, so this job only reads them. The function no-ops when no prediction file is present, so
    this job is safe to run on every session.

    Args:
        session: The loaded session whose acquisition system selects the tracking function.
        video_data_directory: The processed video-data directory the tracking outputs are written into.
        job_id: The unique hexadecimal identifier for the tracking job.
        tracker: The video ProcessingTracker instance for recording job state transitions.
    """
    with tracked_job(tracker=tracker, job_id=job_id):
        resolve_video_tracking(system=session.acquisition_system)(
            session=session, output_directory=video_data_directory
        )


def _run_motion_energy(
    session: SessionData,
    camera_name: str,
    video_data_directory: Path,
    job_id: str,
    tracker: ProcessingTracker,
    *,
    workers: int,
    display_progress: bool,
    executor: ProcessPoolExecutor | None,
) -> None:
    """Measures one camera's recording into a per-frame motion-energy feather, as one tracked job.

    Locates the camera's recording in the session's raw camera-data directory and writes its motion energy into the
    processed video-data directory. A camera whose recording is absent is skipped, so a rig that ran one of its
    cameras leaves the shared video tracker clear on the camera it did not run.

    Args:
        session: The loaded session whose raw camera-data directory supplies the recording.
        camera_name: The colloquial manifest name of the camera to measure, which names both its recording and its
            output feather.
        video_data_directory: The processed video-data directory the motion-energy feather is written into.
        job_id: The unique hexadecimal identifier for this camera's energy job.
        tracker: The video ProcessingTracker instance for recording job state transitions.
        workers: The number of worker processes to decode the recording with.
        display_progress: Determines whether per-chunk completion is reported as the recording is measured.
        executor: An optional process pool to decode the recording's chunks in. When None, a pool is created and torn
            down for this recording. A recording that plans a single decode chunk runs in-process either way and
            touches no pool.
    """
    # Resolved inside the tracked job so that a missing recording and a decode failure alike are recorded against a
    # job that actually started, rather than leaving the tracker unable to explain why the job never ran.
    with tracked_job(tracker=tracker, job_id=job_id):
        video_path = resolve_camera_video(
            camera_data_directory=session.raw_data.camera_data_path,
            session_name=session.session_name,
            camera_name=camera_name,
        )
        if video_path is None:
            console.echo(
                message=(
                    f"No '{camera_name}' recording was found in the raw camera_data directory of session "
                    f"'{session.session_name}'. Skipping its motion-energy measurement."
                ),
                level=LogLevel.INFO,
            )
            return

        compute_camera_motion_energy(
            video_path=video_path,
            output_path=video_data_directory.joinpath(f"{camera_name}{MOTION_ENERGY_SUFFIX}"),
            workers=workers,
            executor=executor,
            display_progress=display_progress,
        )
