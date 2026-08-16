"""Provides the sizing pass that resolves each job's cores and working set from the data it will process."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from functools import cache
from dataclasses import dataclass

import cv2
from cindra import (
    WORKER_MEMORY_MB,
    SPAWNED_CHILD_MEMORY_MB,
    MEMORY_ESTIMATE_TOLERANCE,
    COMBINED_METADATA_FILENAME,
    MULTI_RECORDING_DIRECTORY_NAME,
    RecordingArrays,
    MultiRecordingJobNames,
    SingleRecordingJobNames,
    resolve_array_path,
    resolve_recording_geometry,
    estimate_multi_recording_job_memory_mb,
    estimate_single_recording_job_memory_mb,
)
import psutil
from natsort import natsorted
from numpy.lib.format import read_magic, read_array_header_1_0, read_array_header_2_0
from ataraxis_video_system import size_archive_job as size_camera_extraction_job
from sollertia_shared_assets import SessionData
from ataraxis_data_structures import LOG_ARCHIVE_SUFFIX
from ataraxis_communication_interface import size_archive_job as size_controller_extraction_job

from ..video import ENERGY_JOB_NAME, TRACKING_JOB_NAME, CAMERA_EXTRACTION_JOB_NAME
from ..forging import MULTIDAY_DISCOVERY_JOB_NAME, MULTIDAY_EXTRACTION_JOB_NAME
from ..runtime import RUNTIME_JOB_NAME
from ..managing import CHECKSUM_JOB_NAME
from ..registries import (
    resolve_two_photon_data_locator,
    resolve_multi_recording_configuration_resolver,
    resolve_single_recording_configuration_resolver,
)
from ..shared_assets import ProcessingPipelines, multi_recording_dataset_directory
from ..microcontrollers import PARSE_JOB_NAME, CONTROLLER_EXTRACTION_JOB_NAME

if TYPE_CHECKING:
    from pathlib import Path

    from cindra import RecordingGeometry, MultiRecordingConfiguration, SingleRecordingConfiguration
    from sollertia_shared_assets import DatasetData

_MEGABYTES_PER_GIGABYTE: int = 1024
"""The megabytes one gigabyte holds, which every reportable estimate is rounded up to a multiple of."""

_BYTES_PER_MEGABYTE: int = 1024 * 1024
"""The divisor converting a byte count into megabytes."""

_SINGLE_PRECISION_BYTES: int = 4
"""The width of one single-precision element, the type every modeled working array holds."""

_ARCHIVE_DIRECTORY_RATIO: float = 4.0
"""The resident memory the runtime log reader holds per byte of archive on disk. Reading an archive builds one
directory entry per logged message, which dominates the decoded payload itself."""

_MODULE_TABLE_RATIO: float = 3.4
"""The resident memory a module parse job holds per byte of the archive its module came from. One module holds a
share of its controller's archive, and partitioning and merging its event streams materializes that share several
times over, so the whole archive bounds the job from above."""

_POSE_PREDICTION_RATIO: float = 6.0
"""The resident memory the pose-tracking job holds per byte of its prediction file. The table is promoted to double
precision and expanded into per-bodypart and per-metric arrays."""

_DECODER_BUFFER_MEMORY_MB: int = 96
"""The resident memory one decode worker holds for its codec reference frames and its decoder state, beyond the frame
buffers the measurement itself retains."""

_RETAINED_FRAME_BUFFERS: int = 2
"""The number of full-resolution single-precision frame buffers a decode worker keeps live. Binning produces a
strided view that retains its full-frame base, and the current and previous binned frames are live at once."""

_CHECKSUM_READER_MEMORY_MB: int = 285
"""The resident memory one checksum worker holds, covering its fixed read chunk and the interpreter and import graph
the worker starts with. The figure holds steady across sessions of any size, because a worker streams its file in
fixed chunks and never holds more than one at a time."""

_TRACE_ARRAY_DIMENSIONS: int = 2
"""The axes a cindra trace array carries, which are its regions and its samples."""

_DISCOVERY_FLOOR_MEMORY_MB: int = 5632
"""The memory a cross-recording discovery job is charged above the worker baseline when cindra refuses to size it. The
value is the allowance this package measured for the stage's clustering pass over this corpus, which is the term that
survives when nothing on disk predicts the job's shape. cindra's own model adds a registration term following the
combined frame and a clustering term quadratic in the regions the dataset spans, and neither is readable from a dataset
whose recordings carry no combined output yet. The stage declares no concurrency ceiling of its own and runs at a
narrow allocation, so this floor is what bounds how many of its jobs one batch admits at once."""

_ASSEMBLY_FLUORESCENCE_COLUMNS: int = 8
"""The fluorescence columns an experiment assembly retains at once. Every column is attached under its own name and
none replaces another, so each stays live in the assembled frame for the rest of the job."""

_ASSEMBLY_WRITE_COPIES: int = 1
"""The copies of the assembled fluorescence volume charged at the write. The write streams the frame it was handed
rather than rebuilding it, so the columns the assembly already holds are what the stage peaks at."""

_SUB_DATASET_BYTES_PER_SAMPLE: int = 512
"""The memory the behavior, runtime, and video sub-datasets hold per sample of the clock they are placed on. Each
emits one array per column and the interpolation that aligns them holds double-precision transients."""

_PERCENT_PER_FRACTION: float = 100.0
"""The divisor converting a percentage into a fraction."""


@dataclass(frozen=True, slots=True)
class _RecordingGeometry:
    """Describes the shape of a two-photon recording as its processing output reports it."""

    regions: int
    """The regions the single-recording pipeline detected."""
    samples: int
    """The samples each region's trace holds."""


@dataclass(frozen=True, slots=True)
class JobFootprint:
    """Describes the resources one job occupies while it runs, as this module's sizing pass resolved them."""

    cores: int
    """The cores the job is dispatched at."""
    memory_mb: int
    """The reportable memory the job holds at its peak, in megabytes."""
    memory_modeled: bool
    """Determines whether the memory figure follows from the job's own input rather than from the worker baseline."""


def resolve_host_memory_mb() -> int:
    """Reads the host's total physical memory.

    Returns:
        The host's total physical memory in megabytes.
    """
    return int(psutil.virtual_memory().total / _BYTES_PER_MEGABYTE)


def estimate_session_job_memory(
    pipeline: ProcessingPipelines, session: SessionData, jobs: list[tuple[str, str, int]]
) -> dict[tuple[str, str], JobFootprint]:
    """Sizes every possible job of one session, reporting the cores it is dispatched at and the memory it holds there.

    Notes:
        Reads on-disk metadata alone, so sizing a session never decodes a frame. The two extraction stages read their
        archive's central directory, which decodes no message and loads no payload. Every estimate scales with the
        input it describes, which keeps it correct on recordings longer, wider, or denser than any previously seen.

        Every term is read from the session's raw acquisition data, so a footprint is available before any stage has
        run. Estimates cover anonymous memory, the term that forces a host to swap and a scheduler to kill a job, so
        the reclaimable pages a memory-mapped stage leaves resident are excluded. Each footprint carries a flag
        stating whether the input its memory scales with was found. A job whose input is absent falls back to the
        worker baseline, which the flag marks as a floor to plan around.

        The stages a dependency owns are sized by that dependency's own sizing pass, which reads the job's input once
        and answers both halves of its model from that read. It picks the width the stage actually runs at, which is
        one core for an input below its parallel threshold and its declared allocation above it, and it estimates the
        memory at that width. Taking both figures whole is what keeps a retune of either half reaching slf without a
        change here, and it is what stops this package from reserving a width the library would never open. Those
        models reject an unreadable input rather than answering with a floor, so each delegated call is guarded and
        its refusal falls back to the declared allocation on this module's own baseline memory.

        Every other stage is this package's own, so it runs at the declared allocation the caller supplied and that
        allocation passes straight through into the footprint.

    Args:
        pipeline: The pipeline the jobs belong to.
        session: The loaded session the jobs operate on.
        jobs: The possible jobs as ``(job_name, specifier, declared_cores)`` triples.

    Returns:
        A dictionary mapping each ``(job_name, specifier)`` pair to the cores the job is dispatched at, its estimated
        memory in megabytes, and a flag that is True when the memory follows from the job's own input rather than
        from the worker baseline alone.
    """
    behavior_directory = session.raw_data.behavior_data_path
    output_root = session.processed_data_path
    geometry: RecordingGeometry | None = None
    configuration: SingleRecordingConfiguration | None = None
    data_path: Path | None = None
    widest_frame_pixels = 0
    if pipeline is ProcessingPipelines.TWO_PHOTON:
        resolve_configuration = resolve_single_recording_configuration_resolver(system=session.acquisition_system)
        configuration = resolve_configuration(session)
        locate_two_photon_data = resolve_two_photon_data_locator(system=session.acquisition_system)
        data_path = locate_two_photon_data(session)
        geometry = _resolve_two_photon_geometry(
            output_root=output_root, data_path=data_path, configuration=configuration
        )
    elif pipeline is ProcessingPipelines.VIDEO:
        widest_frame_pixels = _resolve_widest_camera_frame_pixels(camera_directory=session.raw_data.camera_data_path)

    footprints: dict[tuple[str, str], JobFootprint] = {}
    for job_name, specifier, cores in jobs:
        # The two extraction stages record the whole footprint their own library resolved, while every other stage
        # models its memory alone and is planned at the declared allocation the closing statement attaches.
        if pipeline is ProcessingPipelines.TWO_PHOTON:
            if geometry is None or configuration is None:
                memory_mb, memory_modeled = _apply_tolerance(memory_mb=WORKER_MEMORY_MB), False
            else:
                memory_mb, memory_modeled = _estimate_two_photon_memory(
                    job_name=job_name,
                    specifier=specifier,
                    output_root=output_root,
                    configuration=configuration,
                    data_path=data_path,
                )
        elif job_name == CHECKSUM_JOB_NAME:
            memory_mb, memory_modeled = _estimate_checksum_memory(cores=cores), True
        elif job_name == CAMERA_EXTRACTION_JOB_NAME:
            footprints[job_name, specifier] = _size_camera_extraction_job(
                archive_path=behavior_directory.joinpath(f"{specifier}{LOG_ARCHIVE_SUFFIX}"), declared_cores=cores
            )
            continue
        elif job_name == CONTROLLER_EXTRACTION_JOB_NAME:
            footprints[job_name, specifier] = _size_controller_extraction_job(
                archive_path=behavior_directory.joinpath(f"{specifier}{LOG_ARCHIVE_SUFFIX}"), declared_cores=cores
            )
            continue
        elif job_name == RUNTIME_JOB_NAME:
            archive = behavior_directory.joinpath(f"{specifier}{LOG_ARCHIVE_SUFFIX}")
            memory_mb, memory_modeled = _estimate_runtime_reader_memory(archive_path=archive, cores=cores)
        elif job_name == ENERGY_JOB_NAME:
            memory_mb, memory_modeled = _estimate_motion_energy_memory(frame_pixels=widest_frame_pixels, cores=cores)
        elif job_name == TRACKING_JOB_NAME:
            memory_mb, memory_modeled = _estimate_widest_file_memory(
                directory=session.raw_data.camera_data_path,
                pattern="*.h5",
                expansion_ratio=_POSE_PREDICTION_RATIO,
            )
        elif job_name == PARSE_JOB_NAME:
            memory_mb, memory_modeled = _estimate_widest_file_memory(
                directory=behavior_directory, pattern=f"*{LOG_ARCHIVE_SUFFIX}", expansion_ratio=_MODULE_TABLE_RATIO
            )
        else:
            memory_mb, memory_modeled = _apply_tolerance(memory_mb=WORKER_MEMORY_MB), False
        footprints[job_name, specifier] = JobFootprint(cores=cores, memory_mb=memory_mb, memory_modeled=memory_modeled)

    return footprints


def estimate_dataset_job_memory(
    dataset: DatasetData, jobs: list[tuple[str, str, int]]
) -> dict[tuple[str, str], JobFootprint]:
    """Sizes every possible forging job from the processed data it will read, reporting its cores and its memory.

    Notes:
        Reads array headers and the presence of the recording metadata alone, so sizing a dataset decodes no
        fluorescence and opens no binary. Each cross-recording stage scales with the processed data the
        single-recording pipeline wrote for the sessions that carry two-photon data.

        Every job is routed to a model rather than to a blanket allowance, since a remote scheduler reserves memory
        per job. A job whose processed input is absent falls back to the worker baseline, which the flag marks as a
        floor to plan around. The two cross-recording stages belong to cindra, so they are sized by cindra's own
        model, which refuses a dataset any recording leaves short rather than sizing it from the recordings that
        happen to be complete. That refusal becomes the fallback here, and the discovery stage carries an allowance
        above the baseline on it, because nothing else bounds how many of its jobs one batch admits at once.

        Every stage a dataset runs holds one width whatever data it reads, cindra's two among them, so the declared
        allocation the caller supplied is the width each of these jobs is dispatched at.

        The per-session assembly stage is this package's own, so no dependency models it and its projection stays
        here.

    Args:
        dataset: The resolved dataset the jobs operate on.
        jobs: The possible jobs as ``(job_name, specifier, declared_cores)`` triples.

    Returns:
        A dictionary mapping each ``(job_name, specifier)`` pair to the cores the job is dispatched at, its estimated
        memory in megabytes, and a flag that is True when the memory follows from the job's own input rather than
        from a flat allowance.
    """
    project_root = dataset.dataset_data_path.parent.parent
    animals = {entry.session: entry.animal for entry in dataset.sessions}
    configuration = _resolve_tracking_configuration(dataset=dataset, project_root=project_root)

    footprints: dict[tuple[str, str], JobFootprint] = {}
    for job_name, specifier, cores in jobs:
        if job_name == MULTIDAY_DISCOVERY_JOB_NAME:
            # The discovery stage runs over one animal, so its specifier names that animal rather than a session.
            memory_mb, memory_modeled = _estimate_multi_recording_memory(
                job_name=MultiRecordingJobNames.DISCOVER,
                specifier=specifier,
                recording_directories=_animal_recording_directories(
                    dataset=dataset, animal=specifier, project_root=project_root
                ),
                configuration=configuration,
                unmodeled_allowance_mb=_DISCOVERY_FLOOR_MEMORY_MB,
            )
        elif job_name == MULTIDAY_EXTRACTION_JOB_NAME:
            memory_mb, memory_modeled = _estimate_multi_recording_memory(
                job_name=MultiRecordingJobNames.EXTRACT,
                specifier=specifier,
                recording_directories=_animal_recording_directories(
                    dataset=dataset, animal=animals.get(specifier, ""), project_root=project_root
                ),
                configuration=configuration,
            )
        else:
            animal = animals.get(specifier, "")
            geometry = _resolve_recording_geometry(project_root=project_root, animal=animal, session=specifier)
            regions = _resolve_tracked_regions(
                dataset=dataset,
                animal=animal,
                session=specifier,
                project_root=project_root,
                configuration=configuration,
            )
            memory_mb, memory_modeled = _estimate_assembly_memory(geometry=geometry, regions=regions)
        footprints[job_name, specifier] = JobFootprint(cores=cores, memory_mb=memory_mb, memory_modeled=memory_modeled)

    return footprints


def _bytes_to_megabytes(byte_count: float) -> int:
    """Converts a byte count into whole megabytes, rounding up so an estimate never understates its demand.

    Args:
        byte_count: The number of bytes to convert.

    Returns:
        The equivalent size in megabytes.
    """
    return max(0, int(byte_count / _BYTES_PER_MEGABYTE) + 1) if byte_count > 0 else 0


def _round_to_gigabyte(memory_mb: int) -> int:
    """Rounds a memory figure up to a whole gigabyte, which is the quantum every estimate is reported at.

    Notes:
        A scheduler reserves memory in whole gigabytes, so an estimate landing mid-gigabyte is rounded there by
        whatever consumes it. Rounding here instead keeps the figure a plan records identical to the figure a
        submission requests, which is what lets a planned batch and a submitted one be compared directly.

        A figure a dependency already sized is rounded here as well. Those models report at their own quantum, which
        is finer than a gigabyte, so this is the boundary where every figure reaches one scale whichever model
        produced it.

    Args:
        memory_mb: The memory to round, in megabytes.

    Returns:
        The memory in megabytes, rounded up to a whole gigabyte.
    """
    return math.ceil(memory_mb / _MEGABYTES_PER_GIGABYTE) * _MEGABYTES_PER_GIGABYTE


def _apply_tolerance(memory_mb: int) -> int:
    """Applies the shared estimate tolerance to a modeled memory figure and rounds it to a whole gigabyte.

    Notes:
        The tolerance is cindra's, which the sibling acquisition libraries carry at the same value, so every stage of
        a mixed batch is weighed on one scale.

    Args:
        memory_mb: The modeled memory in megabytes, before any margin.

    Returns:
        The reportable memory in megabytes, carrying the tolerance and rounded up to a whole gigabyte.
    """
    return _round_to_gigabyte(memory_mb=int(memory_mb * MEMORY_ESTIMATE_TOLERANCE) + 1)


def _size_camera_extraction_job(archive_path: Path, declared_cores: int) -> JobFootprint:
    """Sizes one camera timestamp extraction job through the video library's own sizing pass.

    Notes:
        The library reads the archive once and answers both halves of its model from that read, picking a single core
        for an archive below its parallel-extraction threshold and its declared allocation above it, then estimating
        the memory at the width it picked. Both figures are taken whole, so this package neither repeats the width
        rule nor reserves cores for a pool the stage would not open.

        Reading the archive reads its central directory alone, and the library refuses an archive it cannot read
        because the job reading it could not run either. That refusal is the same condition this module reports as an
        unmodeled floor, so the job falls back to the allocation its type declares.

    Args:
        archive_path: The path to the log archive the job reads.
        declared_cores: The cores the job's type declares, which stand in when the archive cannot be read.

    Returns:
        The job's footprint, whose modeled flag is True when the archive was found and read.
    """
    try:
        sizing = size_camera_extraction_job(archive_path=archive_path)
    except FileNotFoundError:
        return JobFootprint(
            cores=declared_cores, memory_mb=_apply_tolerance(memory_mb=WORKER_MEMORY_MB), memory_modeled=False
        )
    return JobFootprint(
        cores=sizing.cores, memory_mb=_round_to_gigabyte(memory_mb=sizing.memory_mb), memory_modeled=True
    )


def _size_controller_extraction_job(archive_path: Path, declared_cores: int) -> JobFootprint:
    """Sizes one microcontroller data extraction job through the communication library's own sizing pass.

    Notes:
        The library reads the archive once and answers both halves of its model from that read, picking a single core
        for an archive below its parallel-extraction threshold and its declared allocation above it, then estimating
        the memory at the width it picked. Both figures are taken whole, so this package neither repeats the width
        rule nor reserves cores for a pool the stage would not open.

        Reading the archive reads its central directory alone, and the library refuses an archive it cannot read
        because the job reading it could not run either. That refusal is the same condition this module reports as an
        unmodeled floor, so the job falls back to the allocation its type declares.

    Args:
        archive_path: The path to the log archive the job reads.
        declared_cores: The cores the job's type declares, which stand in when the archive cannot be read.

    Returns:
        The job's footprint, whose modeled flag is True when the archive was found and read.
    """
    try:
        sizing = size_controller_extraction_job(archive_path=archive_path)
    except FileNotFoundError:
        return JobFootprint(
            cores=declared_cores, memory_mb=_apply_tolerance(memory_mb=WORKER_MEMORY_MB), memory_modeled=False
        )
    return JobFootprint(
        cores=sizing.cores, memory_mb=_round_to_gigabyte(memory_mb=sizing.memory_mb), memory_modeled=True
    )


def _estimate_runtime_reader_memory(archive_path: Path, cores: int) -> tuple[int, bool]:
    """Estimates the memory one runtime log job holds.

    Notes:
        The stage splits one log archive across a worker pool, and every worker opens the archive itself, so the
        archive's directory is held once per allocated core. The runtime pipeline is this package's own, so no
        dependency models it.

    Args:
        archive_path: The path to the log archive the job reads.
        cores: The cores the job is allocated, which is how many readers it opens.

    Returns:
        The reportable memory in megabytes and a flag that is True when the archive was found and read.
    """
    if not archive_path.is_file():
        return _apply_tolerance(memory_mb=WORKER_MEMORY_MB), False
    per_reader = _bytes_to_megabytes(byte_count=archive_path.stat().st_size * _ARCHIVE_DIRECTORY_RATIO)
    return _apply_tolerance(memory_mb=WORKER_MEMORY_MB + cores * (per_reader + SPAWNED_CHILD_MEMORY_MB)), True


def _estimate_checksum_memory(cores: int) -> int:
    """Estimates the memory one raw-data checksum job holds.

    Notes:
        Each worker streams its file in fixed chunks and holds one at a time, so the figure is flat across every
        session size. The parent retains one pending result per file, which stays below the rounding this estimate
        already carries.

    Args:
        cores: The cores the job is allocated, which is how many files it hashes at once.

    Returns:
        The reportable memory in megabytes.
    """
    return _apply_tolerance(memory_mb=WORKER_MEMORY_MB + cores * _CHECKSUM_READER_MEMORY_MB)


def _resolve_widest_camera_frame_pixels(camera_directory: Path) -> int:
    """Reads the pixel count of the largest frame among a video session's camera recordings.

    Notes:
        Reads container metadata only, so no frame is decoded. The largest frame is a property of the session rather
        than of any one job, so it is read once and shared by every motion-energy job the session dispatches.

    Args:
        camera_directory: The raw camera directory holding the session's recordings.

    Returns:
        The pixels the largest recorded frame holds, or zero when the session holds no readable recordings.
    """
    if not camera_directory.is_dir():
        return 0

    widest = 0
    for recording in natsorted(camera_directory.glob("*.mp4")):
        capture = cv2.VideoCapture(str(recording))
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()
        widest = max(widest, max(0, width) * max(0, height))
    return widest


def _estimate_motion_energy_memory(frame_pixels: int, cores: int) -> tuple[int, bool]:
    """Estimates the memory one video motion-energy job holds, from the frame it decodes.

    Notes:
        Each decode worker holds the binned frame and its predecessor. Binning slices a strided view out of a
        full-resolution buffer, and a view retains its base, so both cost the whole frame. The frame buffers are a
        small share of a worker's memory next to the cost of the worker itself. Charging every job of a session its
        widest frame therefore stays on the safe side at a cost well inside the tolerance.

    Args:
        frame_pixels: The pixels the frame this job decodes holds.
        cores: The cores the job is allocated, which bounds how many chunks it decodes at once.

    Returns:
        The reportable memory in megabytes and a flag that is always True, because the per-core decoder and child
        cost is modeled whether or not a readable recording supplied a frame.
    """
    per_worker = (
        _bytes_to_megabytes(byte_count=frame_pixels * _SINGLE_PRECISION_BYTES * _RETAINED_FRAME_BUFFERS)
        + _DECODER_BUFFER_MEMORY_MB
        + SPAWNED_CHILD_MEMORY_MB
    )
    return _apply_tolerance(memory_mb=WORKER_MEMORY_MB + cores * per_worker), True


def _resolve_two_photon_geometry(
    output_root: Path, data_path: Path, configuration: SingleRecordingConfiguration
) -> RecordingGeometry | None:
    """Reads a two-photon recording's shape through cindra's own geometry resolver.

    Notes:
        Resolved once per session rather than once per job, because the resolution reads the acquisition metadata and
        one source file header while every job of the recording answers to the same shape.

        The result gates the session's two-photon estimates rather than sizing them, since cindra sizes each of its
        stages from the recording itself. A session whose geometry resolves to None carries no readable raw imaging
        data, so every one of its jobs falls back to the worker baseline.

    Args:
        output_root: The output root the recording's cindra configuration was given.
        data_path: The raw imaging directory holding the recording's source files.
        configuration: The recording's resolved processing configuration, consulted for the image names the
            conversion stage excludes.

    Returns:
        The recording's geometry, or None when the session holds no readable raw imaging data.
    """
    try:
        geometry = resolve_recording_geometry(
            output_root=output_root,
            data_path=data_path,
            ignored_file_names=tuple(configuration.file_io.ignored_file_names),
        )
    except OSError, ValueError, RuntimeError:
        return None
    return geometry if geometry.planes else None


def _estimate_two_photon_memory(
    job_name: str,
    specifier: str,
    output_root: Path,
    configuration: SingleRecordingConfiguration,
    data_path: Path | None,
) -> tuple[int, bool]:
    """Estimates the memory one two-photon job holds, through cindra's own per-stage model.

    Notes:
        cindra rejects a stage it cannot size, either because the recording carries no readable raw imaging data or
        because a per-plane specifier names a plane the recording does not hold. Both are the condition this module
        reports as an unmodeled floor, so the refusal is translated rather than propagated.

    Args:
        job_name: The tracker job name identifying the stage.
        specifier: The plane specifier for a registration or processing job, empty for the binarization and
            combination stages.
        output_root: The output root the recording's cindra configuration was given.
        configuration: The recording's resolved processing configuration.
        data_path: The raw imaging directory, consulted when the recording carries no output yet.

    Returns:
        The reportable memory in megabytes and a flag that is True when cindra sized the stage.
    """
    try:
        memory_mb = estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames(job_name),
            specifier=specifier,
            output_root=output_root,
            configuration=configuration,
            data_path=data_path,
        )
    except FileNotFoundError, ValueError:
        return _apply_tolerance(memory_mb=WORKER_MEMORY_MB), False
    return _round_to_gigabyte(memory_mb=memory_mb), True


def _estimate_multi_recording_memory(
    job_name: MultiRecordingJobNames,
    specifier: str,
    recording_directories: tuple[Path, ...],
    configuration: MultiRecordingConfiguration | None,
    unmodeled_allowance_mb: int = 0,
) -> tuple[int, bool]:
    """Estimates the memory one cross-recording job holds, through cindra's own per-stage model.

    Notes:
        Both cross-recording stages read every recording of the animal they run over, so the whole recording set is
        handed to the model whichever stage is being sized. cindra refuses a set any recording leaves short, which is
        the condition this module reports as an unmodeled floor.

        A stage whose refusal would otherwise report the bare worker baseline carries an allowance above it instead,
        which stands in for the terms cindra's model would have read. That floor is what bounds how many of the
        stage's jobs a batch admits at once, since a narrow allocation alone lets a core budget admit a great many.

    Args:
        job_name: The cindra stage the job runs.
        specifier: The job's tracker specifier, which names a session for the extraction stage and an animal for the
            discovery stage.
        recording_directories: The cindra output directory of every recording the job spans.
        configuration: The dataset's resolved multi-recording configuration, or None when its acquisition system
            donates none.
        unmodeled_allowance_mb: The memory charged above the worker baseline when cindra refuses to size the stage.

    Returns:
        The reportable memory in megabytes and a flag that is True when cindra sized the stage.
    """
    if configuration is None or not recording_directories:
        return _apply_tolerance(memory_mb=WORKER_MEMORY_MB + unmodeled_allowance_mb), False
    try:
        memory_mb = estimate_multi_recording_job_memory_mb(
            job_name=job_name,
            specifier=specifier,
            recording_directories=recording_directories,
            configuration=configuration,
        )
    except FileNotFoundError, ValueError, RuntimeError:
        return _apply_tolerance(memory_mb=WORKER_MEMORY_MB + unmodeled_allowance_mb), False
    return _round_to_gigabyte(memory_mb=memory_mb), True


def _estimate_widest_file_memory(directory: Path, pattern: str, expansion_ratio: float) -> tuple[int, bool]:
    """Estimates memory from the largest file in a directory matching a pattern, for a stage whose input is one of
    several files it may read.

    Notes:
        Inputs are discovered by extension rather than by name, because the naming of an acquired file belongs to the
        acquisition system that produced it.

    Args:
        directory: The directory to search.
        pattern: The glob pattern the candidate files match.
        expansion_ratio: The resident memory a job holds per byte of the file it reads.

    Returns:
        The reportable memory in megabytes and a flag that is True when a candidate file was found and measured.
    """
    if not directory.is_dir():
        return _apply_tolerance(memory_mb=WORKER_MEMORY_MB), False
    candidates = sorted(directory.glob(pattern), key=lambda path: path.stat().st_size, reverse=True)
    if not candidates:
        return _apply_tolerance(memory_mb=WORKER_MEMORY_MB), False
    widest = candidates[0]
    return _apply_tolerance(
        memory_mb=WORKER_MEMORY_MB + _bytes_to_megabytes(byte_count=widest.stat().st_size * expansion_ratio)
    ), True


@cache
def _two_photon_output_directory(project_root: Path, animal: str, session: str) -> Path:
    """Resolves a session's single-recording two-photon output directory through the session hierarchy.

    Notes:
        Cached, because one dataset's estimates resolve the same session from several stages and each resolution
        otherwise re-reads that session's marker.

    Args:
        project_root: The path to the project's root directory.
        animal: The animal the session belongs to.
        session: The session name whose output directory is resolved.

    Returns:
        The path to the session's cindra output directory.
    """
    return SessionData.load(session_path=project_root.joinpath(animal, session)).processed_data.cindra_data_path


def _animal_recording_directories(dataset: DatasetData, animal: str, project_root: Path) -> tuple[Path, ...]:
    """Resolves the cindra output directory of every recording one animal contributes to a dataset.

    Notes:
        This is the recording set the animal's materialized multi-recording configuration names, resolved from the
        project root rather than read back from that file, so a dataset whose configurations have not been written
        yet is still sizable. A session that has moved to long-term storage contributes no directory.

    Args:
        dataset: The resolved dataset the animal belongs to.
        animal: The animal whose recordings are resolved.
        project_root: The path to the project's root directory.

    Returns:
        The cindra output directory of every recording still present under the project root.
    """
    directories: list[Path] = []
    for entry in dataset.get_sessions_for_animal(animal=animal):
        if not project_root.joinpath(animal, entry.session).is_dir():
            continue
        directories.append(
            _two_photon_output_directory(project_root=project_root, animal=animal, session=entry.session)
        )
    return tuple(directories)


def _resolve_recording_geometry(project_root: Path, animal: str, session: str) -> _RecordingGeometry | None:
    """Reads a processed recording's shape from the arrays the single-recording pipeline wrote.

    Notes:
        The combined metadata archive gates the result alongside the trace array, because both are written by the
        combination stage and a recording that has not reached the end of it carries no output the forging stages can
        read. Neither file's contents are loaded, so the resolution reads array headers and directory entries alone.

    Args:
        project_root: The path to the project's root directory.
        animal: The animal the session belongs to.
        session: The session name whose processed output is read.

    Returns:
        The recording's geometry, or None when the session holds no processed imaging output.
    """
    directory = _two_photon_output_directory(project_root=project_root, animal=animal, session=session)
    traces = _read_array_shape(
        array_path=resolve_array_path(root_path=directory, array=RecordingArrays.CELL_FLUORESCENCE)
    )
    if traces is None or not directory.joinpath(COMBINED_METADATA_FILENAME).is_file():
        return None
    return _RecordingGeometry(regions=traces[0], samples=traces[1])


def _read_array_shape(array_path: Path) -> tuple[int, int] | None:
    """Parses the shape a two-dimensional array's own header reports.

    Args:
        array_path: The path to the array whose header is parsed.

    Returns:
        The array's two extents, or None when it is absent or carries another rank.
    """
    if not array_path.is_file():
        return None
    with array_path.open("rb") as array_file:
        reader = read_array_header_1_0 if read_magic(array_file) == (1, 0) else read_array_header_2_0
        shape, _, _ = reader(array_file)
    if len(shape) != _TRACE_ARRAY_DIMENSIONS:
        return None
    return int(shape[0]), int(shape[1])


def _resolve_tracking_configuration(dataset: DatasetData, project_root: Path) -> MultiRecordingConfiguration | None:
    """Resolves the multi-recording configuration the dataset's acquisition system donates.

    Notes:
        Read from the system registry rather than from the file ``define_forging_dataset`` materializes, so the
        parameters are available for a dataset whose configurations have not been written yet.

        Resolved from the first session still present under the project root, because a dataset outlives the source
        data of the animals it has already forged. Estimating a newly added animal therefore does not depend on
        sessions that have moved to long-term storage.

    Args:
        dataset: The resolved dataset whose acquisition system donates the configuration.
        project_root: The path to the project's root directory.

    Returns:
        The resolved configuration, or None when no session remains under the project root, or when the dataset's
        sessions need no multi-day processing.
    """
    resolve_configuration = resolve_multi_recording_configuration_resolver(system=dataset.acquisition_system)
    for entry in dataset.sessions:
        session_path = project_root.joinpath(entry.animal, entry.session)
        if not session_path.is_dir():
            continue
        return resolve_configuration(SessionData.load(session_path=session_path))
    return None


def _resolve_tracked_regions(
    dataset: DatasetData,
    animal: str,
    session: str,
    project_root: Path,
    configuration: MultiRecordingConfiguration | None,
) -> int:
    """Resolves how many regions a session's multi-day arrays hold.

    Notes:
        Reads the multi-day array directly once the session's multi-day extraction job has written it, and bounds the
        count from the single-recording geometries until then. Tracking keeps a cluster whenever it appears in enough
        of the animal's recordings, so the pooled region count divided by that minimum bounds the templates, narrowed
        again to the widest single recording the animal holds.

    Args:
        dataset: The resolved dataset the session belongs to.
        animal: The animal the session belongs to.
        session: The session name whose tracked regions are resolved.
        project_root: The path to the project's root directory.
        configuration: The resolved multi-recording configuration, which reports the prevalence a cluster must reach.

    Returns:
        The tracked region count, or the bound standing in for it.
    """
    entries = dataset.get_sessions_for_animal(animal=animal)
    geometries = [
        geometry
        for entry in entries
        if (geometry := _resolve_recording_geometry(project_root=project_root, animal=animal, session=entry.session))
        is not None
    ]
    if not geometries:
        return 1

    tracked = _read_array_shape(
        array_path=resolve_array_path(
            root_path=_two_photon_output_directory(project_root=project_root, animal=animal, session=session).joinpath(
                MULTI_RECORDING_DIRECTORY_NAME,
                multi_recording_dataset_directory(animal_id=animal, dataset_name=dataset.name),
            ),
            array=RecordingArrays.CELL_FLUORESCENCE,
        )
    )
    if tracked is not None:
        return tracked[0]

    prevalence = configuration.roi_tracking.mask_prevalence if configuration is not None else 0.0
    minimum_recordings = max(1, math.ceil(prevalence / _PERCENT_PER_FRACTION * len(geometries)))
    pooled = sum(geometry.regions for geometry in geometries) // minimum_recordings
    # A template is one cluster of regions drawn from several recordings, so the count settles at the scale of a
    # single recording's own regions rather than the pooled total the prevalence term alone allows.
    return max(1, min(pooled, max(geometry.regions for geometry in geometries)))


def _estimate_assembly_memory(geometry: _RecordingGeometry | None, regions: int) -> tuple[int, bool]:
    """Estimates the memory one per-session assembly job holds.

    Notes:
        The assembled frame retains every fluorescence column it attaches, and the write that closes the job rechunks
        the frame into a second copy of the whole thing. A session carrying no fluorescence falls back to the worker
        baseline, which the flag marks as a floor to plan around.

    Args:
        geometry: The session's processed geometry.
        regions: The regions each retained fluorescence column spans.

    Returns:
        The reportable memory in megabytes and a flag stating whether the fluorescence geometry was found.
    """
    if geometry is None:
        return _apply_tolerance(memory_mb=WORKER_MEMORY_MB), False

    columns = (
        _ASSEMBLY_FLUORESCENCE_COLUMNS * _ASSEMBLY_WRITE_COPIES * geometry.samples * regions * _SINGLE_PRECISION_BYTES
    )
    sub_datasets = geometry.samples * _SUB_DATASET_BYTES_PER_SAMPLE
    return (
        _apply_tolerance(memory_mb=WORKER_MEMORY_MB + _bytes_to_megabytes(byte_count=columns + sub_datasets)),
        True,
    )
