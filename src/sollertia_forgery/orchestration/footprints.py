"""Provides the sizing pass that resolves each job's cores and working set from the data it will process."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from hashlib import sha256
from functools import cache
from dataclasses import dataclass

import cv2
from cindra import (
    WORKER_MEMORY_MB,
    SPAWNED_CHILD_MEMORY_MB,
    MEMORY_ESTIMATE_TOLERANCE,
    RecordingArrays,
    MultiRecordingJobNames,
    SingleRecordingJobNames,
    resolve_array_path,
    resolve_output_path,
    resolve_dataset_path,
    is_recording_processed,
    size_multi_recording_job,
    size_single_recording_job,
)
import polars as pl
import psutil
from natsort import natsorted
from numpy.lib.format import read_magic, read_array_header_1_0, read_array_header_2_0
from ataraxis_video_system import (
    OutputLayout,
    size_archive_job as size_camera_extraction_job,
)
from ataraxis_base_utilities import console
from sollertia_shared_assets import SessionData, SessionTypes
from ataraxis_data_structures import find_log_archives, read_archive_message_count
from ataraxis_communication_interface import size_archive_job as size_controller_extraction_job

from ..video import ENERGY_JOB_NAME, RENAME_JOB_NAME, TRACKING_JOB_NAME, CAMERA_EXTRACTION_JOB_NAME
from ..forging import FORGING_JOB_NAME, MULTIDAY_DISCOVERY_JOB_NAME, MULTIDAY_EXTRACTION_JOB_NAME
from ..runtime import RUNTIME_JOB_NAME
from ..managing import CHECKSUM_JOB_NAME
from ..registries import (
    resolve_pose_prediction_locator,
    resolve_two_photon_data_locator,
    resolve_forging_admission_pipelines,
    resolve_multi_recording_configuration_resolver,
    resolve_single_recording_configuration_resolver,
)
from ..shared_assets import ProcessingPipelines, multi_recording_dataset_name
from ..microcontrollers import PARSE_JOB_NAME, CONTROLLER_EXTRACTION_JOB_NAME

if TYPE_CHECKING:
    from pathlib import Path

    from cindra import MultiRecordingConfiguration, SingleRecordingConfiguration
    from sollertia_shared_assets import DatasetData

_MODEL_VERSION_DIGITS: int = 12
"""The width of the digest that identifies this module's sizing model."""

_MEGABYTES_PER_GIGABYTE: int = 1024
"""The megabytes one gigabyte holds. Every reportable estimate is rounded up to a multiple of that figure."""

_BYTES_PER_MEGABYTE: int = 1024 * 1024
"""The divisor converting a byte count into megabytes."""

_SINGLE_PRECISION_BYTES: int = 4
"""The width of one single-precision element, the type every modeled working array holds."""

_ARCHIVE_DIRECTORY_BYTES_PER_MESSAGE: int = 768
"""The resident memory the runtime log reader holds per message its archive carries. Reading an archive builds one
directory entry per logged message, and that directory dominates the decoded payload itself, so the cost follows the
message count rather than the bytes those messages occupy."""

_MODULE_TABLE_RATIO: float = 3.4
"""The resident memory a module parse job holds per byte of the archive from which its module came. One module holds
a share of its controller's archive, and partitioning and merging its event streams materializes that share several
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

_CHECKSUM_READER_MEMORY_MB: int = 190
"""The resident memory one checksum worker holds, covering its fixed read chunk and the interpreter and import graph
loaded at its start. The figure holds steady across sessions of any size, because a worker streams its file in fixed
chunks and never holds more than one at a time."""

_TRACE_ARRAY_DIMENSIONS: int = 2
"""The axes a cindra trace array carries, which are its regions and its samples."""

_ASSEMBLY_SINGLE_DAY_COLUMNS: int = 4
"""The fluorescence columns an experiment assembly retains at the recording's own detected region count. Every column
is attached under its own name and none replaces another, so each stays live in the assembled frame for the rest of
the job."""

_ASSEMBLY_MULTI_DAY_COLUMNS: int = 4
"""The fluorescence columns an experiment assembly retains at the count of regions tracked across the animal's
recordings. Tracking keeps a region only where it appears in enough of those recordings, so these columns are narrower
than their single-day counterparts."""

_ASSEMBLY_WRITE_COPIES: int = 1
"""The copies of the assembled fluorescence volume charged at the write. The write streams the frame it was handed
rather than rebuilding it, so the stage peaks at the columns the assembly already holds."""

_SUB_DATASET_BYTES_PER_SAMPLE: int = 512
"""The memory the behavior, runtime, and video sub-datasets hold per sample of the clock on which they are placed.
Each emits one array per column and the interpolation that aligns them holds double-precision transients."""

_MINIMUM_CLOCK_SAMPLES: int = 2
"""The fewest samples a camera timestamp feather must hold for the assembly of a session that records no imaging to
settle on it. That assembly places its columns on a camera clock, and a clock is defined by a mean frame rate, which
needs at least two timestamps to state one."""

_PERCENT_PER_FRACTION: float = 100.0
"""The divisor converting a percentage into a fraction."""

_MODULE_SPECIFIER_SEPARATOR: str = "-"
"""The separator joining the controller, type, and identifier segments of a module parse job's specifier. The leading
segment names the controller that recorded the module. The job's estimate reads that controller's archive."""

_ARCHIVE_JOB_NAMES: frozenset[str] = frozenset(
    {CAMERA_EXTRACTION_JOB_NAME, CONTROLLER_EXTRACTION_JOB_NAME, RUNTIME_JOB_NAME}
)
"""The job types whose input is one source's log archive, which the data-structures locator resolves for them.

Notes:
    Each of these jobs is specified by the identifier of the source that recorded its archive, so one locating pass
    over the session's raw behavior data answers every one of them.
"""


@dataclass(frozen=True, slots=True)
class JobFootprint:
    """Describes the resources one job occupies while it runs, as this module's sizing pass resolved them.

    Notes:
        Carries the two fields every dependency's own sizing record carries, so a stage that a library owns and a
        stage that this package owns describe themselves the same way. Both halves follow from the job's input, and a
        job whose input cannot be read raises rather than reporting a footprint nothing measured.
    """

    cores: int
    """The cores the job occupies."""
    memory_mb: int
    """The reportable memory the job holds at its peak, in megabytes."""


def resolve_model_version() -> str:
    """Resolves the identifier of the sizing model this module implements.

    Notes:
        The identifier digests the name and the value of every private constant this module defines, so retuning any
        one of them answers with a new identifier and every plan stamped with the old one is estimated again. A
        constant holding a collection is ordered before it is digested, which keeps one tuning answering with one
        identifier across processes.

    Returns:
        The hexadecimal identifier of the sizing model.
    """
    digest = sha256()
    for name, value in sorted(globals().items()):
        if not name.startswith("_") or not name.isupper():
            continue
        ordered = sorted(value) if isinstance(value, frozenset | set) else value
        digest.update(f"{name}={ordered!r}\n".encode())
    return digest.hexdigest()[:_MODEL_VERSION_DIGITS]


def resolve_host_memory_mb() -> int:
    """Reads the host's total physical memory.

    Returns:
        The host's total physical memory in megabytes.
    """
    return int(psutil.virtual_memory().total / _BYTES_PER_MEGABYTE)


def size_session_jobs(
    pipeline: ProcessingPipelines, session: SessionData, jobs: list[tuple[str, str, int]]
) -> dict[tuple[str, str], JobFootprint]:
    """Sizes every possible job of one session, reporting the cores it occupies and the memory it holds there.

    Notes:
        Reads on-disk metadata alone, so sizing a session never decodes a frame. The two extraction stages read their
        archive's central directory, which decodes no message and loads no payload. Every estimate scales with the
        input it describes, which keeps it correct on recordings longer, wider, or denser than any previously seen.

        Every term is read from the session's raw acquisition data, so a footprint is available before any stage has
        run. Estimates cover anonymous memory, the term that forces a host to swap and a scheduler to kill a job, so
        the reclaimable pages a memory-mapped stage leaves resident are excluded.

        The stages that a dependency owns are sized by that dependency's own sizing pass, which reads the job's input
        once and answers both halves of its model from that read. It picks the width at which the stage actually
        runs, which for the extraction stages is one core for an archive below their parallel threshold and their
        declared allocation above it, and it estimates the memory at that width. Taking both figures whole is what
        keeps a retune of either half reaching slf without a change here, and it is what stops this package from
        reserving a width the library would never open.

        Every other stage is this package's own, so it is modeled here in the same shape, where one call answers both
        halves from the job's input. A stage whose cost holds one width, whatever data it reads, reports the allocation
        its type declared, which reaches this pass alongside the job.

        No stage answers with a floor. A job whose input cannot be read is a job that cannot run, so the refusal
        raised by the read propagates to the caller, which drops the target holding it rather than planning it at a
        figure nothing measured.

    Args:
        pipeline: The pipeline that owns the jobs.
        session: The loaded session on which the jobs operate.
        jobs: The possible jobs as ``(job_name, specifier, declared_cores)`` triples.

    Returns:
        A dictionary mapping each ``(job_name, specifier)`` pair to the cores the job occupies and the memory it
        holds there, in megabytes.

    Raises:
        FileNotFoundError: If a job's input cannot be read, in which case the job that reads it cannot run either.
        OSError: If any directory beneath the session's raw behavior data cannot be read while its log archives are
            located.
        ValueError: If a two-photon job's specifier names an imaging plane the recording does not hold, or if the
            recording's acquisition metadata cannot be parsed. Also raised when the session's raw behavior data holds
            more than one archive for a source, or when a job name routes to no sizing model.
    """
    # Every job of the two-photon pipeline runs a cindra stage, so the whole pipeline is sized by cindra's own pass
    # and the recording's configuration and raw imaging location are resolved once for the session that carries them.
    if pipeline is ProcessingPipelines.TWO_PHOTON:
        resolve_configuration = resolve_single_recording_configuration_resolver(system=session.acquisition_system)
        locate_two_photon_data = resolve_two_photon_data_locator(system=session.acquisition_system)
        configuration = resolve_configuration(session)
        data_path = locate_two_photon_data(session)
        return {
            (job_name, specifier): _size_two_photon_job(
                job_name=job_name,
                specifier=specifier,
                output_root=session.processed_data_path,
                configuration=configuration,
                data_path=data_path,
            )
            for job_name, specifier, _ in jobs
        }

    behavior_directory = session.raw_data.behavior_data_path
    camera_directory = session.raw_data.camera_data_path
    widest_frame_pixels = (
        _resolve_widest_camera_frame_pixels(camera_directory=camera_directory)
        if pipeline is ProcessingPipelines.VIDEO
        else 0
    )

    # Locating a source's archive belongs to the library that writes it, and one pass answers every archive-reading
    # job the session holds. A source that the pass cannot resolve raises there, which is the same refusal each sizing
    # model raises for an input it cannot read.
    archives = _resolve_job_archives(behavior_directory=behavior_directory, jobs=jobs)

    footprints: dict[tuple[str, str], JobFootprint] = {}
    for job_name, specifier, cores in jobs:
        if job_name == CHECKSUM_JOB_NAME:
            footprint = _size_checksum_job(cores=cores)
        elif job_name == CAMERA_EXTRACTION_JOB_NAME:
            footprint = _size_camera_extraction_job(archive_path=archives[specifier])
        elif job_name == CONTROLLER_EXTRACTION_JOB_NAME:
            footprint = _size_controller_extraction_job(archive_path=archives[specifier])
        elif job_name == RUNTIME_JOB_NAME:
            footprint = _size_runtime_job(archive_path=archives[specifier], cores=cores)
        elif job_name == ENERGY_JOB_NAME:
            footprint = _size_motion_energy_job(frame_pixels=widest_frame_pixels, cores=cores)
        elif job_name == TRACKING_JOB_NAME:
            footprint = _size_pose_tracking_job(session=session, cores=cores)
        elif job_name == PARSE_JOB_NAME:
            footprint = _size_module_parse_job(
                archive_path=archives[specifier.split(_MODULE_SPECIFIER_SEPARATOR)[0]], cores=cores
            )
        elif job_name == RENAME_JOB_NAME:
            footprint = _size_rename_job(cores=cores)
        else:
            message = (
                f"Unable to size job '{job_name}' of session '{session.session_name}'. The job type routes to no "
                f"sizing model, and a job whose resources nothing resolves cannot be admitted to a batch."
            )
            console.error(message=message, error=ValueError)
        footprints[job_name, specifier] = footprint

    return footprints


def size_dataset_jobs(dataset: DatasetData, jobs: list[tuple[str, str, int]]) -> dict[tuple[str, str], JobFootprint]:
    """Sizes every possible forging job from the processed data it will read, reporting its cores and its memory.

    Notes:
        Reads array headers, feather metadata and the presence of the recording metadata alone, so sizing a dataset
        decodes no fluorescence and reads no timestamp. Each cross-recording stage scales with the processed data the
        single-recording pipeline wrote for the sessions that carry two-photon data.

        Every job is routed to a model rather than to a blanket allowance, since a remote scheduler reserves memory
        per job. The two cross-recording stages belong to cindra, so both halves of their figures are cindra's own
        sizing pass, which refuses a dataset that any recording leaves short rather than sizing it from the recordings
        that happen to be complete. That refusal propagates, because a stage that cindra will not size is a stage the
        dataset cannot run until its recordings are complete.

        The per-session assembly stage is this package's own, so no dependency models it and its projection stays
        here. Its width holds one value whatever data it reads, so it reports the allocation its type declared. Which
        of its two models applies follows from the acquisition system's admission policy: a session type that joins a
        dataset without completing the two-photon pipeline is assembled from its behavior sources onto a camera clock,
        so it is sized from that clock rather than from fluorescence it never recorded.

    Args:
        dataset: The resolved dataset on which the jobs operate.
        jobs: The possible jobs as ``(job_name, specifier, declared_cores)`` triples.

    Returns:
        A dictionary mapping each ``(job_name, specifier)`` pair to the cores the job occupies and the memory it
        holds there, in megabytes.

    Raises:
        FileNotFoundError: If a job's processed input cannot be read, in which case the job that reads it cannot run
            either.
        ValueError: If the dataset's acquisition system donates no multi-recording configuration, if a job name
            routes to no sizing model, or if the dataset's session type joins no dataset for its acquisition system.
    """
    project_root = dataset.dataset_data_path.parent.parent
    animals = {entry.session: entry.animal for entry in dataset.sessions}
    configuration = _resolve_tracking_configuration(dataset=dataset, project_root=project_root)

    footprints: dict[tuple[str, str], JobFootprint] = {}
    for job_name, specifier, cores in jobs:
        if job_name == MULTIDAY_DISCOVERY_JOB_NAME:
            # The discovery stage runs over one animal, so its specifier names that animal rather than a session.
            footprint = _size_multi_recording_job(
                job_name=MultiRecordingJobNames.DISCOVER,
                specifier=specifier,
                recording_directories=_animal_recording_directories(
                    dataset=dataset, animal=specifier, project_root=project_root
                ),
                configuration=configuration,
            )
        elif job_name == MULTIDAY_EXTRACTION_JOB_NAME:
            footprint = _size_multi_recording_job(
                job_name=MultiRecordingJobNames.EXTRACT,
                specifier=specifier,
                recording_directories=_animal_recording_directories(
                    dataset=dataset, animal=animals.get(specifier, ""), project_root=project_root
                ),
                configuration=configuration,
            )
        elif job_name == FORGING_JOB_NAME:
            footprint = _size_forging_job(
                dataset=dataset,
                animal=animals.get(specifier, ""),
                session=specifier,
                project_root=project_root,
                configuration=configuration,
                cores=cores,
            )
        else:
            message = (
                f"Unable to size job '{job_name}' of dataset '{dataset.name}'. The job type routes to no sizing "
                f"model, and a job whose resources nothing resolves cannot be admitted to a batch."
            )
            console.error(message=message, error=ValueError)
        footprints[job_name, specifier] = footprint

    return footprints


@dataclass(frozen=True, slots=True)
class _RecordingGeometry:
    """Describes the shape of a two-photon recording as its processing output reports it."""

    regions: int
    """The regions the single-recording pipeline detected."""
    samples: int
    """The samples each region's trace holds."""


def _resolve_job_archives(behavior_directory: Path, jobs: list[tuple[str, str, int]]) -> dict[str, Path]:
    """Locates the log archive every archive-reading job of one session consumes.

    Notes:
        The archive filename that a source writes is the data-structures library's own contract, so the sources are
        handed to its locator rather than having their filenames rebuilt here. One traversal resolves every source,
        which is the same pass the acquisition libraries' own job resolvers make.

    Args:
        behavior_directory: The session's raw behavior data directory, whose tree holds every archive it recorded.
        jobs: The possible jobs as ``(job_name, specifier, declared_cores)`` triples, whose archive-reading members
            carry the identifier of the source that recorded their archive.

    Returns:
        The path to the archive of every source the jobs read, keyed by that source identifier. Empty when the jobs
        read no archive.

    Raises:
        FileNotFoundError: If the behavior data directory is absent, or if a source recorded no archive, in which case
            the job reading it cannot run either.
        OSError: If any directory beneath the behavior data directory cannot be read.
        ValueError: If the tree holds more than one archive for a source, which leaves the job's input ambiguous.
    """
    sources = {specifier for job_name, specifier, _ in jobs if job_name in _ARCHIVE_JOB_NAMES}
    # A parse job is specified by its module rather than by a source, and the controller that recorded the module is
    # the leading segment of that name, so the archive its estimate reads is recovered from the specifier.
    sources.update(
        specifier.split(_MODULE_SPECIFIER_SEPARATOR)[0] for job_name, specifier, _ in jobs if job_name == PARSE_JOB_NAME
    )
    if not sources:
        return {}
    return find_log_archives(log_directory=behavior_directory, source_ids=sorted(sources))


def _bytes_to_megabytes(byte_count: float) -> int:
    """Converts a byte count into whole megabytes, rounding up so an estimate never understates its demand.

    Args:
        byte_count: The number of bytes to convert.

    Returns:
        The equivalent size in megabytes.
    """
    return max(0, int(byte_count / _BYTES_PER_MEGABYTE) + 1) if byte_count > 0 else 0


def _round_to_gigabyte(memory_mb: int) -> int:
    """Rounds a memory figure up to a whole gigabyte, which is the quantum at which every estimate is reported.

    Notes:
        A scheduler reserves memory in whole gigabytes, so an estimate landing mid-gigabyte is rounded there by
        whatever consumes it. Rounding here instead keeps the figure a plan records identical to the figure a
        submission requests, so a planned batch and a submitted one compare directly.

        A figure that a dependency already sized is rounded here as well. The acquisition libraries report at a 256
        megabyte quantum while cindra already reports at a whole gigabyte, so this is the boundary where every figure
        reaches one scale whichever model produced it.

    Args:
        memory_mb: The memory to round, in megabytes.

    Returns:
        The memory in megabytes, rounded up to a whole gigabyte.
    """
    return math.ceil(memory_mb / _MEGABYTES_PER_GIGABYTE) * _MEGABYTES_PER_GIGABYTE


def _apply_tolerance(memory_mb: int) -> int:
    """Applies the shared estimate tolerance to a modeled memory figure and rounds it to a whole gigabyte.

    Notes:
        The tolerance is cindra's, which the video library carries at the same value while the communication library
        carries a wider one, so a mixed batch is weighed on close but not identical scales.

    Args:
        memory_mb: The modeled memory in megabytes, before any margin.

    Returns:
        The reportable memory in megabytes, carrying the tolerance and rounded up to a whole gigabyte.
    """
    return _round_to_gigabyte(memory_mb=int(memory_mb * MEMORY_ESTIMATE_TOLERANCE) + 1)


def _size_camera_extraction_job(archive_path: Path) -> JobFootprint:
    """Sizes one camera timestamp extraction job through the video library's own sizing pass.

    Notes:
        The library reads the archive once and answers both halves of its model from that read. It picks a single
        core for an archive below its parallel-extraction threshold and its declared allocation above it, then
        estimates the memory at the width it picked. Both figures are taken whole, so this package neither repeats the
        width rule nor reserves cores for a pool the stage would not open.

        Reading the archive reads its central directory alone, and the library refuses an archive it cannot read
        because the job reading it could not run either. That refusal is left to propagate, since a job with no
        readable input is a job to drop from the workflow rather than one to plan at a guessed figure.

    Args:
        archive_path: The path to the log archive the job reads.

    Returns:
        The job's footprint, holding the library's own width and its memory at that width.

    Raises:
        FileNotFoundError: If the archive cannot be read, in which case the job that reads it cannot run either.
    """
    sizing = size_camera_extraction_job(archive_path=archive_path)
    return JobFootprint(cores=sizing.cores, memory_mb=_round_to_gigabyte(memory_mb=sizing.memory_mb))


def _size_controller_extraction_job(archive_path: Path) -> JobFootprint:
    """Sizes one microcontroller data extraction job through the communication library's own sizing pass.

    Notes:
        The library reads the archive once and answers both halves of its model from that read. It picks a single
        core for an archive below its parallel-extraction threshold and its declared allocation above it, then
        estimates the memory at the width it picked. Both figures are taken whole, so this package neither repeats the
        width rule nor reserves cores for a pool the stage would not open.

        Reading the archive reads its central directory alone, and the library refuses an archive it cannot read
        because the job reading it could not run either. That refusal is left to propagate, since a job with no
        readable input is a job to drop from the workflow rather than one to plan at a guessed figure.

    Args:
        archive_path: The path to the log archive the job reads.

    Returns:
        The job's footprint, holding the library's own width and its memory at that width.

    Raises:
        FileNotFoundError: If the archive cannot be read, in which case the job that reads it cannot run either.
    """
    sizing = size_controller_extraction_job(archive_path=archive_path)
    return JobFootprint(cores=sizing.cores, memory_mb=_round_to_gigabyte(memory_mb=sizing.memory_mb))


def _size_runtime_job(archive_path: Path, cores: int) -> JobFootprint:
    """Sizes one runtime log job from the archive it reads.

    Notes:
        Every reader opens the archive itself, so each one holds the archive's directory. A single core decodes in the
        job's own process, which holds one directory and starts no child. A wider allocation opens one child per
        allocated core, and the job's own process keeps the reader that planned the batches, so the directory is held
        once more than the child count. The runtime pipeline is this package's own, so no dependency models it, and
        the pool it opens is the allocation its type declared.

    Args:
        archive_path: The path to the log archive the job reads, as the shared locating pass resolved it.
        cores: The cores the job is allocated, which is how many readers it opens.

    Returns:
        The job's footprint, holding the declared width and the memory the readers hold at it.

    Raises:
        FileNotFoundError: If the archive cannot be read, in which case the job that reads it cannot run either.
    """
    per_reader = _bytes_to_megabytes(
        byte_count=read_archive_message_count(archive_path=archive_path) * _ARCHIVE_DIRECTORY_BYTES_PER_MESSAGE
    )
    children = cores if cores > 1 else 0
    readers = children + 1
    return JobFootprint(
        cores=cores,
        memory_mb=_apply_tolerance(
            memory_mb=WORKER_MEMORY_MB + readers * per_reader + children * SPAWNED_CHILD_MEMORY_MB
        ),
    )


def _size_checksum_job(cores: int) -> JobFootprint:
    """Sizes one raw-data checksum job from the readers it opens.

    Notes:
        Each worker streams its file in fixed chunks and holds one at a time, so the figure is flat across every
        session size. The parent retains one pending result per file, which stays below the rounding this estimate
        already carries. The stage is this package's own and gains nothing from a width the data picks, so it runs at
        the allocation its type declared.

    Args:
        cores: The cores the job is allocated, which is how many files it hashes at once.

    Returns:
        The job's footprint, holding the declared width and the memory its readers hold at it.
    """
    return JobFootprint(
        cores=cores, memory_mb=_apply_tolerance(memory_mb=WORKER_MEMORY_MB + cores * _CHECKSUM_READER_MEMORY_MB)
    )


def _size_rename_job(cores: int) -> JobFootprint:
    """Sizes one camera timestamp rename job.

    Notes:
        The stage performs a fixed handful of filesystem operations and reads no recording, so it holds a worker and
        nothing besides. No input scales the estimate, so the worker itself is the whole model rather than a floor
        standing in for one.

    Args:
        cores: The cores the job is allocated, which its type declares.

    Returns:
        The job's footprint, holding the declared width and one worker's memory.
    """
    return JobFootprint(cores=cores, memory_mb=_apply_tolerance(memory_mb=WORKER_MEMORY_MB))


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


def _size_motion_energy_job(frame_pixels: int, cores: int) -> JobFootprint:
    """Sizes one video motion-energy job from the frame it decodes.

    Notes:
        Each decode worker holds the binned frame and its predecessor. Binning slices a strided view out of a
        full-resolution buffer, and a view retains its base, so both cost the whole frame. The frame buffers are a
        small share of a worker's memory next to the cost of the worker itself. Charging every job of a session its
        widest frame therefore stays on the safe side at a cost well inside the tolerance.

        The stage is this package's own and opens one decoder per core it holds, so it runs at the allocation its
        type declared and the per-core decoder and child cost is charged at that width.

    Args:
        frame_pixels: The pixels held by the frame this job decodes.
        cores: The cores the job is allocated, which bounds how many chunks it decodes at once.

    Returns:
        The job's footprint, holding the declared width and the memory its decoders hold at it.
    """
    per_worker = (
        _bytes_to_megabytes(byte_count=frame_pixels * _SINGLE_PRECISION_BYTES * _RETAINED_FRAME_BUFFERS)
        + _DECODER_BUFFER_MEMORY_MB
        + SPAWNED_CHILD_MEMORY_MB
    )
    return JobFootprint(cores=cores, memory_mb=_apply_tolerance(memory_mb=WORKER_MEMORY_MB + cores * per_worker))


def _size_two_photon_job(
    job_name: str,
    specifier: str,
    output_root: Path,
    configuration: SingleRecordingConfiguration,
    data_path: Path | None,
) -> JobFootprint:
    """Sizes one two-photon job through cindra's own per-stage sizing pass.

    Notes:
        cindra reads the recording once and answers both halves of its model from that read, so the width at which a
        stage runs is the measured knee of its own scaling curve rather than a figure this package repeats. Taking
        both figures whole is what keeps a retune of either half reaching slf without a change here.

        cindra rejects a stage it cannot size, either because the recording carries no readable raw imaging data or
        because a per-plane specifier names a plane the recording does not hold. Both refusals propagate, since a
        recording that cindra will not size is a recording whose stages cannot run.

    Args:
        job_name: The tracker job name identifying the stage.
        specifier: The plane specifier for a registration or processing job, empty for the binarization and
            combination stages.
        output_root: The output root the recording's cindra configuration was given.
        configuration: The recording's resolved processing configuration.
        data_path: The raw imaging directory, consulted when the recording carries no output yet.

    Returns:
        The job's footprint, holding cindra's own width for the stage and its memory at that width.

    Raises:
        FileNotFoundError: If the recording carries neither pipeline output nor readable raw imaging data, in which
            case no stage of it can run.
        ValueError: If the specifier names an imaging plane the recording does not hold, or if both inputs were
            readable and still describe no whole imaging plane.
    """
    sizing = size_single_recording_job(
        job_name=SingleRecordingJobNames(job_name),
        specifier=specifier,
        output_root=output_root,
        configuration=configuration,
        data_path=data_path,
    )
    return JobFootprint(cores=sizing.cores, memory_mb=_round_to_gigabyte(memory_mb=sizing.memory_mb))


def _size_multi_recording_job(
    job_name: MultiRecordingJobNames,
    specifier: str,
    recording_directories: tuple[Path, ...],
    configuration: MultiRecordingConfiguration | None,
) -> JobFootprint:
    """Sizes one cross-recording job through cindra's own per-stage sizing pass.

    Notes:
        Both cross-recording stages read every recording of the animal over which they run, so the whole recording set
        is handed to cindra whichever stage is being sized, and both halves of the figure come back from that one read.

        cindra refuses a set that any recording leaves short rather than sizing it from the recordings that happen to
        be complete, and that refusal propagates. A dataset whose recordings carry no combined output cannot run
        either stage yet, so it is dropped from the workflow rather than planned at a floor.

    Args:
        job_name: The cindra stage the job runs.
        specifier: The job's tracker specifier, which names a session for the extraction stage and an animal for the
            discovery stage.
        recording_directories: The cindra output directory of every recording the job spans.
        configuration: The dataset's resolved multi-recording configuration, or None when its acquisition system
            donates none.

    Returns:
        The job's footprint, holding cindra's own width for the stage and its memory at that width.

    Raises:
        FileNotFoundError: If the job spans no recording, if any recording it spans carries no combined metadata
            archive, or if any recording reports no regions in its combined trace array, in which case neither
            cross-recording stage can run.
        ValueError: If the dataset's acquisition system donates no multi-recording configuration, which leaves the
            stage without the parameters its sizing needs.
    """
    if configuration is None:
        message = (
            f"Unable to size the '{job_name.value}' job of '{specifier}'. The dataset resolved no multi-recording "
            f"configuration, either because its acquisition system tracks no regions across the sessions it holds "
            f"or because none of those sessions remains under the project root, so the stage carries neither sizing "
            f"parameters nor input data."
        )
        console.error(message=message, error=ValueError)
    sizing = size_multi_recording_job(
        job_name=job_name,
        specifier=specifier,
        recording_directories=recording_directories,
        configuration=configuration,
    )
    return JobFootprint(cores=sizing.cores, memory_mb=_round_to_gigabyte(memory_mb=sizing.memory_mb))


def _size_pose_tracking_job(session: SessionData, cores: int) -> JobFootprint:
    """Sizes one pose-tracking job from the prediction file its session carries.

    Notes:
        The stage reads predictions written upstream and never runs inference, so its working set follows the table it
        reads. The file is named by the system that produces it, so it is resolved through that system's donated
        locator. That locator answers with the file that the tracking worker opens. The stage is this package's own and
        its own fan-out is fixed, so it runs at the allocation its type declared.

    Args:
        session: The loaded session whose pose predictions the job reads.
        cores: The cores the job is allocated, which its type declares.

    Returns:
        The job's footprint, holding the declared width and the memory the prediction file implies.

    Raises:
        FileNotFoundError: If the session carries no pose prediction, in which case the job reading one cannot run
            either.
    """
    prediction = resolve_pose_prediction_locator(system=session.acquisition_system)(session=session)
    if prediction is None:
        message = (
            f"Unable to size the pose-tracking job of session '{session.session_name}'. The session carries no pose "
            f"prediction, so nothing states how much the job holds and the job could not run either."
        )
        console.error(message=message, error=FileNotFoundError)
    return JobFootprint(
        cores=cores,
        memory_mb=_apply_tolerance(
            memory_mb=WORKER_MEMORY_MB
            + _bytes_to_megabytes(byte_count=prediction.stat().st_size * _POSE_PREDICTION_RATIO)
        ),
    )


def _size_module_parse_job(archive_path: Path, cores: int) -> JobFootprint:
    """Sizes one module parse job from the log archive its own controller recorded.

    Notes:
        One module holds a share of its controller's archive, and the whole archive bounds that share from above, so
        the archive of the controller that recorded the module is what the job is charged. The stage is a single pass
        over one module's extracted table, so it runs at the allocation its type declared.

    Args:
        archive_path: The path to the log archive the job's controller recorded.
        cores: The cores the job is allocated, which its type declares.

    Returns:
        The job's footprint, holding the declared width and the memory its controller's archive implies.
    """
    return JobFootprint(
        cores=cores,
        memory_mb=_apply_tolerance(
            memory_mb=WORKER_MEMORY_MB
            + _bytes_to_megabytes(byte_count=archive_path.stat().st_size * _MODULE_TABLE_RATIO)
        ),
    )


@cache
def _two_photon_output_root(project_root: Path, animal: str, session: str) -> Path:
    """Resolves the output root given to a session's two-photon processing.

    Notes:
        This is the root under which cindra creates its own output directory, so every location beneath it is resolved
        through cindra's own resolvers rather than by rebuilding its layout here.

        Cached, because one dataset's estimates resolve the same session from several stages and each resolution
        otherwise re-reads that session's marker.

    Args:
        project_root: The path to the project's root directory.
        animal: The animal that owns the session.
        session: The session name whose output root is resolved.

    Returns:
        The path to the session's processed-data root, which is the output root its cindra jobs were given.
    """
    return SessionData.load(session_path=project_root.joinpath(animal, session)).processed_data_path


@cache
def _video_output_root(project_root: Path, animal: str, session: str) -> Path:
    """Resolves the directory into which a session's video processing wrote its per-camera timestamp feathers.

    Notes:
        The directory is read from the shared hierarchy's own accessor rather than rebuilt here, which is the same
        contract every other location this pass reads follows.

        Cached on the same terms as the two-photon output root, because a session's marker is otherwise re-read for
        every stage that resolves a location beneath it.

    Args:
        project_root: The path to the project's root directory.
        animal: The animal that owns the session.
        session: The session name whose processed video directory is resolved.

    Returns:
        The path to the session's processed video-data directory.
    """
    return SessionData.load(session_path=project_root.joinpath(animal, session)).processed_data.video_data_path


def _animal_recording_directories(dataset: DatasetData, animal: str, project_root: Path) -> tuple[Path, ...]:
    """Resolves the cindra output directory of every recording one animal contributes to a dataset.

    Notes:
        This is the recording set named by the animal's materialized multi-recording configuration, resolved from the
        project root rather than read back from that file, so a dataset whose configurations have not been written
        yet is still sizable. A session that has moved to long-term storage contributes no directory.

    Args:
        dataset: The resolved dataset that holds the animal.
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
            resolve_output_path(
                output_root=_two_photon_output_root(project_root=project_root, animal=animal, session=entry.session)
            )
        )
    return tuple(directories)


def _resolve_recording_geometry(project_root: Path, animal: str, session: str) -> _RecordingGeometry | None:
    """Reads a processed recording's shape from the arrays the single-recording pipeline wrote.

    Notes:
        cindra's own completion predicate gates the result alongside the trace array, because a recording that has
        not reached the end of the combination stage carries no output the forging stages can read. Neither file's
        contents are loaded, so the resolution reads one array header and one directory entry.

    Args:
        project_root: The path to the project's root directory.
        animal: The animal that owns the session.
        session: The session name whose processed output is read.

    Returns:
        The recording's geometry, or None when the session holds no processed imaging output.
    """
    output_root = _two_photon_output_root(project_root=project_root, animal=animal, session=session)
    if not is_recording_processed(output_root=output_root):
        return None
    traces = _read_array_shape(
        array_path=resolve_array_path(
            root_path=resolve_output_path(output_root=output_root), array=RecordingArrays.CELL_FLUORESCENCE
        )
    )
    if traces is None:
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
        Read from the system registry rather than from the file that ``define_forging_dataset`` materializes, so the
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
        dataset: The resolved dataset that holds the session.
        animal: The animal that owns the session.
        session: The session name whose tracked regions are resolved.
        project_root: The path to the project's root directory.
        configuration: The resolved multi-recording configuration, which reports the prevalence a cluster must reach.

    Returns:
        The tracked region count, or the bound standing in for it.
    """
    entries = dataset.get_sessions_for_animal(animal=animal)
    geometries = [
        geometry
        # A session that has moved to long-term storage cannot be loaded at all, so it is skipped before its geometry
        # is read rather than failing the whole animal's estimate. A dataset outlives the source data of the animals it
        # has already forged, and the recording set from which this bound is drawn skips a relocated session on the
        # same terms, so reading one here would make an estimate depend on data the dataset no longer needs.
        for entry in entries
        if project_root.joinpath(animal, entry.session).is_dir()
        and (geometry := _resolve_recording_geometry(project_root=project_root, animal=animal, session=entry.session))
        is not None
    ]
    if not geometries:
        return 1

    tracked = _read_array_shape(
        array_path=resolve_array_path(
            root_path=resolve_dataset_path(
                output_root=_two_photon_output_root(project_root=project_root, animal=animal, session=session),
                dataset_name=multi_recording_dataset_name(animal_id=animal, dataset_name=dataset.name),
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


def _size_forging_job(
    dataset: DatasetData,
    animal: str,
    session: str,
    project_root: Path,
    configuration: MultiRecordingConfiguration | None,
    cores: int,
) -> JobFootprint:
    """Sizes one per-session assembly job from the processed output it reads.

    Notes:
        The assembled frame retains every fluorescence column it attaches. The write that closes the job streams the
        frame it was handed rather than rebuilding it, so the shape of the session's own fluorescence is what the job
        is charged. The columns fall into two groups, each spanning its own region count. A single-day column spans
        every region the recording detected, while a multi-day column spans only the regions tracked across the animal.
        The stage is this package's own, and its fan-out is a fixed handful of threads, so it runs at the allocation its
        type declared.

        A session carrying no fluorescence has nothing to assemble, so its refusal propagates rather than resolving
        to a floor. That holds for a session whose type must complete the two-photon pipeline before it joins a
        dataset at all. A type admitted without that pipeline records no imaging by design, and its assembly attaches
        no fluorescence column, so this model does not describe its job and the behavior-only model sizes it instead.

    Args:
        dataset: The resolved dataset that holds the session.
        animal: The animal that owns the session.
        session: The session name whose assembly job is sized.
        project_root: The path to the project's root directory.
        configuration: The dataset's resolved multi-recording configuration, or None when its acquisition system
            donates none.
        cores: The cores the job is allocated, which its type declares.

    Returns:
        The job's footprint, holding the declared width and the memory the assembled frame holds at it.

    Raises:
        FileNotFoundError: If a session whose type requires imaging carries no processed imaging output, or if a
            session whose type requires none carries no camera clock, in which case the job assembling it cannot run
            either.
        ValueError: If the dataset's acquisition system is unknown, if its session type falls outside the platform
            vocabulary, or if that type joins no dataset for the system and therefore matches neither model.
    """
    # The admission policy states which pipelines a session type must complete before it joins a dataset, so a type
    # that joins without the two-photon pipeline is one the assembler builds from behavior sources alone. Sizing it
    # from fluorescence would charge columns its job never attaches, so the model is selected by the policy rather
    # than falling back to it when imaging happens to be absent.
    if not _requires_imaging(dataset=dataset):
        return _size_behavior_assembly_job(project_root=project_root, animal=animal, session=session, cores=cores)

    geometry = _resolve_recording_geometry(project_root=project_root, animal=animal, session=session)
    if geometry is None:
        message = (
            f"Unable to size the assembly job of session '{session}'. The session carries no processed imaging "
            f"output, so nothing states the shape of the frame the job assembles and the job could not run either."
        )
        console.error(message=message, error=FileNotFoundError)
    regions = _resolve_tracked_regions(
        dataset=dataset, animal=animal, session=session, project_root=project_root, configuration=configuration
    )

    retained_regions = _ASSEMBLY_SINGLE_DAY_COLUMNS * geometry.regions + _ASSEMBLY_MULTI_DAY_COLUMNS * regions
    columns = retained_regions * _ASSEMBLY_WRITE_COPIES * geometry.samples * _SINGLE_PRECISION_BYTES
    sub_datasets = geometry.samples * _SUB_DATASET_BYTES_PER_SAMPLE
    return JobFootprint(
        cores=cores,
        memory_mb=_apply_tolerance(memory_mb=WORKER_MEMORY_MB + _bytes_to_megabytes(byte_count=columns + sub_datasets)),
    )


def _requires_imaging(dataset: DatasetData) -> bool:
    """Reports whether the sessions a dataset holds must carry two-photon output to have joined it.

    Notes:
        Read from the acquisition system's own admission policy rather than from a session-type literal, so a system
        that admits a behavior-only session type inherits the behavior-only estimate with no change here. A dataset
        holds one session type, which the shared hierarchy enforces both when a dataset is defined and when a session
        is added to one, so the policy is read once for the dataset rather than once per session.

        A session type the policy omits joins no dataset at all, so neither model describes the assembly of a
        dataset carrying it. Such a type is refused here rather than read as requiring no imaging, because this pass
        is public and does not itself run the admission check: a hand-made dataset marker naming an unadmitted type
        reaches it directly, and answering it with a behavior-only figure would reserve memory for a job the forging
        pipeline never plans.

    Args:
        dataset: The resolved dataset whose session type is examined.

    Returns:
        True when the dataset's session type must complete the two-photon pipeline before joining a dataset, and
        False when it joins carrying no imaging at all.

    Raises:
        ValueError: If the dataset's acquisition system is unknown, if its session type falls outside the platform
            vocabulary, or if that type joins no dataset for the system.
    """
    requirements = resolve_forging_admission_pipelines(system=dataset.acquisition_system)

    # An absent entry states that the type joins no dataset, which is a different answer from a type admitted while
    # requiring no imaging. Collapsing the two would size a dataset the pipeline refuses to forge, so the absence is
    # reported on the same terms the admission check reports it.
    required = requirements.get(SessionTypes(dataset.session_type))
    if required is None:
        admissible = ", ".join(sorted(str(session_type) for session_type in requirements))
        message = (
            f"Unable to size the assembly job of dataset '{dataset.name}'. Its session type "
            f"'{dataset.session_type}' joins no dataset for the '{dataset.acquisition_system}' acquisition system, "
            f"which admits the session type(s): {admissible}, so no model describes the job assembling it."
        )
        console.error(message=message, error=ValueError)

    return ProcessingPipelines.TWO_PHOTON in required


def _resolve_widest_camera_clock_samples(video_data_path: Path) -> int:
    """Reads how many samples the widest camera clock a session recorded holds.

    Notes:
        Every camera the video pipeline processed writes one timestamp per frame it acquired, under the filename that
        library's own layout states, so a feather's row count is that camera's sample count and the filenames are
        read from the layout rather than rebuilt here. The pipeline publishes each parsed feather under its canonical
        name as well, and both names carry the same rows, so counting one twice moves no figure.

        The pattern matches every camera the pipeline published a feather for, including one whose name a given
        acquisition system's own assembler does not read. Counting those keeps this module free of any system's
        camera manifest and keeps the figure an upper bound, which is the direction a reservation may err in.

        Only each feather's metadata is read, which states the rows it holds without decoding one of them.

    Args:
        video_data_path: The processed video-data directory holding the session's per-camera timestamp feathers.

    Returns:
        The samples the widest camera clock holds, or zero when the session carries no timestamp feather at all.
    """
    return max(
        (
            int(pl.scan_ipc(source=timestamps_path).select(pl.len()).collect().item())
            for timestamps_path in video_data_path.glob(f"*{OutputLayout.TIMESTAMPS_INFIX}{OutputLayout.FILE_SUFFIX}")
        ),
        default=0,
    )


def _size_behavior_assembly_job(project_root: Path, animal: str, session: str, cores: int) -> JobFootprint:
    """Sizes one per-session assembly job of a session type that records no imaging.

    Notes:
        The assembly of such a session attaches no fluorescence column at all. It places its behavior, runtime and
        video columns on the clock of the slowest camera the session recorded, so the samples that clock holds are the
        height of the frame the job builds and the whole data-dependent charge follows from them. The clock is settled
        after the columns are stacked and before they are clipped to the session bounds, so the job peaks at the full
        recorded height rather than at the height it writes.

        Which camera is the slowest follows from the mean rate each one held, which its timestamps alone state. The
        cameras of one session run over the same span, so the widest clock holds at least the samples the slowest one
        does and bounds the frame from above at the cost of a metadata read. Charging that bound keeps the estimate on
        the safe side while leaving the timestamps themselves unread.

        A session whose cameras left no usable clock has nothing to place its columns on, so its refusal propagates
        rather than resolving to a floor.

        The clock read here is a deliberate upper bound taken from the video pipeline's output contract rather than a
        replay of the assembler's own clock resolution, and the two do not agree in every case. This pass counts the
        rows of every timestamp feather that pipeline published and accepts any feather holding the samples a mean
        rate needs. The assembler a system donates is stricter: it reads the cameras its own manifest names and
        accepts a clock only where its timestamps span a positive duration. So a session whose only feather comes
        from a camera outside that manifest, or whose timestamps span no duration, is sized here and refused there.
        The residual surfaces as a job that was planned and then failed on its own missing clock rather than as a
        planning refusal, which is the direction an estimate may err in, since a figure this pass reports only
        reserves memory.

    Args:
        project_root: The path to the project's root directory.
        animal: The animal that owns the session.
        session: The session name whose assembly job is sized.
        cores: The cores the job is allocated, which its type declares.

    Returns:
        The job's footprint, holding the declared width and the memory the assembled frame holds at it.

    Raises:
        FileNotFoundError: If the session carries no camera timestamp feather holding the samples a clock needs, in
            which case nothing states the height of the frame the job builds.
    """
    samples = _resolve_widest_camera_clock_samples(
        video_data_path=_video_output_root(project_root=project_root, animal=animal, session=session)
    )
    if samples < _MINIMUM_CLOCK_SAMPLES:
        message = (
            f"Unable to size the assembly job of session '{session}'. The session carries no camera timestamp feather "
            f"holding at least {_MINIMUM_CLOCK_SAMPLES} frames, so nothing states the reference clock the job places "
            f"its columns on and the job could not run either."
        )
        console.error(message=message, error=FileNotFoundError)

    return JobFootprint(
        cores=cores,
        memory_mb=_apply_tolerance(
            memory_mb=WORKER_MEMORY_MB + _bytes_to_megabytes(byte_count=samples * _SUB_DATASET_BYTES_PER_SAMPLE)
        ),
    )
