"""Provides the memory estimators that size each job's working set from the raw acquisition data it will process."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from dataclasses import dataclass

import cv2
from cindra import SingleRecordingJobNames, SingleRecordingConfiguration
import psutil
from natsort import natsorted
from tifffile import TiffFile
from cindra.io import TIFF_EXTENSIONS, PARAMETERS_FILENAME

from ..video import ENERGY_JOB_NAME, TRACKING_JOB_NAME, TIMESTAMP_JOB_NAME
from ..runtime import RUNTIME_JOB_NAME
from ..managing import CHECKSUM_JOB_NAME
from .pipelines import ProcessingPipelines
from ..registries import resolve_two_photon_data_locator, resolve_single_recording_configuration_resolver
from ..shared_assets import LOG_ARCHIVE_SUFFIX
from ..microcontrollers import PARSE_JOB_NAME, EXTRACTION_JOB_NAME

if TYPE_CHECKING:
    from pathlib import Path

    from sollertia_shared_assets import SessionData

_MEMORY_ESTIMATE_TOLERANCE: float = 1.15
"""The margin applied to every estimate before it is reported. It covers the working sets a model does not enumerate
and the variation between recordings of the same shape. The penalty for understating is asymmetric, since a local
batch overcommits its host and a scheduled job is killed outright, so estimates round up."""

_WORKER_MEMORY_MB: int = 384
"""The resident memory a worker occupies before it runs any job, covering the interpreter and the package's import
graph. This term is charged once per job, while the per-child allowance is charged for every core the job holds."""

_SUBPROCESS_MEMORY_MB: int = 200
"""The resident memory each child of a job's own worker pool occupies before it touches data. Processes are spawned
rather than forked, so every child re-imports the module its target function lives in."""

_BYTES_PER_MEGABYTE: int = 1024 * 1024
"""The divisor converting a byte count into megabytes."""

_SINGLE_PRECISION_BYTES: int = 4
"""The width of one single-precision element, the type every modeled working array holds."""

_RAW_SAMPLE_BYTES: int = 2
"""The width of one raw imaging sample as it is read from and written to the converted binary."""

_ARCHIVE_DIRECTORY_RATIO: float = 2.6
"""The resident memory a log-archive reader holds per byte of archive on disk. Reading an archive builds one
directory entry per logged message, which dominates the decoded payload itself."""

_MODULE_TABLE_RATIO: float = 3.4
"""The resident memory a module parse job holds per byte of the archive its module came from. One module holds a
share of its controller's archive, and partitioning and merging its event streams materializes that share several
times over, so the whole archive bounds the job from above."""

_POSE_PREDICTION_RATIO: float = 6.0
"""The resident memory the pose-tracking job holds per byte of its prediction file. The table is promoted to double
precision and expanded into per-bodypart and per-metric arrays."""

_DECODER_BUFFER_MEMORY_MB: int = 32
"""The resident memory one decode worker holds for its codec reference frames, beyond the frame buffers the
measurement itself retains."""

_RETAINED_FRAME_BUFFERS: int = 2
"""The number of full-resolution single-precision frame buffers a decode worker keeps live. Binning produces a
strided view that retains its full-frame base, and the current and previous binned frames are live at once."""

_DETECTION_ARRAY_MULTIPLIER: int = 3
"""The number of copies of the binned movie live at the two-photon detection peak. Computing the temporal standard
deviation holds the frames, their difference, and the squared difference at once."""

_BINARIZATION_BATCH_COPIES: int = 2
"""The number of copies of a raw frame batch binarization holds. Indexing each plane out of the batch allocates a
second full batch alongside the one that was read."""

_CHECKSUM_READER_MEMORY_MB: int = 56
"""The resident memory one checksum worker holds, covering its fixed read chunk and the small module its target
function lives in. It replaces the general per-child allowance for this stage, because a checksum worker re-imports
only the hashing module rather than this package's import graph. Measured at 49 MB per worker across a sweep from
one to sixty-four workers, then rounded up."""

_COMBINATION_MEMORY_MB: int = 16384
"""The memory the combination job is charged. The stage concatenates every plane's traces into dense arrays, so its
size follows the region count that only detection resolves, which no reading of the raw data predicts. It therefore
takes a flat allowance comparable to one plane job rather than a projection. The allowance covers the trace volume a
recording of any attainable length produces, so the estimate tolerance does not apply on top."""


@dataclass(frozen=True, slots=True)
class _RawImagingGeometry:
    """Describes the shape of a two-photon recording as its raw acquisition data reports it.

    Notes:
        Every field is read from the raw imaging directory, so the geometry is available before any stage of the
        pipeline has run. The heights follow each region's own line span, matching how the conversion stage slices a
        multi-region frame into planes.
    """

    frame_count: int
    """The samples each plane holds."""
    sampling_rate: float
    """The rate at which the recording sampled each plane."""
    raw_frame_pixels: int
    """The pixels one unsliced acquisition frame holds, which the conversion stage reads a batch of at a time."""
    plane_extents: tuple[tuple[int, int], ...]
    """The height and width of every plane, ordered by plane index."""


def resolve_host_memory_mb() -> int:
    """Reads the host's total physical memory.

    Returns:
        The host's total physical memory in megabytes.
    """
    return int(psutil.virtual_memory().total / _BYTES_PER_MEGABYTE)


def estimate_session_job_memory(
    pipeline: ProcessingPipelines, session: SessionData, jobs: list[tuple[str, str, int]]
) -> dict[tuple[str, str], tuple[int, bool]]:
    """Estimates the memory every runnable job of one session occupies at its allocated core count.

    Notes:
        Reads on-disk metadata alone, so estimating a session never decodes a frame or opens a log archive. Every
        estimate scales with the input it describes, which keeps it correct on recordings longer, wider, or denser
        than any previously seen.

        Every term is read from the session's raw acquisition data, so an estimate is available before any stage has
        run. Estimates cover anonymous memory, the term that forces a host to swap and a scheduler to kill a job, so
        the reclaimable pages a memory-mapped stage leaves resident are excluded. Each estimate carries a flag
        stating whether the input it scales with was found. A job whose input is absent falls back to the worker
        baseline, which the flag marks as a floor to plan around.

    Args:
        pipeline: The pipeline the jobs belong to.
        session: The loaded session the jobs operate on.
        jobs: The runnable jobs as ``(job_name, specifier, cores)`` triples.

    Returns:
        A dictionary mapping each ``(job_name, specifier)`` pair to its estimated memory in megabytes and a flag that
        is True when the estimate follows from the job's own input rather than from the worker baseline alone.
    """
    behavior_directory = session.raw_data.behavior_data_path
    geometry: _RawImagingGeometry | None = None
    configuration: SingleRecordingConfiguration | None = None
    widest_frame_pixels = 0
    if pipeline is ProcessingPipelines.TWO_PHOTON:
        resolve_configuration = resolve_single_recording_configuration_resolver(system=session.acquisition_system)
        configuration = resolve_configuration(session)
        geometry = _resolve_raw_imaging_geometry(session=session, configuration=configuration)
    elif pipeline is ProcessingPipelines.VIDEO:
        widest_frame_pixels = _resolve_widest_camera_frame_pixels(camera_directory=session.raw_data.camera_data_path)

    estimates: dict[tuple[str, str], tuple[int, bool]] = {}
    for job_name, specifier, cores in jobs:
        if pipeline is ProcessingPipelines.TWO_PHOTON:
            modeled = (
                _estimate_two_photon_memory(
                    job_name=job_name, specifier=specifier, geometry=geometry, configuration=configuration
                )
                if geometry is not None and configuration is not None
                else _apply_tolerance(memory_mb=_WORKER_MEMORY_MB)
            )
        elif job_name == CHECKSUM_JOB_NAME:
            modeled = _estimate_checksum_memory(cores=cores)
        elif job_name in {RUNTIME_JOB_NAME, EXTRACTION_JOB_NAME, TIMESTAMP_JOB_NAME}:
            archive = behavior_directory.joinpath(f"{specifier}{LOG_ARCHIVE_SUFFIX}")
            modeled = _estimate_archive_reader_memory(archive_path=archive, cores=cores)
        elif job_name == ENERGY_JOB_NAME:
            modeled = _estimate_motion_energy_memory(frame_pixels=widest_frame_pixels, cores=cores)
        elif job_name == TRACKING_JOB_NAME:
            modeled = _estimate_widest_file_memory(
                directory=session.raw_data.camera_data_path,
                pattern="*.h5",
                expansion_ratio=_POSE_PREDICTION_RATIO,
            )
        elif job_name == PARSE_JOB_NAME:
            modeled = _estimate_widest_file_memory(
                directory=behavior_directory, pattern=f"*{LOG_ARCHIVE_SUFFIX}", expansion_ratio=_MODULE_TABLE_RATIO
            )
        else:
            modeled = _apply_tolerance(memory_mb=_WORKER_MEMORY_MB)
        estimates[job_name, specifier] = (modeled, modeled > _apply_tolerance(memory_mb=_WORKER_MEMORY_MB))

    return estimates


def _resolve_raw_imaging_geometry(
    session: SessionData, configuration: SingleRecordingConfiguration
) -> _RawImagingGeometry | None:
    """Reads a two-photon recording's shape from its raw acquisition data.

    Notes:
        Reads the acquisition parameters and the headers of the first and last image files, so the cost stays flat
        as a recording grows and no pixel data is decoded. The sample count follows from the pages those two files
        report, since a recording's files hold an equal share apart from the last. Region line spans give each
        plane's height and the image header gives the shared width, which is how the conversion stage sizes a plane.

        The image set and its order match the ones the conversion stage builds. Both skip the names the
        configuration excludes and order numbered files naturally, so both agree on which file holds the recording's
        final samples.

    Args:
        session: The loaded session whose raw imaging directory is read.
        configuration: The recording's resolved processing configuration, consulted for the image names the
            conversion stage excludes.

    Returns:
        The recording's geometry, or None when the session holds no readable raw imaging data.
    """
    locate_two_photon_data = resolve_two_photon_data_locator(system=session.acquisition_system)
    data_path = locate_two_photon_data(session)
    if not data_path.is_dir():
        return None

    parameter_files = natsorted(data_path.rglob(PARAMETERS_FILENAME))
    if not parameter_files:
        return None
    parameters = json.loads(parameter_files[0].read_text())

    # Collects into a set first, since a case-insensitive filesystem returns one file under several of the extension
    # spellings and counting it twice would inflate the sample count.
    ignored_names = tuple(configuration.file_io.ignored_file_names)
    discovered = {
        path.resolve()
        for extension in TIFF_EXTENSIONS
        for path in data_path.glob(f"*.{extension}")
        if path.stem not in ignored_names
    }
    images = natsorted(discovered)
    if not images:
        return None

    with TiffFile(images[0]) as first_image:
        leading_pages = len(first_image.pages)
        frame_shape = first_image.pages[0].shape
    with TiffFile(images[-1]) as final_image:
        trailing_pages = len(final_image.pages)

    volumes = max(1, int(parameters.get("plane_number", 1)) * int(parameters.get("channel_number", 1)))
    total_pages = (len(images) - 1) * leading_pages + trailing_pages
    base_height, base_width = int(frame_shape[-2]), int(frame_shape[-1])

    region_lines = parameters.get("roi_lines") or []
    extents = tuple((int(lines[-1]) - int(lines[0]) + 1, base_width) for lines in region_lines if lines) or (
        (base_height, base_width),
    )

    return _RawImagingGeometry(
        frame_count=max(1, total_pages // volumes),
        sampling_rate=float(parameters.get("frame_rate", 1.0)),
        raw_frame_pixels=base_height * base_width,
        plane_extents=extents,
    )


def _bytes_to_megabytes(byte_count: float) -> int:
    """Converts a byte count into whole megabytes, rounding up so an estimate never understates its demand.

    Args:
        byte_count: The number of bytes to convert.

    Returns:
        The equivalent size in megabytes.
    """
    return max(0, int(byte_count / _BYTES_PER_MEGABYTE) + 1) if byte_count > 0 else 0


def _apply_tolerance(memory_mb: int) -> int:
    """Applies the shared estimate tolerance to a modeled memory figure.

    Args:
        memory_mb: The modeled memory in megabytes, before any margin.

    Returns:
        The reportable memory in megabytes, rounded up.
    """
    return int(memory_mb * _MEMORY_ESTIMATE_TOLERANCE) + 1


def _estimate_archive_reader_memory(archive_path: Path, cores: int) -> int:
    """Estimates the memory a runtime, microcontroller-extraction, or camera-timestamp job holds.

    Notes:
        Each of those stages splits one log archive across a worker pool, and every worker opens the archive itself,
        so the archive's directory is held once per allocated core.

    Args:
        archive_path: The path to the log archive the job reads.
        cores: The cores the job is allocated, which is how many readers it opens.

    Returns:
        The reportable memory in megabytes.
    """
    if not archive_path.is_file():
        return _apply_tolerance(memory_mb=_WORKER_MEMORY_MB)
    per_reader = _bytes_to_megabytes(byte_count=archive_path.stat().st_size * _ARCHIVE_DIRECTORY_RATIO)
    return _apply_tolerance(memory_mb=_WORKER_MEMORY_MB + cores * (per_reader + _SUBPROCESS_MEMORY_MB))


def _estimate_checksum_memory(cores: int) -> int:
    """Estimates the memory one raw-data checksum job holds.

    Notes:
        The only estimator here that does not scale with the size of its input. Each worker streams its file in
        fixed chunks and holds one at a time, so a session of a few megabytes and one of eighty gigabytes cost the
        same. The parent retains one pending result per file, which the largest session in this corpus keeps under a
        megabyte, so it stays below the rounding this estimate already carries.

    Args:
        cores: The cores the job is allocated, which is how many files it hashes at once.

    Returns:
        The reportable memory in megabytes.
    """
    return _apply_tolerance(memory_mb=_WORKER_MEMORY_MB + cores * _CHECKSUM_READER_MEMORY_MB)


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


def _estimate_motion_energy_memory(frame_pixels: int, cores: int) -> int:
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
        The reportable memory in megabytes.
    """
    per_worker = (
        _bytes_to_megabytes(byte_count=frame_pixels * _SINGLE_PRECISION_BYTES * _RETAINED_FRAME_BUFFERS)
        + _DECODER_BUFFER_MEMORY_MB
        + _SUBPROCESS_MEMORY_MB
    )
    return _apply_tolerance(memory_mb=_WORKER_MEMORY_MB + cores * per_worker)


def _estimate_binarization_memory(geometry: _RawImagingGeometry, configuration: SingleRecordingConfiguration) -> int:
    """Estimates the memory the two-photon conversion stage holds, from the raw frame it reads a batch of at a time.

    Args:
        geometry: The recording's raw geometry.
        configuration: The recording's resolved processing configuration.

    Returns:
        The reportable memory in megabytes.
    """
    batch_bytes = (
        configuration.registration.batch_size
        * geometry.raw_frame_pixels
        * _RAW_SAMPLE_BYTES
        * _BINARIZATION_BATCH_COPIES
    )
    return _apply_tolerance(memory_mb=_WORKER_MEMORY_MB + _bytes_to_megabytes(byte_count=batch_bytes))


def _estimate_plane_processing_memory(
    extent: tuple[int, int], geometry: _RawImagingGeometry, configuration: SingleRecordingConfiguration
) -> int:
    """Estimates the memory one two-photon plane-processing job holds, from that plane's shape and sample count.

    Notes:
        Detection rather than registration sets the peak. The movie is binned down to at most the configured binned
        sample count, and the temporal standard deviation then holds the binned frames, their difference, and the
        squared difference at once. Registration is bounded by its own batch and stays below this peak. Registration
        later narrows a plane to the region that stayed in frame, so the raw extent used here is the wider bound.

    Args:
        extent: The plane's height and width in pixels.
        geometry: The recording's raw geometry.
        configuration: The recording's resolved processing configuration.

    Returns:
        The reportable memory in megabytes.
    """
    height, width = extent
    detection = configuration.roi_detection

    # Mirrors cindra's own bin sizing, which takes the coarsest of a single sample, the ratio that caps the binned
    # sample count, and the transient decay window.
    decay_samples = round(configuration.main.tau * geometry.sampling_rate)
    bin_size = max(1, geometry.frame_count // max(1, detection.maximum_binned_frames), decay_samples)
    binned_samples = max(1, geometry.frame_count // bin_size)

    peak_bytes = _DETECTION_ARRAY_MULTIPLIER * binned_samples * height * width * _SINGLE_PRECISION_BYTES
    return _apply_tolerance(memory_mb=_WORKER_MEMORY_MB + _bytes_to_megabytes(byte_count=peak_bytes))


def _estimate_two_photon_memory(
    job_name: str,
    specifier: str,
    geometry: _RawImagingGeometry,
    configuration: SingleRecordingConfiguration,
) -> int:
    """Routes a two-photon job to the estimator matching its stage.

    Args:
        job_name: The tracker job name identifying the stage.
        specifier: The plane specifier for a processing job, empty for the other stages.
        geometry: The recording's raw geometry.
        configuration: The recording's resolved processing configuration.

    Returns:
        The reportable memory in megabytes.
    """
    if job_name == str(SingleRecordingJobNames.BINARIZE):
        return _estimate_binarization_memory(geometry=geometry, configuration=configuration)
    if job_name == str(SingleRecordingJobNames.COMBINE):
        return _COMBINATION_MEMORY_MB

    estimates = [
        _estimate_plane_processing_memory(extent=extent, geometry=geometry, configuration=configuration)
        for extent in geometry.plane_extents
    ]
    plane_index = _resolve_plane_index(specifier=specifier)
    if plane_index is not None and 0 <= plane_index < len(estimates):
        return estimates[plane_index]
    # Charges the largest per-plane estimate when the specifier does not resolve, so an unmatched job never
    # underestimates.
    return max(estimates)


def _resolve_plane_index(specifier: str) -> int | None:
    """Reads the plane index a processing job's specifier carries.

    Args:
        specifier: The job specifier, which names a plane by its index.

    Returns:
        The plane index, or None when the specifier does not name one.
    """
    digits = specifier.rsplit("_", maxsplit=1)[-1]
    return int(digits) if digits.isdigit() else None


def _estimate_widest_file_memory(directory: Path, pattern: str, expansion_ratio: float) -> int:
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
        The reportable memory in megabytes.
    """
    if not directory.is_dir():
        return _apply_tolerance(memory_mb=_WORKER_MEMORY_MB)
    candidates = sorted(directory.glob(pattern), key=lambda path: path.stat().st_size, reverse=True)
    if not candidates:
        return _apply_tolerance(memory_mb=_WORKER_MEMORY_MB)
    widest = candidates[0]
    return _apply_tolerance(
        memory_mb=_WORKER_MEMORY_MB + _bytes_to_megabytes(byte_count=widest.stat().st_size * expansion_ratio)
    )
