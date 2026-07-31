"""Provides the memory estimators that size each job's working set from the raw acquisition data it will process."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING
from functools import cache
from dataclasses import dataclass

import cv2
import numpy as np
from cindra import SingleRecordingJobNames
import psutil
from natsort import natsorted
from tifffile import TiffFile
from cindra.io import TIFF_EXTENSIONS, PARAMETERS_FILENAME
from numpy.lib.format import read_magic, read_array_header_1_0, read_array_header_2_0
from cindra.allocation import PLANE_SPECIFIER_PREFIX
from sollertia_shared_assets import SessionData

from ..video import ENERGY_JOB_NAME, TRACKING_JOB_NAME, TIMESTAMP_JOB_NAME
from ..forging import MULTIDAY_DISCOVERY_JOB_NAME, MULTIDAY_EXTRACTION_JOB_NAME
from ..runtime import RUNTIME_JOB_NAME
from ..managing import CHECKSUM_JOB_NAME
from ..registries import (
    resolve_two_photon_data_locator,
    resolve_multi_recording_configuration_resolver,
    resolve_single_recording_configuration_resolver,
)
from ..shared_assets import (
    LOG_ARCHIVE_SUFFIX,
    ProcessingPipelines,
    multi_recording_dataset_directory,
)
from ..microcontrollers import PARSE_JOB_NAME, EXTRACTION_JOB_NAME

if TYPE_CHECKING:
    from pathlib import Path

    from cindra import MultiRecordingConfiguration, SingleRecordingConfiguration
    from sollertia_shared_assets import DatasetData

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

_MEGABYTES_PER_GIGABYTE: int = 1024
"""The megabytes one gigabyte holds, which every reportable estimate is rounded up to a multiple of."""

_BYTES_PER_MEGABYTE: int = 1024 * 1024
"""The divisor converting a byte count into megabytes."""

_SINGLE_PRECISION_BYTES: int = 4
"""The width of one single-precision element, the type every modeled working array holds."""

_RAW_SAMPLE_BYTES: int = 2
"""The width of one raw imaging sample as it is read from and written to the converted binary."""

_ARCHIVE_DIRECTORY_RATIO: float = 4.0
"""The resident memory a log-archive reader holds per byte of archive on disk. Reading an archive builds one
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

_DETECTION_ARRAY_MULTIPLIER: int = 4
"""The number of copies of the binned movie live at the two-photon detection peak. Computing the temporal standard
deviation holds the frames, their difference, and the squared difference at once, alongside the working copy the
reduction accumulates into."""

_BINARIZATION_BATCH_COPIES: int = 2
"""The number of copies of a raw frame batch binarization holds. Indexing each plane out of the batch allocates a
second full batch alongside the one that was read."""

_CHECKSUM_READER_MEMORY_MB: int = 285
"""The resident memory one checksum worker holds, covering its fixed read chunk and the interpreter and import graph
the worker starts with. The figure holds steady across sessions of any size, because a worker streams its file in
fixed chunks and never holds more than one at a time."""

_FLUORESCENCE_FILENAME: str = "cell_fluorescence.npy"
"""The cindra array whose header reports a recording's region and sample counts. Only the header is parsed, so a
recording of any length costs one small read and no part of it is mapped."""

_COMBINED_METADATA_FILENAME: str = "combined_metadata.npz"
"""The cindra archive reporting the combined field extent every multi-day stage works at."""

_MULTI_RECORDING_DIRECTORY: str = "multi_recording"
"""The processed-output subdirectory holding an animal's multi-day results, one directory per tracked dataset."""

_TRACE_ARRAY_DIMENSIONS: int = 2
"""The axes a cindra trace array carries, which are its regions and its samples."""

_DISCOVERY_PLANES_PER_RECORDING: int = 12
"""The single-precision planes a discovery job holds per recording beyond its pairwise cache. The planes cover that
recording's accumulated and cached deformation fields, its scale-space pyramid, its transformed reference images, and
the per-thread warp transients live alongside them."""

_DISCOVERY_CLUSTERING_MEMORY_MB: int = 2048
"""The memory the cross-recording clustering stage is charged. The stage builds a pairwise matrix over the regions
falling inside one spatial bin, so its size follows local region crowding, which no reading of the processed data
predicts. The allowance covers the crowding this corpus produces."""

_EXTRACTION_TRACE_COPIES: int = 4
"""The copies of a recording's traces the extraction stage retains, which are the cell, neuropil, subtracted, and
spike arrays it returns together. The stages that derive the later three release their working arrays, so the
retained set rather than any transient peak sizes this term."""

_EXTRACTION_BATCH_BYTES_PER_PIXEL: int = 6
"""The memory one extraction batch holds per combined pixel, covering the batch at its stored width and the
single-precision copy the kernel consumes."""

_EXTRACTION_BATCH_RETENTION: int = 20
"""The batch working sets an extraction job holds at its peak. The stage reads its recording in batches and releases
each one, but the allocator returns little of that memory between iterations, so the peak settles far above the
working set of any single batch. The retained multiple varies between runs of identical work, so this covers the
widest settling point rather than a typical one."""

_ASSEMBLY_FLUORESCENCE_COLUMNS: int = 8
"""The fluorescence columns an experiment assembly retains at once. Every column is attached under its own name and
none replaces another, so each stays live in the assembled frame for the rest of the job."""

_ASSEMBLY_WRITE_COPIES: int = 3
"""The copies of the assembled fluorescence volume charged at the write. Writing rechunks a frame the earlier stages
left fragmented, which materializes the whole frame a second time beside the one already resident, and the allocator
holds a further share of what the column builds released."""

_SUB_DATASET_BYTES_PER_SAMPLE: int = 512
"""The memory the behavior, runtime, and video sub-datasets hold per sample of the clock they are placed on. Each
emits one array per column and the interpolation that aligns them holds double-precision transients."""

_PERCENT_PER_FRACTION: float = 100.0
"""The divisor converting a percentage into a fraction."""

_COMBINATION_MEMORY_MB: int = 8192
"""The memory the combination job is charged. The stage concatenates every plane's traces into dense arrays, so its
size follows the region count that only detection resolves, which no reading of the raw data predicts. It therefore
takes a flat allowance rather than a projection, covering the trace volume a recording of any attainable length
produces."""

_REGISTRATION_MEMORY_MB: int = 14848
"""The memory one plane-registration job is charged.

Notes:
    Charged flat rather than projected from the plane. The stage streams the plane in batches whose size its
    configuration fixes, yet its footprint neither follows that batch nor the plane's own extent, and across planes
    it moves opposite to the extent rather than with it. What the footprint does follow has not been established, so
    a projection built on any of those terms would be a shape the stage does not have.

    The allowance therefore covers the widest footprint the stage has been observed to reach. Narrowing it wants the
    driver identified first, since a plane far outside the observed shapes is the case a flat allowance cannot
    reason about."""


@dataclass(frozen=True, slots=True)
class _RawImagingGeometry:
    """Describes the shape of a two-photon recording as its raw acquisition data reports it.

    Notes:
        Every field is read from the raw imaging directory, so the geometry is available before any stage of the
        pipeline has run. The heights follow each region's own line span, matching how the conversion stage slices a
        multi-region frame into planes.
    """

    sample_count: int
    """The samples one plane holds, as the floor across the interleave positions the conversion stage fills."""
    sampling_rate: float
    """The rate at which the recording sampled each plane."""
    raw_frame_pixels: int
    """The pixels one unsliced acquisition frame holds, which the conversion stage reads a batch of at a time."""
    plane_extents: tuple[tuple[int, int], ...]
    """The height and width of every plane, ordered by plane index."""


@dataclass(frozen=True, slots=True)
class _RecordingGeometry:
    """Describes the shape of a two-photon recording as its processing output reports it."""

    regions: int
    """The regions the single-recording pipeline detected."""
    samples: int
    """The samples each region's trace holds."""
    pixels: int
    """The pixels one combined multi-plane frame holds, which every multi-day stage works at."""


def resolve_host_memory_mb() -> int:
    """Reads the host's total physical memory.

    Returns:
        The host's total physical memory in megabytes.
    """
    return int(psutil.virtual_memory().total / _BYTES_PER_MEGABYTE)


def estimate_session_job_memory(
    pipeline: ProcessingPipelines, session: SessionData, jobs: list[tuple[str, str, int]]
) -> dict[tuple[str, str], tuple[int, bool]]:
    """Estimates the memory every possible job of one session occupies at its allocated core count.

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
        jobs: The possible jobs as ``(job_name, specifier, cores)`` triples.

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


def estimate_dataset_job_memory(
    dataset: DatasetData, jobs: list[tuple[str, str, int]]
) -> dict[tuple[str, str], tuple[int, bool]]:
    """Estimates the memory every possible forging job occupies, from the processed data it will read.

    Notes:
        Reads array headers and the recording metadata alone, so estimating a dataset decodes no fluorescence and
        opens no binary. Each two-photon stage scales with the processed data the single-recording pipeline wrote for
        the sessions that carry two-photon data.

        Every job is routed to a model rather than to a blanket allowance, since a remote scheduler reserves memory
        per job. A job whose processed input is absent falls back to the worker baseline, which the flag marks as a
        floor to plan around.

    Args:
        dataset: The resolved dataset the jobs operate on.
        jobs: The possible jobs as ``(job_name, specifier, cores)`` triples.

    Returns:
        A dictionary mapping each ``(job_name, specifier)`` pair to its estimated memory in megabytes and a flag that
        is True when the estimate follows from the job's own input rather than from a flat allowance.
    """
    project_root = dataset.dataset_data_path.parent.parent
    animals = {entry.session: entry.animal for entry in dataset.sessions}
    configuration = _resolve_tracking_configuration(dataset=dataset, project_root=project_root)

    estimates: dict[tuple[str, str], tuple[int, bool]] = {}
    for job_name, specifier, _cores in jobs:
        if job_name == MULTIDAY_DISCOVERY_JOB_NAME:
            estimates[job_name, specifier] = _estimate_discovery_memory(
                dataset=dataset, animal=specifier, project_root=project_root
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
            if job_name == MULTIDAY_EXTRACTION_JOB_NAME:
                estimates[job_name, specifier] = _estimate_extraction_memory(
                    geometry=geometry, regions=regions, configuration=configuration
                )
            else:
                estimates[job_name, specifier] = _estimate_assembly_memory(geometry=geometry, regions=regions)

    return estimates


def _resolve_raw_imaging_geometry(
    session: SessionData, configuration: SingleRecordingConfiguration
) -> _RawImagingGeometry | None:
    """Reads a two-photon recording's shape from its raw acquisition data.

    Notes:
        Reads the acquisition parameters and the headers of the first and last image files, so the cost stays flat
        as a recording grows and no pixel data is decoded. The sample count follows from the pages those two files
        report, since a recording's files hold an equal share apart from the last. Dividing those pages by the
        interleave stride gives the floor across the interleave positions, and every position below the remainder
        holds one sample more.

        Region line spans give each plane's height and the first image's header gives the shared width, which is how
        the conversion stage sizes a plane. That stage requires every image it discovers to carry the same frame
        shape.

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
        sample_count=max(1, total_pages // volumes),
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
    """Applies the shared estimate tolerance to a modeled memory figure and rounds it to a whole gigabyte.

    Notes:
        A scheduler reserves memory in whole gigabytes, so an estimate landing mid-gigabyte is rounded there by
        whatever consumes it. Rounding here instead keeps the figure a plan records identical to the figure a
        submission requests, which is what lets a planned batch and a submitted one be compared directly.

    Args:
        memory_mb: The modeled memory in megabytes, before any margin.

    Returns:
        The reportable memory in megabytes, carrying the tolerance and rounded up to a whole gigabyte.
    """
    reportable = int(memory_mb * _MEMORY_ESTIMATE_TOLERANCE) + 1
    return math.ceil(reportable / _MEGABYTES_PER_GIGABYTE) * _MEGABYTES_PER_GIGABYTE


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
        Each worker streams its file in fixed chunks and holds one at a time, so the figure is flat across every
        session size. The parent retains one pending result per file, which stays below the rounding this estimate
        already carries.

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


def _estimate_plane_registration_memory() -> int:
    """Reports the memory one two-photon plane-registration job is charged.

    Notes:
        Takes a flat allowance rather than a projection from the plane, for the reasons ``_REGISTRATION_MEMORY_MB``
        records. Kept as a function so the stage is routed like every other, and so a projection can replace the
        allowance here once what the footprint follows is known.

    Returns:
        The reportable memory in megabytes.
    """
    return _apply_tolerance(memory_mb=_REGISTRATION_MEMORY_MB)


def _estimate_plane_processing_memory(
    extent: tuple[int, int], geometry: _RawImagingGeometry, configuration: SingleRecordingConfiguration
) -> int:
    """Estimates the memory one two-photon plane-processing job holds, from that plane's shape and sample count.

    Notes:
        Detection sets the peak. The movie is binned down to at most the configured binned sample count, and the
        temporal standard deviation then holds the binned frames, their difference, and the squared difference at
        once. Registration narrows a plane to the region that stayed in frame before this stage reads it, so the raw
        extent used here is the wider bound.

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
    bin_size = max(1, geometry.sample_count // max(1, detection.maximum_binned_frames), decay_samples)
    binned_samples = max(1, geometry.sample_count // bin_size)

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
        specifier: The plane specifier for a registration or processing job, empty for the binarization and
            combination stages.
        geometry: The recording's raw geometry.
        configuration: The recording's resolved processing configuration.

    Returns:
        The reportable memory in megabytes.
    """
    if job_name == str(SingleRecordingJobNames.BINARIZE):
        return _estimate_binarization_memory(geometry=geometry, configuration=configuration)
    if job_name == str(SingleRecordingJobNames.COMBINE):
        return _apply_tolerance(memory_mb=_COMBINATION_MEMORY_MB)

    if job_name == str(SingleRecordingJobNames.REGISTER):
        estimates = [_estimate_plane_registration_memory() for _ in geometry.plane_extents]
    else:
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
    """Reads the plane index a per-plane job's specifier carries.

    Args:
        specifier: The job specifier, which names a plane by its index behind cindra's plane specifier prefix.

    Returns:
        The plane index, or None when the specifier does not name one.
    """
    if not specifier.startswith(PLANE_SPECIFIER_PREFIX):
        return None
    digits = specifier.removeprefix(PLANE_SPECIFIER_PREFIX)
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


def _resolve_recording_geometry(project_root: Path, animal: str, session: str) -> _RecordingGeometry | None:
    """Reads a processed recording's shape from the arrays the single-recording pipeline wrote.

    Args:
        project_root: The path to the project's root directory.
        animal: The animal the session belongs to.
        session: The session name whose processed output is read.

    Returns:
        The recording's geometry, or None when the session holds no processed imaging output.
    """
    directory = _two_photon_output_directory(project_root=project_root, animal=animal, session=session)
    traces = _read_array_shape(array_path=directory.joinpath(_FLUORESCENCE_FILENAME))
    metadata_path = directory.joinpath(_COMBINED_METADATA_FILENAME)
    if traces is None or not metadata_path.is_file():
        return None

    with np.load(file=metadata_path) as metadata:
        pixels = int(metadata["combined_height"][0]) * int(metadata["combined_width"][0])
    return _RecordingGeometry(regions=traces[0], samples=traces[1], pixels=pixels)


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

    Args:
        dataset: The resolved dataset whose acquisition system donates the configuration.
        project_root: The path to the project's root directory.

    Returns:
        The resolved configuration, or None when the dataset holds no session, or when its sessions need no
        multi-day processing.
    """
    if not dataset.sessions:
        return None
    entry = dataset.sessions[0]
    resolve_configuration = resolve_multi_recording_configuration_resolver(system=dataset.acquisition_system)
    return resolve_configuration(SessionData.load(session_path=project_root.joinpath(entry.animal, entry.session)))


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
        array_path=_two_photon_output_directory(project_root=project_root, animal=animal, session=session).joinpath(
            _MULTI_RECORDING_DIRECTORY,
            multi_recording_dataset_directory(animal_id=animal, dataset_name=dataset.name),
            _FLUORESCENCE_FILENAME,
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


def _estimate_discovery_memory(dataset: DatasetData, animal: str, project_root: Path) -> tuple[int, bool]:
    """Estimates the memory one cross-recording discovery job holds for a whole animal.

    Notes:
        Registration caches one deformation per unordered recording pair and never evicts it, so the plane count
        grows with the square of the animal's recording count. The clustering stage that follows sizes itself from
        local region crowding, which nothing on disk predicts, so it contributes a flat allowance.

    Args:
        dataset: The resolved dataset the animal belongs to.
        animal: The animal whose recordings are registered against each other.
        project_root: The path to the project's root directory.

    Returns:
        The reportable memory in megabytes and a flag stating whether the recording geometry was found.
    """
    geometries = [
        geometry
        for entry in dataset.get_sessions_for_animal(animal=animal)
        if (geometry := _resolve_recording_geometry(project_root=project_root, animal=animal, session=entry.session))
        is not None
    ]
    if not geometries:
        return _apply_tolerance(memory_mb=_WORKER_MEMORY_MB + _DISCOVERY_CLUSTERING_MEMORY_MB), False

    recordings = len(geometries)
    widest = max(geometry.pixels for geometry in geometries)
    planes = recordings * (recordings - 1) + _DISCOVERY_PLANES_PER_RECORDING * recordings
    registration = _bytes_to_megabytes(byte_count=planes * widest * _SINGLE_PRECISION_BYTES)
    return _apply_tolerance(memory_mb=_WORKER_MEMORY_MB + registration + _DISCOVERY_CLUSTERING_MEMORY_MB), True


def _estimate_extraction_memory(
    geometry: _RecordingGeometry | None, regions: int, configuration: MultiRecordingConfiguration | None
) -> tuple[int, bool]:
    """Estimates the memory one aligned-fluorescence extraction job holds for a single recording.

    Args:
        geometry: The recording's processed geometry.
        regions: The tracked regions the job extracts.
        configuration: The resolved multi-recording configuration, which reports the batch the job reads in.

    Returns:
        The reportable memory in megabytes and a flag stating whether the recording geometry was found.
    """
    if geometry is None or configuration is None:
        return _apply_tolerance(memory_mb=_WORKER_MEMORY_MB), False

    traces = _EXTRACTION_TRACE_COPIES * regions * geometry.samples * _SINGLE_PRECISION_BYTES
    batch = configuration.signal_extraction.batch_size * geometry.pixels * _EXTRACTION_BATCH_BYTES_PER_PIXEL
    retained = _EXTRACTION_BATCH_RETENTION * batch
    return _apply_tolerance(memory_mb=_WORKER_MEMORY_MB + _bytes_to_megabytes(byte_count=traces + retained)), True


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
        return _apply_tolerance(memory_mb=_WORKER_MEMORY_MB), False

    columns = (
        _ASSEMBLY_FLUORESCENCE_COLUMNS * _ASSEMBLY_WRITE_COPIES * geometry.samples * regions * _SINGLE_PRECISION_BYTES
    )
    sub_datasets = geometry.samples * _SUB_DATASET_BYTES_PER_SAMPLE
    return (
        _apply_tolerance(memory_mb=_WORKER_MEMORY_MB + _bytes_to_megabytes(byte_count=columns + sub_datasets)),
        True,
    )
