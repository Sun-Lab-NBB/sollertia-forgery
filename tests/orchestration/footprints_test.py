"""Contains tests for the sizing pass that resolves each job's cores and working set from the data it reads."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import numpy as np
from cindra import (
    WORKER_MEMORY_MB,
    PARAMETERS_FILENAME,
    PLANE_SPECIFIER_PREFIX,
    SPAWNED_CHILD_MEMORY_MB,
    MultiRecordingJobNames,
    SingleRecordingJobNames,
    size_multi_recording_job,
    size_single_recording_job,
)
import pandas as pd
import polars as pl
import pytest
import tifffile
from ataraxis_video_system import (
    CAMERA_MANIFEST_FILENAME,
    CAMERA_EXTRACTION_JOB_CORES,
    OutputLayout,
    CameraManifest,
    CameraSourceData,
    ExtractedDataColumns,
    size_archive_job as size_camera_extraction_job,
)
from sollertia_shared_assets import (
    DatasetData,
    SessionData,
    SubjectData,
    SurgeryData,
    SessionTypes,
    ProcedureData,
    DatasetSession,
)
from ataraxis_data_structures import read_archive_message_count
from ataraxis_communication_interface import (
    CONTROLLER_EXTRACTION_JOB_CORES,
    size_archive_job as size_controller_extraction_job,
)

from sollertia_forgery.video import (
    ENERGY_JOB_NAME,
    RENAME_JOB_NAME,
    TRACKING_JOB_NAME,
    CAMERA_EXTRACTION_JOB_NAME,
)
from sollertia_forgery.forging import (
    FORGING_JOB_NAME,
    MULTIDAY_DISCOVERY_JOB_NAME,
    MULTIDAY_EXTRACTION_JOB_NAME,
)
from sollertia_forgery.runtime import RUNTIME_JOB_NAME
from sollertia_forgery.managing import CHECKSUM_JOB_NAME
from sollertia_forgery.registries import (
    resolve_two_photon_data_locator,
    resolve_multi_recording_configuration_resolver,
    resolve_single_recording_configuration_resolver,
)
from sollertia_forgery.orchestration import footprints as footprints_module
from sollertia_forgery.shared_assets import ProcessingPipelines
from sollertia_forgery.microcontrollers import PARSE_JOB_NAME, CONTROLLER_EXTRACTION_JOB_NAME
from sollertia_forgery.video.motion_energy import MINIMUM_CHUNK_FRAMES, _plan_chunks
from sollertia_forgery.mesoscope_vr.metadata import BehaviorDataFiles
from sollertia_forgery.orchestration.footprints import (
    _POSE_TABLE_COPIES,
    _MODEL_VERSION_DIGITS,
    _DOUBLE_PRECISION_BYTES,
    _RETAINED_FRAME_BUFFERS,
    _SINGLE_PRECISION_BYTES,
    _UNRESOLVED_FRAME_COUNT,
    _DECODER_BUFFER_MEMORY_MB,
    _ARCHIVE_DIRECTORY_BYTES_PER_MESSAGE,
    JobFootprint,
    _EnergyRecording,
    _apply_tolerance,
    _read_array_shape,
    size_dataset_jobs,
    size_session_jobs,
    _round_to_gigabyte,
    _bytes_to_megabytes,
    resolve_model_version,
    _read_pose_table_shape,
    _size_motion_energy_job,
    _resolve_tracked_regions,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Mapping, Callable, Sequence

    from numpy.typing import NDArray

_WORKER_ONLY_MB: int = _apply_tolerance(memory_mb=WORKER_MEMORY_MB)
"""The memory charged to a stage holding one worker and nothing besides, which is the whole model for a job that
reads no input of its own."""

_PARALLEL_ARCHIVE_MESSAGES: int = 35_000
"""The messages a synthetic log archive holds to earn the parallel extraction path from both acquisition libraries.
The video library carries the higher of the two message thresholds, so an archive this size clears both."""

_FRAME_HEIGHT: int = 300
"""The line count of one unsliced acquisition frame that the synthetic imaging stacks carry."""

_FRAME_WIDTH: int = 64
"""The pixel width of one acquisition frame, shared by every plane the conversion stage slices out."""

_REGION_LINES: list[list[int]] = [[1, 100], [101, 300], []]
"""The per-region line spans declared by the synthetic acquisition parameters. The parameters name no roi_number, so
cindra reads the recording as single-region and discards every span."""

_SAMPLING_RATE: float = 10.0
"""The volume acquisition rate declared by the synthetic acquisition parameters, from which cindra derives a per-plane
rate of half this figure across the two declared planes."""

_CHECKSUM_READER_MEMORY_MB: int = 208
"""The resident memory the checksum model charges one reader, which is the shared cost of one spawned child plus the
eight-megabyte buffer that child streams every file through. The tunable terms of a model this package owns are
stated here rather than imported back out of it, so that retuning one moves this expectation instead of moving both
sides of the comparison together."""

_UNDERSTATED_CHECKSUM_READER_MEMORY_MB: int = 190
"""The figure the checksum model charged one reader while it stood below the shared spawned-child cost. Kept here so
the comparison below states what the correction is worth rather than only that some figure is reported."""

_CHECKSUM_WIDE_CORES: int = 16
"""The cores the wider of the two synthetic checksum jobs declares. Every estimate is reported at a whole gigabyte,
and a reader's whole cost is a fraction of that quantum, so the two candidate reader figures land in one bucket at the
narrower width and in different buckets only here."""

_CHECKSUM_WIDE_MEMORY_MB: int = 5120
"""The memory the checksum model reports for the wider job, stated outright rather than recomputed, so the expectation
does not move with the model it checks."""

_UNDERSTATED_CHECKSUM_WIDE_MEMORY_MB: int = 4096
"""The memory the same job would report were its readers modeled below the shared spawned-child cost, which is one
whole gigabyte less than the memory those readers hold."""

_ASSEMBLY_SINGLE_DAY_COLUMNS: int = 4
"""The fluorescence columns the per-session assembly model charges at the recording's own detected region count,
anchored on the same terms."""

_ASSEMBLY_MULTI_DAY_COLUMNS: int = 4
"""The fluorescence columns the same model charges at the count of regions tracked across the animal's recordings,
anchored on the same terms."""

_ASSEMBLY_WRITE_COPIES: int = 1
"""The copies of the assembled fluorescence volume the same model charges at the write, anchored on the same terms."""

_ASSEMBLY_LOAD_TRANSIENT_COLUMNS: int = 1
"""The extra fluorescence columns, at the recording's own detected region count, the same model charges for the
masked selection an assembly holds beside the contiguous copy it builds from it while loading one single-day column,
anchored on the same terms."""

_SUB_DATASET_BYTES_PER_SAMPLE: int = 512
"""The memory the same model charges the assembled behavior, runtime, and video columns per sample of the reference
clock on which they are placed, anchored on the same terms."""

_SOURCE_INPUT_BYTES_PER_SAMPLE: int = 128
"""The memory the same model charges one source of an assembly per sample of that source's own clock, anchored on the
same terms. It is a quarter of the charge one sample of the assembled frame carries, so the two terms cannot be read
off one another and a session whose sources tower over its reference clock is charged for both."""

_WIDE_ARCHIVE_BYTES: int = 200 * 1024 * 1024
"""The size of the wider of the two synthetic log archives. Every estimate is reported at a whole gigabyte and the
parse model charges a worker above the archive term, so an archive below roughly 150 megabytes reports the floor
gigabyte whatever its length. This archive clears that boundary, which is what lets a figure taken from the wrong
archive, or from a share of the right one, differ from the expected one."""

_WIDE_ARCHIVE_MEMORY_MB: int = 2048
"""The memory the parse model reports for the wide archive, stated outright rather than recomputed, so the
expectation does not move with the model it checks."""

_NARROW_ARCHIVE_MEMORY_MB: int = 1024
"""The memory the same model reports for a controller whose archive holds a handful of messages, which is the figure
a parse job charged its own narrow archive receives rather than the one its wide-archive neighbour receives."""

_WIDE_CLOCK_FRAMES: int = 3_000_000
"""The frames the wider of the two synthetic camera clocks holds. Every estimate is reported at a whole gigabyte, and
the behavior-only model charges one sub-dataset term per sample above a worker, so any clock below roughly one
million samples reports the same single gigabyte whatever its length. This clock is long enough to clear two of those
boundaries, which is what lets a figure taken from the wrong clock, or from fluorescence, differ from the expected
one."""

_NARROW_CLOCK_FRAMES: int = 1_200_000
"""The frames the narrower clock holds. It spans the same duration as the wide clock at a lower rate, which makes its
camera the slower one, and it lands one whole gigabyte below the wide clock rather than in the same bucket."""

_WIDE_CLOCK_MEMORY_MB: int = 3072
"""The memory the behavior-only model reports for the wide clock, stated outright rather than recomputed, so the
expectation does not move with the model it checks."""

_NARROW_CLOCK_MEMORY_MB: int = 2048
"""The memory the same model reports for the narrow clock, which is the figure an estimate settling on the slower
camera would report instead."""

_WIDE_CLOCK_PERIOD_US: int = 1_000
"""The microseconds between consecutive frames of the wide clock."""

_NARROW_CLOCK_PERIOD_US: int = 2_500
"""The microseconds between consecutive frames of the narrow clock, which spans the same duration at a lower rate."""

_SOURCED_REFERENCE_FRAMES: int = 1_000_000
"""The frames the slower camera of the source-charged session acquired, which is the clock its assembly settles on and
therefore the height of the frame that assembly builds."""

_SOURCED_REFERENCE_PERIOD_US: int = 2_500
"""The microseconds between consecutive frames of that slower camera."""

_SOURCED_FAST_FRAMES: int = 5_000_000
"""The frames the faster camera of the same session acquired over the same span. Its feathers are read at this height
and only then interpolated onto the reference clock, so they stand five times taller than the frame the job builds."""

_SOURCED_FAST_PERIOD_US: int = 500
"""The microseconds between consecutive frames of that faster camera, a fifth of the slower camera's period."""

_SOURCED_BEHAVIOR_SAMPLES: int = 2_000_000
"""The samples the session's encoder feather holds. A behavior source carries its own clock as much as a camera does,
so it stands at its own height in the estimate rather than at the reference clock's."""

_SOURCED_ASSEMBLY_MEMORY_MB: int = 3072
"""The memory the behavior-only model reports for that session, stated outright rather than recomputed, so the
expectation does not move with the model it checks."""

_SOURCED_FRAME_ONLY_MEMORY_MB: int = 1024
"""The memory the same session would report were the sources left out and the reference clock charged alone, which is
the figure the estimate carried while it under-reserved the job."""

_SOURCED_SOURCES_ONLY_MEMORY_MB: int = 2048
"""The memory the same session would report were the assembled frame left out and its sources charged alone."""

_SOURCED_CAMERAS_ONLY_MEMORY_MB: int = 2048
"""The memory the same session would report were its behavior feathers left out of the source term, counting the
cameras alone."""

_SOURCED_WIDEST_CLOCK_MEMORY_MB: int = 5120
"""The memory the same session would report were the assembled frame placed on the widest clock the session recorded
rather than on the clock its assembler settles on."""

_FACE_SOURCE_ID: str = "51"
"""The manifest source identifier of the first synthetic camera, which is the specifier its motion-energy job
carries."""

_BODY_SOURCE_ID: str = "73"
"""The manifest source identifier of the second synthetic camera, kept distinct so a session plans two independently
sized motion-energy jobs."""

_FACE_CAMERA: str = "face_camera"
"""The colloquial manifest name of the first synthetic camera, which names its recording on disk."""

_BODY_CAMERA: str = "body_camera"
"""The colloquial manifest name of the second synthetic camera."""

_ENERGY_JOB_CORES: int = 8
"""The cores a synthetic motion-energy job declares. It bounds the chunks the stage plans, so it is the width at which
a recording long enough to fill every chunk is charged."""

_LARGE_FRAME_SHAPE: tuple[int, int] = (514, 1024)
"""The height and width of the larger synthetic recording's frame. Both extents are even so the encoder accepts
them."""

_POOLED_REGION_TEST_REGIONS: int = 25_000
"""The regions each recording of the incomplete-set comparison reports. The discovery stage is quadratic in the pooled
region count of the set it spans, and every estimate is reported at a whole gigabyte, so the recordings are made wide
enough that halving the set moves the figure across a gigabyte boundary rather than inside one."""

_POOLED_REGION_TEST_SAMPLES: int = 64
"""The samples each of those recordings' traces hold, kept short so a recording wide enough to move the quadratic term
is still cheap to write."""

_POSE_TABLE_ROWS: int = 20_000
"""The frames the synthetic pose prediction table holds."""

_POSE_TABLE_BODYPARTS: int = 13
"""The bodyparts that table carries, which is the canonical count the pupil stage reads. Each contributes a
horizontal position, a vertical position, and a likelihood, so the table stands three times this wide."""

_POSE_COORDINATES_PER_BODYPART: int = 3
"""The columns one bodypart contributes to the prediction table."""

_POSE_COMPRESSION_LEVEL: int = 9
"""The compression level the compressed synthetic prediction is written at. A DeepLabCut deployment can carry such a
setting, and it shrinks the file without narrowing the table the stage expands in memory."""

_POSE_TABLE_COPIES_OVERRIDE: float = 400.0
"""The copies term the compression comparison runs under. A table small enough to write in a test spans a fraction of
the gigabyte every estimate is rounded to, so the term is raised until the two candidate models land in different
buckets. The term under test is the base those copies multiply, not the copies themselves."""

_POSE_TABLE_MEMORY_MB: int = 4096
"""The memory the pose model reports for that table under the raised copies term, whatever the writer compressed."""

_POSE_COMPRESSED_FILE_MEMORY_MB: int = 1024
"""The memory a model charging the prediction's size on disk would report for the compressed copy of that same table,
three whole gigabytes below the memory the stage goes on to hold."""

_SMALL_FRAME_SHAPE: tuple[int, int] = (48, 64)
"""The height and width of the smaller synthetic recording's frame, kept small so a recording long enough to plan
several decode chunks is still cheap to encode."""

_CHUNKED_FRAME_COUNT: int = MINIMUM_CHUNK_FRAMES * 2
"""The frames the longer synthetic recording holds, which is exactly enough for the stage to plan two decode chunks
and open a pool. Every frame below the shared minimum is decoded by the job's own process instead."""

_SINGLE_CHUNK_FRAME_COUNT: int = 4
"""The frames the shorter synthetic recording holds, which is far below the shared chunk minimum, so the stage decodes
it in the job's own process and opens no pool at all."""

_WIDE_ENERGY_FRAME_PIXELS: int = 1280 * 1024
"""The pixels one frame of the wider camera of the direct model comparison holds. The frame buffers are a small share
of a decode worker's cost, so the two frames compared below are set far enough apart to report different whole
gigabytes once every worker the allocation permits is open."""

_NARROW_ENERGY_FRAME_PIXELS: int = 640 * 480
"""The pixels one frame of the narrower camera of the same comparison holds."""

_FULL_WIDTH_ENERGY_CORES: int = 52
"""The cores the direct model comparison declares. The frame term only clears a gigabyte boundary once enough decoders
are open to multiply it, which is the width at which the recording set below is compared."""

_WIDE_ENERGY_MEMORY_MB: int = 19456
"""The memory the motion-energy model reports for the wider frame at that width, stated outright rather than
recomputed, so the expectation does not move with the model it checks."""

_NARROW_ENERGY_MEMORY_MB: int = 18432
"""The memory the same model reports for the narrower frame at the same width, one whole gigabyte below the wider
frame's, which is the figure a job charged the wrong camera's frame would report."""

_SINGLE_CHUNK_ENERGY_MEMORY_MB: int = 1024
"""The memory the same model reports for the narrower frame when the recording is too short to plan more than one
decode chunk. The job decodes in its own process, so it holds one decoder and starts no child at all."""

_FULL_CHUNK_ENERGY_MEMORY_MB: int = 6144
"""The memory the same model reports for the same frame once the recording is long enough to fill every chunk the
allocation permits. It is six times the single-chunk figure, which is what charging the allocation rather than the
chunk plan reserved for every short clip a rig records."""

_UNCHUNKED_WIDE_FRAME_PIXELS: int = 52_500_000
"""The pixels one frame of the recording that proves a single-chunk job starts no child holds. One spawned child costs
a fraction of the gigabyte every estimate is rounded to, so the frame beside it has to stand at the width where that
fraction crosses a boundary. No camera records such a frame: this is the width at which the branch is observable at
the quantum the estimate reports, not a frame a rig produces."""

_UNCHUNKED_WIDE_MEMORY_MB: int = 1024
"""The memory the motion-energy model reports for that frame when the recording plans a single chunk, which covers one
decoder held in the job's own process."""

_UNCHUNKED_WIDE_WITH_CHILD_MEMORY_MB: int = 2048
"""The memory the same frame would report were the job charged a spawned child it never starts, which is one whole
gigabyte above the memory the job holds."""

_CHUNKED_ENERGY_JOB_CORES: int = 16
"""The cores the chunk-count comparison declares, which is the motion-energy allocation the dispatch table itself
carries."""


def write_surgery_metadata(session: SessionData, genotype: str = "GP5.17") -> Path:
    """Writes the surgery metadata from which the Mesoscope-VR cindra resolvers read an animal's genotype.

    Args:
        session: The session whose raw data receives the metadata file.
        genotype: The genotype recorded on the animal, which selects the indicator-tuned configuration.

    Returns:
        The path to the written metadata file.
    """
    path = session.raw_data.surgery_metadata_path
    path.parent.mkdir(parents=True, exist_ok=True)
    SurgeryData(
        subject=SubjectData(
            id=int(session.animal_id),
            ear_punch="none",
            sex="F",
            genotype=genotype,
            date_of_birth_us=0,
            weight_g=25.0,
            cage=1,
            location_housed="vivarium",
            status="alive",
        ),
        procedure=ProcedureData(
            surgery_start_us=0,
            surgery_end_us=1,
            surgeon="tester",
            protocol="synthetic",
            surgery_notes="none",
            post_op_notes="none",
        ),
        drugs=[],
        implants=[],
        injections=[],
    ).to_yaml(file_path=path)
    return path


def write_acquisition_parameters(directory: Path, *, region_lines: list[list[int]] | None = None) -> Path:
    """Writes a cindra acquisition parameters file describing the synthetic recording's shape.

    Args:
        directory: The raw imaging directory into which the file is written.
        region_lines: The per-region line spans to declare, or None to declare none at all.

    Returns:
        The path to the written parameters file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    parameters: dict[str, object] = {"frame_rate": _SAMPLING_RATE, "plane_number": 2, "channel_number": 1}
    if region_lines is not None:
        parameters["roi_lines"] = region_lines
    path = directory.joinpath(PARAMETERS_FILENAME)
    path.write_text(json.dumps(parameters))
    return path


def write_imaging_stack(directory: Path, name: str, pages: int) -> Path:
    """Writes one multi-page acquisition image whose header reports the synthetic recording's frame shape.

    Args:
        directory: The raw imaging directory into which the image is written.
        name: The filename to write, including its extension.
        pages: The number of pages the image holds.

    Returns:
        The path to the written image.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory.joinpath(name)
    tifffile.imwrite(path, np.zeros((pages, _FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.int16))
    return path


def write_raw_imaging(session: SessionData, *, region_lines: list[list[int]] | None = _REGION_LINES) -> Path:
    """Builds a complete synthetic raw imaging directory holding two acquisition images and their parameters.

    Args:
        session: The session whose raw data receives the imaging directory.
        region_lines: The per-region line spans declared by the parameters.

    Returns:
        The path to the raw imaging directory.
    """
    directory = session.raw_data_path.joinpath("mesoscope_data")
    write_acquisition_parameters(directory=directory, region_lines=region_lines)
    write_imaging_stack(directory=directory, name="recording_00001.tif", pages=10)
    write_imaging_stack(directory=directory, name="recording_00002.tif", pages=6)
    # The configuration excludes this name, so the reader must leave it out of both the count and the ordering.
    write_imaging_stack(directory=directory, name="zstack.tif", pages=400)
    return directory


def write_trace_array(path: Path, shape: tuple[int, ...], *, version: tuple[int, int] = (1, 0)) -> Path:
    """Writes an array whose header alone reports the extents the estimators read.

    Args:
        path: The path to which the array is written.
        shape: The extents the header reports.
        version: The array format version under which the header is written.

    Returns:
        The path to the written array.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as array_file:
        np.lib.format.write_array(array_file, np.zeros(shape, dtype=np.float32), version=version)
    return path


def write_combined_metadata(directory: Path, *, height: int, width: int, frame_count: int | None = None) -> Path:
    """Writes the cindra archive reporting the combined field extent at which every multi-day stage works.

    The frame count is written only when one is stated, because cindra added that field after the field extent and
    reads a missing one back as zero. An archive written without it is what a recording processed by an earlier
    cindra carries, which is the shape every fixture here poses unless it needs cindra's own extraction model to
    scale with the frames.

    Args:
        directory: The processed output directory into which the archive is written.
        height: The combined field height in pixels.
        width: The combined field width in pixels.
        frame_count: The frames the archive records for the combined view, or None to write an archive recording
            none, as an earlier cindra release left it.

    Returns:
        The path to the written archive.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory.joinpath("combined_metadata.npz")
    payload = {"combined_height": np.array([height]), "combined_width": np.array([width])}
    if frame_count is not None:
        payload["frame_count"] = np.array([frame_count])
    np.savez(path, **payload)
    return path


def write_trace_header(path: Path, shape: tuple[int, ...]) -> Path:
    """Writes an array header that reports the given extents while holding none of the values it describes.

    Notes:
        Every estimator reads a trace array's header alone, so a header standing on its own states the same extents a
        densely written array of those extents would state. This is what lets a fixture pose a recording whose region
        count separates one whole-gigabyte bucket from the next without writing the gigabytes such a recording holds.

    Args:
        path: The path to which the header is written.
        shape: The extents the header reports.

    Returns:
        The path to the written header.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as array_file:
        np.lib.format.write_array_header_1_0(
            array_file, {"descr": "<f4", "fortran_order": False, "shape": tuple(shape)}
        )
    return path


def write_processed_recording(
    session: SessionData,
    *,
    regions: int,
    samples: int,
    height: int = 128,
    width: int = 96,
    dense: bool = True,
    record_frame_count: bool = False,
) -> Path:
    """Writes the single-recording outputs from which a dataset job's estimate is sized.

    Args:
        session: The session whose processed data receives the outputs.
        regions: The regions the recording's traces hold.
        samples: The samples each trace holds.
        height: The combined field height in pixels.
        width: The combined field width in pixels.
        dense: Whether the trace array holds the values its header describes. A recording posed at a scale that
            separates gigabyte buckets is written as a header alone, since that is all the estimators read.
        record_frame_count: Whether the metadata archive records the combined frame count alongside the field extent.
            The traces and the archive are written from one sample count here, so a recording recording it reports the
            same frames the traces span. Left False by default, which poses the archive an earlier cindra wrote and
            which this package's own sample count must survive.

    Returns:
        The path to the session's cindra output directory.
    """
    directory = session.processed_data.cindra_data_path
    traces = directory.joinpath("cell_fluorescence.npy")
    if dense:
        write_trace_array(path=traces, shape=(regions, samples))
    else:
        write_trace_header(path=traces, shape=(regions, samples))
    write_combined_metadata(
        directory=directory, height=height, width=width, frame_count=samples if record_frame_count else None
    )
    return directory


def write_camera_clock(session: SessionData, *, camera: str, frames: int, period_us: int = 33_000) -> Path:
    """Writes one camera's timestamp feather, which is the clock a session recording no imaging is assembled onto.

    The feather is published under the canonical name the video library's own layout states, which is the name the
    assembly reads and the name the estimate counts the rows of.

    Args:
        session: The session whose processed video data receives the feather.
        camera: The colloquial camera name under which the feather is published.
        frames: The frames the camera acquired, which is the samples its clock holds.
        period_us: The microseconds separating consecutive frames, which sets the camera's mean rate.

    Returns:
        The path to the written feather.
    """
    directory = session.processed_data.video_data_path
    directory.mkdir(parents=True, exist_ok=True)
    path = directory.joinpath(f"{camera}{OutputLayout.TIMESTAMPS_INFIX}{OutputLayout.FILE_SUFFIX}")
    timestamps = np.arange(frames, dtype=np.uint64) * np.uint64(period_us)
    pl.DataFrame({ExtractedDataColumns.FRAME_TIME: timestamps}).write_ipc(file=path, compression="uncompressed")
    return path


def write_behavior_source(session: SessionData, source_file: BehaviorDataFiles, samples: int) -> Path:
    """Writes one module-parsed behavior feather, which the assembly reads at that feather's own height.

    The feather carries the shape the encoder parser writes: one timestamp and one traveled-distance value per
    logged sample. The assembly reads it whole and interpolates it onto the reference clock, so its own height is
    what it contributes to the estimate.

    Args:
        session: The session whose processed microcontroller data receives the feather.
        source_file: The canonical filename under which the parser publishes the feather.
        samples: The samples the feather holds.

    Returns:
        The path to the written feather.
    """
    directory = session.processed_data.microcontroller_data_path
    directory.mkdir(parents=True, exist_ok=True)
    path = directory.joinpath(source_file)
    pl.DataFrame(
        {
            "time_us": np.arange(samples, dtype=np.uint64),
            "traveled_distance_cm": np.arange(samples, dtype=np.float64),
        }
    ).write_ipc(file=path, compression="uncompressed")
    return path


def write_camera_manifest(session: SessionData, cameras: Mapping[str, str]) -> Path:
    """Writes the acquisition-time camera manifest through which a motion-energy specifier reaches its camera.

    The specifier a motion-energy job carries is a source identifier, and the colloquial name that locates that
    camera's recording on disk lives in this manifest, so a session that registers no camera leaves every one of its
    energy jobs measuring no recording.

    Args:
        session: The session whose raw behavior data receives the manifest.
        cameras: The colloquial name of every registered camera, keyed by its source identifier.

    Returns:
        The path to the written manifest.
    """
    directory = session.raw_data.behavior_data_path
    directory.mkdir(parents=True, exist_ok=True)
    path = directory.joinpath(CAMERA_MANIFEST_FILENAME)
    CameraManifest(
        sources=[CameraSourceData(id=int(source_id), name=name) for source_id, name in cameras.items()]
    ).to_yaml(file_path=path)
    return path


def write_camera_recording(
    session: SessionData, camera: str, frames: NDArray[np.uint8], writer: Callable[..., Path]
) -> Path:
    """Writes one camera's recording under the deterministic name the motion-energy stage itself resolves.

    Args:
        session: The session whose raw camera data receives the recording.
        camera: The colloquial manifest name of the camera that recorded it.
        frames: The frame stack the recording holds, whose extents the estimate reads back.
        writer: The fixture that encodes the frame stack into a container.

    Returns:
        The path to the written recording.
    """
    directory = session.raw_data.camera_data_path
    directory.mkdir(parents=True, exist_ok=True)
    return writer(directory.joinpath(f"{session.session_name}_{camera}.mp4"), frames)


def energy_memory(frame_pixels: int, frame_count: int, cores: int) -> int:
    """Reports the figure the motion-energy model gives a job decoding the named recording.

    The chunk count is asked of the stage's own planner rather than restated here, so an expectation built by this
    helper follows the stage a retuned threshold would move rather than a rule written beside it.

    Args:
        frame_pixels: The pixels one frame of the job's own recording holds.
        frame_count: The frames the recording's container reports, or ``_UNRESOLVED_FRAME_COUNT`` when the pass read
            no container at all.
        cores: The cores the job is allocated, which bounds how many decoders it opens.

    Returns:
        The reportable memory in megabytes.
    """
    frame_buffers = _bytes_to_megabytes(byte_count=frame_pixels * _SINGLE_PRECISION_BYTES * _RETAINED_FRAME_BUFFERS)
    chunks = cores if frame_count < 0 else len(_plan_chunks(frame_count=frame_count, workers=cores))
    if chunks == 1:
        return _apply_tolerance(memory_mb=WORKER_MEMORY_MB + frame_buffers + _DECODER_BUFFER_MEMORY_MB)
    per_worker = frame_buffers + _DECODER_BUFFER_MEMORY_MB + SPAWNED_CHILD_MEMORY_MB
    return _apply_tolerance(memory_mb=WORKER_MEMORY_MB + chunks * per_worker)


def build_dataset(
    project_root: Path, name: str, sessions: Sequence[SessionData], session_type: SessionTypes
) -> DatasetData:
    """Creates a forged dataset hierarchy naming the given sessions, through the shared hierarchy's own creator.

    Args:
        project_root: The project root under which the dataset is created and from which its sessions are resolved.
        name: The dataset name.
        sessions: The acquired sessions the dataset covers.
        session_type: The session type every covered session shares.

    Returns:
        The created dataset.
    """
    return DatasetData.create(
        name=name,
        project=project_root.stem,
        session_type=session_type,
        acquisition_system=sessions[0].acquisition_system,
        sessions=tuple(
            DatasetSession(session=session.session_name, animal=str(session.animal_id)) for session in sessions
        ),
        datasets_root=project_root,
        column_descriptions={"time_us": "The sample timestamp."},
    )


def two_photon_estimates(session: SessionData, jobs: list[tuple[str, str, int]]) -> dict[tuple[str, str], JobFootprint]:
    """Sizes the named two-photon jobs of one session.

    Args:
        session: The session on which the jobs operate.
        jobs: The jobs as name, specifier, and declared core triples.

    Returns:
        The footprint of every named job.
    """
    return size_session_jobs(pipeline=ProcessingPipelines.TWO_PHOTON, session=session, jobs=jobs)


def camera_library_footprint(archive: Path) -> JobFootprint:
    """Reports the footprint the video library's own sizing pass gives one archive, at slf's scale.

    Args:
        archive: The log archive the extraction job reads.

    Returns:
        The library's cores and its memory, rounded up to the whole gigabyte on which every estimate lands.
    """
    sizing = size_camera_extraction_job(archive_path=archive)
    return JobFootprint(cores=sizing.cores, memory_mb=_round_to_gigabyte(memory_mb=sizing.memory_mb))


def controller_library_footprint(archive: Path) -> JobFootprint:
    """Reports the footprint the communication library's own sizing pass gives one archive, at slf's scale.

    Args:
        archive: The log archive the extraction job reads.

    Returns:
        The library's cores and its memory, rounded up to the whole gigabyte on which every estimate lands.
    """
    sizing = size_controller_extraction_job(archive_path=archive)
    return JobFootprint(cores=sizing.cores, memory_mb=_round_to_gigabyte(memory_mb=sizing.memory_mb))


def write_sized_archive(path: Path, size_bytes: int) -> Path:
    """Writes a log archive that reports the requested size on disk without holding that many bytes.

    Notes:
        The parse model stats its controller's archive and reads nothing inside it, so a sparse file of the requested
        length states the same figure a densely written archive of that length would state, at no cost in disk.

    Args:
        path: The path the archive is written to, whose name carries the source identifier the locating pass matches.
        size_bytes: The size the written archive reports.

    Returns:
        The path to the written archive.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as archive:
        archive.truncate(size_bytes)
    return path


def cindra_single_recording_footprint(
    session: SessionData, job_name: SingleRecordingJobNames, specifier: str
) -> JobFootprint:
    """Reports the footprint cindra's own sizing pass gives one single-recording stage, at slf's scale.

    Args:
        session: The session on which the job operates.
        job_name: The cindra stage the job runs.
        specifier: The job's tracker specifier, which names a plane for the per-plane stages.

    Returns:
        cindra's own width for the stage and its memory, rounded up to the whole gigabyte on which every estimate lands.
    """
    resolve_configuration = resolve_single_recording_configuration_resolver(system=session.acquisition_system)
    locate_two_photon_data = resolve_two_photon_data_locator(system=session.acquisition_system)
    sizing = size_single_recording_job(
        job_name=job_name,
        specifier=specifier,
        output_root=session.processed_data_path,
        configuration=resolve_configuration(session),
        data_path=locate_two_photon_data(session=session),
    )
    return JobFootprint(cores=sizing.cores, memory_mb=_round_to_gigabyte(memory_mb=sizing.memory_mb))


def cindra_multi_recording_footprint(
    dataset: DatasetData,
    sessions: Sequence[SessionData],
    job_name: MultiRecordingJobNames,
    specifier: str,
    planned_roi_count: int | None = None,
) -> JobFootprint:
    """Reports the footprint cindra's own sizing pass gives one cross-recording stage, at slf's scale.

    Args:
        dataset: The dataset whose acquisition system donates the multi-recording configuration.
        sessions: The sessions whose cindra output directories are spanned by the stage.
        job_name: The cindra stage the job runs.
        specifier: The job's tracker specifier, naming a session for extraction and an animal for discovery.
        planned_roi_count: The tracked templates cindra is told to plan the stage for, or None to let it draw its own
            bound from the per-recording region counts.

    Returns:
        cindra's own width for the stage and its memory, rounded up to the whole gigabyte on which every estimate lands.
    """
    resolve_configuration = resolve_multi_recording_configuration_resolver(system=dataset.acquisition_system)
    sizing = size_multi_recording_job(
        job_name=job_name,
        specifier=specifier,
        recording_directories=[session.processed_data.cindra_data_path for session in sessions],
        configuration=resolve_configuration(sessions[0]),
        planned_roi_count=planned_roi_count,
    )
    return JobFootprint(cores=sizing.cores, memory_mb=_round_to_gigabyte(memory_mb=sizing.memory_mb))


def assembly_memory(
    samples: int, regions: int, tracked_regions: int | None = None, source_samples: Sequence[int] = ()
) -> int:
    """Reports the figure the per-session assembly model gives a recording of the named shape.

    Args:
        samples: The samples each retained fluorescence column holds, which is the fluorescence clock the assembled
            frame is placed on.
        regions: The regions the recording itself detected, which the single-day columns span.
        tracked_regions: The regions tracked across the animal, which the multi-day columns span. Defaults to the
            recording's own count.
        source_samples: The samples each source the assembly reads holds on that source's own clock. Defaults to no
            source, which is the shape of a session carrying none of the feathers its assembler reads.

    Returns:
        The reportable memory in megabytes.
    """
    retained = _ASSEMBLY_SINGLE_DAY_COLUMNS * regions + _ASSEMBLY_MULTI_DAY_COLUMNS * (
        regions if tracked_regions is None else tracked_regions
    )
    columns = retained * _ASSEMBLY_WRITE_COPIES * samples * _SINGLE_PRECISION_BYTES
    transient = _ASSEMBLY_LOAD_TRANSIENT_COLUMNS * regions * samples * _SINGLE_PRECISION_BYTES
    sources = sum(source_samples) * _SOURCE_INPUT_BYTES_PER_SAMPLE
    return _apply_tolerance(
        memory_mb=WORKER_MEMORY_MB
        + _bytes_to_megabytes(byte_count=columns + transient + samples * _SUB_DATASET_BYTES_PER_SAMPLE + sources)
    )


# Session estimates


def test_checksum_memory_scales_with_the_readers_a_job_opens(experiment_session: SessionData) -> None:
    """Verifies that the checksum estimate follows the cores a job holds, since each core streams one file in fixed
    chunks.
    """
    estimates = size_session_jobs(
        pipeline=ProcessingPipelines.CHECKSUM,
        session=experiment_session,
        jobs=[(CHECKSUM_JOB_NAME, "", 8), (CHECKSUM_JOB_NAME, "wide", _CHECKSUM_WIDE_CORES)],
    )

    narrow = estimates[CHECKSUM_JOB_NAME, ""]
    wide = estimates[CHECKSUM_JOB_NAME, "wide"]
    assert narrow.memory_mb == _apply_tolerance(memory_mb=WORKER_MEMORY_MB + 8 * _CHECKSUM_READER_MEMORY_MB)
    assert wide.memory_mb > narrow.memory_mb
    # The checksum stage is this package's own, so each job is planned at the allocation it was handed.
    assert (narrow.cores, wide.cores) == (8, _CHECKSUM_WIDE_CORES)


def test_a_checksum_reader_is_charged_the_spawned_child_it_is(experiment_session: SessionData) -> None:
    """Verifies that a checksum worker is charged the shared cost of a spawned child plus the buffer it streams files
    through, rather than a figure standing below that shared cost.

    The pool that opens the workers is spawn-started, so each worker pays the interpreter and import graph the shared
    figure covers and holds its read buffer above it. The width compared here is the one at which the correction is
    worth a whole reportable gigabyte, so a model charging the understated figure reports a different bucket rather
    than the same one.
    """
    estimates = size_session_jobs(
        pipeline=ProcessingPipelines.CHECKSUM,
        session=experiment_session,
        jobs=[(CHECKSUM_JOB_NAME, "", _CHECKSUM_WIDE_CORES)],
    )

    assert estimates[CHECKSUM_JOB_NAME, ""].memory_mb == _CHECKSUM_WIDE_MEMORY_MB
    # A reader modeled below the shared spawned-child cost reserves the job a whole gigabyte less than it holds.
    assert (
        _apply_tolerance(memory_mb=WORKER_MEMORY_MB + _CHECKSUM_WIDE_CORES * _UNDERSTATED_CHECKSUM_READER_MEMORY_MB)
        == _UNDERSTATED_CHECKSUM_WIDE_MEMORY_MB
    )
    assert estimates[CHECKSUM_JOB_NAME, ""].memory_mb > _UNDERSTATED_CHECKSUM_WIDE_MEMORY_MB
    # Expressing the reader as the shared figure plus its buffer is what makes a retune of that shared figure reach
    # this model, so the reader can never again be modeled as cheaper than the child it runs in.
    assert footprints_module._CHECKSUM_READER_MEMORY_MB >= SPAWNED_CHILD_MEMORY_MB


def test_an_archive_reader_estimate_scales_with_the_archive_on_disk(
    experiment_session: SessionData, write_log_archive: Callable[..., Path]
) -> None:
    """Verifies that every worker of a log-reading stage opens the archive itself, so the estimate is charged once per
    core.
    """
    archive = write_log_archive(experiment_session.raw_data.behavior_data_path.joinpath("51_log.npz"), 51, [(5, b"ab")])
    per_reader = _bytes_to_megabytes(
        byte_count=read_archive_message_count(archive_path=archive) * _ARCHIVE_DIRECTORY_BYTES_PER_MESSAGE
    )

    estimates = size_session_jobs(
        pipeline=ProcessingPipelines.RUNTIME,
        session=experiment_session,
        jobs=[(RUNTIME_JOB_NAME, "51", 4), (CONTROLLER_EXTRACTION_JOB_NAME, "51", 8)],
    )

    # A four-core job opens four children and keeps the reader that planned their batches, so five readers hold the
    # archive's directory while four children carry their own cost.
    assert estimates[RUNTIME_JOB_NAME, "51"] == JobFootprint(
        cores=4,
        memory_mb=_apply_tolerance(memory_mb=WORKER_MEMORY_MB + 5 * per_reader + 4 * SPAWNED_CHILD_MEMORY_MB),
    )
    # The extraction stage belongs to the communication library, so both halves of its figure are that library's own
    # sizing pass, with the memory rounded to the gigabyte on which every reportable estimate lands.
    assert estimates[CONTROLLER_EXTRACTION_JOB_NAME, "51"] == controller_library_footprint(archive=archive)


def test_a_runtime_job_whose_archive_was_never_written_is_refused(experiment_session: SessionData) -> None:
    """Verifies that a log-reading stage holds memory in proportion to an archive, so an absent archive names no figure
    and the job that would read it cannot run either.
    """
    experiment_session.raw_data.behavior_data_path.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="Unable to find the log archive of source '51'"):
        size_session_jobs(
            pipeline=ProcessingPipelines.RUNTIME,
            session=experiment_session,
            jobs=[(RUNTIME_JOB_NAME, "51", 4)],
        )


def test_a_camera_extraction_estimate_follows_the_video_library_model(
    experiment_session: SessionData, write_log_archive: Callable[..., Path]
) -> None:
    """Verifies that the camera timestamp stage belongs to the video library, so slf reports that library's figures
    unchanged.
    """
    archive = write_log_archive(experiment_session.raw_data.behavior_data_path.joinpath("77_log.npz"), 77, [(5, b"ab")])

    estimates = size_session_jobs(
        pipeline=ProcessingPipelines.VIDEO,
        session=experiment_session,
        jobs=[(CAMERA_EXTRACTION_JOB_NAME, "77", 8)],
    )

    assert estimates[CAMERA_EXTRACTION_JOB_NAME, "77"] == camera_library_footprint(archive=archive)
    # The controller archive of source 99 was never written, and each library refuses an archive it cannot read
    # because the job reading it could not run either. That refusal reaches the caller unchanged.
    with pytest.raises(FileNotFoundError, match="Unable to find the log archive of source '99'"):
        size_session_jobs(
            pipeline=ProcessingPipelines.VIDEO,
            session=experiment_session,
            jobs=[(CONTROLLER_EXTRACTION_JOB_NAME, "99", 8)],
        )
    with pytest.raises(FileNotFoundError, match="Unable to find the log archive of source '99'"):
        size_session_jobs(
            pipeline=ProcessingPipelines.VIDEO,
            session=experiment_session,
            jobs=[(CAMERA_EXTRACTION_JOB_NAME, "99", 8)],
        )


def test_an_extraction_job_is_sized_at_the_width_its_own_archive_earns(
    experiment_session: SessionData, write_log_archive: Callable[..., Path]
) -> None:
    """Verifies that each library picks a width from the archive its job reads, so a small archive runs sequentially
    whatever allocation this package declared for the stage.
    """
    archive = write_log_archive(
        experiment_session.raw_data.behavior_data_path.joinpath("77_log.npz"),
        77,
        [(index, bytes(400)) for index in range(64)],
    )

    estimates = size_session_jobs(
        pipeline=ProcessingPipelines.VIDEO,
        session=experiment_session,
        jobs=[(CAMERA_EXTRACTION_JOB_NAME, "77", 16), (CONTROLLER_EXTRACTION_JOB_NAME, "77", 16)],
    )

    # Sixty-four messages sit well below either library's parallel-extraction threshold, so neither job opens a pool
    # and neither holds the sixteen cores this call declared for it.
    assert estimates[CAMERA_EXTRACTION_JOB_NAME, "77"] == camera_library_footprint(archive=archive)
    assert estimates[CAMERA_EXTRACTION_JOB_NAME, "77"].cores == 1
    assert estimates[CONTROLLER_EXTRACTION_JOB_NAME, "77"] == controller_library_footprint(archive=archive)
    assert estimates[CONTROLLER_EXTRACTION_JOB_NAME, "77"].cores == 1


def test_an_archive_above_the_parallel_threshold_earns_each_librarys_declared_width(
    experiment_session: SessionData, write_log_archive: Callable[..., Path]
) -> None:
    """Verifies that an archive dense enough to repay a pool earns the allocation declared by the library that owns
    the stage, and the memory that comes back is the memory of that width rather than of the width this package
    requested.
    """
    archive = write_log_archive(
        experiment_session.raw_data.behavior_data_path.joinpath("77_log.npz"),
        77,
        [(index, b"ab") for index in range(_PARALLEL_ARCHIVE_MESSAGES)],
    )

    estimates = size_session_jobs(
        pipeline=ProcessingPipelines.VIDEO,
        session=experiment_session,
        jobs=[(CAMERA_EXTRACTION_JOB_NAME, "77", 1), (CONTROLLER_EXTRACTION_JOB_NAME, "77", 1)],
    )

    assert estimates[CAMERA_EXTRACTION_JOB_NAME, "77"] == camera_library_footprint(archive=archive)
    assert estimates[CAMERA_EXTRACTION_JOB_NAME, "77"].cores == CAMERA_EXTRACTION_JOB_CORES
    assert estimates[CONTROLLER_EXTRACTION_JOB_NAME, "77"] == controller_library_footprint(archive=archive)
    assert estimates[CONTROLLER_EXTRACTION_JOB_NAME, "77"].cores == CONTROLLER_EXTRACTION_JOB_CORES


def test_a_parse_estimate_follows_the_archive_of_its_own_controller(
    experiment_session: SessionData, write_log_archive: Callable[..., Path]
) -> None:
    """Verifies each parse job is charged the archive of the controller that recorded its own module, rather than the
    widest archive the session holds.
    """
    behavior = experiment_session.raw_data.behavior_data_path
    wider = write_sized_archive(path=behavior.joinpath("51_log.npz"), size_bytes=_WIDE_ARCHIVE_BYTES)
    owned = write_log_archive(path=behavior.joinpath("52_log.npz"), source_id=52, messages=[(1, b"a")])

    estimates = size_session_jobs(
        pipeline=ProcessingPipelines.MICROCONTROLLER,
        session=experiment_session,
        jobs=[(PARSE_JOB_NAME, "51-1-1", 1), (PARSE_JOB_NAME, "52-1-1", 1)],
    )

    # The two archives report different whole gigabytes, so the module of the narrow controller is charged the narrow
    # figure while its wide-archive neighbour is charged the wide one. A model reading either archive for both jobs
    # would move one of the two.
    assert wider.stat().st_size > owned.stat().st_size
    assert estimates[PARSE_JOB_NAME, "51-1-1"] == JobFootprint(cores=1, memory_mb=_WIDE_ARCHIVE_MEMORY_MB)
    assert estimates[PARSE_JOB_NAME, "52-1-1"] == JobFootprint(cores=1, memory_mb=_NARROW_ARCHIVE_MEMORY_MB)


def test_every_parse_job_of_one_controller_carries_the_bound_they_share(
    experiment_session: SessionData,
) -> None:
    """Verifies that the parse jobs of one controller are each charged that controller's whole archive rather than a
    share of it, which is the shared bound the model states and not a per-module estimate.
    """
    behavior = experiment_session.raw_data.behavior_data_path
    write_sized_archive(path=behavior.joinpath("51_log.npz"), size_bytes=_WIDE_ARCHIVE_BYTES)

    shared = size_session_jobs(
        pipeline=ProcessingPipelines.MICROCONTROLLER,
        session=experiment_session,
        jobs=[(PARSE_JOB_NAME, "51-2-1", 1), (PARSE_JOB_NAME, "51-4-1", 1)],
    )
    alone = size_session_jobs(
        pipeline=ProcessingPipelines.MICROCONTROLLER,
        session=experiment_session,
        jobs=[(PARSE_JOB_NAME, "51-2-1", 1)],
    )

    # Two modules of one controller are charged one figure between them, and it is the whole archive rather than a
    # share of it: the same module sized as its controller's only parse job receives exactly the same figure, so
    # nothing divides the archive among the jobs that read it.
    assert shared[PARSE_JOB_NAME, "51-2-1"] == shared[PARSE_JOB_NAME, "51-4-1"]
    assert shared[PARSE_JOB_NAME, "51-2-1"] == alone[PARSE_JOB_NAME, "51-2-1"]

    # The archive stands a whole gigabyte above the floor, so a model dividing it between the two jobs would report a
    # different figure here rather than the same one after rounding.
    assert shared[PARSE_JOB_NAME, "51-2-1"].memory_mb == _WIDE_ARCHIVE_MEMORY_MB


def test_a_parse_estimate_is_refused_when_its_controller_recorded_no_archive(
    experiment_session: SessionData, write_log_archive: Callable[..., Path]
) -> None:
    """Verifies that a module whose controller recorded no archive states nothing with which the stage scales, so the
    job is refused rather than planned at a guess.
    """
    write_log_archive(
        path=experiment_session.raw_data.behavior_data_path.joinpath("51_log.npz"),
        source_id=51,
        messages=[(1, b"a")],
    )

    with pytest.raises(FileNotFoundError, match="Unable to find the log archive of source '52'"):
        size_session_jobs(
            pipeline=ProcessingPipelines.MICROCONTROLLER,
            session=experiment_session,
            jobs=[(PARSE_JOB_NAME, "52-1-1", 1)],
        )


def test_a_parse_estimate_is_refused_when_the_behavior_directory_is_absent(
    experiment_session: SessionData,
) -> None:
    """Verifies that a session that recorded no behavior data carries no directory to search, which is refused the same
    way.
    """
    assert not experiment_session.raw_data.behavior_data_path.is_dir()

    with pytest.raises(FileNotFoundError):
        size_session_jobs(
            pipeline=ProcessingPipelines.MICROCONTROLLER,
            session=experiment_session,
            jobs=[(PARSE_JOB_NAME, "52-1-1", 1)],
        )


def test_each_motion_energy_job_is_charged_the_recording_of_its_own_camera(
    experiment_session: SessionData,
    write_grayscale_video: Callable[..., Path],
    write_dlc_predictions: Callable[..., Path],
) -> None:
    """Verifies that a motion-energy job reads only the recording of the camera its specifier names, so it is charged
    that camera's own frame and that camera's own length rather than whatever the session's other camera recorded.

    The two cameras are written so that one plans a single decode chunk while the other plans a pool, which puts the
    two jobs in different reportable gigabytes. A job charged its sibling's recording therefore reports its sibling's
    figure rather than collapsing onto the quantum every estimate is rounded to.
    """
    write_camera_manifest(
        session=experiment_session, cameras={_FACE_SOURCE_ID: _FACE_CAMERA, _BODY_SOURCE_ID: _BODY_CAMERA}
    )
    write_camera_recording(
        session=experiment_session,
        camera=_FACE_CAMERA,
        frames=np.zeros((_SINGLE_CHUNK_FRAME_COUNT, *_LARGE_FRAME_SHAPE), dtype=np.uint8),
        writer=write_grayscale_video,
    )
    write_camera_recording(
        session=experiment_session,
        camera=_BODY_CAMERA,
        frames=np.zeros((_CHUNKED_FRAME_COUNT, *_SMALL_FRAME_SHAPE), dtype=np.uint8),
        writer=write_grayscale_video,
    )
    points: Mapping[str, NDArray[np.float64]] = {"eye_top": np.zeros((32, 3), dtype=np.float64)}
    write_dlc_predictions(
        path=experiment_session.raw_data.camera_data_path.joinpath("51_cameraDLC_eye_tracking.h5"), points=points
    )

    estimates = size_session_jobs(
        pipeline=ProcessingPipelines.VIDEO,
        session=experiment_session,
        jobs=[
            (ENERGY_JOB_NAME, _FACE_SOURCE_ID, _ENERGY_JOB_CORES),
            (ENERGY_JOB_NAME, _BODY_SOURCE_ID, _ENERGY_JOB_CORES),
            (TRACKING_JOB_NAME, "", 1),
            (RENAME_JOB_NAME, "", 1),
        ],
    )

    assert estimates[ENERGY_JOB_NAME, _FACE_SOURCE_ID] == JobFootprint(
        cores=_ENERGY_JOB_CORES,
        memory_mb=energy_memory(
            frame_pixels=_LARGE_FRAME_SHAPE[0] * _LARGE_FRAME_SHAPE[1],
            frame_count=_SINGLE_CHUNK_FRAME_COUNT,
            cores=_ENERGY_JOB_CORES,
        ),
    )
    assert estimates[ENERGY_JOB_NAME, _BODY_SOURCE_ID] == JobFootprint(
        cores=_ENERGY_JOB_CORES,
        memory_mb=energy_memory(
            frame_pixels=_SMALL_FRAME_SHAPE[0] * _SMALL_FRAME_SHAPE[1],
            frame_count=_CHUNKED_FRAME_COUNT,
            cores=_ENERGY_JOB_CORES,
        ),
    )
    # The two jobs land in different reportable gigabytes, so neither figure could have come from the other camera.
    assert (
        estimates[ENERGY_JOB_NAME, _FACE_SOURCE_ID].memory_mb != estimates[ENERGY_JOB_NAME, _BODY_SOURCE_ID].memory_mb
    )
    # The pose prediction is charged the table its own metadata reports rather than the size of the file holding it.
    assert estimates[TRACKING_JOB_NAME, ""] == JobFootprint(
        cores=1,
        memory_mb=_apply_tolerance(
            memory_mb=WORKER_MEMORY_MB
            + _bytes_to_megabytes(byte_count=32 * 3 * _DOUBLE_PRECISION_BYTES * _POSE_TABLE_COPIES)
        ),
    )
    # Renaming performs a fixed handful of filesystem operations and reads no recording, so one worker is the whole
    # model rather than a floor standing in for one.
    assert estimates[RENAME_JOB_NAME, ""] == JobFootprint(cores=1, memory_mb=_WORKER_ONLY_MB)


def test_a_recording_too_short_to_chunk_is_charged_one_decoder_and_no_child(
    experiment_session: SessionData, write_grayscale_video: Callable[..., Path]
) -> None:
    """Verifies that a job whose recording is shorter than one decode chunk is charged the single decoder it opens in
    its own process rather than the decoders its full core allocation would permit.

    The stage opens a pool only when it plans more than one chunk, and it plans one chunk for every recording below
    the shared chunk minimum. A rig recording calibration and training clips beside full sessions runs many such jobs,
    and charging each of them the allocation is most of what those jobs reserve.
    """
    write_camera_manifest(session=experiment_session, cameras={_FACE_SOURCE_ID: _FACE_CAMERA})
    write_camera_recording(
        session=experiment_session,
        camera=_FACE_CAMERA,
        frames=np.zeros((_SINGLE_CHUNK_FRAME_COUNT, *_SMALL_FRAME_SHAPE), dtype=np.uint8),
        writer=write_grayscale_video,
    )

    estimates = size_session_jobs(
        pipeline=ProcessingPipelines.VIDEO,
        session=experiment_session,
        jobs=[(ENERGY_JOB_NAME, _FACE_SOURCE_ID, _CHUNKED_ENERGY_JOB_CORES)],
    )

    assert estimates[ENERGY_JOB_NAME, _FACE_SOURCE_ID].memory_mb == _SINGLE_CHUNK_ENERGY_MEMORY_MB
    # The same job charged one decoder per allocated core reserves six times what the stage goes on to hold.
    assert (
        _size_motion_energy_job(
            recording=_EnergyRecording(
                frame_pixels=_SMALL_FRAME_SHAPE[0] * _SMALL_FRAME_SHAPE[1],
                frame_count=MINIMUM_CHUNK_FRAMES * _CHUNKED_ENERGY_JOB_CORES,
            ),
            cores=_CHUNKED_ENERGY_JOB_CORES,
        ).memory_mb
        == _FULL_CHUNK_ENERGY_MEMORY_MB
    )
    # The stage keeps the allocation it declared whatever chunk plan its recording earns, since the allocation is what
    # bounds that plan.
    assert estimates[ENERGY_JOB_NAME, _FACE_SOURCE_ID].cores == _CHUNKED_ENERGY_JOB_CORES


def test_a_single_chunk_job_is_charged_no_spawned_child() -> None:
    """Verifies that a job planning one decode chunk is charged the decoder it holds in its own process and no spawned
    child, since the stage opens a pool only once it plans more than one chunk.

    A spawned child costs a fraction of the gigabyte every estimate is rounded to, so the frame compared here stands
    at the width where that fraction crosses a boundary rather than at any width a camera records.
    """
    single_chunk = _size_motion_energy_job(
        recording=_EnergyRecording(frame_pixels=_UNCHUNKED_WIDE_FRAME_PIXELS, frame_count=MINIMUM_CHUNK_FRAMES - 1),
        cores=_CHUNKED_ENERGY_JOB_CORES,
    )

    assert single_chunk.memory_mb == _UNCHUNKED_WIDE_MEMORY_MB
    frame_buffers = _bytes_to_megabytes(
        byte_count=_UNCHUNKED_WIDE_FRAME_PIXELS * _SINGLE_PRECISION_BYTES * _RETAINED_FRAME_BUFFERS
    )
    # Charging that job the one child a pool would start reserves a whole gigabyte the job never holds.
    assert (
        _apply_tolerance(
            memory_mb=WORKER_MEMORY_MB + frame_buffers + _DECODER_BUFFER_MEMORY_MB + SPAWNED_CHILD_MEMORY_MB
        )
        == _UNCHUNKED_WIDE_WITH_CHILD_MEMORY_MB
    )
    assert single_chunk.memory_mb < _UNCHUNKED_WIDE_WITH_CHILD_MEMORY_MB


def test_the_decoders_a_motion_energy_job_is_charged_are_the_chunks_the_stage_plans() -> None:
    """Verifies that the model charges the decoders the stage opens rather than restating a chunk rule beside it, by
    reading the chunk count off the stage's own planner for every shape of recording a job can carry.
    """
    for frame_count in (
        MINIMUM_CHUNK_FRAMES - 1,
        MINIMUM_CHUNK_FRAMES,
        MINIMUM_CHUNK_FRAMES * 3,
        MINIMUM_CHUNK_FRAMES * _CHUNKED_ENERGY_JOB_CORES,
        MINIMUM_CHUNK_FRAMES * _CHUNKED_ENERGY_JOB_CORES * 4,
    ):
        chunks = len(_plan_chunks(frame_count=frame_count, workers=_CHUNKED_ENERGY_JOB_CORES))
        frame_buffers = _bytes_to_megabytes(
            byte_count=_NARROW_ENERGY_FRAME_PIXELS * _SINGLE_PRECISION_BYTES * _RETAINED_FRAME_BUFFERS
        )
        # A single chunk decodes in the job's own process, so it holds one decoder and starts no child.
        expected = (
            _apply_tolerance(memory_mb=WORKER_MEMORY_MB + frame_buffers + _DECODER_BUFFER_MEMORY_MB)
            if chunks == 1
            else _apply_tolerance(
                memory_mb=WORKER_MEMORY_MB
                + chunks * (frame_buffers + _DECODER_BUFFER_MEMORY_MB + SPAWNED_CHILD_MEMORY_MB)
            )
        )
        assert (
            _size_motion_energy_job(
                recording=_EnergyRecording(frame_pixels=_NARROW_ENERGY_FRAME_PIXELS, frame_count=frame_count),
                cores=_CHUNKED_ENERGY_JOB_CORES,
            ).memory_mb
            == expected
        )


def test_a_job_whose_recording_fills_every_chunk_is_charged_its_own_frame() -> None:
    """Verifies that the frame term still separates two cameras once a recording is long enough to open every decoder
    the allocation permits, which is the only width at which a frame moves the reported gigabyte at all.
    """
    long_enough = MINIMUM_CHUNK_FRAMES * _FULL_WIDTH_ENERGY_CORES

    wide = _size_motion_energy_job(
        recording=_EnergyRecording(frame_pixels=_WIDE_ENERGY_FRAME_PIXELS, frame_count=long_enough),
        cores=_FULL_WIDTH_ENERGY_CORES,
    )
    narrow = _size_motion_energy_job(
        recording=_EnergyRecording(frame_pixels=_NARROW_ENERGY_FRAME_PIXELS, frame_count=long_enough),
        cores=_FULL_WIDTH_ENERGY_CORES,
    )

    assert wide.memory_mb == _WIDE_ENERGY_MEMORY_MB
    assert narrow.memory_mb == _NARROW_ENERGY_MEMORY_MB


def test_a_motion_energy_job_whose_camera_the_manifest_omits_is_charged_its_full_width(
    experiment_session: SessionData, write_grayscale_video: Callable[..., Path]
) -> None:
    """Verifies that a specifier the camera manifest does not register is charged no frame and the full width its
    allocation permits, rather than refused, since the stage skips such a camera and completes.

    No container was read for that specifier, so nothing states how many chunks the stage would plan, and the width
    the allocation permits is the only figure that cannot understate the job. Its sibling recorded, and the two report
    different reportable gigabytes, so a job charged the session's recordings rather than its own camera's would
    report that recording's figure here.
    """
    write_camera_manifest(session=experiment_session, cameras={_FACE_SOURCE_ID: _FACE_CAMERA})
    write_camera_recording(
        session=experiment_session,
        camera=_FACE_CAMERA,
        frames=np.zeros((_SINGLE_CHUNK_FRAME_COUNT, *_LARGE_FRAME_SHAPE), dtype=np.uint8),
        writer=write_grayscale_video,
    )

    estimates = size_session_jobs(
        pipeline=ProcessingPipelines.VIDEO,
        session=experiment_session,
        jobs=[
            (ENERGY_JOB_NAME, _FACE_SOURCE_ID, _ENERGY_JOB_CORES),
            (ENERGY_JOB_NAME, _BODY_SOURCE_ID, _ENERGY_JOB_CORES),
        ],
    )

    assert estimates[ENERGY_JOB_NAME, _BODY_SOURCE_ID].memory_mb == energy_memory(
        frame_pixels=0, frame_count=_UNRESOLVED_FRAME_COUNT, cores=_ENERGY_JOB_CORES
    )
    assert (
        estimates[ENERGY_JOB_NAME, _BODY_SOURCE_ID].memory_mb != estimates[ENERGY_JOB_NAME, _FACE_SOURCE_ID].memory_mb
    )


def test_a_registered_camera_that_left_no_recording_is_charged_no_frame(
    experiment_session: SessionData, write_grayscale_video: Callable[..., Path]
) -> None:
    """Verifies that a camera the manifest registers but which recorded nothing is charged its decoders alone, since
    the stage reports such a camera as skipped and completes rather than failing.

    Its sibling camera did record, so the figure states that the absent camera was resolved on its own name rather
    than falling back to whatever recording the directory happened to hold.
    """
    write_camera_manifest(
        session=experiment_session, cameras={_FACE_SOURCE_ID: _FACE_CAMERA, _BODY_SOURCE_ID: _BODY_CAMERA}
    )
    write_camera_recording(
        session=experiment_session,
        camera=_FACE_CAMERA,
        frames=np.zeros((_SINGLE_CHUNK_FRAME_COUNT, *_LARGE_FRAME_SHAPE), dtype=np.uint8),
        writer=write_grayscale_video,
    )

    estimates = size_session_jobs(
        pipeline=ProcessingPipelines.VIDEO,
        session=experiment_session,
        jobs=[(ENERGY_JOB_NAME, _BODY_SOURCE_ID, _ENERGY_JOB_CORES)],
    )

    assert estimates[ENERGY_JOB_NAME, _BODY_SOURCE_ID].memory_mb == energy_memory(
        frame_pixels=0, frame_count=_UNRESOLVED_FRAME_COUNT, cores=_ENERGY_JOB_CORES
    )


def test_a_video_estimate_charges_the_decoders_when_the_session_recorded_no_camera(
    experiment_session: SessionData,
) -> None:
    """Verifies that a session carrying no camera directory reports no frame, which leaves motion energy on the per-core
    decoder and child cost that its model charges whatever the recording holds.
    """
    estimates = size_session_jobs(
        pipeline=ProcessingPipelines.VIDEO, session=experiment_session, jobs=[(ENERGY_JOB_NAME, "51", 16)]
    )

    energy = estimates[ENERGY_JOB_NAME, "51"]
    # No frame contributes no pixels, so the estimate covers the decoder buffers and the children alone.
    assert energy.memory_mb == _apply_tolerance(
        memory_mb=WORKER_MEMORY_MB + 16 * (_DECODER_BUFFER_MEMORY_MB + SPAWNED_CHILD_MEMORY_MB)
    )


def test_a_pose_estimate_is_charged_the_table_rather_than_the_file_holding_it(
    experiment_session: SessionData, write_dlc_predictions: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that the pose estimate follows the table's own row and column counts, so a prediction whose writer
    compressed it is charged what the stage expands in memory rather than what it occupies on disk.

    The two files below hold the identical table and differ only in compression, and the compressed one is small
    enough that a model charging its size on disk reports three whole gigabytes less than the stage goes on to hold.
    An under-reserved job is killed by the scheduler, so this is the direction the estimate may never take.
    """
    points: Mapping[str, NDArray[np.float64]] = {
        f"point_{index}": np.zeros((_POSE_TABLE_ROWS, _POSE_COORDINATES_PER_BODYPART), dtype=np.float64)
        for index in range(_POSE_TABLE_BODYPARTS)
    }
    camera_directory = experiment_session.raw_data.camera_data_path
    uncompressed = write_dlc_predictions(path=camera_directory.joinpath("51_cameraDLC_eye_tracking.h5"), points=points)
    compressed = write_dlc_predictions(
        path=camera_directory.joinpath("compressed.h5"), points=points, compression_level=_POSE_COMPRESSION_LEVEL
    )

    # The raised copies term is what puts a table this small in a bucket the two candidate models cannot share.
    monkeypatch.setattr(footprints_module, "_POSE_TABLE_COPIES", _POSE_TABLE_COPIES_OVERRIDE)

    uncompressed_estimate = size_session_jobs(
        pipeline=ProcessingPipelines.VIDEO, session=experiment_session, jobs=[(TRACKING_JOB_NAME, "", 1)]
    )
    uncompressed.unlink()
    compressed.rename(target=camera_directory.joinpath("51_cameraDLC_eye_tracking.h5"))
    compressed_estimate = size_session_jobs(
        pipeline=ProcessingPipelines.VIDEO, session=experiment_session, jobs=[(TRACKING_JOB_NAME, "", 1)]
    )

    columns = _POSE_TABLE_BODYPARTS * _POSE_COORDINATES_PER_BODYPART
    assert uncompressed_estimate[TRACKING_JOB_NAME, ""].memory_mb == _POSE_TABLE_MEMORY_MB
    # The same table read out of a file a fraction of the size reports the same figure, because the stage holds the
    # table rather than the file.
    assert compressed_estimate[TRACKING_JOB_NAME, ""].memory_mb == _POSE_TABLE_MEMORY_MB
    assert (
        _apply_tolerance(
            memory_mb=WORKER_MEMORY_MB
            + _bytes_to_megabytes(
                byte_count=_POSE_TABLE_ROWS * columns * _DOUBLE_PRECISION_BYTES * _POSE_TABLE_COPIES_OVERRIDE
            )
        )
        == _POSE_TABLE_MEMORY_MB
    )
    # A model charging the compressed file's size on disk lands three gigabytes below the memory the stage holds.
    assert (
        _apply_tolerance(
            memory_mb=WORKER_MEMORY_MB
            + _bytes_to_megabytes(
                byte_count=camera_directory.joinpath("51_cameraDLC_eye_tracking.h5").stat().st_size
                * _POSE_TABLE_COPIES_OVERRIDE
            )
        )
        == _POSE_COMPRESSED_FILE_MEMORY_MB
    )


def test_a_pose_table_reports_one_shape_whatever_its_writer_compressed(
    tmp_path: Path, write_dlc_predictions: Callable[..., Path]
) -> None:
    """Verifies that the row and column counts come from the table's own metadata, which compression does not move."""
    points: Mapping[str, NDArray[np.float64]] = {
        f"point_{index}": np.zeros((_POSE_TABLE_ROWS, _POSE_COORDINATES_PER_BODYPART), dtype=np.float64)
        for index in range(_POSE_TABLE_BODYPARTS)
    }
    uncompressed = write_dlc_predictions(path=tmp_path.joinpath("plain.h5"), points=points)
    compressed = write_dlc_predictions(
        path=tmp_path.joinpath("packed.h5"), points=points, compression_level=_POSE_COMPRESSION_LEVEL
    )

    shape = (_POSE_TABLE_ROWS, _POSE_TABLE_BODYPARTS * _POSE_COORDINATES_PER_BODYPART)
    assert _read_pose_table_shape(prediction=uncompressed) == shape
    assert _read_pose_table_shape(prediction=compressed) == shape
    # The files themselves differ by an order of magnitude, which is exactly what the shape read does not inherit.
    assert compressed.stat().st_size < uncompressed.stat().st_size


def test_a_pose_prediction_stating_no_table_width_is_refused(tmp_path: Path) -> None:
    """Verifies that a prediction whose frame reports no row and column count is refused rather than charged a figure
    taken from the size of the file, which is the read this model exists to replace.
    """
    prediction = tmp_path.joinpath("fixed_layout.h5")
    columns = pd.MultiIndex.from_tuples(
        [("scorer", "eye_top", coordinate) for coordinate in ("x", "y", "likelihood")],
        names=["scorer", "bodyparts", "coords"],
    )
    pd.DataFrame(data=np.zeros((8, 3), dtype=np.float64), columns=columns).to_hdf(
        path_or_buf=prediction, key="df_with_missing", format="fixed"
    )

    with pytest.raises(ValueError, match="reports no row") as failure:
        _read_pose_table_shape(prediction=prediction)

    assert "under-reserve the job" in " ".join(str(failure.value).split())


def test_a_pose_prediction_holding_several_tables_is_refused(tmp_path: Path) -> None:
    """Verifies that a prediction holding more than one table is refused, since the stage reads it through a call that
    names no key and that call requires the file to hold exactly one.
    """
    prediction = tmp_path.joinpath("two_tables.h5")
    columns = pd.MultiIndex.from_tuples(
        [("scorer", "eye_top", coordinate) for coordinate in ("x", "y", "likelihood")],
        names=["scorer", "bodyparts", "coords"],
    )
    frame = pd.DataFrame(data=np.zeros((8, 3), dtype=np.float64), columns=columns)
    frame.to_hdf(path_or_buf=prediction, key="df_with_missing", format="table")
    frame.to_hdf(path_or_buf=prediction, key="second_pass", format="table", append=False)

    with pytest.raises(ValueError, match="must hold exactly one table") as failure:
        _read_pose_table_shape(prediction=prediction)

    assert "but it holds 2" in " ".join(str(failure.value).split())


def test_a_pose_estimate_is_refused_when_the_session_carries_no_prediction(
    experiment_session: SessionData,
) -> None:
    """Verifies that a session holding no pose prediction states nothing with which the stage scales, so the job is
    refused rather than planned at a guess.
    """
    with pytest.raises(FileNotFoundError, match="carries no pose prediction"):
        size_session_jobs(
            pipeline=ProcessingPipelines.VIDEO, session=experiment_session, jobs=[(TRACKING_JOB_NAME, "51", 1)]
        )


def test_a_camera_directory_holding_no_recording_reports_no_frame(experiment_session: SessionData) -> None:
    """Verifies that an empty camera directory is read the same way an absent one is, which is as no recorded frame at
    all.
    """
    experiment_session.raw_data.camera_data_path.mkdir(parents=True, exist_ok=True)

    estimates = size_session_jobs(
        pipeline=ProcessingPipelines.VIDEO, session=experiment_session, jobs=[(ENERGY_JOB_NAME, "51", 4)]
    )

    assert estimates[ENERGY_JOB_NAME, "51"].memory_mb == _apply_tolerance(
        memory_mb=WORKER_MEMORY_MB + 4 * (_DECODER_BUFFER_MEMORY_MB + SPAWNED_CHILD_MEMORY_MB)
    )


def test_a_session_job_naming_a_stage_nothing_models_is_refused(experiment_session: SessionData) -> None:
    """Verifies that every job of a session is routed to a model of its own, so a name reaching the end of that routing
    describes a stage nothing sizes and is refused rather than admitted to a batch at an allowance nobody chose.
    """
    experiment_session.raw_data.behavior_data_path.mkdir(parents=True, exist_ok=True)

    # Matches the unwrapped opening of the message, since the console formatter wraps long lines.
    with pytest.raises(ValueError, match="Unable to size job") as failure:
        size_session_jobs(
            pipeline=ProcessingPipelines.RUNTIME,
            session=experiment_session,
            jobs=[("a_stage_no_pipeline_declares", "51", 4)],
        )

    assert "routes to no sizing model" in " ".join(str(failure.value).split())


# Two-photon estimates


def test_two_photon_stages_are_sized_from_the_raw_recording_geometry(experiment_session: SessionData) -> None:
    """Verifies that each stage reads the shape its own working set follows, which the raw acquisition data reports."""
    write_surgery_metadata(session=experiment_session)
    write_raw_imaging(session=experiment_session)

    estimates = two_photon_estimates(
        session=experiment_session,
        jobs=[
            (str(SingleRecordingJobNames.BINARIZE), "", 4),
            (str(SingleRecordingJobNames.COMBINE), "", 1),
            (str(SingleRecordingJobNames.REGISTER), f"{PLANE_SPECIFIER_PREFIX}0", 8),
            (str(SingleRecordingJobNames.REGISTER), f"{PLANE_SPECIFIER_PREFIX}1", 8),
            (str(SingleRecordingJobNames.PROCESS), f"{PLANE_SPECIFIER_PREFIX}0", 10),
        ],
    )

    # Every two-photon stage belongs to cindra, so both halves of each figure are cindra's own sizing pass, with the
    # memory rounded to the reportable gigabyte rather than projected again here.
    for job_name, specifier in (
        (SingleRecordingJobNames.BINARIZE, ""),
        (SingleRecordingJobNames.COMBINE, ""),
        (SingleRecordingJobNames.REGISTER, f"{PLANE_SPECIFIER_PREFIX}0"),
        (SingleRecordingJobNames.REGISTER, f"{PLANE_SPECIFIER_PREFIX}1"),
        (SingleRecordingJobNames.PROCESS, f"{PLANE_SPECIFIER_PREFIX}0"),
    ):
        assert estimates[str(job_name), specifier] == cindra_single_recording_footprint(
            session=experiment_session, job_name=job_name, specifier=specifier
        )
    # cindra picks each stage's width itself, so the allocation this call declared does not survive the sizing pass.
    assert estimates[str(SingleRecordingJobNames.BINARIZE), ""].cores != 4


def test_a_plane_job_whose_specifier_names_no_plane_takes_the_widest_plane(
    experiment_session: SessionData,
) -> None:
    """Verifies a specifier naming no index at all is charged the largest per-plane figure, so it never understates."""
    write_surgery_metadata(session=experiment_session)
    write_raw_imaging(session=experiment_session)

    estimates = two_photon_estimates(
        session=experiment_session,
        jobs=[
            (str(SingleRecordingJobNames.REGISTER), f"{PLANE_SPECIFIER_PREFIX}1", 8),
            (str(SingleRecordingJobNames.REGISTER), "an_unstructured_specifier", 8),
            (str(SingleRecordingJobNames.PROCESS), f"{PLANE_SPECIFIER_PREFIX}x", 10),
            (str(SingleRecordingJobNames.PROCESS), f"{PLANE_SPECIFIER_PREFIX}1", 10),
        ],
    )

    widest_registration = cindra_single_recording_footprint(
        session=experiment_session, job_name=SingleRecordingJobNames.REGISTER, specifier=""
    )
    widest_processing = cindra_single_recording_footprint(
        session=experiment_session, job_name=SingleRecordingJobNames.PROCESS, specifier=""
    )
    assert estimates[str(SingleRecordingJobNames.REGISTER), "an_unstructured_specifier"] == widest_registration
    # A specifier that does name a plane held by the recording is charged that plane alone, not the widest one.
    assert estimates[
        str(SingleRecordingJobNames.REGISTER), f"{PLANE_SPECIFIER_PREFIX}1"
    ] == cindra_single_recording_footprint(
        session=experiment_session,
        job_name=SingleRecordingJobNames.REGISTER,
        specifier=f"{PLANE_SPECIFIER_PREFIX}1",
    )
    # A prefix carrying no readable index names no plane either, so it is charged the same widest figure.
    assert estimates[str(SingleRecordingJobNames.PROCESS), f"{PLANE_SPECIFIER_PREFIX}x"] == widest_processing


def test_a_plane_job_naming_a_plane_the_recording_does_not_hold_is_refused(experiment_session: SessionData) -> None:
    """Verifies that cindra refuses to size a plane the recording never held, and that refusal reaches the caller
    unchanged.
    """
    write_surgery_metadata(session=experiment_session)
    write_raw_imaging(session=experiment_session)

    with pytest.raises(ValueError, match="does not hold"):
        two_photon_estimates(
            session=experiment_session,
            jobs=[(str(SingleRecordingJobNames.REGISTER), f"{PLANE_SPECIFIER_PREFIX}99", 8)],
        )


def test_a_recording_declaring_no_regions_is_read_as_one_full_frame_plane(experiment_session: SessionData) -> None:
    """Verifies acquisition parameters naming no region span leave one plane covering the whole acquisition frame."""
    write_surgery_metadata(session=experiment_session)
    write_raw_imaging(session=experiment_session, region_lines=None)

    estimates = two_photon_estimates(
        session=experiment_session, jobs=[(str(SingleRecordingJobNames.REGISTER), f"{PLANE_SPECIFIER_PREFIX}0", 8)]
    )

    assert estimates[
        str(SingleRecordingJobNames.REGISTER), f"{PLANE_SPECIFIER_PREFIX}0"
    ] == cindra_single_recording_footprint(
        session=experiment_session,
        job_name=SingleRecordingJobNames.REGISTER,
        specifier=f"{PLANE_SPECIFIER_PREFIX}0",
    )


def test_two_photon_jobs_are_refused_when_the_session_holds_no_raw_imaging(experiment_session: SessionData) -> None:
    """Verifies that a session with no imaging directory names no geometry, and a recording that cindra will not size
    is a recording whose stages cannot run.
    """
    write_surgery_metadata(session=experiment_session)

    with pytest.raises(FileNotFoundError, match="Neither of its two inputs resolved"):
        two_photon_estimates(
            session=experiment_session,
            jobs=[(str(SingleRecordingJobNames.BINARIZE), "", 4), (str(SingleRecordingJobNames.COMBINE), "", 1)],
        )


def test_two_photon_jobs_are_refused_when_the_imaging_directory_holds_no_parameters(
    experiment_session: SessionData,
) -> None:
    """Verifies an imaging directory carrying no acquisition parameters reports no shape for scaling an estimate."""
    write_surgery_metadata(session=experiment_session)
    experiment_session.raw_data_path.joinpath("mesoscope_data").mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="Neither of its two inputs resolved"):
        two_photon_estimates(session=experiment_session, jobs=[(str(SingleRecordingJobNames.BINARIZE), "", 4)])


def test_two_photon_jobs_are_refused_when_every_image_is_excluded(experiment_session: SessionData) -> None:
    """Verifies that the reader skips the names the configuration excludes, so a directory holding only those reports no
    image.
    """
    write_surgery_metadata(session=experiment_session)
    directory = experiment_session.raw_data_path.joinpath("mesoscope_data")
    write_acquisition_parameters(directory=directory, region_lines=_REGION_LINES)
    write_imaging_stack(directory=directory, name="zstack.tif", pages=8)

    # The parameters were read here, so the refusal names the frames the conversion could not count rather than the
    # unresolved inputs an unreadable recording reports.
    with pytest.raises(FileNotFoundError, match="so the frames its conversion reads cannot be counted"):
        two_photon_estimates(session=experiment_session, jobs=[(str(SingleRecordingJobNames.BINARIZE), "", 4)])


def test_two_photon_jobs_are_refused_when_the_acquisition_parameters_cannot_be_read(
    experiment_session: SessionData,
) -> None:
    """Verifies that a parameters file rejected by the reader names no geometry, and that refusal propagates to the
    caller sizing the batch rather than resolving to a figure nothing measured.
    """
    write_surgery_metadata(session=experiment_session)
    directory = experiment_session.raw_data_path.joinpath("mesoscope_data")
    write_raw_imaging(session=experiment_session)
    directory.joinpath(PARAMETERS_FILENAME).write_text("{not valid json")

    with pytest.raises(ValueError, match="double quotes"):
        two_photon_estimates(session=experiment_session, jobs=[(str(SingleRecordingJobNames.BINARIZE), "", 4)])


# Dataset estimates


def test_dataset_stages_are_sized_from_the_processed_recordings_they_read(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies that discovery, extraction, and assembly each scale with the single-recording output their own stage
    consumes.
    """
    first = session_factory(animal_id="305", experiment_name="test_experiment")
    second = session_factory(animal_id="305", experiment_name="test_experiment")
    for session in (first, second):
        write_surgery_metadata(session=session)
        write_processed_recording(session=session, regions=200, samples=4000)
    dataset = build_dataset(
        project_root=project_root,
        name="ds_processed",
        sessions=[first, second],
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
    )

    estimates = size_dataset_jobs(
        dataset=dataset,
        jobs=[
            (MULTIDAY_DISCOVERY_JOB_NAME, "305", 30),
            (MULTIDAY_EXTRACTION_JOB_NAME, first.session_name, 16),
            (FORGING_JOB_NAME, first.session_name, 1),
        ],
    )

    # Both cross-recording stages belong to cindra, so both halves of each figure are cindra's own sizing pass over
    # the animal's whole recording set, with the memory rounded to the reportable gigabyte.
    assert estimates[MULTIDAY_DISCOVERY_JOB_NAME, "305"] == cindra_multi_recording_footprint(
        dataset=dataset, sessions=[first, second], job_name=MultiRecordingJobNames.DISCOVER, specifier="305"
    )
    assert estimates[MULTIDAY_EXTRACTION_JOB_NAME, first.session_name] == cindra_multi_recording_footprint(
        dataset=dataset,
        sessions=[first, second],
        job_name=MultiRecordingJobNames.EXTRACT,
        specifier=first.session_name,
    )
    # cindra picks each stage's width itself, so the allocation this call declared does not survive the sizing pass.
    assert estimates[MULTIDAY_DISCOVERY_JOB_NAME, "305"].cores != 30
    # The per-session assembly is this package's own stage, so its projection stays here and it keeps the allocation
    # it was handed. Half prevalence over two recordings keeps a cluster appearing in one of them, so the four hundred
    # pooled regions bound the templates at four hundred rather than at either recording's own two hundred.
    assert estimates[FORGING_JOB_NAME, first.session_name] == JobFootprint(
        cores=1, memory_mb=assembly_memory(samples=4000, regions=200, tracked_regions=400)
    )


def test_the_assembly_estimate_charges_the_assembled_frame_a_single_time(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies that the write that closes an assembly job streams the frame it was handed rather than rebuilding it, so
    the stage peaks at the columns the assembly already holds.

    The reportable figure is rounded up to the whole gigabyte, which leaves a recording of the scale used by the
    other dataset tests unable to tell one copy of its fluorescence from two. This recording is therefore written
    large enough that a second copy of its retained columns would push the reported estimate from one whole gigabyte
    to two, which is the scale at which the copy count becomes visible.
    """
    session = session_factory(animal_id="305", experiment_name="test_experiment")
    write_surgery_metadata(session=session)
    write_processed_recording(session=session, regions=48, samples=150_000)
    dataset = build_dataset(
        project_root=project_root,
        name="ds_one_copy",
        sessions=[session],
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
    )

    estimates = size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, session.session_name, 1)])

    assert estimates[FORGING_JOB_NAME, session.session_name] == JobFootprint(
        cores=1, memory_mb=assembly_memory(samples=150_000, regions=48)
    )


def test_an_imaging_assembly_is_charged_the_sources_it_reads_at_their_own_heights(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies that the assembly of an imaging session is charged its behavior sources at the heights those sources
    stand at, rather than at the fluorescence clock its assembled frame is placed on.

    The assembler places every column on the fluorescence clock, but it reads each source at that source's own rate
    first and interpolates onto that clock afterwards, and it reads the sources together rather than one at a time.
    A camera running far faster than the imaging therefore holds an array far taller than the frame it lands in, and
    charging that array at the imaging rate reserves a fraction of what the job holds.

    The four million samples the camera clock here holds carry the estimate a whole gigabyte bucket above what the
    same session reports with no source at all, so the reported figure names the source family as its own term
    rather than one the fluorescence clock already covered.
    """
    session = session_factory(animal_id="305", experiment_name="test_experiment")
    write_surgery_metadata(session=session)
    write_processed_recording(session=session, regions=48, samples=20_000)
    write_camera_clock(session=session, camera="face_camera", frames=4_000_000, period_us=1)
    dataset = build_dataset(
        project_root=project_root,
        name="ds_imaging_sources",
        sessions=[session],
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
    )

    estimates = size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, session.session_name, 1)])

    assert estimates[FORGING_JOB_NAME, session.session_name] == JobFootprint(
        cores=1, memory_mb=assembly_memory(samples=20_000, regions=48, source_samples=(4_000_000,))
    )
    # Charging the same session no source family at all lands a whole gigabyte lower, so the assertion above could
    # not have been met by an estimate that omitted the term.
    assert assembly_memory(samples=20_000, regions=48, source_samples=(4_000_000,)) != assembly_memory(
        samples=20_000, regions=48
    )


def test_dataset_stages_are_refused_for_a_session_carrying_no_processed_output(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies that a session with no single-recording output names no geometry for scaling any stage, so every
    stage that would read it is refused rather than planned at a floor.
    """
    session = session_factory(animal_id="305", experiment_name="test_experiment")
    write_surgery_metadata(session=session)
    dataset = build_dataset(
        project_root=project_root,
        name="ds_unprocessed",
        sessions=[session],
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
    )

    # cindra refuses a recording set carrying no processed output rather than sizing it from a guess, and the
    # assembly stage has no fluorescence of its own to read either.
    for job, refusal in (
        ((MULTIDAY_DISCOVERY_JOB_NAME, "305", 30), "carry no combined"),
        ((MULTIDAY_EXTRACTION_JOB_NAME, session.session_name, 16), "carry no combined"),
        ((FORGING_JOB_NAME, session.session_name, 1), "carries no processed imaging"),
    ):
        with pytest.raises(FileNotFoundError, match=refusal):
            size_dataset_jobs(dataset=dataset, jobs=[job])


def test_extraction_is_refused_for_a_dataset_its_system_performs_no_tracking_for(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies that mesoscope-VR tracks cells across experiment sessions alone, so a training dataset resolves no
    configuration, and that the assembly of such a dataset is sized from the clock its own assembler reads.

    A training session joins a dataset without completing the two-photon pipeline, so its assembly attaches no
    fluorescence column even where a stray single-recording output happens to sit beside it. The admission policy
    therefore selects the model rather than standing in for it when imaging is absent, which this session pins by
    carrying a processed recording that the estimate must leave out of its figure.
    """
    session = session_factory(animal_id="321", session_type=SessionTypes.RUN_TRAINING)
    write_surgery_metadata(session=session)
    write_processed_recording(session=session, regions=64, samples=1000)
    write_camera_clock(
        session=session, camera="face_camera", frames=_WIDE_CLOCK_FRAMES, period_us=_WIDE_CLOCK_PERIOD_US
    )
    dataset = build_dataset(
        project_root=project_root, name="ds_training", sessions=[session], session_type=SessionTypes.RUN_TRAINING
    )

    # Extraction is sized against the configuration, so a dataset donating none names no figure for the stage.
    with pytest.raises(ValueError, match="resolved no multi-recording"):
        size_dataset_jobs(dataset=dataset, jobs=[(MULTIDAY_EXTRACTION_JOB_NAME, session.session_name, 16)])

    estimates = size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, session.session_name, 1)])

    # Assembly needs no configuration, and this session type records no imaging, so the figure holds the camera
    # data alone and charges none of the fluorescence the stray recording output reports. The stray
    # recording is small enough to report a single gigabyte through the imaging model, while the camera clock reports
    # three through this one, so a figure read off that recording could not pass this assertion.
    assert estimates[FORGING_JOB_NAME, session.session_name] == JobFootprint(cores=1, memory_mb=_WIDE_CLOCK_MEMORY_MB)


def test_the_assembly_of_a_session_recording_no_imaging_is_sized_from_the_clock_its_assembler_settles_on(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies that a session whose type joins a dataset without imaging is sized from the reference clock its own
    acquisition system's assembler settles on, which is the slowest camera's.

    The two clocks span the same duration at different rates, so the slower camera records the fewer samples. The
    faster camera's clock is the wider one, and the two are written far enough apart to report different whole
    gigabytes, so a figure taken from the clock the assembler would not settle on reports the wide figure rather than
    collapsing onto the quantum every estimate is rounded to.
    """
    session = session_factory(animal_id="321", session_type=SessionTypes.RUN_TRAINING)
    write_camera_clock(
        session=session, camera="face_camera", frames=_WIDE_CLOCK_FRAMES, period_us=_WIDE_CLOCK_PERIOD_US
    )
    write_camera_clock(
        session=session, camera="body_camera", frames=_NARROW_CLOCK_FRAMES, period_us=_NARROW_CLOCK_PERIOD_US
    )
    dataset = build_dataset(
        project_root=project_root, name="ds_clocked", sessions=[session], session_type=SessionTypes.RUN_TRAINING
    )

    estimates = size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, session.session_name, 1)])

    # No fluorescence column is attached at all, so the sub-dataset term the settled clock implies is the whole
    # data-dependent charge the stage carries above its worker. The faster camera's clock reports a gigabyte more, so
    # an estimate bounded by the widest clock the session recorded reports the wide figure and fails here.
    assert _WIDE_CLOCK_MEMORY_MB != _NARROW_CLOCK_MEMORY_MB
    assert estimates[FORGING_JOB_NAME, session.session_name] == JobFootprint(cores=1, memory_mb=_NARROW_CLOCK_MEMORY_MB)


def test_the_assembly_of_a_session_recording_no_imaging_is_sized_from_its_frame_and_from_its_sources(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies that the estimate charges both families of arrays the assembly holds: the frame it builds at the
    reference clock's height, and every source it reads at that source's own height.

    The two families stand at different heights and neither bounds the other. The faster camera writes five times the
    reference clock's samples and the encoder writes twice them, so a figure covering the frame alone reserves a
    fraction of what the job holds, which is what got the job killed. A figure covering the sources alone, one
    counting the cameras and leaving the behavior feathers out, and one placing the frame on the widest clock the
    session recorded rather than on the clock its assembler settles on each land on a different whole gigabyte from
    the one this model reports, so none of them passes here.
    """
    session = session_factory(animal_id="321", session_type=SessionTypes.RUN_TRAINING)
    write_camera_clock(
        session=session, camera="body_camera", frames=_SOURCED_REFERENCE_FRAMES, period_us=_SOURCED_REFERENCE_PERIOD_US
    )
    write_camera_clock(
        session=session, camera="face_camera", frames=_SOURCED_FAST_FRAMES, period_us=_SOURCED_FAST_PERIOD_US
    )
    write_behavior_source(session=session, source_file=BehaviorDataFiles.ENCODER, samples=_SOURCED_BEHAVIOR_SAMPLES)
    dataset = build_dataset(
        project_root=project_root, name="ds_sourced", sessions=[session], session_type=SessionTypes.RUN_TRAINING
    )

    estimates = size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, session.session_name, 1)])

    # Every wrong model lands in its own gigabyte bucket, so the equality below discriminates against each of them
    # rather than collapsing onto the quantum the estimates are rounded to.
    assert (
        len(
            {
                _SOURCED_ASSEMBLY_MEMORY_MB,
                _SOURCED_FRAME_ONLY_MEMORY_MB,
                _SOURCED_WIDEST_CLOCK_MEMORY_MB,
            }
        )
        == 3
    )
    assert _SOURCED_SOURCES_ONLY_MEMORY_MB != _SOURCED_ASSEMBLY_MEMORY_MB
    assert _SOURCED_CAMERAS_ONLY_MEMORY_MB != _SOURCED_ASSEMBLY_MEMORY_MB
    assert estimates[FORGING_JOB_NAME, session.session_name] == JobFootprint(
        cores=1, memory_mb=_SOURCED_ASSEMBLY_MEMORY_MB
    )


def test_a_camera_clock_the_assembler_would_not_settle_on_is_left_out_of_the_reference(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies that the estimate places the assembled frame on exactly the clocks its system's assembler accepts,
    since a clock that assembler refuses states the height of a frame the job never builds.

    The session carries three feathers the video pipeline published. One comes from a camera outside the fixed set the
    assembler reads, and one spans no duration at all, so neither can serve as a reference clock however many rows it
    holds. Both are wider than the one clock that qualifies, so an estimate placing the frame on the rows of every
    published feather reports the wide figure here.

    The two roles a feather plays are separate. The zero-span feather comes from a camera the assembler does read, so
    its rows still stand in the source term even though the frame is not placed on them, while the feather from the
    unread camera stands in neither.
    """
    session = session_factory(animal_id="321", session_type=SessionTypes.RUN_TRAINING)
    write_camera_clock(session=session, camera="left_camera", frames=_WIDE_CLOCK_FRAMES, period_us=1)
    write_camera_clock(session=session, camera="face_camera", frames=_WIDE_CLOCK_FRAMES, period_us=0)
    write_camera_clock(
        session=session, camera="body_camera", frames=_NARROW_CLOCK_FRAMES, period_us=_NARROW_CLOCK_PERIOD_US
    )
    dataset = build_dataset(
        project_root=project_root, name="ds_unreadable", sessions=[session], session_type=SessionTypes.RUN_TRAINING
    )

    estimates = size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, session.session_name, 1)])

    assert estimates[FORGING_JOB_NAME, session.session_name] == JobFootprint(cores=1, memory_mb=_NARROW_CLOCK_MEMORY_MB)


def test_the_assembly_of_a_session_recording_no_imaging_is_refused_without_a_camera_clock(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies that a session recording no imaging and no usable reference clock is refused rather than sized at a
    floor, which is the answer its assembly worker gives for it as well.

    A clock is defined by a mean frame rate, so a feather holding a single frame states none, and a feather written by
    a camera the assembler does not read is not a candidate at all however many frames it holds. Each of those
    sessions is refused on the same terms as one whose cameras wrote no feather.

    The last two sessions pin the two conditions a candidate feather has to meet, each as the only candidate the
    session carries so that dropping either condition changes this outcome. A feather holding no row at all is the
    only input the frame-count condition ever decides: every shorter clock a session can carry already holds one row,
    which spans no duration and is refused on that condition instead. And a feather whose frames all share one
    timestamp is here the only camera of the set that wrote anything, so it would be settled on were its span not
    weighed, whatever rate an unguarded reading assigned it.
    """
    session = session_factory(animal_id="321", session_type=SessionTypes.RUN_TRAINING)
    dataset = build_dataset(
        project_root=project_root, name="ds_clockless", sessions=[session], session_type=SessionTypes.RUN_TRAINING
    )

    with pytest.raises(FileNotFoundError, match="No camera timestamp feather"):
        size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, session.session_name, 1)])

    write_camera_clock(session=session, camera="face_camera", frames=1)

    with pytest.raises(FileNotFoundError, match="No camera timestamp feather"):
        size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, session.session_name, 1)])

    write_camera_clock(session=session, camera="left_camera", frames=_NARROW_CLOCK_FRAMES)

    with pytest.raises(FileNotFoundError, match="No camera timestamp feather"):
        size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, session.session_name, 1)])

    write_camera_clock(session=session, camera="face_camera", frames=0)

    with pytest.raises(FileNotFoundError, match="No camera timestamp feather"):
        size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, session.session_name, 1)])

    write_camera_clock(session=session, camera="face_camera", frames=_NARROW_CLOCK_FRAMES, period_us=0)

    with pytest.raises(FileNotFoundError, match="No camera timestamp feather"):
        size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, session.session_name, 1)])


def test_the_assembly_of_a_dataset_whose_session_type_joins_no_dataset_is_refused(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies that a dataset whose session type the acquisition system's admission policy omits is refused rather
    than sized from the behavior model.

    The policy maps a type to the pipelines it must complete, and a type it omits joins no dataset at all. Reading
    that absence as an empty requirement would size such a dataset as though it were admitted while recording no
    imaging. This pass is public and runs no admission check of its own, so a marker naming an unadmitted type reaches
    it directly and must be refused there. The session carries a usable camera clock, so the refusal can only come
    from the type rather than from data the pass could not read.
    """
    session = session_factory(animal_id="321", session_type=SessionTypes.WINDOW_CHECKING)
    write_camera_clock(session=session, camera="face_camera", frames=2500)
    dataset = build_dataset(
        project_root=project_root,
        name="ds_unadmitted",
        sessions=[session],
        session_type=SessionTypes.WINDOW_CHECKING,
    )

    with pytest.raises(ValueError, match="joins no dataset"):
        size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, session.session_name, 1)])


def test_a_dataset_naming_no_session_resolves_no_tracking_configuration(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies that a dataset holding no session donates no configuration, which leaves its cross-recording stages
    unsizable.
    """
    session = session_factory(animal_id="305", experiment_name="test_experiment")
    write_surgery_metadata(session=session)
    write_processed_recording(session=session, regions=50, samples=800)
    dataset = build_dataset(
        project_root=project_root, name="ds_empty", sessions=[session], session_type=SessionTypes.MESOSCOPE_EXPERIMENT
    )
    dataset.sessions = ()
    dataset.save()
    emptied = DatasetData.load(dataset_path=dataset.dataset_data_path.parent)

    # No session leaves no recording directory to hand cindra and no session from which to resolve a configuration.
    with pytest.raises(ValueError, match="resolved no multi-recording"):
        size_dataset_jobs(dataset=emptied, jobs=[(MULTIDAY_DISCOVERY_JOB_NAME, "305", 30)])


def test_a_complete_recording_set_sizes_and_an_incomplete_one_is_refused(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies that a cross-recording job is sized over the whole recording set the dataset names for its animal, and
    that a planning host holding fewer of those sessions is refused rather than sized over the subset it holds.

    The stage the job runs reads back the animal's materialized configuration, which names every session the dataset
    holds for the animal whatever this host can see. Discovery is quadratic in the pooled region count of that set, so
    an animal sized from half its sessions is reserved a quarter of what the job goes on to hold, and a job reserved
    less than it holds is killed by the scheduler and cancels every dependent scheduled behind it.
    """
    first = session_factory(animal_id="305", experiment_name="test_experiment")
    second = session_factory(animal_id="305", experiment_name="test_experiment")
    for session in (first, second):
        write_surgery_metadata(session=session)
        write_processed_recording(
            session=session, regions=_POOLED_REGION_TEST_REGIONS, samples=_POOLED_REGION_TEST_SAMPLES
        )
    dataset = build_dataset(
        project_root=project_root,
        name="ds_relocated",
        sessions=[first, second],
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
    )

    # The complete set is sized over every recording the dataset names, which is the set the job itself runs over.
    complete = size_dataset_jobs(dataset=dataset, jobs=[(MULTIDAY_DISCOVERY_JOB_NAME, "305", 30)])
    assert complete[MULTIDAY_DISCOVERY_JOB_NAME, "305"] == cindra_multi_recording_footprint(
        dataset=dataset, sessions=[first, second], job_name=MultiRecordingJobNames.DISCOVER, specifier="305"
    )

    shutil.rmtree(project_root.joinpath("305", first.session_name))

    # Matches the unwrapped opening of the message, since the console formatter wraps long lines.
    with pytest.raises(FileNotFoundError, match="Unable to size the cross-recording jobs") as failure:
        size_dataset_jobs(dataset=dataset, jobs=[(MULTIDAY_DISCOVERY_JOB_NAME, "305", 30)])

    reported = " ".join(str(failure.value).split())
    # The refusal names the sessions the host is missing, which are the ones to stage before the animal is planned.
    assert first.session_name in reported
    assert second.session_name not in reported
    # Sizing the subset would have reported a figure, and a reported figure is what a scheduler reserves. The stage is
    # quadratic in the pooled region count, so that figure sits whole gigabytes below the memory the job holds.
    subset = cindra_multi_recording_footprint(
        dataset=dataset, sessions=[second], job_name=MultiRecordingJobNames.DISCOVER, specifier="305"
    )
    assert subset.memory_mb < complete[MULTIDAY_DISCOVERY_JOB_NAME, "305"].memory_mb


def test_an_extraction_job_of_an_incomplete_recording_set_is_refused(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies that the extraction stage is refused on the same terms as the discovery stage, since both are handed
    the animal's whole recording set and both are sized from it.
    """
    first = session_factory(animal_id="305", experiment_name="test_experiment")
    second = session_factory(animal_id="305", experiment_name="test_experiment")
    for session in (first, second):
        write_surgery_metadata(session=session)
        write_processed_recording(session=session, regions=120, samples=900)
    dataset = build_dataset(
        project_root=project_root,
        name="ds_relocated_extraction",
        sessions=[first, second],
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
    )
    shutil.rmtree(project_root.joinpath("305", first.session_name))

    with pytest.raises(FileNotFoundError, match="Unable to size the cross-recording jobs") as failure:
        size_dataset_jobs(dataset=dataset, jobs=[(MULTIDAY_EXTRACTION_JOB_NAME, second.session_name, 16)])

    assert first.session_name in " ".join(str(failure.value).split())


def test_an_assembly_job_of_an_incomplete_recording_set_is_refused(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies that the tracked-region bound is drawn from the animal's whole recording set or from none of it, so a
    planning host holding fewer of the animal's sessions than the dataset names is refused rather than bounded from
    the ones it holds.

    Both terms of the bound fall when the set is narrowed, and the missing recording's own region count cannot be
    recovered without its geometry, so there is nothing to bound it from. The refusal names the sessions to stage, on
    the same terms the cross-recording sizer names them.
    """
    sessions = [session_factory(animal_id="305", experiment_name="test_experiment") for _ in range(3)]
    for session in sessions:
        write_surgery_metadata(session=session)
        write_processed_recording(session=session, regions=120, samples=900)
    dataset = build_dataset(
        project_root=project_root,
        name="ds_incomplete_assembly",
        sessions=sessions,
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
    )
    for session in sessions[:2]:
        shutil.rmtree(project_root.joinpath("305", session.session_name))

    with pytest.raises(FileNotFoundError, match="Unable to size the assembly job of session") as failure:
        size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, sessions[2].session_name, 1)])

    # The console formatter wraps the message, so it is unwrapped before the named sessions are matched in it.
    reported = " ".join(str(failure.value).split())
    assert all(session.session_name in reported for session in sessions[:2])
    assert "absent under the project root" in reported


def test_the_tracked_region_bound_is_drawn_from_the_animals_whole_recording_set(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies that a host holding every session the dataset names bounds the tracked regions from all of them, so
    the pooled sum and the recording count the prevalence divides both span the whole set.

    The animal contributes one recording far wider than the rest, which is the recording that carries the bound. Its
    presence and absence separate whole gigabyte buckets, so this figure states that the bound spanned the whole set
    rather than whichever part of it a host happened to hold.

    The odd recording count here is what makes the prevalence a true ceiling: half of five rounds up to a minimum of
    three, so this fixture is the one that pins that rounding.
    """
    sessions = [session_factory(animal_id="305", experiment_name="test_experiment") for _ in range(5)]
    for index, session in enumerate(sessions):
        write_surgery_metadata(session=session)
        write_processed_recording(session=session, regions=5000 if index == 4 else 1000, samples=100_000, dense=False)
    dataset = build_dataset(
        project_root=project_root,
        name="ds_whole_set",
        sessions=sessions,
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
    )

    estimates = size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, sessions[0].session_name, 1)])

    # Half of five recordings rounds up to a cluster appearing in three of them, so the nine thousand pooled regions
    # bound the templates at three thousand.
    assert estimates[FORGING_JOB_NAME, sessions[0].session_name] == JobFootprint(
        cores=1, memory_mb=assembly_memory(samples=100_000, regions=1000, tracked_regions=3000)
    )
    # Dropping the widest recording from the set would pool four thousand regions over a minimum of two and bound the
    # result at two thousand, which is a different gigabyte bucket rather than a saving the rounding absorbs.
    assert assembly_memory(samples=100_000, regions=1000, tracked_regions=3000) != assembly_memory(
        samples=100_000, regions=1000, tracked_regions=2000
    )


def test_a_complete_recording_set_bounds_the_assembly_and_an_unreadable_entry_is_refused(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies that the completeness of the recording set is judged on the geometries the bound is drawn from rather
    than on the session directories holding them, so an entry whose processed output was removed while its directory
    stands is refused rather than silently dropped from the bound.

    Such an entry passes a directory check and then contributes to neither of the bound's terms, so the pooled sum and
    the recording count the prevalence divides both fall while the set reads as whole. The removed entry here is the
    widest of the five, and the same fixture is sized before and after the removal, so the figure the directory check
    would have reported is the one this assertion pair rules out.
    """
    sessions = [session_factory(animal_id="305", experiment_name="test_experiment") for _ in range(5)]
    for index, session in enumerate(sessions):
        write_surgery_metadata(session=session)
        write_processed_recording(session=session, regions=5000 if index == 4 else 1000, samples=100_000, dense=False)
    dataset = build_dataset(
        project_root=project_root,
        name="ds_unreadable_entry",
        sessions=sessions,
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
    )

    # The complete set is bounded over every recording the dataset names, which is the set the tracking runs over.
    complete = size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, sessions[0].session_name, 1)])
    assert complete[FORGING_JOB_NAME, sessions[0].session_name] == JobFootprint(
        cores=1, memory_mb=assembly_memory(samples=100_000, regions=1000, tracked_regions=3000)
    )

    # The widest recording keeps its session directory while the processed output the geometry is read from is gone.
    shutil.rmtree(sessions[4].processed_data.cindra_data_path)
    assert project_root.joinpath("305", sessions[4].session_name).is_dir()

    with pytest.raises(FileNotFoundError, match="Unable to size the assembly job of session") as failure:
        size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, sessions[0].session_name, 1)])

    # The console formatter wraps the message, so it is unwrapped before the named session is matched in it.
    reported = " ".join(str(failure.value).split())
    assert sessions[4].session_name in reported
    # The entry is named as one carrying no readable output rather than as one absent from the project root, since
    # staging a directory that already stands is not the remedy this entry needs.
    assert "carrying no readable processed imaging output" in reported
    assert "absent under the project root" not in reported
    # Every session that still resolves is left out of the named set, so the operator remedies only what is broken.
    # The session being sized is named by the refusal's opening rather than by that set, so it is checked separately.
    assert all(session.session_name not in reported for session in sessions[1:4])
    assert reported.count(sessions[0].session_name) == 1
    # Bounding the four surviving recordings would pool four thousand regions over a minimum of two and land at two
    # thousand, which is two thirds of the complete set's figure and a whole gigabyte bucket below it. That figure is
    # what a scheduler would have reserved, and a job reserved below what it holds is killed and cancels its
    # dependents, so the refusal above stands in place of reporting it.
    assert (
        assembly_memory(samples=100_000, regions=1000, tracked_regions=2000)
        < complete[FORGING_JOB_NAME, sessions[0].session_name].memory_mb
    )


def test_a_recording_set_entry_whose_session_marker_cannot_be_read_is_refused(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies that an entry whose directory stands while its own session marker cannot be read is classified by this
    bound and named in the refusal, rather than propagating a marker-loading failure that names neither the bound nor
    the session being sized.

    The geometry resolver loads each entry's marker to reach its output root, and that load refuses a hierarchy
    holding no marker or more than one, which is what a partial transfer or a half-deleted session leaves behind. Such
    an entry is as unmeasured as one carrying no processed output, so it is refused on the same terms rather than
    dropped from the bound. It needs a different remedy from either an absent directory or a missing output, so it is
    named in a clause of its own.
    """
    sessions = [session_factory(animal_id="305", experiment_name="test_experiment") for _ in range(3)]
    for session in sessions:
        write_surgery_metadata(session=session)
        write_processed_recording(session=session, regions=120, samples=900)
    dataset = build_dataset(
        project_root=project_root,
        name="ds_unmarked_entry",
        sessions=sessions,
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
    )

    # The half-deleted session: its directory and its processed output stand while the marker naming that output's
    # root is gone, which is the state a partial transfer leaves behind.
    sessions[1].raw_data_path.joinpath("session_data.yaml").unlink()
    assert project_root.joinpath("305", sessions[1].session_name).is_dir()

    with pytest.raises(FileNotFoundError, match="Unable to size the assembly job of session") as failure:
        size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, sessions[0].session_name, 1)])

    # The console formatter wraps the message, so it is unwrapped before the named session is matched in it.
    reported = " ".join(str(failure.value).split())
    assert sessions[1].session_name in reported
    # Named in its own clause, since neither staging a directory that already stands nor reprocessing output that is
    # already there is the remedy this entry needs.
    assert "carrying no readable session marker" in reported
    assert "absent under the project root" not in reported
    assert "carrying no readable processed imaging output" not in reported
    # The entries that still resolve are left out of the named set, and the session being sized is named only by the
    # refusal's opening rather than by that set.
    assert sessions[2].session_name not in reported
    assert reported.count(sessions[0].session_name) == 1


def test_a_recording_whose_metadata_is_absent_is_refused(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies that the combined field extent is read from the metadata archive, so an output missing it names no
    geometry and the assembly job that would read it is refused.
    """
    session = session_factory(animal_id="305", experiment_name="test_experiment")
    write_surgery_metadata(session=session)
    traces = write_trace_array(
        path=session.processed_data.cindra_data_path.joinpath("cell_fluorescence.npy"), shape=(120, 900)
    )
    dataset = build_dataset(
        project_root=project_root,
        name="ds_partial",
        sessions=[session],
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
    )

    with pytest.raises(FileNotFoundError, match="carries no processed imaging"):
        size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, session.session_name, 1)])

    # The traces themselves read cleanly, so the missing archive alone is what leaves the geometry unresolved.
    assert _read_array_shape(array_path=traces) == (120, 900)


def test_a_trace_array_of_another_rank_is_refused(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies that a fluorescence array carrying a rank other than regions by samples is not read as a recording
    geometry.

    The rank is rejected on both sides of the two axes that make up the geometry, so an array carrying a further
    axis is refused rather than read as its leading two extents.
    """
    session = session_factory(animal_id="305", experiment_name="test_experiment")
    write_surgery_metadata(session=session)
    directory = session.processed_data.cindra_data_path
    flat = write_trace_array(path=directory.joinpath("cell_fluorescence.npy"), shape=(120,))
    write_combined_metadata(directory=directory, height=64, width=64)
    dataset = build_dataset(
        project_root=project_root, name="ds_rank", sessions=[session], session_type=SessionTypes.MESOSCOPE_EXPERIMENT
    )

    with pytest.raises(FileNotFoundError, match="carries no processed imaging"):
        size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, session.session_name, 1)])

    # The metadata archive is present here, so the rank the header reports is the whole basis of the refusal.
    assert _read_array_shape(array_path=flat) is None
    volume = write_trace_array(path=directory.joinpath("volume.npy"), shape=(4, 120, 900))
    assert _read_array_shape(array_path=volume) is None


def test_a_second_format_version_header_is_parsed_the_same_way(tmp_path: Path) -> None:
    """Verifies that array headers are parsed at whichever format version wrote them, so both versions report the same
    extents.
    """
    first = write_trace_array(path=tmp_path.joinpath("first.npy"), shape=(12, 34), version=(1, 0))
    second = write_trace_array(path=tmp_path.joinpath("second.npy"), shape=(12, 34), version=(2, 0))

    assert _read_array_shape(array_path=first) == (12, 34)
    assert _read_array_shape(array_path=second) == (12, 34)
    assert _read_array_shape(array_path=tmp_path.joinpath("absent.npy")) is None


def test_the_pooled_region_bound_is_not_narrowed_to_one_recordings_own_regions(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies that the bound stands above the regions any one of the animal's recordings detected, and that the
    headroom ceiling is what caps it once the pooled ceiling rises past that headroom.

    Tracking keeps a cluster on the count of distinct recordings it spans alone, so nothing requires a surviving
    cluster to hold a region from any particular recording and a set of equally wide recordings can yield more
    templates than any one of them detected regions. Narrowing to one recording's own count would under-reserve such
    an animal, and a job reserved below what it holds is killed by the scheduler and cancels every dependent
    scheduled behind it.

    The two ceilings the bound takes the smaller of are separated here. Half prevalence over four recordings of a
    thousand regions pools four thousand over a minimum of two and allows two thousand templates, while the headroom
    ceiling allows a thousand and a half, so the headroom term is the one charged. That is the direction a pooled-only
    bound gets wrong, and it grows with the recording count: this fixture is the small end of the case that twenty
    recordings of fifteen thousand regions carry to twice the memory such an animal holds.

    The pooled ceiling is the tighter of the two over the odd five-recording set that
    test_the_tracked_region_bound_is_drawn_from_the_animals_whole_recording_set carries, so that test pins the pooled
    term and its prevalence rounding while this one pins the headroom term. Between them the smaller of the two is
    taken in both directions.

    The three figures the assertions separate land in three different gigabyte buckets rather than in one the
    rounding absorbs, so the reported figure names the headroom ceiling as its source and excludes both a pooled-only
    bound and a bound narrowed to one recording.
    """
    sessions = [session_factory(animal_id="305", experiment_name="test_experiment") for _ in range(4)]
    for session in sessions:
        write_surgery_metadata(session=session)
        write_processed_recording(session=session, regions=1000, samples=100_000, dense=False)
    dataset = build_dataset(
        project_root=project_root,
        name="ds_pooled",
        sessions=sessions,
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
    )

    estimates = size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, sessions[0].session_name, 1)])

    # The most populated recording detected a thousand regions and the headroom allows half as many again, so the
    # multi-day columns are charged fifteen hundred templates.
    assert estimates[FORGING_JOB_NAME, sessions[0].session_name] == JobFootprint(
        cores=1, memory_mb=assembly_memory(samples=100_000, regions=1000, tracked_regions=1500)
    )
    # Narrowing to the widest recording's own count would charge a thousand templates, which is a different gigabyte
    # bucket rather than a difference the rounding absorbs, and it is an under-estimate of what the job attaches.
    assert assembly_memory(samples=100_000, regions=1000, tracked_regions=1500) != assembly_memory(
        samples=100_000, regions=1000, tracked_regions=1000
    )
    # Taking the pooled ceiling alone would charge two thousand, which is a third gigabyte bucket, so the figure
    # reported above could not have come from it either.
    assert assembly_memory(samples=100_000, regions=1000, tracked_regions=1500) != assembly_memory(
        samples=100_000, regions=1000, tracked_regions=2000
    )


def test_a_dataset_donating_no_configuration_is_bounded_over_one_recording(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies that a dataset whose acquisition system donates no multi-recording configuration divides its pooled
    ceiling by one recording rather than by none, so the bound answers a figure instead of dividing by nothing.

    A configuration is what states the prevalence a cluster must reach, and nothing states it when a system donates
    none, which is the state this pass documents as its own. The prevalence then reads as zero, the recording count it
    scales reads as zero with it, and an unfloored divisor divides the pooled region count by nothing at all. One
    recording is also the honest answer in that state: nothing requires a cluster to span more than one recording, so
    the pooled ceiling relaxes to the whole pooled sum.

    The configuration is handed in as none directly rather than provoked through a dataset, because the sole
    registered acquisition system donates a configuration for every session type it admits into a dataset. The state
    belongs to the donation contract rather than to any dataset this pipeline builds, and a system that admits a
    session type while performing no cross-recording tracking for it reaches the state through its donation alone.

    The thirteen hundred regions the set pools stand below the fifteen hundred the headroom ceiling allows, so the
    pooled term is the one the bound takes and the divisor this fixture pins is the one that term carries.
    """
    sessions = [session_factory(animal_id="305", experiment_name="test_experiment") for _ in range(2)]
    for session, regions in zip(sessions, (1000, 300), strict=True):
        write_surgery_metadata(session=session)
        write_processed_recording(session=session, regions=regions, samples=100_000, dense=False)
    dataset = build_dataset(
        project_root=project_root,
        name="ds_unconfigured",
        sessions=sessions,
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
    )

    tracked_regions = _resolve_tracked_regions(
        dataset=dataset,
        animal="305",
        session=sessions[0].session_name,
        project_root=project_root,
        configuration=None,
    )

    assert tracked_regions == 1300


def test_the_headroom_ceiling_rounds_its_fractional_template_up(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies that the headroom ceiling rounds the fractional template an odd region count leaves it up rather than
    down, so the bound never falls below the templates the tracking can keep.

    Every other fixture reaching this bound carries an even region count, and half of an even count is whole, so both
    directions of rounding answer alike there. The odd count here separates them: a thousand and one regions at half
    again is fifteen hundred and one and a half, which rounding up charges as fifteen hundred and two and rounding
    down would charge as fifteen hundred and one.

    The assertion is made on the bound itself rather than on the memory a job carrying it reports, because a reported
    figure is rounded up to a whole gigabyte and one region of fluorescence stands far below that quantum, so the
    difference between the two directions would not survive into any estimate the sizing pass reports.

    The pooled ceiling stands above the headroom one over this set, so the headroom term is the one the bound takes:
    half prevalence over two recordings keeps a cluster appearing in one of them, which leaves the two thousand and
    two pooled regions of the set undivided and well above the headroom the widest recording allows.
    """
    sessions = [session_factory(animal_id="305", experiment_name="test_experiment") for _ in range(2)]
    for session in sessions:
        write_surgery_metadata(session=session)
        write_processed_recording(session=session, regions=1001, samples=1000, dense=False)
    dataset = build_dataset(
        project_root=project_root,
        name="ds_odd_headroom",
        sessions=sessions,
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
    )
    resolve_configuration = resolve_multi_recording_configuration_resolver(system=dataset.acquisition_system)

    tracked_regions = _resolve_tracked_regions(
        dataset=dataset,
        animal="305",
        session=sessions[0].session_name,
        project_root=project_root,
        configuration=resolve_configuration(sessions[0]),
    )

    assert tracked_regions == 1502


def test_a_stated_region_count_is_charged_in_place_of_the_tracked_region_bound(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies that a caller that reasonably knows how many regions its tracking keeps states that count and is
    charged it, and that the recording set the bound is drawn from is then neither gathered nor gated.

    The bound exists because the tracking has not run when the job that reads its output is planned. A caller holding
    a figure for it from outside this pass is better informed than the bound is, so it states one, on the same terms
    cindra's own single-recording sizing takes a planned region count in place of its detection ceiling.

    The recording set is gathered for the bound alone, and the completeness gate over that set exists to keep the
    bound from being drawn over part of it. Neither serves a stated count, so both are skipped, which the second half
    of this test pins by refusing the very same dataset once the count is withdrawn.

    The four hundred stated here and the fifteen hundred the animal's recording set bounds sit two gigabyte buckets
    apart rather than in one the rounding absorbs, so the reported figure names the stated count as its source.
    """
    sessions = [session_factory(animal_id="305", experiment_name="test_experiment") for _ in range(2)]
    for session in sessions:
        write_surgery_metadata(session=session)
        write_processed_recording(session=session, regions=1000, samples=100_000, dense=False)
    dataset = build_dataset(
        project_root=project_root,
        name="ds_stated_count",
        sessions=sessions,
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
    )
    jobs = [(FORGING_JOB_NAME, sessions[0].session_name, 1)]

    stated = size_dataset_jobs(dataset=dataset, jobs=jobs, planned_roi_count=400)

    assert stated[FORGING_JOB_NAME, sessions[0].session_name] == JobFootprint(
        cores=1, memory_mb=assembly_memory(samples=100_000, regions=1000, tracked_regions=400)
    )
    # The same dataset without a stated count is charged the bound its recording set carries, which is the headroom
    # ceiling over the thousand regions each of its two recordings detected. The two calls differ in the stated count
    # alone, so that count is what the figure above is attributed to.
    assert size_dataset_jobs(dataset=dataset, jobs=jobs)[FORGING_JOB_NAME, sessions[0].session_name] == JobFootprint(
        cores=1, memory_mb=assembly_memory(samples=100_000, regions=1000, tracked_regions=1500)
    )
    assert assembly_memory(samples=100_000, regions=1000, tracked_regions=400) != assembly_memory(
        samples=100_000, regions=1000, tracked_regions=1500
    )

    # Narrowing the recording set refuses the bound, since it would be drawn over part of the set the tracking spans.
    shutil.rmtree(project_root.joinpath("305", sessions[1].session_name))
    with pytest.raises(FileNotFoundError, match="Unable to size the assembly job of session"):
        size_dataset_jobs(dataset=dataset, jobs=jobs)

    # The stated count answers the same narrowed dataset, because the gate guards the bound rather than the job.
    narrowed = size_dataset_jobs(dataset=dataset, jobs=jobs, planned_roi_count=400)

    assert narrowed[FORGING_JOB_NAME, sessions[0].session_name] == JobFootprint(
        cores=1, memory_mb=assembly_memory(samples=100_000, regions=1000, tracked_regions=400)
    )


def test_a_stated_region_count_reaches_the_cross_recording_stages(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies that a stated template count is handed to cindra's own sizing pass rather than spent on this
    package's assembly model alone, so the stages that produce the templates are planned for the same count as the
    stage that reads them.

    The count a caller states is a property of the tracking the dataset will run, and the two cross-recording stages
    are the stages that run it. cindra takes the same override for exactly that reason, and reads it in its tracked
    extraction stage, whose traces span the templates. Its discovery stage scales with the regions each recording
    reports rather than with the templates it produces, so the count reaches it and changes nothing there, which the
    third assertion pins so the forwarding is not read as moving a figure it does not own.

    The recordings here record their combined frame count, because cindra's extraction model multiplies the templates
    by that count and an archive recording none collapses the whole term to zero, which would leave the stated count
    invisible whatever it held. The four hundred stated and the bound cindra draws for itself sit two gigabyte buckets
    apart rather than in one the rounding absorbs, so the reported figure names the stated count as its source.
    """
    sessions = [session_factory(animal_id="305", experiment_name="test_experiment") for _ in range(2)]
    for session in sessions:
        write_surgery_metadata(session=session)
        write_processed_recording(session=session, regions=1000, samples=100_000, dense=False, record_frame_count=True)
    dataset = build_dataset(
        project_root=project_root,
        name="ds_forwarded_count",
        sessions=sessions,
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
    )
    jobs = [
        (MULTIDAY_DISCOVERY_JOB_NAME, "305", 30),
        (MULTIDAY_EXTRACTION_JOB_NAME, sessions[0].session_name, 16),
    ]

    stated = size_dataset_jobs(dataset=dataset, jobs=jobs, planned_roi_count=400)
    unstated = size_dataset_jobs(dataset=dataset, jobs=jobs)

    # The extraction figure is cindra's own figure at the stated count, which is what the forwarding is for.
    assert stated[MULTIDAY_EXTRACTION_JOB_NAME, sessions[0].session_name] == cindra_multi_recording_footprint(
        dataset=dataset,
        sessions=sessions,
        job_name=MultiRecordingJobNames.EXTRACT,
        specifier=sessions[0].session_name,
        planned_roi_count=400,
    )
    # The two calls differ in the stated count alone and land in different gigabyte buckets, so that count is what
    # the figure above is attributed to rather than anything the fixture holds.
    assert (
        stated[MULTIDAY_EXTRACTION_JOB_NAME, sessions[0].session_name].memory_mb
        != unstated[MULTIDAY_EXTRACTION_JOB_NAME, sessions[0].session_name].memory_mb
    )
    # The discovery stage does not read the count, so it reports the same figure either way. The forwarding reaches
    # it all the same, since which stages read the count is cindra's model to state rather than this pass's.
    assert stated[MULTIDAY_DISCOVERY_JOB_NAME, "305"] == unstated[MULTIDAY_DISCOVERY_JOB_NAME, "305"]


@pytest.mark.parametrize(
    "damage",
    [
        b"",
        b"PK\x03\x04 truncated before the entry directory",
        b"this was never a zip archive at all",
    ],
    ids=["emptied", "truncated", "overwritten"],
)
def test_a_recording_whose_metadata_archive_cannot_be_read_is_refused_as_unreadable_output(
    project_root: Path, session_factory: Callable[..., SessionData], damage: bytes
) -> None:
    """Verifies that a recording whose combined metadata archive stands on disk but cannot be read is classified as
    carrying no readable processed output, rather than crashing the sizing pass or being reported under a remedy that
    would not repair it.

    The archive is a compressed entry store, so a half-written or overwritten one fails in the shape of whichever
    layer first reaches the damage, and those shapes are unrelated exception types. One of them is OSError, which this
    bound classifies as an unreadable session marker when it comes from the marker load, so an unguarded archive read
    would name a damaged recording under a remedy that repairs session hierarchies rather than processed output.

    Every damaged archive here leaves the session directory, the session marker and the trace array standing, so the
    only thing the refusal can be reacting to is the archive itself.
    """
    sessions = [session_factory(animal_id="305", experiment_name="test_experiment") for _ in range(3)]
    for session in sessions:
        write_surgery_metadata(session=session)
        write_processed_recording(session=session, regions=120, samples=900)
    dataset = build_dataset(
        project_root=project_root,
        name="ds_damaged_archive",
        sessions=sessions,
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
    )

    archive = sessions[1].processed_data.cindra_data_path.joinpath("combined_metadata.npz")
    archive.write_bytes(damage)
    # Everything else the resolution reads is left intact, so the archive is the only unreadable input.
    assert sessions[1].processed_data.cindra_data_path.joinpath("cell_fluorescence.npy").is_file()
    assert sessions[1].raw_data_path.joinpath("session_data.yaml").is_file()

    with pytest.raises(FileNotFoundError, match="Unable to size the assembly job of session") as failure:
        size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, sessions[0].session_name, 1)])

    # The console formatter wraps the message, so it is unwrapped before the named session is matched in it.
    reported = " ".join(str(failure.value).split())
    assert sessions[1].session_name in reported
    # Reprocessing the recording is the remedy a damaged archive needs, so it is named in that clause rather than in
    # the one that asks for a repaired session hierarchy or a staged directory.
    assert "carrying no readable processed imaging output" in reported
    assert "carrying no readable session marker" not in reported
    assert "absent under the project root" not in reported
    # The entries that still resolve are left out of the named set, so the operator remedies only what is broken.
    assert sessions[2].session_name not in reported


def test_a_recording_whose_combined_field_is_empty_names_no_geometry(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies that a recording whose metadata archive describes an empty combined field is refused, rather than
    being measured because the archive that describes it happens to exist.

    A recording is judged complete on what its archive reports rather than on the archive's presence, since a combined
    view of no extent is not output any stage can read. cindra's own multi-recording sizing refuses such a recording
    on those same terms, so refusing it here keeps the set this bound is drawn from equal to the set the tracking
    stages will accept, which is the agreement the bound depends on to not under-reserve.
    """
    sessions = [session_factory(animal_id="305", experiment_name="test_experiment") for _ in range(2)]
    for session in sessions:
        write_surgery_metadata(session=session)
        write_processed_recording(session=session, regions=120, samples=900)
    dataset = build_dataset(
        project_root=project_root,
        name="ds_empty_field",
        sessions=sessions,
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
    )

    # The archive stands and parses; it reports a combined field of no extent, which is what a combination stage that
    # contributed no plane leaves behind.
    write_combined_metadata(directory=sessions[1].processed_data.cindra_data_path, height=0, width=96)
    assert sessions[1].processed_data.cindra_data_path.joinpath("combined_metadata.npz").is_file()
    assert sessions[1].processed_data.cindra_data_path.joinpath("cell_fluorescence.npy").is_file()

    with pytest.raises(FileNotFoundError, match="Unable to size the assembly job of session") as failure:
        size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, sessions[0].session_name, 1)])

    # The console formatter wraps the message, so it is unwrapped before the named session is matched in it.
    reported = " ".join(str(failure.value).split())
    assert sessions[1].session_name in reported
    assert "carrying no readable processed imaging output" in reported
    assert "carrying no readable session marker" not in reported


def test_an_assembly_job_carrying_an_animal_specifier_bounds_its_regions_at_one(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies the defensive floor the bound falls to when its specifier resolves no recording set at all.

    The forging job universe never reaches this floor. '_build_forging_universe' emits an assembly job only for a
    session the dataset lists, so a specifier reaching the sizing pass through the pipeline always resolves a real
    animal and a non-empty entry list. This test therefore pins defensive behavior for a caller reaching the public
    sizing pass directly, rather than a state the pipeline produces.

    The discovery stage is specified by its animal while assembly is specified by its session, so a specifier naming
    an animal matches no session the dataset lists and pools no recordings. The bound settles at the one template an
    empty recording set allows rather than failing the whole batch's sizing pass, and an empty set is not read as a
    set this host could not measure, which would refuse it.

    The recording is posed at a scale where the single template and the two thousand regions it detected land in
    different gigabyte buckets, so the reported figure states which of the two the multi-day columns were charged.
    """
    session = session_factory(animal_id="305", experiment_name="test_experiment")
    write_surgery_metadata(session=session)
    write_processed_recording(session=session, regions=2000, samples=100_000, dense=False)
    dataset = build_dataset(
        project_root=project_root,
        name="ds_animal_specifier",
        sessions=[session],
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
    )

    estimates = size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, "305", 1)])

    # The animal directory resolves the recording, so the single-day columns are charged that recording's own two
    # thousand regions, while the multi-day columns are charged the single template an empty recording set allows.
    assert estimates[FORGING_JOB_NAME, "305"] == JobFootprint(
        cores=1, memory_mb=assembly_memory(samples=100_000, regions=2000, tracked_regions=1)
    )
    # Charging the multi-day columns the recording's own regions instead would double the retained width into the
    # next gigabyte bucket, so the reported figure could not have come from a bound over a non-empty recording set.
    assert assembly_memory(samples=100_000, regions=2000, tracked_regions=1) != assembly_memory(
        samples=100_000, regions=2000
    )


def test_a_dataset_job_naming_a_stage_nothing_models_is_refused(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Verifies that every forging job is routed to a model of its own the same way a session's jobs are, so a name
    reaching the end of that routing is refused rather than admitted to a batch at an allowance nobody chose.
    """
    session = session_factory(animal_id="321", session_type=SessionTypes.RUN_TRAINING)
    write_surgery_metadata(session=session)
    dataset = build_dataset(
        project_root=project_root, name="ds_unrouted", sessions=[session], session_type=SessionTypes.RUN_TRAINING
    )

    # Matches the unwrapped opening of the message, since the console formatter wraps long lines.
    with pytest.raises(ValueError, match="Unable to size job") as failure:
        size_dataset_jobs(dataset=dataset, jobs=[("a_stage_no_pipeline_declares", session.session_name, 1)])

    assert "routes to no sizing model" in " ".join(str(failure.value).split())


# Sizing model identity


def test_one_tuning_of_the_sizing_constants_answers_with_one_identifier() -> None:
    """Verifies that the identifier holds steady while the constants behind it do, so a plan stays readable."""
    assert resolve_model_version() == resolve_model_version()
    assert len(resolve_model_version()) == _MODEL_VERSION_DIGITS


def test_retuning_a_scalar_sizing_constant_answers_with_another_identifier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a retuned ratio answers with another identifier, so every plan it stamped is estimated again."""
    tuned = resolve_model_version()

    monkeypatch.setattr(footprints_module, "_POSE_TABLE_COPIES", _POSE_TABLE_COPIES + 1.0)

    assert resolve_model_version() != tuned


def test_retuning_a_collection_constant_answers_with_another_identifier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a constant holding a collection reaches the identifier alongside the scalar ones."""
    tuned = resolve_model_version()

    monkeypatch.setattr(footprints_module, "_ARCHIVE_JOB_NAMES", frozenset({CHECKSUM_JOB_NAME}))

    assert resolve_model_version() != tuned


# Shared conversions


def test_byte_conversion_rounds_up_and_reports_nothing_for_nothing() -> None:
    """Verifies that a converted byte count never understates its demand, while an empty input converts to no memory at
    all.
    """
    assert _bytes_to_megabytes(byte_count=0) == 0
    assert _bytes_to_megabytes(byte_count=1) == 1
    assert _bytes_to_megabytes(byte_count=1024 * 1024) == 2


def test_the_camera_manifest_is_read_once_for_every_motion_energy_job_of_a_session(
    experiment_session: SessionData,
    write_grayscale_video: Callable[..., Path],
    moving_block_frames: NDArray[np.uint8],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that one read of the acquisition-time manifest answers every motion-energy job of a session, since the
    manifest names every camera at once and re-reading it per job would walk the raw behavior tree once per camera.
    """
    write_camera_manifest(
        session=experiment_session, cameras={_FACE_SOURCE_ID: _FACE_CAMERA, _BODY_SOURCE_ID: _BODY_CAMERA}
    )
    write_camera_recording(
        session=experiment_session, camera=_FACE_CAMERA, frames=moving_block_frames, writer=write_grayscale_video
    )
    write_camera_recording(
        session=experiment_session, camera=_BODY_CAMERA, frames=moving_block_frames, writer=write_grayscale_video
    )

    resolutions: list[Path] = []
    resolve_manifest_jobs = footprints_module.resolve_jobs

    def counted_resolve_jobs(log_directory: Path):
        resolutions.append(log_directory)
        return resolve_manifest_jobs(log_directory=log_directory)

    monkeypatch.setattr(footprints_module, "resolve_jobs", counted_resolve_jobs)

    size_session_jobs(
        pipeline=ProcessingPipelines.VIDEO,
        session=experiment_session,
        jobs=[(ENERGY_JOB_NAME, _FACE_SOURCE_ID, 4), (ENERGY_JOB_NAME, _BODY_SOURCE_ID, 4)],
    )

    assert resolutions == [experiment_session.raw_data.behavior_data_path]


def test_a_session_planning_no_motion_energy_job_reads_no_camera_manifest(
    experiment_session: SessionData, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that the frame pre-pass gates on the presence of a motion-energy job rather than on the pipeline that
    owns one, so a session planning none opens nothing at all.
    """

    def refuse_resolve_jobs(log_directory: Path):
        message = f"The camera manifest under '{log_directory}' must not be read for a session planning no energy job."
        raise AssertionError(message)

    monkeypatch.setattr(footprints_module, "resolve_jobs", refuse_resolve_jobs)

    estimates = size_session_jobs(
        pipeline=ProcessingPipelines.VIDEO, session=experiment_session, jobs=[(RENAME_JOB_NAME, "", 1)]
    )

    assert estimates[RENAME_JOB_NAME, ""] == JobFootprint(cores=1, memory_mb=_WORKER_ONLY_MB)
