"""Tests the sizing pass that resolves each job's cores and working set from the data it reads."""

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
import pytest
import tifffile
from ataraxis_video_system import (
    CAMERA_EXTRACTION_JOB_CORES,
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
from sollertia_forgery.shared_assets import ProcessingPipelines, multi_recording_dataset_directory
from sollertia_forgery.microcontrollers import PARSE_JOB_NAME, CONTROLLER_EXTRACTION_JOB_NAME
from sollertia_forgery.orchestration.footprints import (
    _ASSEMBLY_WRITE_COPIES,
    _POSE_PREDICTION_RATIO,
    _RETAINED_FRAME_BUFFERS,
    _SINGLE_PRECISION_BYTES,
    _ARCHIVE_DIRECTORY_RATIO,
    _DECODER_BUFFER_MEMORY_MB,
    _CHECKSUM_READER_MEMORY_MB,
    _SUB_DATASET_BYTES_PER_SAMPLE,
    _ASSEMBLY_FLUORESCENCE_COLUMNS,
    JobFootprint,
    _apply_tolerance,
    _read_array_shape,
    size_dataset_jobs,
    size_session_jobs,
    _round_to_gigabyte,
    _bytes_to_megabytes,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Mapping, Callable, Sequence

    from numpy.typing import NDArray

WORKER_ONLY_MB: int = _apply_tolerance(memory_mb=WORKER_MEMORY_MB)
"""The memory a stage holding one worker and nothing besides is charged, which is the whole model for a job that
reads no input of its own."""

PARALLEL_ARCHIVE_MESSAGES: int = 35_000
"""The messages a synthetic log archive holds to earn the parallel extraction path from both acquisition libraries.
The video library carries the higher of the two message thresholds, so an archive this size clears both."""

FRAME_HEIGHT: int = 300
"""The line count of one unsliced acquisition frame the synthetic imaging stacks carry."""

FRAME_WIDTH: int = 64
"""The pixel width of one acquisition frame, shared by every plane the conversion stage slices out."""

REGION_LINES: list[list[int]] = [[1, 100], [101, 300], []]
"""The per-region line spans the synthetic acquisition parameters declare, holding one empty span the reader drops."""

SAMPLING_RATE: float = 10.0
"""The per-plane sampling rate the synthetic acquisition parameters declare."""


def write_surgery_metadata(session: SessionData, genotype: str = "GP5.17") -> Path:
    """Writes the surgery metadata the Mesoscope-VR cindra resolvers read an animal's genotype from.

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
        directory: The raw imaging directory the file is written into.
        region_lines: The per-region line spans to declare, or None to declare none at all.

    Returns:
        The path to the written parameters file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    parameters: dict[str, object] = {"frame_rate": SAMPLING_RATE, "plane_number": 2, "channel_number": 1}
    if region_lines is not None:
        parameters["roi_lines"] = region_lines
    path = directory.joinpath(PARAMETERS_FILENAME)
    path.write_text(json.dumps(parameters))
    return path


def write_imaging_stack(directory: Path, name: str, pages: int) -> Path:
    """Writes one multi-page acquisition image whose header reports the synthetic recording's frame shape.

    Args:
        directory: The raw imaging directory the image is written into.
        name: The filename to write, including its extension.
        pages: The number of pages the image holds.

    Returns:
        The path to the written image.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory.joinpath(name)
    tifffile.imwrite(path, np.zeros((pages, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.int16))
    return path


def write_raw_imaging(session: SessionData, *, region_lines: list[list[int]] | None = REGION_LINES) -> Path:
    """Builds a complete synthetic raw imaging directory holding two acquisition images and their parameters.

    Args:
        session: The session whose raw data receives the imaging directory.
        region_lines: The per-region line spans the parameters declare.

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
        path: The path to write the array to.
        shape: The extents the header reports.
        version: The array format version the header is written under.

    Returns:
        The path to the written array.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as array_file:
        np.lib.format.write_array(array_file, np.zeros(shape, dtype=np.float32), version=version)
    return path


def write_combined_metadata(directory: Path, *, height: int, width: int) -> Path:
    """Writes the cindra archive reporting the combined field extent every multi-day stage works at.

    Args:
        directory: The processed output directory the archive is written into.
        height: The combined field height in pixels.
        width: The combined field width in pixels.

    Returns:
        The path to the written archive.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory.joinpath("combined_metadata.npz")
    np.savez(path, combined_height=np.array([height]), combined_width=np.array([width]))
    return path


def write_processed_recording(
    session: SessionData, *, regions: int, samples: int, height: int = 128, width: int = 96
) -> Path:
    """Writes the single-recording outputs a dataset job's estimate is sized from.

    Args:
        session: The session whose processed data receives the outputs.
        regions: The regions the recording's traces hold.
        samples: The samples each trace holds.
        height: The combined field height in pixels.
        width: The combined field width in pixels.

    Returns:
        The path to the session's cindra output directory.
    """
    directory = session.processed_data.cindra_data_path
    write_trace_array(path=directory.joinpath("cell_fluorescence.npy"), shape=(regions, samples))
    write_combined_metadata(directory=directory, height=height, width=width)
    return directory


def build_dataset(
    project_root: Path, name: str, sessions: Sequence[SessionData], session_type: SessionTypes
) -> DatasetData:
    """Creates a forged dataset hierarchy naming the given sessions, through the shared hierarchy's own creator.

    Args:
        project_root: The project root the dataset is created under, which is where its sessions are resolved from.
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
        session: The session the jobs operate on.
        jobs: The jobs as name, specifier, and declared core triples.

    Returns:
        The footprint of every named job.
    """
    return size_session_jobs(pipeline=ProcessingPipelines.TWO_PHOTON, session=session, jobs=jobs)


def camera_library_footprint(archive: Path) -> JobFootprint:
    """Reports the footprint the video library's own sizing pass gives one archive, at the scale slf reports it on.

    Args:
        archive: The log archive the extraction job reads.

    Returns:
        The library's cores and its memory, rounded up to the whole gigabyte every estimate lands on.
    """
    sizing = size_camera_extraction_job(archive_path=archive)
    return JobFootprint(cores=sizing.cores, memory_mb=_round_to_gigabyte(memory_mb=sizing.memory_mb))


def controller_library_footprint(archive: Path) -> JobFootprint:
    """Reports the footprint the communication library's own sizing pass gives one archive, at slf's scale.

    Args:
        archive: The log archive the extraction job reads.

    Returns:
        The library's cores and its memory, rounded up to the whole gigabyte every estimate lands on.
    """
    sizing = size_controller_extraction_job(archive_path=archive)
    return JobFootprint(cores=sizing.cores, memory_mb=_round_to_gigabyte(memory_mb=sizing.memory_mb))


def cindra_single_recording_footprint(
    session: SessionData, job_name: SingleRecordingJobNames, specifier: str
) -> JobFootprint:
    """Reports the footprint cindra's own sizing pass gives one single-recording stage, at slf's scale.

    Args:
        session: The session the job operates on.
        job_name: The cindra stage the job runs.
        specifier: The job's tracker specifier, which names a plane for the per-plane stages.

    Returns:
        cindra's own width for the stage and its memory, rounded up to the whole gigabyte every estimate lands on.
    """
    resolve_configuration = resolve_single_recording_configuration_resolver(system=session.acquisition_system)
    locate_two_photon_data = resolve_two_photon_data_locator(system=session.acquisition_system)
    sizing = size_single_recording_job(
        job_name=job_name,
        specifier=specifier,
        output_root=session.processed_data_path,
        configuration=resolve_configuration(session),
        data_path=locate_two_photon_data(session),
    )
    return JobFootprint(cores=sizing.cores, memory_mb=_round_to_gigabyte(memory_mb=sizing.memory_mb))


def cindra_multi_recording_footprint(
    dataset: DatasetData, sessions: Sequence[SessionData], job_name: MultiRecordingJobNames, specifier: str
) -> JobFootprint:
    """Reports the footprint cindra's own sizing pass gives one cross-recording stage, at slf's scale.

    Args:
        dataset: The dataset whose acquisition system donates the multi-recording configuration.
        sessions: The sessions whose cindra output directories the stage spans.
        job_name: The cindra stage the job runs.
        specifier: The job's tracker specifier, naming a session for extraction and an animal for discovery.

    Returns:
        cindra's own width for the stage and its memory, rounded up to the whole gigabyte every estimate lands on.
    """
    resolve_configuration = resolve_multi_recording_configuration_resolver(system=dataset.acquisition_system)
    sizing = size_multi_recording_job(
        job_name=job_name,
        specifier=specifier,
        recording_directories=[session.processed_data.cindra_data_path for session in sessions],
        configuration=resolve_configuration(sessions[0]),
    )
    return JobFootprint(cores=sizing.cores, memory_mb=_round_to_gigabyte(memory_mb=sizing.memory_mb))


def assembly_memory(samples: int, regions: int) -> int:
    """Reports the figure the per-session assembly model gives a recording of the named shape.

    Args:
        samples: The samples each retained fluorescence column holds.
        regions: The regions each retained fluorescence column spans.

    Returns:
        The reportable memory in megabytes.
    """
    columns = _ASSEMBLY_FLUORESCENCE_COLUMNS * _ASSEMBLY_WRITE_COPIES * samples * regions * _SINGLE_PRECISION_BYTES
    return _apply_tolerance(
        memory_mb=WORKER_MEMORY_MB + _bytes_to_megabytes(byte_count=columns + samples * _SUB_DATASET_BYTES_PER_SAMPLE)
    )


# Session estimates


def test_checksum_memory_scales_with_the_readers_a_job_opens(experiment_session: SessionData) -> None:
    """The checksum estimate follows the cores a job holds, since each core streams one file in fixed chunks."""
    estimates = size_session_jobs(
        pipeline=ProcessingPipelines.CHECKSUM,
        session=experiment_session,
        jobs=[(CHECKSUM_JOB_NAME, "", 8), (CHECKSUM_JOB_NAME, "wide", 16)],
    )

    narrow = estimates[CHECKSUM_JOB_NAME, ""]
    wide = estimates[CHECKSUM_JOB_NAME, "wide"]
    assert narrow.memory_mb == _apply_tolerance(memory_mb=WORKER_MEMORY_MB + 8 * _CHECKSUM_READER_MEMORY_MB)
    assert wide.memory_mb > narrow.memory_mb
    # The checksum stage is this package's own, so each job is planned at the allocation it was handed.
    assert (narrow.cores, wide.cores) == (8, 16)


def test_an_archive_reader_estimate_scales_with_the_archive_on_disk(
    experiment_session: SessionData, write_log_archive: Callable[..., Path]
) -> None:
    """Every worker of a log-reading stage opens the archive itself, so the estimate is charged once per core."""
    archive = write_log_archive(experiment_session.raw_data.behavior_data_path.joinpath("51_log.npz"), 51, [(5, b"ab")])
    per_reader = _bytes_to_megabytes(byte_count=archive.stat().st_size * _ARCHIVE_DIRECTORY_RATIO)

    estimates = size_session_jobs(
        pipeline=ProcessingPipelines.RUNTIME,
        session=experiment_session,
        jobs=[(RUNTIME_JOB_NAME, "51", 4), (CONTROLLER_EXTRACTION_JOB_NAME, "51", 8)],
    )

    assert estimates[RUNTIME_JOB_NAME, "51"] == JobFootprint(
        cores=4,
        memory_mb=_apply_tolerance(memory_mb=WORKER_MEMORY_MB + 4 * (per_reader + SPAWNED_CHILD_MEMORY_MB)),
    )
    # The extraction stage belongs to the communication library, so both halves of its figure are that library's own
    # sizing pass, with the memory rounded to the gigabyte every reportable estimate lands on.
    assert estimates[CONTROLLER_EXTRACTION_JOB_NAME, "51"] == controller_library_footprint(archive=archive)


def test_a_runtime_job_whose_archive_was_never_written_is_refused(experiment_session: SessionData) -> None:
    """A log-reading stage holds memory in proportion to an archive, so an absent archive names no figure and the
    job that would read it cannot run either.
    """
    experiment_session.raw_data.behavior_data_path.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError):
        size_session_jobs(
            pipeline=ProcessingPipelines.RUNTIME,
            session=experiment_session,
            jobs=[(RUNTIME_JOB_NAME, "51", 4)],
        )


def test_a_camera_extraction_estimate_follows_the_video_library_model(
    experiment_session: SessionData, write_log_archive: Callable[..., Path]
) -> None:
    """The camera timestamp stage belongs to the video library, so slf reports that library's figures unchanged."""
    archive = write_log_archive(experiment_session.raw_data.behavior_data_path.joinpath("77_log.npz"), 77, [(5, b"ab")])

    estimates = size_session_jobs(
        pipeline=ProcessingPipelines.VIDEO,
        session=experiment_session,
        jobs=[(CAMERA_EXTRACTION_JOB_NAME, "77", 8)],
    )

    assert estimates[CAMERA_EXTRACTION_JOB_NAME, "77"] == camera_library_footprint(archive=archive)
    # The controller archive of source 99 was never written, and each library refuses an archive it cannot read
    # because the job reading it could not run either. That refusal reaches the caller unchanged.
    with pytest.raises(FileNotFoundError):
        size_session_jobs(
            pipeline=ProcessingPipelines.VIDEO,
            session=experiment_session,
            jobs=[(CONTROLLER_EXTRACTION_JOB_NAME, "99", 8)],
        )
    with pytest.raises(FileNotFoundError):
        size_session_jobs(
            pipeline=ProcessingPipelines.VIDEO,
            session=experiment_session,
            jobs=[(CAMERA_EXTRACTION_JOB_NAME, "99", 8)],
        )


def test_an_extraction_job_is_sized_at_the_width_its_own_archive_earns(
    experiment_session: SessionData, write_log_archive: Callable[..., Path]
) -> None:
    """Each library picks a width from the archive its job reads, so a small archive runs sequentially whatever
    allocation this package declared for the stage.
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
    """An archive dense enough to repay a pool earns the allocation the library that owns the stage declares, and the
    memory that comes back is the memory of that width rather than of the width this package asked for.
    """
    archive = write_log_archive(
        experiment_session.raw_data.behavior_data_path.joinpath("77_log.npz"),
        77,
        [(index, b"ab") for index in range(PARALLEL_ARCHIVE_MESSAGES)],
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


def test_a_parse_estimate_follows_the_widest_archive_in_the_behavior_directory(
    experiment_session: SessionData, write_log_archive: Callable[..., Path]
) -> None:
    """A parse job reads one module's share of its controller's archive, so the widest archive bounds it."""
    behavior = experiment_session.raw_data.behavior_data_path
    write_log_archive(behavior.joinpath("51_log.npz"), 51, [(1, b"a")])
    widest = write_log_archive(behavior.joinpath("52_log.npz"), 52, [(index, bytes(400)) for index in range(20)])

    estimates = size_session_jobs(
        pipeline=ProcessingPipelines.MICROCONTROLLER,
        session=experiment_session,
        jobs=[(PARSE_JOB_NAME, "52_1_1", 1)],
    )

    assert widest.stat().st_size > behavior.joinpath("51_log.npz").stat().st_size
    assert estimates[PARSE_JOB_NAME, "52_1_1"] == JobFootprint(
        cores=1,
        memory_mb=_apply_tolerance(memory_mb=WORKER_MEMORY_MB + _bytes_to_megabytes(widest.stat().st_size * 3.4)),
    )


def test_a_parse_estimate_is_refused_when_the_behavior_directory_holds_no_archive(
    experiment_session: SessionData,
) -> None:
    """A directory carrying no candidate file states nothing the stage scales with, so the job is refused rather
    than planned at a guess.
    """
    experiment_session.raw_data.behavior_data_path.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError):
        size_session_jobs(
            pipeline=ProcessingPipelines.MICROCONTROLLER,
            session=experiment_session,
            jobs=[(PARSE_JOB_NAME, "52_1_1", 1)],
        )


def test_a_parse_estimate_is_refused_when_the_behavior_directory_is_absent(
    experiment_session: SessionData,
) -> None:
    """A session that recorded no behavior data carries no directory to search, which is refused the same way."""
    assert not experiment_session.raw_data.behavior_data_path.is_dir()

    with pytest.raises(FileNotFoundError):
        size_session_jobs(
            pipeline=ProcessingPipelines.MICROCONTROLLER,
            session=experiment_session,
            jobs=[(PARSE_JOB_NAME, "52_1_1", 1)],
        )


def test_a_video_estimate_charges_every_job_the_widest_recorded_frame(
    experiment_session: SessionData,
    write_grayscale_video: Callable[..., Path],
    write_dlc_predictions: Callable[..., Path],
    moving_block_frames: NDArray[np.uint8],
) -> None:
    """The widest frame is a property of the session, so every motion-energy job of it is charged that frame."""
    camera = experiment_session.raw_data.camera_data_path
    camera.mkdir(parents=True, exist_ok=True)
    write_grayscale_video(camera.joinpath("51_camera.mp4"), moving_block_frames)
    write_grayscale_video(camera.joinpath("73_camera.mp4"), moving_block_frames[:, :40, :32])
    points: Mapping[str, NDArray[np.float64]] = {"eye_top": np.zeros((32, 3), dtype=np.float64)}
    predictions = write_dlc_predictions(camera.joinpath("51_camera.h5"), points)

    estimates = size_session_jobs(
        pipeline=ProcessingPipelines.VIDEO,
        session=experiment_session,
        jobs=[
            (ENERGY_JOB_NAME, "51", 16),
            (TRACKING_JOB_NAME, "51", 1),
            (RENAME_JOB_NAME, "51", 1),
        ],
    )

    widest_pixels = moving_block_frames.shape[1] * moving_block_frames.shape[2]
    per_worker = (
        _bytes_to_megabytes(byte_count=widest_pixels * _SINGLE_PRECISION_BYTES * _RETAINED_FRAME_BUFFERS)
        + _DECODER_BUFFER_MEMORY_MB
        + SPAWNED_CHILD_MEMORY_MB
    )
    assert estimates[ENERGY_JOB_NAME, "51"] == JobFootprint(
        cores=16, memory_mb=_apply_tolerance(memory_mb=WORKER_MEMORY_MB + 16 * per_worker)
    )
    assert estimates[TRACKING_JOB_NAME, "51"] == JobFootprint(
        cores=1,
        memory_mb=_apply_tolerance(
            memory_mb=WORKER_MEMORY_MB
            + _bytes_to_megabytes(byte_count=predictions.stat().st_size * _POSE_PREDICTION_RATIO)
        ),
    )
    # Renaming performs a fixed handful of filesystem operations and reads no recording, so one worker is the whole
    # model rather than a floor standing in for one.
    assert estimates[RENAME_JOB_NAME, "51"] == JobFootprint(cores=1, memory_mb=WORKER_ONLY_MB)


def test_a_video_estimate_charges_the_decoders_when_the_session_recorded_no_camera(
    experiment_session: SessionData,
) -> None:
    """A session carrying no camera directory reports no frame, which leaves motion energy on the per-core decoder
    and child cost its model charges whatever the recording holds, while pose tracking is refused outright.
    """
    estimates = size_session_jobs(
        pipeline=ProcessingPipelines.VIDEO, session=experiment_session, jobs=[(ENERGY_JOB_NAME, "51", 16)]
    )

    energy = estimates[ENERGY_JOB_NAME, "51"]
    # No frame contributes no pixels, so the estimate covers the decoder buffers and the children alone.
    assert energy.memory_mb == _apply_tolerance(
        memory_mb=WORKER_MEMORY_MB + 16 * (_DECODER_BUFFER_MEMORY_MB + SPAWNED_CHILD_MEMORY_MB)
    )
    # Pose tracking reads a prediction file written upstream, and a session holding none names no figure at all.
    with pytest.raises(FileNotFoundError):
        size_session_jobs(
            pipeline=ProcessingPipelines.VIDEO, session=experiment_session, jobs=[(TRACKING_JOB_NAME, "51", 1)]
        )


def test_a_camera_directory_holding_no_recording_reports_no_frame(experiment_session: SessionData) -> None:
    """An empty camera directory is read the same way an absent one is, which is as no recorded frame at all."""
    experiment_session.raw_data.camera_data_path.mkdir(parents=True, exist_ok=True)

    estimates = size_session_jobs(
        pipeline=ProcessingPipelines.VIDEO, session=experiment_session, jobs=[(ENERGY_JOB_NAME, "51", 4)]
    )

    assert estimates[ENERGY_JOB_NAME, "51"].memory_mb == _apply_tolerance(
        memory_mb=WORKER_MEMORY_MB + 4 * (_DECODER_BUFFER_MEMORY_MB + SPAWNED_CHILD_MEMORY_MB)
    )


def test_a_session_job_naming_a_stage_nothing_models_is_refused(experiment_session: SessionData) -> None:
    """Every job of a session is routed to a model of its own, so a name reaching the end of that routing describes
    a stage nothing sizes and is refused rather than admitted to a batch at an allowance nobody chose.
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
    """Each stage reads the shape its own working set follows, which the raw acquisition data reports."""
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
    """A specifier naming no index at all is charged the largest per-plane figure, so it never understates."""
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
    # A specifier that does name a plane the recording holds is charged that plane alone, not the widest one.
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
    """cindra refuses to size a plane the recording never held, and that refusal reaches the caller unchanged."""
    write_surgery_metadata(session=experiment_session)
    write_raw_imaging(session=experiment_session)

    with pytest.raises(ValueError, match="does not hold"):
        two_photon_estimates(
            session=experiment_session,
            jobs=[(str(SingleRecordingJobNames.REGISTER), f"{PLANE_SPECIFIER_PREFIX}99", 8)],
        )


def test_a_recording_declaring_no_regions_is_read_as_one_full_frame_plane(experiment_session: SessionData) -> None:
    """Acquisition parameters naming no region span leave one plane covering the whole acquisition frame."""
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
    """A session with no imaging directory names no geometry, and a recording cindra will not size is a recording
    whose stages cannot run.
    """
    write_surgery_metadata(session=experiment_session)

    with pytest.raises(FileNotFoundError):
        two_photon_estimates(
            session=experiment_session,
            jobs=[(str(SingleRecordingJobNames.BINARIZE), "", 4), (str(SingleRecordingJobNames.COMBINE), "", 1)],
        )


def test_two_photon_jobs_are_refused_when_the_imaging_directory_holds_no_parameters(
    experiment_session: SessionData,
) -> None:
    """An imaging directory carrying no acquisition parameters reports no shape to scale an estimate with."""
    write_surgery_metadata(session=experiment_session)
    experiment_session.raw_data_path.joinpath("mesoscope_data").mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError):
        two_photon_estimates(session=experiment_session, jobs=[(str(SingleRecordingJobNames.BINARIZE), "", 4)])


def test_two_photon_jobs_are_refused_when_every_image_is_excluded(experiment_session: SessionData) -> None:
    """The reader skips the names the configuration excludes, so a directory holding only those reports no image."""
    write_surgery_metadata(session=experiment_session)
    directory = experiment_session.raw_data_path.joinpath("mesoscope_data")
    write_acquisition_parameters(directory=directory, region_lines=REGION_LINES)
    write_imaging_stack(directory=directory, name="zstack.tif", pages=8)

    with pytest.raises(FileNotFoundError):
        two_photon_estimates(session=experiment_session, jobs=[(str(SingleRecordingJobNames.BINARIZE), "", 4)])


def test_two_photon_jobs_are_refused_when_the_acquisition_parameters_cannot_be_read(
    experiment_session: SessionData,
) -> None:
    """A parameters file the reader rejects names no geometry, and that refusal propagates to the caller sizing the
    batch rather than resolving to a figure nothing measured.
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
    """Discovery, extraction, and assembly each scale with the single-recording output their own stage consumes."""
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
    # it was handed. Tracking bounds the regions it retains to the widest single recording the animal holds.
    assert estimates[FORGING_JOB_NAME, first.session_name] == JobFootprint(
        cores=1, memory_mb=assembly_memory(samples=4000, regions=200)
    )


def test_a_written_multi_day_array_replaces_the_tracked_region_bound(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Once the multi-day extraction has written its array, the tracked count is read rather than bounded."""
    session = session_factory(animal_id="305", experiment_name="test_experiment")
    write_surgery_metadata(session=session)
    directory = write_processed_recording(session=session, regions=300, samples=2000)
    dataset = build_dataset(
        project_root=project_root,
        name="ds_tracked",
        sessions=[session],
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
    )
    tracked_directory = directory.joinpath(
        "multi_recording", multi_recording_dataset_directory(animal_id="305", dataset_name=dataset.name)
    )
    write_trace_array(path=tracked_directory.joinpath("cell_fluorescence.npy"), shape=(7, 2000))

    tracked = size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, session.session_name, 1)])[
        FORGING_JOB_NAME, session.session_name
    ]

    # Seven tracked regions cost far less than the three hundred the single recording detected.
    tracked_directory.joinpath("cell_fluorescence.npy").unlink()
    bounded = size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, session.session_name, 1)])[
        FORGING_JOB_NAME, session.session_name
    ]
    # Tracking narrows the region count, a saving the gigabyte rounding absorbs at this fixture's scale.
    assert tracked.memory_mb <= bounded.memory_mb


def test_dataset_stages_are_refused_for_a_session_carrying_no_processed_output(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """A session the single-recording pipeline never wrote for names no geometry to scale any stage with, so every
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
    for job in (
        (MULTIDAY_DISCOVERY_JOB_NAME, "305", 30),
        (MULTIDAY_EXTRACTION_JOB_NAME, session.session_name, 16),
        (FORGING_JOB_NAME, session.session_name, 1),
    ):
        with pytest.raises(FileNotFoundError):
            size_dataset_jobs(dataset=dataset, jobs=[job])


def test_extraction_is_refused_for_a_dataset_its_system_performs_no_tracking_for(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Mesoscope-VR tracks cells across experiment sessions alone, so a training dataset resolves no configuration."""
    session = session_factory(animal_id="321", session_type=SessionTypes.RUN_TRAINING)
    write_surgery_metadata(session=session)
    write_processed_recording(session=session, regions=64, samples=1000)
    dataset = build_dataset(
        project_root=project_root, name="ds_training", sessions=[session], session_type=SessionTypes.RUN_TRAINING
    )

    # Extraction is sized against the configuration, so a dataset donating none names no figure for the stage.
    with pytest.raises(ValueError, match="resolved no multi-recording"):
        size_dataset_jobs(dataset=dataset, jobs=[(MULTIDAY_EXTRACTION_JOB_NAME, session.session_name, 16)])

    estimates = size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, session.session_name, 1)])

    # Assembly needs no configuration, so it stays sized from the geometry the session's own output reports.
    assert estimates[FORGING_JOB_NAME, session.session_name] == JobFootprint(
        cores=1, memory_mb=assembly_memory(samples=1000, regions=64)
    )


def test_a_dataset_naming_no_session_resolves_no_tracking_configuration(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """A dataset holding no session donates no configuration, which leaves its cross-recording stages unsizable."""
    session = session_factory(animal_id="305", experiment_name="test_experiment")
    write_surgery_metadata(session=session)
    write_processed_recording(session=session, regions=50, samples=800)
    dataset = build_dataset(
        project_root=project_root, name="ds_empty", sessions=[session], session_type=SessionTypes.MESOSCOPE_EXPERIMENT
    )
    dataset.sessions = ()
    dataset.save()
    emptied = DatasetData.load(dataset_path=dataset.dataset_data_path.parent)

    # No session leaves no recording directory to hand cindra and no session to resolve a configuration from.
    with pytest.raises(ValueError, match="resolved no multi-recording"):
        size_dataset_jobs(dataset=emptied, jobs=[(MULTIDAY_DISCOVERY_JOB_NAME, "305", 30)])


def test_a_session_that_left_the_project_root_contributes_no_recording(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """A dataset outlives the source data of the animals it has forged, so a session moved to long-term storage is
    skipped rather than resolved.
    """
    first = session_factory(animal_id="305", experiment_name="test_experiment")
    second = session_factory(animal_id="305", experiment_name="test_experiment")
    for session in (first, second):
        write_surgery_metadata(session=session)
        write_processed_recording(session=session, regions=120, samples=900)
    dataset = build_dataset(
        project_root=project_root,
        name="ds_relocated",
        sessions=[first, second],
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
    )
    shutil.rmtree(project_root.joinpath("305", first.session_name))

    estimates = size_dataset_jobs(dataset=dataset, jobs=[(MULTIDAY_DISCOVERY_JOB_NAME, "305", 30)])

    # The surviving recording is the whole set the stage is sized from, and the configuration is resolved from it too.
    assert estimates[MULTIDAY_DISCOVERY_JOB_NAME, "305"] == cindra_multi_recording_footprint(
        dataset=dataset, sessions=[second], job_name=MultiRecordingJobNames.DISCOVER, specifier="305"
    )


def test_a_recording_whose_metadata_is_absent_is_refused(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """The combined field extent is read from the metadata archive, so an output missing it names no geometry and
    the assembly job that would read it is refused.
    """
    session = session_factory(animal_id="305", experiment_name="test_experiment")
    write_surgery_metadata(session=session)
    write_trace_array(path=session.processed_data.cindra_data_path.joinpath("cell_fluorescence.npy"), shape=(120, 900))
    dataset = build_dataset(
        project_root=project_root,
        name="ds_partial",
        sessions=[session],
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
    )

    with pytest.raises(FileNotFoundError):
        size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, session.session_name, 1)])


def test_a_trace_array_of_another_rank_is_refused(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """A fluorescence array carrying a rank other than regions by samples is not read as a recording geometry."""
    session = session_factory(animal_id="305", experiment_name="test_experiment")
    write_surgery_metadata(session=session)
    directory = session.processed_data.cindra_data_path
    write_trace_array(path=directory.joinpath("cell_fluorescence.npy"), shape=(120,))
    write_combined_metadata(directory=directory, height=64, width=64)
    dataset = build_dataset(
        project_root=project_root, name="ds_rank", sessions=[session], session_type=SessionTypes.MESOSCOPE_EXPERIMENT
    )

    with pytest.raises(FileNotFoundError):
        size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, session.session_name, 1)])


def test_a_second_format_version_header_is_parsed_the_same_way(tmp_path: Path) -> None:
    """Array headers are parsed at whichever format version wrote them, so both versions report the same extents."""
    first = write_trace_array(path=tmp_path.joinpath("first.npy"), shape=(12, 34), version=(1, 0))
    second = write_trace_array(path=tmp_path.joinpath("second.npy"), shape=(12, 34), version=(2, 0))

    assert _read_array_shape(array_path=first) == (12, 34)
    assert _read_array_shape(array_path=second) == (12, 34)
    assert _read_array_shape(array_path=tmp_path.joinpath("absent.npy")) is None


def test_the_pooled_region_bound_narrows_to_one_recordings_own_regions(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """A template clusters regions drawn from several recordings, so the bound settles at one recording's scale."""
    sessions = [session_factory(animal_id="305", experiment_name="test_experiment") for _ in range(4)]
    for index, session in enumerate(sessions):
        write_surgery_metadata(session=session)
        write_processed_recording(session=session, regions=100 + index, samples=1500)
    dataset = build_dataset(
        project_root=project_root,
        name="ds_pooled",
        sessions=sessions,
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
    )

    estimates = size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, sessions[0].session_name, 1)])

    # Half prevalence over four recordings pools 406 regions into 203, which the widest recording narrows to 103.
    assert estimates[FORGING_JOB_NAME, sessions[0].session_name] == JobFootprint(
        cores=1, memory_mb=assembly_memory(samples=1500, regions=103)
    )


def test_an_assembly_job_carrying_an_animal_specifier_bounds_its_regions_at_one(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """The discovery stage is specified by its animal while assembly is specified by its session, so a specifier
    naming an animal matches no session the dataset lists and resolves no animal to pool recordings from. The bound
    settles at the one template an empty recording set allows rather than failing the whole batch's sizing pass.
    """
    session = session_factory(animal_id="305", experiment_name="test_experiment")
    write_surgery_metadata(session=session)
    write_processed_recording(session=session, regions=250, samples=1100)
    dataset = build_dataset(
        project_root=project_root,
        name="ds_animal_specifier",
        sessions=[session],
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
    )

    estimates = size_dataset_jobs(dataset=dataset, jobs=[(FORGING_JOB_NAME, "305", 1)])

    # The animal directory resolves the recording whose samples the job is charged, while the regions it is charged
    # across are the single template rather than the two hundred and fifty that recording detected. The two figures
    # differ by a saving the gigabyte rounding absorbs at this fixture's scale.
    assert estimates[FORGING_JOB_NAME, "305"] == JobFootprint(
        cores=1, memory_mb=assembly_memory(samples=1100, regions=1)
    )


def test_a_dataset_job_naming_a_stage_nothing_models_is_refused(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Every forging job is routed to a model of its own the same way a session's jobs are, so a name reaching the
    end of that routing is refused rather than admitted to a batch at an allowance nobody chose.
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


# Shared conversions


def test_byte_conversion_rounds_up_and_reports_nothing_for_nothing() -> None:
    """A converted byte count never understates its demand, while an empty input converts to no memory at all."""
    assert _bytes_to_megabytes(byte_count=0) == 0
    assert _bytes_to_megabytes(byte_count=1) == 1
    assert _bytes_to_megabytes(byte_count=1024 * 1024) == 2


def test_the_camera_directory_is_read_once_even_when_it_is_removed_mid_session(
    experiment_session: SessionData,
    write_grayscale_video: Callable[..., Path],
    moving_block_frames: NDArray[np.uint8],
) -> None:
    """The widest frame is resolved before any job is sized, so removing the recordings does not change a figure."""
    camera = experiment_session.raw_data.camera_data_path
    camera.mkdir(parents=True, exist_ok=True)
    write_grayscale_video(camera.joinpath("51_camera.mp4"), moving_block_frames)

    with_recordings = size_session_jobs(
        pipeline=ProcessingPipelines.VIDEO, session=experiment_session, jobs=[(ENERGY_JOB_NAME, "51", 4)]
    )
    shutil.rmtree(camera)
    without_recordings = size_session_jobs(
        pipeline=ProcessingPipelines.VIDEO, session=experiment_session, jobs=[(ENERGY_JOB_NAME, "51", 4)]
    )

    # A recorded frame adds pixels on top of the decoder and child cost, which the gigabyte rounding absorbs here.
    assert with_recordings[ENERGY_JOB_NAME, "51"].memory_mb >= without_recordings[ENERGY_JOB_NAME, "51"].memory_mb
