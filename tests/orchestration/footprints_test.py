"""Tests the memory estimators that size each job's working set from the acquisition and processed data it reads."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import numpy as np
from cindra import SingleRecordingJobNames
import pytest
import tifffile
from cindra.io import PARAMETERS_FILENAME
from cindra.allocation import PLANE_SPECIFIER_PREFIX
from sollertia_shared_assets import (
    DatasetData,
    SessionData,
    SurgeryData,
    SessionTypes,
    DatasetSession,
)
from sollertia_shared_assets.data_classes.surgery_data import SubjectData, ProcedureData

from sollertia_forgery.video import ENERGY_JOB_NAME, RENAME_JOB_NAME, TRACKING_JOB_NAME, TIMESTAMP_JOB_NAME
from sollertia_forgery.forging import (
    FORGING_JOB_NAME,
    MULTIDAY_DISCOVERY_JOB_NAME,
    MULTIDAY_EXTRACTION_JOB_NAME,
)
from sollertia_forgery.runtime import RUNTIME_JOB_NAME
from sollertia_forgery.managing import CHECKSUM_JOB_NAME
from sollertia_forgery.shared_assets import ProcessingPipelines, multi_recording_dataset_directory
from sollertia_forgery.microcontrollers import PARSE_JOB_NAME, EXTRACTION_JOB_NAME
from sollertia_forgery.orchestration.footprints import (
    _WORKER_MEMORY_MB,
    _SUBPROCESS_MEMORY_MB,
    _COMBINATION_MEMORY_MB,
    _POSE_PREDICTION_RATIO,
    _RETAINED_FRAME_BUFFERS,
    _SINGLE_PRECISION_BYTES,
    _ARCHIVE_DIRECTORY_RATIO,
    _DECODER_BUFFER_MEMORY_MB,
    _DISCOVERY_CLUSTERING_MEMORY_MB,
    _apply_tolerance,
    _read_array_shape,
    _bytes_to_megabytes,
    _resolve_plane_index,
    estimate_dataset_job_memory,
    estimate_session_job_memory,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Mapping, Callable, Sequence

    from numpy.typing import NDArray

BASELINE_MB: int = _apply_tolerance(memory_mb=_WORKER_MEMORY_MB)
"""The figure every estimate falls back to when the input it would scale with is absent."""

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


def two_photon_estimates(
    session: SessionData, jobs: list[tuple[str, str, int]]
) -> dict[tuple[str, str], tuple[int, bool]]:
    """Estimates the named two-photon jobs of one session.

    Args:
        session: The session the jobs operate on.
        jobs: The jobs as name, specifier, and core triples.

    Returns:
        The estimate and modeled flag of every named job.
    """
    return estimate_session_job_memory(pipeline=ProcessingPipelines.TWO_PHOTON, session=session, jobs=jobs)


# Session estimates


def test_checksum_memory_scales_with_the_readers_a_job_opens(experiment_session: SessionData) -> None:
    """The checksum estimate follows the cores a job holds, since each core streams one file in fixed chunks."""
    estimates = estimate_session_job_memory(
        pipeline=ProcessingPipelines.CHECKSUM,
        session=experiment_session,
        jobs=[(CHECKSUM_JOB_NAME, "", 8), (CHECKSUM_JOB_NAME, "wide", 16)],
    )

    narrow, narrow_modeled = estimates[CHECKSUM_JOB_NAME, ""]
    wide, wide_modeled = estimates[CHECKSUM_JOB_NAME, "wide"]
    assert narrow == _apply_tolerance(memory_mb=_WORKER_MEMORY_MB + 8 * 56)
    assert wide > narrow
    assert narrow_modeled
    assert wide_modeled


def test_an_archive_reader_estimate_scales_with_the_archive_on_disk(
    experiment_session: SessionData, write_log_archive: Callable[..., Path]
) -> None:
    """Every worker of a log-reading stage opens the archive itself, so the estimate is charged once per core."""
    archive = write_log_archive(experiment_session.raw_data.behavior_data_path.joinpath("51_log.npz"), 51, [(5, b"ab")])
    per_reader = _bytes_to_megabytes(byte_count=archive.stat().st_size * _ARCHIVE_DIRECTORY_RATIO)

    estimates = estimate_session_job_memory(
        pipeline=ProcessingPipelines.RUNTIME,
        session=experiment_session,
        jobs=[(RUNTIME_JOB_NAME, "51", 4), (EXTRACTION_JOB_NAME, "51", 8), (TIMESTAMP_JOB_NAME, "77", 8)],
    )

    assert estimates[RUNTIME_JOB_NAME, "51"] == (
        _apply_tolerance(memory_mb=_WORKER_MEMORY_MB + 4 * (per_reader + _SUBPROCESS_MEMORY_MB)),
        True,
    )
    assert estimates[EXTRACTION_JOB_NAME, "51"][0] > estimates[RUNTIME_JOB_NAME, "51"][0]
    # The camera archive of source 77 was never written, so its estimate falls back to the worker baseline.
    assert estimates[TIMESTAMP_JOB_NAME, "77"] == (BASELINE_MB, False)


def test_a_parse_estimate_follows_the_widest_archive_in_the_behavior_directory(
    experiment_session: SessionData, write_log_archive: Callable[..., Path]
) -> None:
    """A parse job reads one module's share of its controller's archive, so the widest archive bounds it."""
    behavior = experiment_session.raw_data.behavior_data_path
    write_log_archive(behavior.joinpath("51_log.npz"), 51, [(1, b"a")])
    widest = write_log_archive(behavior.joinpath("52_log.npz"), 52, [(index, bytes(400)) for index in range(20)])

    estimates = estimate_session_job_memory(
        pipeline=ProcessingPipelines.MICROCONTROLLER,
        session=experiment_session,
        jobs=[(PARSE_JOB_NAME, "52_1_1", 1)],
    )

    assert widest.stat().st_size > behavior.joinpath("51_log.npz").stat().st_size
    modeled, is_modeled = estimates[PARSE_JOB_NAME, "52_1_1"]
    assert modeled == _apply_tolerance(memory_mb=_WORKER_MEMORY_MB + _bytes_to_megabytes(widest.stat().st_size * 3.4))
    assert is_modeled


def test_a_parse_estimate_falls_back_when_the_behavior_directory_holds_no_archive(
    experiment_session: SessionData,
) -> None:
    """A directory carrying no candidate file leaves the stage on the worker baseline rather than on a guess."""
    experiment_session.raw_data.behavior_data_path.mkdir(parents=True, exist_ok=True)

    estimates = estimate_session_job_memory(
        pipeline=ProcessingPipelines.MICROCONTROLLER,
        session=experiment_session,
        jobs=[(PARSE_JOB_NAME, "52_1_1", 1)],
    )

    assert estimates[PARSE_JOB_NAME, "52_1_1"] == (BASELINE_MB, False)


def test_a_parse_estimate_falls_back_when_the_behavior_directory_is_absent(
    experiment_session: SessionData,
) -> None:
    """A session that recorded no behavior data carries no directory to search, which is the same floor."""
    assert not experiment_session.raw_data.behavior_data_path.is_dir()

    estimates = estimate_session_job_memory(
        pipeline=ProcessingPipelines.MICROCONTROLLER,
        session=experiment_session,
        jobs=[(PARSE_JOB_NAME, "52_1_1", 1)],
    )

    assert estimates[PARSE_JOB_NAME, "52_1_1"] == (BASELINE_MB, False)


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

    estimates = estimate_session_job_memory(
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
        + _SUBPROCESS_MEMORY_MB
    )
    assert estimates[ENERGY_JOB_NAME, "51"] == (_apply_tolerance(memory_mb=_WORKER_MEMORY_MB + 16 * per_worker), True)
    assert estimates[TRACKING_JOB_NAME, "51"] == (
        _apply_tolerance(
            memory_mb=_WORKER_MEMORY_MB
            + _bytes_to_megabytes(byte_count=predictions.stat().st_size * _POSE_PREDICTION_RATIO)
        ),
        True,
    )
    # Renaming performs a fixed handful of filesystem operations, so it models nothing and takes the floor.
    assert estimates[RENAME_JOB_NAME, "51"] == (BASELINE_MB, False)


def test_a_video_estimate_falls_back_when_the_session_recorded_no_camera(experiment_session: SessionData) -> None:
    """A session carrying no camera directory reports no frame, which leaves motion energy on the worker baseline."""
    estimates = estimate_session_job_memory(
        pipeline=ProcessingPipelines.VIDEO,
        session=experiment_session,
        jobs=[(ENERGY_JOB_NAME, "51", 16), (TRACKING_JOB_NAME, "51", 1)],
    )

    energy, energy_modeled = estimates[ENERGY_JOB_NAME, "51"]
    # No frame contributes no pixels, so the estimate covers the decoder buffers and the children alone.
    assert energy == _apply_tolerance(
        memory_mb=_WORKER_MEMORY_MB + 16 * (_DECODER_BUFFER_MEMORY_MB + _SUBPROCESS_MEMORY_MB)
    )
    assert energy_modeled
    assert estimates[TRACKING_JOB_NAME, "51"] == (BASELINE_MB, False)


def test_a_camera_directory_holding_no_recording_reports_no_frame(experiment_session: SessionData) -> None:
    """An empty camera directory is read the same way an absent one is, which is as no recorded frame at all."""
    experiment_session.raw_data.camera_data_path.mkdir(parents=True, exist_ok=True)

    estimates = estimate_session_job_memory(
        pipeline=ProcessingPipelines.VIDEO, session=experiment_session, jobs=[(ENERGY_JOB_NAME, "51", 4)]
    )

    assert estimates[ENERGY_JOB_NAME, "51"][0] == _apply_tolerance(
        memory_mb=_WORKER_MEMORY_MB + 4 * (_DECODER_BUFFER_MEMORY_MB + _SUBPROCESS_MEMORY_MB)
    )


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

    # Conversion reads one batch of the unsliced frame at a time, which the configured batch size and the frame set.
    binarize_bytes = 100 * FRAME_HEIGHT * FRAME_WIDTH * 2 * 2
    assert estimates[str(SingleRecordingJobNames.BINARIZE), ""] == (
        _apply_tolerance(memory_mb=_WORKER_MEMORY_MB + _bytes_to_megabytes(byte_count=binarize_bytes)),
        True,
    )
    # Combination takes a flat allowance, which the shared tolerance does not apply on top of.
    assert estimates[str(SingleRecordingJobNames.COMBINE), ""] == (_COMBINATION_MEMORY_MB, True)
    # The second region spans twice the lines of the first, so its registration batch costs twice as much.
    first_plane = estimates[str(SingleRecordingJobNames.REGISTER), f"{PLANE_SPECIFIER_PREFIX}0"][0]
    second_plane = estimates[str(SingleRecordingJobNames.REGISTER), f"{PLANE_SPECIFIER_PREFIX}1"][0]
    assert second_plane > first_plane
    assert estimates[str(SingleRecordingJobNames.PROCESS), f"{PLANE_SPECIFIER_PREFIX}0"][1]


def test_a_plane_job_whose_specifier_does_not_resolve_takes_the_widest_plane(
    experiment_session: SessionData,
) -> None:
    """An unmatched specifier is charged the largest per-plane figure, so it never understates its demand."""
    write_surgery_metadata(session=experiment_session)
    write_raw_imaging(session=experiment_session)

    estimates = two_photon_estimates(
        session=experiment_session,
        jobs=[
            (str(SingleRecordingJobNames.REGISTER), f"{PLANE_SPECIFIER_PREFIX}1", 8),
            (str(SingleRecordingJobNames.REGISTER), f"{PLANE_SPECIFIER_PREFIX}9", 8),
            (str(SingleRecordingJobNames.REGISTER), "an_unstructured_specifier", 8),
            (str(SingleRecordingJobNames.PROCESS), f"{PLANE_SPECIFIER_PREFIX}x", 10),
            (str(SingleRecordingJobNames.PROCESS), f"{PLANE_SPECIFIER_PREFIX}1", 10),
        ],
    )

    widest_registration = estimates[str(SingleRecordingJobNames.REGISTER), f"{PLANE_SPECIFIER_PREFIX}1"][0]
    # An index past the last plane and a specifier naming no index both resolve to the widest plane's figure.
    assert estimates[str(SingleRecordingJobNames.REGISTER), f"{PLANE_SPECIFIER_PREFIX}9"][0] == widest_registration
    assert estimates[str(SingleRecordingJobNames.REGISTER), "an_unstructured_specifier"][0] == widest_registration
    assert (
        estimates[str(SingleRecordingJobNames.PROCESS), f"{PLANE_SPECIFIER_PREFIX}x"][0]
        == estimates[str(SingleRecordingJobNames.PROCESS), f"{PLANE_SPECIFIER_PREFIX}1"][0]
    )


def test_a_recording_declaring_no_regions_is_read_as_one_full_frame_plane(experiment_session: SessionData) -> None:
    """Acquisition parameters naming no region span leave one plane covering the whole acquisition frame."""
    write_surgery_metadata(session=experiment_session)
    write_raw_imaging(session=experiment_session, region_lines=None)

    estimates = two_photon_estimates(
        session=experiment_session, jobs=[(str(SingleRecordingJobNames.REGISTER), f"{PLANE_SPECIFIER_PREFIX}0", 8)]
    )

    batch_bytes = 100 * FRAME_HEIGHT * FRAME_WIDTH * _SINGLE_PRECISION_BYTES * 3
    assert estimates[str(SingleRecordingJobNames.REGISTER), f"{PLANE_SPECIFIER_PREFIX}0"] == (
        _apply_tolerance(memory_mb=_WORKER_MEMORY_MB + _bytes_to_megabytes(byte_count=batch_bytes)),
        True,
    )


def test_two_photon_jobs_fall_back_when_the_session_holds_no_raw_imaging(experiment_session: SessionData) -> None:
    """A session with no imaging directory names no geometry, so every stage takes the worker baseline."""
    write_surgery_metadata(session=experiment_session)

    estimates = two_photon_estimates(
        session=experiment_session,
        jobs=[(str(SingleRecordingJobNames.BINARIZE), "", 4), (str(SingleRecordingJobNames.COMBINE), "", 1)],
    )

    assert estimates[str(SingleRecordingJobNames.BINARIZE), ""] == (BASELINE_MB, False)
    assert estimates[str(SingleRecordingJobNames.COMBINE), ""] == (BASELINE_MB, False)


def test_two_photon_jobs_fall_back_when_the_imaging_directory_holds_no_parameters(
    experiment_session: SessionData,
) -> None:
    """An imaging directory carrying no acquisition parameters reports no shape to scale an estimate with."""
    write_surgery_metadata(session=experiment_session)
    experiment_session.raw_data_path.joinpath("mesoscope_data").mkdir(parents=True, exist_ok=True)

    estimates = two_photon_estimates(session=experiment_session, jobs=[(str(SingleRecordingJobNames.BINARIZE), "", 4)])

    assert estimates[str(SingleRecordingJobNames.BINARIZE), ""] == (BASELINE_MB, False)


def test_two_photon_jobs_fall_back_when_every_image_is_excluded(experiment_session: SessionData) -> None:
    """The reader skips the names the configuration excludes, so a directory holding only those reports no image."""
    write_surgery_metadata(session=experiment_session)
    directory = experiment_session.raw_data_path.joinpath("mesoscope_data")
    write_acquisition_parameters(directory=directory, region_lines=REGION_LINES)
    write_imaging_stack(directory=directory, name="zstack.tif", pages=8)

    estimates = two_photon_estimates(session=experiment_session, jobs=[(str(SingleRecordingJobNames.BINARIZE), "", 4)])

    assert estimates[str(SingleRecordingJobNames.BINARIZE), ""] == (BASELINE_MB, False)


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

    estimates = estimate_dataset_job_memory(
        dataset=dataset,
        jobs=[
            (MULTIDAY_DISCOVERY_JOB_NAME, "305", 30),
            (MULTIDAY_EXTRACTION_JOB_NAME, first.session_name, 16),
            (FORGING_JOB_NAME, first.session_name, 1),
        ],
    )

    discovery, discovery_modeled = estimates[MULTIDAY_DISCOVERY_JOB_NAME, "305"]
    # Two recordings hold one pairwise deformation plus the per-recording planes, over the combined field extent.
    planes = 2 * (2 - 1) + 12 * 2
    registration = _bytes_to_megabytes(byte_count=planes * 128 * 96 * _SINGLE_PRECISION_BYTES)
    assert discovery == _apply_tolerance(memory_mb=_WORKER_MEMORY_MB + registration + _DISCOVERY_CLUSTERING_MEMORY_MB)
    assert discovery_modeled
    assert estimates[MULTIDAY_EXTRACTION_JOB_NAME, first.session_name][1]
    assert estimates[MULTIDAY_EXTRACTION_JOB_NAME, first.session_name][0] > BASELINE_MB
    assert estimates[FORGING_JOB_NAME, first.session_name][1]
    assert estimates[FORGING_JOB_NAME, first.session_name][0] > BASELINE_MB


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

    tracked = estimate_dataset_job_memory(dataset=dataset, jobs=[(FORGING_JOB_NAME, session.session_name, 1)])[
        FORGING_JOB_NAME, session.session_name
    ]

    # Seven tracked regions cost far less than the three hundred the single recording detected.
    tracked_directory.joinpath("cell_fluorescence.npy").unlink()
    bounded = estimate_dataset_job_memory(dataset=dataset, jobs=[(FORGING_JOB_NAME, session.session_name, 1)])[
        FORGING_JOB_NAME, session.session_name
    ]
    assert tracked[0] < bounded[0]
    assert tracked[1]


def test_dataset_stages_fall_back_for_a_session_carrying_no_processed_output(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """A session the single-recording pipeline never wrote for names no geometry to scale any stage with."""
    session = session_factory(animal_id="305", experiment_name="test_experiment")
    write_surgery_metadata(session=session)
    dataset = build_dataset(
        project_root=project_root,
        name="ds_unprocessed",
        sessions=[session],
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
    )

    estimates = estimate_dataset_job_memory(
        dataset=dataset,
        jobs=[
            (MULTIDAY_DISCOVERY_JOB_NAME, "305", 30),
            (MULTIDAY_EXTRACTION_JOB_NAME, session.session_name, 16),
            (FORGING_JOB_NAME, session.session_name, 1),
        ],
    )

    assert estimates[MULTIDAY_DISCOVERY_JOB_NAME, "305"] == (
        _apply_tolerance(memory_mb=_WORKER_MEMORY_MB + _DISCOVERY_CLUSTERING_MEMORY_MB),
        False,
    )
    assert estimates[MULTIDAY_EXTRACTION_JOB_NAME, session.session_name] == (BASELINE_MB, False)
    assert estimates[FORGING_JOB_NAME, session.session_name] == (BASELINE_MB, False)


def test_extraction_falls_back_for_a_dataset_its_system_performs_no_tracking_for(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """Mesoscope-VR tracks cells across experiment sessions alone, so a training dataset resolves no configuration."""
    session = session_factory(animal_id="321", session_type=SessionTypes.RUN_TRAINING)
    write_surgery_metadata(session=session)
    write_processed_recording(session=session, regions=64, samples=1000)
    dataset = build_dataset(
        project_root=project_root, name="ds_training", sessions=[session], session_type=SessionTypes.RUN_TRAINING
    )

    estimates = estimate_dataset_job_memory(
        dataset=dataset,
        jobs=[
            (MULTIDAY_EXTRACTION_JOB_NAME, session.session_name, 16),
            (FORGING_JOB_NAME, session.session_name, 1),
        ],
    )

    # Extraction reads its batch width from the configuration, so no configuration leaves it on the floor.
    assert estimates[MULTIDAY_EXTRACTION_JOB_NAME, session.session_name] == (BASELINE_MB, False)
    # Assembly needs no configuration, so it stays sized from the geometry the session's own output reports.
    assert estimates[FORGING_JOB_NAME, session.session_name][1]
    assert estimates[FORGING_JOB_NAME, session.session_name][0] > BASELINE_MB


def test_a_dataset_naming_no_session_resolves_no_tracking_configuration(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """A dataset holding no session donates no configuration, and its planned jobs still resolve to a figure."""
    session = session_factory(animal_id="305", experiment_name="test_experiment")
    write_surgery_metadata(session=session)
    write_processed_recording(session=session, regions=50, samples=800)
    dataset = build_dataset(
        project_root=project_root, name="ds_empty", sessions=[session], session_type=SessionTypes.MESOSCOPE_EXPERIMENT
    )
    dataset.sessions = ()
    dataset.save()
    emptied = DatasetData.load(dataset_path=dataset.dataset_data_path.parent)

    estimates = estimate_dataset_job_memory(dataset=emptied, jobs=[(MULTIDAY_DISCOVERY_JOB_NAME, "305", 30)])

    assert estimates[MULTIDAY_DISCOVERY_JOB_NAME, "305"] == (
        _apply_tolerance(memory_mb=_WORKER_MEMORY_MB + _DISCOVERY_CLUSTERING_MEMORY_MB),
        False,
    )


def test_a_recording_whose_metadata_is_absent_reports_no_geometry(
    project_root: Path, session_factory: Callable[..., SessionData]
) -> None:
    """The combined field extent is read from the metadata archive, so an output missing it names no geometry."""
    session = session_factory(animal_id="305", experiment_name="test_experiment")
    write_surgery_metadata(session=session)
    write_trace_array(path=session.processed_data.cindra_data_path.joinpath("cell_fluorescence.npy"), shape=(120, 900))
    dataset = build_dataset(
        project_root=project_root,
        name="ds_partial",
        sessions=[session],
        session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
    )

    estimates = estimate_dataset_job_memory(dataset=dataset, jobs=[(FORGING_JOB_NAME, session.session_name, 1)])

    assert estimates[FORGING_JOB_NAME, session.session_name] == (BASELINE_MB, False)


def test_a_trace_array_of_another_rank_reports_no_geometry(
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

    estimates = estimate_dataset_job_memory(dataset=dataset, jobs=[(FORGING_JOB_NAME, session.session_name, 1)])

    assert estimates[FORGING_JOB_NAME, session.session_name] == (BASELINE_MB, False)


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

    estimates = estimate_dataset_job_memory(
        dataset=dataset, jobs=[(MULTIDAY_EXTRACTION_JOB_NAME, sessions[0].session_name, 16)]
    )

    # Half prevalence over four recordings pools 406 regions into 203, which the widest recording narrows to 103.
    traces = 4 * 103 * 1500 * _SINGLE_PRECISION_BYTES
    retained = 20 * (500 * 128 * 96 * 6)
    assert estimates[MULTIDAY_EXTRACTION_JOB_NAME, sessions[0].session_name] == (
        _apply_tolerance(memory_mb=_WORKER_MEMORY_MB + _bytes_to_megabytes(byte_count=traces + retained)),
        True,
    )


# Shared conversions


def test_byte_conversion_rounds_up_and_reports_nothing_for_nothing() -> None:
    """A converted byte count never understates its demand, while an empty input converts to no memory at all."""
    assert _bytes_to_megabytes(byte_count=0) == 0
    assert _bytes_to_megabytes(byte_count=1) == 1
    assert _bytes_to_megabytes(byte_count=1024 * 1024) == 2


@pytest.mark.parametrize(
    ("specifier", "expected"),
    [
        (f"{PLANE_SPECIFIER_PREFIX}0", 0),
        (f"{PLANE_SPECIFIER_PREFIX}12", 12),
        (f"{PLANE_SPECIFIER_PREFIX}x", None),
        ("", None),
        ("registration", None),
    ],
)
def test_a_plane_specifier_reports_the_index_it_names(specifier: str, expected: int | None) -> None:
    """A per-plane job names its plane behind the shared prefix, and anything else names no plane."""
    assert _resolve_plane_index(specifier=specifier) == expected


def test_the_camera_directory_is_read_once_even_when_it_is_removed_mid_session(
    experiment_session: SessionData,
    write_grayscale_video: Callable[..., Path],
    moving_block_frames: NDArray[np.uint8],
) -> None:
    """The widest frame is resolved before any job is sized, so removing the recordings does not change a figure."""
    camera = experiment_session.raw_data.camera_data_path
    camera.mkdir(parents=True, exist_ok=True)
    write_grayscale_video(camera.joinpath("51_camera.mp4"), moving_block_frames)

    with_recordings = estimate_session_job_memory(
        pipeline=ProcessingPipelines.VIDEO, session=experiment_session, jobs=[(ENERGY_JOB_NAME, "51", 4)]
    )
    shutil.rmtree(camera)
    without_recordings = estimate_session_job_memory(
        pipeline=ProcessingPipelines.VIDEO, session=experiment_session, jobs=[(ENERGY_JOB_NAME, "51", 4)]
    )

    assert with_recordings[ENERGY_JOB_NAME, "51"][0] > without_recordings[ENERGY_JOB_NAME, "51"][0]
