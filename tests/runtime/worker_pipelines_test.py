"""Tests the system-agnostic runtime log processing pipeline and the archive decoder that feeds it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from sollertia_shared_assets import SessionTypes
from ataraxis_data_structures import (
    LOG_ARCHIVE_SUFFIX,
    LogArchiveReader,
    ProcessingStatus,
    ProcessingTracker,
)

from sollertia_forgery.runtime import (
    RUNTIME_JOB_NAME,
    discover_runtime_jobs,
    runtime_job_prerequisites,
    run_runtime_processing_pipeline,
)
from sollertia_forgery.runtime.pipeline import _decode_batch, _decode_archive
from sollertia_forgery.mesoscope_vr.runtime import RUNTIME_SOURCE_ID
from sollertia_forgery.mesoscope_vr.metadata import BehaviorDataFiles

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from sollertia_shared_assets import SessionData

_PARALLEL_MESSAGE_COUNT: int = 2000
"""The number of messages an archive needs before the reader splits it into more than one batch, which is what selects
the pipeline's parallel decode path."""

_SYSTEM_STATE_CODE: int = 1
"""The leading payload byte the Mesoscope-VR parser routes to the system-state feather."""

_RUNTIME_STATE_CODE: int = 2
"""The leading payload byte the Mesoscope-VR parser routes to the runtime-state feather."""


def _session_path(session: SessionData) -> Path:
    """Resolves the root session directory every pipeline entry point takes as its argument.

    Args:
        session: The loaded session whose root directory to resolve.

    Returns:
        The path to the session directory holding the raw and processed data trees.
    """
    return session.raw_data_path.parent


def _archive_path(session: SessionData) -> Path:
    """Resolves where the Mesoscope-VR runtime DataLogger archive lives for the target session.

    Args:
        session: The loaded session whose runtime archive path to resolve.

    Returns:
        The path to the session's ``1_log.npz`` archive inside its raw behavior-data directory.
    """
    return session.raw_data.behavior_data_path.joinpath(f"{RUNTIME_SOURCE_ID}{LOG_ARCHIVE_SUFFIX}")


@pytest.fixture
def runtime_archive(training_session: SessionData, write_log_archive: Callable[..., Path]) -> Path:
    """Writes a small Mesoscope-VR runtime archive carrying two system-state and one runtime-state message.

    Args:
        training_session: The session the archive is written for.
        write_log_archive: The writer that builds a real DataLogger archive.

    Returns:
        The path to the written archive.
    """
    return write_log_archive(
        _archive_path(training_session),
        int(RUNTIME_SOURCE_ID),
        [
            (100, bytes([_SYSTEM_STATE_CODE, 1])),
            (200, bytes([_RUNTIME_STATE_CODE, 7])),
            (300, bytes([_SYSTEM_STATE_CODE, 2])),
        ],
    )


@pytest.fixture
def parallel_archive(tmp_path: Path, write_log_archive: Callable[..., Path]) -> Path:
    """Writes an archive large enough for the reader to split it across more than one decode batch.

    Args:
        tmp_path: The directory the archive is written under.
        write_log_archive: The writer that builds a real DataLogger archive.

    Returns:
        The path to the written archive.
    """
    messages = [(index * 10, bytes([_SYSTEM_STATE_CODE, index % 251])) for index in range(_PARALLEL_MESSAGE_COUNT)]
    return write_log_archive(
        tmp_path.joinpath(f"{RUNTIME_SOURCE_ID}{LOG_ARCHIVE_SUFFIX}"), int(RUNTIME_SOURCE_ID), messages
    )


def test_discovering_runtime_jobs_reports_the_single_source_job(training_session: SessionData) -> None:
    """The universe is always the one runtime job keyed by the acquisition system's source id."""
    session, universe, possible = discover_runtime_jobs(session_path=_session_path(training_session))

    assert session.session_name == training_session.session_name
    assert universe == [(RUNTIME_JOB_NAME, RUNTIME_SOURCE_ID)]
    # No archive has been written yet, so the single job is not possible.
    assert possible == []


def test_discovering_runtime_jobs_admits_the_job_once_the_archive_exists(
    training_session: SessionData, runtime_archive: Path
) -> None:
    """A present archive is the only condition the discovery step tests, so the possible subset fills in."""
    assert runtime_archive.is_file()

    _session, universe, possible = discover_runtime_jobs(session_path=_session_path(training_session))

    assert possible == universe


def test_runtime_job_prerequisites_are_empty(training_session: SessionData) -> None:
    """The single-job pipeline orders nothing, so every job maps to an empty prerequisite tuple."""
    session, universe, _possible = discover_runtime_jobs(session_path=_session_path(training_session))

    assert runtime_job_prerequisites(session=session, universe=universe) == {(RUNTIME_JOB_NAME, RUNTIME_SOURCE_ID): ()}


def test_running_the_pipeline_writes_the_state_feathers(training_session: SessionData, runtime_archive: Path) -> None:
    """The pipeline decodes the archive and hands it to the registered parser, which writes the state feathers."""
    onset_us = int(LogArchiveReader(archive_path=runtime_archive).onset_timestamp_us)

    run_runtime_processing_pipeline(session_path=_session_path(training_session))

    output_directory = training_session.processed_data.runtime_data_path
    system_states = pl.read_ipc(output_directory.joinpath(BehaviorDataFiles.SYSTEM_STATE))
    runtime_states = pl.read_ipc(output_directory.joinpath(BehaviorDataFiles.RUNTIME_STATE))

    assert system_states["time_us"].to_list() == [onset_us + 100, onset_us + 300]
    assert system_states["system_state"].to_list() == [1, 2]
    assert runtime_states["time_us"].to_list() == [onset_us + 200]
    assert runtime_states["runtime_state"].to_list() == [7]


def test_running_the_pipeline_records_the_job_as_succeeded(
    training_session: SessionData, runtime_archive: Path
) -> None:
    """The runtime tracker carries exactly the one job the pipeline runs, marked complete."""
    assert runtime_archive.is_file()

    run_runtime_processing_pipeline(session_path=_session_path(training_session), workers=1)

    tracker = ProcessingTracker(file_path=training_session.processed_data.runtime_tracker_path)
    snapshot = tracker.snapshot()
    job_identifier = ProcessingTracker.generate_job_id(job_name=RUNTIME_JOB_NAME, specifier=RUNTIME_SOURCE_ID)

    assert list(snapshot) == [job_identifier]
    assert snapshot[job_identifier].status == ProcessingStatus.SUCCEEDED
    assert snapshot[job_identifier].job_name == RUNTIME_JOB_NAME
    assert snapshot[job_identifier].specifier == RUNTIME_SOURCE_ID


def test_running_the_pipeline_without_an_archive_names_the_expected_file(training_session: SessionData) -> None:
    """An absent archive leaves the single job impossible, which this pipeline escalates to a failure."""
    with pytest.raises(FileNotFoundError, match=r"No runtime log archive '1_log\.npz' was found"):
        run_runtime_processing_pipeline(session_path=_session_path(training_session))


def test_a_failing_parser_marks_the_job_failed(
    session_factory: Callable[..., SessionData], write_log_archive: Callable[..., Path]
) -> None:
    """A parser error is recorded on the tracked job and re-raised unchanged."""
    # An experiment session reads its experiment configuration snapshot, and this one was acquired without it, so the
    # registered parser fails for a real reason once the decode hands it the messages.
    session = session_factory(session_type=SessionTypes.MESOSCOPE_EXPERIMENT)
    write_log_archive(_archive_path(session), int(RUNTIME_SOURCE_ID), [(100, bytes([_SYSTEM_STATE_CODE, 1]))])

    with pytest.raises(FileNotFoundError, match="Unable to load experiment configuration for session"):
        run_runtime_processing_pipeline(session_path=_session_path(session))

    tracker = ProcessingTracker(file_path=session.processed_data.runtime_tracker_path)
    job_identifier = ProcessingTracker.generate_job_id(job_name=RUNTIME_JOB_NAME, specifier=RUNTIME_SOURCE_ID)
    state = tracker.snapshot()[job_identifier]

    assert state.status == ProcessingStatus.FAILED
    assert str(state.error_message).startswith("Unable to load experiment configuration for session")


def test_single_batch_decode_returns_every_message_in_archive_order(runtime_archive: Path) -> None:
    """A short archive fits one batch, so the decode reads it in one in-process pass."""
    onset_us = int(LogArchiveReader(archive_path=runtime_archive).onset_timestamp_us)

    decoded = _decode_archive(archive_path=runtime_archive, workers=-1, display_progress=False)

    assert decoded.schema == pl.Schema({"time_us": pl.UInt64, "payload": pl.Binary})
    assert decoded["time_us"].to_list() == [onset_us + 100, onset_us + 200, onset_us + 300]
    assert decoded["payload"].to_list() == [
        bytes([_SYSTEM_STATE_CODE, 1]),
        bytes([_RUNTIME_STATE_CODE, 7]),
        bytes([_SYSTEM_STATE_CODE, 2]),
    ]


def test_a_single_worker_decodes_a_multi_batch_archive_in_process(parallel_archive: Path) -> None:
    """A one-worker request keeps the decode in this process even when the reader offers several batches."""
    assert len(LogArchiveReader(archive_path=parallel_archive).get_batches(workers=1)) > 1

    decoded = _decode_archive(archive_path=parallel_archive, workers=1, display_progress=False)

    assert decoded.height == _PARALLEL_MESSAGE_COUNT
    assert decoded["payload"].to_list()[:3] == [
        bytes([_SYSTEM_STATE_CODE, 0]),
        bytes([_SYSTEM_STATE_CODE, 1]),
        bytes([_SYSTEM_STATE_CODE, 2]),
    ]


@pytest.mark.parametrize("display_progress", [False, True])
def test_the_parallel_decode_reassembles_the_archive_in_order(
    parallel_archive: Path, *, display_progress: bool
) -> None:
    """Batches decoded across worker processes are placed back at their batch index, so the order is the archive's."""
    reference = _decode_archive(archive_path=parallel_archive, workers=1, display_progress=False)

    decoded = _decode_archive(archive_path=parallel_archive, workers=2, display_progress=display_progress)

    assert decoded.equals(reference)


def test_decoding_one_batch_returns_its_timestamps_and_payloads(runtime_archive: Path) -> None:
    """The unit of work a decode worker runs reads only the keys it is handed, with the onset supplied to it."""
    reader = LogArchiveReader(archive_path=runtime_archive)
    onset_us = reader.onset_timestamp_us
    keys = reader.get_batches(workers=1)[0]

    timestamps, payloads = _decode_batch(archive_path=runtime_archive, onset_us=onset_us, keys=keys[:2])

    assert timestamps.dtype == np.uint64
    assert timestamps.tolist() == [int(onset_us) + 100, int(onset_us) + 200]
    assert payloads == [bytes([_SYSTEM_STATE_CODE, 1]), bytes([_RUNTIME_STATE_CODE, 7])]
