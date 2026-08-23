"""Tests the session raw-data integrity pipeline that establishes, confirms, or condemns the stored data checksum."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from ataraxis_base_utilities import console
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from sollertia_forgery.managing import (
    CHECKSUM_JOB_NAME,
    discover_checksum_jobs,
    checksum_job_prerequisites,
    run_checksum_processing_pipeline,
)
import sollertia_forgery.managing.checksum as checksum_module
from sollertia_forgery.managing.checksum import _CHECKSUM_EXCLUDED_FILES, _has_checksummable_data

if TYPE_CHECKING:
    from pathlib import Path

    from sollertia_shared_assets import SessionData


@pytest.fixture
def reported_messages(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Collects every message the shared console is asked to echo, so a reported outcome is assertable.

    Args:
        monkeypatch: The fixture used to replace the console's echo method for the duration of one test.

    Returns:
        The list the recorder appends each echoed message to, in the order they were emitted.
    """
    messages: list[str] = []
    monkeypatch.setattr(console, "echo", lambda message, **_keywords: messages.append(message))
    return messages


def read_job_state(session: SessionData) -> tuple[str, str | None]:
    """Reads the status name and error message of the session's single checksum job.

    Args:
        session: The session whose checksum tracker to read.

    Returns:
        A tuple of the job's status name and the error message it recorded.
    """
    jobs = ProcessingTracker(file_path=session.raw_data.checksum_tracker_path).snapshot()
    state = next(iter(jobs.values()))
    return state.status.name, state.error_message


def test_regeneration_establishes_the_stored_checksum(training_session: SessionData) -> None:
    """A session that was never checksummed gets its baseline written and its job recorded as succeeded."""
    assert not training_session.raw_data.checksum_path.is_file()

    run_checksum_processing_pipeline(session_path=training_session.raw_data_path.parent, regenerate_checksum=True)

    stored = training_session.raw_data.checksum_path.read_text().strip()
    assert len(stored) == 32
    assert read_job_state(session=training_session) == ("SUCCEEDED", None)


def test_verification_confirms_untouched_raw_data(training_session: SessionData) -> None:
    """Recomputing an unchanged directory reproduces the stored value, which the pipeline records as a success."""
    session_path = training_session.raw_data_path.parent
    run_checksum_processing_pipeline(session_path=session_path, regenerate_checksum=True, workers=1)
    baseline = training_session.raw_data.checksum_path.read_text().strip()

    run_checksum_processing_pipeline(session_path=session_path, workers=1)

    assert training_session.raw_data.checksum_path.read_text().strip() == baseline
    assert read_job_state(session=training_session) == ("SUCCEEDED", None)


def test_verification_condemns_raw_data_that_changed(training_session: SessionData) -> None:
    """A mismatch is recorded as a job failure naming both checksums, which is how corruption is surfaced."""
    session_path = training_session.raw_data_path.parent
    run_checksum_processing_pipeline(session_path=session_path, regenerate_checksum=True, workers=1)
    stored = training_session.raw_data.checksum_path.read_text().strip()
    training_session.raw_data_path.joinpath("intruder.bin").write_bytes(b"unexpected")

    run_checksum_processing_pipeline(session_path=session_path, workers=1)

    status, error_message = read_job_state(session=training_session)
    assert status == "FAILED"
    assert error_message is not None
    assert "Raw data integrity compromised" in error_message
    assert f"stored checksum '{stored}'" in error_message


def test_a_compromised_session_announces_its_outcome(
    training_session: SessionData, reported_messages: list[str]
) -> None:
    """The mismatch verdict is reported alongside the preamble, so an operator sees corruption as it is found."""
    session_path = training_session.raw_data_path.parent
    run_checksum_processing_pipeline(session_path=session_path, regenerate_checksum=True, workers=1)
    training_session.raw_data_path.joinpath("intruder.bin").write_bytes(b"unexpected")

    run_checksum_processing_pipeline(session_path=session_path, workers=1, display_progress=True)

    assert reported_messages[0].startswith("Resolving the data integrity checksum")
    assert reported_messages[1].endswith("raw data integrity: Compromised.")


def test_a_verified_session_announces_its_outcome(training_session: SessionData, reported_messages: list[str]) -> None:
    """The success message reports the verdict, so an operator reads the outcome without opening the tracker."""
    session_path = training_session.raw_data_path.parent
    run_checksum_processing_pipeline(session_path=session_path, regenerate_checksum=True, workers=1)

    run_checksum_processing_pipeline(session_path=session_path, workers=1, display_progress=True)

    assert reported_messages[-1].endswith("raw data integrity: Verified.")


def test_verification_without_a_stored_value_reports_how_to_establish_one(training_session: SessionData) -> None:
    """Verification needs a baseline, so its absence names regeneration rather than reading as a mismatch."""
    with pytest.raises(FileNotFoundError, match="No checksum file exists at"):
        run_checksum_processing_pipeline(session_path=training_session.raw_data_path.parent, workers=1)


def test_a_write_fault_marks_the_job_failed_before_it_propagates(training_session: SessionData) -> None:
    """A regeneration that cannot store its result records the fault on the tracker and re-raises it unchanged.

    The stored value's location is occupied by a directory, so writing the freshly computed checksum faults inside
    the calculation itself, which is the one place this pipeline's failure envelope has to cover.
    """
    training_session.raw_data.checksum_path.mkdir()

    with pytest.raises(IsADirectoryError):
        run_checksum_processing_pipeline(
            session_path=training_session.raw_data_path.parent, regenerate_checksum=True, workers=1
        )

    status, error_message = read_job_state(session=training_session)
    assert status == "FAILED"
    assert error_message is not None
    assert error_message.startswith("IsADirectoryError: ")


def test_discovery_reports_the_single_job_the_pipeline_owns(training_session: SessionData) -> None:
    """The universe is one job specified by the session, and an acquired session makes that job possible."""
    session, universe, possible = discover_checksum_jobs(session_path=training_session.raw_data_path.parent)

    assert session.session_name == training_session.session_name
    assert universe == [(CHECKSUM_JOB_NAME, training_session.session_name)]
    assert possible == universe


def test_every_checksum_job_declares_no_prerequisite(training_session: SessionData) -> None:
    """The single job depends on nothing, so the ordering contract maps it to an empty tuple."""
    session, universe, _possible = discover_checksum_jobs(session_path=training_session.raw_data_path.parent)

    assert checksum_job_prerequisites(session=session, universe=universe) == {
        (CHECKSUM_JOB_NAME, training_session.session_name): ()
    }


def test_a_session_with_nothing_to_checksum_is_refused_before_the_tracker_is_touched(
    training_session: SessionData, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session whose raw data never arrived makes no job possible, so the run is refused by name and the tracker is
    left exactly as it was found.
    """
    monkeypatch.setattr(checksum_module, "_has_checksummable_data", lambda raw_data_path: False)  # noqa: ARG005

    with pytest.raises(ValueError, match="holds no file the checksum covers"):
        run_checksum_processing_pipeline(
            session_path=training_session.raw_data_path.parent, regenerate_checksum=True, workers=1
        )

    assert ProcessingTracker(file_path=training_session.raw_data.checksum_tracker_path).snapshot() == {}


def test_a_session_that_lost_its_raw_data_keeps_its_recorded_verdict(
    training_session: SessionData, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session whose tracker already records a verdict keeps it when its raw data is later archived off, since the
    refusal is decided from the data rather than from the tracker's ignorance of the job.
    """
    run_checksum_processing_pipeline(
        session_path=training_session.raw_data_path.parent, regenerate_checksum=True, workers=1
    )
    tracker_path = training_session.raw_data.checksum_tracker_path
    recorded = ProcessingTracker(file_path=tracker_path).snapshot()
    assert all(state.status is ProcessingStatus.SUCCEEDED for state in recorded.values())

    monkeypatch.setattr(checksum_module, "_has_checksummable_data", lambda raw_data_path: False)  # noqa: ARG005
    with pytest.raises(ValueError, match="holds no file the checksum covers"):
        run_checksum_processing_pipeline(
            session_path=training_session.raw_data_path.parent, regenerate_checksum=True, workers=1
        )

    assert ProcessingTracker(file_path=tracker_path).snapshot() == recorded


def test_a_directory_holding_only_bookkeeping_files_has_nothing_to_checksum(tmp_path: Path) -> None:
    """The checksum file, its tracker, and the tracker lock are excluded, so a directory of only those is empty."""
    raw_data = tmp_path.joinpath("raw_data")
    raw_data.mkdir()
    for name in _CHECKSUM_EXCLUDED_FILES:
        raw_data.joinpath(name).write_text("bookkeeping")

    assert not _has_checksummable_data(raw_data_path=raw_data)

    raw_data.joinpath("acquired.bin").write_bytes(b"data")
    assert _has_checksummable_data(raw_data_path=raw_data)


def test_an_absent_raw_data_directory_has_nothing_to_checksum(tmp_path: Path) -> None:
    """A session whose acquired data never arrived carries no coverable file at all."""
    assert not _has_checksummable_data(raw_data_path=tmp_path.joinpath("never_acquired"))
