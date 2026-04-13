"""Provides assets for calculating and verifying session data integrity checksums."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_base_utilities import LogLevel, console, resolve_worker_count
from sollertia_shared_assets import SessionData
from ataraxis_data_structures import ProcessingTracker, calculate_directory_checksum

if TYPE_CHECKING:
    from pathlib import Path

CHECKSUM_TRACKER_FILENAME: str = "checksum_processing_tracker.yaml"
"""The filename for the processing tracker placed in the session's raw_data directory alongside the ax_checksum.txt
file."""

_CHECKSUM_TRACKER_LOCK_FILENAME: str = CHECKSUM_TRACKER_FILENAME + ".lock"
"""The lock filename associated with the checksum processing tracker."""

CHECKSUM_JOB_NAME: str = "checksum_resolution"
"""The job name used to identify checksum resolution jobs in processing trackers."""

_CHECKSUM_EXCLUDED_FILES: set[str] = {"ax_checksum.txt", CHECKSUM_TRACKER_FILENAME, _CHECKSUM_TRACKER_LOCK_FILENAME}
"""The set of filenames excluded from checksum calculation. Includes the checksum file itself, the processing
tracker, and its lock file to prevent the tracker presence from altering the checksum value."""


def resolve_checksum(
    session_path: Path,
    *,
    regenerate_checksum: bool = False,
    workers: int = -1,
    display_progress: bool = False,
) -> None:
    """Generates the checksum of the session's raw_data directory and either compares it against the checksum stored
    in the ax_checksum.txt file or overwrites the stored checksum.

    Notes:
        Initializes a processing tracker in the session's raw_data directory alongside the ax_checksum.txt file,
        runs the checksum computation, and records the outcome. If the checksums match, the job is marked as
        completed. If the checksums do not match (data corruption), the job is marked as failed. The tracker file
        and its lock file are excluded from the checksum calculation to prevent the tracker presence from altering
        the checksum value.

    Args:
        session_path: The path to the root data directory of the session to be processed.
        regenerate_checksum: Determines whether to update the checksum stored in the ax_checksum.txt file instead of
            verifying its integrity.
        workers: The number of processes to use for parallel checksum calculation. Setting this to a value less than
            1 uses all available CPU cores (minus reserved cores). Setting this to 1 conducts the calculation
            sequentially without spawning additional processes.
        display_progress: Determines whether to display console messages and a progress bar during checksum
            calculation.

    Raises:
        FileNotFoundError: If the source path does not contain a valid session data hierarchy.
    """
    # Loads session data to resolve the raw_data path where the tracker and checksum file live.
    session_data = SessionData.load(session_path=session_path)

    # Initializes the processing tracker in the raw_data directory alongside the checksum file.
    tracker = ProcessingTracker(file_path=session_data.raw_data_path.joinpath(CHECKSUM_TRACKER_FILENAME))
    job_ids = tracker.initialize_jobs(jobs=[(CHECKSUM_JOB_NAME, session_data.session_name)])
    job_id = job_ids[0]

    # Marks the job as running.
    tracker.start_job(job_id=job_id)
    try:
        if display_progress:
            console.echo(
                message=f"Resolving the data integrity checksum for the session '{session_data.session_name}'...",
                level=LogLevel.INFO,
            )

        # Resolves the process count and calculates the checksum for the raw_data directory. If the
        # 'save_checksum' flag is True, this guarantees that the check below succeeds as the function replaces
        # the checksum in the ax_checksum.txt file with the newly calculated value.
        resolved_workers = resolve_worker_count(requested_workers=workers)
        calculated_checksum = calculate_directory_checksum(
            directory=session_data.raw_data_path,
            num_processes=resolved_workers,
            progress=display_progress,
            save_checksum=regenerate_checksum,
            excluded_files=_CHECKSUM_EXCLUDED_FILES,
        )

        # Loads the checksum stored inside the ax_checksum.txt file.
        checksum_path = session_data.raw_data_path.joinpath("ax_checksum.txt")
        with checksum_path.open() as f:
            stored_checksum = f.read().strip()

        # If the two checksums do not match, this indicates data corruption.
        if stored_checksum != calculated_checksum:
            tracker.fail_job(job_id=job_id)
            if display_progress:
                console.echo(
                    message=f"Session '{session_data.session_name}' raw data integrity: Compromised.",
                    level=LogLevel.ERROR,
                )
        else:
            tracker.complete_job(job_id=job_id)
            if display_progress:
                console.echo(
                    message=f"Session '{session_data.session_name}' raw data integrity: Verified.",
                    level=LogLevel.SUCCESS,
                )

    except Exception:
        # Marks the job as failed and re-raises any unexpected errors.
        tracker.fail_job(job_id=job_id)
        raise
