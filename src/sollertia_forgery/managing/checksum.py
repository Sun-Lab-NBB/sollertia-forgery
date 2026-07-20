"""Provides assets for calculating and verifying session data integrity checksums."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_base_utilities import LogLevel, console, resolve_worker_count
from sollertia_shared_assets import SessionData, RawDataFiles, ProcessingTrackers
from ataraxis_data_structures import ProcessingTracker, calculate_directory_checksum

from ..shared_assets import prepare_tracker

if TYPE_CHECKING:
    from pathlib import Path

CHECKSUM_JOB_NAME: str = "checksum_resolution"
"""The job name used to identify checksum resolution jobs in processing trackers."""

_CHECKSUM_TRACKER_LOCK_FILENAME: str = ProcessingTrackers.CHECKSUM + ".lock"
"""The lock filename associated with the checksum processing tracker."""

_CHECKSUM_EXCLUDED_FILES: set[str] = {
    str(RawDataFiles.CHECKSUM),
    str(ProcessingTrackers.CHECKSUM),
    _CHECKSUM_TRACKER_LOCK_FILENAME,
}
"""The set of filenames excluded from checksum calculation. Includes the checksum file itself, the processing
tracker, and its lock file to prevent the tracker presence from altering the checksum value. Only includes files
canonically found under the 'raw_data' session data directory."""


def resolve_checksum(
    session_path: Path,
    *,
    regenerate_checksum: bool = False,
    workers: int = -1,
    display_progress: bool = False,
) -> None:
    """Verifies the integrity of the session's raw_data directory by comparing its computed checksum against the
    value stored in ax_checksum.txt, or regenerates that stored value.

    Records the outcome on a checksum processing tracker in the raw_data directory: completed on a match, failed on a
    mismatch (indicating corruption).

    Args:
        session_path: The path to the root data directory of the session to be processed.
        regenerate_checksum: Determines whether to overwrite the stored ax_checksum.txt value with the freshly
            computed checksum instead of verifying against it.
        workers: The number of parallel worker processes. Values below 1 request all available cores minus reserved
            cores. A value of 1 disables parallelism.
        display_progress: Determines whether to emit console messages and a progress bar during calculation.

    Raises:
        FileNotFoundError: If the source path does not contain a valid session data hierarchy.
    """
    # Loads session data to resolve the raw_data path where the tracker and checksum file live.
    session_data = SessionData.load(session_path=session_path)

    # Initializes the processing tracker in the raw_data directory alongside the checksum file. Applies stale
    # entry detection so that foreign or outdated job entries are reset before the new job is registered.
    tracker = ProcessingTracker(file_path=session_data.raw_data.checksum_tracker_path)
    jobs = [(CHECKSUM_JOB_NAME, session_data.session_name)]
    prepare_tracker(tracker=tracker, jobs=jobs, universe=jobs)
    job_id = ProcessingTracker.generate_job_id(job_name=CHECKSUM_JOB_NAME, specifier=session_data.session_name)

    tracker.start_job(job_id=job_id)
    try:
        if display_progress:
            console.echo(
                message=f"Resolving the data integrity checksum for the session '{session_data.session_name}'...",
                level=LogLevel.INFO,
            )

        # Resolves the process count and calculates the checksum for the raw_data directory. If the
        # 'regenerate_checksum' flag is True (forwarded as save_checksum), this guarantees that the check below
        # succeeds as the function replaces the checksum in the ax_checksum.txt file with the newly calculated value.
        resolved_workers = resolve_worker_count(requested_workers=workers)
        calculated_checksum = calculate_directory_checksum(
            directory=session_data.raw_data_path,
            num_processes=resolved_workers,
            progress=display_progress,
            save_checksum=regenerate_checksum,
            excluded_files=_CHECKSUM_EXCLUDED_FILES,
        )

        # Loads the checksum stored inside the ax_checksum.txt file.
        checksum_path = session_data.raw_data.checksum_path
        with checksum_path.open() as file:
            stored_checksum = file.read().strip()

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
