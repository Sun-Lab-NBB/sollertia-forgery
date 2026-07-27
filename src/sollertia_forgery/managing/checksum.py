"""Provides the session raw-data integrity pipeline, which verifies or regenerates the stored data checksum."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_base_utilities import LogLevel, console, resolve_worker_count
from sollertia_shared_assets import SessionData, RawDataFiles, ProcessingTrackers
from ataraxis_data_structures import ProcessingTracker, calculate_directory_checksum

from ..shared_assets import pinned_worker_threads

if TYPE_CHECKING:
    from pathlib import Path

CHECKSUM_JOB_NAME: str = "checksum_resolution"
"""The job name identifying the checksum resolution job in the checksum processing tracker
(``ProcessingTrackers.CHECKSUM``), where this pipeline records the job's state."""

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


def run_checksum_processing_pipeline(
    session_path: Path,
    *,
    regenerate_checksum: bool = False,
    workers: int = -1,
    display_progress: bool = False,
) -> None:
    """Verifies the integrity of the session's raw_data directory by comparing its computed checksum against the
    value stored in ax_checksum.txt, or regenerates that stored value.

    Notes:
        This is a single-stage pipeline that produces exactly one job, which resolves the whole raw_data directory.
        The job runs in one of two modes. Verification compares the recomputed checksum against the stored value and
        records a failure on any mismatch, which is what marks the data as corrupted. Regeneration overwrites the
        stored value with the freshly computed one, re-baselining a session whose raw data changed by intent.

        Every path this pipeline reads and writes lives under raw_data, while every other session pipeline writes
        only under processed_data. A checksum job therefore runs safely alongside any other pipeline's job for the
        same session.

    Args:
        session_path: The path to the root data directory of the session to be processed.
        regenerate_checksum: Determines whether to overwrite the stored ax_checksum.txt value with the freshly
            computed checksum instead of verifying against it.
        workers: The number of parallel worker processes. Values below 1 request all available cores minus reserved
            cores. A value of 1 disables parallelism.
        display_progress: Determines whether to emit console messages and a progress bar during calculation.

    Raises:
        FileNotFoundError: If the source path does not contain a valid session data hierarchy, or if verification is
            requested for a session that stores no checksum value.
    """
    session, universe, runnable = discover_checksum_jobs(session_path=session_path)
    job_id = ProcessingTracker.generate_job_id(job_name=CHECKSUM_JOB_NAME, specifier=session.session_name)

    # Initializes the processing tracker in the raw_data directory alongside the checksum file. Aligning against the
    # universe resets foreign or outdated job entries while preserving the state of the jobs this pipeline produces.
    tracker = ProcessingTracker(file_path=session.raw_data.checksum_tracker_path)
    tracker.align_jobs(jobs=runnable, universe=universe)

    checksum_path = session.raw_data.checksum_path

    # Verification needs a stored value to compare against, so its absence is a hard error rather than a mismatch.
    # Regeneration writes that value, so it runs on a session that has never been checksummed.
    if not regenerate_checksum and not checksum_path.is_file():
        message = (
            f"Unable to verify the data integrity checksum for the session '{session.session_name}'. No checksum "
            f"file exists at '{checksum_path}'. Regenerate the session's checksum to establish the stored value "
            f"this verification compares against."
        )
        console.error(message=message, error=FileNotFoundError)

    tracker.start_job(job_id=job_id)
    try:
        if display_progress:
            console.echo(
                message=f"Resolving the data integrity checksum for the session '{session.session_name}'...",
                level=LogLevel.INFO,
            )

        # Resolves the process count and calculates the checksum for the raw_data directory. If the
        # 'regenerate_checksum' flag is True (forwarded as save_checksum), this guarantees that the check below
        # succeeds as the function replaces the checksum in the ax_checksum.txt file with the newly calculated value.
        # Hashing fans one file per worker across a pool of its own, and each of those workers sizes its library
        # thread pools while importing, so the caps are placed here rather than inside them.
        resolved_workers = resolve_worker_count(requested_workers=workers)
        with pinned_worker_threads():
            calculated_checksum = calculate_directory_checksum(
                directory=session.raw_data_path,
                num_processes=resolved_workers,
                progress=display_progress,
                save_checksum=regenerate_checksum,
                excluded_files=_CHECKSUM_EXCLUDED_FILES,
            )

        # Loads the checksum stored inside the ax_checksum.txt file.
        with checksum_path.open() as file:
            stored_checksum = file.read().strip()

        # If the two checksums do not match, this indicates data corruption.
        if stored_checksum != calculated_checksum:
            tracker.fail_job(
                job_id=job_id,
                error_message=(
                    f"Raw data integrity compromised: recomputed checksum '{calculated_checksum}' does not match "
                    f"stored checksum '{stored_checksum}'."
                ),
            )
            if display_progress:
                console.echo(
                    message=f"Session '{session.session_name}' raw data integrity: Compromised.",
                    level=LogLevel.ERROR,
                )
        else:
            tracker.complete_job(job_id=job_id)
            if display_progress:
                console.echo(
                    message=f"Session '{session.session_name}' raw data integrity: Verified.",
                    level=LogLevel.SUCCESS,
                )

    except Exception as error:
        # Marks the job as failed and re-raises any unexpected errors.
        tracker.fail_job(job_id=job_id, error_message=f"{type(error).__name__}: {error}")
        raise


def discover_checksum_jobs(session_path: Path) -> tuple[SessionData, list[tuple[str, str]], list[tuple[str, str]]]:
    """Resolves the checksum pipeline's job universe and runnable subset for the target session.

    Notes:
        The checksum pipeline produces exactly one job, so the universe is always the single
        ``(CHECKSUM_JOB_NAME, session_name)`` pair. Both pipeline modes share that job, because a session carries one
        integrity state whether the run establishes it or confirms it. That job is runnable once the session holds
        raw data the checksum covers, which excludes the checksum file, the tracker, and the tracker lock. This is
        pure discovery that reads no file contents and mutates nothing.

    Args:
        session_path: The path to the root session directory containing the session data hierarchy.

    Returns:
        A tuple of the loaded session, the job universe as a list of ``(job_name, specifier)`` pairs, and the runnable
        subset of that universe.

    Raises:
        FileNotFoundError: If the source path does not contain a valid session data hierarchy.
    """
    session = SessionData.load(session_path=session_path)
    universe = [(CHECKSUM_JOB_NAME, session.session_name)]
    runnable = list(universe) if _has_checksummable_data(raw_data_path=session.raw_data_path) else []
    return session, universe, runnable


def checksum_job_prerequisites(
    universe: list[tuple[str, str]],
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]:
    """Returns the intra-pipeline job ordering for the checksum pipeline.

    Notes:
        The checksum pipeline produces a single job with no upstream dependency, so every job maps to an empty
        prerequisite tuple. This mirrors the prerequisite contract the other worker packages publish, so a batch
        layer can validate ordering uniformly across pipelines.

    Args:
        universe: The job universe as returned by ``discover_checksum_jobs``.

    Returns:
        A mapping of each job to its tuple of prerequisite jobs, which is always empty for the checksum pipeline.
    """
    return dict.fromkeys(universe, ())


def _has_checksummable_data(raw_data_path: Path) -> bool:
    """Determines whether a session's raw data directory holds at least one file the checksum covers.

    Notes:
        Stops at the first qualifying file, so the cost is a partial directory walk rather than a full census. A
        session holding only the excluded bookkeeping files has nothing to checksum, which is what distinguishes an
        acquired session from one whose raw data never arrived.

    Args:
        raw_data_path: The path to the session's raw data directory.

    Returns:
        True when the directory holds a file outside the excluded set, and False otherwise.
    """
    if not raw_data_path.is_dir():
        return False
    return any(path.is_file() and path.name not in _CHECKSUM_EXCLUDED_FILES for path in raw_data_path.rglob("*"))
