"""Provides the processing-tracker job-registry alignment helper and the tracked-job execution envelope shared
across library pipelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from contextlib import contextmanager

from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import ProcessingTracker

if TYPE_CHECKING:
    from collections.abc import Iterator


def prepare_tracker(tracker: ProcessingTracker, jobs: list[tuple[str, str]], universe: list[tuple[str, str]]) -> None:
    """Aligns a processing tracker's job registry with the jobs requested for the current pipeline invocation.

    Notes:
        Foreign entries are detected against the full ``universe`` of producible jobs, not the requested ``jobs``
        subset, so a subset invocation aligns the tracker without wiping its siblings' completed state. The helper
        initializes a missing tracker, additively registers requested jobs absent from a valid tracker, and is a
        no-op when all requested jobs are present. It warns and rebuilds only when the tracker holds entries outside
        the universe.

    Args:
        tracker: The processing tracker bound to the target directory.
        jobs: The ``(job_name, specifier)`` tuples the current invocation intends to execute.
        universe: The ``(job_name, specifier)`` tuples enumerating every job the current input set could produce,
            used only for foreign-entry detection. Callers whose requested set is always the full universe pass the
            same list as ``jobs``.
    """
    universe_ids = {
        ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier) for job_name, specifier in universe
    }
    requested_ids = {
        ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier) for job_name, specifier in jobs
    }

    if not tracker.file_path.exists():
        tracker.initialize_jobs(jobs=jobs)
        return

    existing_ids = set(tracker.find_jobs(job_name="").keys())
    foreign_ids = existing_ids - universe_ids

    if foreign_ids:
        console.echo(
            message=(
                f"The processing tracker at '{tracker.file_path}' contains {len(foreign_ids)} job entries "
                f"that are not part of the current job universe. Resetting and reinitializing the tracker to "
                f"match the requested jobs. Foreign job IDs: {sorted(foreign_ids)}."
            ),
            level=LogLevel.WARNING,
        )
        tracker.reset()
        tracker.initialize_jobs(jobs=jobs)
        return

    if not requested_ids.issubset(existing_ids):
        tracker.initialize_jobs(jobs=jobs)


@contextmanager
def tracked_job(tracker: ProcessingTracker, job_id: str) -> Iterator[None]:
    """Runs a single tracked processing job, recording its start, completion, or failure on the processing tracker.

    Notes:
        The system-agnostic counterpart to the acquisition libraries' ``execute_job`` bindings (such as
        ataraxis-video-system's and ataraxis-communication-interface's): it owns the tracker state machine (start,
        then complete on normal return or fail on exception) and leaves the job body to the caller. The job is
        completed only when the wrapped block returns normally. Any exception marks the job failed (recording its
        message) and is re-raised unchanged.

    Args:
        tracker: The processing tracker that records this job's state transitions.
        job_id: The unique hexadecimal identifier of the job to run, as produced by
            ``ProcessingTracker.generate_job_id``.

    Raises:
        Exception: Re-raises any exception raised inside the wrapped block, after marking the job failed.
    """
    tracker.start_job(job_id=job_id)
    try:
        yield
    except Exception as exception:
        tracker.fail_job(job_id=job_id, error_message=str(exception))
        raise
    else:
        tracker.complete_job(job_id=job_id)
