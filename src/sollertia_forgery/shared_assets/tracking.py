"""Provides the processing-tracker status reporters and the tracked-job execution envelope shared across library
pipelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
from contextlib import contextmanager

from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ataraxis_data_structures import JobState


def summarize_tracker(jobs: dict[str, JobState]) -> dict[str, Any]:
    """Converts a processing tracker's job registry into structured per-job details and summary counts.

    Notes:
        Emits every field ``JobState`` carries, so a consumer that snapshots tracker state can serialize a job
        faithfully.

    Args:
        jobs: The tracker's job registry, as returned by ``ProcessingTracker.snapshot``.

    Returns:
        A dictionary containing per-job details in ``jobs`` and aggregate counts in ``summary``. Each job entry
        carries ``job_id``, ``job_name``, ``specifier``, ``status``, ``executor_id``, ``started_at``, and
        ``completed_at``, plus ``error_message`` when the job recorded a failure reason.
    """
    job_details: list[dict[str, Any]] = []
    succeeded_count = 0
    failed_count = 0
    running_count = 0
    scheduled_count = 0

    for job_id, job_state in jobs.items():
        status = job_state.status

        if status == ProcessingStatus.SUCCEEDED:
            succeeded_count += 1
        elif status == ProcessingStatus.FAILED:
            failed_count += 1
        elif status == ProcessingStatus.RUNNING:
            running_count += 1
        else:
            scheduled_count += 1

        entry: dict[str, Any] = {
            "job_id": job_id,
            "job_name": job_state.job_name,
            "specifier": job_state.specifier,
            "status": status.name,
            "executor_id": job_state.executor_id,
            "started_at": job_state.started_at,
            "completed_at": job_state.completed_at,
        }
        if job_state.error_message is not None:
            entry["error_message"] = job_state.error_message
        job_details.append(entry)

    return {
        "jobs": job_details,
        "summary": {
            "total": len(jobs),
            "succeeded": succeeded_count,
            "failed": failed_count,
            "running": running_count,
            "scheduled": scheduled_count,
        },
    }


def derive_tracker_status(
    summary: dict[str, Any],
) -> Literal["failed", "completed", "processing", "not_started", "in_progress"]:
    """Derives a high-level processing status label from a tracker summary's job counts.

    Applies a fixed priority: ``failed`` if any job failed, ``completed`` if all succeeded, ``processing`` if any
    are running, ``not_started`` if all are scheduled, and ``in_progress`` otherwise. A summary with no jobs at all
    resolves to ``in_progress``, since the ``completed`` and ``not_started`` branches each require at least one
    tracked job.

    Args:
        summary: A dictionary containing ``total``, ``succeeded``, ``failed``, ``running``, and ``scheduled``
            counts, as produced by ``summarize_tracker``.

    Returns:
        The label matching the highest-priority condition the counts satisfy.
    """
    total = summary.get("total", 0)
    if summary.get("failed", 0) > 0:
        return "failed"
    if summary.get("succeeded", 0) == total and total > 0:
        return "completed"
    if summary.get("running", 0) > 0:
        return "processing"
    if summary.get("scheduled", 0) == total and total > 0:
        return "not_started"
    return "in_progress"


@contextmanager
def tracked_job(tracker: ProcessingTracker, job_id: str) -> Iterator[None]:
    """Runs a single tracked processing job, recording its start, completion, or failure on the processing tracker.

    Notes:
        Owns the tracker state machine and leaves the job body to the caller. The guard spans the wrapped block and
        the completion call, so an ``Exception`` from either marks the job failed, records its message, and re-raises
        it unchanged. A ``BaseException`` such as ``KeyboardInterrupt`` propagates with the job left running.

    Args:
        tracker: The processing tracker that records this job's state transitions.
        job_id: The unique hexadecimal identifier of the job to run, as produced by
            ``ProcessingTracker.generate_job_id``.

    Yields:
        None. The tracker holds the job in its running state for the duration of the block.

    Raises:
        ValueError: If the tracker does not track a job with the provided identifier.
        TimeoutError: If the tracker's lock file cannot be acquired within its timeout period.
        Exception: Re-raises any exception raised inside the wrapped block, after marking the job failed.
    """
    tracker.start_job(job_id=job_id)
    try:
        yield
        tracker.complete_job(job_id=job_id)
    except Exception as exception:
        tracker.fail_job(job_id=job_id, error_message=str(exception))
        raise
