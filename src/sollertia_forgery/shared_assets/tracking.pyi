from typing import Any, Literal
from contextlib import contextmanager
from collections.abc import Iterator

from ataraxis_data_structures import (
    JobState as JobState,
    ProcessingTracker as ProcessingTracker,
)

def summarize_tracker(jobs: dict[str, JobState]) -> dict[str, Any]: ...
def derive_tracker_status(
    summary: dict[str, Any],
) -> Literal["failed", "completed", "processing", "not_started", "in_progress"]: ...
@contextmanager
def tracked_job(tracker: ProcessingTracker, job_id: str) -> Iterator[None]: ...
