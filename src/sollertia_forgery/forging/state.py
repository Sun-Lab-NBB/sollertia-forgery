"""Provides the dataset state artifact that serializes every forging job the dataset's tracker records into one
shippable table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from natsort import natsorted
from filelock import FileLock
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import ProcessingTracker, atomic_write

from .pipeline import (
    FORGING_JOB_NAME,
    MULTIDAY_DISCOVERY_JOB_NAME,
    MULTIDAY_EXTRACTION_JOB_NAME,
    forging_tracker_path,
)
from ..shared_assets import natural_sort

if TYPE_CHECKING:
    from pathlib import Path

    from sollertia_shared_assets import DatasetData

DATASET_STATE_FILENAME: str = "dataset_state.feather"
"""The filename of the dataset state artifact, written at the dataset's root beside its forging tracker. The remote
backend resolves the same artifact from a server path, without loading the dataset."""

_LOCK_TIMEOUT_SECONDS: float = 20.0
"""The period a writer waits for the state file's lock before giving up, matching the project manifest's writer."""

ANIMAL_SCOPE: str = "animal"
"""The scope label of a forging job specified by the animal it processes."""

SESSION_SCOPE: str = "session"
"""The scope label of a forging job specified by the session it processes."""

DATASET_JOB_SCOPES: dict[str, str] = {
    MULTIDAY_DISCOVERY_JOB_NAME: ANIMAL_SCOPE,
    MULTIDAY_EXTRACTION_JOB_NAME: SESSION_SCOPE,
    FORGING_JOB_NAME: SESSION_SCOPE,
}
"""Maps each forging job name to the unit its specifier names.

Notes:
    Cross-recording discovery runs once per animal and reads that animal's whole session set, while extraction and
    assembly each run once per session. A reader therefore resolves a row's subject from this scope rather than
    assuming the specifier names a session.
"""

DATASET_STATE_SCHEMA: dict[str, pl.datatypes.classes.DataTypeClass | pl.DataType] = {
    "dataset": pl.String,
    "animal": pl.String,
    "session": pl.String,
    "scope": pl.String,
    "job_id": pl.String,
    "job_name": pl.String,
    "specifier": pl.String,
    "status": pl.String,
    "executor_id": pl.String,
    "error_message": pl.String,
    "started_at": pl.UInt64,
    "completed_at": pl.UInt64,
}
"""The column layout of the dataset state artifact, one row per forging job.

Notes:
    ``animal`` and ``session`` are resolved alongside the raw ``specifier`` so the table joins against the project job
    artifact on the pair that identifies a session across animals. An animal-scoped row carries no session, and the
    timestamps follow the stack-wide microsecond-epoch convention that artifact already stores.
"""


def dataset_state_path(dataset: DatasetData) -> Path:
    """Resolves the path to a dataset's state artifact.

    Args:
        dataset: The resolved dataset whose state artifact to locate.

    Returns:
        The path to the dataset's state .feather file.
    """
    return dataset.dataset_data_path.parent.joinpath(DATASET_STATE_FILENAME)


def generate_dataset_state(dataset: DatasetData, *, display_progress: bool = False) -> Path:
    """Builds and saves the state artifact for one forged dataset.

    Reads the dataset's forging tracker and writes one row per tracked job, carrying that job's scope, its resolved
    subject, and everything the tracker records about it. A file lock serializes concurrent writers.

    Args:
        dataset: The resolved dataset whose forging state to serialize.
        display_progress: Determines whether to emit a completion message once the artifact is written. Generation
            reads a single tracker, so no progress bar is displayed.

    Returns:
        The path the state artifact was written to.

    Raises:
        Timeout: If the state file's lock cannot be acquired within the timeout period.
        ValueError: If the dataset's forging tracker records a job name that declares no scope in
            ``DATASET_JOB_SCOPES``.
    """
    state_path = dataset_state_path(dataset=dataset)
    lock = FileLock(str(state_path.with_suffix(state_path.suffix + ".lock")))

    with lock.acquire(timeout=_LOCK_TIMEOUT_SECONDS):
        rows = _build_job_rows(dataset=dataset)
        frame = natural_sort(
            frame=pl.DataFrame(data=rows, schema=DATASET_STATE_SCHEMA, strict=False),
            by=["animal", "session", "job_name"],
            nulls_last=True,
        )
        # Published through a temporary file renamed over the destination, which also creates the destination's
        # parent. A reader of the artifact takes no lock, so an in-place rewrite would let it read a torn file.
        with atomic_write(file_path=state_path, binary=True) as file:
            frame.write_ipc(file=file, compression="uncompressed")

    if display_progress:
        console.echo(
            message=f"Dataset '{dataset.name}' state: Generated with {len(rows)} job(s).", level=LogLevel.SUCCESS
        )
    return state_path


def _build_job_rows(dataset: DatasetData) -> list[dict[str, str | int | None]]:
    """Reads a dataset's forging tracker into one row per tracked job.

    Notes:
        A session-scoped job resolves its animal through the dataset's own session list. A dataset that dropped a
        session during a rebuild reports that session's outstanding rows with no animal until the next pipeline run
        realigns the tracker, so a state read stays available across that window.

    Args:
        dataset: The resolved dataset whose tracker to read.

    Returns:
        The list of row mappings, one per tracked job, empty when the dataset has no tracker or no tracked jobs.

    Raises:
        ValueError: If the tracker records a job name that declares no scope.
    """
    tracker_path = forging_tracker_path(dataset=dataset)
    if not tracker_path.is_file():
        return []

    # The summary carries every field the tracker records for a job, so one read serves both the scope check and the
    # rows built from it.
    jobs = ProcessingTracker(file_path=tracker_path).summarize()["jobs"]
    if not jobs:
        return []

    animal_of_session = {entry.session: entry.animal for entry in dataset.sessions}

    unscoped = natsorted({entry["job_name"] for entry in jobs if entry["job_name"] not in DATASET_JOB_SCOPES})
    if unscoped:
        message = (
            f"Unable to serialize the state of dataset '{dataset.name}'. Its forging tracker records job name(s) "
            f"{unscoped}, which declare no scope. Every forging job name must declare the unit its specifier names "
            f"in DATASET_JOB_SCOPES."
        )
        console.error(message=message, error=ValueError)

    rows: list[dict[str, str | int | None]] = []
    for entry in jobs:
        scope = DATASET_JOB_SCOPES[entry["job_name"]]
        specifier = entry["specifier"]
        session = specifier if scope == SESSION_SCOPE else None
        rows.append(
            {
                "dataset": dataset.name,
                "animal": specifier if scope == ANIMAL_SCOPE else animal_of_session.get(specifier),
                "session": session,
                "scope": scope,
                "job_id": entry["job_id"],
                "job_name": entry["job_name"],
                "specifier": specifier,
                "status": entry["status"],
                "executor_id": entry["executor_id"],
                "error_message": entry.get("error_message"),
                "started_at": entry["started_at"],
                "completed_at": entry["completed_at"],
            }
        )
    return rows
