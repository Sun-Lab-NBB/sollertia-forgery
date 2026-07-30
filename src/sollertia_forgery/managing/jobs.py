"""Provides the project job artifact that records every tracked job of every per-session pipeline as its own row."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

PROJECT_JOBS_SCHEMA: dict[str, pl.datatypes.classes.DataTypeClass | pl.DataType] = {
    "animal": pl.String,
    "session": pl.String,
    "pipeline": pl.String,
    "job_id": pl.String,
    "job_name": pl.String,
    "specifier": pl.String,
    "status": pl.String,
    "executor_id": pl.String,
    "error_message": pl.String,
    "started_at": pl.UInt64,
    "completed_at": pl.UInt64,
}
"""The column layout of the project job artifact, one row per tracked job.

Notes:
    Carries every ``ataraxis_data_structures.JobState`` field alongside its registry ``job_id`` and the ``pipeline``
    that produced it. The subject columns name the session the job belongs to, so a reader joins the artifact against
    the manifest on the animal and session pair.
"""


def project_jobs_path(project_directory: Path) -> Path:
    """Resolves the path to the project job artifact under the target project's root directory.

    This is the single source of the artifact's filename, so the writer and every consumer that locates it derive the
    same path.

    Args:
        project_directory: The path to the project's root directory.

    Returns:
        The path to the project's job .feather file.
    """
    return project_directory.joinpath(f"{project_directory.stem}_jobs.feather")


def write_project_jobs(project_directory: Path, job_rows: list[dict[str, Any]]) -> Path:
    """Writes the project job artifact from the rows the manifest's walk collected.

    Notes:
        Takes no lock of its own, because the manifest's writer calls this while holding the lock that serializes the
        whole generation. Writing both artifacts under one lock is what keeps them from disagreeing about a session.

        Stored uncompressed so a reader memory-maps it rather than decoding it, which is what makes a filtered read
        cost the same whether a project holds fifty sessions or eight hundred.

    Args:
        project_directory: The path to the project's root directory.
        job_rows: The job rows to record, each carrying the animal and session that recorded it.

    Returns:
        The path the artifact was written to.
    """
    jobs_path = project_jobs_path(project_directory=project_directory)
    pl.DataFrame(data=job_rows, schema=PROJECT_JOBS_SCHEMA, strict=False).sort(
        by=["animal", "session", "pipeline", "job_name", "specifier"], nulls_last=True
    ).write_ipc(file=jobs_path, compression="uncompressed")
    return jobs_path
