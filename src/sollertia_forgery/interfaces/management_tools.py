"""Provides the Model Context Protocol (MCP) tools for generating, inspecting, and querying a project manifest.

The manifest is deployed per project rather than per session, and it snapshots state the session pipelines have
already recorded. It therefore runs between execution graphs rather than inside one, and these tools run it directly
instead of admitting it to the shared batch pool.
"""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any
from pathlib import Path

from sollertia_shared_assets import ProcessingTrackers
from ataraxis_data_structures import ProcessingTracker

from ..managing import (
    MANIFEST_JOB_NAME,
    ProjectManifest,
    project_manifest_path,
    generate_project_manifest,
)
from .mcp_instance import mcp

if TYPE_CHECKING:
    import polars as pl

_DEFAULT_ROW_LIMIT: int = 100
"""The rows a query returns when the caller names no limit. The manifest carries one job entry per pipeline job per
session, so an unbounded projection of a fully processed project runs to megabytes."""

_ORCHESTRATION_COLUMNS: tuple[str, ...] = ("animal", "session", "session_path", "tracker_paths")
"""The columns every query returns regardless of the projection the caller asks for. They are what turns a manifest
row back into an argument for the batch tools, so omitting them would leave a result an agent cannot act on."""

_HEAVY_COLUMNS: frozenset[str] = frozenset({"jobs", "notes"})
"""The columns a projection omits by default. ``jobs`` holds one struct per tracked job and ``notes`` holds free
text, so both are requested explicitly rather than carried in a survey of many sessions."""


@mcp.tool()
def generate_project_manifest_tool(project_path: str) -> dict[str, Any]:
    """Regenerates the target project's manifest file and returns a summary of the resulting snapshot.

    Runs to completion before returning, because generation reads only small metadata files and finishes in about a
    second for a few hundred sessions. Any existing manifest for the project is overwritten with a fresh snapshot.
    Run this between execution graphs, since a manifest generated while jobs are running captures a state that is
    already stale by the time it is read.

    Args:
        project_path: The absolute path to the project's root data directory.

    Returns:
        A response dict carrying every field of the manifest summary, alongside ``project_path``, ``manifest_path``,
        and the ``elapsed_seconds`` generation took. Returns an error when the project directory holds no sessions
        or cannot be read.
    """
    directory = Path(project_path)
    start = perf_counter()
    try:
        generate_project_manifest(project_directory=directory)
    except Exception as exception:
        return _error_response(message=f"Unable to generate the manifest for '{project_path}'. {exception}")
    elapsed = perf_counter() - start

    manifest_path = project_manifest_path(project_directory=directory)
    manifest = ProjectManifest(manifest_file=manifest_path)
    return _ok_response(
        project_path=str(directory),
        manifest_path=str(manifest_path),
        elapsed_seconds=round(elapsed, 3),
        **manifest.summarize(),
    )


@mcp.tool()
def read_project_manifest_tool(
    project_path: str,
    animal: int | None = None,
    columns: list[str] | None = None,
    limit: int = _DEFAULT_ROW_LIMIT,
) -> dict[str, Any]:
    """Reads rows out of an existing project manifest, projected to the requested columns and capped in length.

    Reads the stored snapshot rather than the session hierarchy, so the cost is independent of how many sessions the
    project holds. The columns that map a row back onto the batch tools are always present, so a result feeds
    ``prepare_batch_tool`` and ``reset_processing_jobs_tool`` without a second lookup.

    Args:
        project_path: The absolute path to the project's root data directory.
        animal: The animal identifier to restrict the result to. Omit to read every animal in the project.
        columns: The manifest columns to return, added to the always-present orchestration columns. Omit for every
            column except the per-job registry and the experimenter notes, which are large enough to request by name.
        limit: The maximum number of rows to return. Values below 1 return every matching row.

    Returns:
        A response dict with ``project_path``, ``manifest_path``, the ``columns`` projected, the ``rows`` returned,
        ``total_rows`` matching before the cap, a ``truncated`` flag, and the ``sessions`` list. Returns an error
        when no manifest exists or the requested animal or columns are absent.
    """
    directory = Path(project_path)
    manifest_path = project_manifest_path(project_directory=directory)
    if not manifest_path.is_file():
        return _error_response(
            message=(
                f"No manifest file exists at '{manifest_path}'. Generate it with generate_project_manifest_tool "
                f"before reading it."
            )
        )

    manifest = ProjectManifest(manifest_file=manifest_path)
    frame = manifest.data

    if animal is not None:
        if animal not in manifest.animals:
            return _error_response(
                message=(
                    f"Animal '{animal}' did not participate in the '{directory.stem}' project. Available animals: "
                    f"{list(manifest.animals)}."
                )
            )
        frame = frame.filter(frame["animal"] == animal)

    projection = _resolve_projection(frame=frame, columns=columns)
    if isinstance(projection, dict):
        return projection

    total_rows = frame.height
    capped = frame if limit < 1 else frame.head(limit)
    return _ok_response(
        project_path=str(directory),
        manifest_path=str(manifest_path),
        columns=projection,
        rows=capped.height,
        total_rows=total_rows,
        truncated=capped.height < total_rows,
        sessions=capped.select(projection).to_dicts(),
    )


@mcp.tool()
def get_manifest_status_tool(project_path: str) -> dict[str, Any]:
    """Reports the state of the target project's last manifest generation from its processing tracker.

    Reads the tracker alone and rescans nothing, so it answers whether the stored manifest is trustworthy without
    paying to rebuild it.

    Args:
        project_path: The absolute path to the project's root data directory.

    Returns:
        A response dict with ``project_path``, ``tracker_path``, ``manifest_path``, whether the manifest file
        ``exists``, and the ``status`` of the generation job alongside any ``error_message`` it recorded. A project
        whose manifest has never been generated reports a ``not_started`` status.
    """
    directory = Path(project_path)
    tracker_path = directory.joinpath(ProcessingTrackers.MANIFEST)
    manifest_path = project_manifest_path(project_directory=directory)

    response: dict[str, Any] = {
        "project_path": str(directory),
        "tracker_path": str(tracker_path),
        "manifest_path": str(manifest_path),
        "exists": manifest_path.is_file(),
        "status": "not_started",
    }
    if not tracker_path.is_file():
        return _ok_response(**response)

    job_id = ProcessingTracker.generate_job_id(job_name=MANIFEST_JOB_NAME, specifier=directory.stem)
    job_state = ProcessingTracker(file_path=tracker_path).snapshot().get(job_id)
    if job_state is None:
        return _ok_response(**response)

    response["status"] = job_state.status.name.lower()
    if job_state.error_message is not None:
        response["error_message"] = job_state.error_message
    return _ok_response(**response)


def _resolve_projection(frame: pl.DataFrame, columns: list[str] | None) -> list[str] | dict[str, Any]:
    """Resolves the column projection a manifest query returns, or an error response naming the unknown columns.

    Args:
        frame: The manifest frame the projection applies to.
        columns: The columns the caller requested, or None for the default projection.

    Returns:
        The ordered column list to project, or an error response dict when a requested column does not exist.
    """
    available = list(frame.columns)
    if columns is None:
        return [column for column in available if column not in _HEAVY_COLUMNS]

    unknown = sorted({column for column in columns if column not in available})
    if unknown:
        return _error_response(message=f"Unknown manifest column(s) {unknown}. Available columns: {available}.")

    requested = list(_ORCHESTRATION_COLUMNS) + [column for column in columns if column not in _ORCHESTRATION_COLUMNS]
    return [column for column in available if column in set(requested)]


def _ok_response(**payload: Any) -> dict[str, Any]:  # noqa: ANN401
    """Constructs a successful response dict with a ``success`` flag set to True."""
    return {"success": True, **payload}


def _error_response(message: str) -> dict[str, Any]:
    """Constructs a failure response dict with a ``success`` flag set to False and the provided error message."""
    return {"success": False, "error": message}
