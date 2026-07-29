"""Provides the Model Context Protocol (MCP) tools for generating, inspecting, and querying a project's state
artifacts.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any
from pathlib import Path

import polars as pl
from sollertia_shared_assets import ProcessingTrackers
from ataraxis_data_structures import ProcessingTracker

from ..managing import (
    MANIFEST_JOB_NAME,
    ProjectManifest,
    project_jobs_path,
    project_manifest_path,
    generate_project_manifest,
)
from .responses import (
    ok_response,
    page_fields,
    count_values,
    project_item,
    resolve_page,
    error_response,
    resolve_detail_limit,
)
from .mcp_instance import mcp

_MANIFEST_AXES: tuple[str, ...] = (
    "animal",
    "type",
    "system",
    "complete",
    "integrity",
    "runtime",
    "microcontroller",
    "video",
    "two_photon",
)
"""The manifest columns a caller may filter sessions by, and the axes its breakdown counts. Every one holds a low
cardinality value, which is what makes a breakdown over it worth reading. The five pipeline columns each hold a 0 or a
1, so their breakdown reports how many sessions have finished that pipeline."""

_MANIFEST_SEMI_FIELDS: tuple[str, ...] = (
    "animal",
    "session",
    "session_path",
    "date",
    "type",
    "system",
    "complete",
    "integrity",
    "runtime",
    "microcontroller",
    "video",
    "two_photon",
)
"""The session fields a semi-detail listing carries, which is the session's identity and whether each of its
pipelines finished."""

_MANIFEST_DETAIL_FIELDS: tuple[str, ...] = ("notes",)
"""The session field detail adds, which is the free-text experimenter notes."""

_JOB_AXES: tuple[str, ...] = ("animal", "pipeline", "job_name", "status")
"""The job columns a caller may filter by, and the axes the job breakdown counts."""

_JOB_SEMI_FIELDS: tuple[str, ...] = ("animal", "session", "pipeline", "job_name", "specifier", "status", "job_id")
"""The job fields a semi-detail listing carries. ``job_id`` is included because it is the key a caller resets a job
by, so a listing that omitted it could not be acted on."""

_JOB_DETAIL_FIELDS: tuple[str, ...] = ("executor_id", "error_message", "started_at", "completed_at")
"""The job fields detail adds, which are the provenance and timing a caller reads when examining one job closely."""


@mcp.tool()
def generate_project_manifest_tool(project_path: str) -> dict[str, Any]:
    """Regenerates the target project's manifest and job artifacts, returning a summary of the resulting snapshot.

    Runs to completion before returning, because generation reads only small metadata files and finishes in about a
    second for a few hundred sessions. Both artifacts are written from one walk under one lock, so they never disagree
    about a session. Run this between execution graphs, since a snapshot taken while jobs run is already stale when it
    is read.

    Args:
        project_path: The absolute path to the project's root data directory.

    Returns:
        A response dict carrying every field of the manifest summary, alongside ``project_path``, ``manifest_path``,
        ``jobs_path``, the ``total_jobs`` the job artifact holds, and the ``elapsed_seconds`` generation took. Returns
        an error when the project directory holds no sessions or cannot be read.
    """
    directory = Path(project_path)
    start = perf_counter()
    try:
        generate_project_manifest(project_directory=directory)
    except Exception as exception:
        return error_response(message=f"Unable to generate the state artifacts for '{project_path}'. {exception}")
    elapsed = perf_counter() - start

    manifest_path = project_manifest_path(project_directory=directory)
    jobs_path = project_jobs_path(project_directory=directory)
    return ok_response(
        project_path=str(directory),
        manifest_path=str(manifest_path),
        jobs_path=str(jobs_path),
        total_jobs=pl.read_ipc(source=jobs_path, memory_map=True).height if jobs_path.is_file() else 0,
        elapsed_seconds=round(elapsed, 3),
        **ProjectManifest(manifest_file=manifest_path).summarize(),
    )


@mcp.tool()
def read_project_manifest_tool(
    project_path: str,
    animal: int | None = None,
    session_type: str | None = None,
    system: str | None = None,
    pipeline_done: dict[str, int] | None = None,
    limit: int | None = None,
    start_row: int = 0,
    *,
    include_items: bool = False,
    detailed: bool = False,
) -> dict[str, Any]:
    """Reads a project's sessions out of its stored manifest, in three widening stages.

    A bare call reports the totals and a ``breakdown`` naming which values each filterable axis holds, which is what
    tells you what there is to filter on. Naming any filter, or asking for the listing, adds a page of sessions.
    Opting into detail adds the experimenter notes.

    Each pipeline column is a gross done indicator, so this answers which sessions are ready and nothing finer. Which
    jobs a pipeline holds, how each fared, and why one failed are read with ``read_project_jobs_tool``.

    The totals and the breakdown span every session in the project regardless of the filters, so narrowing what is
    listed never distorts what is reported. Reads the stored snapshot rather than the session hierarchy, so the cost is
    independent of how much the project holds.

    Args:
        project_path: The absolute path to the project's root data directory.
        animal: The animal identifier to restrict the listing to.
        session_type: The session type to restrict the listing to, as reported by the ``type`` breakdown axis.
        system: The acquisition system to restrict the listing to.
        pipeline_done: Whether each named pipeline must be finished, as ``1`` for done and ``0`` for not done, for
            example ``{"video": 0}`` to list the sessions whose video pipeline is outstanding. Column names come from
            the breakdown. Which jobs of that pipeline failed, and why, are read with ``read_project_jobs_tool``.
        limit: The sessions to list. Defaults to 200, or to 50 when detail is requested. A value at or below zero
            lists every match.
        start_row: The match index to begin the listing at. Follow ``next_start_row`` to walk a long result.
        include_items: Determines whether to list sessions when no filter is named.
        detailed: Determines whether the listed sessions carry their experimenter notes.

    Returns:
        A response dict with ``project_path``, ``manifest_path``, ``total_sessions``, and a ``breakdown`` per axis.
        Carries a ``sessions`` list with ``rows``, ``matched_rows``, ``start_row``, and ``next_start_row`` whenever a
        filter is named or the listing is requested. Returns an error when no manifest exists or a filter names a value
        the project does not hold.
    """
    directory = Path(project_path)
    manifest_path = project_manifest_path(project_directory=directory)
    if not manifest_path.is_file():
        return error_response(
            message=(
                f"No manifest exists at '{manifest_path}'. Generate it with generate_project_manifest_tool before "
                f"reading it."
            )
        )

    frame = pl.read_ipc(source=manifest_path, memory_map=True)
    response = ok_response(
        project_path=str(directory),
        manifest_path=str(manifest_path),
        total_sessions=frame.height,
        breakdown=_breakdown(frame=frame, axes=_MANIFEST_AXES),
    )

    selectors: dict[str, Any] = {"animal": animal, "type": session_type, "system": system}
    if pipeline_done is not None:
        selectors.update(pipeline_done)
    narrowed = {column: value for column, value in selectors.items() if value is not None}
    if not narrowed and not include_items:
        return response

    matched = frame
    for column, value in narrowed.items():
        if column not in frame.columns:
            return error_response(message=f"Unknown manifest column '{column}'. Available: {sorted(frame.columns)}.")
        available = sorted({str(entry) for entry in frame[column].to_list()})
        if str(value) not in available:
            return error_response(message=f"No session has '{column}' of '{value}'. Available: {available}.")
        matched = matched.filter(pl.col(column).cast(pl.String) == str(value))

    fields = (*_MANIFEST_SEMI_FIELDS, *_MANIFEST_DETAIL_FIELDS) if detailed else _MANIFEST_SEMI_FIELDS
    window = resolve_page(
        total=matched.height,
        limit=resolve_detail_limit(limit=limit, detailed=detailed),
        start_row=start_row,
    )
    page = matched.slice(window.start, window.length)
    response["sessions"] = [project_item(item=item, fields=fields) for item in page.to_dicts()]
    response.update(page_fields(window=window, total=matched.height, listed=page.height))
    return response


@mcp.tool()
def read_project_jobs_tool(
    project_path: str,
    animal: str | None = None,
    session: str | None = None,
    pipelines: list[str] | None = None,
    job_names: list[str] | None = None,
    status: str | None = None,
    limit: int | None = None,
    start_row: int = 0,
    *,
    include_items: bool = False,
    detailed: bool = False,
) -> dict[str, Any]:
    """Reads a project's tracked jobs out of its stored job artifact, in three widening stages.

    A bare call reports the totals and a ``breakdown`` naming every animal, pipeline, job type, and status the project
    holds, which is how you find what needs attention without listing anything. Naming a filter adds a page of jobs
    carrying identity and status. Opting into detail adds the executor, the timestamps, and any recorded error.

    This is the tool for reading job state across a whole project. It reads the stored artifact rather than the
    trackers, so a snapshot pulled from another host answers without any access to the data it describes.

    Args:
        project_path: The absolute path to the project's root data directory.
        animal: The animal identifier to restrict the listing to.
        session: The session name to restrict the listing to.
        pipelines: The pipelines to restrict the listing to.
        job_names: The job type names to restrict the listing to, such as ``motion_energy``.
        status: The tracker status to restrict the listing to, such as ``FAILED``.
        limit: The jobs to list. Defaults to 200, or to 50 when detail is requested. A value at or below zero lists
            every match, which is how a caller reading under a tight filter takes the whole result at once.
        start_row: The match index to begin the listing at. Follow ``next_start_row`` to walk a long result.
        include_items: Determines whether to list jobs when no filter is named.
        detailed: Determines whether the listed jobs carry the executor, timestamps, and error text.

    Returns:
        A response dict with ``project_path``, ``jobs_path``, ``total_jobs``, and a ``breakdown`` per axis. Carries a
        ``jobs`` list with ``rows``, ``matched_rows``, ``start_row``, and ``next_start_row`` whenever a filter is named
        or the listing is requested. Returns an error when no job artifact exists or a filter names a value the project
        does not hold.
    """
    directory = Path(project_path)
    jobs_path = project_jobs_path(project_directory=directory)
    if not jobs_path.is_file():
        return error_response(
            message=(
                f"No job artifact exists at '{jobs_path}'. Generate it with generate_project_manifest_tool before "
                f"reading it."
            )
        )

    frame = pl.read_ipc(source=jobs_path, memory_map=True)
    response = ok_response(
        project_path=str(directory),
        jobs_path=str(jobs_path),
        total_jobs=frame.height,
        breakdown=_breakdown(frame=frame, axes=_JOB_AXES),
    )

    singles: dict[str, str | None] = {"animal": animal, "session": session, "status": status}
    multiples: dict[str, list[str] | None] = {"pipeline": pipelines, "job_name": job_names}
    if not any(value is not None for value in (*singles.values(), *multiples.values())) and not include_items:
        return response

    matched = frame
    for column, value in singles.items():
        if value is None:
            continue
        rejection = _reject_unknown(frame=frame, column=column, values=[value])
        if rejection is not None:
            return rejection
        matched = matched.filter(pl.col(column) == value)
    for column, values in multiples.items():
        if values is None:
            continue
        rejection = _reject_unknown(frame=frame, column=column, values=values)
        if rejection is not None:
            return rejection
        matched = matched.filter(pl.col(column).is_in(values))

    fields = (*_JOB_SEMI_FIELDS, *_JOB_DETAIL_FIELDS) if detailed else _JOB_SEMI_FIELDS
    window = resolve_page(
        total=matched.height, limit=resolve_detail_limit(limit=limit, detailed=detailed), start_row=start_row
    )
    page = matched.slice(window.start, window.length)
    response["jobs"] = [project_item(item=item, fields=fields) for item in page.to_dicts()]
    response.update(page_fields(window=window, total=matched.height, listed=page.height))
    return response


@mcp.tool()
def get_manifest_status_tool(project_path: str) -> dict[str, Any]:
    """Reports the state of the target project's last state-artifact generation from its processing tracker.

    Reads the tracker alone and rescans nothing, so it answers whether the stored artifacts are trustworthy without
    paying to rebuild them.

    Args:
        project_path: The absolute path to the project's root data directory.

    Returns:
        A response dict with ``project_path``, ``tracker_path``, ``manifest_path``, ``jobs_path``, whether each
        artifact ``exists``, and the ``status`` of the generation job alongside any ``error_message`` it recorded. A
        project whose artifacts have never been generated reports a ``not_started`` status.
    """
    directory = Path(project_path)
    tracker_path = directory.joinpath(ProcessingTrackers.MANIFEST)
    manifest_path = project_manifest_path(project_directory=directory)
    jobs_path = project_jobs_path(project_directory=directory)

    response: dict[str, Any] = {
        "project_path": str(directory),
        "tracker_path": str(tracker_path),
        "manifest_path": str(manifest_path),
        "jobs_path": str(jobs_path),
        "exists": {"manifest": manifest_path.is_file(), "jobs": jobs_path.is_file()},
        "status": "not_started",
    }
    if not tracker_path.is_file():
        return ok_response(**response)

    job_id = ProcessingTracker.generate_job_id(job_name=MANIFEST_JOB_NAME, specifier=directory.stem)
    job_state = ProcessingTracker(file_path=tracker_path).snapshot().get(job_id)
    if job_state is None:
        return ok_response(**response)

    response["status"] = job_state.status.name.lower()
    if job_state.error_message is not None:
        response["error_message"] = job_state.error_message
    return ok_response(**response)


def _breakdown(frame: pl.DataFrame, axes: tuple[str, ...]) -> dict[str, dict[str, int]]:
    """Counts how many rows carry each value of every filterable axis.

    Notes:
        This is what a bare call reports in place of a listing. It names the values a caller can filter on and how much
        each would match, so an agent orients itself on one response rather than paging a whole artifact.

    Args:
        frame: The whole artifact.
        axes: The columns to count, which are the columns a caller may filter by.

    Returns:
        A dictionary mapping each present axis to its value counts.
    """
    return {axis: count_values(values=frame[axis].to_list()) for axis in axes if axis in frame.columns}


def _reject_unknown(frame: pl.DataFrame, column: str, values: list[str]) -> dict[str, Any] | None:
    """Builds the error response for a filter naming a value the artifact does not hold.

    Notes:
        Reports what is available rather than returning an empty page, because an empty page and a mistyped filter look
        identical to a caller otherwise.

    Args:
        frame: The whole artifact.
        column: The column being filtered.
        values: The values the caller named.

    Returns:
        The error response, or None when every named value is present.
    """
    if column not in frame.columns:
        return error_response(message=f"Unknown column '{column}'. Available: {sorted(frame.columns)}.")
    available = sorted({str(entry) for entry in frame[column].to_list() if entry is not None})
    unknown = sorted({value for value in values if value not in available})
    if unknown:
        return error_response(message=f"No job has '{column}' in {unknown}. Available: {available}.")
    return None
