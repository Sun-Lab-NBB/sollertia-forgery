"""Provides the Model Context Protocol (MCP) tools for planning what a unit's jobs will cost and for reading the
project-level projection of those plans.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any
from pathlib import Path

import polars as pl

from .mcp_instance import mcp
from ..orchestration import (
    DATASET_UNIT,
    SESSION_UNIT,
    project_plan_path,
    resolve_dataset_plan,
    resolve_session_plan,
    generate_project_plan,
)

_DEFAULT_ROW_LIMIT: int = 200
"""The rows a plan query returns when the caller names no limit. A fully planned project carries one row per job of
every session, so an unbounded projection runs to thousands of rows."""

_UNIT_KINDS: tuple[str, ...] = (SESSION_UNIT, DATASET_UNIT)
"""The unit kinds the projection holds, which is what a caller may filter it by."""


@mcp.tool()
def plan_session_jobs_tool(session_paths: list[str], *, regenerate_plan: bool = False) -> dict[str, Any]:
    """Records what every processing job of one or more sessions will cost, caching the figures beside each session.

    Estimation reads each session's raw acquisition data, opening video containers and image headers, so this is the
    expensive half of planning and the reason it is a call of its own. A session already carrying a plan keeps every
    recorded figure and is only extended with jobs the cache does not hold, so re-running this is cheap and safe.

    A session that cannot be planned is reported in its own entry and does not abort the others.

    Args:
        session_paths: The session root directories to plan.
        regenerate_plan: Determines whether to re-estimate the figures a cache already holds. Leave False unless a
            deliberate retune should be adopted, since a submission may already have been sized against the recorded
            figures.

    Returns:
        A response dict with ``total_units``, ``total_jobs``, the ``elapsed_seconds`` planning took, and a ``units``
        list carrying each session's ``session_path``, ``unit_name``, ``plan_path``, ``job_count``, and
        ``summed_memory_mb``, or an ``error``.
    """
    return _plan_units(unit_paths=session_paths, unit_kind=SESSION_UNIT, regenerate_plan=regenerate_plan)


@mcp.tool()
def plan_dataset_jobs_tool(dataset_paths: list[str], *, regenerate_plan: bool = False) -> dict[str, Any]:
    """Records what every forging job of one or more datasets will cost, caching the figures at each dataset root.

    A dataset's figures follow from the single-day outputs its jobs consume, and admission already requires a session
    to carry those outputs, so a dataset is plannable as soon as its hierarchy is defined.

    Args:
        dataset_paths: The dataset root directories to plan.
        regenerate_plan: Determines whether to re-estimate the figures a cache already holds.

    Returns:
        A response dict with ``total_units``, ``total_jobs``, the ``elapsed_seconds`` planning took, and a ``units``
        list carrying each dataset's ``dataset_path``, ``unit_name``, ``plan_path``, ``job_count``, and
        ``summed_memory_mb``, or an ``error``.
    """
    return _plan_units(unit_paths=dataset_paths, unit_kind=DATASET_UNIT, regenerate_plan=regenerate_plan)


@mcp.tool()
def generate_project_plan_tool(project_path: str) -> dict[str, Any]:
    """Projects every plan cache under a project into one table at the project root.

    Reads the caches alone and estimates nothing, so this is the cheap half of planning and the half that ships. Pull
    the resulting file to size a remote submission without reading the data root it was planned against.

    A unit carrying no cache contributes no rows, so plan the units first and treat a job absent from the projection
    as unplanned rather than free.

    Args:
        project_path: The absolute path to the project's root data directory.

    Returns:
        A response dict with ``project_path``, ``plan_path``, ``total_jobs``, ``summed_memory_mb``,
        ``widest_job_cores``, ``jobs_without_a_modeled_estimate``, a per-pipeline ``breakdown``, and the
        ``elapsed_seconds`` the projection took. Returns an error when the project cannot be read.
    """
    directory = Path(project_path)
    start = perf_counter()
    try:
        plan_path = generate_project_plan(project_directory=directory)
    except Exception as exception:
        return _error_response(message=f"Unable to project the plans under '{project_path}'. {exception}")
    elapsed = perf_counter() - start

    frame = pl.read_ipc(source=plan_path, memory_map=True)
    return _ok_response(
        project_path=str(directory),
        plan_path=str(plan_path),
        elapsed_seconds=round(elapsed, 3),
        **_plan_totals(frame=frame),
        breakdown=_plan_breakdown(frame=frame),
    )


@mcp.tool()
def read_project_plan_tool(
    project_path: str,
    unit_kind: str | None = None,
    pipelines: list[str] | None = None,
    limit: int = _DEFAULT_ROW_LIMIT,
) -> dict[str, Any]:
    """Reads the planned cores and memory of a project's jobs out of its stored projection.

    Reads the stored table rather than any unit's data, so the cost is independent of how much the project holds.
    Filters narrow the rows returned, while the totals and the breakdown always span every row the projection holds,
    so narrowing what is listed never distorts what is reported.

    Args:
        project_path: The absolute path to the project's root data directory.
        unit_kind: Restricts the listed rows to one unit kind, either ``session`` or ``dataset``.
        pipelines: Restricts the listed rows to these pipelines.
        limit: The maximum number of rows to list. Values below 1 list every matching row.

    Returns:
        A response dict with ``project_path``, ``plan_path``, the whole-projection totals and per-pipeline
        ``breakdown``, the ``rows`` listed, ``matched_rows`` before the cap, a ``truncated`` flag, and the ``jobs``
        list. Returns an error when no projection exists or a filter names an unknown value.
    """
    directory = Path(project_path)
    plan_path = project_plan_path(project_directory=directory)
    if not plan_path.is_file():
        return _error_response(
            message=(
                f"No plan projection exists at '{plan_path}'. Plan the project's units, then run "
                f"generate_project_plan_tool before reading it."
            )
        )

    frame = pl.read_ipc(source=plan_path, memory_map=True)
    matched = frame

    if unit_kind is not None:
        if unit_kind not in _UNIT_KINDS:
            return _error_response(message=f"Unknown unit kind '{unit_kind}'. Available: {', '.join(_UNIT_KINDS)}.")
        matched = matched.filter(pl.col("unit_kind") == unit_kind)

    if pipelines is not None:
        available = sorted(set(frame["pipeline"].to_list()))
        unknown = sorted({pipeline for pipeline in pipelines if pipeline not in available})
        if unknown:
            return _error_response(message=f"Unknown pipeline(s) {unknown}. Available: {available}.")
        matched = matched.filter(pl.col("pipeline").is_in(pipelines))

    capped = matched if limit < 1 else matched.head(limit)
    return _ok_response(
        project_path=str(directory),
        plan_path=str(plan_path),
        **_plan_totals(frame=frame),
        breakdown=_plan_breakdown(frame=frame),
        rows=capped.height,
        matched_rows=matched.height,
        truncated=capped.height < matched.height,
        jobs=capped.to_dicts(),
    )


def _plan_units(unit_paths: list[str], unit_kind: str, *, regenerate_plan: bool) -> dict[str, Any]:
    """Plans every unit of one kind, reporting each independently.

    Args:
        unit_paths: The unit root directories to plan.
        unit_kind: Whether the units are sessions or datasets, which selects the resolver and names the response key.
        regenerate_plan: Determines whether to re-estimate the figures a cache already holds.

    Returns:
        The response dict the calling tool returns.
    """
    resolve = resolve_session_plan if unit_kind == SESSION_UNIT else resolve_dataset_plan
    path_key = "session_path" if unit_kind == SESSION_UNIT else "dataset_path"

    units: list[dict[str, Any]] = []
    total_jobs = 0
    start = perf_counter()
    for unit_path in unit_paths:
        try:
            plan = resolve(Path(unit_path), regenerate_plan=regenerate_plan)
        except Exception as exception:
            units.append({path_key: unit_path, "error": str(exception), "job_count": 0})
            continue
        total_jobs += len(plan.entries)
        units.append(
            {
                path_key: unit_path,
                "unit_name": plan.unit_name,
                "job_count": len(plan.entries),
                "summed_memory_mb": sum(entry.memory_mb for entry in plan.entries),
            }
        )

    return _ok_response(
        total_units=len(units),
        total_jobs=total_jobs,
        elapsed_seconds=round(perf_counter() - start, 3),
        units=units,
    )


def _plan_totals(frame: pl.DataFrame) -> dict[str, Any]:
    """Summarizes a plan projection into the figures a submission is sized against.

    Args:
        frame: The whole plan projection.

    Returns:
        A dictionary with the total jobs, the summed and largest memory, the widest core allocation, and how many jobs
        carry no modeled estimate.
    """
    if frame.height == 0:
        return {
            "total_jobs": 0,
            "summed_memory_mb": 0,
            "largest_job_memory_mb": 0,
            "widest_job_cores": 0,
            "jobs_without_a_modeled_estimate": 0,
        }
    return {
        "total_jobs": frame.height,
        "summed_memory_mb": int(frame["memory_mb"].sum()),
        "largest_job_memory_mb": int(frame["memory_mb"].max()),  # type: ignore[arg-type]
        "widest_job_cores": int(frame["cores"].max()),  # type: ignore[arg-type]
        "jobs_without_a_modeled_estimate": int(frame.filter(~pl.col("memory_modeled")).height),
    }


def _plan_breakdown(frame: pl.DataFrame) -> list[dict[str, Any]]:
    """Groups a plan projection by pipeline, which is the grain a caller admits work at.

    Args:
        frame: The whole plan projection.

    Returns:
        A list of per-pipeline entries, each carrying the pipeline, its job count, its summed memory, and its widest
        core allocation, ordered by pipeline.
    """
    if frame.height == 0:
        return []
    grouped = (
        frame.group_by("unit_kind", "pipeline")
        .agg(
            pl.len().alias("jobs"),
            pl.col("memory_mb").sum().alias("summed_memory_mb"),
            pl.col("cores").max().alias("widest_job_cores"),
        )
        .sort("unit_kind", "pipeline")
    )
    return grouped.to_dicts()


def _ok_response(**payload: Any) -> dict[str, Any]:  # noqa: ANN401
    """Constructs a successful response dict with a ``success`` flag set to True."""
    return {"success": True, **payload}


def _error_response(message: str) -> dict[str, Any]:
    """Constructs a failure response dict with a ``success`` flag set to False and the provided error message."""
    return {"success": False, "error": message}
