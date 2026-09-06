"""Provides the Model Context Protocol (MCP) tools for planning what a unit's jobs will cost and for reading the
project-level projection of those plans.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

import polars as pl
from ataraxis_time import PrecisionTimer, TimerPrecisions

from .responses import (
    ok_response,
    page_fields,
    project_item,
    resolve_page,
    error_response,
    reject_unknown,
    frame_breakdown,
    resolve_detail_limit,
    resolve_elapsed_seconds,
)
from .mcp_instance import mcp
from ..orchestration import (
    DATASET_UNIT,
    SESSION_UNIT,
    PROJECT_PLAN_SCHEMA,
    project_plan_path,
    resolve_project_root,
)
from .host_resolution import (
    HOST_LABELS,
    reported_project_path,
    resolve_execution_host,
    resolve_readable_project,
    unsupported_host_message,
)

_PLAN_AXES: tuple[str, ...] = ("unit_kind", "animal", "dataset", "pipeline", "job_name")
"""The axes a plan breakdown counts, each of which is also a column by which a caller may filter."""

_PLAN_SEMI_FIELDS: tuple[str, ...] = (
    "unit_kind",
    "animal",
    "session",
    "dataset",
    "pipeline",
    "job_name",
    "specifier",
    "cores",
    "memory_mb",
    "resident_mb",
)
"""The job fields a semi-detail listing carries, which are the job's subject, its identity, and its figures. Both
memory figures are carried, because a local pool is budgeted against the anonymous one while the scheduler is given
the resident one, so a reader sizing a batch needs whichever matches the host it targets."""

_PLAN_DETAIL_FIELDS: tuple[str, ...] = ("job_id", "memory_modeled", "prerequisite_ids")
"""The job fields detail adds, which are the tracked job's identifier, whether a model of the job's own input produced
its memory figure, and its prerequisite jobs."""


@mcp.tool()
def plan_session_jobs_tool(
    session_paths: list[str], host: str = "local", *, regenerate_plan: bool = False
) -> dict[str, Any]:
    """Records what every processing job of one or more sessions will cost, caching the figures beside each session.

    Estimation reads each session's raw acquisition data, opening video containers and image headers, so this is the
    expensive half of planning and the reason it is a call of its own. A session already carrying a plan keeps every
    recorded figure and is only extended with jobs the cache does not hold, so re-running this is cheap and safe.

    A session that cannot be planned is reported in its own entry and does not abort the others. A job whose input the
    sizing pass cannot read is left out of its session's plan and reported with the reason, leaving every job beside it
    planned.

    Args:
        session_paths: The session root directories to plan, which are paths ON THE SERVER for ``remote``.
        host: Where the data sits, either ``local`` for this machine or ``remote`` for the configured compute server.
        regenerate_plan: Determines whether to re-estimate the figures a cache already holds. Leave False unless a
            deliberate retune should be adopted, since a submission may already have been sized against the recorded
            figures. A cache stamped with another resource-model version is re-estimated whatever this asks.

    Returns:
        A response dict with ``host``, ``total_units``, ``total_jobs``, and the ``elapsed_seconds`` planning took.
        Carries a ``units`` list, whose entries hold each session's ``unit_path``, ``unit_name``, ``job_count``, and
        ``summed_memory_mb``, and ``summed_resident_mb``, or its ``unit_path``, a ``job_count`` of zero, and the
        ``error`` that stopped it. A ``local`` entry also holds the ``unsized_jobs`` refusals its sizing pass
        recorded, mapped to their reasons.
    """
    return _plan_units(unit_paths=session_paths, unit_kind=SESSION_UNIT, host=host, regenerate_plan=regenerate_plan)


@mcp.tool()
def plan_dataset_jobs_tool(
    dataset_paths: list[str], host: str = "local", *, regenerate_plan: bool = False
) -> dict[str, Any]:
    """Records what every forging job of one or more datasets will cost, caching the figures at each dataset root.

    A dataset's figures follow from the single-day outputs its jobs consume, and admission already requires a session
    to carry those outputs, so a dataset is plannable as soon as its hierarchy is defined. A job whose input the sizing
    pass cannot read is left out of its dataset's plan and reported with the reason, leaving every job beside it
    planned.

    Args:
        dataset_paths: The dataset root directories to plan, which are paths ON THE SERVER for ``remote``.
        host: Where the data sits, either ``local`` for this machine or ``remote`` for the configured compute server.
        regenerate_plan: Determines whether to re-estimate the figures a cache already holds. A cache stamped with
            another resource-model version is re-estimated whatever this asks.

    Returns:
        A response dict with ``host``, ``total_units``, ``total_jobs``, and the ``elapsed_seconds`` planning took.
        Carries a ``units`` list, whose entries hold each dataset's ``unit_path``, ``unit_name``, ``job_count``, and
        ``summed_memory_mb``, and ``summed_resident_mb``, or its ``unit_path``, a ``job_count`` of zero, and the
        ``error`` that stopped it. A ``local`` entry also holds the ``unsized_jobs`` refusals its sizing pass
        recorded, mapped to their reasons.
    """
    return _plan_units(unit_paths=dataset_paths, unit_kind=DATASET_UNIT, host=host, regenerate_plan=regenerate_plan)


@mcp.tool()
def generate_project_plan_tool(project_path: str, host: str = "local") -> dict[str, Any]:
    """Projects every plan cache under a project into one table at the project root.

    Reads the caches alone and estimates nothing, so this is the cheap half of planning and the half that ships. Pull
    the resulting file to size a remote submission without reading the data root against which it was planned.

    A unit carrying no cache contributes no rows, so plan the units first and treat a job absent from the projection
    as unplanned rather than free.

    Args:
        project_path: The absolute path to the project's root data directory, which is a path ON THE SERVER for
            ``remote``.
        host: Where the data sits, either ``local`` for this machine or ``remote`` for the configured compute server.

    Returns:
        A response dict with ``project_path``, ``host``, ``plan_path``, ``total_jobs``, ``summed_memory_mb``,
        ``largest_job_memory_mb``, ``summed_resident_mb``, ``largest_job_resident_mb``, ``widest_job_cores``, a
        per-unit-kind and per-pipeline ``pipeline_totals``, and the
        ``elapsed_seconds`` the projection took. Returns an error when the project cannot be read.
    """
    if host not in HOST_LABELS:
        return error_response(message=unsupported_host_message(host=host))

    directory = Path(project_path)
    plan_path = project_plan_path(project_directory=directory)
    timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)
    try:
        with resolve_execution_host(host=host) as execution_host:
            # Naming no unit leaves the projection alone to run, since this reprojects what the units already planned.
            execution_host.plan(project_root=directory, unit_paths=[], unit_kind=SESSION_UNIT, replan=False)
            rows = execution_host.read_rows(path=plan_path)
    except Exception as exception:
        return error_response(message=f"Unable to project the plans under '{project_path}'. {exception}")

    frame = pl.DataFrame(data=rows, schema=PROJECT_PLAN_SCHEMA, strict=False)
    return ok_response(
        project_path=str(directory),
        host=host,
        plan_path=str(plan_path),
        elapsed_seconds=resolve_elapsed_seconds(timer=timer),
        **_plan_totals(frame=frame),
        pipeline_totals=_plan_breakdown(frame=frame),
    )


@mcp.tool()
def read_project_plan_tool(
    project_path: str,
    host: str = "local",
    unit_kind: str | None = None,
    animal: str | None = None,
    dataset: str | None = None,
    pipelines: list[str] | None = None,
    job_names: list[str] | None = None,
    limit: int | None = None,
    start_row: int = 0,
    *,
    include_items: bool = False,
    detailed: bool = False,
) -> dict[str, Any]:
    """Reads the planned cores and memory of a project's jobs out of its stored projection, in three widening stages.

    A bare call reports the figures against which a submission is sized, alongside a ``breakdown`` naming every unit
    kind, animal, dataset, pipeline, and job type the projection holds. An axis holding more distinct values than the
    shared cap reports how many it holds in place of its counts, and filtering on that axis reaches the jobs
    themselves. Naming a filter adds a page of planned jobs carrying their subject and their figures. Opting into
    detail adds each job's tracked identifier, whether a model of the job's own input produced its memory figure, and
    its prerequisite jobs.

    The totals and the breakdown span every planned job regardless of the filters, so narrowing what is listed never
    distorts what is reported. Reads the stored table rather than any unit's data, so the cost is independent of how
    much the project holds.

    Args:
        project_path: The absolute path to the project's root data directory, which is a path ON THE SERVER for
            ``remote``, where only its final component names the project.
        host: Where the project sits, either ``local`` for this machine or ``remote`` for the configured
            compute server. A remote read mirrors the project's artifacts onto this machine and reads the
            mirror, so it reports what the project currently records without regenerating anything.
        unit_kind: Restricts the listing to one unit kind, either ``session`` or ``dataset``.
        animal: Restricts the listing to one animal's sessions.
        dataset: Restricts the listing to one dataset's forging jobs.
        pipelines: Restricts the listing to these pipelines.
        job_names: Restricts the listing to these job type names.
        limit: The jobs to list. Defaults to 200, or to 50 when detail is requested. A value at or below zero lists
            every match, which is how a caller reading under a tight filter takes the whole result at once.
        start_row: The match index at which to begin the listing. Follow ``next_start_row`` to walk a long result.
        include_items: Determines whether to list jobs when no filter is named.
        detailed: Determines whether the listed jobs report their tracked identifier, whether a model of their own
            input produced their memory figure, and their prerequisite jobs.

    Returns:
        A response dict with ``project_path``, ``plan_path``, the whole-projection totals, and a ``breakdown`` per
        axis. Carries a ``jobs`` list with ``rows``, ``matched_rows``, ``start_row``, and ``next_start_row`` whenever a
        filter is named or the listing is requested. Returns an error when no projection exists or a filter names a
        value the projection does not hold.
    """
    if host not in HOST_LABELS:
        return error_response(message=unsupported_host_message(host=host))
    try:
        directory = resolve_readable_project(project_path=project_path, host=host)
    except Exception as exception:
        return error_response(message=f"Unable to read the {host} project. {exception}")

    plan_path = project_plan_path(project_directory=directory)
    if not plan_path.is_file():
        return error_response(
            message=(
                f"No plan projection exists at '{plan_path}'. Plan the project's units, then run "
                f"generate_project_plan_tool before reading it."
            )
        )

    # A projection written by another model states a narrower column set than this reader totals, so the read and the
    # totals answer through the envelope rather than raising out of the session.
    try:
        frame = pl.read_ipc(source=plan_path, memory_map=True)
        totals = _plan_totals(frame=frame)
        breakdown = frame_breakdown(frame=frame, axes=_PLAN_AXES)
    except Exception as exception:
        return error_response(
            message=(
                f"Unable to read the plan projection at '{plan_path}'. {exception} Regenerate it with "
                f"generate_project_plan_tool, which rebuilds the table from the units' own caches."
            )
        )

    response = ok_response(
        project_path=reported_project_path(project_path=project_path, directory=directory, host=host),
        plan_path=str(plan_path),
        **totals,
        breakdown=breakdown,
    )

    singles: dict[str, str | None] = {"unit_kind": unit_kind, "animal": animal, "dataset": dataset}
    multiples: dict[str, list[str] | None] = {"pipeline": pipelines, "job_name": job_names}
    if not any(value is not None for value in (*singles.values(), *multiples.values())) and not include_items:
        return response

    matched = frame
    for column, value in singles.items():
        if value is None:
            continue
        rejection = reject_unknown(frame=frame, column=column, values=[value], subject="planned job")
        if rejection is not None:
            return rejection
        matched = matched.filter(pl.col(column) == value)
    for column, values in multiples.items():
        if values is None:
            continue
        rejection = reject_unknown(frame=frame, column=column, values=values, subject="planned job")
        if rejection is not None:
            return rejection
        matched = matched.filter(pl.col(column).is_in(values))

    fields = (*_PLAN_SEMI_FIELDS, *_PLAN_DETAIL_FIELDS) if detailed else _PLAN_SEMI_FIELDS
    window = resolve_page(
        total=matched.height, limit=resolve_detail_limit(limit=limit, detailed=detailed), start_row=start_row
    )
    page = matched.slice(offset=window.start, length=window.length)
    response["jobs"] = [project_item(item=item, fields=fields) for item in page.to_dicts()]
    response.update(page_fields(window=window, total=matched.height, listed=page.height))
    return response


def _plan_units(unit_paths: list[str], unit_kind: str, host: str, *, regenerate_plan: bool) -> dict[str, Any]:
    """Plans every unit of one kind on the named host, reporting each independently.

    Args:
        unit_paths: The unit root directories to plan.
        unit_kind: The kind of processing unit the paths name, either ``session`` or ``dataset``.
        host: Where the data sits, either ``local`` or ``remote``.
        regenerate_plan: Determines whether to re-estimate the figures a cache already holds.

    Returns:
        The response dict the calling tool returns.
    """
    if host not in HOST_LABELS:
        return error_response(message=unsupported_host_message(host=host))

    units = [Path(path) for path in unit_paths]
    if not units:
        return error_response(message="No processing unit was named.")

    try:
        project_root = resolve_project_root(unit_paths=units, unit_kind=unit_kind)
    except ValueError as exception:
        return error_response(message=str(exception))

    timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)
    try:
        with resolve_execution_host(host=host) as execution_host:
            planned = execution_host.plan(
                project_root=project_root, unit_paths=units, unit_kind=unit_kind, replan=regenerate_plan
            )
    except Exception as exception:
        return error_response(message=f"Unable to plan the {host} units. {exception}")

    return ok_response(
        host=host,
        total_units=len(planned),
        total_jobs=sum(int(entry["job_count"]) for entry in planned),
        elapsed_seconds=resolve_elapsed_seconds(timer=timer),
        units=planned,
    )


def _plan_totals(frame: pl.DataFrame) -> dict[str, Any]:
    """Summarizes a plan projection into the figures against which a submission is sized.

    Args:
        frame: The whole plan projection.

    Returns:
        A dictionary with the total jobs, the summed and largest figure of both memory terms, and the widest core
        allocation.

    Notes:
        Both memory terms are totaled, because a caller sizing work for this machine's pool budgets against the
        anonymous term while a caller sizing a scheduler submission budgets against the resident one. Reporting the
        anonymous total alone leaves the second caller under-requesting by whatever its jobs map.
    """
    if frame.height == 0:
        return {
            "total_jobs": 0,
            "summed_memory_mb": 0,
            "largest_job_memory_mb": 0,
            "summed_resident_mb": 0,
            "largest_job_resident_mb": 0,
            "widest_job_cores": 0,
        }
    return {
        "total_jobs": frame.height,
        "summed_memory_mb": int(frame["memory_mb"].sum()),
        "largest_job_memory_mb": int(frame["memory_mb"].max()),  # type: ignore[arg-type]
        "summed_resident_mb": int(frame["resident_mb"].sum()),
        "largest_job_resident_mb": int(frame["resident_mb"].max()),  # type: ignore[arg-type]
        "widest_job_cores": int(frame["cores"].max()),  # type: ignore[arg-type]
    }


def _plan_breakdown(frame: pl.DataFrame) -> list[dict[str, Any]]:
    """Groups a plan projection by unit kind and pipeline, which is the grain at which a caller admits work.

    Args:
        frame: The whole plan projection.

    Returns:
        A list of entries, each carrying the unit kind, the pipeline, its job count, both of its summed memory
        figures, and its widest core allocation, ordered by unit kind and then by pipeline.
    """
    if frame.height == 0:
        return []
    grouped = (
        frame.group_by("unit_kind", "pipeline")
        .agg(
            pl.len().alias("jobs"),
            pl.col("memory_mb").sum().alias("summed_memory_mb"),
            pl.col("resident_mb").sum().alias("summed_resident_mb"),
            pl.col("cores").max().alias("widest_job_cores"),
        )
        .sort("unit_kind", "pipeline")
    )
    return grouped.to_dicts()
