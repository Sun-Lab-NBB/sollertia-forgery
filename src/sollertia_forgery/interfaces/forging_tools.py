"""Provides the Model Context Protocol (MCP) tools for defining the dataset hierarchy a forging batch runs against and
for snapshotting the state of its forging jobs.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

import polars as pl
from sollertia_shared_assets import DatasetData

from ..forging import (
    MULTIDAY_EXTRACTION_JOB_NAME,
    dataset_state_path,
    forging_tracker_path,
    define_forging_dataset,
    generate_dataset_state,
    discover_project_datasets,
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
from ..orchestration import resolve_job_cores

_DATASET_SEMI_FIELDS: tuple[str, ...] = (
    "name",
    "dataset_path",
    "session_type",
    "acquisition_system",
    "session_count",
    "animal_count",
)
"""The fields a dataset listing carries, which is the dataset's identity and how much it holds. Its job counts are
absent because reading them opens one stored table per dataset, so detail asks for them."""

_STATE_AXES: tuple[str, ...] = ("scope", "animal", "job_name", "status")
"""The snapshot columns a caller may filter by, and the axes its breakdown counts."""

_STATE_SEMI_FIELDS: tuple[str, ...] = ("animal", "session", "scope", "job_name", "specifier", "status", "job_id")
"""The job fields a semi-detail listing carries. ``job_id`` is included because it is the key a caller resets a job
by."""

_STATE_DETAIL_FIELDS: tuple[str, ...] = ("executor_id", "error_message", "started_at", "completed_at")
"""The job fields detail adds, which are the provenance and timing a caller reads when examining one job closely."""


@mcp.tool()
def define_forging_dataset_tool(
    project_path: str,
    dataset_name: str,
    session_names: list[str],
    recreate_animals: list[str] | None = None,
    *,
    force_recreate: bool = False,
) -> dict[str, Any]:
    """Creates or extends a forged dataset hierarchy and materializes the per-animal configurations it resolves.

    Every forging job runs against a hierarchy this tool established, so a dataset is defined here before its jobs
    are prepared, and preparing a dataset this tool has not built reports an error. A per-animal configuration is
    written only for the animals whose acquisition system resolves a multi-recording configuration, which is the
    case for sessions carrying two-photon imaging data. Each configuration names the thread count the batch layer
    budgets a cross-recording job, since those stages read that count from the file rather than from a call.

    Provided sessions the dataset does not hold are appended, so a dataset grows by naming the sessions to add. An
    animal already in the dataset is frozen, because widening its session set invalidates the outputs already forged
    for the sessions it keeps. Name that animal in ``recreate_animals`` to rebuild it from the provided sessions
    while every other animal keeps its data, which also returns that animal's tracked jobs to the scheduled state.

    Args:
        project_path: The path to the project's root directory holding the animal and session data directories. The
            dataset hierarchy is created under this directory.
        dataset_name: The unique name of the dataset to create or extend.
        session_names: The session names the dataset must contain.
        recreate_animals: The identifiers of animals already in the dataset to rebuild from the sessions provided
            for them. Omit to leave every existing animal frozen.
        force_recreate: Determines whether to delete the whole existing dataset hierarchy and rebuild it from the
            provided session list. Mutually exclusive with ``recreate_animals``.

    Returns:
        A response dict with the ``dataset_name``, the ``dataset_path`` the hierarchy was built at, the
        ``tracker_path`` its jobs record on, the ``session_count`` and ``animal_count`` the dataset now holds, and
        the ``animals`` it covers. Returns an error when the resolution policy rejects the request.
    """
    try:
        dataset = define_forging_dataset(
            name=dataset_name,
            session_names=tuple(session_names),
            project_root=Path(project_path),
            workers=resolve_job_cores(job_name=MULTIDAY_EXTRACTION_JOB_NAME),
            force_recreate=force_recreate,
            recreate_animals=tuple(recreate_animals or ()),
        )
    except Exception as exception:
        return error_response(message=str(exception))

    return ok_response(
        dataset_name=dataset.name,
        dataset_path=str(dataset.dataset_data_path.parent),
        tracker_path=str(forging_tracker_path(dataset=dataset)),
        session_count=len(dataset.sessions),
        animal_count=len(dataset.animals),
        animals=[animal.animal for animal in dataset.animals],
    )


@mcp.tool()
def generate_dataset_state_tool(dataset_paths: list[str]) -> dict[str, Any]:
    """Snapshots each named dataset's forging job state into a shippable feather file at the dataset root.

    Reads the dataset's forging tracker and rewrites the snapshot, so it is cheap enough to run before deciding
    what to forge and again once a run finishes. A dataset that cannot be read is reported in its own entry and does
    not abort the others.

    Args:
        dataset_paths: The dataset root directories to snapshot.

    Returns:
        A response dict with ``total_units``, ``total_jobs``, and a ``units`` list carrying each dataset's
        ``dataset_path``, ``dataset_name``, ``state_path``, ``job_count``, and a ``summary`` counting its jobs by
        status, or an ``error``.
    """
    units: list[dict[str, Any]] = []
    total_jobs = 0
    for dataset_path in dataset_paths:
        try:
            dataset = DatasetData.load(dataset_path=Path(dataset_path))
            state_path = generate_dataset_state(dataset=dataset)
        except Exception as exception:
            units.append({"dataset_path": dataset_path, "error": str(exception), "job_count": 0})
            continue

        frame = pl.read_ipc(source=state_path, memory_map=True)
        total_jobs += frame.height
        units.append(
            {
                "dataset_path": dataset_path,
                "dataset_name": dataset.name,
                "state_path": str(state_path),
                "job_count": frame.height,
                "summary": _status_counts(frame=frame),
            }
        )

    return ok_response(total_units=len(units), total_jobs=total_jobs, units=units)


@mcp.tool()
def read_dataset_state_tool(
    dataset_path: str,
    scope: str | None = None,
    animal: str | None = None,
    session: str | None = None,
    job_names: list[str] | None = None,
    status: str | None = None,
    limit: int | None = None,
    start_row: int = 0,
    *,
    include_items: bool = False,
    detailed: bool = False,
) -> dict[str, Any]:
    """Reads a dataset's forging job state out of its stored snapshot, in three widening stages.

    A bare call reports the totals and a ``breakdown`` naming every scope, animal, job type, and status the dataset
    holds, which is how you find what needs attention without listing anything. Naming a filter adds a page of jobs
    carrying their subject and status. Opting into detail adds the executor, the timestamps, and any recorded error.

    Reads the stored table rather than the tracker, so a snapshot pulled from a remote host answers without any access
    to the data it describes. The totals and the breakdown span every job regardless of the filters.

    Args:
        dataset_path: The absolute path to the dataset's root directory.
        scope: Restricts the listing to jobs of one scope, either ``animal`` or ``session``.
        animal: Restricts the listing to one animal's jobs.
        session: Restricts the listing to one session's jobs.
        job_names: Restricts the listing to these forging job type names.
        status: Restricts the listing to one tracker status, such as ``FAILED``.
        limit: The jobs to list. Defaults to 200, or to 50 when detail is requested. A value at or below zero lists
            every match.
        start_row: The match index to begin the listing at. Follow ``next_start_row`` to walk a long result.
        include_items: Determines whether to list jobs when no filter is named.
        detailed: Determines whether the listed jobs carry the executor, timestamps, and error text.

    Returns:
        A response dict with ``dataset_path``, ``state_path``, a ``summary`` counting every job by status, and a
        ``breakdown`` per axis. Carries a ``jobs`` list with ``rows``, ``matched_rows``, ``start_row``, and
        ``next_start_row`` whenever a filter is named or the listing is requested. Returns an error when no snapshot
        exists or a filter names a value the snapshot does not hold.
    """
    try:
        dataset = DatasetData.load(dataset_path=Path(dataset_path))
    except Exception as exception:
        return error_response(message=f"Unable to load the dataset at '{dataset_path}'. {exception}")

    state_path = dataset_state_path(dataset=dataset)
    if not state_path.is_file():
        return error_response(
            message=(
                f"No dataset state snapshot exists at '{state_path}'. Run generate_dataset_state_tool before reading "
                f"it."
            )
        )

    frame = pl.read_ipc(source=state_path, memory_map=True)
    response = ok_response(
        dataset_path=dataset_path,
        state_path=str(state_path),
        summary=_status_counts(frame=frame),
        breakdown={axis: count_values(values=frame[axis].to_list()) for axis in _STATE_AXES if axis in frame.columns},
    )

    singles: dict[str, str | None] = {"scope": scope, "animal": animal, "session": session, "status": status}
    if not any(value is not None for value in (*singles.values(), job_names)) and not include_items:
        return response

    matched = frame
    for column, value in singles.items():
        if value is None:
            continue
        available = sorted({str(entry) for entry in frame[column].to_list() if entry is not None})
        if value not in available:
            return error_response(message=f"No job has '{column}' of '{value}'. Available: {available}.")
        matched = matched.filter(pl.col(column) == value)
    if job_names is not None:
        available = sorted({str(entry) for entry in frame["job_name"].to_list() if entry is not None})
        unknown = sorted({name for name in job_names if name not in available})
        if unknown:
            return error_response(message=f"No job has 'job_name' in {unknown}. Available: {available}.")
        matched = matched.filter(pl.col("job_name").is_in(job_names))

    fields = (*_STATE_SEMI_FIELDS, *_STATE_DETAIL_FIELDS) if detailed else _STATE_SEMI_FIELDS
    window = resolve_page(
        total=matched.height, limit=resolve_detail_limit(limit=limit, detailed=detailed), start_row=start_row
    )
    page = matched.slice(window.start, window.length)
    response["jobs"] = [project_item(item=item, fields=fields) for item in page.to_dicts()]
    response.update(page_fields(window=window, total=matched.height, listed=page.height))
    return response


@mcp.tool()
def list_project_datasets_tool(
    project_path: str,
    session: str | None = None,
    animal: str | None = None,
    limit: int | None = None,
    start_row: int = 0,
    *,
    detailed: bool = False,
) -> dict[str, Any]:
    """Lists the forged datasets stored under a project, and which of them hold a given session or animal.

    This is the tool that answers what datasets a project holds and whether a session has been forged into any of
    them, which no other tool reports. The manifest is session-rowed and says nothing about datasets, because a
    dataset's own artifacts own that fact.

    A project holds a handful of datasets rather than thousands, so the listing is the summary and appears in every
    response. Naming a session or an animal narrows it to the datasets holding them. Opting into detail reads each
    listed dataset's state snapshot and adds its job counts by status, which is the expensive half since it opens one
    stored table per dataset.

    Args:
        project_path: The absolute path to the project's root data directory.
        session: The session name to restrict the listing to the datasets holding it.
        animal: The animal identifier to restrict the listing to the datasets holding it.
        limit: The datasets to list. Defaults to 200, or to 50 when detail is requested. A value at or below zero lists
            every match.
        start_row: The match index to begin the listing at. Follow ``next_start_row`` to walk a long result.
        detailed: Determines whether each listed dataset reports its animals and its job counts by status, read from
            its state snapshot.

    Returns:
        A response dict with ``project_path``, ``total_datasets``, ``total_memberships`` summed across datasets, a
        ``breakdown`` per session type and acquisition system, and a ``datasets`` list alongside ``rows``,
        ``matched_rows``, ``start_row``, and ``next_start_row``. Each entry carries the dataset's ``name``,
        ``dataset_path``, ``session_type``, ``acquisition_system``, ``session_count``, and ``animal_count``. Returns an
        error when the project directory cannot be read.
    """
    try:
        datasets = discover_project_datasets(project_root=Path(project_path))
    except Exception as exception:
        return error_response(message=f"Unable to discover the datasets under '{project_path}'. {exception}")

    # Keeps each dataset alongside its rendered fields rather than inside them, so the response never carries the
    # loaded object and filtering reads the dataset directly.
    entries: list[tuple[DatasetData, dict[str, Any]]] = [
        (
            dataset,
            {
                "name": dataset.name,
                "dataset_path": str(dataset.dataset_data_path.parent),
                "session_type": str(dataset.session_type),
                "acquisition_system": str(dataset.acquisition_system),
                "session_count": len(dataset.sessions),
                "animal_count": len(dataset.animals),
            },
        )
        for dataset in datasets
    ]

    response = ok_response(
        project_path=project_path,
        total_datasets=len(entries),
        total_memberships=sum(len(dataset.sessions) for dataset, _ in entries),
        breakdown={
            "session_type": count_values(values=[fields["session_type"] for _, fields in entries]),
            "acquisition_system": count_values(values=[fields["acquisition_system"] for _, fields in entries]),
        },
    )

    matched = entries
    if session is not None:
        matched = [pair for pair in matched if any(entry.session == session for entry in pair[0].sessions)]
    if animal is not None:
        matched = [pair for pair in matched if any(entry.animal == animal for entry in pair[0].sessions)]

    window = resolve_page(
        total=len(matched), limit=resolve_detail_limit(limit=limit, detailed=detailed), start_row=start_row
    )
    listed: list[dict[str, Any]] = []
    for dataset, fields in matched[window.start : window.stop]:
        rendered = project_item(item=fields, fields=_DATASET_SEMI_FIELDS)
        if detailed:
            rendered.update(_dataset_state_summary(dataset=dataset))
            rendered["animals"] = sorted({entry.animal for entry in dataset.sessions})
        listed.append(rendered)

    response["datasets"] = listed
    response.update(page_fields(window=window, total=len(matched), listed=len(listed)))
    return response


def _dataset_state_summary(dataset: DatasetData) -> dict[str, Any]:
    """Reads one dataset's forging job counts from its stored state snapshot.

    Notes:
        Reports the snapshot's absence rather than falling back to the dataset's tracker, because every job-level fact
        is read from the artifact that owns it. A dataset whose snapshot was never generated is told to generate one.

    Args:
        dataset: The dataset whose state snapshot to read.

    Returns:
        A dictionary carrying whether the snapshot exists and, when it does, its job counts by status.
    """
    state_path = dataset_state_path(dataset=dataset)
    if not state_path.is_file():
        return {"state_exists": False}
    return {"state_exists": True, "jobs": _status_counts(frame=pl.read_ipc(source=state_path, memory_map=True))}


def _status_counts(frame: pl.DataFrame) -> dict[str, int]:
    """Counts a dataset state snapshot's jobs by tracker status.

    Args:
        frame: The whole state snapshot.

    Returns:
        A dictionary mapping each status the snapshot holds to how many jobs report it, alongside the total.
    """
    counts = {str(status): int(count) for status, count in frame["status"].value_counts().iter_rows()}
    return {"total": frame.height, **dict(sorted(counts.items()))}
