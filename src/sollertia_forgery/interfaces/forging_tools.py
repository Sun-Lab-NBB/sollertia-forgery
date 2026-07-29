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
)
from .mcp_instance import mcp
from ..orchestration import resolve_job_cores

_DEFAULT_ROW_LIMIT: int = 200
"""The rows a state query returns when the caller names no limit. A dataset carries one row per forging job across
every animal and session it holds."""


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
        return _error_response(message=str(exception))

    return _ok_response(
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

    return _ok_response(total_units=len(units), total_jobs=total_jobs, units=units)


@mcp.tool()
def read_dataset_state_tool(
    dataset_path: str,
    scope: str | None = None,
    status_filter: str | None = None,
    limit: int = _DEFAULT_ROW_LIMIT,
) -> dict[str, Any]:
    """Reads a dataset's forging job state out of its stored snapshot.

    Reads the stored table rather than the tracker, so a snapshot pulled from a remote host answers without any access
    to the data it describes. Filters narrow the rows listed, while the summary always spans every row the snapshot
    holds, so narrowing what is listed never distorts what is reported.

    Args:
        dataset_path: The absolute path to the dataset's root directory.
        scope: Restricts the listed rows to jobs of one scope, either ``animal`` or ``session``.
        status_filter: Restricts the listed rows to one tracker status, such as ``FAILED`` or ``SUCCEEDED``.
        limit: The maximum number of rows to list. Values below 1 list every matching row.

    Returns:
        A response dict with ``dataset_path``, ``state_path``, a ``summary`` counting every job by status, the ``rows``
        listed, ``matched_rows`` before the cap, a ``truncated`` flag, and the ``jobs`` list. Returns an error when no
        snapshot exists or a filter names a value the snapshot does not hold.
    """
    try:
        dataset = DatasetData.load(dataset_path=Path(dataset_path))
    except Exception as exception:
        return _error_response(message=f"Unable to load the dataset at '{dataset_path}'. {exception}")

    state_path = dataset_state_path(dataset=dataset)
    if not state_path.is_file():
        return _error_response(
            message=(
                f"No dataset state snapshot exists at '{state_path}'. Run generate_dataset_state_tool before reading "
                f"it."
            )
        )

    frame = pl.read_ipc(source=state_path, memory_map=True)
    matched = frame

    for column, value in (("scope", scope), ("status", status_filter)):
        if value is None:
            continue
        available = sorted(set(frame[column].to_list()))
        if value not in available:
            return _error_response(message=f"Unknown {column} '{value}'. Available: {available}.")
        matched = matched.filter(pl.col(column) == value)

    capped = matched if limit < 1 else matched.head(limit)
    return _ok_response(
        dataset_path=dataset_path,
        state_path=str(state_path),
        summary=_status_counts(frame=frame),
        rows=capped.height,
        matched_rows=matched.height,
        truncated=capped.height < matched.height,
        jobs=capped.to_dicts(),
    )


def _status_counts(frame: pl.DataFrame) -> dict[str, int]:
    """Counts a dataset state snapshot's jobs by tracker status.

    Args:
        frame: The whole state snapshot.

    Returns:
        A dictionary mapping each status the snapshot holds to how many jobs report it, alongside the total.
    """
    counts = {str(status): int(count) for status, count in frame["status"].value_counts().iter_rows()}
    return {"total": frame.height, **dict(sorted(counts.items()))}


def _ok_response(**payload: Any) -> dict[str, Any]:  # noqa: ANN401
    """Constructs a successful response dict with a ``success`` flag set to True."""
    return {"success": True, **payload}


def _error_response(message: str) -> dict[str, Any]:
    """Constructs a failure response dict with a ``success`` flag set to False and the provided error message."""
    return {"success": False, "error": message}
