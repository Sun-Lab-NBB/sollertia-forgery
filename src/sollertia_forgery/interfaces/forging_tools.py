"""Provides the Model Context Protocol (MCP) tools for defining the dataset hierarchy against which a forging batch runs
and for snapshotting the state of its forging jobs.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

import polars as pl
from sollertia_shared_assets import DatasetData

from ..forging import (
    DATASET_STATE_FILENAME,
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
    reject_unknown,
    frame_breakdown,
    resolve_detail_limit,
)
from .mcp_instance import mcp
from ..orchestration import (
    DATASET_UNIT,
    REMOTE_HOST_LABEL,
    RemoteHost,
    connect_to_server,
)
from .host_resolution import (
    HOST_LABELS,
    resolve_readable_project,
    unsupported_host_message,
)

_DATASET_SEMI_FIELDS: tuple[str, ...] = (
    "name",
    "dataset_path",
    "session_type",
    "acquisition_system",
    "session_count",
    "animal_count",
)
"""The fields a dataset listing carries, which is the dataset's identity and how much it holds. Detail adds the job
counts, since reading them opens one stored table per dataset."""

_STATE_AXES: tuple[str, ...] = ("scope", "animal", "job_name", "status")
"""The snapshot columns by which a caller may filter, and the axes its breakdown counts."""

_STATE_SEMI_FIELDS: tuple[str, ...] = ("animal", "session", "scope", "job_name", "specifier", "status", "job_id")
"""The job fields a semi-detail listing carries. ``job_id`` is included because it is the identifier a reset
targets."""

_STATE_DETAIL_FIELDS: tuple[str, ...] = ("executor_id", "error_message", "started_at", "completed_at")
"""The job fields detail adds, which are the provenance and timing a caller reads when examining one job closely."""


@mcp.tool()
def define_forging_dataset_tool(
    project_path: str,
    dataset_name: str,
    session_names: list[str],
    recreate_animals: list[str] | None = None,
    host: str = "local",
    *,
    force_recreate: bool = False,
) -> dict[str, Any]:
    """Creates or extends a forged dataset hierarchy and materializes the per-animal configurations it resolves.

    Every forging job runs against a hierarchy this tool established, so a dataset is defined here before its jobs
    are prepared, and preparing a dataset this tool has not built reports an error. A per-animal configuration is
    written only for the animals whose acquisition system resolves a multi-recording configuration, which is the
    case for sessions carrying two-photon imaging data. The configuration file carries the recording set, the
    qualified dataset name, and the progress flag.

    Provided sessions that the dataset does not hold are appended, so a dataset grows by naming the sessions to add. An
    animal already in the dataset is frozen, because widening its session set invalidates the outputs already forged for
    the sessions it keeps. Name that animal in ``recreate_animals`` to rebuild it from the provided sessions while every
    other animal keeps its data, which also returns that animal's tracked jobs to the scheduled state.

    Args:
        project_path: The path to the project's root directory holding the animal and session data directories. The
            dataset hierarchy is created under this directory.
        dataset_name: The unique name of the dataset to create or extend.
        session_names: The session names the dataset must contain.
        recreate_animals: The identifiers of animals already in the dataset to rebuild from the sessions provided
            for them. Omit to leave every existing animal frozen.
        host: Where the data sits, either ``local`` for this machine or ``remote`` for the configured compute server.
            A remote definition reports the dataset it built without its shape, since the hierarchy cannot be loaded
            from this machine.
        force_recreate: Determines whether to delete the whole existing dataset hierarchy and rebuild it from the
            provided session list. Mutually exclusive with ``recreate_animals``.

    Returns:
        A response dict with the ``dataset_name``, the ``dataset_path`` at which the hierarchy was built, the
        ``tracker_path`` recording its jobs, the ``session_count`` and ``animal_count`` the dataset now holds, and the
        ``animals`` it covers. Returns an error when the resolution policy rejects the request.
    """
    if host not in HOST_LABELS:
        return error_response(message=unsupported_host_message(host=host))

    if host == REMOTE_HOST_LABEL:
        try:
            with connect_to_server() as server:
                RemoteHost(server=server).define_dataset(
                    project_root=Path(project_path),
                    dataset_name=dataset_name,
                    session_names=session_names,
                    recreate_animals=recreate_animals or (),
                    force_recreate=force_recreate,
                )
        except Exception as exception:
            return error_response(message=f"Unable to define the remote dataset '{dataset_name}'. {exception}")

        # The hierarchy sits on the server, so nothing here can load it to report its shape. A caller reads what it
        # holds through the dataset state artifact.
        return ok_response(
            dataset_name=dataset_name,
            host=host,
            dataset_path=str(Path(project_path).joinpath(dataset_name)),
            message=(
                "Defined the dataset on the server. Read what it now holds with generate_dataset_state_tool followed "
                "by read_dataset_state_tool, since the hierarchy cannot be loaded from this machine."
            ),
        )

    try:
        dataset = define_forging_dataset(
            name=dataset_name,
            session_names=tuple(session_names),
            project_root=Path(project_path),
            force_recreate=force_recreate,
            recreate_animals=tuple(recreate_animals or ()),
        )
    except Exception as exception:
        return error_response(message=f"Unable to define the local dataset '{dataset_name}'. {exception}")

    return ok_response(
        dataset_name=dataset.name,
        dataset_path=str(dataset.dataset_data_path.parent),
        tracker_path=str(forging_tracker_path(dataset=dataset)),
        session_count=len(dataset.sessions),
        animal_count=len(dataset.animals),
        animals=[animal.animal for animal in dataset.animals],
    )


@mcp.tool()
def generate_dataset_state_tool(dataset_paths: list[str], host: str = "local") -> dict[str, Any]:
    """Snapshots each named dataset's forging job state into a shippable feather file at the dataset root.

    Reads the dataset's forging tracker and rewrites the snapshot, so it is cheap enough to run before deciding
    what to forge and again once a run finishes. A ``local`` snapshot reports a dataset it cannot read in its own entry
    and leaves the others alone, while a ``remote`` snapshot fails the whole call, since one failure anywhere in the
    server-side sequence aborts it.

    Args:
        dataset_paths: The dataset root directories to snapshot, which are paths ON THE SERVER for ``remote``.
        host: Where the data sits, either ``local`` for this machine or ``remote`` for the configured compute server.
            A remote snapshot reads its rows back off the server, since the dataset cannot be loaded from this machine.

    Returns:
        A response dict with ``host``, ``total_units``, ``total_jobs``, and a ``units`` list carrying each dataset's
        ``dataset_path``, ``dataset_name``, ``job_count``, and a ``summary`` counting its jobs by status, or an
        ``error``. A local snapshot also carries each ``state_path``.
    """
    if host not in HOST_LABELS:
        return error_response(message=unsupported_host_message(host=host))
    if host == REMOTE_HOST_LABEL:
        return _generate_remote_dataset_state(dataset_paths=dataset_paths)

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

    return ok_response(host=host, total_units=len(units), total_jobs=total_jobs, units=units)


@mcp.tool()
def read_dataset_state_tool(
    dataset_path: str,
    host: str = "local",
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
        dataset_path: The absolute path to the dataset's root directory, which is a path ON THE SERVER for ``remote``,
            where the project is resolved from the parent directory's name.
        host: Where the project sits, either ``local`` for this machine or ``remote`` for the configured
            compute server. A remote read mirrors the project's artifacts onto this machine and reads the
            mirror, so it reports what the project currently records without regenerating anything.
        scope: Restricts the listing to jobs of one scope, either ``animal`` or ``session``.
        animal: Restricts the listing to one animal's jobs.
        session: Restricts the listing to one session's jobs.
        job_names: Restricts the listing to these forging job type names.
        status: Restricts the listing to one tracker status, such as ``FAILED``.
        limit: The jobs to list. Defaults to 200, or to 50 when detail is requested. A value at or below zero lists
            every match.
        start_row: The match index at which to begin the listing. Follow ``next_start_row`` to walk a long result.
        include_items: Determines whether to list jobs when no filter is named.
        detailed: Determines whether the listed jobs carry the executor, timestamps, and error text.

    Returns:
        A response dict with ``dataset_path``, ``state_path``, a ``summary`` counting every job by status, and a
        ``breakdown`` per axis. Carries a ``jobs`` list with ``rows``, ``matched_rows``, ``start_row``, and
        ``next_start_row`` whenever a filter is named or the listing is requested. Returns an error when no snapshot
        exists or a filter names a value the snapshot does not hold.
    """
    if host not in HOST_LABELS:
        return error_response(message=unsupported_host_message(host=host))
    try:
        # A remote dataset is mirrored through its project, since the mirror reproduces the project directory by name.
        project = resolve_readable_project(project_path=str(Path(dataset_path).parent), host=host)
        dataset = DatasetData.load(dataset_path=project.joinpath(Path(dataset_path).name))
    except Exception as exception:
        return error_response(message=f"Unable to load the {host} dataset at '{dataset_path}'. {exception}")

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
        breakdown=frame_breakdown(frame=frame, axes=_STATE_AXES),
    )

    singles: dict[str, str | None] = {"scope": scope, "animal": animal, "session": session, "status": status}
    if not any(value is not None for value in (*singles.values(), job_names)) and not include_items:
        return response

    matched = frame
    for column, value in singles.items():
        if value is None:
            continue
        rejection = reject_unknown(frame=frame, column=column, values=[value], subject="job")
        if rejection is not None:
            return rejection
        matched = matched.filter(pl.col(column) == value)
    if job_names is not None:
        rejection = reject_unknown(frame=frame, column="job_name", values=job_names, subject="job")
        if rejection is not None:
            return rejection
        matched = matched.filter(pl.col("job_name").is_in(job_names))

    fields = (*_STATE_SEMI_FIELDS, *_STATE_DETAIL_FIELDS) if detailed else _STATE_SEMI_FIELDS
    window = resolve_page(
        total=matched.height, limit=resolve_detail_limit(limit=limit, detailed=detailed), start_row=start_row
    )
    page = matched.slice(offset=window.start, length=window.length)
    response["jobs"] = [project_item(item=item, fields=fields) for item in page.to_dicts()]
    response.update(page_fields(window=window, total=matched.height, listed=page.height))
    return response


@mcp.tool()
def list_project_datasets_tool(
    project_path: str,
    host: str = "local",
    session: str | None = None,
    animal: str | None = None,
    limit: int | None = None,
    start_row: int = 0,
    *,
    detailed: bool = False,
) -> dict[str, Any]:
    """Lists the forged datasets stored under a project, and which of them hold a given session or animal.

    A project holds a handful of datasets rather than thousands, so the listing is the summary and appears in every
    response. Naming a session or an animal narrows it to the datasets holding them. Opting into detail reads each
    listed dataset's state snapshot and adds its job counts by status, which is the expensive half since it opens one
    stored table per dataset.

    Args:
        project_path: The absolute path to the project's root data directory.
        host: Where the project sits, either ``local`` for this machine or ``remote`` for the configured
            compute server. A remote read mirrors the project's artifacts onto this machine and reads the
            mirror, so it reports what the project currently records without regenerating anything.
        session: The session name to restrict the listing to the datasets holding it.
        animal: The animal identifier to restrict the listing to the datasets holding it.
        limit: The datasets to list. Defaults to 200, or to 50 when detail is requested. A value at or below zero lists
            every match.
        start_row: The match index at which to begin the listing. Follow ``next_start_row`` to walk a long result.
        detailed: Determines whether each listed dataset reports its animals and its job counts by status, read from
            its state snapshot.

    Returns:
        A response dict with ``project_path``, ``total_datasets``, ``total_memberships`` summed across datasets, a
        ``breakdown`` per session type and acquisition system, and a ``datasets`` list alongside ``rows``,
        ``matched_rows``, ``start_row``, and ``next_start_row``. Each entry carries the dataset's ``name``,
        ``dataset_path``, ``session_type``, ``acquisition_system``, ``session_count``, and ``animal_count``. A detailed
        entry adds ``state_exists``, its ``jobs`` counts by status when a snapshot exists, and the ``animals`` it
        covers. Returns an error when the project directory cannot be read.
    """
    if host not in HOST_LABELS:
        return error_response(message=unsupported_host_message(host=host))
    try:
        datasets = discover_project_datasets(
            project_root=resolve_readable_project(project_path=project_path, host=host)
        )
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


def _generate_remote_dataset_state(dataset_paths: list[str]) -> dict[str, Any]:
    """Rewrites each named dataset's state artifact on the server and reports what it now records.

    Notes:
        The hierarchy sits on the server, so the rows are read back off it rather than loaded from a dataset this
        machine cannot open.

    Args:
        dataset_paths: The dataset root directories on the server.

    Returns:
        The response dict the calling tool returns.
    """
    units = [Path(path) for path in dataset_paths]
    try:
        with connect_to_server() as server:
            host = RemoteHost(server=server)
            for unit in units:
                host.generate_state(project_root=unit.parent, unit_paths=[unit], unit_kind=DATASET_UNIT)
            rows = {str(unit): host.read_rows(path=unit.joinpath(DATASET_STATE_FILENAME)) for unit in units}
    except Exception as exception:
        return error_response(message=f"Unable to generate the remote dataset state. {exception}")

    reported = [
        {
            "dataset_path": str(unit),
            "dataset_name": unit.name,
            "job_count": len(rows[str(unit)]),
            "summary": count_values(values=[row["status"] for row in rows[str(unit)]]),
        }
        for unit in units
    ]
    return ok_response(
        host=REMOTE_HOST_LABEL,
        total_units=len(reported),
        total_jobs=sum(entry["job_count"] for entry in reported),
        units=reported,
    )
