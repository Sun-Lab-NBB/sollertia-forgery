"""Provides the Model Context Protocol (MCP) tools for authoring and reading the remote compute server configuration,
discovering the data a project holds on that server, and reading the server's scheduler queue and job accounting.
"""

from __future__ import annotations

import uuid
import shlex
from typing import Any
from pathlib import Path

import yaml
from ataraxis_data_structures import direct_write

from ..server import (
    ServerConfiguration,
    discover_project_markers,
    get_server_configuration,
    get_server_configuration_path,
)
from .responses import (
    ok_response,
    page_fields,
    count_values,
    project_item,
    resolve_page,
    bounded_counts,
    error_response,
    resolve_detail_limit,
)
from .mcp_instance import mcp
from ..orchestration import DATASET_UNIT, SESSION_UNIT, connect_to_server

_MASKED_PASSWORD: str = "<masked>"  # noqa: S105 - literal masking placeholder, not a real password.
"""The placeholder every tool substitutes for the stored password when it reports a configuration back. The write tool
refuses to persist it, since a caller editing what a read returned would otherwise replace the password with this
literal."""

_DISCOVERY_UNIT_KINDS: frozenset[str] = frozenset({SESSION_UNIT, DATASET_UNIT})
"""The processing unit kinds a project's marker files resolve, which are the values by which a discovery listing
filters."""

_DISCOVERY_FIELDS: tuple[str, ...] = ("unit_kind", "animal", "session", "dataset", "unit_path")
"""The fields a discovered unit carries, which name the unit and give the server-side absolute path every remote tool
takes as its argument. A dataset entry carries no animal or session, and a session entry carries no dataset."""

_ACCOUNTING_VIEW: str = "accounting"
"""The scheduler view reporting what every allocation of a finished or running job actually consumed."""

_QUEUE_VIEW: str = "queue"
"""The scheduler view reporting the allocations the scheduler currently holds queued or running."""

_SCHEDULER_VIEWS: frozenset[str] = frozenset({_ACCOUNTING_VIEW, _QUEUE_VIEW})
"""The views a scheduler read may request."""

_ALL_USERS: str = "all"
"""The user name that lifts the per-user restriction and covers every account on the server."""

_ACCOUNTING_COLUMNS: dict[str, str] = {
    "JobID": "job_id",
    "JobName": "job_name",
    "State": "state",
    "Elapsed": "elapsed",
    "NCPUS": "cores",
    "AveCPU": "peak_step_cpu_time",
    "ReqMem": "requested_memory",
    "MaxRSS": "maximum_resident_memory",
    "AveRSS": "peak_step_resident_memory",
    "MaxVMSize": "maximum_virtual_memory",
}
"""Maps each accounting column the query requests onto the key under which a response reports it. The query requests
the columns in this order, and a parse reads them by the response's own header row."""

_QUEUE_FORMAT: str = "%i|%P|%j|%u|%T|%D|%C|%m|%M|%l|%L"
"""The queue format the query requests, which separates its columns by the same character the accounting query uses,
so both views split a line on one separator."""

_QUEUE_FIELDS: tuple[str, ...] = (
    "job_id",
    "partition",
    "job_name",
    "user",
    "state",
    "nodes",
    "cores",
    "requested_memory",
    "elapsed",
    "time_limit",
    "time_left",
)
"""The keys under which a response reports the queue columns, in the order ``_QUEUE_FORMAT`` requests them. The queue
names its own columns after the format specifiers rather than after these keys, so a parse reads them by position."""

_ACCOUNTING_AXES: tuple[str, ...] = ("state", "job_name")
"""The accounting fields by which a caller may filter, which are the axes an accounting breakdown counts."""

_QUEUE_AXES: tuple[str, ...] = ("state", "partition", "user", "job_name")
"""The queue fields by which a caller may filter, which are the axes a queue breakdown counts."""

_ACCOUNTING_SEMI_FIELDS: tuple[str, ...] = ("job_id", "job_name", "state", "elapsed", "cores")
"""The accounting fields a semi-detail listing carries, which are the allocation's identity, its outcome, and how long
it ran."""

_ACCOUNTING_DETAIL_FIELDS: tuple[str, ...] = (
    "peak_step_cpu_time",
    "requested_memory",
    "maximum_resident_memory",
    "peak_step_resident_memory",
    "maximum_virtual_memory",
)
"""The accounting fields detail adds, which are what the allocation requested against what it occupied."""

_QUEUE_SEMI_FIELDS: tuple[str, ...] = ("job_id", "partition", "job_name", "user", "state", "cores")
"""The queue fields a semi-detail listing carries, which are the allocation's identity, its owner, and its state."""

_QUEUE_DETAIL_FIELDS: tuple[str, ...] = ("nodes", "requested_memory", "elapsed", "time_limit", "time_left")
"""The queue fields detail adds, which are the allocation's shape and how much of its walltime remains."""

_FIELD_SEPARATOR: str = "|"
"""The character separating the columns of a scheduler response."""

_STEP_SEPARATOR: str = "."
"""The character separating a job step's identifier from the identifier of the job that holds it."""

_EXTERN_STEP_NAME: str = ".extern"
"""The step the scheduler records for every job's external shell. It carries no accounting figure, so a merge skips
it."""

_MINIMUM_TABLE_LINES: int = 2
"""The smallest line count a scheduler response with data carries, being the header line plus one record."""

_MEASURED_SIZE_FIELDS: frozenset[str] = frozenset(
    {"maximum_resident_memory", "peak_step_resident_memory", "maximum_virtual_memory"}
)
"""The accounting fields the scheduler measures per step and reports as a memory size. A job takes the largest figure
any of its steps recorded, so each key names the widest step rather than a figure folded across them."""

_MEASURED_DURATION_FIELDS: frozenset[str] = frozenset({"peak_step_cpu_time"})
"""The accounting fields the scheduler measures per step and reports as a duration, folded the same way the size fields
are."""

_SIZE_SUFFIX_SCALES: dict[str, float] = {"K": 1024.0, "M": 1024.0**2, "G": 1024.0**3, "T": 1024.0**4, "P": 1024.0**5}
"""Maps each suffix the scheduler appends to a memory figure onto the byte count that suffix represents. A figure
carrying no suffix is already in bytes."""

_SECONDS_PER_DAY: float = 86_400.0
"""The seconds in the day count that leads a scheduler duration."""

_SECONDS_PER_MINUTE: float = 60.0
"""The scale by which each colon-separated field of a scheduler duration exceeds the field to its right."""

_DAY_SEPARATOR: str = "-"
"""The character separating the day count of a scheduler duration from the clock time that follows it."""

_TIME_SEPARATOR: str = ":"
"""The character separating the hour, minute, and second fields of a scheduler duration."""


@mcp.tool()
def read_server_configuration_tool() -> dict[str, Any]:
    """Loads the ServerConfiguration from the working directory with the password masked.

    Returns:
        A response dict with ``data`` containing the server configuration payload. The password field is
        replaced with the literal string ``"<masked>"`` for security.
    """
    try:
        instance = get_server_configuration()
    except (OSError, ValueError) as exception:
        return error_response(message=f"Unable to read the server configuration. {exception}")
    serialized = _render_configuration(instance=instance)
    serialized["password"] = _MASKED_PASSWORD
    return ok_response(data=serialized)


@mcp.tool()
def write_server_configuration_tool(
    configuration_payload: dict[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Creates or replaces the ServerConfiguration YAML in the working directory.

    Args:
        configuration_payload: The complete ServerConfiguration payload. Supply ``username``, ``password``, ``host``,
            ``root``, and ``environment``, since an omitted field is persisted as an empty string rather than rejected.
            The ``password`` must be the account's real password, since the placeholder that the read tool reports in
            its place is refused rather than stored.
        overwrite: Determines whether to overwrite an existing server configuration file.

    Returns:
        A response dict with ``file_path`` and ``data`` containing the validated payload with the password masked.
        Returns an error when the payload carries the read tool's password placeholder.
    """
    if configuration_payload.get("password") == _MASKED_PASSWORD:
        return error_response(
            message=(
                f"Unable to write the server configuration. The 'password' field carries the '{_MASKED_PASSWORD}' "
                f"placeholder that the read tool reports in place of the stored password, so writing it would replace "
                f"the real password with that literal and leave every later connection unable to authenticate. "
                f"Supply the account's actual password."
            )
        )

    try:
        file_path = get_server_configuration_path()
    except FileNotFoundError as exception:
        return error_response(message=f"Unable to resolve the server configuration path. {exception}")

    if file_path.exists() and not overwrite:
        return error_response(
            message=(
                f"Unable to write the server configuration. A file already exists at '{file_path}'. Pass "
                f"overwrite=True to replace it."
            )
        )

    # Keeps the temporary file ending in .yaml because YamlConfig.from_yaml rejects non-.yaml paths. The scratch file
    # exists to give from_yaml a path to read rather than to publish anything, so it is written with direct_write,
    # which also creates the configuration directory. The atomic_write helper is wrong here, because nothing ever reads
    # this path and its flush and rename would only pay to publish a file the next statement deletes. The durable half
    # of the operation is instance.to_yaml() below, which writes through atomic_write itself.
    temporary_path = file_path.with_name(f".{file_path.stem}.{uuid.uuid4().hex[:8]}.tmp.yaml")

    try:
        with direct_write(file_path=temporary_path) as temporary_file:
            yaml.safe_dump(data=configuration_payload, stream=temporary_file, sort_keys=False)
        instance = ServerConfiguration.from_yaml(file_path=temporary_path)
    except Exception as exception:
        return error_response(message=f"Unable to validate the supplied server configuration payload. {exception}")
    finally:
        temporary_path.unlink(missing_ok=True)

    try:
        instance.to_yaml(file_path=file_path)
    except Exception as exception:
        return error_response(message=f"Unable to write the server configuration to '{file_path}'. {exception}")

    serialized = _render_configuration(instance=instance)
    serialized["password"] = _MASKED_PASSWORD
    return ok_response(file_path=str(file_path), data=serialized)


@mcp.tool()
def discover_remote_project_tool(
    project: str,
    unit_kind: str | None = None,
    animals: list[str] | None = None,
    sessions: list[str] | None = None,
    datasets: list[str] | None = None,
    limit: int | None = None,
    start_row: int = 0,
    *,
    include_items: bool = False,
    include_sessions: bool = True,
) -> dict[str, Any]:
    """Enumerates the sessions and forged datasets a project holds on the compute server, in two widening stages.

    Every tool that names ``remote`` takes paths as the server itself resolves them, and this is where those paths
    originate. A bare call reports how much the project holds alongside a ``breakdown`` naming every animal and
    dataset. An axis holding more distinct values than the shared cap reports how many it holds in place of its counts,
    and filtering on that axis reaches the units themselves. Naming a filter adds a page of units, each carrying the
    absolute server-side path a remote tool takes as its argument.

    The whole tree is read in one server-side search, so the cost is one round trip rather than one per directory.

    Args:
        project: The project to discover. Only the final component of the value names the project, which is resolved
            under the server's configured data root.
        unit_kind: Restricts the listing to one unit kind, either ``session`` or ``dataset``.
        animals: Restricts the listing to these animals' sessions.
        sessions: Restricts the listing to these session names.
        datasets: Restricts the listing to these forged datasets.
        limit: The units to list. Defaults to 200. A value at or below zero lists every match, which is how a caller
            reading under a tight filter takes the whole result at once.
        start_row: The match index at which to begin the listing. Follow ``next_start_row`` to walk a long result.
        include_items: Determines whether to list units when no filter is named.
        include_sessions: Determines whether the search covers the project's acquired sessions alongside its datasets.
            Leave True unless the datasets alone are wanted, since the shallower search reads fewer directories and
            therefore succeeds on a project whose session directories another account owns.

    Returns:
        A response dict with ``project``, the ``project_path`` to which the server resolves it, ``total_sessions``,
        ``total_datasets``, and a ``breakdown`` per unit kind, animal, and dataset. Carries a ``units`` list with
        ``rows``, ``matched_rows``, ``start_row``, and ``next_start_row`` whenever a filter is named or the listing is
        requested, where each entry gives the unit's ``unit_kind``, ``unit_path``, and either its ``animal`` and
        ``session`` or its ``dataset``. Returns an error when the unit kind is unknown, when the project value names no
        final component, when the server holds no such project, or when the search reached only part of its tree.
    """
    if unit_kind is not None and unit_kind not in _DISCOVERY_UNIT_KINDS:
        return error_response(message=f"Unknown unit kind '{unit_kind}'. Available: {sorted(_DISCOVERY_UNIT_KINDS)}.")

    project_name = Path(project).name
    if not project_name:
        return error_response(
            message=(
                f"Unable to discover the '{project}' project on the compute server. Only the final component of the "
                f"value names the project, and that component is empty, so the value resolves to the data root itself."
            )
        )

    try:
        with connect_to_server() as server:
            project_path = server.root.joinpath(project_name)
            if not server.is_directory(remote_path=project_path):
                return error_response(
                    message=(
                        f"Unable to discover the '{project}' project on the compute server. The server holds no "
                        f"directory at '{project_path}', which is where its configured data root resolves that name."
                    )
                )
            markers = discover_project_markers(
                project_path=project_path, server=server, include_sessions=include_sessions
            )
    except Exception as exception:
        return error_response(message=f"Unable to discover the '{project}' project on the compute server. {exception}")

    entries: list[dict[str, str]] = [
        {"unit_kind": DATASET_UNIT, "dataset": dataset.name, "unit_path": str(dataset)} for dataset in markers.datasets
    ]
    entries.extend(
        {
            "unit_kind": SESSION_UNIT,
            "animal": session.animal,
            "session": session.session,
            "unit_path": str(project_path.joinpath(session.animal, session.session)),
        }
        for session in markers.sessions
    )

    response = ok_response(
        project=project_path.name,
        project_path=str(project_path),
        total_sessions=len(markers.sessions),
        total_datasets=len(markers.datasets),
        breakdown={
            "unit_kind": count_values(values=[entry["unit_kind"] for entry in entries]),
            "animal": bounded_counts(values=[entry["animal"] for entry in entries if "animal" in entry]),
            "dataset": bounded_counts(values=[entry["dataset"] for entry in entries if "dataset" in entry]),
        },
    )

    selectors: dict[str, list[str] | None] = {
        "unit_kind": [unit_kind] if unit_kind is not None else None,
        "animal": animals,
        "session": sessions,
        "dataset": datasets,
    }
    if not any(values is not None for values in selectors.values()) and not include_items:
        return response

    matched = [
        entry
        for entry in entries
        if all(values is None or entry.get(field) in values for field, values in selectors.items())
    ]
    window = resolve_page(
        total=len(matched), limit=resolve_detail_limit(limit=limit, detailed=False), start_row=start_row
    )
    page = matched[window.start : window.stop]
    response["units"] = [project_item(item=entry, fields=_DISCOVERY_FIELDS) for entry in page]
    response.update(page_fields(window=window, total=len(matched), listed=len(page)))
    return response


@mcp.tool()
def read_scheduler_jobs_tool(
    view: str = _ACCOUNTING_VIEW,
    user: str | None = None,
    job_ids: list[str] | None = None,
    job_names: list[str] | None = None,
    states: list[str] | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    limit: int | None = None,
    start_row: int = 0,
    *,
    include_items: bool = False,
    detailed: bool = False,
) -> dict[str, Any]:
    """Reads the compute server's own record of its allocations, in three widening stages.

    The ``accounting`` view covers every allocation for which the scheduler still holds a record, including the ones
    that have already finished, and its detail carries what each one truly occupied. The ``queue`` view covers the
    allocations the scheduler currently holds, and its detail carries their shape and their remaining walltime.

    A bare call reports the totals alongside a ``breakdown`` naming every state, job type, and, for the queue, every
    partition and user. An axis holding more distinct values than the shared cap reports how many it holds in place of
    its counts, and filtering on that axis reaches the allocations themselves. Naming a filter adds a page of
    allocations, and opting into detail adds each one's resources.

    This asks the scheduler rather than this machine's submission ledger, so it answers for an allocation another
    machine submitted and for one that has already settled.

    Args:
        view: The record to read, either ``accounting`` for what allocations consumed or ``queue`` for what the
            scheduler currently holds.
        user: The account whose allocations to read. Defaults to the account the server configuration authenticates
            with. Pass ``all`` to cover every account.
        job_ids: The scheduler allocation identifiers to read, to which the query is then scoped. Naming any identifier
            bypasses the account and date restrictions, so a caller reads an allocation another account submitted.
        job_names: Restricts the listing to these scheduler job names.
        states: Restricts the listing to these scheduler states, such as ``COMPLETED``, ``FAILED``, or ``RUNNING``.
        start_time: The earliest start date an accounting record may carry, as ``YYYY-MM-DD`` or
            ``YYYY-MM-DD HH:MM:SS``.
        end_time: The latest end date an accounting record may carry, in the same formats.
        limit: The allocations to list. Defaults to 200, or to 50 when detail is requested. A value at or below zero
            lists every match.
        start_row: The match index at which to begin the listing. Follow ``next_start_row`` to walk a long result.
        include_items: Determines whether to list allocations when no filter is named.
        detailed: Determines whether the listed allocations report their resources.

    Returns:
        A response dict with the ``view`` read, the ``user`` it covered, ``total_jobs``, and a ``breakdown`` per axis.
        Carries a ``jobs`` list with ``rows``, ``matched_rows``, ``start_row``, and ``next_start_row`` whenever a
        filter is named or the listing is requested. An accounting entry gives ``job_id``, ``job_name``, ``state``,
        ``elapsed``, and ``cores``, and its detail adds ``peak_step_cpu_time``, ``requested_memory``,
        ``maximum_resident_memory``, ``peak_step_resident_memory``, and ``maximum_virtual_memory``. A queue entry gives
        ``job_id``, ``partition``, ``job_name``, ``user``, ``state``, and ``cores``, and its detail adds ``nodes``,
        ``requested_memory``, ``elapsed``, ``time_limit``, and ``time_left``. Returns an error when the view is
        unknown, when the server cannot be reached, when the scheduler rejects the query, or when a filter names a
        value the scheduler's answer does not hold.
    """
    if view not in _SCHEDULER_VIEWS:
        return error_response(message=f"Unknown scheduler view '{view}'. Available: {sorted(_SCHEDULER_VIEWS)}.")

    try:
        with connect_to_server() as server:
            account = server.user if user is None else user
            command = (
                _accounting_command(user=account, job_ids=job_ids, start_time=start_time, end_time=end_time)
                if view == _ACCOUNTING_VIEW
                else _queue_command(user=account, job_ids=job_ids)
            )
            result = server.execute_command(command=command)
    except Exception as exception:
        return error_response(message=f"Unable to reach the compute server's scheduler. {exception}")

    if result.return_code != 0:
        return error_response(
            message=(
                f"Unable to read the scheduler's {view} records. The command '{command}' exited with the status "
                f"{result.return_code} and reported: {result.stderr.strip()}."
            )
        )

    rows = (
        _parse_accounting_rows(output=result.stdout)
        if view == _ACCOUNTING_VIEW
        else _parse_queue_rows(output=result.stdout)
    )
    axes = _ACCOUNTING_AXES if view == _ACCOUNTING_VIEW else _QUEUE_AXES
    response = ok_response(
        view=view,
        user=account,
        total_jobs=len(rows),
        breakdown={axis: bounded_counts(values=[row.get(axis, "") for row in rows]) for axis in axes},
    )

    selectors: dict[str, list[str] | None] = {"job_name": job_names, "state": states}
    if not any(values is not None for values in selectors.values()) and job_ids is None and not include_items:
        return response

    for field_name, values in selectors.items():
        if values is None:
            continue
        rejection = _reject_unmatched(
            field=field_name, values=values, available={row.get(field_name, "") for row in rows}
        )
        if rejection is not None:
            return rejection

    matched = [
        row for row in rows if all(values is None or row.get(field) in values for field, values in selectors.items())
    ]
    if view == _ACCOUNTING_VIEW:
        fields = (*_ACCOUNTING_SEMI_FIELDS, *_ACCOUNTING_DETAIL_FIELDS) if detailed else _ACCOUNTING_SEMI_FIELDS
    else:
        fields = (*_QUEUE_SEMI_FIELDS, *_QUEUE_DETAIL_FIELDS) if detailed else _QUEUE_SEMI_FIELDS
    window = resolve_page(
        total=len(matched), limit=resolve_detail_limit(limit=limit, detailed=detailed), start_row=start_row
    )
    page = matched[window.start : window.stop]
    response["jobs"] = [project_item(item=row, fields=fields) for row in page]
    response.update(page_fields(window=window, total=len(matched), listed=len(page)))
    return response


@mcp.tool()
def pull_remote_path_tool(remote_path: str, destination: str) -> dict[str, Any]:
    """Copies a file or directory off the compute server onto this machine.

    The server holds every artifact a remote run produces, and the project-state readers mirror only the manifest, the
    job table, and the plan. Carries anything else back, which covers a session's processed data, one feather, and the
    standard output and
    error a batch's allocations wrote.

    A directory is copied whole, with its tree beneath it. The copy lands inside the destination directory under the
    remote path's own final component, and the destination is created when it does not exist.

    Args:
        remote_path: The absolute path, on the server, to the file or directory to copy.
        destination: The absolute path to the local directory that receives the copy.

    Returns:
        A response dict with the ``remote_path`` copied, the ``local_path`` at which the copy landed, whether the copy
        ``is_directory``, the ``total_files`` it holds, and the ``total_bytes`` it occupies. Returns an error when the
        server holds nothing at the named path, when a file already stands where the destination directory would be
        created, and when the copy or the read of what it landed fails.
    """
    target = Path(destination)
    source = Path(remote_path)
    try:
        with connect_to_server() as server:
            if not server.exists(remote_path=source):
                return error_response(
                    message=(
                        f"Unable to copy '{remote_path}' off the compute server. The server holds no file or directory "
                        f"at that path."
                    )
                )
            is_directory = server.is_directory(remote_path=source)
            target.mkdir(parents=True, exist_ok=True)
            local_path = target.joinpath(source.name)
            server.pull(local_path=local_path, remote_path=source)
    except Exception as exception:
        return error_response(message=f"Unable to copy '{remote_path}' off the compute server. {exception}")

    copied = sorted(path for path in local_path.rglob("*") if path.is_file()) if is_directory else [local_path]
    return ok_response(
        remote_path=str(source),
        local_path=str(local_path),
        is_directory=is_directory,
        total_files=len(copied),
        total_bytes=sum(path.stat().st_size for path in copied),
    )


def _render_configuration(instance: ServerConfiguration) -> dict[str, Any]:
    """Converts a ServerConfiguration instance into a JSON-friendly dict.

    Args:
        instance: The configuration to render.

    Returns:
        A dictionary carrying the username, password, host, root, and environment the configuration holds.
    """
    return {
        "username": instance.username,
        "password": instance.password,
        "host": instance.host,
        "root": instance.root,
        "environment": instance.environment,
    }


def _accounting_command(user: str, job_ids: list[str] | None, start_time: str | None, end_time: str | None) -> str:
    """Builds the command that reads the scheduler's accounting records.

    Notes:
        Naming an allocation scopes the query to it alone, so the account and the dates are left off. That is what
        lets a caller read an allocation another account submitted.

    Args:
        user: The account whose allocations to read, or ``all`` to cover every account.
        job_ids: The allocations to read, or None to read the ones matching the remaining restrictions.
        start_time: The earliest start date a record may carry.
        end_time: The latest end date a record may carry.

    Returns:
        The shell command to execute on the server.
    """
    arguments = ["sacct", "-o", ",".join(_ACCOUNTING_COLUMNS), "--parsable2", "--units=G"]
    if job_ids:
        arguments.extend(("-j", ",".join(job_ids)))
        return shlex.join(arguments)

    if user.lower() == _ALL_USERS:
        arguments.append("-a")
    else:
        arguments.extend(("-u", user))
    if start_time is not None:
        arguments.append(f"--starttime={start_time}")
    if end_time is not None:
        arguments.append(f"--endtime={end_time}")
    return shlex.join(arguments)


def _queue_command(user: str, job_ids: list[str] | None) -> str:
    """Builds the command that reads the allocations the scheduler currently holds.

    Args:
        user: The account whose allocations to read, or ``all`` to cover every account.
        job_ids: The allocations to read, or None to read the ones the account holds.

    Returns:
        The shell command to execute on the server.
    """
    arguments = ["squeue", "-o", _QUEUE_FORMAT]
    if job_ids:
        arguments.extend(("-j", ",".join(job_ids)))
        return shlex.join(arguments)

    if user.lower() != _ALL_USERS:
        arguments.extend(("-u", user))
    return shlex.join(arguments)


def _parse_accounting_rows(output: str) -> list[dict[str, str]]:
    """Parses the scheduler's accounting response into one entry per allocation.

    Args:
        output: The response the accounting command wrote.

    Returns:
        One entry per allocation, keyed by the response keys onto which the accounting columns map.
    """
    lines = [line for line in output.strip().split("\n") if line.strip()]
    if len(lines) < _MINIMUM_TABLE_LINES:
        return []

    # The response names its own columns in its first line, so the keys follow what the scheduler answered rather than
    # the columns for which the query asked.
    keys = [_ACCOUNTING_COLUMNS.get(column, column) for column in lines[0].split(_FIELD_SEPARATOR)]
    rows = [
        dict(zip(keys, values, strict=True))
        for line in lines[1:]
        if len(values := line.split(_FIELD_SEPARATOR)) == len(keys)
    ]
    return _merge_accounting_rows(rows=rows)


def _merge_accounting_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Merges the step records the scheduler writes for one job into the record of the job that holds them.

    Notes:
        The scheduler records a job's requested resources and its final state on the job itself, while it records the
        memory and processor time the job occupied on the steps below it. Merging is therefore what makes one record
        carry both halves.

        A measured field takes the largest figure any step recorded, compared as a number so the suffix the scheduler
        appends orders the values. The record carries the winning step's own text, which leaves every figure in the
        units the scheduler chose. Every remaining field takes the first value a record supplied, since the job row
        already carries it.

    Args:
        rows: The parsed records, in the order the scheduler wrote them.

    Returns:
        One record per job, carrying the job's own fields and the largest figure its steps measured.
    """
    merged: dict[str, dict[str, str]] = {}
    for row in rows:
        identifier = row.get("job_id", "")
        if _EXTERN_STEP_NAME in identifier:
            continue

        job_identifier = identifier.split(_STEP_SEPARATOR)[0]
        held = merged.get(job_identifier)
        if held is None:
            merged[job_identifier] = dict(row)
            continue
        for key, value in row.items():
            if not value:
                continue
            if key in _MEASURED_SIZE_FIELDS or key in _MEASURED_DURATION_FIELDS:
                if _exceeds_held(field=key, value=value, held=held.get(key, "")):
                    held[key] = value
            elif not held.get(key):
                held[key] = value
    return list(merged.values())


def _exceeds_held(field: str, value: str, held: str) -> bool:
    """Determines whether one step's measured figure is larger than the figure the job already holds.

    Args:
        field: The response key under which the figure is reported.
        value: The figure the step recorded.
        held: The figure the job holds, which is empty until a step supplies one.

    Returns:
        True when the step's figure parses and either exceeds the held one or is the first to parse.
    """
    parse = _parse_duration if field in _MEASURED_DURATION_FIELDS else _parse_size
    candidate = parse(value=value)
    if candidate is None:
        return False
    current = parse(value=held)
    return current is None or candidate > current


def _parse_size(value: str) -> float | None:
    """Parses one memory figure the scheduler reported into its byte count.

    Args:
        value: The figure as the scheduler rendered it, such as a magnitude followed by a unit suffix.

    Returns:
        The byte count the figure represents, or None when it carries no magnitude.
    """
    text = value.strip()
    index = 0
    while index < len(text) and (text[index].isdigit() or text[index] == "."):
        index += 1
    try:
        magnitude = float(text[:index])
    except ValueError:
        return None
    return magnitude * _SIZE_SUFFIX_SCALES.get(text[index : index + 1].upper(), 1.0)


def _parse_duration(value: str) -> float | None:
    """Parses one duration the scheduler reported into the number of seconds it represents.

    Notes:
        A duration leads with an optional day count and follows it with colon-separated clock fields, each of which
        exceeds the field to its right by sixty. Every field is folded into a running total at that scale, so one
        pass serves a duration carrying hours and one carrying minutes alone.

    Args:
        value: The duration as the scheduler rendered it.

    Returns:
        The number of seconds the duration represents, or None when its fields do not parse as numbers.
    """
    text = value.strip()
    days, separator, clock = text.partition(_DAY_SEPARATOR)
    if not separator:
        days, clock = "0", text

    seconds = 0.0
    try:
        for field_value in clock.split(_TIME_SEPARATOR):
            seconds = seconds * _SECONDS_PER_MINUTE + float(field_value)
        return float(days) * _SECONDS_PER_DAY + seconds
    except ValueError:
        return None


def _parse_queue_rows(output: str) -> list[dict[str, str]]:
    """Parses the scheduler's queue response into one entry per allocation.

    Args:
        output: The response the queue command wrote.

    Returns:
        One entry per allocation, keyed by the response keys onto which the queue columns map.
    """
    lines = [line for line in output.strip().split("\n") if line.strip()]
    if len(lines) < _MINIMUM_TABLE_LINES:
        return []

    # The queue names its columns after the format specifiers rather than after the reported keys, so the header line
    # is dropped and the columns are read by position.
    return [
        dict(zip(_QUEUE_FIELDS, [value.strip() for value in values], strict=True))
        for line in lines[1:]
        if len(values := line.split(_FIELD_SEPARATOR)) == len(_QUEUE_FIELDS)
    ]


def _reject_unmatched(field: str, values: list[str], available: set[str]) -> dict[str, Any] | None:
    """Builds the error response for a filter naming a value the scheduler's answer does not hold.

    Notes:
        Reports what is available rather than returning an empty page, because an empty page and a mistyped filter
        look identical to a caller otherwise. An answer holding no record at all is left to page as empty, since a
        query that matched nothing has no values to offer.

    Args:
        field: The field being filtered.
        values: The values the caller named.
        available: The values the answer holds for that field.

    Returns:
        The error response, or None when every named value is present.
    """
    if not available:
        return None
    unknown = sorted({value for value in values if value not in available})
    if not unknown:
        return None
    return error_response(message=f"No scheduler record has '{field}' in {unknown}. Available: {sorted(available)}.")
