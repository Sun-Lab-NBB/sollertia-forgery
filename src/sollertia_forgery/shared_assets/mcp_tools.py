"""Provides shared Model Context Protocol (MCP) tools used across multiple library packages."""

from __future__ import annotations  # pragma: no cover

from typing import Any  # pragma: no cover
from pathlib import Path  # pragma: no cover

from sollertia_shared_assets import SessionData, SessionTypes, DatasetSession  # pragma: no cover

from .metadata import ProjectManifest  # pragma: no cover
from ..interfaces import mcp  # pragma: no cover
from .mcp_orchestration import validate_directory  # pragma: no cover
from .session_discovery import filter_sessions, discover_sessions  # pragma: no cover

_STATUS_COLUMNS: frozenset[str] = frozenset({"complete", "integrity", "cindra", "behavior", "video"})
"""The manifest column names that store boolean-like UInt8 processing status flags, cast to native bools by
``get_project_manifest_tool`` for readability."""


@mcp.tool()  # pragma: no cover
def discover_sessions_tool(  # pragma: no cover
    root_directory: str,
    session_types: list[str] | None = None,
) -> dict[str, Any]:
    """Discovers sessions under a project root directory.

    Recursively searches for ``session_data.yaml`` marker files to identify session root directories, loads each
    session via :meth:`SessionData.load`, and returns metadata. Accepts an optional list of session type strings
    to filter sessions by eligibility. When no filter is provided, all discovered sessions are returned as
    eligible.

    Args:
        root_directory: The absolute path to the root directory to search. Searched recursively.
        session_types: An optional list of session type strings to filter by (e.g.,
            ``["lick_training", "mesoscope_experiment"]``). When provided, each session entry includes an
            ``eligible`` flag and the ``session_paths`` list contains only matching sessions. When omitted, all
            sessions are returned as eligible.

    Returns:
        A dictionary containing a ``sessions`` list where each entry has ``session_path``, ``session_name``,
        ``animal_id``, ``session_type``, ``acquisition_system``, ``raw_data_path``, ``processed_data_path``,
        and ``eligible`` keys, a flat ``session_paths`` list of eligible session roots, and aggregate counts.
        Sessions that fail to load produce entries with ``session_path``, ``eligible=False``, and ``error``.
    """
    error = validate_directory(root_directory)
    if error is not None:
        return {"error": error}

    root_path = Path(root_directory)

    # Converts the optional session type string list to a frozenset of SessionTypes enum values.
    types_filter: frozenset[SessionTypes] | None = None
    if session_types is not None:
        try:
            types_filter = frozenset(SessionTypes(t) for t in session_types)
        except ValueError as error:
            return {"error": f"Invalid session type in filter: {error}"}

    # Discovers all session root paths under the target directory.
    try:
        session_paths = discover_sessions(root_path=root_path)
    except PermissionError as error:
        return {"error": f"Permission denied during search: {error}"}

    # Loads metadata for each discovered session and applies optional type filtering.
    sessions_output: list[dict[str, Any]] = []
    eligible_paths: list[str] = []

    for session_root in session_paths:
        try:
            session = SessionData.load(session_path=session_root)
        except Exception as error:
            sessions_output.append(
                {
                    "session_path": str(session_root),
                    "eligible": False,
                    "error": f"Unable to load session: {error}",
                }
            )
            continue

        eligible = session.session_type in types_filter if types_filter is not None else True
        entry: dict[str, Any] = {
            "session_path": str(session_root),
            "session_name": session.session_name,
            "animal_id": session.animal_id,
            "session_type": str(session.session_type),
            "acquisition_system": str(session.acquisition_system),
            "raw_data_path": str(session.raw_data_path),
            "processed_data_path": str(session.processed_data_path),
            "eligible": eligible,
        }

        if eligible:
            eligible_paths.append(str(session_root))

        sessions_output.append(entry)

    return {
        "sessions": sessions_output,
        "session_paths": eligible_paths,
        "total_sessions": len(sessions_output),
        "total_eligible": len(eligible_paths),
    }


@mcp.tool()  # pragma: no cover
def get_project_manifest_tool(  # pragma: no cover
    manifest_file: str,
    animal: int | None = None,
    session: str | None = None,
    *,
    include_notes: bool = False,
) -> dict[str, Any]:
    """Reads a project manifest ``.feather`` file and returns structured project metadata with per-session data.

    Loads the manifest via :class:`ProjectManifest`, computes aggregate statistics with
    :meth:`~ProjectManifest.summarize`, and returns per-session rows as dictionaries. Supports two retrieval
    modes: when ``session`` is provided, returns full data for that single session including experimenter notes,
    which is useful for answering detailed questions about a specific session. When ``session`` is omitted,
    returns data for all sessions (optionally filtered by ``animal``), with notes excluded by default.

    Args:
        manifest_file: The absolute path to the ``.feather`` manifest file.
        animal: An optional animal identifier. When provided, only sessions belonging to that animal are included
            in the ``sessions`` list. Ignored when ``session`` is specified. The ``summary`` always reflects the
            full manifest regardless of this filter.
        session: An optional session identifier for targeted retrieval. When provided, returns full data for that
            single session including experimenter notes, ignoring ``animal`` and ``include_notes``.
        include_notes: Determines whether to include the ``notes`` column in each session entry. Defaults to
            ``False`` to keep responses concise. Ignored when ``session`` is specified, as notes are always
            included in targeted retrieval mode.

    Returns:
        A dictionary containing the ``manifest_file`` path, a ``summary`` with aggregate statistics, a
        ``sessions`` list of per-session dictionaries, and ``total_sessions`` count. Returns an ``error`` key
        on failure.
    """
    file_path = Path(manifest_file)

    if not file_path.exists():
        return {"error": f"Manifest file does not exist: {manifest_file}"}

    if not file_path.is_file():
        return {"error": f"Path is not a file: {manifest_file}"}

    try:
        manifest = ProjectManifest(manifest_file=file_path)
    except Exception as error:
        return {"error": f"Unable to read manifest file: {error}"}

    # Computes aggregate statistics from the full manifest.
    summary = manifest.summarize()

    # Targeted session retrieval mode — returns full data for a single session including notes.
    if session is not None:
        session_df = manifest.get_session_data(session=session)
        if session_df.is_empty():
            available = list(manifest.get_sessions(animal=None, exclude_incomplete=False))
            return {"error": f"Session '{session}' not found in manifest. Available sessions: {available}."}

        row: dict[str, Any] = session_df.to_dicts()[0]
        for column in _STATUS_COLUMNS:
            if column in row:
                row[column] = bool(row[column])

        return {
            "manifest_file": manifest_file,
            "summary": summary,
            "sessions": [row],
            "total_sessions": 1,
        }

    # Validates the animal filter against the manifest's known animals.
    if animal is not None and animal not in manifest.animals:
        return {"error": f"Animal ID '{animal}' not found in manifest. Available animals: {list(manifest.animals)}."}

    # Retrieves per-session data using ProjectManifest's public filtering API.
    session_names = manifest.get_sessions(animal=animal, exclude_incomplete=False)

    session_rows: list[dict[str, Any]] = []

    for session_name in session_names:
        session_df = manifest.get_session_data(session=session_name)
        if session_df.is_empty():
            continue

        row = session_df.to_dicts()[0]

        # Excludes the notes column unless explicitly requested.
        if not include_notes and "notes" in row:
            del row["notes"]

        # Casts boolean-like UInt8 status columns to native bools for readability.
        for column in _STATUS_COLUMNS:
            if column in row:
                row[column] = bool(row[column])

        session_rows.append(row)

    return {
        "manifest_file": manifest_file,
        "summary": summary,
        "sessions": session_rows,
        "total_sessions": len(session_rows),
    }


@mcp.tool()  # pragma: no cover
def filter_sessions_tool(  # pragma: no cover
    sessions: list[dict[str, Any]],
    start_date: str | None = None,
    end_date: str | None = None,
    include_sessions: list[str] | None = None,
    exclude_sessions: list[str] | None = None,
    include_animals: list[str] | None = None,
    exclude_animals: list[str] | None = None,
    *,
    utc_timezone: bool = True,
) -> dict[str, Any]:
    """Filters a list of session entries by date range and inclusion/exclusion criteria.

    Designed for agentic chaining with :func:`discover_sessions_tool`: accepts the ``sessions`` list from its
    output and returns a filtered subset with the same structure. Each input entry must contain ``session_name``
    and ``animal_id`` keys. Animal filtering is applied before session filtering, and exclusion takes precedence
    over inclusion.

    Args:
        sessions: A list of session entry dictionaries, each containing at least ``session_name`` and
            ``animal_id`` keys. Typically, the ``sessions`` list from :func:`discover_sessions_tool`.
        start_date: Sessions recorded on or after this date are included. Accepts formats like ``YYYY-MM-DD``
            or ``YYYY-MM-DD HH:MM:SS``. When ``None``, no start bound is applied.
        end_date: Sessions recorded on or before this date are included. Date-only values include the entire
            day. When ``None``, no end bound is applied.
        include_sessions: Session names to include regardless of date range, unless overridden by
            ``exclude_sessions``.
        exclude_sessions: Session names to exclude from results. Takes precedence over all other inclusion
            criteria.
        include_animals: Animal identifiers to include. When provided, only sessions from these animals are
            considered.
        exclude_animals: Animal identifiers to exclude. Takes precedence over ``include_animals``.
        utc_timezone: Determines whether to interpret date boundaries and session timestamps in UTC. When
            ``False``, uses America/New_York timezone.

    Returns:
        A dictionary containing a filtered ``sessions`` list, a ``session_paths`` list of eligible session
        roots, and ``total_sessions`` / ``total_eligible`` counts. Structurally identical to the output of
        :func:`discover_sessions_tool` for downstream chaining.
    """
    # Builds DatasetSession objects and a reverse map from session name to the original entry.
    dataset_sessions: set[DatasetSession] = set()
    session_map: dict[str, dict[str, Any]] = {}
    invalid_entries: list[dict[str, Any]] = []

    for entry in sessions:
        session_name = entry.get("session_name")
        animal_id = entry.get("animal_id")

        if session_name is None or animal_id is None:
            invalid_entries.append({**entry, "filter_error": "Missing required 'session_name' or 'animal_id' field."})
            continue

        dataset_sessions.add(DatasetSession(session=str(session_name), animal=str(animal_id)))
        session_map[str(session_name)] = entry

    # Applies the shared filtering logic.
    filtered = filter_sessions(
        sessions=dataset_sessions,
        start_date=start_date,
        end_date=end_date,
        include_sessions=set(include_sessions) if include_sessions else None,
        exclude_sessions=set(exclude_sessions) if exclude_sessions else None,
        include_animals=set(include_animals) if include_animals else None,
        exclude_animals=set(exclude_animals) if exclude_animals else None,
        utc_timezone=utc_timezone,
    )

    # Maps filtered results back to the original entry dictionaries.
    filtered_entries = sorted(
        (
            session_map[dataset_session.session]
            for dataset_session in filtered
            if dataset_session.session in session_map
        ),
        key=lambda session_entry: session_entry.get("session_name", ""),
    )

    # Builds the session_paths list from eligible filtered entries.
    eligible_paths = sorted(
        entry["session_path"] for entry in filtered_entries if entry.get("eligible", True) and "session_path" in entry
    )

    result: dict[str, Any] = {
        "sessions": filtered_entries,
        "session_paths": eligible_paths,
        "total_sessions": len(filtered_entries),
        "total_eligible": len(eligible_paths),
    }

    if invalid_entries:
        result["invalid_entries"] = invalid_entries

    return result
