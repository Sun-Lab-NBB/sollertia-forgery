"""Provides shared Model Context Protocol (MCP) tools used across multiple library packages."""

from __future__ import annotations  # pragma: no cover

from typing import Any  # pragma: no cover
from pathlib import Path  # pragma: no cover

from sollertia_shared_assets import SessionData, SessionTypes  # pragma: no cover

from ..interfaces import mcp  # pragma: no cover
from .mcp_orchestration import validate_directory  # pragma: no cover
from .session_discovery import discover_sessions  # pragma: no cover


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
