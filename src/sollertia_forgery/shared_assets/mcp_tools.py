"""Provides shared Model Context Protocol (MCP) tools used across multiple library packages."""

from __future__ import annotations

from typing import Any
from pathlib import Path

from .metadata import ProjectManifest
from ..interfaces import mcp

_STATUS_COLUMNS: frozenset[str] = frozenset({"complete", "integrity", "cindra", "behavior", "video"})
"""The manifest column names that store boolean-like UInt8 processing status flags, cast to native bools by
``get_project_manifest_tool`` for readability."""


@mcp.tool()
def get_project_manifest_tool(
    manifest_file: str,
    animal: int | None = None,
    session: str | None = None,
    *,
    include_notes: bool = False,
) -> dict[str, Any]:
    """Reads a project manifest ``.feather`` file and returns structured project metadata with per-session data.

    Loads the manifest via ``ProjectManifest``, computes aggregate statistics with its ``summarize`` method, and
    returns per-session rows as dictionaries. Supports two retrieval
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
