"""Provides shared session discovery, path resolution, and session filtering assets used across multiple library
packages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from datetime import datetime
from zoneinfo import ZoneInfo

from dateutil import parser
from sollertia_shared_assets import SessionData

from .mcp_orchestration import SESSION_MARKER_FILENAME

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Iterator

    from ..forging.dataset_data import DatasetSession

_SESSION_NAME_COMPONENTS: int = 7
"""The number of hyphen-separated components in a valid session name (YYYY-MM-DD-HH-MM-SS-microseconds)."""


def get_session_root_from_marker(marker_path: Path) -> Path:
    """Returns the session root directory from a ``session_data.yaml`` marker path.

    The marker lives at ``{session_root}/raw_data/session_data.yaml``, so the session root is two directory
    levels above the marker file.

    Args:
        marker_path: The absolute path to a ``session_data.yaml`` file.

    Returns:
        The path to the session root directory (the grandparent of the marker).
    """
    return marker_path.parents[1]


def discover_sessions(root_path: Path) -> list[Path]:
    """Discovers session root directories under a root path by locating ``session_data.yaml`` markers.

    Recursively searches for ``SESSION_MARKER_FILENAME`` files and derives each session's root directory
    from the marker location. Returns only the resolved paths without loading or validating session data,
    making this function suitable as a lightweight discovery primitive for both MCP tools and internal
    pipeline code.

    Args:
        root_path: The absolute path to the directory to search recursively.

    Returns:
        A sorted list of absolute paths to session root directories found under ``root_path``.

    Raises:
        PermissionError: If the search encounters a directory it cannot read.
    """
    return sorted(
        get_session_root_from_marker(marker_path=marker) for marker in root_path.rglob(SESSION_MARKER_FILENAME)
    )


def iter_sessions(root_path: Path) -> Iterator[SessionData]:
    """Discovers and lazily loads every ``SessionData`` instance under the target root directory.

    Thin composite over :func:`discover_sessions` and :meth:`SessionData.load` so that callers (manifest
    generation, MCP status aggregators) can switch from project-wide ``rglob`` scans to typed iteration
    in one call. The returned iterator yields sessions in the same sorted order as ``discover_sessions``.

    Args:
        root_path: The absolute path to the directory to search recursively for session markers.

    Yields:
        Each :class:`SessionData` instance loaded from a session marker found under ``root_path``.
    """
    for session_root in discover_sessions(root_path=root_path):
        yield SessionData.load(session_path=session_root)


def filter_sessions(
    sessions: set[DatasetSession],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    include_sessions: set[str] | None = None,
    exclude_sessions: set[str] | None = None,
    include_animals: set[str] | None = None,
    exclude_animals: set[str] | None = None,
    utc_timezone: bool = True,
) -> set[DatasetSession]:
    """Filters the input set of dataset sessions based on the specified date ranges and inclusion/exclusion criteria.

    This function provides a general-purpose filtering mechanism for selecting a subset of all available sessions.
    Animal filtering is carried out before the session filtering. Exclusion filtering takes precedence over inclusion
    filtering.

    Args:
        sessions: The set of DatasetSession instances representing the sessions to be filtered.
        start_date: The start date for the date range filter. Sessions recorded on or after this date are included.
            Accepts various date formats (e.g., 'YYYY-MM-DD', 'YYYY-MM-DD HH:MM:SS'). If None, no start date filter
            is applied.
        end_date: The end date for the date range filter. Sessions recorded on or before this date are included.
            If only a date is provided (no time), the filter includes the entire day. If None, no end date filter
            is applied.
        include_sessions: A set of session names to include regardless of the date range. These sessions are included
            even if they fall outside the start_date/end_date range, unless they are in exclude_sessions.
        exclude_sessions: A set of session names to exclude from the results. This takes precedence over all other
            inclusion criteria.
        include_animals: A set of animal names to include. If specified, only sessions from these animals are
            considered. If None, sessions from all animals are considered.
        exclude_animals: A set of animal names to exclude. Sessions from these animals are removed from the results.
            This takes precedence over include_animals.
        utc_timezone: Determines whether to interpret date boundaries and session timestamps in UTC (True) or
            America/New_York (False) timezone. Session names reflect the UTC timestamps, but when this is False,
            the function converts them to America/New_York for comparison.

    Returns:
        A set of DatasetSession instances that match the filtering criteria.
    """
    # Applies animal exclusion filter (takes precedence over animal inclusion).
    if exclude_animals:
        sessions = {s for s in sessions if s.animal not in exclude_animals}

    # Applies animal inclusion filter.
    if include_animals:
        sessions = {s for s in sessions if s.animal in include_animals}

    # Applies session exclusion filter (takes precedence over all session inclusion criteria).
    if exclude_sessions:
        sessions = {s for s in sessions if s.session not in exclude_sessions}

    # Applies date range and session inclusion filters. Sessions are included if they fall within the date range
    # OR are in include_sessions.
    if start_date is not None or end_date is not None or include_sessions:
        # Parses date boundaries.
        parsed_start = (
            _parse_date_boundary(start_date, is_end_date=False, utc_timezone=utc_timezone) if start_date else None
        )
        parsed_end = _parse_date_boundary(end_date, is_end_date=True, utc_timezone=utc_timezone) if end_date else None

        filtered = set()
        for session in sessions:
            # Checks if the session is explicitly included.
            if include_sessions and session.session in include_sessions:
                filtered.add(session)
                continue

            # Checks if the session falls within the date range.
            session_date = _parse_session_date(session.session, utc_timezone=utc_timezone)
            if session_date is not None:
                in_range = True
                if parsed_start and session_date < parsed_start:
                    in_range = False
                if parsed_end and session_date > parsed_end:
                    in_range = False
                if in_range:
                    filtered.add(session)

        sessions = filtered

    return sessions


def _parse_date_boundary(date_string: str, *, is_end_date: bool = False, utc_timezone: bool = True) -> datetime:
    """Parses the input date and time string preserving any time information provided.

    Args:
        date_string: A date and time string in various formats (YYYY-MM-DD or with time).
        is_end_date: Determines whether to set the time component to the end of the day when only the date data
            is provided in the string.
        utc_timezone: Determines whether to interpret the date string as UTC. When ``False``, interprets it as
            America/New_York.

    Returns:
        The timezone-aware datetime object constructed from the input string's data.
    """
    parsed = parser.parse(date_string)

    # Checks if only the date was provided (parser defaults to midnight).
    date_only = "T" not in date_string and " " not in date_string and ":" not in date_string

    if date_only and is_end_date:
        # Makes end dates inclusive of the entire day.
        parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=999999)

    # Determines the target timezone based on the utc_timezone flag.
    target_tz = ZoneInfo("UTC") if utc_timezone else ZoneInfo("America/New_York")

    # Ensures timezone awareness and returns the parsed data.
    return parsed.replace(tzinfo=target_tz) if parsed.tzinfo is None else parsed.astimezone(target_tz)


def _parse_session_date(session_name: str, *, utc_timezone: bool = True) -> datetime | None:
    """Parses the session name to extract its acquisition datetime.

    Session names follow the format 'YYYY-MM-DD-HH-MM-SS-microseconds' and encode the session's acquisition
    timestamp in the UTC timezone.

    Args:
        session_name: The unique identifier of the session.
        utc_timezone: Determines whether to return the datetime in UTC. When ``False``, converts to
            America/New_York timezone.

    Returns:
        The timezone-aware datetime object representing when the session was acquired, or None if the session name
        does not follow the expected format.
    """
    parts = session_name.split("-")
    if len(parts) != _SESSION_NAME_COMPONENTS:
        return None

    try:
        year, month, day, hour, minute, second, microseconds = parts
        # Session names always store UTC timestamps.
        utc_dt = datetime(
            year=int(year),
            month=int(month),
            day=int(day),
            hour=int(hour),
            minute=int(minute),
            second=int(second),
            microsecond=int(microseconds),
            tzinfo=ZoneInfo("UTC"),
        )
        # Returns in UTC or converts to America/New_York based on the flag.
        if utc_timezone:
            return utc_dt
        return utc_dt.astimezone(ZoneInfo("America/New_York"))
    except ValueError, IndexError:
        return None
