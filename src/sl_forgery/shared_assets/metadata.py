"""This module provides the metadata assets that allow other library modules to work with the project data stored on
remote compute servers.
"""

from typing import TYPE_CHECKING
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import dataclass

import polars as pl
from dateutil import parser
from ataraxis_base_utilities import console

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class SessionMetadata:
    """Encapsulates the identity metadata for a single data acquisition session.

    Attributes:
        session: The unique identifier of the session. Session names follow the format
            'YYYY-MM-DD-HH-MM-SS-microseconds' and encode the session's acquisition timestamp.
        animal: The unique identifier of the animal that participated in the session.
    """

    session: str
    """The unique identifier of the session."""
    animal: str
    """The unique identifier of the animal that participated in the session."""


def filter_sessions(
    sessions: set[SessionMetadata],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    include_sessions: set[str] | None = None,
    exclude_sessions: set[str] | None = None,
    include_animals: set[str] | None = None,
    exclude_animals: set[str] | None = None,
    utc_timezone: bool = True,
) -> set[SessionMetadata]:
    """Filters the input set of sessions based on the specified date ranges and inclusion/exclusion criteria.

    This function provides a general-purpose filtering mechanism for selecting a subset of all available sessions.
    Animal filtering is carried out before the session filtering. Exclusion filtering takes precedence over inclusion
    filtering.

    Args:
        sessions: The set of SessionMetadata instances representing the sessions to be filtered.
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
        A set of SessionMetadata instances that match the filtering criteria.
    """
    # Step 1: Applies animal exclusion filter (takes precedence over animal inclusion)
    if exclude_animals:
        sessions = {s for s in sessions if s.animal not in exclude_animals}

    # Step 2: Applies animal inclusion filter
    if include_animals:
        sessions = {s for s in sessions if s.animal in include_animals}

    # Step 3: Applies session exclusion filter (takes precedence over all session inclusion criteria)
    if exclude_sessions:
        sessions = {s for s in sessions if s.session not in exclude_sessions}

    # Step 4: Applies date range and session inclusion filters
    # Sessions are included if they fall within the date range OR are in include_sessions
    if start_date is not None or end_date is not None or include_sessions:
        # Parses date boundaries
        parsed_start = (
            _parse_date_boundary(start_date, is_end_date=False, utc_timezone=utc_timezone) if start_date else None
        )
        parsed_end = _parse_date_boundary(end_date, is_end_date=True, utc_timezone=utc_timezone) if end_date else None

        filtered = set()
        for session in sessions:
            # Checks if the session is explicitly included
            if include_sessions and session.session in include_sessions:
                filtered.add(session)
                continue

            # Checks if the session falls within the date range
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
        is_end_date: If True and only the date data is provided in the string, sets the time component to the
            end of the day.
        utc_timezone: If True, interprets the date string as UTC. If False, interprets it as America/New_York.

    Returns:
        The timezone-aware datetime object constructed from the input string's data.
    """
    parsed = parser.parse(date_string)

    # Checks if only the date was provided (parser defaults to midnight)
    date_only = "T" not in date_string and " " not in date_string and ":" not in date_string

    if date_only and is_end_date:
        # Makes end dates inclusive of the entire day
        parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=999999)

    # Determines the target timezone based on the utc_timezone flag
    target_tz = ZoneInfo("UTC") if utc_timezone else ZoneInfo("America/New_York")

    # Ensures timezone awareness and returns the parsed data
    return parsed.replace(tzinfo=target_tz) if parsed.tzinfo is None else parsed.astimezone(target_tz)


# The number of hyphen-separated components in a valid session name (YYYY-MM-DD-HH-MM-SS-microseconds)
_SESSION_NAME_COMPONENTS = 7


def _parse_session_date(session_name: str, *, utc_timezone: bool = True) -> datetime | None:
    """Parses the session name to extract its acquisition datetime.

    Session names follow the format 'YYYY-MM-DD-HH-MM-SS-microseconds' and encode the session's acquisition
    timestamp in the UTC timezone.

    Args:
        session_name: The unique identifier of the session.
        utc_timezone: If True, returns the datetime in UTC. If False, converts to America/New_York timezone.

    Returns:
        The timezone-aware datetime object representing when the session was acquired, or None if the session name
        does not follow the expected format.
    """
    parts = session_name.split("-")
    if len(parts) != _SESSION_NAME_COMPONENTS:
        return None

    try:
        year, month, day, hour, minute, second, microseconds = parts
        # Session names always store UTC timestamps
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
        # Returns in UTC or converts to America/New_York based on the flag
        if utc_timezone:
            return utc_dt
        return utc_dt.astimezone(ZoneInfo("America/New_York"))
    except (ValueError, IndexError):
        return None


class ProjectManifest:
    """Provides methods for visualizing and working with the data stored inside the managed project manifest .feather
    file.

    Notes:
        This class provides the entry-point API for working with Sun lab's research project data. It is used by most
        data processing and analysis dataset formation pipelines to work with the processed project's data.

    Args:
        manifest_file: The path to the .feather manifest file that stores the snapshot of the target project's state.

    Attributes:
        _data: The Polars DataFrame that stores the snapshot of the project's state.
        _animal_string: Determines whether animal IDs are stored as strings or unsigned integers.
    """

    def __init__(self, manifest_file: Path) -> None:
        # Reads the data from the target manifest file into the class attribute.
        self._data: pl.DataFrame = pl.read_ipc(source=manifest_file, use_pyarrow=True, memory_map=True)

        # Determines whether animal IDs are stored as strings or as numbers.
        self._animal_string = False
        schema = self._data.collect_schema()
        if isinstance(schema["animal"], pl.String):
            self._animal_string = True

    def print_data(self) -> None:
        """Prints the entire contents of the manifest file to the terminal."""
        with pl.Config(
            set_tbl_rows=-1,  # Displays all rows (-1 means unlimited)
            set_tbl_cols=-1,  # Displays all columns (-1 means unlimited)
            set_tbl_hide_column_data_types=True,
            set_tbl_cell_alignment="LEFT",
            set_tbl_width_chars=250,  # Sets table width to 250 characters
            set_fmt_str_lengths=600,  # Allows longer strings to display properly (default is 32)
        ):
            print(self._data)  # noqa: T201

    def print_summary(self, animal: str | int | None = None) -> None:
        """Prints a summary view of the manifest file to the terminal, excluding the 'experimenter notes' data for
        each session.

        This data view is optimized for tracking which processing steps have been applied to each of the project's data
        acquisition sessions.

        Args:
            animal: The unique identifier of the animal for which to display the data. If provided, this method only
                displays the data for that animal. Otherwise, it displays the data for all animals.
        """
        summary_cols = [
            "animal",
            "date",
            "session",
            "type",
            "system",
            "complete",
            "integrity",
            "suite2p",
            "behavior",
            "video",
        ]

        # Retrieves the data.
        df = self._data.select(summary_cols)

        # Optionally filters the data for the target animal.
        if animal is not None:
            # Ensures that the 'animal' argument has the same type as the data inside the DataFrame.
            animal = str(animal) if self._animal_string else int(animal)
            df = df.filter(pl.col("animal") == animal)

        # Ensures the data displays properly.
        with pl.Config(
            set_tbl_rows=-1,
            set_tbl_cols=-1,
            set_tbl_width_chars=250,
            set_tbl_hide_column_data_types=True,
            set_tbl_cell_alignment="CENTER",
        ):
            print(df)  # noqa: T201

    def print_notes(self, animal: str | int | None = None) -> None:
        """Prints the animal ID, session ID, and experimenter notes data for each project's session to the terminal.

        This data view is optimized for determining what data acquisition sessions have been carried out and checking
        the outcomes of each session recorded in the experimenter notes.

        Args:
            animal: The unique identifier of the animal for which to display the data. If provided, this method only
                displays the data for that animal. Otherwise, it displays the data for all animals.
        """
        # Pre-selects the columns to display.
        df = self._data.select(["animal", "date", "session", "type", "system", "notes"])

        # Optionally filters the data for the target animal.
        if animal is not None:
            # Ensures that the 'animal' argument has the same type as the data inside the DataFrame.
            animal = str(animal) if self._animal_string else int(animal)

            df = df.filter(pl.col("animal") == animal)

        # Prints the extracted data.
        with pl.Config(
            set_tbl_rows=-1,
            set_tbl_cols=-1,
            set_tbl_hide_column_data_types=True,
            set_tbl_cell_alignment="LEFT",
            set_tbl_width_chars=170,  # Wider columns for notes
            set_fmt_str_lengths=2000,  # Allows very long strings for notes
        ):
            print(df)  # noqa: T201

    @property
    def animals(self) -> tuple[str, ...]:
        """Returns the unique identifiers for each animal participating in the project."""
        # If animal IDs are stored as integers, converts them to string to support consistent return types.
        return tuple(
            [str(animal) for animal in self._data.select("animal").unique().sort("animal").to_series().to_list()]
        )

    def _get_filtered_sessions(
        self,
        animal: str | int | None = None,
        *,
        exclude_incomplete: bool = True,
    ) -> tuple[str, ...]:
        """Builds a tuple of unique session identifiers with optional filtering.

        Notes:
            User-facing methods call this worker method under-the-hood to fetch the filtered tuple of session IDs.

        Args:
            animal: An optional animal identifier for which to retrieve the sessions. If set to None, the method
                returns the session IDs for all animals participating in the project.
            exclude_incomplete: Determines whether to exclude incomplete sessions from the output tuple.

        Returns:
            The tuple of unique session identifiers matching the filter criteria.

        Raises:
            ValueError: If the specified animal identifier is not found in the manifest file.
        """
        data = self._data

        # Filters by animal if specified.
        if animal is not None:
            # Ensures that the 'animal' argument has the same type as the data inside the DataFrame.
            animal = str(animal) if self._animal_string else int(animal)

            if animal not in self.animals:
                message = f"Animal ID '{animal}' not found in the project manifest. Available animals: {self.animals}."
                console.error(message=message, error=ValueError)

            data = data.filter(pl.col("animal") == animal)

        # Optionally filters out incomplete sessions.
        if exclude_incomplete:
            data = data.filter(pl.col("complete") == 1)

        # Formats and returns session IDs to the caller.
        sessions = data.select("session").sort("session").to_series().to_list()
        return tuple(sessions)

    def get_sessions(
        self,
        animal: str | int | None = None,
        *,
        exclude_incomplete: bool = True,
    ) -> tuple[str, ...]:
        """Returns the unique identifiers for all sessions that match the input filtering criteria.

        Args:
            animal: An optional animal identifier for which to retrieve the session identifiers. If set to None, the
                method returns the session IDs for all animals participating in the project.
            exclude_incomplete: Determines whether to exclude incomplete sessions from the output tuple.

        Returns:
            The tuple of session identifiers matching the filtering criteria.

        Raises:
            ValueError: If the specified animal is not found in the manifest file.
        """
        return self._get_filtered_sessions(
            animal=animal,
            exclude_incomplete=exclude_incomplete,
        )

    def get_session_data(self, session: str) -> pl.DataFrame:
        """Returns a Polars DataFrame that stores detailed information about the current acquisition and processing
        state of the specified session.

        Args:
            session: The unique identifier of the session for which to retrieve the data.

        Returns:
            A Polars DataFrame with the following columns: 'animal', 'date', 'notes', 'session', 'type', 'system',
            'complete', 'integrity', 'suite2p', 'behavior', 'video'.
        """
        return self._data.filter(pl.col("session").eq(session))

    def get_animal_for_session(self, session: str) -> str:
        """Returns the unique identifier of the animal that participated in the specified session.

        Args:
            session: The unique identifier of the session for which to retrieve the participating animal's identifier.

        Returns:
            The unique identifier of the animal that participated in the specified session.

        Raises:
            ValueError: If the specified session is not found in the manifest file.
        """
        # Filters the data for the specified session.
        df = self._data.filter(pl.col("session") == session)

        # Checks if the session exists.
        if df.is_empty():
            message = (
                f"Session ID '{session}' not found in the project manifest. "
                f"Available sessions: {self.get_sessions(animal=None, exclude_incomplete=False)}."
            )
            console.error(message=message, error=ValueError)

        # Extracts the animal ID.
        animal_id = df.select("animal").item()

        # Returns the animal ID with the appropriate type.
        return str(animal_id)

    def get_system_for_session(self, session: str) -> str:
        """Returns the data acquisition system used to acquire the specified session's data.

        Args:
            session: The unique identifier of the session for which to retrieve the data acquisition system.

        Returns:
            The data acquisition system used to acquire the specified session's data.

        Raises:
            ValueError: If the specified session is not found in the manifest file.
        """
        # Filters the data for the specified session.
        df = self._data.filter(pl.col("session") == session)

        # Checks if the session exists.
        if df.is_empty():
            message = (
                f"Session ID '{session}' not found in the project manifest. "
                f"Available sessions: {self.get_sessions(animal=None, exclude_incomplete=False)}."
            )
            console.error(message=message, error=ValueError)

        # Extracts and returns the acquisition system used to acquire the session.
        return str(df.select("system").item())

    @property
    def data(self) -> pl.DataFrame:
        """Returns the Polars DataFrame instance that stores the managed manifest file's data."""
        return self._data
