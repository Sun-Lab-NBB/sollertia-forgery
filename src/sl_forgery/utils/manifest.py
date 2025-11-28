"""This module provides assets for working with project manifest .feather files."""

from pathlib import Path

import polars as pl
from ataraxis_base_utilities import console


class ProjectManifest:
    """Wraps the contents of a Sun lab project manifest .feather file and exposes methods for visualizing and
    working with the data stored inside the file.

    This class functions as a high-level API for working with Sun lab projects. It is used both to visualize the
    current state of various projects and during automated data processing to determine which processing steps to
    apply to different sessions.

    Args:
        manifest_file: The path to the .feather manifest file that stores the target project's state data.

    Attributes:
        _data: Stores the manifest data as a Polars DataFrame.
        _animal_string: Determines whether animal IDs are stored as strings or unsigned integers.
    """

    def __init__(self, manifest_file: Path):
        # Reads the data from the target manifest file into the class attribute
        self._data: pl.DataFrame = pl.read_ipc(source=manifest_file, use_pyarrow=True)

        # Determines whether animal IDs are stored as strings or as numbers
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
            set_tbl_width_chars=250,  # Sets table width to 200 characters
            set_fmt_str_lengths=600,  # Allows longer strings to display properly (default is 32)
        ):
            print(self._data)

    def print_summary(self, animal: str | int | None = None) -> None:
        """Prints a summary view of the manifest file to the terminal, excluding the 'experimenter notes' data for
        each session.

        This data view is optimized for tracking which processing steps have been applied to each session inside the
        project.

        Args:
            animal: The ID of the animal for which to display the data. If an ID is provided, this method will only
                display the data for that animal. Otherwise, it will display the data for all animals.
        """
        summary_cols = [
            "animal",
            "date",
            "session",
            "type",
            "system",
            "complete",
            "integrity",
            "prepared",
            "suite2p",
            "behavior",
            "video",
            "archived",
        ]

        # Retrieves the data
        df = self._data.select(summary_cols)

        # Optionally filters the data for the target animal
        if animal is not None:
            # Ensures that the 'animal' argument has the same type as the data inside the DataFrame.
            if self._animal_string:
                animal = str(animal)
            else:
                animal = int(animal)
            df = df.filter(pl.col("animal") == animal)

        # Ensures the data displays properly
        with pl.Config(
            set_tbl_rows=-1,
            set_tbl_cols=-1,
            set_tbl_width_chars=250,
            set_tbl_hide_column_data_types=True,
            set_tbl_cell_alignment="CENTER",
        ):
            print(df)

    def print_notes(self, animal: str | int | None = None) -> None:
        """Prints only animal, session, and notes data from the manifest file.

        This data view is optimized for experimenters to check what sessions have been recorded for each animal in the
        project and refresh their memory on the outcomes of each session using experimenter notes.

        Args:
            animal: The ID of the animal for which to display the data. If an ID is provided, this method will only
                display the data for that animal. Otherwise, it will display the data for all animals.
        """
        # Pre-selects the columns to display
        df = self._data.select(["animal", "date", "session", "type", "system", "notes"])

        # Optionally filters the data for the target animal
        if animal is not None:
            # Ensures that the 'animal' argument has the same type as the data inside the DataFrame.
            if self._animal_string:
                animal = str(animal)
            else:
                animal = int(animal)

            df = df.filter(pl.col("animal") == animal)

        #  Prints the extracted data
        with pl.Config(
            set_tbl_rows=-1,
            set_tbl_cols=-1,
            set_tbl_hide_column_data_types=True,
            set_tbl_cell_alignment="LEFT",
            set_tbl_width_chars=170,  # Wider columns for notes
            set_fmt_str_lengths=2000,  # Allows very long strings for notes
        ):
            print(df)

    @property
    def animals(self) -> tuple[str, ...]:
        """Returns all unique animal IDs stored inside the manifest file.

        This provides a tuple of all animal IDs participating in the target project.
        """
        # If animal IDs are stored as integers, converts them to string to support consistent return types.
        return tuple(
            [str(animal) for animal in self._data.select("animal").unique().sort("animal").to_series().to_list()]
        )

    def _get_filtered_sessions(
        self,
        animal: str | int | None = None,
        exclude_incomplete: bool = True,
    ) -> tuple[str, ...]:
        """This worker method is used to get a list of sessions with optional filtering.

        User-facing methods call this worker under-the-hood to fetch the filtered tuple of sessions.

        Args:
            animal: An optional animal ID to filter the sessions. If set to None, the method returns sessions for all
                animals.
            exclude_incomplete: Determines whether to exclude sessions not marked as 'complete' from the output
                list.

        Returns:
            The tuple of session IDs matching the filter criteria.

        Raises:
            ValueError: If the specified animal is not found in the manifest file.
        """
        data = self._data

        # Filter by animal if specified
        if animal is not None:
            # Ensures that the 'animal' argument has the same type as the data inside the DataFrame.
            if self._animal_string:
                animal = str(animal)
            else:
                animal = int(animal)

            if animal not in self.animals:
                message = f"Animal ID '{animal}' not found in the project manifest. Available animals: {self.animals}."
                console.error(message=message, error=ValueError)

            data = data.filter(pl.col("animal") == animal)

        # Optionally filters out incomplete sessions
        if exclude_incomplete:
            data = data.filter(pl.col("complete") == 1)

        # Formats and returns session IDs to the caller
        sessions = data.select("session").sort("session").to_series().to_list()
        return tuple(sessions)

    @property
    def sessions(self) -> tuple[str, ...]:
        """Returns all session IDs stored inside the manifest file.

        This property provides a tuple of all sessions, independent of the participating animal, that were recorded as
        part of the target project. Use the get_sessions() method to get the list of session tuples with filtering.
        """
        return self._get_filtered_sessions(animal=None, exclude_incomplete=False)

    def get_sessions(
        self,
        animal: str | int | None = None,
        exclude_incomplete: bool = True,
    ) -> tuple[str, ...]:
        """Returns requested session IDs based on selected filtering criteria.

        This method provides a tuple of sessions based on the specified filters. If no animal is specified, returns
        sessions for all animals in the project.

        Args:
            animal: An optional animal ID to filter the sessions. If set to None, the method returns sessions for all
                animals.
            exclude_incomplete: Determines whether to exclude sessions not marked as 'complete' from the output
                list.

        Returns:
            The tuple of session IDs matching the filter criteria.

        Raises:
            ValueError: If the specified animal is not found in the manifest file.
        """
        return self._get_filtered_sessions(
            animal=animal,
            exclude_incomplete=exclude_incomplete,
        )

    def get_session_info(self, session: str) -> pl.DataFrame:
        """Returns a Polars DataFrame that stores detailed information for the specified session.

        Since session IDs are unique, it is expected that filtering by session ID is enough to get the requested
        information.

        Args:
            session: The ID of the session for which to retrieve the data.

        Returns:
            A Polars DataFrame with the following columns: 'animal', 'date', 'notes', 'session', 'type', 'system',
            'complete', 'integrity', 'suite2p', 'behavior', 'video', 'archived'.
        """
        df = self._data
        df = df.filter(pl.col("session").eq(session))
        return df

    def get_animal_for_session(self, session: str) -> str:
        """Returns the animal ID associated with the specified session.

        Since session IDs are unique in the manifest, each session belongs to exactly one animal.

        Args:
            session: The ID of the session for which to retrieve the animal ID.

        Returns:
            The animal ID associated with the specified session, formatted as a string.

        Raises:
            ValueError: If the specified session is not found in the manifest file.
        """
        # Filters the data for the specified session
        df = self._data.filter(pl.col("session") == session)

        # Checks if the session exists
        if df.is_empty():
            message = f"Session ID '{session}' not found in the project manifest. Available sessions: {self.sessions}."
            console.error(message=message, error=ValueError)

        # Extracts the animal ID
        animal_id = df.select("animal").item()

        # Returns the animal ID with the appropriate type
        return str(animal_id)

    def get_system_for_session(self, session: str) -> str:
        """Returns the data acquisition system associated with the specified session.

        Args:
            session: The ID of the session for which to retrieve the data acquisition system.

        Returns:
            The data acquisition system associated with the specified session.

        Raises:
            ValueError: If the specified session is not found in the manifest file.
        """
        # Filters the data for the specified session
        df = self._data.filter(pl.col("session") == session)

        # Checks if the session exists
        if df.is_empty():
            message = f"Session ID '{session}' not found in the project manifest. Available sessions: {self.sessions}."
            console.error(message=message, error=ValueError)

        # Extracts and returns the acquisition system used to acquire the session
        return str(df.select("system").item())

    @property
    def data(self) -> pl.DataFrame:
        """Returns Polars DataFrame object wrapped by the class instance."""
        return self._data
