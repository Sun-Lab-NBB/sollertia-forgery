"""Provides shared metadata assets used across sollertia-forgery: the BehaviorDataFiles and StimulusMode enumerations
and the ProjectManifest class for visualizing and querying project manifest .feather files.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

import polars as pl
from ataraxis_base_utilities import console

if TYPE_CHECKING:
    from pathlib import Path


class BehaviorDataFiles(StrEnum):
    """Enumerates the canonical filenames of the behavior feather files written into the session's
    ``processed_data/behavior_data`` directory by the sollertia-forgery processing pipeline and consumed by the
    sollertia-forgery dataset forging pipeline.

    Notes:
        This is the contract between the ``processing`` and ``forging`` subpackages: the processing pipeline writes
        each file under these exact names, and the forging pipeline reads them back by the same names. All entries
        are forgery-internal and must not be referenced from outside the library.
    """

    ENCODER = "encoder_data.feather"
    """The encoder module feather holding the traveled-distance time series derived from the running-wheel encoder."""
    VALVE = "valve_data.feather"
    """The water valve module feather holding water-dispensing events and cumulative dispensed volume."""
    GAS_PUFF = "gas_puff_data.feather"
    """The gas puff valve module feather holding aversive-stimulus dispensing events."""
    LICK = "lick_data.feather"
    """The lick module feather holding thresholded lick events from the capacitive sensor."""
    BRAKE = "brake_data.feather"
    """The brake module feather holding the instantaneous brake torque applied to the running wheel."""
    TORQUE = "torque_data.feather"
    """The torque module feather holding the instantaneous torque exerted on the running wheel by the animal."""
    SCREEN = "screen_data.feather"
    """The screen module feather holding VR screen on/off state transitions."""
    MESOSCOPE_FRAME = "mesoscope_frame_data.feather"
    """The TTL module feather holding mesoscope scan-frame pulse edges used to align fluorescence to behavior."""
    SYSTEM_STATE = "system_state_data.feather"
    """The runtime feather holding system-state code transitions extracted from the acquisition runtime's log
    archive."""
    RUNTIME_STATE = "runtime_state_data.feather"
    """The runtime feather holding experiment-state code transitions extracted from the acquisition runtime's log
    archive."""
    REINFORCING_GUIDANCE = "reinforcing_guidance_state_data.feather"
    """The runtime feather holding reinforcing guidance-state transitions (written only when reinforcing guidance
    events were recorded during the session)."""
    AVERSIVE_GUIDANCE = "aversive_guidance_state_data.feather"
    """The runtime feather holding aversive guidance-state transitions (written only when aversive guidance events
    were recorded during the session)."""
    VR_CUE = "vr_cue_data.feather"
    """The runtime feather holding VR wall-cue transitions along the corridor."""
    VR_TRIGGER_ZONE = "vr_trigger_zone_data.feather"
    """The runtime feather holding VR trigger-zone entry and exit events."""
    TRIAL = "trial_data.feather"
    """The runtime feather holding per-trial metadata (trial index, trial type, traveled distance at trial start)."""
    FACE_CAMERA_TIMESTAMPS = "face_camera_timestamps.feather"
    """The face-camera timestamp feather hardlinked from the ataraxis-video-system processed camera_timestamps output
    under its legacy sollertia-forgery name."""
    BODY_CAMERA_TIMESTAMPS = "body_camera_timestamps.feather"
    """The body-camera timestamp feather hardlinked from the ataraxis-video-system processed camera_timestamps output
    under its legacy sollertia-forgery name."""


class StimulusMode(StrEnum):
    """Defines the semantic meaning of the stimulus delivered when a trial's stimulus trigger zone fires.

    Notes:
        Projects each upstream trial-class type onto the analysis-relevant axis of "what does the animal experience
        when the trigger fires." Decoupled from TriggerType, which describes the activation
        mechanism rather than the resulting outcome. Analysis modules use this enum to gate semantically appropriate
        pipelines (e.g., reward-cell analysis runs only against trial types whose stimulus_mode is REWARD).
    """

    REWARD = "reward"
    """Indicates an appetitive stimulus (e.g., water delivery in a WaterRewardTrial) delivered when the trigger
    condition is met."""
    AVERSIVE = "aversive"
    """Indicates an aversive stimulus (e.g., gas puff in a GasPuffTrial) delivered when the trigger condition fails."""


class ProjectManifest:
    """Provides methods for visualizing and working with the data stored inside the managed project manifest .feather
    file.

    Notes:
        This class provides the entry-point API for working with Sollertia research project data. It is used by most
        data processing and analysis dataset formation pipelines to work with the processed project's data.

    Args:
        manifest_file: The path to the .feather manifest file that stores the snapshot of the target project's state.

    Attributes:
        _data: The Polars DataFrame that stores the snapshot of the project's state.
    """

    def __init__(self, manifest_file: Path) -> None:
        # Reads the data from the target manifest file into the class attribute.
        self._data: pl.DataFrame = pl.read_ipc(source=manifest_file, memory_map=True)

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
            console.echo(message=str(self._data), raw=True)

    def print_summary(self, animal: int | None = None) -> None:
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
            "cindra",
            "behavior",
            "video",
            "multi_recording_datasets",
            "multi_recording_complete",
        ]

        # Retrieves the data.
        df = self._data.select(summary_cols)

        # Optionally filters the data for the target animal.
        if animal is not None:
            df = df.filter(pl.col("animal") == int(animal))

        # Ensures the data displays properly.
        with pl.Config(
            set_tbl_rows=-1,
            set_tbl_cols=-1,
            set_tbl_width_chars=250,
            set_tbl_hide_column_data_types=True,
            set_tbl_cell_alignment="CENTER",
        ):
            console.echo(message=str(df), raw=True)

    def print_notes(self, animal: int | None = None) -> None:
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
            df = df.filter(pl.col("animal") == int(animal))

        # Prints the extracted data.
        with pl.Config(
            set_tbl_rows=-1,
            set_tbl_cols=-1,
            set_tbl_hide_column_data_types=True,
            set_tbl_cell_alignment="LEFT",
            set_tbl_width_chars=170,  # Wider columns for notes
            set_fmt_str_lengths=2000,  # Allows very long strings for notes
        ):
            console.echo(message=str(df), raw=True)

    @property
    def animals(self) -> tuple[int, ...]:
        """Returns the unique identifiers for each animal participating in the project."""
        return tuple(self._data.select("animal").unique().sort("animal").to_series().to_list())

    def get_sessions(
        self,
        animal: int | None = None,
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
            A Polars DataFrame containing all manifest columns for the specified session, including 'animal', 'date',
            'session', 'type', 'system', 'notes', 'complete', 'integrity', 'cindra', 'behavior', 'video',
            'multi_recording_datasets', and 'multi_recording_complete'.
        """
        return self._data.filter(pl.col("session").eq(session))

    def get_animal_for_session(self, session: str) -> int:
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

        # Extracts and returns the animal ID.
        return int(df.select("animal").item())

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

    def summarize(self) -> dict[str, Any]:
        """Returns a structured summary of the project manifest for programmatic consumption.

        Computes aggregate statistics across all sessions including per-pipeline completion counts and
        multi-recording dataset membership. Designed for MCP tool responses where a structured dictionary is
        more useful than a printed table.

        Returns:
            A dictionary containing ``total_sessions``, ``total_animals``, ``animals``, ``session_types``,
            ``acquisition_systems``, per-pipeline completion counts, ``multi_recording_datasets`` summary,
            ``columns``, and ``total_rows``.
        """
        data = self._data
        total_rows = data.height

        # Computes per-pipeline completion counts from the boolean (UInt8) status columns.
        complete_count = int(data.filter(pl.col("complete") == 1).height)
        integrity_count = int(data.filter(pl.col("integrity") == 1).height)
        cindra_count = int(data.filter(pl.col("cindra") == 1).height)
        behavior_count = int(data.filter(pl.col("behavior") == 1).height)
        video_count = int(data.filter(pl.col("video") == 1).height)

        # Computes session type distribution.
        session_types: dict[str, int] = {}
        for row in data.select("type").to_series().to_list():
            session_types[str(row)] = session_types.get(str(row), 0) + 1

        # Computes acquisition system distribution.
        acquisition_systems: dict[str, int] = {}
        for row in data.select("system").to_series().to_list():
            acquisition_systems[str(row)] = acquisition_systems.get(str(row), 0) + 1

        # Builds the multi-recording dataset summary by exploding the list columns and grouping by dataset
        # name. Each dataset entry reports the number of sessions it spans and its completion status.
        dataset_summary: dict[str, Any] = {"total_datasets": 0, "datasets": []}
        if "multi_recording_datasets" in data.columns and "multi_recording_complete" in data.columns:
            # Filters to rows that have at least one dataset entry, then explodes both list columns in
            # parallel so each row represents a single (session, dataset, complete) triple.
            has_datasets = data.filter(pl.col("multi_recording_datasets").list.len() > 0)
            if has_datasets.height > 0:
                exploded = has_datasets.select(
                    "session", "multi_recording_datasets", "multi_recording_complete"
                ).explode("multi_recording_datasets", "multi_recording_complete")

                # Groups by dataset name to compute per-dataset session count and completion status.
                grouped = exploded.group_by("multi_recording_datasets").agg(
                    pl.col("session").count().alias("session_count"),
                    pl.col("multi_recording_complete").max().alias("complete"),
                )

                datasets: list[dict[str, Any]] = [
                    {
                        "name": row["multi_recording_datasets"],
                        "session_count": int(row["session_count"]),
                        "complete": bool(row["complete"]),
                    }
                    for row in grouped.iter_rows(named=True)
                ]

                dataset_summary = {
                    "total_datasets": len(datasets),
                    "datasets": sorted(datasets, key=lambda d: d["name"]),
                }

        return {
            "total_sessions": total_rows,
            "total_animals": len(self.animals),
            "animals": list(self.animals),
            "session_types": session_types,
            "acquisition_systems": acquisition_systems,
            "complete_count": complete_count,
            "integrity_verified_count": integrity_count,
            "cindra_processed_count": cindra_count,
            "behavior_processed_count": behavior_count,
            "video_processed_count": video_count,
            "multi_recording_datasets": dataset_summary,
            "columns": data.columns,
            "total_rows": total_rows,
        }

    @property
    def data(self) -> pl.DataFrame:
        """Returns the Polars DataFrame instance that stores the managed manifest file's data."""
        return self._data

    def _get_filtered_sessions(
        self,
        animal: int | None = None,
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
