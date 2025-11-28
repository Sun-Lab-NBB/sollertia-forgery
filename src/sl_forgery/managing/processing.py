"""This module provides the assets for managing Sun lab session data stored on remote compute servers and
generating snapshots of the lab's research project's states. Assets from this module form the foundation for all
other data processing and analysis pipelines available from this library.
"""

from pathlib import Path
from datetime import datetime

import pytz
import polars as pl
from filelock import FileLock
from sl_shared_assets import (
    SessionData,
    SessionTypes,
    ProcessingTracker,
    RunTrainingDescriptor,
    LickTrainingDescriptor,
    WindowCheckingDescriptor,
    MesoscopeExperimentDescriptor,
    delete_directory,
    transfer_directory,
    calculate_directory_checksum,
)
from ataraxis_base_utilities import LogLevel, console

from ..server import ManagingTrackers, ProcessingTrackers


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

    def get_session_info(self, session: str) -> pl.DataFrame:
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


def transfer_session(
    source_path: Path,
    destination_path: Path | None = None,
    *,
    remove_source: bool = False,
) -> None:
    """Transfers the session's data from source to destination or deletes the session's data directory.

    Notes:
        If the destination path is None and remove_source is True, this function deletes the source directory instead
        of transferring it.

    Args:
        source_path: The path to the source session's data directory to be transferred.
        destination_path: The path to the destination directory where to transfer the session's data.
        remove_source: Determines whether to delete the source session directory after completing the transfer. If the
            destination path is not provided, this command deletes the source session directory without transferring.
    """
    # If the destination path is None and remove_source is True, deletes the source directory.
    if destination_path is None and remove_source:
        console.echo(
            message=f"Deleting the session directory at '{source_path}'...",
            level=LogLevel.INFO,
        )
        delete_directory(directory_path=source_path)
        console.echo(
            message=f"Session directory at '{source_path}': Deleted.",
            level=LogLevel.SUCCESS,
        )
        return

    console.echo(
        message=f"Transferring the session from '{source_path}' to '{destination_path}'...",
        level=LogLevel.INFO,
    )

    # Transfers the session data to the destination.
    transfer_directory(
        source=source_path,
        destination=destination_path,
        num_threads=0,
        verify_integrity=False,
        remove_source=remove_source,
        progress=True,
    )

    console.echo(
        message=f"Session transferred to '{destination_path}'.",
        level=LogLevel.SUCCESS,
    )


def resolve_checksum(
    session_path: Path,
    job_id: str,
    *,
    regenerate_checksum: bool = False,
) -> None:
    """Generates the checksum of the session's raw_data directory and either compares it against the checksum stored in
    the ax_checksum.txt file or overwrites the checksum stored in the file.

    Notes:
        Primarily, this function is used to verify the integrity of the session's data before running unsupervised data
        processing pipelines.

    Args:
        session_path: The path to the root data directory of the session to be processed.
        job_id: The unique identifier of this processing job.
        regenerate_checksum: Determines whether to update the checksum stored in the ax_checksum.txt file instead of
            verifying its integrity.
    """
    # Loads session data layout.
    session_data = SessionData.load(session_path=session_path)

    # Initializes the ProcessingTracker instance.
    tracker = ProcessingTracker(
        file_path=session_data.tracking_data.tracking_data_path.joinpath(ManagingTrackers.CHECKSUM)
    )

    # Marks the job as running.
    tracker.start_job(job_id=job_id)
    try:
        console.echo(
            message=f"Resolving the data integrity checksum for the session '{session_data.session_name}'...",
            level=LogLevel.INFO,
        )

        # Regenerates the checksum for the raw_data directory. Note, if the 'save_checksum' flag is True, this
        # guarantees that the check below succeeds as the function replaces the checksum in the ax_checksum.txt file
        # with the newly calculated value.
        calculated_checksum = calculate_directory_checksum(
            directory=session_data.raw_data.raw_data_path, progress=True, save_checksum=regenerate_checksum
        )

        # Loads the checksum stored inside the ax_checksum.txt file.
        with session_data.raw_data.checksum_path.open() as f:
            stored_checksum = f.read().strip()

        # If the two checksums do not match, this indicates data corruption.
        if stored_checksum != calculated_checksum:
            tracker.fail_job(job_id=job_id)
            console.echo(
                message=f"Session '{session_data.session_name}' raw data integrity: Compromised.", level=LogLevel.ERROR
            )

        else:
            # Sets the tracker to indicate that the runtime completed successfully.
            tracker.complete_job(job_id=job_id)
            console.echo(
                message=f"Session '{session_data.session_name}' raw data integrity: Verified.", level=LogLevel.SUCCESS
            )

    except Exception:
        # If the code reaches this section, this means that the runtime encountered an error.
        tracker.fail_job(job_id=job_id)
        raise  # Re-raises the exception to log the error message.


def generate_project_manifest(
    project_directory: Path,
    job_id: str,
) -> None:
    """Builds and saves the project manifest .feather file under the target project's root directory.

    Notes:
        The manifest file is primarily used to capture and move project's state information between machines, typically
        in the context of working with data stored on a remote compute server or cluster.

    Args:
        project_directory: The path to the processed project's root directory.
        job_id: The unique identifier of this processing job.
    """
    if not project_directory.exists():
        message = (
            f"Unable to generate the project manifest file for the '{project_directory.stem}; project. "
            f"The specified project directory does not exist."
        )
        console.error(message=message, error=FileNotFoundError)

    # Finds the root directories for all project's sessions.
    session_directories = [directory.parents[1] for directory in project_directory.rglob("session_data.yaml")]

    if len(session_directories) == 0:
        message = (
            f"Unable to generate the project manifest file for the requested project {project_directory.stem}. The "
            f"project directory does not contain any session data. To generate the manifest file, the project must "
            f"contain the data for at least one session."
        )
        console.error(message=message, error=FileNotFoundError)

    # Pre-creates the 'manifest' dictionary structure.
    manifest: dict[str, list[str | bool | datetime | int]] = {
        "animal": [],  # Animal IDs.
        "session": [],  # Session names.
        "date": [],  # Session names stored as timezone-aware date-time objects in EST.
        "type": [],  # Session types (e.g., mesoscope experiment, run training, etc.).
        "system": [],  # The acquisition system used to acquire the session (e.g., mesoscope-vr, etc.).
        "notes": [],  # The experimenter notes about the session.
        # Determines whether the session's data is complete and ready for unsupervised processing.
        "complete": [],
        # Determines whether the session's data integrity has been verified.
        "integrity": [],
        # Determines whether the session has been processed with the single-day s2p pipeline.
        "suite2p": [],
        # Determines whether the session has been processed with the behavior extraction pipeline.
        "behavior": [],
        # Determines whether the session has been processed with the DeepLabCut (video tracking) pipeline.
        "video": [],
    }

    # Resolves the path to the manifest .feather file to be created and the .lock file used to ensure only a single
    # process can be working on the manifest file at the same time.
    manifest_path = project_directory.joinpath(f"{project_directory.stem}_manifest.feather")
    manifest_lock = manifest_path.with_suffix(manifest_path.suffix + ".lock")

    # Also instantiates the processing tracker for the manifest file in the same directory.
    runtime_tracker = ProcessingTracker(file_path=project_directory.joinpath(ManagingTrackers.MANIFEST))

    # Acquires the lock file, ensuring only this specific process can work with the manifest data.
    lock = FileLock(str(manifest_lock))
    with lock.acquire(timeout=20.0):
        # Initializes the tracker with the job and marks it as running.
        runtime_tracker.initialize_jobs(job_ids=[job_id])
        runtime_tracker.start_job(job_id=job_id)
        try:
            # Loops over each session of every animal in the project and extracts session ID information and
            # information about which processing steps have been successfully applied to the session.
            for directory in session_directories:
                # Skips processing directories without files (sessions with empty raw_data directories).
                if not any(directory.joinpath("raw_data").glob("*")):
                    continue

                # Instantiates the SessionData instance to resolve the paths to all session's data files and locations.
                session_data = SessionData.load(session_path=directory)

                # Extracts ID and data path information from the SessionData instance.
                manifest["animal"].append(session_data.animal_id)
                manifest["session"].append(session_data.session_name)
                manifest["type"].append(session_data.session_type)
                manifest["system"].append(session_data.acquisition_system)

                # Parses session name into the date-time object to simplify working with date-time data in the future.
                date_time_components = session_data.session_name.split("-")
                date_time = datetime(
                    year=int(date_time_components[0]),
                    month=int(date_time_components[1]),
                    day=int(date_time_components[2]),
                    hour=int(date_time_components[3]),
                    minute=int(date_time_components[4]),
                    second=int(date_time_components[5]),
                    microsecond=int(date_time_components[6]),
                    tzinfo=pytz.UTC,
                )

                # Converts from UTC to EST / EDT for user convenience.
                eastern = pytz.timezone("America/New_York")
                date_time = date_time.astimezone(eastern)
                manifest["date"].append(date_time)

                # Depending on the session type, instantiates the appropriate descriptor instance and uses it to read
                # the experimenter notes and completeness status.
                is_complete: bool
                if session_data.session_type == SessionTypes.LICK_TRAINING:
                    descriptor: LickTrainingDescriptor = LickTrainingDescriptor.from_yaml(
                        file_path=session_data.raw_data.session_descriptor_path
                    )
                    manifest["notes"].append(descriptor.experimenter_notes)
                    is_complete = not descriptor.incomplete
                elif session_data.session_type == SessionTypes.RUN_TRAINING:
                    descriptor: RunTrainingDescriptor = RunTrainingDescriptor.from_yaml(
                        file_path=session_data.raw_data.session_descriptor_path
                    )
                    manifest["notes"].append(descriptor.experimenter_notes)
                    is_complete = not descriptor.incomplete
                elif session_data.session_type == SessionTypes.MESOSCOPE_EXPERIMENT:
                    descriptor: MesoscopeExperimentDescriptor = MesoscopeExperimentDescriptor.from_yaml(
                        file_path=session_data.raw_data.session_descriptor_path
                    )
                    manifest["notes"].append(descriptor.experimenter_notes)
                    is_complete = not descriptor.incomplete
                elif session_data.session_type == SessionTypes.WINDOW_CHECKING:
                    # sl-experiment version 3.0.0 added session descriptors to Window Checking runtimes. Since the file
                    # does not exist in prior versions, this section is written to statically handle the discrepancy.
                    try:
                        descriptor: WindowCheckingDescriptor = WindowCheckingDescriptor.from_yaml(
                            file_path=session_data.raw_data.session_descriptor_path
                        )
                        manifest["notes"].append(descriptor.experimenter_notes)
                        is_complete = not descriptor.incomplete
                    except Exception:
                        manifest["notes"].append("N/A")
                        is_complete = False
                else:
                    # Raises an error if an unsupported session type is encountered.
                    message = (
                        f"Unsupported session type '{session_data.session_type}' encountered for session "
                        f"'{directory.stem}' when generating the manifest file for the project "
                        f"{project_directory.stem}. Currently, only the following session types are supported: "
                        f"{tuple(SessionTypes)}."
                    )
                    console.error(message=message, error=ValueError)
                    # Fallback to appease mypy, should not be reachable
                    raise ValueError(message)  # noqa: TRY301

                # Marks the session as complete based on the descriptor's incomplete field.
                manifest["complete"].append(is_complete)

                # Data integrity verification status.
                tracker = ProcessingTracker(
                    file_path=session_data.tracking_data.tracking_data_path.joinpath(ManagingTrackers.CHECKSUM)
                )
                manifest["integrity"].append(tracker.complete)

                # If the session is incomplete or unverified, marks all processing steps as FALSE, as automatic
                # processing is disabled for incomplete sessions and, therefore, it could not have been processed.
                if not manifest["complete"][-1] or not manifest["integrity"][-1]:
                    manifest["suite2p"].append(False)
                    manifest["behavior"].append(False)
                    manifest["video"].append(False)
                    continue  # Cycles to the next session

                # Suite2p (single-day) processing status.
                tracker = ProcessingTracker(
                    file_path=session_data.tracking_data.tracking_data_path.joinpath(ProcessingTrackers.SUITE2P)
                )
                manifest["suite2p"].append(tracker.complete)

                # Behavior data processing status.
                tracker = ProcessingTracker(
                    file_path=session_data.tracking_data.tracking_data_path.joinpath(ProcessingTrackers.BEHAVIOR)
                )
                manifest["behavior"].append(tracker.complete)

                # DeepLabCut (video) processing status.
                tracker = ProcessingTracker(
                    file_path=session_data.tracking_data.tracking_data_path.joinpath(ProcessingTrackers.VIDEO)
                )
                manifest["video"].append(tracker.complete)

            # If all animal IDs are integer-convertible, stores them as numbers to promote proper sorting.
            # Otherwise, stores them as strings. The latter options are primarily kept for compatibility with Tyche
            # data.
            animal_type: type[pl.UInt64 | pl.String]
            if all(str(animal).isdigit() for animal in manifest["animal"]):
                # Converts all strings to integers.
                manifest["animal"] = [int(animal) for animal in manifest["animal"]]
                animal_type = pl.UInt64  # Uint64 for future proofing
            else:
                animal_type = pl.String

            # Converts the manifest dictionary to a Polars Dataframe.
            schema = {
                "animal": animal_type,
                "date": pl.Datetime,
                "session": pl.String,
                "type": pl.String,
                "system": pl.String,
                "notes": pl.String,
                "complete": pl.UInt8,
                "integrity": pl.UInt8,
                "suite2p": pl.UInt8,
                "behavior": pl.UInt8,
                "video": pl.UInt8,
            }
            df = pl.DataFrame(manifest, schema=schema, strict=False)

            # Sorts the DataFrame by animal and then session. Since animal IDs are monotonically increasing according to
            # Sun lab standards and session 'names' are based on acquisition timestamps, the sort order is
            # chronological.
            sorted_df = df.sort(["animal", "session"])

            # Saves the generated manifest to the project-specific manifest .feather file for further processing.
            sorted_df.write_ipc(file=manifest_path, compression="lz4")

            # The processing is now complete.
            runtime_tracker.complete_job(job_id=job_id)

        except Exception:
            # If the code reaches this section, this means that the runtime encountered an error.
            runtime_tracker.fail_job(job_id=job_id)
            raise  # Re-raises the exception to log the error message.
