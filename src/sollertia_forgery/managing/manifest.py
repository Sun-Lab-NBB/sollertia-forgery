"""Provides assets for generating, visualizing, and querying the project manifest .feather file that captures the
snapshot of a project's state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from datetime import UTC, datetime

import polars as pl
from natsort import natsorted
from filelock import FileLock
from ataraxis_base_utilities import console
from sollertia_shared_assets import (
    SessionTypes,
    ProcessingTrackers,
    iterate_sessions,
)
from ataraxis_data_structures import ProcessingTracker
from sollertia_shared_assets.registries import DESCRIPTOR_REGISTRY

from ..shared_assets import prepare_tracker

if TYPE_CHECKING:
    from pathlib import Path

    from sollertia_shared_assets import SessionData

MANIFEST_JOB_NAME: str = "manifest_generation"
"""The job name used to identify manifest generation jobs in processing trackers."""


def generate_project_manifest(project_directory: Path) -> None:
    """Builds and saves the project manifest .feather file under the target project's root directory.

    The manifest captures one row per session with its acquisition metadata and per-pipeline processing status. A
    file lock serializes concurrent writers, and the outcome is recorded on a manifest processing tracker in the
    project root.

    Args:
        project_directory: The path to the processed project's root directory.

    Raises:
        FileNotFoundError: If the project directory does not exist or contains no session data.
        ValueError: If an unsupported session type is encountered.
        Timeout: If the manifest .feather file lock cannot be acquired within 20 seconds.
    """
    if not project_directory.exists():
        message = (
            f"Unable to generate the project manifest file for the '{project_directory.stem}' project. "
            f"The specified project directory does not exist."
        )
        console.error(message=message, error=FileNotFoundError)

    # Discovers and loads every session under the project once. Both the multi-recording registry and the
    # per-session manifest rows consume this list, avoiding a second project-wide scan and redundant
    # SessionData loads.
    sessions: list[SessionData] = list(iterate_sessions(root_path=project_directory))

    if not sessions:
        message = (
            f"Unable to generate the project manifest file for the '{project_directory.stem}' project. The "
            f"project directory does not contain any session data. To generate the manifest file, the project must "
            f"contain the data for at least one session."
        )
        console.error(message=message, error=FileNotFoundError)

    # Resolves the path to the manifest .feather file to be created and the .lock file used to ensure only a single
    # process can be working on the manifest file at the same time.
    manifest_path = project_directory.joinpath(f"{project_directory.stem}_manifest.feather")
    manifest_lock = manifest_path.with_suffix(manifest_path.suffix + ".lock")

    # Initializes the processing tracker in the project directory alongside the manifest output. Applies stale
    # entry detection so that foreign or outdated job entries are reset before the new job is registered.
    tracker = ProcessingTracker(file_path=project_directory.joinpath(ProcessingTrackers.MANIFEST))
    jobs = [(MANIFEST_JOB_NAME, project_directory.stem)]
    prepare_tracker(tracker=tracker, jobs=jobs, universe=jobs)
    job_id = ProcessingTracker.generate_job_id(job_name=MANIFEST_JOB_NAME, specifier=project_directory.stem)

    # Acquires the lock file, ensuring only this specific process can work with the manifest data.
    lock = FileLock(str(manifest_lock))
    with lock.acquire(timeout=20.0):
        tracker.start_job(job_id=job_id)
        try:
            manifest: dict[str, list] = {
                # Animal IDs.
                "animal": [],
                # Session names.
                "session": [],
                # Session acquisition time as a timezone-aware datetime in the host machine's local time.
                "date": [],
                # Session types (e.g., mesoscope experiment, run training, etc.).
                "type": [],
                # The acquisition system used to acquire the session (e.g., mesoscope-vr, etc.).
                "system": [],
                # The experimenter notes about the session.
                "notes": [],
                # Determines whether the session's data is complete and ready for unsupervised processing.
                "complete": [],
                # Determines whether the session's data integrity has been verified.
                "integrity": [],
                # Determines whether the session has been processed with the two-photon (cindra) processing pipeline.
                "two_photon": [],
                # Determines whether the session has been processed with the runtime processing pipeline.
                "runtime": [],
                # Determines whether the session has been processed with the DeepLabCut (video tracking) pipeline.
                "video": [],
                # Stores the cindra multi-recording dataset names the session belongs to (empty list if none).
                "multi_recording_datasets": [],
                # Stores per-dataset completion status, aligned by index with multi_recording_datasets.
                "multi_recording_complete": [],
            }

            # Builds the cindra multi-recording dataset completion registry from the canonical
            # ``cindra/multi_recording`` subdirectory of every session, rather than rescanning the whole
            # project. The tracker only lives on the main recording, so the registry later resolves
            # completion status for datasets discovered on non-main sessions.
            multi_recording_registry: dict[str, bool] = {}
            for session_data in sessions:
                multi_recording_root = session_data.processed_data.cindra_multi_recording_path
                if not multi_recording_root.is_dir():
                    continue
                for dataset_dir in multi_recording_root.iterdir():
                    if not dataset_dir.is_dir():
                        continue
                    tracker_path = dataset_dir.joinpath(ProcessingTrackers.CINDRA_MULTI_RECORDING)
                    if not tracker_path.is_file():
                        continue
                    # The forging pipeline writes the dataset directory as ``{animal_id}_{base_name}`` for
                    # collision avoidance when batching multiple animals under one forged dataset. The manifest
                    # surfaces the unqualified base name, so the animal_id prefix is stripped here.
                    dataset_name = _strip_animal_prefix(
                        qualified_name=dataset_dir.name, animal_id=str(session_data.animal_id)
                    )
                    multi_recording_registry[dataset_name] = ProcessingTracker(file_path=tracker_path).complete

            # Loops over each session of every animal in the project and extracts session ID information and
            # information about which processing steps have been successfully applied to the session.
            for session_data in sessions:
                # Skips processing directories without files (sessions with empty raw_data directories).
                if not any(session_data.raw_data_path.glob("*")):
                    continue

                # Extracts ID and data path information from the SessionData instance.
                manifest["animal"].append(session_data.animal_id)
                manifest["session"].append(session_data.session_name)
                manifest["type"].append(session_data.session_type)
                manifest["system"].append(session_data.acquisition_system)

                # Parses the session name (a UTC timestamp) into a timezone-aware datetime in the host machine's
                # local time.
                date_time_components = session_data.session_name.split("-")
                date_time = datetime(
                    year=int(date_time_components[0]),
                    month=int(date_time_components[1]),
                    day=int(date_time_components[2]),
                    hour=int(date_time_components[3]),
                    minute=int(date_time_components[4]),
                    second=int(date_time_components[5]),
                    microsecond=int(date_time_components[6]),
                    tzinfo=UTC,
                ).astimezone()
                manifest["date"].append(date_time)

                # Loads the session descriptor to extract experimenter notes and completeness status. Some legacy
                # Window Checking sessions lack descriptors, so a missing file is handled gracefully for that session
                # type only.
                descriptor_path = session_data.raw_data.session_descriptor_path
                descriptor_class = DESCRIPTOR_REGISTRY.get(SessionTypes(session_data.session_type))
                if descriptor_class is None:
                    message = (
                        f"Unable to generate the manifest file for the '{project_directory.stem}' project. An "
                        f"unsupported session type '{session_data.session_type}' was encountered for session "
                        f"'{session_data.session_name}'. Currently, only the following session types are supported: "
                        f"{tuple(SessionTypes)}."
                    )
                    console.error(message=message, error=ValueError)

                try:
                    # DESCRIPTOR_REGISTRY types its values as the base YamlConfig, so the shared descriptor fields
                    # the manifest reads (every registered descriptor declares them) need an attribute-defined ignore.
                    descriptor = descriptor_class.from_yaml(file_path=descriptor_path)
                    is_complete = not descriptor.incomplete  # type: ignore[attr-defined]
                    manifest["notes"].append(descriptor.experimenter_notes)  # type: ignore[attr-defined]
                except Exception:
                    if session_data.session_type != SessionTypes.WINDOW_CHECKING:
                        raise
                    is_complete = False
                    manifest["notes"].append("N/A")

                manifest["complete"].append(is_complete)

                # Resolves data integrity verification status from the canonical checksum tracker path.
                checksum_tracker = _load_tracker_if_exists(tracker_path=session_data.raw_data.checksum_tracker_path)
                is_verified = checksum_tracker.complete if checksum_tracker is not None else False
                manifest["integrity"].append(is_verified)

                # If the session is incomplete or unverified, marks all processing steps as FALSE, as automatic
                # processing is disabled for incomplete sessions and, therefore, it could not have been processed.
                if not is_complete or not is_verified:
                    manifest["two_photon"].append(False)
                    manifest["runtime"].append(False)
                    manifest["video"].append(False)
                    manifest["multi_recording_datasets"].append([])
                    manifest["multi_recording_complete"].append([])
                    continue

                # Resolves two-photon, runtime, and DeepLabCut (video) processing status from canonical tracker paths
                # exposed by SessionData.
                two_photon_tracker = _load_tracker_if_exists(
                    tracker_path=session_data.processed_data.two_photon_tracker_path
                )
                manifest["two_photon"].append(two_photon_tracker.complete if two_photon_tracker is not None else False)

                runtime_tracker = _load_tracker_if_exists(
                    tracker_path=session_data.processed_data.runtime_tracker_path
                )
                manifest["runtime"].append(runtime_tracker.complete if runtime_tracker is not None else False)

                video_tracker = _load_tracker_if_exists(tracker_path=session_data.processed_data.video_tracker_path)
                manifest["video"].append(video_tracker.complete if video_tracker is not None else False)

                # Resolves multi-recording dataset membership by enumerating the session's
                # ``cindra/multi_recording`` subdirectories, then looks up each dataset's completion status
                # from the project-wide registry built above.
                multi_recording_root = session_data.processed_data.cindra_multi_recording_path
                session_datasets: list[str] = []
                session_dataset_complete: list[bool] = []
                if multi_recording_root.is_dir():
                    for dataset_dir in natsorted(multi_recording_root.iterdir()):
                        if not dataset_dir.is_dir():
                            continue
                        dataset_name = _strip_animal_prefix(
                            qualified_name=dataset_dir.name, animal_id=str(session_data.animal_id)
                        )
                        session_datasets.append(dataset_name)
                        session_dataset_complete.append(multi_recording_registry.get(dataset_name, False))
                manifest["multi_recording_datasets"].append(session_datasets)
                manifest["multi_recording_complete"].append(session_dataset_complete)

            # Converts animal IDs from strings to integers for proper numeric sorting.
            manifest["animal"] = [int(animal) for animal in manifest["animal"]]

            # Converts the manifest dictionary to a Polars DataFrame.
            schema: dict[str, pl.datatypes.classes.DataTypeClass | pl.DataType] = {
                "animal": pl.UInt64,
                "date": pl.Datetime,
                "session": pl.String,
                "type": pl.String,
                "system": pl.String,
                "notes": pl.String,
                "complete": pl.UInt8,
                "integrity": pl.UInt8,
                "two_photon": pl.UInt8,
                "runtime": pl.UInt8,
                "video": pl.UInt8,
                "multi_recording_datasets": pl.List(pl.String),
                "multi_recording_complete": pl.List(pl.UInt8),
            }
            manifest_frame = pl.DataFrame(data=manifest, schema=schema, strict=False)

            # Sorts the DataFrame by animal, then session. Animal IDs are monotonically increasing per Sollertia
            # standards and session names are acquisition timestamps, so rows are grouped by animal and ordered
            # chronologically within each animal group.
            sorted_manifest = manifest_frame.sort(by=["animal", "session"])

            # Saves the generated manifest to the project-specific uncompressed .feather file to allow
            # memory-mapped reads.
            sorted_manifest.write_ipc(file=manifest_path, compression="uncompressed")

            tracker.complete_job(job_id=job_id)

        except Exception:
            # Records the manifest job as failed before re-raising so the tracker reflects the aborted run.
            tracker.fail_job(job_id=job_id)
            raise


class ProjectManifest:
    """Provides methods for visualizing and working with the data stored inside the managed project manifest .feather
    file.

    Notes:
        This class provides the entry-point API for working with Sollertia research project data. It is used by most
        data processing and dataset forging pipelines to work with the processed project's data.

    Args:
        manifest_file: The path to the .feather manifest file that stores the snapshot of the target project's state.

    Attributes:
        _data: The Polars DataFrame that stores the snapshot of the project's state.
    """

    def __init__(self, manifest_file: Path) -> None:
        self._data: pl.DataFrame = pl.read_ipc(source=manifest_file, memory_map=True)

    def __repr__(self) -> str:
        """Returns a string representation of the ProjectManifest instance."""
        return f"ProjectManifest(sessions={self._data.height})"

    def print_data(self) -> None:
        """Prints the entire contents of the manifest file to the terminal."""
        with pl.Config(
            set_tbl_rows=-1,  # Displays all rows (-1 means unlimited)
            set_tbl_cols=-1,  # Displays all columns (-1 means unlimited)
            set_tbl_hide_column_data_types=True,
            set_tbl_cell_alignment="LEFT",
            set_tbl_width_chars=250,
            set_fmt_str_lengths=600,  # Allows longer strings to display properly (default is 30)
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
            "two_photon",
            "runtime",
            "video",
            "multi_recording_datasets",
            "multi_recording_complete",
        ]

        data_frame = self._data.select(summary_cols)

        # Optionally filters the data for the target animal.
        if animal is not None:
            data_frame = data_frame.filter(pl.col("animal") == int(animal))

        # Ensures the data displays properly.
        with pl.Config(
            set_tbl_rows=-1,
            set_tbl_cols=-1,
            set_tbl_width_chars=250,
            set_tbl_hide_column_data_types=True,
            set_tbl_cell_alignment="CENTER",
        ):
            console.echo(message=str(data_frame), raw=True)

    def print_notes(self, animal: int | None = None) -> None:
        """Prints the animal ID, session date, session ID, session type, acquisition system, and experimenter notes
        data for each project's session to the terminal.

        This data view is optimized for determining what data acquisition sessions have been carried out and checking
        the outcomes of each session recorded in the experimenter notes.

        Args:
            animal: The unique identifier of the animal for which to display the data. If provided, this method only
                displays the data for that animal. Otherwise, it displays the data for all animals.
        """
        # Pre-selects the columns to display.
        data_frame = self._data.select(["animal", "date", "session", "type", "system", "notes"])

        # Optionally filters the data for the target animal.
        if animal is not None:
            data_frame = data_frame.filter(pl.col("animal") == int(animal))

        # Prints the extracted data.
        with pl.Config(
            set_tbl_rows=-1,
            set_tbl_cols=-1,
            set_tbl_hide_column_data_types=True,
            set_tbl_cell_alignment="LEFT",
            set_tbl_width_chars=170,  # Wider columns for notes
            set_fmt_str_lengths=2000,  # Allows very long strings for notes
        ):
            console.echo(message=str(data_frame), raw=True)

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
            'session', 'type', 'system', 'notes', 'complete', 'integrity', 'two_photon', 'runtime', 'video',
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
        data_frame = self._data.filter(pl.col("session") == session)

        if data_frame.is_empty():
            message = (
                f"Unable to look up the participating animal using session ID '{session}'. The session is not "
                f"found in the project manifest. Available sessions: "
                f"{self.get_sessions(animal=None, exclude_incomplete=False)}."
            )
            console.error(message=message, error=ValueError)

        return int(data_frame.select("animal").item())

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
        data_frame = self._data.filter(pl.col("session") == session)

        if data_frame.is_empty():
            message = (
                f"Unable to look up the acquisition system using session ID '{session}'. The session is not "
                f"found in the project manifest. Available sessions: "
                f"{self.get_sessions(animal=None, exclude_incomplete=False)}."
            )
            console.error(message=message, error=ValueError)

        return str(data_frame.select("system").item())

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
        two_photon_count = int(data.filter(pl.col("two_photon") == 1).height)
        runtime_count = int(data.filter(pl.col("runtime") == 1).height)
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
                    "datasets": natsorted(datasets, key=lambda dataset: dataset["name"]),
                }

        return {
            "total_sessions": total_rows,
            "total_animals": len(self.animals),
            "animals": list(self.animals),
            "session_types": session_types,
            "acquisition_systems": acquisition_systems,
            "complete_count": complete_count,
            "integrity_verified_count": integrity_count,
            "two_photon_processed_count": two_photon_count,
            "runtime_processed_count": runtime_count,
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
                message = (
                    f"Unable to filter sessions using animal ID '{animal}'. The animal is not found in the "
                    f"project manifest. Available animals: {self.animals}."
                )
                console.error(message=message, error=ValueError)

            data = data.filter(pl.col("animal") == animal)

        # Optionally filters out incomplete sessions.
        if exclude_incomplete:
            data = data.filter(pl.col("complete") == 1)

        # Formats and returns session IDs to the caller.
        sessions = data.select("session").sort("session").to_series().to_list()
        return tuple(sessions)


def _strip_animal_prefix(qualified_name: str, animal_id: str) -> str:
    """Strips the ``{animal_id}_`` prefix from a cindra multi-recording dataset directory name.

    The slf forging pipeline prepends the animal identifier to the dataset name to produce collision-free
    output directories when batching multiple animals with the same forged dataset. This helper reverses that
    qualification so manifest consumers see the logical base name instead of the filesystem-qualified name.

    Args:
        qualified_name: The on-disk directory name produced by the forging pipeline.
        animal_id: The animal identifier prepended to qualify the on-disk dataset directory name.

    Returns:
        The dataset name with the ``{animal_id}_`` prefix removed when present, or the input unchanged when
        the prefix is absent.
    """
    prefix = f"{animal_id}_"
    if qualified_name.startswith(prefix):
        return qualified_name[len(prefix) :]
    return qualified_name


def _load_tracker_if_exists(tracker_path: Path) -> ProcessingTracker | None:
    """Returns a ProcessingTracker bound to the target path when it exists, or None otherwise.

    Args:
        tracker_path: The canonical path to the processing tracker YAML file.

    Returns:
        A ProcessingTracker instance when the file is present on disk, or None when it is missing.
    """
    if not tracker_path.is_file():
        return None
    return ProcessingTracker(file_path=tracker_path)
