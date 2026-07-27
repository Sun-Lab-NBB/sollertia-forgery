"""Provides assets for generating, visualizing, and querying the project manifest .feather file that captures the
snapshot of a project's state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from datetime import UTC, datetime

import polars as pl
from natsort import natsorted
from filelock import FileLock
from ataraxis_base_utilities import LogLevel, console
from sollertia_shared_assets import (
    DatasetData,
    SessionData,
    SessionTypes,
    ProcessingTrackers,
    iterate_sessions,
)
from ataraxis_data_structures import ProcessingTracker
from sollertia_shared_assets.registries import DESCRIPTOR_REGISTRY

from ..shared_assets import (
    summarize_tracker,
    derive_tracker_status,
)

if TYPE_CHECKING:
    from pathlib import Path

MANIFEST_JOB_NAME: str = "manifest_generation"
"""The job name used to identify manifest generation jobs in processing trackers."""

PIPELINE_STATUS_COLUMNS: dict[str, str] = {
    "checksum": "integrity",
    "runtime": "runtime",
    "microcontroller": "microcontroller",
    "video": "video",
    "two_photon": "two_photon",
}
"""Maps each per-session pipeline to the manifest column that carries its rolled-up status. The checksum pipeline's
status lives under the ``integrity`` column, so the mapping's keys differ from its values."""

JOB_STRUCT: pl.Struct = pl.Struct(
    {
        "pipeline": pl.String,
        "job_id": pl.String,
        "job_name": pl.String,
        "specifier": pl.String,
        "status": pl.String,
        "executor_id": pl.String,
        "error_message": pl.String,
        "started_at": pl.UInt64,
        "completed_at": pl.UInt64,
    }
)
"""The element type of the manifest's ``jobs`` column. Mirrors the per-job entry that ``summarize_tracker`` emits,
which carries every ``ataraxis_data_structures.JobState`` field plus the ``job_id`` registry key, with a
``pipeline`` discriminator prepended. Exploding the column yields one row per tracked job across every pipeline of a
session."""

DATASET_STRUCT: pl.Struct = pl.Struct(
    {
        "name": pl.String,
        "complete": pl.Boolean,
    }
)
"""The element type of the manifest's ``datasets`` column. Each struct pairs the name of a forged dataset the
session belongs to with that dataset's forging completion status, so a session's dataset membership stays
self-describing within a single column."""


def project_manifest_path(project_directory: Path) -> Path:
    """Resolves the path to the project manifest .feather file under the target project's root directory.

    This is the single source of truth for the manifest filename, so both the manifest writer and any consumer that
    locates the manifest derive the same path.

    Args:
        project_directory: The path to the project's root directory.

    Returns:
        The path to the project manifest .feather file.
    """
    return project_directory.joinpath(f"{project_directory.stem}_manifest.feather")


def generate_project_manifest(project_directory: Path, *, display_progress: bool = False) -> None:
    """Builds and saves the project manifest .feather file under the target project's root directory.

    The manifest captures one row per session with its acquisition metadata and per-pipeline processing status. A
    file lock serializes concurrent writers, and the outcome is recorded on a manifest processing tracker in the
    project root.

    Args:
        project_directory: The path to the processed project's root directory.
        display_progress: Determines whether to emit a preamble message when generation starts and a completion
            message when the manifest is written. Generation is fast enough that no progress bar is displayed.

    Raises:
        FileNotFoundError: If the project directory does not exist, contains no session data, or contains a session
            without its descriptor file.
        ValueError: If an unsupported session type is encountered.
        Timeout: If the manifest .feather file lock cannot be acquired within 20 seconds.
    """
    if not project_directory.exists():
        message = (
            f"Unable to generate the project manifest file for the '{project_directory.stem}' project. "
            f"The specified project directory does not exist."
        )
        console.error(message=message, error=FileNotFoundError)

    if display_progress:
        console.echo(
            message=f"Generating the project manifest for the '{project_directory.stem}' project...",
            level=LogLevel.INFO,
        )

    # Discovers and loads every session under the project once, so the per-session manifest rows are built without
    # redundant SessionData loads.
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
    manifest_path = project_manifest_path(project_directory=project_directory)
    manifest_lock = manifest_path.with_suffix(manifest_path.suffix + ".lock")

    # Initializes the processing tracker in the project directory alongside the manifest output. Applies stale
    # entry detection so that foreign or outdated job entries are reset before the new job is registered.
    tracker = ProcessingTracker(file_path=project_directory.joinpath(ProcessingTrackers.MANIFEST))
    jobs = [(MANIFEST_JOB_NAME, project_directory.stem)]
    tracker.align_jobs(jobs=jobs, universe=jobs)
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
                # The session's location relative to the project root, as '<animal_id>/<session_name>'. Stored
                # relative rather than absolute so a manifest generated on one machine resolves against any data
                # root, which is what lets an orchestrator map a manifest row back to a directory to process.
                "session_path": [],
                # Session acquisition time as a timezone-aware UTC datetime, matching the UTC session name.
                "date": [],
                # The session type, a SessionTypes enumeration value.
                "type": [],
                # The acquisition system that recorded the session, an AcquisitionSystems enumeration value.
                "system": [],
                # The experimenter notes about the session.
                "notes": [],
                # Determines whether the session's data is complete and ready for unsupervised processing.
                "complete": [],
                # The rolled-up status label of the checksum (data-integrity) pipeline. A label rather than a
                # boolean, so a failed pipeline is distinguishable from one that has not started.
                "integrity": [],
                # The rolled-up status label of the two-photon (cindra) processing pipeline.
                "two_photon": [],
                # The rolled-up status label of the runtime processing pipeline.
                "runtime": [],
                # The rolled-up status label of the microcontroller processing pipeline.
                "microcontroller": [],
                # The rolled-up status label of the video (timestamp, tracking, motion energy) pipeline.
                "video": [],
                # The complete job registry of every per-session pipeline's tracker, one entry per job, carrying
                # per-job status, failure reasons, executor identifiers, and timing.
                "jobs": [],
                # Maps each pipeline to its tracker's location relative to the project root, so a consumer can
                # reset or inspect a tracker without re-deriving the session hierarchy.
                "tracker_paths": [],
                # The forged datasets the session belongs to, one struct per dataset pairing the dataset name with
                # its forging completion status. Empty list when the session joins no forged dataset.
                "datasets": [],
            }

            # Builds a map from each session to the forged datasets it belongs to. A forged dataset is a top-level
            # directory carrying a ``dataset.yaml`` marker under the project root. Its member sessions come from the
            # dataset hierarchy, and its completion is the roll-up of its forging tracker, shared by every session in
            # the dataset.
            session_datasets: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for dataset_directory in natsorted(project_directory.iterdir(), key=lambda path: path.name):
                if not dataset_directory.is_dir() or not dataset_directory.joinpath("dataset.yaml").is_file():
                    continue
                dataset = DatasetData.load(dataset_path=dataset_directory)
                forging_jobs = ProcessingTracker(
                    file_path=dataset_directory.joinpath(ProcessingTrackers.FORGING)
                ).snapshot()
                complete = bool(
                    forging_jobs
                    and derive_tracker_status(summary=summarize_tracker(jobs=forging_jobs)["summary"]) == "completed"
                )
                for dataset_session in dataset.sessions:
                    session_datasets.setdefault((dataset_session.animal, dataset_session.session), []).append(
                        {"name": dataset.name, "complete": complete}
                    )

            # Loops over each session of every animal in the project and extracts session ID information and
            # information about which processing steps have been successfully applied to the session.
            for session_data in sessions:
                # Skips sessions whose raw_data directory is empty. A fully acquired session carries its marker and
                # acquired data under raw_data, so an empty raw_data marks an aborted or not-yet-acquired session
                # that must not enter the manifest.
                if not any(session_data.raw_data_path.glob("*")):
                    continue

                for column, value in _build_session_row(
                    session_data=session_data, project_directory=project_directory
                ).items():
                    manifest[column].append(value)

                # Attaches the forged-dataset membership resolved above, keyed on the session's animal and name.
                manifest["datasets"].append(
                    session_datasets.get((str(session_data.animal_id), session_data.session_name), [])
                )

            # Converts animal IDs from strings to integers for proper numeric sorting.
            manifest["animal"] = [int(animal) for animal in manifest["animal"]]

            # Converts the manifest dictionary to a Polars DataFrame.
            schema: dict[str, pl.datatypes.classes.DataTypeClass | pl.DataType] = {
                "animal": pl.UInt64,
                "date": pl.Datetime,
                "session": pl.String,
                "session_path": pl.String,
                "type": pl.String,
                "system": pl.String,
                "notes": pl.String,
                "complete": pl.UInt8,
                "integrity": pl.String,
                "two_photon": pl.String,
                "runtime": pl.String,
                "microcontroller": pl.String,
                "video": pl.String,
                "jobs": pl.List(JOB_STRUCT),
                "tracker_paths": pl.Struct(dict.fromkeys(PIPELINE_STATUS_COLUMNS, pl.String)),
                "datasets": pl.List(DATASET_STRUCT),
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

            if display_progress:
                console.echo(
                    message=f"Project '{project_directory.stem}' manifest: Generated.",
                    level=LogLevel.SUCCESS,
                )

        except Exception as exception:
            # Records the manifest job as failed before re-raising so the tracker reflects the aborted run.
            tracker.fail_job(job_id=job_id, error_message=str(exception))
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
            set_tbl_rows=-1,
            set_tbl_cols=-1,
            set_tbl_hide_column_data_types=True,
            set_tbl_cell_alignment="LEFT",
            set_tbl_width_chars=250,
            set_fmt_str_lengths=600,
        ):
            console.echo(message=str(self._data), raw=True)

    def print_summary(self, animal: int | None = None) -> None:
        """Prints a summary view of the manifest file to the terminal, excluding the 'experimenter notes' data for
        each session.

        This data view is optimized for tracking which processing steps have been applied to each of the project's data
        acquisition sessions. The 'session' column shows the per-animal 1-based session index, and the 'date' column
        shows the UTC acquisition time truncated to the second. Every pipeline status column and each dataset's
        completion flag collapse to 1 (completed) or 0 (otherwise), matching the numeric 'complete' column, so the
        operator reads a single done or not-done convention. The stored manifest keeps the full status labels for the
        orchestration layer.

        Args:
            animal: The unique identifier of the animal for which to display the data. If provided, this method only
                displays the data for that animal. Otherwise, it displays the data for all animals.
        """
        summary_cols = [
            "animal",
            "session",
            "date",
            "type",
            "system",
            "complete",
            "integrity",
            "two_photon",
            "runtime",
            "microcontroller",
            "video",
            "datasets",
        ]

        # The pipeline status columns collapse to a binary done indicator for this human-facing view. The detailed
        # labels are retained in the stored manifest columns that the orchestration layer reads.
        data_frame = (
            self._display_frame()
            .with_columns(
                # Maps each pipeline status to 1 when completed and 0 otherwise, matching the numeric 'complete' column.
                *(
                    (pl.col(column) == "completed").cast(pl.UInt8).alias(column)
                    for column in PIPELINE_STATUS_COLUMNS.values()
                ),
                # Each dataset's completion follows the same 0/1 convention as the pipeline status columns.
                pl.col("datasets")
                .list.eval(
                    pl.struct(
                        pl.element().struct.field("name").alias("name"),
                        pl.element().struct.field("complete").cast(pl.UInt8).alias("complete"),
                    )
                )
                .alias("datasets"),
            )
            .select(summary_cols)
        )

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
        """Prints the animal ID, per-animal session index, session date, session type, acquisition system, and
        experimenter notes data for each project's session to the terminal.

        This data view is optimized for determining what data acquisition sessions have been carried out and checking
        the outcomes of each session recorded in the experimenter notes. The 'session' column shows the per-animal
        1-based session index, and the 'date' column shows the UTC acquisition time truncated to the second.

        Args:
            animal: The unique identifier of the animal for which to display the data. If provided, this method only
                displays the data for that animal. Otherwise, it displays the data for all animals.
        """
        # Pre-selects the columns to display.
        data_frame = self._display_frame().select(["animal", "session", "date", "type", "system", "notes"])

        if animal is not None:
            data_frame = data_frame.filter(pl.col("animal") == int(animal))

        # Prints the extracted data.
        with pl.Config(
            set_tbl_rows=-1,
            set_tbl_cols=-1,
            set_tbl_hide_column_data_types=True,
            set_tbl_cell_alignment="LEFT",
            set_tbl_width_chars=170,
            set_fmt_str_lengths=2000,
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
            A Polars DataFrame containing all manifest columns for the specified session: 'animal', 'date',
            'session', 'session_path', 'type', 'system', 'notes', 'complete', the per-pipeline status columns
            ('integrity', 'two_photon', 'runtime', 'microcontroller', 'video'), 'jobs', 'tracker_paths', and
            'datasets'.
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

        Computes aggregate statistics across all sessions including per-pipeline status distributions and
        forged dataset membership. Designed for MCP tool responses where a structured dictionary is
        more useful than a printed table.

        Returns:
            A dictionary containing ``total_sessions``, ``total_animals``, ``animals``, ``session_types``,
            ``acquisition_systems``, ``complete_count``, ``pipeline_status_counts`` (the per-status session
            distribution of every per-session pipeline), the ``datasets`` summary, ``columns``, and ``total_rows``.
        """
        data = self._data
        total_rows = data.height

        # Counts sessions marked complete from the boolean (UInt8) completeness column.
        complete_count = int(data.filter(pl.col("complete") == 1).height)

        # The full distribution separates the sessions that failed a pipeline from those that have not run it.
        pipeline_status_counts: dict[str, dict[str, int]] = {}
        for pipeline, column in PIPELINE_STATUS_COLUMNS.items():
            if column not in data.columns:
                continue
            distribution: dict[str, int] = {}
            for value in data.select(column).to_series().to_list():
                distribution[str(value)] = distribution.get(str(value), 0) + 1
            pipeline_status_counts[pipeline] = distribution

        # Computes session type distribution.
        session_types: dict[str, int] = {}
        for row in data.select("type").to_series().to_list():
            session_types[str(row)] = session_types.get(str(row), 0) + 1

        # Computes acquisition system distribution.
        acquisition_systems: dict[str, int] = {}
        for row in data.select("system").to_series().to_list():
            acquisition_systems[str(row)] = acquisition_systems.get(str(row), 0) + 1

        # Builds the dataset summary by exploding the datasets column into one struct per membership and grouping
        # by dataset name. Each dataset entry reports the number of sessions it spans and its forging completion.
        dataset_summary: dict[str, Any] = {"total_datasets": 0, "datasets": []}
        if "datasets" in data.columns:
            # Filters to rows that belong to at least one dataset, then explodes and unnests the struct list so
            # each row represents a single (session, name, complete) triple.
            has_datasets = data.filter(pl.col("datasets").list.len() > 0)
            if has_datasets.height > 0:
                exploded = has_datasets.select("session", "datasets").explode("datasets").unnest("datasets")

                # Groups by dataset name to compute per-dataset session count and completion status.
                grouped = exploded.group_by("name").agg(
                    pl.col("session").count().alias("session_count"),
                    pl.col("complete").max().alias("complete"),
                )

                datasets: list[dict[str, Any]] = [
                    {
                        "name": row["name"],
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
            "pipeline_status_counts": pipeline_status_counts,
            "datasets": dataset_summary,
            "columns": data.columns,
            "total_rows": total_rows,
        }

    @property
    def data(self) -> pl.DataFrame:
        """Returns the Polars DataFrame instance that stores the managed manifest file's data."""
        return self._data

    def _display_frame(self) -> pl.DataFrame:
        """Returns a manifest copy prepared for terminal display, with the session name replaced by a per-animal
        1-based session index and the acquisition date truncated to the second as a timezone-aware UTC datetime.

        The stored 'session' and 'date' columns are left untouched on the underlying data, so this transformation
        only affects the printed views and never the identifiers the other query methods resolve against.
        """
        return self._data.sort(by=["animal", "session"]).with_columns(
            pl.int_range(1, pl.len() + 1).over("animal").alias("session"),
            pl.col("date").dt.truncate("1s").alias("date"),
        )

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

        if animal is not None:
            if animal not in self.animals:
                message = (
                    f"Unable to filter sessions using animal ID '{animal}'. The animal is not found in the "
                    f"project manifest. Available animals: {self.animals}."
                )
                console.error(message=message, error=ValueError)

            data = data.filter(pl.col("animal") == animal)

        if exclude_incomplete:
            data = data.filter(pl.col("complete") == 1)

        sessions = data.select("session").sort("session").to_series().to_list()
        return tuple(sessions)


def _build_session_row(session_data: SessionData, project_directory: Path) -> dict[str, Any]:
    """Builds every manifest column for a single session except the dataset-membership column.

    Notes:
        The dataset-membership column is excluded here because it resolves against a project-wide dataset map rather
        than the session alone, so the caller appends it after this returns.

        Every pipeline is read for every session regardless of completeness or integrity. Suppressing processing
        state for an incomplete session would make an unprocessable session indistinguishable from an unprocessed
        one, which is exactly the distinction an orchestrator needs.

    Args:
        session_data: The loaded session to snapshot.
        project_directory: The project's root directory, used to relativize the emitted paths.

    Returns:
        A mapping of manifest column name to that column's value for this session.

    Raises:
        ValueError: If the session's type has no registered descriptor class.
    """
    # Parses the session name, a UTC timestamp, into a timezone-aware UTC datetime.
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
    )

    # Loads the session descriptor to extract experimenter notes and completeness status. Every session carries a
    # valid descriptor, so a missing or unparseable descriptor propagates as an error.
    descriptor_class = DESCRIPTOR_REGISTRY.get(SessionTypes(session_data.session_type))
    if descriptor_class is None:
        message = (
            f"Unable to generate the manifest file for the '{project_directory.stem}' project. An unsupported "
            f"session type '{session_data.session_type}' was encountered for session "
            f"'{session_data.session_name}'. Currently, only the following session types are supported: "
            f"{tuple(SessionTypes)}."
        )
        console.error(message=message, error=ValueError)

    # DESCRIPTOR_REGISTRY types its values as the base YamlConfig, so the shared descriptor fields the manifest
    # reads (every registered descriptor declares them) need an attribute-defined ignore.
    descriptor = descriptor_class.from_yaml(file_path=session_data.raw_data.session_descriptor_path)

    # Resolves the canonical tracker path of every per-session pipeline from the SessionData grammar.
    session_trackers: dict[str, Path] = {
        "checksum": session_data.raw_data.checksum_tracker_path,
        "runtime": session_data.processed_data.runtime_tracker_path,
        "microcontroller": session_data.processed_data.microcontroller_tracker_path,
        "video": session_data.processed_data.video_tracker_path,
        "two_photon": session_data.processed_data.two_photon_tracker_path,
    }

    row: dict[str, Any] = {
        "animal": session_data.animal_id,
        "session": session_data.session_name,
        "session_path": f"{session_data.animal_id}/{session_data.session_name}",
        "date": date_time,
        "type": session_data.session_type,
        "system": session_data.acquisition_system,
        "notes": descriptor.experimenter_notes,  # type: ignore[attr-defined]
        "complete": not descriptor.incomplete,  # type: ignore[attr-defined]
    }

    session_jobs: list[dict[str, Any]] = []
    tracker_locations: dict[str, str] = {}
    for pipeline, tracker_path in session_trackers.items():
        status, pipeline_jobs = _read_pipeline_state(pipeline=pipeline, tracker_path=tracker_path)
        row[PIPELINE_STATUS_COLUMNS[pipeline]] = status
        session_jobs.extend(pipeline_jobs)
        tracker_locations[pipeline] = _relative_path(path=tracker_path, project_directory=project_directory)

    row["jobs"] = session_jobs
    row["tracker_paths"] = tracker_locations
    return row


def _relative_path(path: Path, project_directory: Path) -> str:
    """Expresses a path under the project hierarchy relative to the project root, using forward slashes.

    Relative locations let a manifest generated against one data root be consumed against another.

    Args:
        path: The absolute path to express relative to the project root.
        project_directory: The project's root directory.

    Returns:
        The path relative to the project root as a forward-slash string, or the unchanged absolute path as a string
        when it does not lie under the project root.
    """
    try:
        return path.relative_to(project_directory).as_posix()
    except ValueError:
        return str(path)


def _read_pipeline_state(pipeline: str, tracker_path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Reads one pipeline's processing tracker into a rolled-up status label and its per-job entries.

    Args:
        pipeline: The pipeline identifier recorded on each emitted job entry.
        tracker_path: The canonical path to the pipeline's processing tracker YAML file.

    Returns:
        A tuple of the rolled-up status label and the list of per-job entries, each carrying the ``pipeline``
        discriminator alongside the full ``JobState`` payload. A tracker that does not exist yields ``not_started``
        and no entries.
    """
    jobs = ProcessingTracker(file_path=tracker_path).snapshot()
    if not jobs:
        return "not_started", []

    status_payload = summarize_tracker(jobs=jobs)
    entries = [{"pipeline": pipeline, **entry} for entry in status_payload["jobs"]]
    return derive_tracker_status(summary=status_payload["summary"]), entries
