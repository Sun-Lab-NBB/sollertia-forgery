"""Provides assets for generating, visualizing, and querying the session-rowed project manifest .feather file that
captures the snapshot of a project's state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections import Counter

import polars as pl
from natsort import natsorted
from filelock import FileLock
from ataraxis_base_utilities import LogLevel, console
from sollertia_shared_assets import (
    DESCRIPTOR_REGISTRY,
    SessionData,
    SessionTypes,
    ProcessingTrackers,
    iterate_sessions,
    parse_session_timestamp,
)
from ataraxis_data_structures import TrackerStatus, ProcessingTracker, atomic_write

from .jobs import write_project_jobs
from ..shared_assets import (
    SESSION_PIPELINES,
    ProcessingPipelines,
    natural_sort,
    resolve_session_tracker_path,
)

if TYPE_CHECKING:
    from pathlib import Path
    from datetime import datetime

MANIFEST_JOB_NAME: str = "manifest_generation"
"""The job name used to identify manifest generation jobs in processing trackers."""

_PIPELINE_STATUS_COLUMNS: dict[ProcessingPipelines, str] = {
    ProcessingPipelines.CHECKSUM: "integrity",
    ProcessingPipelines.RUNTIME: "runtime",
    ProcessingPipelines.MICROCONTROLLER: "microcontroller",
    ProcessingPipelines.VIDEO: "video",
    ProcessingPipelines.TWO_PHOTON: "two_photon",
}
"""Maps each per-session pipeline to the manifest column that reports whether it finished for a session. The checksum
pipeline reports under the ``integrity`` column, so the mapping's values differ from its keys.

Notes:
    Each column holds 1 when every job of that pipeline succeeded and 0 otherwise, matching the numeric ``complete``
    column. The manifest answers whether a pipeline is done, and nothing finer.

    Every pipeline in ``SESSION_PIPELINES`` declares a column here, since the manifest reports one per pipeline a
    session carries a tracker for.
"""

_PROJECT_MANIFEST_SCHEMA: dict[str, pl.datatypes.classes.DataTypeClass | pl.DataType] = {
    "animal": pl.String,
    "date": pl.Datetime,
    "session": pl.String,
    "session_path": pl.String,
    "type": pl.String,
    "system": pl.String,
    "notes": pl.String,
    "complete": pl.UInt8,
    "integrity": pl.UInt8,
    "two_photon": pl.UInt8,
    "runtime": pl.UInt8,
    "microcontroller": pl.UInt8,
    "video": pl.UInt8,
}
"""The column layout of the project manifest artifact, one row per session.

Notes:
    The ``animal`` column is text, matching the project job, dataset state, and project plan artifacts, so a reader
    joins any pair of them on the animal and session key without casting either side.

    The ``complete`` column and the five pipeline columns each hold a 0 or a 1, so a reader applies one done or
    not-done convention across all six.
"""


def project_manifest_path(project_directory: Path) -> Path:
    """Resolves the path to the project manifest .feather file under the target project's root directory.

    Args:
        project_directory: The path to the project's root directory.

    Returns:
        The path to the project manifest .feather file.
    """
    return project_directory.joinpath(f"{project_directory.stem}_manifest.feather")


def generate_project_manifest(project_directory: Path, *, display_progress: bool = False) -> None:
    """Builds and saves the project manifest .feather file under the target project's root directory.

    The manifest captures one row per session with its acquisition metadata and per-pipeline processing status. It
    also writes the project job artifact, ``<project>_jobs.feather``, into the same root, publishing it before the
    manifest itself. A file lock serializes concurrent writers, and the outcome is recorded on a manifest processing
    tracker in the project root.

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

    manifest_path = project_manifest_path(project_directory=project_directory)
    manifest_lock = manifest_path.with_suffix(manifest_path.suffix + ".lock")

    # Applies stale entry detection so that foreign or outdated job entries are discarded before the new job is
    # registered.
    tracker = ProcessingTracker(file_path=project_directory.joinpath(ProcessingTrackers.MANIFEST))
    jobs = [(MANIFEST_JOB_NAME, project_directory.stem)]
    tracker.align_jobs(jobs=jobs, universe=jobs)
    job_id = ProcessingTracker.generate_job_id(job_name=MANIFEST_JOB_NAME, specifier=project_directory.stem)

    job_rows: list[dict[str, str | int | None]] = []

    lock = FileLock(str(manifest_lock))
    with lock.acquire(timeout=20.0):
        tracker.start_job(job_id=job_id)
        try:
            manifest: dict[str, list[str | datetime | bool | int | None]] = {
                "animal": [],
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
                "notes": [],
                # Determines whether the session's data is complete and ready for unsupervised processing.
                "complete": [],
                # Determines whether the checksum (data-integrity) pipeline finished for this session, stored
                # as 1 when every job succeeded and 0 otherwise. Which jobs failed, and why, are read from the
                # project job artifact.
                "integrity": [],
                # Determines whether the two-photon (cindra) processing pipeline finished for this session.
                "two_photon": [],
                # Determines whether the runtime processing pipeline finished for this session.
                "runtime": [],
                # Determines whether the microcontroller processing pipeline finished for this session.
                "microcontroller": [],
                # Determines whether the video (timestamp, tracking, motion energy) pipeline finished for this
                # session.
                "video": [],
            }

            for session_data in sessions:
                # Skips sessions whose raw_data directory is empty. A fully acquired session carries its marker and
                # acquired data under raw_data, so an empty raw_data marks an aborted or not-yet-acquired session
                # that must not enter the manifest.
                if not any(session_data.raw_data_path.glob("*")):
                    continue

                row, session_jobs = _build_session_row(session_data=session_data, project_directory=project_directory)
                for column, value in row.items():
                    manifest[column].append(value)

                # The per-job registry is written to its own job-rowed artifact rather than nested inside a session
                # row, so a reader pages it one job at a time.
                job_rows.extend(session_jobs)

            manifest_frame = pl.DataFrame(data=manifest, schema=_PROJECT_MANIFEST_SCHEMA, strict=False)

            # Groups the rows by animal and orders each animal's sessions chronologically, since session names are
            # acquisition timestamps. The animal identifier is text, so the ordering is natural rather than
            # lexicographic and animal 2 precedes animal 10.
            sorted_manifest = natural_sort(frame=manifest_frame, by=["animal", "session"])

            # The job artifact is published first, because the two renames are ordered rather than simultaneous and a
            # reader takes no lock. Landing the detail before the summary leaves the reader at worst holding job rows
            # for a session the manifest does not list yet, which a join on the documented key drops. The reverse
            # order would show a manifest row whose jobs are absent, which reads as a session nothing has processed.
            write_project_jobs(project_directory=project_directory, job_rows=job_rows)

            # Saves the generated manifest to the project-specific uncompressed .feather file to allow
            # memory-mapped reads. Published through a temporary file renamed over the destination, so a reader
            # that memory-maps the artifact without taking this lock observes either the previous manifest or the
            # complete new one, never a partially rewritten file.
            with atomic_write(file_path=manifest_path, binary=True) as file:
                sorted_manifest.write_ipc(file=file, compression="uncompressed")

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

    def print_summary(self, animal: str | None = None) -> None:
        """Prints a summary view of the manifest file to the terminal, excluding the session path and the
        experimenter notes data for each session.

        This data view is optimized for tracking which processing steps have been applied to each of the project's data
        acquisition sessions. The ``session`` column shows the per-animal 1-based session index, and the ``date``
        column shows the UTC acquisition time truncated to the second. Every pipeline column reports 1 when that
        pipeline finished and 0 otherwise, matching the numeric ``complete`` column, so the operator reads one done or
        not-done convention throughout.

        Args:
            animal: The unique identifier of the animal for which to display the data. If provided, this method only
                displays the data for that animal. Otherwise, it displays the data for all animals.
        """
        summary_columns = [
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
        ]

        # The stored pipeline columns are already the 0/1 indicator this view wants, so nothing needs reducing.
        data_frame = self._display_frame().select(summary_columns)

        if animal is not None:
            data_frame = data_frame.filter(pl.col("animal") == animal)

        with pl.Config(
            set_tbl_rows=-1,
            set_tbl_cols=-1,
            set_tbl_width_chars=250,
            set_tbl_hide_column_data_types=True,
            set_tbl_cell_alignment="CENTER",
        ):
            console.echo(message=str(data_frame), raw=True)

    def print_notes(self, animal: str | None = None) -> None:
        """Prints the animal ID, per-animal session index, session date, session type, acquisition system, and
        experimenter notes data for each project's session to the terminal.

        This data view is optimized for determining what data acquisition sessions have been carried out and checking
        the outcomes of each session recorded in the experimenter notes. The ``session`` column shows the per-animal
        1-based session index, and the ``date`` column shows the UTC acquisition time truncated to the second.

        Args:
            animal: The unique identifier of the animal for which to display the data. If provided, this method only
                displays the data for that animal. Otherwise, it displays the data for all animals.
        """
        data_frame = self._display_frame().select(["animal", "session", "date", "type", "system", "notes"])

        if animal is not None:
            data_frame = data_frame.filter(pl.col("animal") == animal)

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
    def data(self) -> pl.DataFrame:
        """Returns the Polars DataFrame instance that stores the managed manifest file's data."""
        return self._data

    @property
    def animals(self) -> tuple[str, ...]:
        """Returns the unique identifiers for each animal participating in the project, in natural order."""
        return tuple(natsorted(self._data.select("animal").unique().to_series().to_list()))

    def get_sessions(
        self,
        animal: str | None = None,
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
            A Polars DataFrame containing all manifest columns for the specified session: ``animal``, ``date``,
            ``session``, ``session_path``, ``type``, ``system``, ``notes``, ``complete``, and the per-pipeline done
            columns ``integrity``, ``two_photon``, ``runtime``, ``microcontroller``, and ``video``.
        """
        return self._data.filter(pl.col("session") == session)

    def get_animal_for_session(self, session: str) -> str:
        """Returns the unique identifier of the animal that participated in the specified session.

        Args:
            session: The unique identifier of the session for which to retrieve the participating animal's identifier.

        Returns:
            The unique identifier of the animal that participated in the specified session.

        Raises:
            ValueError: If the specified session is not found in the manifest file.
        """
        data_frame = self._data.filter(pl.col("session") == session)

        if data_frame.is_empty():
            message = (
                f"Unable to look up the participating animal using session ID '{session}'. The session is not "
                f"found in the project manifest. Available sessions: "
                f"{self.get_sessions(animal=None, exclude_incomplete=False)}."
            )
            console.error(message=message, error=ValueError)

        return str(data_frame.select("animal").item())

    def get_system_for_session(self, session: str) -> str:
        """Returns the data acquisition system used to acquire the specified session's data.

        Args:
            session: The unique identifier of the session for which to retrieve the data acquisition system.

        Returns:
            The data acquisition system used to acquire the specified session's data.

        Raises:
            ValueError: If the specified session is not found in the manifest file.
        """
        data_frame = self._data.filter(pl.col("session") == session)

        if data_frame.is_empty():
            message = (
                f"Unable to look up the acquisition system using session ID '{session}'. The session is not "
                f"found in the project manifest. Available sessions: "
                f"{self.get_sessions(animal=None, exclude_incomplete=False)}."
            )
            console.error(message=message, error=ValueError)

        return str(data_frame.select("system").item())

    def summarize(self) -> dict[str, int | list[str] | dict[str, int] | dict[str, dict[str, int]]]:
        """Returns a structured summary of the project manifest for programmatic consumption.

        Counts the sessions each pipeline finished, alongside the session type and acquisition system distributions.
        Reports on sessions only, since a dataset's forging state is recorded on that dataset's own artifacts.

        Returns:
            A dictionary containing ``total_sessions``, ``total_animals``, ``animals``, ``session_types``,
            ``acquisition_systems``, ``complete_count``, ``pipeline_status_counts`` (the count of sessions that
            finished and of sessions that did not, keyed by pipeline identifier), ``columns``, and ``total_rows``.
        """
        data = self._data
        total_rows = data.height

        complete_count = int(data.filter(pl.col("complete") == 1).height)

        # Counts the sessions each pipeline finished. Which jobs failed, and why, are read from the job artifact.
        pipeline_status_counts: dict[str, dict[str, int]] = {}
        for pipeline, column in _PIPELINE_STATUS_COLUMNS.items():
            if column not in data.columns:
                continue
            finished = int(data.filter(pl.col(column) == 1).height)
            pipeline_status_counts[pipeline.value] = {"done": finished, "not_done": total_rows - finished}

        session_types: dict[str, int] = dict(Counter(str(row) for row in data.select("type").to_series().to_list()))
        acquisition_systems: dict[str, int] = dict(
            Counter(str(row) for row in data.select("system").to_series().to_list())
        )

        return {
            "total_sessions": total_rows,
            "total_animals": len(self.animals),
            "animals": list(self.animals),
            "session_types": session_types,
            "acquisition_systems": acquisition_systems,
            "complete_count": complete_count,
            "pipeline_status_counts": pipeline_status_counts,
            "columns": data.columns,
            "total_rows": total_rows,
        }

    def _display_frame(self) -> pl.DataFrame:
        """Returns a manifest copy prepared for terminal display, with the session name replaced by a per-animal
        1-based session index and the acquisition date truncated to the second as a timezone-aware UTC datetime.

        The stored ``session`` and ``date`` columns are left untouched on the underlying data, so this transformation
        only affects the printed views and never the identifiers the other query methods resolve against.
        """
        return natural_sort(frame=self._data, by=["animal", "session"]).with_columns(
            pl.int_range(1, pl.len() + 1).over("animal").alias("session"),
            pl.col("date").dt.truncate("1s").alias("date"),
        )

    def _get_filtered_sessions(
        self,
        animal: str | None = None,
        *,
        exclude_incomplete: bool = True,
    ) -> tuple[str, ...]:
        """Builds a tuple of unique session identifiers with optional filtering.

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

        return tuple(natsorted(data.select("session").to_series().to_list()))


def _build_session_row(
    session_data: SessionData, project_directory: Path
) -> tuple[dict[str, str | datetime | bool | int | None], list[dict[str, str | int | None]]]:
    """Builds every manifest column for a single session, alongside that session's job rows.

    Notes:
        The job rows are returned separately rather than nested in the manifest row, because they are written to the
        project's own job-rowed artifact where a reader pages them one job at a time.

        Every pipeline is read for every session regardless of completeness or integrity. Suppressing processing
        state for an incomplete session would make an unprocessable session indistinguishable from an unprocessed
        one, which is exactly the distinction an orchestrator needs.

    Args:
        session_data: The loaded session to snapshot.
        project_directory: The project's root directory, named in the error raised for an unsupported session type.

    Returns:
        A tuple of the manifest row, as a mapping of column name to value, and the session's job rows, each carrying
        the animal and session that recorded it.

    Raises:
        ValueError: If the session's type has no registered descriptor class.
        FileNotFoundError: If the session carries no descriptor file at its canonical raw-data path.
    """
    # Parses the session name, a UTC timestamp, into a timezone-aware UTC datetime. A name that does not follow the
    # session naming format yields None, which the manifest stores as a null date rather than aborting the walk.
    date_time = parse_session_timestamp(session_name=session_data.session_name)

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

    row: dict[str, str | datetime | bool | int | None] = {
        "animal": session_data.animal_id,
        "session": session_data.session_name,
        "session_path": f"{session_data.animal_id}/{session_data.session_name}",
        "date": date_time,
        "type": session_data.session_type,
        "system": session_data.acquisition_system,
        "notes": descriptor.experimenter_notes,  # type: ignore[attr-defined]
        "complete": not descriptor.incomplete,  # type: ignore[attr-defined]
    }

    session_jobs: list[dict[str, str | int | None]] = []
    for pipeline in SESSION_PIPELINES:
        tracker_path = resolve_session_tracker_path(session=session_data, pipeline=pipeline)
        status, pipeline_jobs = _read_pipeline_state(pipeline=pipeline, tracker_path=tracker_path)
        row[_PIPELINE_STATUS_COLUMNS[pipeline]] = int(status == TrackerStatus.COMPLETED)
        session_jobs.extend(pipeline_jobs)

    # Each job row carries the session that recorded it, since the rows of every session are written to one artifact.
    subject = {"animal": str(session_data.animal_id), "session": session_data.session_name}
    return row, [{**subject, **entry} for entry in session_jobs]


def _read_pipeline_state(
    pipeline: ProcessingPipelines, tracker_path: Path
) -> tuple[TrackerStatus, list[dict[str, str | int | None]]]:
    """Reads one pipeline's processing tracker into a rolled-up status label and its per-job entries.

    Notes:
        A tracker holding no jobs is reported as not started, rather than through the label the tracker resolves an
        empty registry to, which is in progress and would read as a pipeline that has already begun.

    Args:
        pipeline: The pipeline whose identifier is recorded on each emitted job entry.
        tracker_path: The canonical path to the pipeline's processing tracker YAML file.

    Returns:
        A tuple of the rolled-up status label and the list of per-job entries, each carrying the ``pipeline``
        discriminator and the registry ``job_id`` alongside the ``JobState`` payload, whose ``error_message`` key is
        absent for a job that recorded no failure. A tracker that does not exist yields ``not_started``
        and no entries.
    """
    status_payload = ProcessingTracker(file_path=tracker_path).summarize()
    if not status_payload["jobs"]:
        return TrackerStatus.NOT_STARTED, []

    entries = [{"pipeline": pipeline.value, **entry} for entry in status_payload["jobs"]]
    return status_payload["status"], entries


def _assert_status_column_coverage() -> None:
    """Verifies that every pipeline a session carries a tracker for declares a manifest status column.

    Notes:
        Runs at import, so a pipeline added to ``SESSION_PIPELINES`` without a status column here fails the moment
        this module loads rather than partway through a generation pass over a project.

    Raises:
        RuntimeError: If a per-session pipeline declares no status column, or a column names a pipeline that no
            session carries a tracker for.
    """
    declared = frozenset(_PIPELINE_STATUS_COLUMNS)
    carried = frozenset(SESSION_PIPELINES)
    if declared != carried:
        message = (
            f"Unable to validate the manifest's pipeline status columns. Every pipeline in SESSION_PIPELINES must "
            f"declare a status column and no column may name a pipeline outside it, but the sets differ by "
            f"{sorted(member.value for member in declared ^ carried)}."
        )
        console.error(message=message, error=RuntimeError)


_assert_status_column_coverage()
