"""Provides assets for generating project manifest .feather files that capture the snapshot of a project's state."""

from __future__ import annotations

from typing import TYPE_CHECKING
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import polars as pl
from filelock import FileLock
from ataraxis_base_utilities import console
from sollertia_shared_assets import (
    SessionData,
    SessionTypes,
    RunTrainingDescriptor,
    LickTrainingDescriptor,
    WindowCheckingDescriptor,
    MesoscopeExperimentDescriptor,
)
from ataraxis_data_structures import ProcessingTracker

from sollertia_forgery.processing import TRACKER_FILENAME as BEHAVIOR_TRACKER_FILENAME

from .checksum import CHECKSUM_TRACKER_FILENAME

if TYPE_CHECKING:
    from pathlib import Path

MANIFEST_TRACKER_FILENAME: str = "manifest_processing_tracker.yaml"
"""The filename for the processing tracker placed in the project's root directory alongside the manifest .feather
file."""

MANIFEST_JOB_NAME: str = "manifest_generation"
"""The job name used to identify manifest generation jobs in processing trackers."""

_SESSION_DESCRIPTOR_FILENAME: str = "session_descriptor.yaml"
"""The expected filename for the session descriptor YAML file stored in each session's raw_data directory."""

_DESCRIPTOR_CLASSES: dict[
    str,
    type[LickTrainingDescriptor | RunTrainingDescriptor | MesoscopeExperimentDescriptor | WindowCheckingDescriptor],
] = {
    SessionTypes.LICK_TRAINING: LickTrainingDescriptor,
    SessionTypes.RUN_TRAINING: RunTrainingDescriptor,
    SessionTypes.MESOSCOPE_EXPERIMENT: MesoscopeExperimentDescriptor,
    SessionTypes.WINDOW_CHECKING: WindowCheckingDescriptor,
}
"""Maps each session type to its corresponding descriptor class. All descriptor classes share the
``experimenter_notes`` and ``incomplete`` attributes used by manifest generation."""

_CINDRA_TRACKER_FILENAME: str = "single_recording_tracker.yaml"
"""The tracker filename used to check whether a session has been processed with the cindra single-recording pipeline.
Located in the session's processed_data output directory."""

_VIDEO_TRACKER_FILENAME: str = "video_processing_tracker.yaml"
"""The tracker filename used to check whether a session has been processed with the DeepLabCut (video tracking)
pipeline. Located in the session's processed_data output directory."""

_MULTI_RECORDING_TRACKER_FILENAME: str = "multi_recording_tracker.yaml"
"""The tracker filename used to check whether a cindra multi-recording dataset has been processed. Located in
the dataset's output directory under the main recording's cindra multi_recording path."""


def generate_project_manifest(project_directory: Path) -> None:
    """Builds and saves the project manifest .feather file under the target project's root directory.

    Notes:
        Initializes a processing tracker in the project's root directory alongside the manifest .feather output,
        runs the manifest generation, and records the outcome. Acquires a file lock on the manifest .feather file
        to ensure only one process writes at a time.

    Args:
        project_directory: The path to the processed project's root directory.

    Raises:
        FileNotFoundError: If the project directory does not exist or contains no session data.
        ValueError: If an unsupported session type is encountered.
    """
    if not project_directory.exists():
        message = (
            f"Unable to generate the project manifest file for the '{project_directory.stem}' project. "
            f"The specified project directory does not exist."
        )
        console.error(message=message, error=FileNotFoundError)

    # Finds the root directories for all project's sessions.
    session_directories = [directory.parents[1] for directory in project_directory.rglob("session_data.yaml")]

    if not session_directories:
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

    # Initializes the processing tracker in the project directory alongside the manifest output.
    tracker = ProcessingTracker(file_path=project_directory.joinpath(MANIFEST_TRACKER_FILENAME))
    job_ids = tracker.initialize_jobs(jobs=[(MANIFEST_JOB_NAME, project_directory.stem)])
    job_id = job_ids[0]

    # Acquires the lock file, ensuring only this specific process can work with the manifest data.
    lock = FileLock(str(manifest_lock))
    with lock.acquire(timeout=20.0):
        # Marks the job as running.
        tracker.start_job(job_id=job_id)
        try:
            # Pre-creates the 'manifest' dictionary structure.
            manifest: dict[str, list] = {
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
                # Determines whether the session has been processed with the cindra single-recording pipeline.
                "cindra": [],
                # Determines whether the session has been processed with the behavior extraction pipeline.
                "behavior": [],
                # Determines whether the session has been processed with the DeepLabCut (video tracking) pipeline.
                "video": [],
                # Stores the cindra multi-recording dataset names the session belongs to (empty list if none).
                "multi_recording_datasets": [],
                # Stores per-dataset completion status, aligned by index with multi_recording_datasets.
                "multi_recording_complete": [],
            }

            # Scans the entire project for cindra multi-recording tracker files to build a dataset completion
            # registry. The tracker only lives on the main recording, so a project-wide scan is needed to
            # resolve completion status for datasets discovered on non-main sessions.
            multi_recording_registry: dict[str, bool] = {}
            for tracker_path in sorted(project_directory.rglob(_MULTI_RECORDING_TRACKER_FILENAME)):
                dataset_name = tracker_path.parent.name
                dataset_tracker = ProcessingTracker(file_path=tracker_path)
                multi_recording_registry[dataset_name] = dataset_tracker.complete

            # Pre-creates the Eastern timezone object for UTC-to-EST/EDT conversion.
            eastern = ZoneInfo("America/New_York")

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

                # Parses session name into a timezone-aware datetime in Eastern time.
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
                ).astimezone(eastern)
                manifest["date"].append(date_time)

                # Loads the session descriptor to extract experimenter notes and completeness status. Window
                # Checking sessions acquired before sollertia-experiment 3.0.0 lack descriptors, so a missing
                # file is handled gracefully for that session type only.
                descriptor_path = session_data.raw_data_path.joinpath(_SESSION_DESCRIPTOR_FILENAME)
                descriptor_class = _DESCRIPTOR_CLASSES.get(session_data.session_type)
                if descriptor_class is None:
                    message = (
                        f"Unsupported session type '{session_data.session_type}' encountered for session "
                        f"'{directory.stem}' when generating the manifest file for the project "
                        f"{project_directory.stem}. Currently, only the following session types are supported: "
                        f"{tuple(SessionTypes)}."
                    )
                    console.error(message=message, error=ValueError)

                try:
                    descriptor = descriptor_class.from_yaml(file_path=descriptor_path)
                    is_complete = not descriptor.incomplete
                    manifest["notes"].append(descriptor.experimenter_notes)
                except Exception:
                    if session_data.session_type != SessionTypes.WINDOW_CHECKING:
                        raise
                    is_complete = False
                    manifest["notes"].append("N/A")

                manifest["complete"].append(is_complete)

                # Resolves data integrity verification status. The checksum tracker lives alongside the
                # checksum file in raw_data.
                checksum_tracker = _find_tracker(
                    search_root=session_data.raw_data_path, tracker_filename=CHECKSUM_TRACKER_FILENAME
                )
                is_verified = checksum_tracker.complete if checksum_tracker is not None else False
                manifest["integrity"].append(is_verified)

                # If the session is incomplete or unverified, marks all processing steps as FALSE, as automatic
                # processing is disabled for incomplete sessions and, therefore, it could not have been processed.
                if not is_complete or not is_verified:
                    manifest["cindra"].append(False)
                    manifest["behavior"].append(False)
                    manifest["video"].append(False)
                    manifest["multi_recording_datasets"].append([])
                    manifest["multi_recording_complete"].append([])
                    continue  # Cycles to the next session

                # Resolves cindra single-recording processing status by searching processed_data for the tracker.
                cindra_tracker = _find_tracker(
                    search_root=session_data.processed_data_path, tracker_filename=_CINDRA_TRACKER_FILENAME
                )
                manifest["cindra"].append(cindra_tracker.complete if cindra_tracker is not None else False)

                # Resolves behavior data processing status by searching processed_data for the tracker.
                behavior_tracker = _find_tracker(
                    search_root=session_data.processed_data_path, tracker_filename=BEHAVIOR_TRACKER_FILENAME
                )
                manifest["behavior"].append(
                    behavior_tracker.complete if behavior_tracker is not None else False
                )

                # Resolves DeepLabCut (video) processing status by searching processed_data for the tracker.
                video_tracker = _find_tracker(
                    search_root=session_data.processed_data_path, tracker_filename=_VIDEO_TRACKER_FILENAME
                )
                manifest["video"].append(video_tracker.complete if video_tracker is not None else False)

                # Resolves multi-recording dataset membership by searching for multi_recording subdirectories
                # under processed_data to discover which datasets this session participates in, then looks up
                # each dataset's completion status from the project-wide registry.
                dataset_dirs = sorted(session_data.processed_data_path.rglob("multi_recording/*/"))
                session_datasets: list[str] = []
                session_dataset_complete: list[bool] = []
                for dataset_dir in dataset_dirs:
                    if dataset_dir.is_dir():
                        dataset_name = dataset_dir.name
                        session_datasets.append(dataset_name)
                        session_dataset_complete.append(multi_recording_registry.get(dataset_name, False))
                manifest["multi_recording_datasets"].append(session_datasets)
                manifest["multi_recording_complete"].append(session_dataset_complete)

            # Converts animal IDs from strings to integers for proper numeric sorting.
            manifest["animal"] = [int(animal) for animal in manifest["animal"]]

            # Converts the manifest dictionary to a Polars Dataframe.
            schema: dict[str, pl.datatypes.classes.DataTypeClass | pl.DataType] = {
                "animal": pl.UInt64,
                "date": pl.Datetime,
                "session": pl.String,
                "type": pl.String,
                "system": pl.String,
                "notes": pl.String,
                "complete": pl.UInt8,
                "integrity": pl.UInt8,
                "cindra": pl.UInt8,
                "behavior": pl.UInt8,
                "video": pl.UInt8,
                "multi_recording_datasets": pl.List(pl.String),
                "multi_recording_complete": pl.List(pl.UInt8),
            }
            manifest_frame = pl.DataFrame(data=manifest, schema=schema, strict=False)

            # Sorts the DataFrame by animal and then session. Since animal IDs are monotonically increasing
            # according to Sollertia standards and session 'names' are based on acquisition timestamps, the
            # sort order is chronological.
            sorted_manifest = manifest_frame.sort(["animal", "session"])

            # Saves the generated manifest to the project-specific uncompressed .feather file to allow
            # memory-mapped reads.
            sorted_manifest.write_ipc(file=manifest_path, compression="uncompressed")

            # Marks the job as completed.
            tracker.complete_job(job_id=job_id)

        except Exception:
            # If the code reaches this section, this means the runtime encountered an error.
            tracker.fail_job(job_id=job_id)
            raise


def _find_tracker(search_root: Path, tracker_filename: str) -> ProcessingTracker | None:
    """Searches for a single processing tracker file by name under the given directory tree.

    Args:
        search_root: The root directory to search recursively.
        tracker_filename: The filename of the tracker to locate.

    Returns:
        A ProcessingTracker instance bound to the discovered file, or None if the file was not found.

    Raises:
        RuntimeError: If more than one matching tracker file is found under the search root.
    """
    candidates = list(search_root.rglob(tracker_filename))
    if len(candidates) == 1:
        return ProcessingTracker(file_path=candidates[0])
    if len(candidates) > 1:
        message = (
            f"Expected at most one '{tracker_filename}' under '{search_root}', but found {len(candidates)}: "
            f"{candidates}."
        )
        console.error(message=message, error=RuntimeError)
    return None
