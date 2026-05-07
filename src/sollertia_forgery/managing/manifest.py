"""Provides assets for generating project manifest .feather files that capture the snapshot of a project's state."""

from __future__ import annotations

from typing import TYPE_CHECKING
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import polars as pl
from filelock import FileLock
from ataraxis_base_utilities import console
from sollertia_shared_assets import (
    SessionTypes,
    ProcessingTrackers,
    RunTrainingDescriptor,
    LickTrainingDescriptor,
    WindowCheckingDescriptor,
    MesoscopeExperimentDescriptor,
    iterate_sessions,
)
from ataraxis_data_structures import ProcessingTracker

from ..shared_assets import prepare_tracker

if TYPE_CHECKING:
    from pathlib import Path

    from sollertia_shared_assets import SessionData

MANIFEST_JOB_NAME: str = "manifest_generation"
"""The job name used to identify manifest generation jobs in processing trackers."""

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
    prepare_tracker(tracker=tracker, jobs=jobs)
    job_id = ProcessingTracker.generate_job_id(job_name=MANIFEST_JOB_NAME, specifier=project_directory.stem)

    # Acquires the lock file, ensuring only this specific process can work with the manifest data.
    lock = FileLock(str(manifest_lock))
    with lock.acquire(timeout=20.0):
        # Marks the job as running.
        tracker.start_job(job_id=job_id)
        try:
            # Pre-creates the 'manifest' dictionary structure.
            manifest: dict[str, list] = {
                # Animal IDs.
                "animal": [],
                # Session names.
                "session": [],
                # Session names stored as timezone-aware date-time objects in EST.
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
                    # Cindra writes the dataset directory as ``{animal_id}_{base_name}`` for collision
                    # avoidance when batching multiple animals under one analysis. The manifest surfaces the
                    # unqualified base name, so the animal_id prefix is stripped here.
                    dataset_name = _strip_animal_prefix(
                        qualified_name=dataset_dir.name, animal_id=str(session_data.animal_id)
                    )
                    multi_recording_registry[dataset_name] = ProcessingTracker(file_path=tracker_path).complete

            # Pre-creates the Eastern timezone object for UTC-to-EST/EDT conversion.
            eastern = ZoneInfo("America/New_York")

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
                descriptor_path = session_data.raw_data.session_descriptor_path
                descriptor_class = _DESCRIPTOR_CLASSES.get(session_data.session_type)
                if descriptor_class is None:
                    message = (
                        f"Unsupported session type '{session_data.session_type}' encountered for session "
                        f"'{session_data.session_name}' when generating the manifest file for the project "
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

                # Resolves data integrity verification status from the canonical checksum tracker path.
                checksum_tracker = _load_tracker_if_exists(tracker_path=session_data.raw_data.checksum_tracker_path)
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

                # Resolves cindra single-recording, behavior, and DeepLabCut (video) processing status from
                # canonical tracker paths exposed by SessionData.
                cindra_tracker = _load_tracker_if_exists(tracker_path=session_data.processed_data.cindra_single_recording_tracker_path)
                manifest["cindra"].append(cindra_tracker.complete if cindra_tracker is not None else False)

                behavior_tracker = _load_tracker_if_exists(tracker_path=session_data.processed_data.behavior_tracker_path)
                manifest["behavior"].append(behavior_tracker.complete if behavior_tracker is not None else False)

                video_tracker = _load_tracker_if_exists(tracker_path=session_data.processed_data.video_tracker_path)
                manifest["video"].append(video_tracker.complete if video_tracker is not None else False)

                # Resolves multi-recording dataset membership by enumerating the session's
                # ``cindra/multi_recording`` subdirectories, then looks up each dataset's completion status
                # from the project-wide registry built above.
                multi_recording_root = session_data.processed_data.cindra_multi_recording_path
                session_datasets: list[str] = []
                session_dataset_complete: list[bool] = []
                if multi_recording_root.is_dir():
                    for dataset_dir in sorted(multi_recording_root.iterdir()):
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
            sorted_manifest = manifest_frame.sort(by=["animal", "session"])

            # Saves the generated manifest to the project-specific uncompressed .feather file to allow
            # memory-mapped reads.
            sorted_manifest.write_ipc(file=manifest_path, compression="uncompressed")

            # Marks the job as completed.
            tracker.complete_job(job_id=job_id)

        except Exception:
            # If the code reaches this section, this means the runtime encountered an error.
            tracker.fail_job(job_id=job_id)
            raise


def _strip_animal_prefix(qualified_name: str, animal_id: str) -> str:
    """Strips the ``{animal_id}_`` prefix from a cindra multi-recording dataset directory name.

    Cindra's ``resolve_dataset_name_tool`` prepends the animal identifier to user-supplied dataset names to
    produce collision-free output directories when batching multiple animals with the same analysis. This
    helper reverses that qualification so manifest consumers see the logical base name instead of the
    filesystem-qualified name.

    Args:
        qualified_name: The on-disk directory name as produced by cindra.
        animal_id: The animal identifier that was prepended by cindra as the specifier.

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
