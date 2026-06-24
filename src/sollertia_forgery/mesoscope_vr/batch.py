"""Provides the Mesoscope-VR batch adapters that wire the behavior and forging pipelines into the system-agnostic
processing tools: discovery/preparation, output verification, cleanup, status overview, concurrency descriptors, and
the picklable per-job workers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path
from dataclasses import dataclass

from natsort import natsorted
from ataraxis_base_utilities import console
from sollertia_shared_assets import (
    Directories,
    SessionData,
    SurgeryData,
    RawDataFiles,
    SessionTypes,
    ProcessingTrackers,
    MesoscopeExperimentDescriptor,
    iterate_sessions,
)
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker, delete_directory

from .forging import FORGING_JOB_NAME, run_forging_pipeline
from ..forging import resolve_dataset
from .processing import discover_behavior_jobs, run_behavior_processing_pipeline
from ..orchestration import (
    GenericPendingJob,
    prepare_tracker,
    read_tracker_status,
    analyze_feather_file,
    derive_tracker_status,
    clean_output_subdirectory,
)
from ..shared_assets import DatasetData, DatasetFiles

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True, slots=True)
class ConcurrencyDescriptor:
    """Describes the per-pipeline concurrency policy consulted by the generic ``execute_jobs_tool``.

    Notes:
        ``cores_per_job`` is the number of CPU cores a single worker subprocess consumes; the generic tool floors
        the user-supplied parallel-job cap by ``worker_budget // cores_per_job``. ``default_max_parallel`` is the
        fallback hard cap applied when the caller does not request an explicit parallel-job ceiling.
    """

    cores_per_job: int
    """The number of CPU cores a single worker subprocess of this pipeline consumes."""
    default_max_parallel: int
    """The default hard cap on concurrently executing jobs when the caller does not request one. A non-positive
    value defers concurrency to the resolved worker budget alone."""


# ----------------------------------------------------------------------------------------------------------------------
# Behavior processing adapters
# ----------------------------------------------------------------------------------------------------------------------

BEHAVIOR_CONCURRENCY: ConcurrencyDescriptor = ConcurrencyDescriptor(cores_per_job=1, default_max_parallel=-1)
"""The behavior pipeline concurrency policy: one core per worker, with concurrency deferred to the worker budget."""


def prepare_behavior_unit(unit: dict[str, Any]) -> dict[str, Any]:
    """Discovers behavior jobs for a single session and initializes its processing tracker.

    Loads the session via behavior discovery, resolves the static ``behavior_data/`` output location under the
    session's processed data hierarchy, initializes (or reads, when already present) the behavior processing
    tracker, and returns the per-session manifest entry. The output location is resolved statically from the
    session's ``SessionData`` marker; the caller never chooses where outputs go.

    Args:
        unit: The behavior unit specification. Must carry a ``session_path`` key with the absolute path to the
            session root directory containing the session data hierarchy.

    Returns:
        A per-session manifest dictionary carrying the session name, the static ``data_path``, the tracker path,
        the enriched job descriptors, and the tracker summary. On discovery failure, returns a dictionary with an
        ``error`` key and empty ``jobs``/``summary``.
    """
    raw_path = unit.get("session_path") or unit.get("unit_path")
    if not raw_path:
        return {"error": "Behavior unit specification must provide a 'session_path'.", "jobs": [], "summary": {}}
    unit_path = Path(raw_path)

    # Runs discovery for the session. Discovery loads the SessionData, validates the session type, and enumerates
    # the runtime and microcontroller jobs available on disk.
    try:
        session, discovered_jobs = discover_behavior_jobs(session_path=unit_path)
    except Exception as error:
        return {"error": f"Discovery failed: {error}", "jobs": [], "summary": {}}

    if not discovered_jobs:
        return {
            "error": "No processable behavior jobs discovered for this session.",
            "jobs": [],
            "summary": {},
        }

    # Resolves the static output location. The ``behavior_data/`` subdirectory always lives under the session's
    # ``processed_data_path``, co-located with the upstream ``camera_timestamps/`` and ``microcontroller_data/``
    # produced by axvs and axci. The caller does not choose where behavior outputs go.
    data_path = session.processed_data.behavior_data_path
    data_path.mkdir(parents=True, exist_ok=True)
    tracker_path = session.processed_data.behavior_tracker_path

    if tracker_path.exists():
        # Idempotent path: returns existing tracker state without rebuilding the job registry.
        try:
            tracker_status = read_tracker_status(tracker_path=tracker_path)
        except Exception:
            tracker_status = {"jobs": [], "summary": {}}

        tracker_jobs_by_id = {
            ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier): (job_name, specifier)
            for job_name, specifier in discovered_jobs
        }

        # Augments each tracker entry with the enclosing dispatch metadata so the caller can feed the manifest
        # directly into the generic execute tool without re-deriving per-job fields.
        enriched_jobs: list[dict[str, Any]] = []
        for tracker_entry in tracker_status.get("jobs", []):
            entry_job_id = tracker_entry["job_id"]
            if entry_job_id not in tracker_jobs_by_id:
                continue
            job_name, specifier = tracker_jobs_by_id[entry_job_id]
            enriched_jobs.append(
                {
                    **tracker_entry,
                    "job_name": job_name,
                    "specifier": specifier,
                    "session_path": str(unit_path),
                    "session_name": session.session_name,
                    "tracker_path": str(tracker_path),
                }
            )

        return {
            "tracker_path": str(tracker_path),
            "data_path": str(data_path),
            "session_name": session.session_name,
            "jobs": enriched_jobs,
            "summary": tracker_status.get("summary", {}),
        }

    # Initializes a new tracker with jobs for the discovered (job_name, specifier) tuples. The helper autoloads on
    # construction so the ProcessingTracker created here is immediately backed by the YAML file once
    # initialize_jobs persists the registry.
    tracker = ProcessingTracker(file_path=tracker_path)
    tracker.initialize_jobs(jobs=discovered_jobs)

    # Derives each job's status from the just-initialized tracker rather than defaulting to SCHEDULED so the
    # manifest reflects any state the tracker already carried.
    try:
        tracker_status = read_tracker_status(tracker_path=tracker_path)
    except Exception:
        tracker_status = {"jobs": [], "summary": {}}
    status_by_id = {entry["job_id"]: entry for entry in tracker_status.get("jobs", [])}

    jobs: list[dict[str, Any]] = []
    for job_name, specifier in discovered_jobs:
        job_id = ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier)
        status_entry = status_by_id.get(job_id, {})
        entry = {
            "job_id": job_id,
            "job_name": job_name,
            "specifier": specifier,
            "status": status_entry.get("status", ProcessingStatus.SCHEDULED.name),
            "session_path": str(unit_path),
            "session_name": session.session_name,
            "tracker_path": str(tracker_path),
        }
        if status_entry.get("error_message") is not None:
            entry["error_message"] = status_entry["error_message"]
        jobs.append(entry)

    return {
        "tracker_path": str(tracker_path),
        "data_path": str(data_path),
        "session_name": session.session_name,
        "jobs": jobs,
        "summary": tracker_status.get(
            "summary",
            {
                "total": len(jobs),
                "succeeded": 0,
                "failed": 0,
                "running": 0,
                "scheduled": len(jobs),
            },
        ),
    }


def verify_behavior_unit(unit_path: Path) -> dict[str, Any]:
    """Verifies the completeness of processed behavior data output for a single session.

    Loads the session's ``SessionData`` marker to resolve ``processed_data_path``, scans the ``behavior_data/``
    subdirectory for feather files produced by the behavior processing pipeline, loads each to confirm
    readability, and reads the processing tracker. Owns the full verification result, including the ``verified``
    boolean and the ``tracker`` block.

    Args:
        unit_path: The path to the session root directory.

    Returns:
        A dictionary containing a ``verified`` flag, per-file results in ``files``, the static ``data_path``,
        tracker status in ``tracker``, and aggregate counts. On failure, returns a dictionary with an ``error``
        key.
    """
    try:
        session = SessionData.load(session_path=unit_path)
    except Exception as error:
        return {"error": f"Unable to load session: {error}"}

    data_path = session.processed_data.behavior_data_path

    if not data_path.exists():
        return {
            "error": (
                f"No '{Directories.BEHAVIOR_DATA}' subdirectory found under '{session.processed_data_path}'. "
                f"Processing may not have been run yet."
            ),
        }

    file_results: list[dict[str, Any]] = []
    all_valid = True

    feather_files = natsorted(data_path.glob("*.feather"))

    for feather_file in feather_files:
        # Reuses the shared feather inspector with zero sample rows, since verify only needs the row count and
        # column list. ``analyze_feather_file`` returns an ``error`` key when the file cannot be read.
        analysis = analyze_feather_file(feather_file=str(feather_file), max_sample_rows=0)
        entry: dict[str, Any] = {"file": str(feather_file), "filename": feather_file.name}

        if "error" in analysis:
            entry["valid"] = False
            entry["error"] = analysis["error"]
            all_valid = False
            file_results.append(entry)
            continue

        summary = analysis.get("summary", {})
        entry["valid"] = True
        entry["columns"] = summary.get("columns", [])
        entry["row_count"] = summary.get("total_rows", 0)
        file_results.append(entry)

    tracker_path = session.processed_data.behavior_tracker_path
    tracker_info: dict[str, Any] = {}
    if tracker_path.exists():
        try:
            tracker_info = read_tracker_status(tracker_path=tracker_path)
        except Exception:
            tracker_info = {"error": "Unable to read tracker file."}

    return {
        "verified": all_valid and bool(feather_files),
        "session_path": str(unit_path),
        "data_path": str(data_path),
        "files": file_results,
        "total_files": len(file_results),
        "tracker": tracker_info,
    }


def clean_behavior_unit(unit_path: Path) -> dict[str, Any]:
    """Deletes the ``behavior_data`` subdirectory under a single session's processed data directory.

    Loads ``SessionData`` to resolve ``processed_data_path``, then removes ``{processed_data_path}/behavior_data/``
    and all of its contents (processed feather files plus the processing tracker) via the shared cleanup helper.

    Args:
        unit_path: The path to the session root directory whose behavior processing output should be deleted.

    Returns:
        A dictionary carrying a ``cleaned`` flag and either ``data_path`` or ``error``, keyed by ``session_path``.
    """
    try:
        session = SessionData.load(session_path=unit_path)
    except Exception as error:
        return {"session_path": str(unit_path), "cleaned": False, "error": f"Unable to load session: {error}"}

    outcome = clean_output_subdirectory(
        output_directory=str(session.processed_data_path),
        subdirectory_name=Directories.BEHAVIOR_DATA,
    )
    # Rewrites the helper's ``output_directory`` key into ``session_path`` so that the MCP surface consistently
    # identifies each entry by its session root.
    outcome.pop("output_directory", None)
    return {"session_path": str(unit_path), **outcome}


def iterate_behavior_overview(root_directory: str) -> Iterator[dict[str, Any]]:
    """Yields per-session behavior status descriptors for every session under a root directory.

    Iterates every session marker under the root via ``iterate_sessions`` and, for each session whose canonical
    ``behavior_tracker_path`` exists, reads the tracker and yields a status descriptor.

    Args:
        root_directory: The absolute path to the root directory to search for sessions.

    Yields:
        Per-session status descriptor dictionaries carrying the session path, the data path, the tracker path,
        the derived status, and the full tracker status payload.
    """
    root_path = Path(root_directory)

    for session in iterate_sessions(root_path=root_path):
        tracker_path = session.processed_data.behavior_tracker_path
        if not tracker_path.is_file():
            continue

        data_path = session.processed_data.behavior_data_path
        session_root = session.raw_data_path.parent
        try:
            status = read_tracker_status(tracker_path=tracker_path)
            summary = status.get("summary", {})
            dir_status = derive_tracker_status(summary=summary)
            yield {
                "session_path": str(session_root),
                "data_path": str(data_path),
                "tracker_path": str(tracker_path),
                "status": dir_status,
                **status,
            }
        except Exception:
            yield {
                "session_path": str(session_root),
                "data_path": str(data_path),
                "tracker_path": str(tracker_path),
                "status": "error",
                "error": "Unable to read tracker file.",
            }


def run_behavior_job(job: GenericPendingJob) -> None:
    """Executes a single behavior processing job in-process via the pipeline's remote mode.

    Serves as the picklable worker callable stored on ``JobExecutionState`` and dispatched to the batch manager's
    ``ProcessPoolExecutor``. Delegates to ``run_behavior_processing_pipeline`` with the job's session path in
    remote mode so that only the single ``(job_name, specifier)`` pair identified by ``job.job_id`` is executed
    against the session. The output location is resolved inside the pipeline from the session's ``SessionData``
    marker.

    Args:
        job: The pending job descriptor carrying the session ``unit_path`` and the ``job_id`` to execute.
    """
    run_behavior_processing_pipeline(
        session_path=job.unit_path,
        job_id=job.job_id,
    )


# ----------------------------------------------------------------------------------------------------------------------
# Forging adapters
# ----------------------------------------------------------------------------------------------------------------------

FORGING_CONCURRENCY: ConcurrencyDescriptor = ConcurrencyDescriptor(cores_per_job=3, default_max_parallel=10)
"""The forging pipeline concurrency policy: three cores per worker (one subprocess plus a two-thread pool that
parallelizes behavior and runtime assembly), defaulting to ten concurrent jobs."""


def prepare_forging_unit(unit: dict[str, Any]) -> dict[str, Any]:
    """Resolves a dataset hierarchy and initializes its forging processing tracker.

    Wraps ``resolve_dataset`` (restricted to mesoscope experiment sessions), resolves the dataset directory and
    forging tracker, aligns the tracker with the session set, and returns the per-dataset manifest entry built
    directly from the in-memory tracker.

    Args:
        unit: The forging unit specification. Must carry a ``name`` (dataset name) and a ``project_root`` key, and
            may carry ``session_names`` (the session names to include; empty reuses an existing dataset) and a
            ``force_recreate`` flag.

    Returns:
        A per-dataset manifest dictionary carrying the dataset name, the dataset path, the project root, the
        tracker path, the enriched job descriptors, and the tracker summary.
    """
    name = unit["name"]
    project_root = Path(unit["project_root"])
    session_names = tuple(unit.get("session_names", ()))
    force_recreate = bool(unit.get("force_recreate", False))

    dataset = resolve_dataset(
        name=name,
        session_names=session_names,
        project_root=project_root,
        required_session_type=SessionTypes.MESOSCOPE_EXPERIMENT,
        force_recreate=force_recreate,
    )

    dataset_path = dataset.dataset_data_path.parent
    tracker_path = dataset_path / ProcessingTrackers.FORGING

    # Prepares the processing tracker and aligns it with the session set.
    tracker = ProcessingTracker(file_path=tracker_path)
    jobs_tuples = [(FORGING_JOB_NAME, entry.session) for entry in dataset.sessions]
    prepare_tracker(tracker=tracker, jobs=jobs_tuples, universe=jobs_tuples)

    # Builds enriched job descriptors directly from the in-memory tracker, which prepare_tracker just aligned.
    # This avoids a redundant YAML deserialization.
    session_by_job_id = {
        ProcessingTracker.generate_job_id(job_name=FORGING_JOB_NAME, specifier=entry.session): entry
        for entry in dataset.sessions
    }

    enriched_jobs: list[dict[str, Any]] = []
    succeeded_count = 0
    failed_count = 0
    running_count = 0
    scheduled_count = 0

    for job_id, job_state in tracker.jobs.items():
        if job_id not in session_by_job_id:
            continue
        session_entry = session_by_job_id[job_id]
        status = job_state.status

        if status == ProcessingStatus.SUCCEEDED:
            succeeded_count += 1
        elif status == ProcessingStatus.FAILED:
            failed_count += 1
        elif status == ProcessingStatus.RUNNING:
            running_count += 1
        else:
            scheduled_count += 1

        entry: dict[str, Any] = {
            "job_id": job_id,
            "job_name": job_state.job_name,
            "specifier": job_state.specifier,
            "status": status.name,
            "session_name": session_entry.session,
            "dataset_name": name,
            "project_root": str(project_root),
            "tracker_path": str(tracker_path),
        }
        if job_state.error_message is not None:
            entry["error_message"] = job_state.error_message
        enriched_jobs.append(entry)

    return {
        "dataset_path": str(dataset_path),
        "tracker_path": str(tracker_path),
        "dataset_name": name,
        "project_root": str(project_root),
        "jobs": enriched_jobs,
        "summary": {
            "total": len(tracker.jobs),
            "succeeded": succeeded_count,
            "failed": failed_count,
            "running": running_count,
            "scheduled": scheduled_count,
        },
    }


def verify_forging_unit(unit_path: Path) -> dict[str, Any]:
    """Verifies the completeness of forged data output for a single dataset.

    Loads the dataset's ``DatasetData`` marker, checks each session's ``data.feather`` and the per-session
    ``session_descriptor.yaml``, and each animal's ``surgery_metadata.yaml``. Reads the forging processing tracker.
    Owns the full verification result, including the ``verified`` boolean and the ``tracker`` block.

    Args:
        unit_path: The path to the dataset root directory (containing ``dataset.yaml``).

    Returns:
        A dictionary containing a ``verified`` flag, per-session results in ``files``, per-animal surgery results
        in ``animals``, tracker status in ``tracker``, and aggregate counts. On failure, returns a dictionary with
        an ``error`` key.
    """
    try:
        dataset = DatasetData.load(dataset_path=unit_path)
    except Exception as load_error:
        return {"error": f"Unable to load dataset: {load_error}"}

    file_results: list[dict[str, Any]] = []
    all_valid = True

    # Checks each session's feather file for existence and readability, plus the per-session experiment descriptor
    # for existence and parseability.
    for session_entry in dataset.sessions:
        data_path = session_entry.data_path
        entry: dict[str, Any] = {
            "session_name": session_entry.session,
            "animal": session_entry.animal,
            "file": str(data_path),
        }

        feather_valid = True
        if not data_path.exists():
            entry["valid"] = False
            entry["error"] = f"{DatasetFiles.DATA} not found."
            all_valid = False
            feather_valid = False
        else:
            # Loads the feather file without sampling to confirm readability and extract metadata.
            analysis = analyze_feather_file(feather_file=str(data_path), max_sample_rows=0)
            if "error" in analysis:
                entry["valid"] = False
                entry["error"] = analysis["error"]
                all_valid = False
                feather_valid = False
            else:
                summary = analysis.get("summary", {})
                entry["valid"] = True
                entry["columns"] = summary.get("columns", [])
                entry["row_count"] = summary.get("total_rows", 0)

        # Verifies the per-session experiment descriptor exists and parses as MesoscopeExperimentDescriptor.
        descriptor_path = session_entry.descriptor_path
        descriptor_entry: dict[str, Any] = {"file": str(descriptor_path)}
        if not descriptor_path.exists():
            descriptor_entry["valid"] = False
            descriptor_entry["error"] = f"{RawDataFiles.SESSION_DESCRIPTOR} not found."
            all_valid = False
        else:
            try:
                MesoscopeExperimentDescriptor.from_yaml(file_path=descriptor_path)
                descriptor_entry["valid"] = True
            except Exception as descriptor_error:
                descriptor_entry["valid"] = False
                descriptor_entry["error"] = f"Unable to load descriptor: {descriptor_error}"
                all_valid = False
        entry["descriptor"] = descriptor_entry

        # Promotes a feather-only valid flag to overall invalid when the descriptor failed.
        if feather_valid and not descriptor_entry["valid"]:
            entry["valid"] = False

        file_results.append(entry)

    # Verifies the per-animal surgery file for each unique animal in the dataset.
    animal_results: list[dict[str, Any]] = []
    for dataset_animal in dataset.animals:
        surgery_path = dataset_animal.surgery_path
        animal_entry: dict[str, Any] = {"animal": dataset_animal.animal, "file": str(surgery_path)}
        if not surgery_path.exists():
            animal_entry["valid"] = False
            animal_entry["error"] = f"{RawDataFiles.SURGERY_METADATA} not found."
            all_valid = False
        else:
            try:
                SurgeryData.from_yaml(file_path=surgery_path)
                animal_entry["valid"] = True
            except Exception as surgery_error:
                animal_entry["valid"] = False
                animal_entry["error"] = f"Unable to load surgery data: {surgery_error}"
                all_valid = False
        animal_results.append(animal_entry)

    # Reads the forging tracker to include per-job pipeline statuses alongside file checks.
    tracker_path = dataset.dataset_data_path.parent / ProcessingTrackers.FORGING
    tracker_info: dict[str, Any] = {}
    if tracker_path.exists():
        try:
            tracker_info = read_tracker_status(tracker_path=tracker_path)
        except Exception:
            tracker_info = {"error": "Unable to read tracker file."}

    return {
        "verified": all_valid and bool(file_results) and bool(animal_results),
        "dataset_path": str(unit_path),
        "dataset_name": dataset.name,
        "files": file_results,
        "total_files": len(file_results),
        "animals": animal_results,
        "total_animals": len(animal_results),
        "tracker": tracker_info,
    }


def clean_forging_unit(unit_path: Path) -> dict[str, Any]:
    """Deletes the full dataset hierarchy for a single dataset.

    Removes the entire dataset directory tree (tracker, dataset metadata, and all per-session ``data.feather``
    files). After cleanup, the same dataset specification can be passed back to the prepare tool to reinitialize
    from scratch.

    Args:
        unit_path: The path to the dataset root directory to delete.

    Returns:
        A dictionary carrying a ``cleaned`` flag and either a ``message`` or ``error``, keyed by ``dataset_path``.
    """
    if not unit_path.exists():
        return {"dataset_path": str(unit_path), "cleaned": True, "message": "Nothing to clean."}

    if not unit_path.is_dir():
        return {"dataset_path": str(unit_path), "cleaned": False, "error": "Path is not a directory."}

    try:
        delete_directory(directory_path=unit_path)
        return {"dataset_path": str(unit_path), "cleaned": True}
    except Exception as delete_error:
        return {"dataset_path": str(unit_path), "cleaned": False, "error": f"Unable to delete: {delete_error}"}


def iterate_forging_overview(root_directory: str) -> Iterator[dict[str, Any]]:
    """Yields per-dataset forging status descriptors for every dataset under a project root.

    Datasets live at the canonical ``<project_root>/<dataset_name>/`` layout and each holds its tracker at
    ``<dataset_root>/forging_tracker.yaml``. Iterates the top-level children of the root and reads the tracker from
    each candidate dataset directory rather than walking the whole project.

    Args:
        root_directory: The absolute path to the project root containing dataset directories.

    Yields:
        Per-dataset status descriptor dictionaries carrying the dataset name, the dataset path, the tracker path,
        the derived status, and the full tracker status payload.
    """
    root_path = Path(root_directory)

    # Iterates top-level children only; datasets are never nested under animals or sessions.
    for dataset_path in natsorted(root_path.iterdir()):
        if not dataset_path.is_dir():
            continue
        tracker_path = dataset_path.joinpath(ProcessingTrackers.FORGING)
        if not tracker_path.is_file():
            continue
        dataset_name = dataset_path.name
        try:
            status = read_tracker_status(tracker_path=tracker_path)
            summary = status.get("summary", {})
            dataset_status = derive_tracker_status(summary=summary)
            yield {
                "dataset_name": dataset_name,
                "dataset_path": str(dataset_path),
                "tracker_path": str(tracker_path),
                "status": dataset_status,
                **status,
            }
        except Exception:
            yield {
                "dataset_name": dataset_name,
                "dataset_path": str(dataset_path),
                "tracker_path": str(tracker_path),
                "status": "error",
                "error": "Unable to read tracker file.",
            }


def run_forging_job(job: GenericPendingJob) -> None:
    """Executes a single forging assembly job in-process via the pipeline's remote mode.

    Serves as the picklable worker callable stored on ``JobExecutionState`` and dispatched to the batch manager's
    ``ProcessPoolExecutor``. Delegates to ``run_forging_pipeline`` with an empty session list (the dataset is
    already materialized) and the job's ``job_id`` in remote mode so that only the single session identified by the
    job ID is assembled.

    Args:
        job: The pending job descriptor carrying the dataset ``name``, the ``project_root``, and the ``job_id``.
    """
    if job.project_root is None:
        message = "The forging worker requires a 'project_root' on the pending job, but none was provided."
        console.error(message=message, error=ValueError)
        raise ValueError(message)  # pragma: no cover

    run_forging_pipeline(
        name=job.name,
        session_names=(),
        project_root=job.project_root,
        job_id=job.job_id,
    )
