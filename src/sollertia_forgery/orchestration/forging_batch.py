"""Provides the system-agnostic forging batch adapters: dataset discovery/preparation, output verification, cleanup,
status overview, the concurrency descriptor, and the picklable per-session worker.

Notes:
    These adapters wrap the agnostic forging pipeline for batch/MCP-style execution. They operate only on the
    system-agnostic dataset hierarchy (``DatasetData``), the forging processing tracker, and the shared assets the
    forging pipeline re-exports, so they carry no acquisition-system coupling. They live in the shared
    ``orchestration`` package alongside the generic batch engine they plug into (the concurrency descriptor, the
    process-pool job manager, and the tracker helpers), and resolve any system-specific behavior through the registry
    rather than importing a system package. They are imported on demand by the batch-registry wiring rather than from
    ``orchestration/__init__`` so the generic engine's public surface stays decoupled from this forging-specific
    adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path

from natsort import natsorted
from ataraxis_base_utilities import console
from sollertia_shared_assets import (
    DatasetData,
    SurgeryData,
    DatasetFiles,
    RawDataFiles,
    ProcessingTrackers,
)
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker, delete_directory

from .local import (
    ConcurrencyDescriptor,
    read_tracker_status,
    analyze_feather_file,
    derive_tracker_status,
)
from ..forging import resolve_dataset
from ..shared_assets import prepare_tracker
from ..forging.pipeline import FORGING_JOB_NAME, run_forging_pipeline

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .local import GenericPendingJob


FORGING_CONCURRENCY: ConcurrencyDescriptor = ConcurrencyDescriptor(cores_per_job=3, default_max_parallel=10)
"""The forging pipeline concurrency policy: three cores per worker (one subprocess plus a two-thread pool that
parallelizes the behavior and runtime sub-dataset assembly inside the donated worker), defaulting to ten concurrent
jobs."""


def prepare_forging_unit(unit: dict[str, Any]) -> dict[str, Any]:
    """Resolves a dataset hierarchy and initializes its forging processing tracker.

    Wraps ``resolve_dataset``, resolves the dataset directory and forging tracker, aligns the tracker with the
    session set, and returns the per-dataset manifest entry built directly from the in-memory tracker.

    Args:
        unit: The forging unit specification. Must carry a ``name`` (dataset name) and a ``project_root`` key, and
            may carry ``session_names`` (the session names to include, an empty list reuses an existing dataset) and a
            ``force_recreate`` flag.

    Returns:
        A per-dataset manifest dictionary carrying the dataset name, the dataset path, the project root, the tracker
        path, the enriched job descriptors, and the tracker summary.
    """
    name = unit["name"]
    project_root = Path(unit["project_root"])
    session_names = tuple(unit.get("session_names", ()))
    force_recreate = bool(unit.get("force_recreate", False))

    dataset = resolve_dataset(
        name=name, session_names=session_names, project_root=project_root, force_recreate=force_recreate
    )

    dataset_path = dataset.dataset_data_path.parent
    tracker_path = dataset_path / ProcessingTrackers.FORGING

    # Prepares the processing tracker and aligns it with the session set.
    tracker = ProcessingTracker(file_path=tracker_path)
    jobs_tuples = [(FORGING_JOB_NAME, entry.session) for entry in dataset.sessions]
    prepare_tracker(tracker=tracker, jobs=jobs_tuples, universe=jobs_tuples)

    # Builds enriched job descriptors directly from the in-memory tracker, which prepare_tracker just aligned. This
    # avoids a redundant YAML deserialization.
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

    Loads the dataset's ``DatasetData`` marker and checks each session's ``data.feather`` (existence and
    readability), confirms every column it writes is described in the dataset's ``data_descriptions.feather``, the
    re-exported shared assets, and each animal's ``surgery_metadata.yaml``. The ``session_descriptor.yaml`` is
    required for every session. The optional ``vr_configuration.yaml`` and ``experiment_configuration.yaml`` are
    reported per session but do not fail verification when absent, since only some session types carry them. Reads
    the forging processing tracker. Owns the full verification result, including the ``verified`` boolean and the
    ``tracker`` block. The check is system-agnostic: it validates only the universal output contract, the
    column-description coverage, and the shared assets, not any system-specific per-session artifact.

    Args:
        unit_path: The path to the dataset root directory (containing ``dataset.yaml``).

    Returns:
        A dictionary containing a ``verified`` flag, a ``descriptions_present`` flag, per-session results in
        ``files`` (each carrying an ``undescribed_columns`` list when a session writes columns absent from the
        dataset's ``data_descriptions.feather``), per-animal surgery results in ``animals``, tracker status in
        ``tracker``, and aggregate counts. On failure, returns a dictionary with an ``error`` key.
    """
    try:
        dataset = DatasetData.load(dataset_path=unit_path)
    except Exception as load_error:
        return {"error": f"Unable to load dataset: {load_error}"}

    # Resolves the per-dataset column-description mapping once so each session's columns can be checked for
    # description coverage. A missing companion feather is a dataset-level defect surfaced here rather than per
    # session, and disables (without crashing) the per-session coverage check below.
    descriptions_present = True
    described_columns: set[str] = set()
    try:
        described_columns = set(dataset.column_descriptions())
    except Exception:
        descriptions_present = False

    file_results: list[dict[str, Any]] = []
    all_valid = descriptions_present

    for session_entry in dataset.sessions:
        data_path = session_entry.data_path
        entry: dict[str, Any] = {
            "session_name": session_entry.session,
            "animal": session_entry.animal,
            "file": str(data_path),
        }

        session_valid = True
        if not data_path.exists():
            entry["valid"] = False
            entry["error"] = f"{DatasetFiles.DATA} not found."
            session_valid = False
        else:
            analysis = analyze_feather_file(feather_file=str(data_path), max_sample_rows=0)
            if "error" in analysis:
                entry["valid"] = False
                entry["error"] = analysis["error"]
                session_valid = False
            else:
                summary = analysis.get("summary", {})
                columns = summary.get("columns", [])
                entry["valid"] = True
                entry["columns"] = columns
                entry["row_count"] = summary.get("total_rows", 0)

                # Enforces the dataset's data-description contract for this session: every column written into the
                # session's data.feather must have an entry in the dataset's data_descriptions.feather. Undescribed
                # columns mark the session (and the dataset) invalid. Skipped when the descriptions feather is
                # absent, which is already recorded as a dataset-level defect.
                if descriptions_present:
                    undescribed_columns = sorted(set(columns) - described_columns)
                    if undescribed_columns:
                        entry["undescribed_columns"] = undescribed_columns
                        session_valid = False

        # Verifies the re-exported shared assets alongside the assembled feather. The session descriptor is part of
        # the universal output contract and is required for every session. The VR and experiment configurations are
        # re-exported only for the session types that carry them, so they are reported per session but never fail
        # verification when absent.
        shared_results: dict[str, Any] = {}
        descriptor_present = session_entry.descriptor_path.is_file()
        shared_results[str(RawDataFiles.SESSION_DESCRIPTOR)] = descriptor_present
        if not descriptor_present:
            session_valid = False
        for filename, asset_path in (
            (RawDataFiles.VR_CONFIGURATION, session_entry.vr_configuration_path),
            (RawDataFiles.EXPERIMENT_CONFIGURATION, session_entry.experiment_configuration_path),
        ):
            shared_results[str(filename)] = asset_path.is_file()
        entry["shared_assets"] = shared_results

        if not session_valid:
            entry["valid"] = False
            all_valid = False
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
        "descriptions_present": descriptions_present,
        "files": file_results,
        "total_files": len(file_results),
        "animals": animal_results,
        "total_animals": len(animal_results),
        "tracker": tracker_info,
    }


def clean_forging_unit(unit_path: Path) -> dict[str, Any]:
    """Deletes the full dataset hierarchy for a single dataset.

    Removes the entire dataset directory tree (tracker, dataset metadata, and all per-session output files). After
    cleanup, the same dataset specification can be passed back to the prepare adapter to reinitialize from scratch.

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
        Per-dataset status descriptor dictionaries carrying the dataset name, the dataset path, the tracker path, the
        derived status, and the full tracker status payload.
    """
    root_path = Path(root_directory)

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

    Serves as the picklable worker callable dispatched to the batch manager's ``ProcessPoolExecutor``. Delegates to
    ``run_forging_pipeline`` with an empty session list (the dataset is already materialized) and the job's
    ``job_id`` in remote mode so only the single session identified by the job ID is assembled.

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
