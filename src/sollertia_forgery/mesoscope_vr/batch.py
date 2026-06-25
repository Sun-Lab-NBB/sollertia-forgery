"""Provides the Mesoscope-VR behavior batch adapters that wire the behavior processing pipeline into the
system-agnostic processing tools: discovery/preparation, output verification, cleanup, status overview, the
concurrency descriptor, and the picklable per-job worker.

Notes:
    Only the behavior pipeline is system-specific and therefore adapted here. The forging pipeline is agnostic and
    owns its own batch adapters in the ``forging`` package, so this module no longer carries any forging adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path

from natsort import natsorted
from sollertia_shared_assets import Directories, SessionData, iterate_sessions
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from .processing import discover_behavior_jobs, run_behavior_processing_pipeline
from ..orchestration import (
    GenericPendingJob,
    ConcurrencyDescriptor,
    read_tracker_status,
    analyze_feather_file,
    derive_tracker_status,
    clean_output_subdirectory,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


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
