"""Provides the pipeline dispatch table that binds each session-processing pipeline to its Stage-2 job resolver, its
picklable batch worker, its concurrency policy, and its processing tracker. The generic batch tools consume this table
to prepare, execute, and monitor any session pipeline without per-pipeline branching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path
from dataclasses import dataclass

from ataraxis_data_structures import ProcessingTracker

from .local import GenericPendingJob, ConcurrencyDescriptor
from ..video import discover_video_jobs, run_video_processing_pipeline
from ..runtime import discover_runtime_jobs, run_runtime_processing_pipeline
from .pipelines import ProcessingPipelines
from ..two_photon import discover_two_photon_jobs, run_two_photon_processing_pipeline
from ..microcontrollers import discover_microcontroller_jobs, run_microcontroller_processing_pipeline

if TYPE_CHECKING:
    from collections.abc import Callable

    from sollertia_shared_assets import SessionData


def run_runtime_job(job: GenericPendingJob, *, cores_per_job: int) -> None:
    """Runs the runtime pipeline for one session as a batch job.

    Notes:
        The runtime pipeline is single-job, so it takes no job identifier. Every job the batch layer dispatches for
        it runs the whole runtime pipeline for its session.

    Args:
        job: The pending job carrying the session root in ``unit_path``.
        cores_per_job: The number of worker processes the runtime decode stage may use.
    """
    run_runtime_processing_pipeline(session_path=job.unit_path, workers=cores_per_job)


def run_microcontroller_job(job: GenericPendingJob, *, cores_per_job: int) -> None:
    """Runs a single microcontroller extraction or parse job for one session.

    Args:
        job: The pending job carrying the session root in ``unit_path`` and the target job in ``job_id``.
        cores_per_job: The number of worker processes the extraction stage may use.
    """
    run_microcontroller_processing_pipeline(session_path=job.unit_path, job_id=job.job_id, workers=cores_per_job)


def run_video_job(job: GenericPendingJob, *, cores_per_job: int) -> None:
    """Runs a single camera timestamp, rename, tracking, or motion-energy job for one session.

    Args:
        job: The pending job carrying the session root in ``unit_path`` and the target job in ``job_id``.
        cores_per_job: The number of worker processes the parse or motion-energy stage may use.
    """
    run_video_processing_pipeline(session_path=job.unit_path, job_id=job.job_id, workers=cores_per_job)


def run_two_photon_job(job: GenericPendingJob, *, cores_per_job: int) -> None:
    """Runs a single two-photon binarization, per-plane processing, or combination job for one session.

    Args:
        job: The pending job carrying the session root in ``unit_path`` and the target job in ``job_id``.
        cores_per_job: The number of numba worker threads cindra may use.
    """
    run_two_photon_processing_pipeline(session_path=job.unit_path, job_id=job.job_id, workers=cores_per_job)


_DEFAULT_CONCURRENCY: ConcurrencyDescriptor = ConcurrencyDescriptor(cores_per_job=1, default_max_parallel=-1)
"""The single-core policy for the runtime, microcontroller, and video pipelines. Each job occupies one core and may
run as widely as the resolved worker budget allows."""

_TWO_PHOTON_CONCURRENCY: ConcurrencyDescriptor = ConcurrencyDescriptor(cores_per_job=8, default_max_parallel=2)
"""The two-photon policy. Each cindra job is internally multi-core and memory-heavy, so it claims several cores and
caps concurrent jobs to bound the memory footprint. The exact values are hardware-dependent and safe to tune."""


@dataclass(frozen=True, slots=True)
class PipelineDispatch:
    """Binds a session-processing pipeline to the assets the generic batch tools drive it with.

    Notes:
        ``discover`` is the pipeline's Stage-2 job resolver, ``worker`` is the picklable callable the process pool
        dispatches per job, ``concurrency`` is the pipeline's core and parallel-job policy, and ``tracker_path``
        resolves the pipeline's processing tracker from a loaded session.
    """

    pipeline: ProcessingPipelines
    """The pipeline this entry dispatches."""
    discover: Callable[[Path], tuple[SessionData, list[tuple[str, str]], list[tuple[str, str]]]]
    """The Stage-2 job resolver returning the loaded session, the job universe, and the runnable subset."""
    worker: Callable[..., None]
    """The picklable module-level worker the process pool invokes with a single job and a keyword core count."""
    concurrency: ConcurrencyDescriptor
    """The pipeline's per-job core count and default concurrent-job cap."""
    tracker_path: Callable[[SessionData], Path]
    """Resolves the pipeline's processing tracker path from a loaded session."""


PIPELINE_DISPATCH: dict[ProcessingPipelines, PipelineDispatch] = {
    ProcessingPipelines.RUNTIME: PipelineDispatch(
        pipeline=ProcessingPipelines.RUNTIME,
        discover=discover_runtime_jobs,
        worker=run_runtime_job,
        concurrency=_DEFAULT_CONCURRENCY,
        tracker_path=lambda session: session.processed_data.runtime_tracker_path,
    ),
    ProcessingPipelines.MICROCONTROLLER: PipelineDispatch(
        pipeline=ProcessingPipelines.MICROCONTROLLER,
        discover=discover_microcontroller_jobs,
        worker=run_microcontroller_job,
        concurrency=_DEFAULT_CONCURRENCY,
        tracker_path=lambda session: session.processed_data.microcontroller_tracker_path,
    ),
    ProcessingPipelines.VIDEO: PipelineDispatch(
        pipeline=ProcessingPipelines.VIDEO,
        discover=discover_video_jobs,
        worker=run_video_job,
        concurrency=_DEFAULT_CONCURRENCY,
        tracker_path=lambda session: session.processed_data.video_tracker_path,
    ),
    ProcessingPipelines.TWO_PHOTON: PipelineDispatch(
        pipeline=ProcessingPipelines.TWO_PHOTON,
        discover=discover_two_photon_jobs,
        worker=run_two_photon_job,
        concurrency=_TWO_PHOTON_CONCURRENCY,
        tracker_path=lambda session: session.processed_data.two_photon_tracker_path,
    ),
}
"""The dispatch entry for every session-processing pipeline the generic batch tools support, keyed by pipeline."""

BATCH_PIPELINES: frozenset[ProcessingPipelines] = frozenset(PIPELINE_DISPATCH)
"""The set of pipelines the generic batch tools support, derived from the dispatch table."""


def resolve_dispatch(pipeline: str | ProcessingPipelines) -> PipelineDispatch | None:
    """Resolves the dispatch entry for a batch pipeline, or None when the identifier is not a supported batch pipeline.

    Args:
        pipeline: The pipeline identifier, either a ProcessingPipelines member or its string value.

    Returns:
        The pipeline's dispatch entry, or None if the identifier is unknown or names a non-batch pipeline.
    """
    try:
        member = ProcessingPipelines(pipeline)
    except ValueError:
        return None
    return PIPELINE_DISPATCH.get(member)


def prepare_pipeline_jobs(dispatch: PipelineDispatch, session_path: Path) -> dict[str, Any]:
    """Discovers a session's runnable jobs, aligns the pipeline tracker, and returns the job descriptors.

    Notes:
        Discovery loads the session and resolves the job universe and its runnable subset through the pipeline's
        Stage-2 resolver. The tracker is aligned against the runnable subset within the universe, so a partial run
        neither wipes sibling jobs nor loses the full-configuration fingerprint. The returned descriptors carry
        everything the execute tool needs to dispatch each job.

    Args:
        dispatch: The pipeline's dispatch entry.
        session_path: The path to the session root to discover jobs for.

    Returns:
        A dictionary with the session name, the tracker path, and a list of job descriptors, each carrying ``job_id``,
        ``job_name``, ``specifier``, ``session_path``, and ``tracker_path``.
    """
    session, universe, runnable = dispatch.discover(session_path)
    tracker_path = dispatch.tracker_path(session)
    tracker_path.parent.mkdir(parents=True, exist_ok=True)
    tracker = ProcessingTracker(file_path=tracker_path)
    tracker.align_jobs(jobs=runnable, universe=universe)

    jobs = [
        {
            "job_id": ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier),
            "job_name": job_name,
            "specifier": specifier,
            "session_path": str(session_path),
            "tracker_path": str(tracker_path),
        }
        for job_name, specifier in runnable
    ]
    return {"session_name": session.session_name, "tracker_path": str(tracker_path), "jobs": jobs}


def build_pending_job(job: dict[str, Any]) -> GenericPendingJob:
    """Builds a GenericPendingJob from a job descriptor emitted by ``prepare_pipeline_jobs``.

    Args:
        job: A job descriptor carrying ``tracker_path``, ``job_id``, and ``session_path``, and optionally
            ``job_name`` and ``specifier``.

    Returns:
        The pending job the batch engine dispatches to a worker.
    """
    return GenericPendingJob(
        tracker_path=Path(job["tracker_path"]),
        job_id=job["job_id"],
        unit_path=Path(job["session_path"]),
        job_name=job.get("job_name", ""),
        specifier=job.get("specifier", ""),
    )
