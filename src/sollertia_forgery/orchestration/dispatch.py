"""Provides the pipeline dispatch table that binds each session-processing pipeline to its job resolver, its
picklable batch worker, its job ordering, and its processing tracker, alongside the cores each job type occupies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path
from functools import cache
from dataclasses import dataclass

from cindra import SingleRecordingJobNames
from ataraxis_base_utilities import console, resolve_worker_count
from ataraxis_data_structures import ProcessingTracker

from .local import RESERVED_CORES, GenericPendingJob
from ..video import (
    ENERGY_JOB_NAME,
    RENAME_JOB_NAME,
    TRACKING_JOB_NAME,
    TIMESTAMP_JOB_NAME,
    discover_video_jobs,
    video_job_prerequisites,
    run_video_processing_pipeline,
)
from ..runtime import (
    RUNTIME_JOB_NAME,
    discover_runtime_jobs,
    runtime_job_prerequisites,
    run_runtime_processing_pipeline,
)
from ..managing import (
    CHECKSUM_JOB_NAME,
    discover_checksum_jobs,
    checksum_job_prerequisites,
    run_checksum_processing_pipeline,
)
from .pipelines import ProcessingPipelines
from .footprints import estimate_session_job_memory
from ..two_photon import (
    discover_two_photon_jobs,
    two_photon_job_prerequisites,
    materialize_cindra_configuration,
    run_two_photon_processing_pipeline,
)
from ..microcontrollers import (
    PARSE_JOB_NAME,
    EXTRACTION_JOB_NAME,
    discover_microcontroller_jobs,
    microcontroller_job_prerequisites,
    run_microcontroller_processing_pipeline,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from sollertia_shared_assets import SessionData


BATCH_PIPELINES: frozenset[ProcessingPipelines] = frozenset(
    {
        ProcessingPipelines.CHECKSUM,
        ProcessingPipelines.RUNTIME,
        ProcessingPipelines.MICROCONTROLLER,
        ProcessingPipelines.VIDEO,
        ProcessingPipelines.TWO_PHOTON,
    }
)
"""The pipelines the generic batch tools support. An import-time check holds this to the dispatch table."""


_JOB_CORE_ALLOCATIONS: dict[str, int] = {
    # Hashes one file per worker, streaming each in fixed chunks, so the stage is bound by how fast the storage
    # delivers bytes rather than by how fast a core hashes them.
    CHECKSUM_JOB_NAME: 8,
    # Decode falls back to a serial path on any archive below its own parallelism threshold, and a runtime archive
    # always is one, so the job is single-core by construction.
    RUNTIME_JOB_NAME: 1,
    # Splits its archive across a worker pool. Every worker re-opens the archive, so the fixed cost per worker does
    # not shrink as workers are added and the speedup flattens well before the core count.
    EXTRACTION_JOB_NAME: 8,
    # A single pass over one module's extracted table, short enough that pool dispatch dominates the work itself.
    PARSE_JOB_NAME: 1,
    # Splits its archive across a worker pool on the same fixed-cost-per-worker terms as controller extraction.
    TIMESTAMP_JOB_NAME: 8,
    # A fixed handful of filesystem operations, independent of recording length.
    RENAME_JOB_NAME: 1,
    # Reads pose predictions written upstream and never runs inference. Its ellipse fit solves once per distinct
    # occlusion pattern rather than once per sample, so its cost holds steady as a recording lengthens.
    TRACKING_JOB_NAME: 1,
    # Decodes the recording in parallel chunks. Throughput is bound by how fast frames move through memory rather
    # than by cores, so it saturates while cores remain. The chunk count is separately bounded by the recording's own
    # length, which caps a short recording below this allocation.
    ENERGY_JOB_NAME: 16,
    # cindra does not consume the worker count for this stage, and it converts a recording into a binary of similar
    # size, so its pace is set by write throughput rather than by cores.
    str(SingleRecordingJobNames.BINARIZE): 2,
    # The only two-photon stage that consumes the worker count, applied as numba threads. cindra documents no benefit
    # past roughly twenty threads per plane and recommends parallelizing across planes instead.
    str(SingleRecordingJobNames.PROCESS): 16,
    # A single-threaded concatenation over every plane's extracted traces.
    str(SingleRecordingJobNames.COMBINE): 1,
}
"""The cores one job of each type occupies, keyed by the tracker job name. Each value follows from how that stage
parallelizes, and every value is safe to retune. Preparing a session that resolves a job type absent from this map
fails for that session, since dispatching it would run it at a width nobody chose."""


@dataclass(frozen=True, slots=True)
class PipelineDispatch:
    """Binds a session-processing pipeline to the assets the generic batch tools drive it with.

    Notes:
        ``discover`` is the pipeline's job resolver, ``worker`` is the picklable callable the process pool
        dispatches per job, ``prerequisites`` is the pipeline's own intra-pipeline job ordering, and ``tracker_path``
        resolves the pipeline's processing tracker from a loaded session. Cores belong to the job type rather than
        to the pipeline, since one pipeline mixes job types that parallelize very differently, so they come from
        ``_JOB_CORE_ALLOCATIONS``.
    """

    pipeline: ProcessingPipelines
    """The pipeline this entry dispatches."""
    discover: Callable[[Path], tuple[SessionData, list[tuple[str, str]], list[tuple[str, str]]]]
    """The job resolver returning the loaded session, the job universe, and the runnable subset."""
    worker: Callable[..., None]
    """The picklable module-level worker the process pool invokes with a single planned job."""
    prerequisites: Callable[[list[tuple[str, str]]], dict[tuple[str, str], tuple[tuple[str, str], ...]]]
    """Resolves each job's upstream jobs, producing the ordering the batch engine dispatches jobs in."""
    tracker_path: Callable[[SessionData], Path]
    """Resolves the pipeline's processing tracker path from a loaded session."""
    output_path: Callable[[SessionData], Path | None]
    """Resolves the directory this pipeline owns outright, which is what a cleanup may remove to return the session
    to its unprocessed state. Resolves to None for a pipeline that writes into a directory it shares with the
    acquired data, since removing that directory would take the inputs with it."""


def run_batch_job(job: GenericPendingJob) -> None:
    """Runs one prepared job of any pipeline, routing on the pipeline the job carries.

    Notes:
        This is the single picklable entry point the shared pool dispatches, which is what lets one pool hold jobs
        from every pipeline at once. Each pipeline's own worker is looked up rather than bound into the job, so the
        descriptor stays a plain data record that pickles cheaply.

    Args:
        job: The pending job carrying its pipeline, its target job identifier, and its planned cores.

    Raises:
        ValueError: If the job names a pipeline the dispatch table does not support.
    """
    dispatch = resolve_dispatch(pipeline=job.pipeline)
    if dispatch is None:
        message = (
            f"Unable to run batch job '{job.job_id}'. The job names pipeline '{job.pipeline}', which is not a "
            f"supported batch pipeline."
        )
        console.error(message=message, error=ValueError)
    dispatch.worker(job)


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
    return _pipeline_dispatch().get(member)


def prepare_pipeline_jobs(
    dispatch: PipelineDispatch, session_path: Path, options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Discovers a session's runnable jobs, aligns the pipeline tracker, and returns the job descriptors.

    Notes:
        Discovery loads the session and resolves the job universe and its runnable subset through the pipeline's
        job resolver. The tracker is aligned against the runnable subset within the universe, so a partial run
        neither wipes sibling jobs nor discards the recorded state of any job the pipeline can still produce. The
        returned descriptors carry everything the execute tool needs to dispatch each job.

        Any options the caller supplies are stamped onto every descriptor unchanged and reach the pipeline's worker
        at dispatch. They do not reach discovery, so the jobs a session resolves stay a property of the data on disk
        rather than of the parameters a run was launched with. That keeps one tracker slot per job however the job
        is parameterized, which is what lets a multimode pipeline record one integrity state per session.

    Args:
        dispatch: The pipeline's dispatch entry.
        session_path: The path to the session root to discover jobs for.
        options: The pipeline-specific parameters to run these jobs with, such as the mode a multi-mode pipeline
            runs in. Pipelines that take no parameters ignore this mapping.

    Returns:
        A dictionary with the session name, the tracker path, and a list of job descriptors, each carrying
        ``job_id``, ``job_name``, ``specifier``, ``session_path``, ``tracker_path``, ``pipeline``, its allocated
        ``cores``, its estimated ``memory_mb``, a ``memory_modeled`` flag, ``prerequisite_ids``, and ``options``.
    """
    session, universe, runnable = dispatch.discover(session_path)
    tracker_path = dispatch.tracker_path(session)
    tracker_path.parent.mkdir(parents=True, exist_ok=True)
    tracker = ProcessingTracker(file_path=tracker_path)
    tracker.align_jobs(jobs=runnable, universe=universe)

    # Resolves the pipeline's own job ordering over the full universe, so a job's upstream stages are named even when
    # this batch does not queue them. The engine then treats an unqueued prerequisite as satisfied only if the
    # tracker already records it as succeeded.
    ordering = dispatch.prerequisites(universe)

    # cindra reads its worker count from the session's configuration rather than from a call argument, and only its
    # per-plane processing stage consumes it, so that stage's allocation is written once here, before any job of the
    # session dispatches. The host's own core count bounds it, since a session prepared on a larger machine would
    # otherwise name more threads than this one can run.
    if dispatch.pipeline is ProcessingPipelines.TWO_PHOTON:
        materialize_cindra_configuration(
            session=session,
            workers=min(
                _JOB_CORE_ALLOCATIONS[str(SingleRecordingJobNames.PROCESS)],
                resolve_worker_count(requested_workers=-1, reserved_cores=RESERVED_CORES),
            ),
        )

    # Sizes each job from the data it will process, at the cores its type is allocated, so the same figures drive
    # local admission and any remote submission that reads the descriptor.
    unregistered = sorted({job_name for job_name, _ in runnable if job_name not in _JOB_CORE_ALLOCATIONS})
    if unregistered:
        message = (
            f"Unable to prepare {dispatch.pipeline.value} jobs for session '{session.session_name}'. No core "
            f"allocation is registered for job type(s) {unregistered}. Every job type a pipeline resolves must "
            f"declare the cores one of its jobs occupies in _JOB_CORE_ALLOCATIONS."
        )
        console.error(message=message, error=ValueError)
    cores = {job_name: _JOB_CORE_ALLOCATIONS[job_name] for job_name, _ in runnable}
    memory = estimate_session_job_memory(
        pipeline=dispatch.pipeline,
        session=session,
        jobs=[(job_name, specifier, cores[job_name]) for job_name, specifier in runnable],
    )

    jobs = [
        {
            "job_id": ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier),
            "job_name": job_name,
            "specifier": specifier,
            "session_path": str(session_path),
            "tracker_path": str(tracker_path),
            "pipeline": dispatch.pipeline.value,
            "cores": cores[job_name],
            "memory_mb": memory.get((job_name, specifier), (0, False))[0],
            "memory_modeled": memory.get((job_name, specifier), (0, False))[1],
            "prerequisite_ids": [
                ProcessingTracker.generate_job_id(job_name=upstream_name, specifier=upstream_specifier)
                for upstream_name, upstream_specifier in ordering.get((job_name, specifier), ())
            ],
            "options": dict(options or {}),
        }
        for job_name, specifier in runnable
    ]
    return {"session_name": session.session_name, "tracker_path": str(tracker_path), "jobs": jobs}


def build_pending_job(job: dict[str, Any]) -> GenericPendingJob:
    """Builds a GenericPendingJob from a job descriptor emitted by ``prepare_pipeline_jobs``.

    Args:
        job: A job descriptor carrying ``tracker_path``, ``job_id``, ``session_path``, ``cores``, and
            ``memory_mb``, and optionally ``job_name``, ``specifier``, ``pipeline``, ``prerequisite_ids``, and
            ``options``.

    Returns:
        The pending job the batch engine dispatches to a worker.

    Raises:
        KeyError: If the descriptor omits a field the engine requires.
    """
    return GenericPendingJob(
        tracker_path=Path(job["tracker_path"]),
        job_id=job["job_id"],
        unit_path=Path(job["session_path"]),
        job_name=job.get("job_name", ""),
        specifier=job.get("specifier", ""),
        pipeline=job.get("pipeline", ""),
        core_weight=int(job["cores"]),
        memory_mb=int(job["memory_mb"]),
        prerequisite_ids=tuple(job.get("prerequisite_ids", ())),
        options=dict(job.get("options") or {}),
    )


def _run_checksum_job(job: GenericPendingJob) -> None:
    """Runs the raw-data integrity pipeline for one session as a batch job.

    Notes:
        The checksum pipeline is single-job, so it takes no job identifier. Its mode rides on the job's options,
        where ``regenerate_checksum`` selects re-baselining over verification. An absent key verifies, which is the
        mode a batch wants by default, since re-baselining is a deliberate correction rather than a routine pass.

    Args:
        job: The pending job carrying the session root in ``unit_path``, its planned cores in ``core_weight``, and
            its mode in ``options``.
    """
    run_checksum_processing_pipeline(
        session_path=job.unit_path,
        regenerate_checksum=bool(job.options.get("regenerate_checksum", False)),
        workers=job.core_weight,
    )


def _run_runtime_job(job: GenericPendingJob) -> None:
    """Runs the runtime pipeline for one session as a batch job.

    Notes:
        The runtime pipeline is single-job, so it takes no job identifier. Every job the batch layer dispatches for
        it runs the whole runtime pipeline for its session.

    Args:
        job: The pending job carrying the session root in ``unit_path`` and its planned cores in ``core_weight``.
    """
    run_runtime_processing_pipeline(session_path=job.unit_path, workers=job.core_weight)


def _run_microcontroller_job(job: GenericPendingJob) -> None:
    """Runs a single microcontroller extraction or parse job for one session.

    Args:
        job: The pending job carrying the session root in ``unit_path``, the target job in ``job_id``, and its
            planned cores in ``core_weight``.
    """
    run_microcontroller_processing_pipeline(session_path=job.unit_path, job_id=job.job_id, workers=job.core_weight)


def _run_video_job(job: GenericPendingJob) -> None:
    """Runs a single camera timestamp, rename, tracking, or motion-energy job for one session.

    Args:
        job: The pending job carrying the session root in ``unit_path``, the target job in ``job_id``, and its
            planned cores in ``core_weight``.
    """
    run_video_processing_pipeline(session_path=job.unit_path, job_id=job.job_id, workers=job.core_weight)


def _run_two_photon_job(job: GenericPendingJob) -> None:
    """Runs a single two-photon binarization, per-plane processing, or combination job for one session.

    Notes:
        cindra reads its thread count from the session's configuration, which the preparation step wrote, so the
        job's own core weight bounds what the batch admits rather than what cindra runs.

    Args:
        job: The pending job carrying the session root in ``unit_path`` and the target job in ``job_id``.
    """
    run_two_photon_processing_pipeline(session_path=job.unit_path, job_id=job.job_id, workers=job.core_weight)


@cache
def _pipeline_dispatch() -> dict[ProcessingPipelines, PipelineDispatch]:
    """Builds the dispatch entry for every session-processing pipeline the generic batch tools support.

    Notes:
        Built on first use rather than at import, so the workers it binds are ordinary private definitions rather
        than names the module has to define ahead of its own constants. The result is cached, so every caller shares
        one table.

    Returns:
        The dispatch entry for each supported pipeline, keyed by pipeline.
    """
    return {
        ProcessingPipelines.CHECKSUM: PipelineDispatch(
            pipeline=ProcessingPipelines.CHECKSUM,
            discover=discover_checksum_jobs,
            worker=_run_checksum_job,
            prerequisites=checksum_job_prerequisites,
            tracker_path=lambda session: session.raw_data.checksum_tracker_path,
            # Writes its stored checksum into raw_data, which holds the acquired data itself, so it owns no
            # directory a cleanup may remove.
            output_path=lambda _session: None,
        ),
        ProcessingPipelines.RUNTIME: PipelineDispatch(
            pipeline=ProcessingPipelines.RUNTIME,
            discover=discover_runtime_jobs,
            worker=_run_runtime_job,
            prerequisites=runtime_job_prerequisites,
            tracker_path=lambda session: session.processed_data.runtime_tracker_path,
            output_path=lambda session: session.processed_data.runtime_data_path,
        ),
        ProcessingPipelines.MICROCONTROLLER: PipelineDispatch(
            pipeline=ProcessingPipelines.MICROCONTROLLER,
            discover=discover_microcontroller_jobs,
            worker=_run_microcontroller_job,
            prerequisites=microcontroller_job_prerequisites,
            tracker_path=lambda session: session.processed_data.microcontroller_tracker_path,
            output_path=lambda session: session.processed_data.microcontroller_data_path,
        ),
        ProcessingPipelines.VIDEO: PipelineDispatch(
            pipeline=ProcessingPipelines.VIDEO,
            discover=discover_video_jobs,
            worker=_run_video_job,
            prerequisites=video_job_prerequisites,
            tracker_path=lambda session: session.processed_data.video_tracker_path,
            output_path=lambda session: session.processed_data.video_data_path,
        ),
        ProcessingPipelines.TWO_PHOTON: PipelineDispatch(
            pipeline=ProcessingPipelines.TWO_PHOTON,
            discover=discover_two_photon_jobs,
            worker=_run_two_photon_job,
            prerequisites=two_photon_job_prerequisites,
            tracker_path=lambda session: session.processed_data.two_photon_tracker_path,
            output_path=lambda session: session.processed_data.cindra_data_path,
        ),
    }


def _assert_dispatch_coverage() -> None:
    """Verifies that every pipeline the batch tools advertise has a dispatch entry.

    Raises:
        RuntimeError: If a supported pipeline has no dispatch entry, or an entry names an unsupported pipeline.
    """
    entries = frozenset(_pipeline_dispatch())
    if entries != BATCH_PIPELINES:
        message = (
            f"Unable to validate the pipeline dispatch table. Every pipeline named in BATCH_PIPELINES must have a "
            f"dispatch entry and no entry may name a pipeline outside it, but the sets differ by "
            f"{sorted(member.value for member in entries ^ BATCH_PIPELINES)}."
        )
        console.error(message=message, error=RuntimeError)


_assert_dispatch_coverage()
