"""Provides the pipeline dispatch table that binds each processing pipeline to its job resolver, its picklable batch
worker, its job ordering, and its processing tracker, alongside the cores each job type occupies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path
from functools import cache
from dataclasses import dataclass

from ataraxis_base_utilities import console, resolve_worker_count
from sollertia_shared_assets import DatasetData, SessionData
from ataraxis_data_structures import ProcessingTracker

from .local import RESERVED_CORES, GenericPendingJob, apply_decode_thread_ceiling
from ..video import (
    ENERGY_JOB_NAME,
    RENAME_JOB_NAME,
    TRACKING_JOB_NAME,
    TIMESTAMP_JOB_NAME,
    discover_video_jobs,
    video_job_prerequisites,
    run_video_processing_pipeline,
)
from ..forging import (
    FORGING_JOB_NAME,
    MULTIDAY_DISCOVERY_JOB_NAME,
    MULTIDAY_EXTRACTION_JOB_NAME,
    FORGING_JOB_CONCURRENCY_LIMITS,
    forging_tracker_path,
    run_forging_pipeline,
    discover_forging_jobs,
    forging_job_prerequisites,
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
from .footprints import estimate_dataset_job_memory, estimate_session_job_memory
from ..two_photon import (
    SingleRecordingJobNames,
    discover_two_photon_jobs,
    two_photon_job_prerequisites,
    run_two_photon_processing_pipeline,
)
from ..shared_assets import ProcessingPipelines, resolve_session_tracker_path
from ..microcontrollers import (
    PARSE_JOB_NAME,
    EXTRACTION_JOB_NAME,
    discover_microcontroller_jobs,
    microcontroller_job_prerequisites,
    run_microcontroller_processing_pipeline,
)

if TYPE_CHECKING:
    from collections.abc import Callable


BATCH_PIPELINES: frozenset[ProcessingPipelines] = frozenset(
    {
        ProcessingPipelines.CHECKSUM,
        ProcessingPipelines.RUNTIME,
        ProcessingPipelines.MICROCONTROLLER,
        ProcessingPipelines.VIDEO,
        ProcessingPipelines.TWO_PHOTON,
        ProcessingPipelines.FORGING,
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
    # Decodes a compressed image set into a binary of comparable size. cindra reads each batch through one keyed call
    # and leaves the decode width to the reader, so the cores this job holds become the threads that decode it.
    str(SingleRecordingJobNames.BINARIZE): 4,
    # Removes motion from one plane and computes its registration-quality components. Its pass over the plane holds
    # every thread busy, and its gain flattens once the batch it aligns stops covering the added cores.
    str(SingleRecordingJobNames.REGISTER): 8,
    # Discovers regions and extracts their fluorescence for one plane. Detection is bound by movie binning and a
    # serial loop, so the stage plateaus while cores remain and running more planes at once pays better.
    str(SingleRecordingJobNames.PROCESS): 10,
    # A single-threaded concatenation over every plane's extracted traces.
    str(SingleRecordingJobNames.COMBINE): 1,
    # Registers an animal's recordings against each other across a thread pool it sizes from this allocation, and
    # keeps every thread busy for the whole pass.
    MULTIDAY_DISCOVERY_JOB_NAME: 30,
    # Gathers each tracked region's pixels through a numba kernel that parallelizes over regions. Every batch the
    # kernel consumes is read serially before it runs, so the stage plateaus well below the width it is given.
    MULTIDAY_EXTRACTION_JOB_NAME: 16,
    # Reads its session's arrays and feathers and writes the merged result. Its own fan-out is a fixed handful of
    # threads, so the stage gains nothing from a wider allocation.
    FORGING_JOB_NAME: 1,
}
"""The cores one job of each type occupies, keyed by the tracker job name. Each value follows from how that stage
parallelizes, and every value is safe to retune. Preparing a session that resolves a job type absent from this map
fails for that session, since dispatching it would run it at a width nobody chose."""


_JOB_CONCURRENCY_LIMITS: dict[str, int] = {
    # Decoder throughput across the host peaks near forty-eight concurrent decoders and falls away past it, and this
    # stage opens one decoder per core it holds. Three jobs at its core allocation sit on that peak, so this ceiling
    # keeps the stage at its best aggregate rate while leaving the cores it would otherwise idle to other work.
    ENERGY_JOB_NAME: 3,
    # Sits at the root of the two-photon chain, so each job that finishes releases that recording's plane jobs. Four
    # run at their full decode rate, which is what returns those plane jobs to the queue soonest.
    str(SingleRecordingJobNames.BINARIZE): 4,
    # The forging pipeline declares its own ceilings, since it owns their job names.
    **FORGING_JOB_CONCURRENCY_LIMITS,
}
"""The jobs of each type that may run at once regardless of the cores the budget could still supply, keyed by the
tracker job name.

Notes:
    This is an admission term separate from the two budgets, and it is a hard ceiling. The budgets bound what the
    host can supply, while this bounds a job type whose own throughput stops climbing before its cores run out. A
    type at the root of a dependency chain takes a ceiling on the same terms, since finishing one root releases the
    stages waiting on it while spreading the same capacity over more roots delays all of them equally.

    Spare cores and spare memory never lift a ceiling recorded here, since a type held by one waits on something the
    spare capacity does not supply. Job types that convert spare capacity into progress belong in
    ``_JOB_CONCURRENCY_RESERVATIONS`` instead.

    A job type absent from this map is limited by the budgets alone. Every value is safe to retune, and a value below
    one is raised to one so a limit can never stall a batch.
"""


_JOB_CONCURRENCY_RESERVATIONS: dict[str, int] = {
    # Gates the plane that waits on it, so holding a share back keeps the stages that wait on no other job running
    # while a recording's planes are still being registered. It converts spare capacity into progress, so the hold is
    # released whenever nothing else can use what it gives up.
    str(SingleRecordingJobNames.REGISTER): 4,
    # Holds back part of the core budget so the stages that wait on no other job keep a share of the host while this
    # one runs. Its cores are the batch's scarcest resource once the two-photon chain opens, and it converts spare
    # capacity into progress, so the hold is released whenever nothing else can use what it gives up.
    str(SingleRecordingJobNames.PROCESS): 5,
}
"""The jobs of each type that run at once while other work can still use the capacity the type gives up, keyed by
the tracker job name.

Notes:
    This is a soft counterpart to ``_JOB_CONCURRENCY_LIMITS``. A reservation exists to leave room for other jobs
    rather than because the type stops gaining from concurrency, so it binds only while other jobs can take that
    room. Admission offers the reserved capacity to every other runnable job first, then releases the reservation
    over whatever capacity remains.

    That release is what keeps a reservation from idling the host. A wide compute stage held to a reservation while
    cores sit unused and its own queue is deep would waste the very capacity the reservation was meant to protect,
    which is the failure this two-pass admission avoids.

    A job type may appear in both tables, where the ceiling stands in every pass and the reservation applies only to
    the first. A job type absent from this map competes for capacity at its full core-derived width.
"""


@dataclass(frozen=True, slots=True)
class PipelineDispatch[UnitT]:
    """Binds a processing pipeline to the assets the generic batch tools drive it with.

    Notes:
        ``discover`` is the pipeline's job resolver, ``worker`` is the picklable callable the process pool
        dispatches per job, ``prerequisites`` is the pipeline's own intra-pipeline job ordering, and ``tracker_path``
        resolves the pipeline's processing tracker from a loaded unit. Cores belong to the job type, since one
        pipeline mixes job types that parallelize very differently.

        ``UnitT`` is whatever the pipeline's resolver loads, which is the processing unit its jobs operate on. A
        session pipeline resolves a session and a dataset pipeline resolves a dataset.
    """

    pipeline: ProcessingPipelines
    """The pipeline this entry dispatches."""
    discover: Callable[[Path], tuple[UnitT, list[tuple[str, str]], list[tuple[str, str]]]]
    """The job resolver returning the loaded unit, the job universe, and the runnable subset."""
    worker: Callable[..., None]
    """The picklable module-level worker the process pool invokes with a single planned job."""
    prerequisites: Callable[[UnitT, list[tuple[str, str]]], dict[tuple[str, str], tuple[tuple[str, str], ...]]]
    """Resolves each job's upstream jobs, producing the ordering the batch engine dispatches jobs in. Takes the loaded
    unit, since a pipeline whose jobs carry specifiers at differing scopes recovers their relation from it."""
    tracker_path: Callable[[UnitT], Path]
    """Resolves the pipeline's processing tracker path from a loaded unit."""
    output_path: Callable[[UnitT], Path | None]
    """Resolves the directory this pipeline owns outright, which is what a cleanup may remove to return the unit
    to its unprocessed state. Resolves to None for a pipeline that writes into a directory it shares with the
    acquired data, since removing that directory would take the inputs with it."""
    unit_name: Callable[[UnitT], str]
    """Resolves the unit's name, which every tool response reports the unit by."""
    estimate_memory: Callable[[UnitT, list[tuple[str, str, int]]], dict[tuple[str, str], tuple[int, bool]]]
    """Estimates the memory each runnable job occupies at its allocated core count, from the data it will process."""
    command: Callable[[GenericPendingJob], tuple[str, ...]]
    """Renders the command line that runs one job on a host holding the data, as an argument vector. The remote
    backend submits this, so one table states both how a job runs in-process and how it runs as a scheduled
    allocation."""


def run_batch_job(job: GenericPendingJob) -> None:
    """Runs one prepared job of any pipeline, routing on the pipeline the job carries.

    Notes:
        This is the single picklable entry point the shared pool dispatches, which is what lets one pool hold jobs
        from every pipeline at once. Each pipeline's own worker is looked up rather than bound into the job, so the
        descriptor stays a plain data record that pickles cheaply.

        The decode pool is scoped to the job's own cores before its worker runs, since one pool worker serves job
        types whose decode widths differ.

    Args:
        job: The pending job carrying its pipeline, its target job identifier, and its planned cores.

    Raises:
        ValueError: If the job names a pipeline the dispatch table does not support.
    """
    apply_decode_thread_ceiling(cores=job.core_weight)
    dispatch = resolve_dispatch(pipeline=job.pipeline)
    if dispatch is None:
        message = (
            f"Unable to run batch job '{job.job_id}'. The job names pipeline '{job.pipeline}', which is not a "
            f"supported batch pipeline."
        )
        console.error(message=message, error=ValueError)
    dispatch.worker(job)


def resolve_job_command(job: GenericPendingJob) -> tuple[str, ...]:
    """Renders the command line that runs one prepared job on a host holding the data it processes.

    Notes:
        Rendered from the same dispatch table the in-process worker routes on, so a job runs the same stage at the same
        width whichever way it is executed. Progress reporting is suppressed, since a scheduled allocation writes its
        output to a log file rather than to a terminal.

    Args:
        job: The pending job carrying its pipeline, its target job identifier, and its planned cores.

    Returns:
        The command as an argument vector, which a caller quotes for the shell it submits to.

    Raises:
        ValueError: If the job names a pipeline the dispatch table does not support.
    """
    dispatch = resolve_dispatch(pipeline=job.pipeline)
    if dispatch is None:
        message = (
            f"Unable to render the command for job '{job.job_id}'. The job names pipeline '{job.pipeline}', which is "
            f"not a supported batch pipeline."
        )
        console.error(message=message, error=ValueError)
    return dispatch.command(job)


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


def resolve_job_cores(job_name: str) -> int:
    """Resolves the cores one job of the named type occupies, narrowed to what this host can supply.

    Notes:
        A stage that reads its own thread count from a configuration file needs this before its jobs dispatch, so
        the file names the width the batch budgeted rather than the host's whole core count.

    Args:
        job_name: The tracker job name whose allocation to resolve.

    Returns:
        The cores one job of that type occupies, or the host's available cores when the type declares no allocation.

    Raises:
        ValueError: If the job type declares no core allocation.
    """
    if job_name not in _JOB_CORE_ALLOCATIONS:
        message = (
            f"Unable to resolve the cores for job type '{job_name}'. Every job type a pipeline resolves must "
            f"declare the cores one of its jobs occupies in _JOB_CORE_ALLOCATIONS."
        )
        console.error(message=message, error=ValueError)
    return min(
        _JOB_CORE_ALLOCATIONS[job_name], resolve_worker_count(requested_workers=-1, reserved_cores=RESERVED_CORES)
    )


def resolve_concurrency_limits(job_names: set[str]) -> dict[str, int]:
    """Resolves the concurrent-job ceiling each queued job type carries beyond the batch's two budgets.

    Notes:
        Only the job types that declare a limit appear in the result, so a caller reads an absent name as bounded by
        the core and memory budgets alone. Declared limits are raised to one, since a ceiling of zero would leave a
        queued job with no admission path at all.

    Args:
        job_names: The job type names present in the batch.

    Returns:
        A dictionary mapping each limited job name to the jobs of that type that may run at once.
    """
    return {
        job_name: max(1, _JOB_CONCURRENCY_LIMITS[job_name])
        for job_name in job_names
        if job_name in _JOB_CONCURRENCY_LIMITS
    }


def resolve_concurrency_reservations(job_names: set[str]) -> dict[str, int]:
    """Resolves the concurrency each queued job type is held to while other work can use the capacity it gives up.

    Notes:
        Only the job types that declare a reservation appear in the result, so a caller reads an absent name as
        competing at its full core-derived width. Declared reservations are raised to one, since a reservation of
        zero would keep a type out of the first admission pass entirely.

    Args:
        job_names: The job type names present in the batch.

    Returns:
        A dictionary mapping each reserved job name to the jobs of that type admitted before the reservation lifts.
    """
    return {
        job_name: max(1, _JOB_CONCURRENCY_RESERVATIONS[job_name])
        for job_name in job_names
        if job_name in _JOB_CONCURRENCY_RESERVATIONS
    }


def prepare_pipeline_jobs[UnitT](
    dispatch: PipelineDispatch[UnitT], unit_path: Path, options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Discovers a unit's runnable jobs, aligns the pipeline tracker, and returns the job descriptors.

    Notes:
        Discovery loads the unit and resolves the job universe and its runnable subset through the pipeline's
        job resolver. The tracker is aligned against the runnable subset within the universe, so a partial run
        neither wipes sibling jobs nor discards the recorded state of any job the pipeline can still produce. The
        returned descriptors carry everything the execute tool needs to dispatch each job.

        Any options the caller supplies are stamped onto every descriptor unchanged and reach the pipeline's worker
        at dispatch. They do not reach discovery, so the jobs a unit resolves stay a property of the data on disk
        rather than of the parameters a run was launched with. That keeps one tracker slot per job however the job
        is parameterized, which is what lets a multimode pipeline record one integrity state per unit.

    Args:
        dispatch: The pipeline's dispatch entry.
        unit_path: The path to the processing unit to discover jobs for, which is a session root for a session
            pipeline and a dataset root for a dataset pipeline.
        options: The pipeline-specific parameters to run these jobs with, such as the mode a multi-mode pipeline
            runs in. Pipelines that take no parameters ignore this mapping.

    Returns:
        A dictionary with the unit name, the tracker path, and a list of job descriptors, each carrying
        ``job_id``, ``job_name``, ``specifier``, ``unit_path``, ``tracker_path``, ``pipeline``, its allocated
        ``cores``, its estimated ``memory_mb``, a ``memory_modeled`` flag, ``prerequisite_ids``, and ``options``.
    """
    unit, universe, runnable = dispatch.discover(unit_path)
    unit_name = dispatch.unit_name(unit)
    tracker_path = dispatch.tracker_path(unit)
    tracker_path.parent.mkdir(parents=True, exist_ok=True)
    tracker = ProcessingTracker(file_path=tracker_path)
    tracker.align_jobs(jobs=runnable, universe=universe)

    # Ordering resolves over the full universe, so a job's upstream stages are named even when this batch does not
    # queue them. The engine then treats an unqueued prerequisite as satisfied only if the tracker already records
    # it as succeeded.
    ordering = dispatch.prerequisites(unit, universe)

    # One set of figures drives both local admission and any remote submission that reads the descriptor.
    unregistered = sorted({job_name for job_name, _ in runnable if job_name not in _JOB_CORE_ALLOCATIONS})
    if unregistered:
        message = (
            f"Unable to prepare {dispatch.pipeline.value} jobs for unit '{unit_name}'. No core allocation is "
            f"registered for job type(s) {unregistered}. Every job type a pipeline resolves must declare the cores "
            f"one of its jobs occupies in _JOB_CORE_ALLOCATIONS."
        )
        console.error(message=message, error=ValueError)
    cores = {job_name: _JOB_CORE_ALLOCATIONS[job_name] for job_name, _ in runnable}
    memory = dispatch.estimate_memory(
        unit, [(job_name, specifier, cores[job_name]) for job_name, specifier in runnable]
    )

    jobs = [
        {
            "job_id": ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier),
            "job_name": job_name,
            "specifier": specifier,
            "unit_path": str(unit_path),
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
    return {"unit_name": unit_name, "tracker_path": str(tracker_path), "jobs": jobs}


def build_pending_job(job: dict[str, Any]) -> GenericPendingJob:
    """Builds a GenericPendingJob from a job descriptor emitted by ``prepare_pipeline_jobs``.

    Args:
        job: A job descriptor carrying ``tracker_path``, ``job_id``, ``unit_path``, ``cores``, and
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
        unit_path=Path(job["unit_path"]),
        job_name=job.get("job_name", ""),
        name=job.get("unit_name", ""),
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
        The runtime pipeline is single-job, so it takes no job identifier.

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
    """Runs a single two-photon binarization, per-plane registration, per-plane processing, or combination job.

    Notes:
        The job's core weight reaches cindra as a call argument, so the stage runs at the width the batch admitted it
        at rather than at a default cindra would resolve on its own.

    Args:
        job: The pending job carrying the session root in ``unit_path`` and the target job in ``job_id``.
    """
    run_two_photon_processing_pipeline(session_path=job.unit_path, job_id=job.job_id, workers=job.core_weight)


def _checksum_command(job: GenericPendingJob) -> tuple[str, ...]:
    """Renders the command that runs the raw-data integrity pipeline for one session.

    Args:
        job: The pending job carrying the session root, its planned cores, and its mode.

    Returns:
        The command as an argument vector.
    """
    command = ["slf", "checksum", "-sp", str(job.unit_path), "-w", str(job.core_weight), "-np"]
    if bool(job.options.get("regenerate_checksum", False)):
        command.append("-rc")
    return tuple(command)


def _runtime_command(job: GenericPendingJob) -> tuple[str, ...]:
    """Renders the command that runs the runtime pipeline for one session.

    Args:
        job: The pending job carrying the session root and its planned cores.

    Returns:
        The command as an argument vector.
    """
    return *_session_command_preamble(job=job), "runtime"


def _microcontroller_command(job: GenericPendingJob) -> tuple[str, ...]:
    """Renders the command that runs one microcontroller extraction or parse job for one session.

    Args:
        job: The pending job carrying the session root, the target job, and its planned cores.

    Returns:
        The command as an argument vector.
    """
    return *_session_command_preamble(job=job), "-id", job.job_id, "microcontroller"


def _video_command(job: GenericPendingJob) -> tuple[str, ...]:
    """Renders the command that runs one camera timestamp, rename, tracking, or motion-energy job for one session.

    Args:
        job: The pending job carrying the session root, the target job, and its planned cores.

    Returns:
        The command as an argument vector.
    """
    return *_session_command_preamble(job=job), "-id", job.job_id, "video"


def _two_photon_command(job: GenericPendingJob) -> tuple[str, ...]:
    """Renders the command that runs one two-photon binarization, per-plane, or combination job for one session.

    Args:
        job: The pending job carrying the session root, the target job, and its planned cores.

    Returns:
        The command as an argument vector.
    """
    return *_session_command_preamble(job=job), "-id", job.job_id, "two-photon"


def _forging_command(job: GenericPendingJob) -> tuple[str, ...]:
    """Renders the command that runs one forging multi-day or assembly job for one dataset.

    Notes:
        Names no session and requests no rebuild, so the command runs the tracked job alone against the hierarchy the
        dataset definition step already built.

    Args:
        job: The pending job carrying the dataset root, the target job, and its planned cores.

    Returns:
        The command as an argument vector.
    """
    return (
        "slf",
        "forge",
        "-dn",
        job.unit_path.name,
        "-pp",
        str(job.unit_path.parent),
        "-id",
        job.job_id,
        "-w",
        str(job.core_weight),
        "-np",
    )


def _session_command_preamble(job: GenericPendingJob) -> tuple[str, ...]:
    """Renders the options every ``slf process`` subcommand shares, which the group parses ahead of the subcommand.

    Args:
        job: The pending job carrying the session root and its planned cores.

    Returns:
        The shared leading arguments of the command.
    """
    return "slf", "process", "-sp", str(job.unit_path), "-w", str(job.core_weight), "-np"


def _session_memory(pipeline: ProcessingPipelines) -> Callable[[SessionData, list[tuple[str, str, int]]], Any]:
    """Binds the session memory estimator to one pipeline.

    Args:
        pipeline: The pipeline whose jobs the bound estimator sizes.

    Returns:
        The estimator that pipeline's dispatch entry declares.
    """
    return lambda session, jobs: estimate_session_job_memory(pipeline=pipeline, session=session, jobs=jobs)


def _session_tracker(pipeline: ProcessingPipelines) -> Callable[[SessionData], Path]:
    """Binds the shared session tracker-path resolver to one pipeline.

    Args:
        pipeline: The pipeline whose tracker the bound resolver locates.

    Returns:
        The tracker resolver that pipeline's dispatch entry declares.
    """
    return lambda session: resolve_session_tracker_path(session=session, pipeline=pipeline)


def _run_forging_job(job: GenericPendingJob) -> None:
    """Runs a single forging multi-day or assembly job for one dataset.

    Notes:
        The dataset is named by the unit directory the job carries, which sits under the project root the pipeline
        resolves its sessions from. The hierarchy the job runs against is built beforehand, so the job takes no
        parameters of its own.

    Args:
        job: The pending job carrying the dataset root in ``unit_path``, the target job in ``job_id``, and its
            planned cores in ``core_weight``.
    """
    run_forging_pipeline(
        name=job.unit_path.name,
        project_root=job.unit_path.parent,
        job_id=job.job_id,
        workers=job.core_weight,
    )


@cache
def _pipeline_dispatch() -> dict[ProcessingPipelines, PipelineDispatch[Any]]:
    """Builds the dispatch entry for every processing pipeline the generic batch tools support.

    Notes:
        Built on first use and cached, so every caller shares one table.

    Returns:
        The dispatch entry for each supported pipeline, keyed by pipeline.
    """
    return {
        ProcessingPipelines.CHECKSUM: PipelineDispatch[SessionData](
            pipeline=ProcessingPipelines.CHECKSUM,
            discover=discover_checksum_jobs,
            worker=_run_checksum_job,
            prerequisites=checksum_job_prerequisites,
            tracker_path=_session_tracker(ProcessingPipelines.CHECKSUM),
            # Writes its stored checksum into raw_data, which holds the acquired data itself, so it owns no
            # directory a cleanup may remove.
            output_path=lambda _session: None,
            unit_name=lambda session: session.session_name,
            estimate_memory=_session_memory(ProcessingPipelines.CHECKSUM),
            command=_checksum_command,
        ),
        ProcessingPipelines.RUNTIME: PipelineDispatch[SessionData](
            pipeline=ProcessingPipelines.RUNTIME,
            discover=discover_runtime_jobs,
            worker=_run_runtime_job,
            prerequisites=runtime_job_prerequisites,
            tracker_path=_session_tracker(ProcessingPipelines.RUNTIME),
            output_path=lambda session: session.processed_data.runtime_data_path,
            unit_name=lambda session: session.session_name,
            estimate_memory=_session_memory(ProcessingPipelines.RUNTIME),
            command=_runtime_command,
        ),
        ProcessingPipelines.MICROCONTROLLER: PipelineDispatch[SessionData](
            pipeline=ProcessingPipelines.MICROCONTROLLER,
            discover=discover_microcontroller_jobs,
            worker=_run_microcontroller_job,
            prerequisites=microcontroller_job_prerequisites,
            tracker_path=_session_tracker(ProcessingPipelines.MICROCONTROLLER),
            output_path=lambda session: session.processed_data.microcontroller_data_path,
            unit_name=lambda session: session.session_name,
            estimate_memory=_session_memory(ProcessingPipelines.MICROCONTROLLER),
            command=_microcontroller_command,
        ),
        ProcessingPipelines.VIDEO: PipelineDispatch[SessionData](
            pipeline=ProcessingPipelines.VIDEO,
            discover=discover_video_jobs,
            worker=_run_video_job,
            prerequisites=video_job_prerequisites,
            tracker_path=_session_tracker(ProcessingPipelines.VIDEO),
            output_path=lambda session: session.processed_data.video_data_path,
            unit_name=lambda session: session.session_name,
            estimate_memory=_session_memory(ProcessingPipelines.VIDEO),
            command=_video_command,
        ),
        ProcessingPipelines.TWO_PHOTON: PipelineDispatch[SessionData](
            pipeline=ProcessingPipelines.TWO_PHOTON,
            discover=discover_two_photon_jobs,
            worker=_run_two_photon_job,
            prerequisites=two_photon_job_prerequisites,
            tracker_path=_session_tracker(ProcessingPipelines.TWO_PHOTON),
            output_path=lambda session: session.processed_data.cindra_data_path,
            unit_name=lambda session: session.session_name,
            estimate_memory=_session_memory(ProcessingPipelines.TWO_PHOTON),
            command=_two_photon_command,
        ),
        ProcessingPipelines.FORGING: PipelineDispatch[DatasetData](
            pipeline=ProcessingPipelines.FORGING,
            discover=discover_forging_jobs,
            worker=_run_forging_job,
            prerequisites=forging_job_prerequisites,
            tracker_path=forging_tracker_path,
            # Owns the dataset hierarchy outright, which holds the assembled feathers and the tracker beside them.
            output_path=lambda dataset: dataset.dataset_data_path.parent,
            unit_name=lambda dataset: dataset.name,
            estimate_memory=estimate_dataset_job_memory,
            command=_forging_command,
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
