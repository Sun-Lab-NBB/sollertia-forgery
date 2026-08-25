"""Provides the single-stage runtime log processing pipeline that decodes the acquisition system's runtime
DataLogger archive into a raw message table and parses it into domain-specific behavior feathers using the parser
registered for the session's acquisition system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import polars as pl
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count
from sollertia_shared_assets import SessionData, ProcessingTrackers
from ataraxis_data_structures import (
    LogArchiveReader,
    ProcessingTracker,
    limit_worker_threads,
    discover_log_archives,
    initialize_worker_threads,
)

from ..registries import resolve_runtime_binding
from ..shared_assets import verify_openmp_runtime

if TYPE_CHECKING:
    from pathlib import Path
    from concurrent.futures import Future

    from numpy.typing import NDArray

RUNTIME_JOB_NAME: str = "runtime_processing"
"""The job name identifying the runtime processing job in the runtime processing tracker
(``ProcessingTrackers.RUNTIME``), where this pipeline records the runtime job's state."""


def run_runtime_processing_pipeline(
    session_path: Path,
    *,
    workers: int = -1,
    display_progress: bool = False,
) -> None:
    """Decodes and parses the acquisition runtime's log archive for the target session.

    Notes:
        This is a single-stage pipeline. It locates the acquisition system's runtime DataLogger archive under the
        session's raw behavior-data directory and decodes it into a raw ``(time_us, payload)`` message table. It then
        hands that table to the registered runtime parser, which writes the system's behavior feathers into the
        session's processed runtime-data directory (``processed_data.runtime_data_path``). The runtime source id and
        parser come from the session's acquisition system.

        The runtime job is the only job this pipeline produces, so it always runs. The registered parser may
        additionally raise system-specific errors (for example ``ValueError`` or ``RuntimeError``) that propagate
        unchanged.

    Args:
        session_path: The path to the root session directory containing the session data hierarchy.
        workers: The number of worker processes the decode stage may use. A value less than 1 uses all available CPU
            cores (minus reserved cores), and 1 forces a single in-process decode.
        display_progress: Determines whether to display a progress bar while decoding a multi-batch archive.

    Raises:
        FileNotFoundError: If the session's runtime log archive is not present in its raw behavior data directory.
        RuntimeError: If the host is macOS and carries no loadable OpenMP runtime for the Numba threading layer.
        ValueError: If the session's acquisition system is unknown (not a valid AcquisitionSystems member), or if the
            runtime log archive carries no valid onset timestamp message.
    """
    # A stage that this pipeline dispatches may reach a parallelized kernel, so a host whose threading layer has no
    # runtime to load fails here rather than partway through a session.
    verify_openmp_runtime()
    session, universe, possible = discover_runtime_jobs(session_path=session_path)
    console.echo(
        message=f"Initializing runtime processing pipeline for session '{session.session_name}'...",
        level=LogLevel.INFO,
    )

    # The single-job universe carries the source id as its specifier.
    source_id = universe[0][1]
    _, parser = resolve_runtime_binding(system=session.acquisition_system)

    log_directory = session.raw_data.behavior_data_path
    output_directory = session.processed_data.runtime_data_path

    # The resolver reports an absent archive as leaving no possible job, which this single-job pipeline escalates to a
    # failure because it leaves nothing to run.
    if not possible:
        message = (
            f"Unable to process runtime data for session '{session.session_name}'. No runtime log archive was found "
            f"for source '{source_id}' in '{log_directory}'. The runtime DataLogger writes exactly one archive per "
            f"session under its fixed source id."
        )
        console.error(message=message, error=FileNotFoundError)

    # The archive filename that a source writes is the data-structures library's own contract, so the path comes from
    # the same indexing helper that discovery used rather than from a name this pipeline rebuilds from the source id.
    archive_path = discover_log_archives(log_directory=log_directory)[source_id]
    job_identifier = ProcessingTracker.generate_job_id(job_name=RUNTIME_JOB_NAME, specifier=source_id)

    output_directory.mkdir(parents=True, exist_ok=True)
    tracker = ProcessingTracker(file_path=output_directory.joinpath(ProcessingTrackers.RUNTIME))
    tracker.align_jobs(jobs=possible, universe=universe)

    console.echo(message=f"Running '{RUNTIME_JOB_NAME}' job with specifier '{source_id}' (ID: {job_identifier})...")
    with tracker.run_job(job_id=job_identifier):
        decoded_messages = _decode_archive(
            archive_path=archive_path, workers=workers, display_progress=display_progress
        )
        parser(decoded_messages=decoded_messages, output_directory=output_directory, session=session)

    console.echo(
        message=f"Runtime processing for session '{session.session_name}' completed successfully.",
        level=LogLevel.SUCCESS,
    )


def discover_runtime_jobs(session_path: Path) -> tuple[SessionData, list[tuple[str, str]], list[tuple[str, str]]]:
    """Resolves the runtime pipeline's job universe and possible subset for the target session.

    Notes:
        The runtime pipeline produces exactly one job, so the universe is always the single
        ``(RUNTIME_JOB_NAME, source_id)`` pair, where the source id is resolved from the session's acquisition system.
        That job is possible only when its DataLogger archive is present on disk. Discovery indexes the archives the
        logger wrote and reads no message, leaving every output file untouched. An absent archive yields an empty
        possible subset, so a batch layer can align the tracker slot against the universe and skip the job.

        The archives are indexed through the data-structures library, which owns the name under which each source writes
        its archive, so a session that recorded no runtime archive is reported without this pipeline restating that
        naming rule. The index covers the logger's own output directory, which is where a session's archives are
        assembled side by side, and the pipeline resolves its own archive through that same helper.

    Args:
        session_path: The path to the root session directory containing the session data hierarchy.

    Returns:
        A tuple of the loaded session, the job universe as a list of ``(job_name, specifier)`` pairs, and the possible
        subset of that universe.

    Raises:
        ValueError: If the session's acquisition system is unknown (not a valid AcquisitionSystems member).
    """
    session = SessionData.load(session_path=session_path)
    source_id, _ = resolve_runtime_binding(system=session.acquisition_system)
    universe = [(RUNTIME_JOB_NAME, source_id)]
    log_directory = session.raw_data.behavior_data_path
    archives = discover_log_archives(log_directory=log_directory) if log_directory.is_dir() else {}
    possible = list(universe) if source_id in archives else []
    return session, universe, possible


def runtime_job_prerequisites(
    session: SessionData,  # noqa: ARG001
    universe: list[tuple[str, str]],
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]:
    """Returns the intra-pipeline job ordering for the runtime pipeline.

    Args:
        session: The loaded session, accepted for the shared dispatch contract and not read by this ordering.
        universe: The job universe as returned by ``discover_runtime_jobs``.

    Returns:
        A mapping of each job to its tuple of prerequisite jobs, which is always empty for the runtime pipeline.
    """
    return dict.fromkeys(universe, ())


def _decode_archive(archive_path: Path, *, workers: int, display_progress: bool) -> pl.DataFrame:
    """Decodes the runtime log archive into a raw ``(time_us, payload)`` message table.

    Notes:
        Reads the DataLogger archive via ``LogArchiveReader``, which resolves the onset timestamp and yields each
        message's absolute timestamp and raw payload bytes. When the reader splits the archive into more than one
        batch and more than one worker is available, the batches are decoded across a worker pool, each worker
        reusing the pre-discovered onset timestamp. A single batch or a single worker reads the archive in one
        in-process bulk pass. The returned table carries the timestamps unchanged and the payloads as opaque bytes.

    Args:
        archive_path: The path to the runtime log archive, as the data-structures locator resolved it.
        workers: The number of worker processes the decode may use. A value less than 1 uses all available CPU cores
            (minus reserved cores), and 1 forces a single in-process decode.
        display_progress: Determines whether to display a per-batch progress bar during a parallel decode.

    Returns:
        A Polars DataFrame with a ``time_us`` UInt64 column of absolute message timestamps and a ``payload`` Binary
        column of the corresponding raw message payloads, in archive order.
    """
    reader = LogArchiveReader(archive_path=archive_path)
    onset_us = reader.onset_timestamp_us
    batches = reader.get_batches(workers=workers)

    resolved_workers = resolve_worker_count(requested_workers=workers)
    if len(batches) <= 1 or resolved_workers <= 1:
        timestamps, payload_arrays = reader.read_all_messages()
        payloads = [payload_array.tobytes() for payload_array in payload_arrays]
    else:
        timestamps, payloads = _decode_batches(
            archive_path=archive_path,
            onset_us=onset_us,
            batches=batches,
            workers=resolved_workers,
            display_progress=display_progress,
        )

    return pl.DataFrame(
        data={
            "time_us": pl.Series(name="time_us", values=timestamps, dtype=pl.UInt64),
            "payload": pl.Series(name="payload", values=payloads, dtype=pl.Binary),
        }
    )


def _decode_batches(
    archive_path: Path,
    onset_us: np.uint64,
    batches: list[list[str]],
    *,
    workers: int,
    display_progress: bool,
) -> tuple[NDArray[np.uint64], list[bytes]]:
    """Decodes the archive's message batches concurrently and reassembles them in archive order.

    Notes:
        Each batch is decoded in its own worker process by ``_decode_batch``, which reuses the onset timestamp
        discovered once in the parent so no worker re-scans for it. Results are placed back at their batch index as
        their futures complete, so the concatenated output preserves the chronological order of the archive's batches
        regardless of completion order.

    Args:
        archive_path: The path to the runtime archive each worker re-opens to read its assigned batch.
        onset_us: The pre-discovered onset timestamp passed to every worker to skip redundant onset scanning.
        batches: The message-key batches produced by the reader, in archive order.
        workers: The resolved worker-process count for the decode pool.
        display_progress: Determines whether to display a per-batch progress bar.

    Returns:
        A tuple of the concatenated UInt64 timestamps and the ordered list of raw payload bytes across all batches.
    """
    timestamp_chunks: list[NDArray[np.uint64]] = [np.array([], dtype=np.uint64) for _ in batches]
    payload_chunks: list[list[bytes]] = [[] for _ in batches]

    # Each decode child re-imports and sizes its library thread pools before any of this code runs inside it, so the
    # caps are placed around the pool's construction rather than inside its workers. numba latches its own ceiling
    # while it is imported and rejects an environment variable that disagrees afterwards, so each child pins it
    # through its own runtime setter in the pool initializer instead.
    with (
        limit_worker_threads(),
        ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker_threads) as executor,
    ):
        future_to_index: dict[Future[tuple[NDArray[np.uint64], list[bytes]]], int] = {
            executor.submit(_decode_batch, archive_path=archive_path, onset_us=onset_us, keys=batch): index
            for index, batch in enumerate(batches)
        }

        progress_context = (
            console.progress(total=len(batches), description="Decoding runtime archive", unit="batch")
            if display_progress
            else nullcontext()
        )

        with progress_context as progress_bar:
            for completed_future in as_completed(future_to_index):
                index = future_to_index[completed_future]
                timestamp_chunks[index], payload_chunks[index] = completed_future.result()
                if progress_bar is not None:
                    progress_bar.update(1)

    timestamps: NDArray[np.uint64] = np.concatenate(timestamp_chunks)
    payloads = [payload for chunk in payload_chunks for payload in chunk]
    return timestamps, payloads


def _decode_batch(archive_path: Path, onset_us: np.uint64, keys: list[str]) -> tuple[NDArray[np.uint64], list[bytes]]:
    """Decodes a single batch of runtime messages into timestamps and raw payload bytes.

    Notes:
        This is the atomic unit of work dispatched to worker processes by the parallel decode path, so it must remain
        importable at module level and accept only picklable arguments. It re-opens the archive with the pre-discovered
        onset timestamp and iterates only the batch's message keys, copying each payload into immutable bytes so the
        result can cross the process boundary.

    Args:
        archive_path: The path to the runtime archive to read.
        onset_us: The pre-discovered onset timestamp, supplied so the worker skips onset discovery.
        keys: The message keys this batch is responsible for decoding.

    Returns:
        A tuple of the batch's UInt64 timestamps and the ordered list of its raw payload bytes.
    """
    reader = LogArchiveReader(archive_path=archive_path, onset_us=onset_us)
    timestamps: list[np.uint64] = []
    payloads: list[bytes] = []
    for message in reader.iter_messages(keys=keys):
        timestamps.append(message.timestamp_us)
        payloads.append(message.payload.tobytes())
    return np.array(timestamps, dtype=np.uint64), payloads
