"""Provides the single-stage runtime log processing pipeline that decodes the acquisition runtime's DataLogger archive
into a raw message table and parses it into domain-specific behavior feathers using the parser registered for the
session's acquisition system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from contextlib import nullcontext
from concurrent.futures import Future, ProcessPoolExecutor, as_completed

import numpy as np
import polars as pl
from ataraxis_base_utilities import LogLevel, console, resolve_worker_count
from sollertia_shared_assets import SessionData, ProcessingTrackers
from ataraxis_data_structures import LogArchiveReader, ProcessingTracker

from ..registries import resolve_runtime_binding
from ..shared_assets import LOG_ARCHIVE_SUFFIX, tracked_job, prepare_tracker

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

RUNTIME_JOB_NAME: str = "runtime_processing"
"""The job name identifying the runtime processing job in the runtime processing tracker
(``ProcessingTrackers.RUNTIME``), where this pipeline records the runtime job's state."""


def run_runtime_processing_pipeline(
    session_path: Path,
    job_id: str | None = None,
    *,
    workers: int = -1,
    display_progress: bool = False,
) -> None:
    """Decodes and parses the acquisition runtime's log archive for the target session.

    Notes:
        This is a single-stage pipeline. It locates the runtime DataLogger archive (``{source_id}_log.npz``) in the
        session's raw behavior-data directory, decodes it into a raw ``(time_us, payload)`` message table, and hands
        that table to the registered runtime parser, which writes the system's behavior feathers into the session's
        processed runtime-data directory (``processed_data.runtime_data_path``). The runtime source id and parser
        are resolved from ``RUNTIME_PARSER_REGISTRY`` by the session's acquisition system, keeping the pipeline
        system-agnostic.

        The decode is the only stage that benefits from parallelism: when the reader splits the archive into more
        than one batch and more than one worker is available, the batches are decoded across a worker pool; otherwise
        the archive is read in a single in-process pass. The parse always runs in-process, so the whole pipeline is
        one tracked job recorded against the ``runtime`` tracker.

        In local mode (job_id is None) the runtime job runs unconditionally; in remote mode the provided job_id must
        match this session's single runtime job. The registered parser may additionally raise system-specific errors
        (for example ``ValueError`` or ``RuntimeError``) that propagate unchanged.

    Args:
        session_path: The path to the root session directory containing the session data hierarchy.
        job_id: The hexadecimal identifier of the runtime job to execute (remote mode). If not provided, the pipeline
            runs the runtime job directly (local mode).
        workers: The number of worker processes the decode stage may use. A value less than 1 uses all available CPU
            cores (minus reserved cores); 1 forces a single in-process decode.
        display_progress: Determines whether to display a progress bar while decoding a multi-batch archive.

    Raises:
        FileNotFoundError: If the session's runtime log archive is not present at its canonical raw behavior data
            location.
        ValueError: If the session's acquisition system has no runtime binding registered, or if the provided job_id
            does not match the runtime job available for this session.
    """
    session = SessionData.load(session_path=session_path)
    console.echo(
        message=f"Initializing runtime processing pipeline for session '{session.session_name}'...",
        level=LogLevel.INFO,
    )

    # Resolves the system's runtime binding: the source id locating its runtime archive and the parser interpreting
    # the decoded messages. The system is inferred from the session, so the pipeline never names a system-specific type.
    source_id, parser = resolve_runtime_binding(session.acquisition_system)

    log_directory = session.raw_data.behavior_data_path
    output_directory = session.processed_data.runtime_data_path

    archive_path = _find_runtime_archive(log_directory=log_directory, source_id=source_id)
    if archive_path is None:
        message = (
            f"Unable to process runtime data for session '{session.session_name}'. No runtime log archive "
            f"'{source_id}{LOG_ARCHIVE_SUFFIX}' was found in '{log_directory}'. The runtime DataLogger writes exactly "
            f"one archive per session under its fixed source id."
        )
        console.error(message=message, error=FileNotFoundError)

    jobs = [(RUNTIME_JOB_NAME, source_id)]
    job_identifier = ProcessingTracker.generate_job_id(job_name=RUNTIME_JOB_NAME, specifier=source_id)

    if job_id is not None and job_id != job_identifier:
        message = (
            f"Unable to execute the requested job with ID '{job_id}'. The identifier does not match the runtime "
            f"processing job available for this session. Valid job ID: '{job_identifier}'."
        )
        console.error(message=message, error=ValueError)

    # The tracker co-locates with the parsed output in ``runtime_data``. The runtime job is the sole entry in its
    # universe, so a re-run aligns the tracker against exactly that job.
    output_directory.mkdir(parents=True, exist_ok=True)
    tracker = ProcessingTracker(file_path=output_directory.joinpath(ProcessingTrackers.RUNTIME))
    prepare_tracker(tracker=tracker, jobs=jobs, universe=jobs)

    console.echo(message=f"Running '{RUNTIME_JOB_NAME}' job with specifier '{source_id}' (ID: {job_identifier})...")
    with tracked_job(tracker=tracker, job_id=job_identifier):
        decoded_messages = _decode_archive(
            archive_path=archive_path, workers=workers, display_progress=display_progress
        )
        parser(decoded_messages=decoded_messages, output_directory=output_directory, session=session)

    console.echo(
        message=f"Runtime processing for session '{session.session_name}' completed successfully.",
        level=LogLevel.SUCCESS,
    )


def _find_runtime_archive(log_directory: Path, source_id: str) -> Path | None:
    """Locates the runtime DataLogger archive for the source id, if it is present under the log directory.

    Notes:
        The runtime DataLogger always writes to a fixed per-system source id, so at most one archive named
        ``{source_id}_log.npz`` is expected directly inside the session's raw behavior data directory. Returns None
        when no archive is present so the caller can raise an actionable error rather than failing deep in the reader.

    Args:
        log_directory: The session's raw behavior data directory holding the runtime log archive.
        source_id: The runtime DataLogger source id donated by the session's acquisition system.

    Returns:
        The path to the runtime log archive, or None if the directory or the archive does not exist.
    """
    if not log_directory.is_dir():
        return None
    archive_path = log_directory.joinpath(f"{source_id}{LOG_ARCHIVE_SUFFIX}")
    return archive_path if archive_path.is_file() else None


def _decode_archive(archive_path: Path, *, workers: int, display_progress: bool) -> pl.DataFrame:
    """Decodes the runtime log archive into a raw ``(time_us, payload)`` message table.

    Notes:
        This is the system-agnostic decode stage. It reads the DataLogger archive via ``LogArchiveReader``, which
        resolves the onset timestamp and yields each message's absolute timestamp and raw payload bytes. When the
        reader splits the archive into more than one batch and more than one worker is available, the batches are
        decoded across a worker pool (each worker reuses the pre-discovered onset timestamp); otherwise the archive
        is read in a single in-process bulk pass. The returned table carries the timestamps unchanged and the
        payloads as opaque bytes, leaving every system-specific interpretation to the registered parser.

    Args:
        archive_path: The path to the runtime ``{source_id}_log.npz`` archive.
        workers: The number of worker processes the decode may use. A value less than 1 uses all available CPU cores
            (minus reserved cores); 1 forces a single in-process decode.
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
        {
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

    with ProcessPoolExecutor(max_workers=workers) as executor:
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
