"""Provides the durable record of the allocations this host has outstanding on the remote compute server's scheduler."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import field, dataclass

from filelock import FileLock
from ataraxis_time import TimestampFormats, TimestampPrecisions, get_timestamp
from ataraxis_data_structures import YamlConfig

from ..server import TERMINAL_JOB_STATUSES, JobStatus, remote_state_path

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Sequence

_LEDGER_FILENAME: str = "submission_ledger.yaml"
"""The filename of the submission ledger, written beside the state artifacts pulled from the server."""

_LOCK_TIMEOUT_SECONDS: float = 20.0
"""The period a writer waits for the ledger's lock before giving up, matching the project manifest's writer."""


@dataclass(frozen=True, slots=True)
class RemoteSubmission:
    """Records one prepared job and the scheduler allocation it was submitted as.

    Notes:
        The job identifier says which stage of which unit this is and the allocation identifier is what the scheduler
        is queried with, so a status read needs both.
    """

    job_id: str = ""
    """The job identifier, which is both the key the project job artifact records this job under and the key that
    joins a pulled artifact to this record."""
    slurm_job_id: str = ""
    """The identifier the scheduler assigned to this allocation."""
    slurm_job_name: str = ""
    """The name the allocation carries in the scheduler's queue."""
    pipeline: str = ""
    """The pipeline this job belongs to."""
    job_name: str = ""
    """The tracker job name identifying this job's stage."""
    specifier: str = ""
    """The specifier differentiating this job from others of its stage within the same unit."""
    unit_path: str = ""
    """The path, on the server, to the processing unit this job operates on."""
    unit_name: str = ""
    """The name of the processing unit this job operates on."""
    cores: int = 1
    """The cores the allocation requested."""
    memory_mb: int = 0
    """The memory estimate this allocation was sized from, in megabytes. The scheduler request itself is this figure
    rounded up to whole gigabytes and floored at one."""
    output_log: str = ""
    """The path, on the server, to the file collecting this allocation's standard output."""
    error_log: str = ""
    """The path, on the server, to the file collecting this allocation's standard error."""


@dataclass(frozen=True, slots=True)
class SubmissionBatch:
    """Records one submitted batch and every allocation it holds."""

    batch_id: str = ""
    """The identifier the preparation issued, which also names the batch's directory on the server."""
    batch_directory: str = ""
    """The path, on the server, to the directory holding this batch's scripts and logs."""
    submitted_at: int = 0
    """The UTC timestamp (microsecond-precision epoch) the batch was submitted at."""
    walltime_minutes: int = 0
    """The wall-time every allocation of this batch requested."""
    submissions: list[RemoteSubmission] = field(default_factory=list)
    """The allocations this batch holds, in the order they were accepted."""

    @property
    def pipelines(self) -> list[str]:
        """Returns the pipelines this batch holds jobs for."""
        return sorted({submission.pipeline for submission in self.submissions})


@dataclass
class SubmissionLedger(YamlConfig):
    """Records every batch this host submitted to the remote compute server's scheduler."""

    batches: list[SubmissionBatch] = field(default_factory=list)
    """The recorded batches, in the order they were last recorded."""

    def resolve_batch(self, batch_id: str) -> SubmissionBatch | None:
        """Returns the recorded batch with the given identifier, or None when the ledger holds no such batch.

        Args:
            batch_id: The identifier of the batch to resolve.

        Returns:
            The recorded batch, or None.
        """
        return next((batch for batch in self.batches if batch.batch_id == batch_id), None)


def ledger_path() -> Path:
    """Returns the path to the submission ledger.

    This is the single source of the ledger's location, so every reader and writer derives the same path.

    Returns:
        The path to the ledger file under the Sollertia platform working directory.
    """
    return remote_state_path().joinpath(_LEDGER_FILENAME)


def read_ledger() -> SubmissionLedger:
    """Reads the submission ledger, treating an absent ledger as holding no batches.

    Returns:
        The recorded ledger.
    """
    path = ledger_path()
    if not path.is_file():
        return SubmissionLedger()
    return SubmissionLedger.from_yaml(file_path=path)


def record_batch(batch: SubmissionBatch) -> SubmissionLedger:
    """Records one submitted batch, replacing any earlier record of the same batch.

    Args:
        batch: The batch to record.

    Returns:
        The ledger as it now stands on disk.

    Raises:
        Timeout: If the ledger's lock cannot be acquired within the timeout period.
    """
    with _ledger_lock():
        ledger = read_ledger()
        ledger.batches = [recorded for recorded in ledger.batches if recorded.batch_id != batch.batch_id]
        ledger.batches.append(batch)
        _save_ledger(ledger=ledger)
        return ledger


def batch_is_settled(batch: SubmissionBatch, statuses: dict[str, JobStatus]) -> bool:
    """Returns True when every allocation the batch holds has reached a state it never leaves.

    Notes:
        An allocation the status map does not cover counts as unfinished, so a partial query never reports a batch it
        did not fully observe as settled.

    Args:
        batch: The batch to test.
        statuses: The observed state of each allocation, keyed by its scheduler identifier.
    """
    return bool(batch.submissions) and all(
        statuses.get(submission.slurm_job_id) in TERMINAL_JOB_STATUSES for submission in batch.submissions
    )


def retire_settled_batches(statuses: dict[str, JobStatus]) -> list[str]:
    """Drops every batch whose allocations have all reached a state they never leave.

    Notes:
        A finished batch is dropped because the ledger names outstanding allocations alone. What its jobs produced is
        read from the project job artifact.

        An allocation the query did not cover counts as unfinished, so a partial query never retires a batch it did
        not fully observe.

    Args:
        statuses: The observed state of each allocation, keyed by its scheduler identifier.

    Returns:
        The identifiers of the batches that were retired.

    Raises:
        Timeout: If the ledger's lock cannot be acquired within the timeout period.
    """
    if not statuses:
        return []

    with _ledger_lock():
        ledger = read_ledger()
        retired = [batch.batch_id for batch in ledger.batches if batch_is_settled(batch=batch, statuses=statuses)]
        if not retired:
            return []
        settled_ids = set(retired)
        ledger.batches = [batch for batch in ledger.batches if batch.batch_id not in settled_ids]
        _save_ledger(ledger=ledger)
        return retired


def forget_batches(batch_ids: Sequence[str]) -> list[str]:
    """Drops the named batches from the ledger, whatever state their allocations are in.

    Args:
        batch_ids: The identifiers of the batches to drop.

    Returns:
        The identifiers the ledger actually held and dropped.

    Raises:
        Timeout: If the ledger's lock cannot be acquired within the timeout period.
    """
    with _ledger_lock():
        ledger = read_ledger()
        named = set(batch_ids)
        dropped = [batch.batch_id for batch in ledger.batches if batch.batch_id in named]
        ledger.batches = [batch for batch in ledger.batches if batch.batch_id not in named]
        _save_ledger(ledger=ledger)
        return dropped


def resolve_batches(ledger: SubmissionLedger, batch_ids: Sequence[str] | None = None) -> list[SubmissionBatch]:
    """Resolves the batches a status or cancellation applies to.

    Args:
        ledger: The recorded ledger.
        batch_ids: The identifiers to resolve, or None to resolve every outstanding batch.

    Returns:
        The resolved batches.
    """
    if batch_ids is not None:
        named = set(batch_ids)
        return [batch for batch in ledger.batches if batch.batch_id in named]
    return list(ledger.batches)


def current_timestamp() -> int:
    """Returns the current UTC timestamp as a microsecond-precision epoch, matching every other timestamp in this
    stack.
    """
    return int(get_timestamp(output_format=TimestampFormats.INTEGER, precision=TimestampPrecisions.MICROSECOND))


def _save_ledger(ledger: SubmissionLedger) -> None:
    """Writes the ledger to disk, taking no lock of its own since every caller already holds one.

    Args:
        ledger: The ledger to write.
    """
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_yaml(file_path=path)


def _ledger_lock() -> FileLock:
    """Returns the acquired lock guarding the ledger file.

    Returns:
        The lock context manager, acquired for the shared timeout period.

    Raises:
        Timeout: If the lock cannot be acquired within the timeout period.
    """
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(path.with_suffix(path.suffix + ".lock"))).acquire(timeout=_LOCK_TIMEOUT_SECONDS)  # type: ignore[return-value]
