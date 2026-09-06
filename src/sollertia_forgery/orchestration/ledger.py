"""Provides the durable record of the allocations this host has outstanding on the remote compute server's scheduler."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import field, replace, dataclass

from filelock import FileLock
from ataraxis_time import TimestampFormats, TimestampPrecisions, get_timestamp
from ataraxis_data_structures import YamlConfig

from ..server import remote_state_path

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Sequence

_LEDGER_FILENAME: str = "submission_ledger.yaml"
"""The filename of the submission ledger, written beside the state artifacts pulled from the server."""

_LOCK_TIMEOUT_SECONDS: float = 20.0
"""The period a writer waits for the ledger's lock before giving up, matching the project manifest's writer."""


@dataclass(frozen=True, slots=True)
class RemoteSubmission:
    """Records one prepared job and the scheduler allocation carrying it.

    Notes:
        The job identifier says which stage of which unit this is, and the scheduler is queried by the allocation
        identifier, so a status read needs both.
    """

    job_id: str = ""
    """The job identifier, which is both the key under which the project job artifact records this job and the key
    that joins a pulled artifact to this record."""
    slurm_job_id: str = ""
    """The identifier the scheduler assigned to this allocation."""
    slurm_job_name: str = ""
    """The name the allocation carries in the scheduler's queue."""
    pipeline: str = ""
    """The pipeline that owns this job."""
    job_name: str = ""
    """The tracker job name identifying this job's stage."""
    specifier: str = ""
    """The specifier differentiating this job from others of its stage within the same unit."""
    unit_path: str = ""
    """The path, on the server, to the processing unit on which this job operates."""
    unit_name: str = ""
    """The name of the processing unit on which this job operates."""
    cores: int = 1
    """The cores the allocation requested."""
    resident_mb: int = 0
    """The resident memory estimate that sized this allocation, in megabytes. The scheduler packs a node by what each
    allocation declares and reclaims the shortfall from a job that declares less than it holds, so it is given the
    resident figure. The request itself is this figure rounded up to whole gigabytes and floored at one."""
    output_log: str = ""
    """The path, on the server, to the file collecting this allocation's standard output."""
    error_log: str = ""
    """The path, on the server, to the file collecting this allocation's standard error."""


@dataclass(frozen=True, slots=True)
class SubmissionBatch:
    """Records one submitted batch and every allocation it holds."""

    batch_id: str = ""
    """The identifier the preparation issued, which also names the batch's directory on the server."""
    batch_ids: list[str] = field(default_factory=list)
    """Every prepared batch this submission dispatched, since one submission may span several. An empty list covers
    the single batch that ``batch_id`` names."""
    batch_directory: str = ""
    """The path, on the server, to the directory holding this batch's scripts and logs."""
    submitted_at: int = 0
    """The UTC timestamp (microsecond-precision epoch) at which the batch was submitted."""
    walltime_minutes: int = 0
    """The wall-time every allocation of this batch requested."""
    submissions: list[RemoteSubmission] = field(default_factory=list)
    """The allocations this batch holds, in the order they were accepted."""

    @property
    def pipelines(self) -> list[str]:
        """Returns the pipelines whose jobs this batch holds."""
        return sorted({submission.pipeline for submission in self.submissions})

    @property
    def covered_batch_ids(self) -> list[str]:
        """Returns every prepared batch this submission dispatched."""
        return list(self.batch_ids) if self.batch_ids else [self.batch_id]


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


def read_ledger() -> SubmissionLedger:
    """Reads the submission ledger, treating an absent ledger as holding no batches.

    Returns:
        The recorded ledger.
    """
    path = _ledger_path()
    if not path.is_file():
        return SubmissionLedger()
    return SubmissionLedger.from_yaml(file_path=path)


def record_batch(batch: SubmissionBatch, resubmitted: Sequence[tuple[str, str]] | None = None) -> SubmissionLedger:
    """Records one submitted batch, replacing any earlier record of the same batch.

    Notes:
        Naming the jobs a submission re-submitted merges the record instead of replacing it outright. Every allocation
        the earlier record held for a job this submission did not cover is carried ahead of the new ones. The merge
        lives here rather than in the caller because the entries it carries forward are then read under the same lock
        that writes them. An allocation a concurrent writer added to the same batch is therefore carried rather than
        dropped by a list read before the lock was taken.

    Args:
        batch: The batch to record.
        resubmitted: The unit path and job identifier of each job this submission re-submitted, which are the entries
            the earlier record must not carry forward. Leave as None to replace the earlier record outright.

    Returns:
        The ledger as it now stands on disk.

    Raises:
        Timeout: If the ledger's lock cannot be acquired within the timeout period.
    """
    with _ledger_lock():
        ledger = read_ledger()
        merged = batch
        already_recorded = ledger.resolve_batch(batch_id=batch.batch_id)
        if resubmitted is not None and already_recorded is not None:
            covered = set(resubmitted)
            carried = [
                entry for entry in already_recorded.submissions if (entry.unit_path, entry.job_id) not in covered
            ]
            merged = replace(batch, submissions=[*carried, *batch.submissions])
        ledger.batches = [recorded for recorded in ledger.batches if recorded.batch_id != batch.batch_id]
        ledger.batches.append(merged)
        _save_ledger(ledger=ledger)
        return ledger


def forget_batches(batch_ids: Sequence[str]) -> list[str]:
    """Drops the named batches from the submission ledger, whatever state their allocations hold.

    Notes:
        This clears the ledger alone, which is the record of the allocations this host has outstanding on the
        scheduler. The prepared documents and the recorded outcomes live in the separate batch registry that
        ``forget_batch_records`` clears.

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
    """Resolves the batches to which a status or cancellation applies.

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


def _ledger_path() -> Path:
    """Returns the path to the submission ledger.

    Returns:
        The path to the ledger file under the Sollertia platform working directory.
    """
    return remote_state_path().joinpath(_LEDGER_FILENAME)


def _save_ledger(ledger: SubmissionLedger) -> None:
    """Writes the ledger to disk, taking no lock of its own since every caller already holds one.

    Args:
        ledger: The ledger to write.
    """
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_yaml(file_path=path)


def _ledger_lock() -> FileLock:
    """Returns the acquired lock guarding the ledger file.

    Returns:
        The lock context manager, acquired for the shared timeout period.

    Raises:
        Timeout: If the lock cannot be acquired within the timeout period.
    """
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(path.with_suffix(path.suffix + ".lock"))).acquire(timeout=_LOCK_TIMEOUT_SECONDS)  # type: ignore[return-value]
