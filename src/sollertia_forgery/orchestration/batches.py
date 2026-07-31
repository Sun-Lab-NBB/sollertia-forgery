"""Provides the on-disk registry of prepared batches, so a batch identifier outlives the process that issued it."""

from __future__ import annotations

from uuid import uuid4
from typing import TYPE_CHECKING, Any
from dataclasses import field, asdict, dataclass

from filelock import FileLock
from ataraxis_base_utilities import console
from ataraxis_data_structures import YamlConfig

from .graph import BatchDocument
from ..server import remote_state_path

if TYPE_CHECKING:
    from pathlib import Path

_BATCH_DIRECTORY_NAME: str = "prepared_batches"
"""The directory holding one file per prepared batch, beside this host's other records of a run."""

_LOCK_TIMEOUT_SECONDS: float = 20.0
"""The period a writer waits for a batch file's lock before giving up, matching the project manifest's writer."""


@dataclass
class PreparedBatch(YamlConfig):
    """Records one prepared batch under the identifier that preparation issued for it.

    Notes:
        Persisting the document is what lets a caller prepare a batch, restart the server that prepared it, and still
        execute it. The document is stored whole, so execution never re-resolves a batch it was handed.
    """

    batch_id: str = ""
    """The identifier that names and resolves this batch."""
    pipeline: str = ""
    """The pipeline the batch dispatches, recorded so a listing names it without parsing the document."""
    host: str = ""
    """The host the batch was prepared against, recorded for the same reason."""
    document: dict[str, Any] = field(default_factory=dict)
    """The batch document's fields, held as a plain mapping so it serializes without a nested dataclass schema."""
    outcome: dict[str, Any] = field(default_factory=dict)
    """What the batch's jobs finally recorded, written at closure and empty until then. This is the durable snapshot a
    caller reads after the run, so a finished batch stays answerable once nothing is running and nothing is queued."""

    def as_document(self) -> BatchDocument:
        """Returns the recorded batch as the document both execution backends dispatch."""
        return BatchDocument(**self.document)


def batch_directory() -> Path:
    """Returns the directory holding every prepared batch this host has recorded.

    This is the single source of the directory's location, so every reader and writer derives the same path.

    Returns:
        The path to the prepared-batch directory under the Sollertia platform working directory.
    """
    return remote_state_path().joinpath(_BATCH_DIRECTORY_NAME)


def batch_path(batch_id: str) -> Path:
    """Resolves where one prepared batch is recorded.

    Args:
        batch_id: The identifier of the batch.

    Returns:
        The path to the batch's file.
    """
    return batch_directory().joinpath(f"{batch_id}.yaml")


def record_prepared_batch(document: BatchDocument) -> str:
    """Records one prepared batch and returns the identifier that resolves it.

    Args:
        document: The prepared batch to record.

    Returns:
        The identifier the batch was recorded under.

    Raises:
        Timeout: If the batch file's lock cannot be acquired within the timeout period.
    """
    batch_id = uuid4().hex[:16]
    path = batch_path(batch_id=batch_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    recorded = PreparedBatch(
        batch_id=batch_id, pipeline=document.pipeline, host=document.host, document=asdict(document)
    )
    with FileLock(str(_lock_path(path=path))).acquire(timeout=_LOCK_TIMEOUT_SECONDS):
        recorded.to_yaml(file_path=path)
    return batch_id


def read_prepared_batch(batch_id: str) -> BatchDocument | None:
    """Reads one recorded batch.

    Args:
        batch_id: The identifier of the batch to read.

    Returns:
        The recorded batch document, or None when this host holds no batch under that identifier.
    """
    path = batch_path(batch_id=batch_id)
    if not path.is_file():
        return None
    return PreparedBatch.from_yaml(file_path=path).as_document()


def read_prepared_batches(batch_ids: list[str]) -> tuple[list[BatchDocument], list[str]]:
    """Reads several recorded batches, reporting the identifiers this host does not hold.

    Args:
        batch_ids: The identifiers to read.

    Returns:
        A tuple of the documents that were found and the identifiers that were not, the latter sorted.
    """
    found: list[BatchDocument] = []
    missing: list[str] = []
    for batch_id in batch_ids:
        document = read_prepared_batch(batch_id=batch_id)
        if document is None:
            missing.append(batch_id)
            continue
        found.append(document)
    return found, sorted(missing)


def resolve_batch_host(documents: list[BatchDocument]) -> str:
    """Resolves the single host a set of prepared batches runs on.

    Notes:
        A batch runs where it was prepared, because its jobs read data that host holds. Executing batches prepared
        against different hosts together is therefore rejected rather than resolved to one of them.

    Args:
        documents: The prepared batches to resolve the host of.

    Returns:
        The name of the host every named batch was prepared against.

    Raises:
        ValueError: If no batch is named, or if the named batches were prepared against different hosts.
    """
    if not documents:
        message = "Unable to resolve the host for an empty set of prepared batches."
        console.error(message=message, error=ValueError)

    hosts = sorted({document.host for document in documents})
    if len(hosts) > 1:
        message = (
            f"Unable to execute batches prepared against the hosts {hosts} together. A batch runs where it was "
            f"prepared, because its jobs read the data that host holds, so dispatch one host's batches at a time."
        )
        console.error(message=message, error=ValueError)
    return hosts[0]


def forget_prepared_batches(batch_ids: list[str]) -> list[str]:
    """Removes the recorded batches this host holds under the named identifiers.

    Args:
        batch_ids: The identifiers to remove.

    Returns:
        The identifiers that were held and removed.
    """
    removed: list[str] = []
    for batch_id in batch_ids:
        path = batch_path(batch_id=batch_id)
        if not path.is_file():
            continue
        path.unlink()
        _lock_path(path=path).unlink(missing_ok=True)
        removed.append(batch_id)
    return removed


def record_batch_outcome(batch_id: str, outcome: dict[str, Any]) -> bool:
    """Writes what a batch's jobs finally recorded onto its own file.

    Notes:
        This is the step that makes a finished batch answerable, so it runs before the batch is retired from anything
        that tracks it as outstanding.

    Args:
        batch_id: The identifier of the batch to record against.
        outcome: The rendered outcome to store.

    Returns:
        True when the batch was held and updated, and False when this host holds no such batch.

    Raises:
        Timeout: If the batch file's lock cannot be acquired within the timeout period.
    """
    path = batch_path(batch_id=batch_id)
    if not path.is_file():
        return False
    with FileLock(str(_lock_path(path=path))).acquire(timeout=_LOCK_TIMEOUT_SECONDS):
        recorded = PreparedBatch.from_yaml(file_path=path)
        recorded.outcome = dict(outcome)
        recorded.to_yaml(file_path=path)
    return True


def read_batch_outcome(batch_id: str) -> dict[str, Any] | None:
    """Reads what a batch's jobs finally recorded.

    Args:
        batch_id: The identifier of the batch to read.

    Returns:
        The stored outcome, or None when this host holds no such batch or the batch has yet to reach closure.
    """
    path = batch_path(batch_id=batch_id)
    if not path.is_file():
        return None
    outcome = PreparedBatch.from_yaml(file_path=path).outcome
    return outcome or None


def _lock_path(path: Path) -> Path:
    """Resolves the lock file guarding one batch file.

    Args:
        path: The batch file the lock guards.

    Returns:
        The path to the lock file.
    """
    return path.with_suffix(path.suffix + ".lock")
