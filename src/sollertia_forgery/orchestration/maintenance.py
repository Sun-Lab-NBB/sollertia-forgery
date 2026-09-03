"""Provides the operations that return a processing unit to an earlier state, by record or by output."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path

from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import ProcessingTracker, delete_directory

from .dispatch import resolve_dispatch

if TYPE_CHECKING:
    from collections.abc import Sequence


def reset_tracked_jobs(pipeline: str, unit_paths: Sequence[Path], job_ids: Sequence[str] = ()) -> list[str]:
    """Returns the named tracked jobs of several units to the scheduled state.

    Notes:
        Resolves each tracker from its unit rather than taking a path, so a caller names what it wants to be reset
        without knowing where the record sits. Identifiers that a unit does not track are dropped, because a tracker
        rejects a request naming a job it does not hold and would then reset nothing at all.

        Every named identifier is applied to every named unit. A job identifier is derived from the job name and the
        specifier alone, so two units of one project share the identifier of the same stage. A caller holding per-unit
        identifiers passes one unit at a time rather than a flat set.

        Naming no identifier resets every job the unit tracks, which is how a caller returns a unit to a clean slate.

    Args:
        pipeline: The pipeline whose jobs to reset.
        unit_paths: The processing units whose records to clear.
        job_ids: The identifiers to reset, or empty to reset every job each unit tracks.

    Returns:
        The identifiers that were reset, which repeats an identifier held by more than one unit. Naming a pipeline
        that the dispatch table does not support returns nothing.
    """
    dispatch = resolve_dispatch(pipeline=pipeline)
    if dispatch is None:
        return []

    requested = set(job_ids)
    reset: list[str] = []
    for unit_path in unit_paths:
        try:
            tracker_path = dispatch.tracker_path(dispatch.load(unit_path))
        except Exception as exception:
            console.echo(
                message=f"Unable to locate the '{pipeline}' tracker for '{unit_path}'. {exception}",
                level=LogLevel.WARNING,
            )
            continue
        if not tracker_path.is_file():
            continue

        tracker = ProcessingTracker(file_path=tracker_path)
        held_job_ids = list(tracker.snapshot())
        targets = [job_id for job_id in held_job_ids if job_id in requested] if requested else held_job_ids
        if targets:
            reset.extend(tracker.reset_jobs(job_ids=targets))
    return reset


def clean_pipeline_output(pipeline: str, unit_paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Removes a pipeline's output and processing tracker for the named units.

    Notes:
        Removes the directory the pipeline owns outright alongside its tracker, so a later preparation rediscovers every
        job from the acquired data rather than resuming a partial run. A pipeline that writes into a directory it shares
        with the acquired data owns none, so cleaning it removes its tracker alone and leaves the inputs in place.

        A pipeline whose stages also write outside the unit declares those directories separately, and a cleanup
        removes them on the same terms. Each of them names the unit being cleaned, so a tree that holds the directories
        of several units keeps the ones the other units own.

        The unit is loaded rather than discovered, since a cleanup needs the unit's own locations and a unit whose
        pipeline never ran still has output to remove. A unit that cannot be loaded is reported and skipped, leaving the
        others cleaned.

    Args:
        pipeline: The pipeline whose output to remove.
        unit_paths: The processing units to clean.

    Returns:
        One entry per removed path, carrying the ``path`` and the ``removed_bytes`` it held. Naming a pipeline that
        the dispatch table does not support removes nothing.
    """
    dispatch = resolve_dispatch(pipeline=pipeline)
    if dispatch is None:
        return []

    removed: list[dict[str, Any]] = []
    for unit_path in unit_paths:
        try:
            unit = dispatch.load(unit_path)
        except Exception as exception:
            console.echo(
                message=f"Unable to load the unit at '{unit_path}'. {exception}",
                level=LogLevel.WARNING,
            )
            continue

        # The directories a pipeline owns outside its unit are resolved before anything is removed, so a resolver
        # that fails leaves the unit untouched rather than partly cleaned.
        external = () if dispatch.external_output_paths is None else dispatch.external_output_paths(unit)

        # The tracker is a file and the directory that a pipeline owns is a directory, so the two are removed on
        # their own terms rather than through one branch that would have to ask which it was handed.
        tracker_path = dispatch.tracker_path(unit)
        if tracker_path.exists():
            removed.append({"path": str(tracker_path), "removed_bytes": _resolve_path_size(path=tracker_path)})
            tracker_path.unlink()
            # The lock file is bookkeeping beside the tracker rather than tracked output of its own, and its path
            # comes from the tracker's own derivation, so the removal cannot disagree with the file the tracker locks.
            Path(ProcessingTracker(file_path=tracker_path).lock_path).unlink(missing_ok=True)

        owned = dispatch.output_path(unit)
        if owned is not None and owned.exists():
            removed.append({"path": str(owned), "removed_bytes": _resolve_path_size(path=owned)})
            delete_directory(directory_path=owned)

        for directory in external:
            if directory.exists():
                removed.append({"path": str(directory), "removed_bytes": _resolve_path_size(path=directory)})
                delete_directory(directory_path=directory)
    return removed


def _resolve_path_size(path: Path) -> int:
    """Sums the bytes a path holds, counting a directory's whole tree and a file's own size.

    Args:
        path: The file or directory to measure.

    Returns:
        The size in bytes.
    """
    if path.is_file():
        return path.stat().st_size
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())
