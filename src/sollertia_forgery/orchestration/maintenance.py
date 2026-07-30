"""Provides the operations that return a processing unit to an earlier state, by record or by output."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ataraxis_base_utilities import console
from ataraxis_data_structures import ProcessingTracker, delete_directory

from .dispatch import resolve_dispatch

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Sequence


def reset_tracked_jobs(pipeline: str, unit_paths: Sequence[Path], job_ids: Sequence[str] = ()) -> list[str]:
    """Returns the named tracked jobs of several units to the scheduled state.

    Notes:
        Resolves each tracker from its unit rather than taking a path, so a caller names what it wants reset without
        knowing where the record sits. Identifiers a unit does not track are dropped, because a tracker rejects a
        request naming a job it does not hold and would then reset nothing at all. That dropping is what lets one call
        carry a whole batch's identifiers and have each unit reset only its own share.

        Naming no identifier resets every job the unit tracks, which is how a caller returns a unit to a clean slate.

    Args:
        pipeline: The pipeline whose jobs to reset.
        unit_paths: The processing units whose records to clear.
        job_ids: The identifiers to reset, or empty to reset every job each unit tracks.

    Returns:
        The identifiers that were reset, which repeats an identifier held by more than one unit.
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
            console.echo(message=f"Unable to locate the '{pipeline}' tracker for '{unit_path}'. {exception}")
            continue
        if not tracker_path.is_file():
            continue

        tracker = ProcessingTracker(file_path=tracker_path)
        held = list(tracker.snapshot())
        targets = held if not requested else [job_id for job_id in held if job_id in requested]
        if targets:
            reset.extend(tracker.reset_jobs(job_ids=targets))
    return reset


def clean_pipeline_output(pipeline: str, unit_paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Removes a pipeline's output and processing tracker for the named units.

    Notes:
        Removes the directory the pipeline owns outright alongside its tracker, so a later preparation rediscovers every
        job from the acquired data rather than resuming a partial run. A pipeline that writes into a directory it shares
        with the acquired data owns none, so cleaning it removes its tracker alone and leaves the inputs in place.

        The unit is loaded rather than discovered, since a cleanup needs the unit's own locations and a unit whose
        pipeline never ran still has output to remove. A unit that cannot be loaded is reported and skipped, leaving the
        others cleaned.

    Args:
        pipeline: The pipeline whose output to remove.
        unit_paths: The processing units to clean.

    Returns:
        One entry per removed path, carrying the ``path`` and the ``removed_bytes`` it held.
    """
    dispatch = resolve_dispatch(pipeline=pipeline)
    if dispatch is None:
        return []

    removed: list[dict[str, Any]] = []
    for unit_path in unit_paths:
        try:
            unit = dispatch.load(unit_path)
        except Exception as exception:
            console.echo(message=f"Unable to load the unit at '{unit_path}'. {exception}")
            continue

        targets = [dispatch.tracker_path(unit)]
        owned = dispatch.output_path(unit)
        if owned is not None:
            targets.append(owned)

        for target in targets:
            if not target.exists():
                continue
            removed.append({"path": str(target), "removed_bytes": resolve_path_size(path=target)})
            if target.is_dir():
                delete_directory(directory_path=target)
            else:
                target.unlink()
                # The tracker's lock file is bookkeeping beside it rather than tracked output of its own.
                target.with_suffix(target.suffix + ".lock").unlink(missing_ok=True)
    return removed


def resolve_path_size(path: Path) -> int:
    """Sums the bytes a path holds, counting a directory's whole tree and a file's own size.

    Args:
        path: The file or directory to measure.

    Returns:
        The size in bytes.
    """
    if path.is_file():
        return path.stat().st_size
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())
