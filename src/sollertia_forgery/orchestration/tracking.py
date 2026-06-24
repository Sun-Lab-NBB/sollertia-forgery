"""Provides the processing-tracker job-registry alignment helper shared across library pipelines."""

from __future__ import annotations

from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import ProcessingTracker


def prepare_tracker(tracker: ProcessingTracker, jobs: list[tuple[str, str]], universe: list[tuple[str, str]]) -> None:
    """Aligns a processing tracker's job registry with the jobs requested for the current pipeline invocation.

    Notes:
        Foreign entries are detected against the ``universe`` of every job the current input set could produce, not
        against the requested ``jobs`` subset. This lets a subset invocation (a single concurrent remote job, or a
        discovery that finds only part of the expected inputs) align the tracker without wiping the completed state
        of its sibling jobs.

        The helper initializes a missing tracker with the requested jobs, additively registers any requested jobs
        absent from an otherwise-valid tracker, and is a no-op when every requested job is already present. Only
        when the tracker holds entries outside the universe (the input set itself has changed) does it warn and
        rebuild the tracker from the requested jobs.

    Args:
        tracker: The processing tracker bound to the target directory.
        jobs: The ``(job_name, specifier)`` tuples the current invocation intends to execute.
        universe: The ``(job_name, specifier)`` tuples enumerating every job the current input set could produce,
            used only for foreign-entry detection. Callers whose requested set is always the full universe pass the
            same list as ``jobs``.
    """
    universe_ids = {
        ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier) for job_name, specifier in universe
    }
    requested_ids = {
        ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier) for job_name, specifier in jobs
    }

    if not tracker.file_path.exists():
        tracker.initialize_jobs(jobs=jobs)
        return

    existing_ids = set(tracker.find_jobs(job_name="").keys())
    foreign_ids = existing_ids - universe_ids

    if foreign_ids:
        console.echo(
            message=(
                f"The processing tracker at '{tracker.file_path}' contains {len(foreign_ids)} job entries "
                f"that are not part of the current job universe. Resetting and reinitializing the tracker to "
                f"match the requested jobs. Foreign job IDs: {sorted(foreign_ids)}."
            ),
            level=LogLevel.WARNING,
        )
        tracker.reset()
        tracker.initialize_jobs(jobs=jobs)
        return

    if not requested_ids.issubset(existing_ids):
        tracker.initialize_jobs(jobs=jobs)
