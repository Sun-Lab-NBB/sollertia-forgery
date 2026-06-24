"""Provides the processing-tracker job-registry alignment helper shared across library pipelines.
"""

from __future__ import annotations

from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import ProcessingTracker


def prepare_tracker(tracker: ProcessingTracker, jobs: list[tuple[str, str]], universe: list[tuple[str, str]]) -> None:
    """Aligns a processing tracker's job registry with the jobs requested for the current pipeline invocation.

    Notes:
        Foreign entries are detected by comparing the tracker's existing job IDs against the ``universe`` of every
        job the current input set could produce, not against the invocation's requested ``jobs`` subset. This lets
        a subset invocation (for example, a single concurrent remote job, or a discovery that only finds part of
        the expected inputs) align the tracker without wiping previously-completed state for its sibling jobs. Only
        entries that fall outside the universe are treated as architectural drift (the input set itself has changed
        since the tracker was last written) and surfaced through a warning before the tracker is rebuilt.

        If the tracker file does not yet exist on disk, the helper initializes it with the requested jobs. If the
        file exists and contains job IDs that are not part of the universe, those entries are classified as foreign
        and the helper emits a warning before resetting and reinitializing the tracker. If the file exists with only
        universe-valid entries but is missing some requested jobs, the helper performs an additive
        ``initialize_jobs`` call that registers the missing entries without clobbering any existing state. If the
        file already contains every requested job, the helper is a no-op, which keeps ``initialize_jobs`` from
        emitting duplicate-entry warnings for the fully-aligned case.

    Args:
        tracker: The ProcessingTracker instance bound to the target directory.
        jobs: The list of ``(job_name, specifier)`` tuples the current invocation intends to execute.
        universe: The list of ``(job_name, specifier)`` tuples enumerating every job the current input set could
            produce. Used exclusively for foreign-entry detection. Callers whose requested set is always the full
            job universe pass the same list as ``jobs``.
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
