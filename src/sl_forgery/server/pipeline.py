"""This module provides the assets for running complex data processing pipelines on remote compute servers. A processing
pipeline represents a higher unit of abstraction relative to the processing job, often leveraging multiple sequential or
parallel jobs to process the data.
"""

import shutil as sh
from typing import TYPE_CHECKING
import contextlib
from dataclasses import field, dataclass

from sl_shared_assets import (
    ProcessingStatus,
    ProcessingTracker,
)
from ataraxis_base_utilities import ensure_directory_exists

from .server import Server, JobStatus

if TYPE_CHECKING:
    from pathlib import Path

    # noinspection PyUnusedImports
    from .job import Job

# Type alias for the jobs' dictionary to improve readability
JobsDict = dict[int, tuple[tuple["Job", "Path"], ...]]


# Maps SLURM's JobStatus values to tracker's ProcessingStatus values for status reconciliation
_SLURM_TO_TRACKER_STATUS: dict[JobStatus, ProcessingStatus | None] = {
    JobStatus.PENDING: None,  # Still in progress, no update needed
    JobStatus.RUNNING: None,  # Still in progress, no update needed
    JobStatus.COMPLETED: ProcessingStatus.SUCCEEDED,
    JobStatus.FAILED: ProcessingStatus.FAILED,
    JobStatus.CANCELLED: ProcessingStatus.FAILED,
    JobStatus.TIMEOUT: ProcessingStatus.FAILED,
    JobStatus.NODE_FAIL: ProcessingStatus.FAILED,
    JobStatus.OUT_OF_MEMORY: ProcessingStatus.FAILED,
    JobStatus.UNKNOWN: None,  # Cannot determine, leave as-is
}


@dataclass()
class ProcessingPipeline:
    """Defines a data processing pipeline to be executed by a remote compute server over one or more processing stages.

    This class provides the interfaces for constructing and executing remote data processing pipelines. After
    instantiation, the class automatically handles all interactions with the remote compute server necessary to run the
    managed pipeline and verify the runtime outcome via the runtime_cycle() method that has to be called cyclically
    until the pipeline is complete.

    Notes:
        Each pipeline is executed as a series of one or more stages with each stage using one or more parallel jobs.
        Therefore, each pipeline can be seen as an execution graph that sequentially submits batches of jobs to the
        remote server. The processing graph for each pipeline is fully resolved at class instantiation.

        This class supports resuming previously interrupted pipelines. If a tracker file already exists on the server,
        the pipeline reconciles the stored job states with SLURM to determine the actual status of each job before
        resuming execution from the appropriate stage.
    """

    pipeline: str
    """Stores the name of the processing pipeline managed by this instance."""
    server: Server
    """Stores the Server instance that interfaces with the remote compute server running the pipeline."""
    data_path: Path
    """Stores the path to the data directory being processed by the tracked pipeline."""
    jobs: JobsDict
    """Stores the dictionary that maps the pipeline processing stage integer-codes to two-element tuples. Each tuple
    stores the Job object and the path to its remote working directory to be submitted to the server as part of
    executing that stage."""
    remote_tracker_path: Path
    """Stores the path to the pipeline's processing tracker .yaml file stored on the remote compute server."""
    local_tracker_path: Path
    """Stores the path to the pipeline's processing tracker .yaml file on the local machine."""
    session: str
    """Stores the unique identifier of the session processed by the tracked pipeline."""
    animal: str
    """Stores the unique identifier of the animal processed by the tracked pipeline."""
    project: str
    """Stores the name of the project processed by the tracked pipeline."""
    keep_job_logs: bool = False
    """Determines whether to keep the logs for the jobs executed as part of the pipeline or (default) to remove
    them after pipeline successfully ends its runtime. If the pipeline fails to complete its runtime, the logs are kept
    regardless of this setting."""
    rerun_completed_jobs: bool = False
    """Determines whether to reset the tracker and rerun all jobs regardless of their completion status. When set to
    True, the pipeline clears the existing tracker and starts all jobs from scratch. When False (default), the
    pipeline preserves completed jobs and only reruns failed or pending jobs."""
    pipeline_status: ProcessingStatus | int = ProcessingStatus.RUNNING
    """Stores the current status of the managed pipeline."""
    _pipeline_stage: int = 0
    """Stores the current stage of the tracked pipeline."""
    _job_ids: dict[int, tuple[str, ...]] = field(default_factory=dict)
    """Maps pipeline stages to tuples of unique job IDs used by the ProcessingTracker instance to track each job's
    runtime state."""
    _tracker_initialized: bool = False
    """Tracks whether the ProcessingTracker has been initialized with job IDs."""

    def __post_init__(self) -> None:
        """Carries out the necessary setup tasks to support pipeline execution."""
        ensure_directory_exists(self.local_tracker_path)  # Ensures that the local temporary directory exists

        # Generates unique job IDs for all jobs in all processing stages using the ProcessingTracker's static method
        for stage, stage_jobs in self.jobs.items():
            stage_job_ids = []
            for job, _ in stage_jobs:
                job_id = ProcessingTracker.generate_job_id(
                    session_path=self.data_path,
                    job_name=job.job_name,
                )
                stage_job_ids.append(job_id)
            self._job_ids[stage] = tuple(stage_job_ids)

    def runtime_cycle(self) -> None:
        """Checks the current status of the tracked pipeline and, if necessary, submits additional batches of jobs to
        the remote server to progress the pipeline.

        This method is the main entry point for all interactions with the processing pipeline managed by this instance.
        It checks the current state of the pipeline, advances the pipeline's processing stage, and submits the necessary
        jobs to the remote server. The runtime manager process should call this method repeatedly (cyclically) to run
        the pipeline until the 'is_running' property of the instance returns False.

        Notes:
            While the 'is_running' property can be used to determine whether the pipeline is still running, to resolve
            the final status of the pipeline (success or failure), the manager process should access the
            'status' instance property.
        """
        # This clause is executed the first time the method is called for the newly initialized instance. It
        # initializes the tracker, reconciles any existing job states, and determines the starting processing stage.
        if self._pipeline_stage == 0:
            self._initialize_tracker()

            # Otherwise, starts or resumes executing the pipeline.
            self._pipeline_stage += 1
            self._submit_jobs()
            return

        # Pulls the tracker's data to check job statuses (read-only)
        try:
            self.server.pull(remote_path=self.remote_tracker_path, local_path=self.local_tracker_path)
        except FileNotFoundError:
            # Tracker disappeared unexpectedly, which indicates user intervention and is interpreted as the user
            # aborting the pipeline.
            self._finalize_pipeline_aborted()
            return

        # Loads the processing tracker's data.
        tracker = ProcessingTracker(file_path=self.local_tracker_path)

        # Checks the status of all jobs in the current stage
        stage_job_ids = self._job_ids[self._pipeline_stage]
        stage_complete = True

        for job_id in stage_job_ids:
            job_status = tracker.get_job_status(job_id)

            # If any job failed, finalizes the pipeline as failed
            if job_status == ProcessingStatus.FAILED:
                self._finalize_pipeline_failure()
                return

            # If the tracker shows the job as RUNNING, reconciles with SLURM to detect externally terminated jobs
            if job_status == ProcessingStatus.RUNNING:
                slurm_status = self.server.get_job_status(slurm_job_id=tracker.jobs[job_id].slurm_job_id)
                mapped_status = _SLURM_TO_TRACKER_STATUS.get(slurm_status)

                # If SLURM reports failure but the tracker shows running, the job was terminated externally
                if mapped_status == ProcessingStatus.FAILED:
                    self._finalize_pipeline_aborted()
                    return

            # If any job is not yet succeeded, the stage is not complete
            if job_status != ProcessingStatus.SUCCEEDED:
                stage_complete = False

        # If the stage is not complete, waits for jobs to finish
        if not stage_complete:
            return

        # All jobs in the current stage have completed successfully.
        # Checks if all stages are complete or advances to the next stage.
        if tracker.complete:
            self._finalize_pipeline_success()
            return

        # Advances to the next stage if one exists
        next_stage = self._pipeline_stage + 1
        if next_stage in self.jobs:
            self._pipeline_stage = next_stage
            self._submit_jobs()
        else:
            # The pipeline does not have further processing stages, but the tracker indicates that the pipeline is
            # incomplete. This is unexpected and indicates that the pipeline is misconfigured or aborted.
            self._finalize_pipeline_aborted()

    def _initialize_tracker(self) -> None:
        """Initializes the ProcessingTracker instance for the managed pipeline, reconciling any existing job states
        with SLURM.

        This method handles both fresh pipeline starts and resumption of previously interrupted pipelines. If a tracker
        file already exists on the server, it reconciles the stored job states with SLURM, aborts any running jobs
        (to allow clean restarts), and pushes the reconciled tracker back to the server.

        Notes:
            This is the ONLY method that writes to the tracker. After initialization, the pipeline only reads the
            tracker to monitor job status. The jobs themselves update the tracker during execution.
        """
        if self._tracker_initialized:
            return

        # Collects all job IDs from all pipeline's stages
        all_job_ids = []
        for stage_job_ids in self._job_ids.values():
            all_job_ids.extend(stage_job_ids)
        all_job_ids_set = set(all_job_ids)

        # Attempts to pull any existing tracker file from the server
        tracker_exists = False
        try:
            self.server.pull(remote_path=self.remote_tracker_path, local_path=self.local_tracker_path)
            tracker_exists = True
        except FileNotFoundError:
            pass  # Tracker file does not exist; a new one will be created

        tracker = ProcessingTracker(file_path=self.local_tracker_path)

        if tracker_exists and tracker.jobs:
            # Checks if the tracker's job IDs match the pipeline's job IDs
            tracker_job_ids_set = set(tracker.jobs.keys())

            if tracker_job_ids_set != all_job_ids_set:
                # Tracker configuration does not match the pipeline; resets and re-initializes the tracker
                tracker.reset()
                tracker.initialize_jobs(all_job_ids)
            elif self.rerun_completed_jobs:
                # User requested to rerun all jobs; aborts any running jobs and resets the tracker
                self._reconcile_and_abort_running_jobs(tracker=tracker)
                tracker.reset()
                tracker.initialize_jobs(all_job_ids)
            else:
                # Tracker matches pipeline configuration; reconciles job states with SLURM
                self._reconcile_and_abort_running_jobs(tracker=tracker)

                if tracker.complete:
                    # All jobs completed; resets the tracker and restarts from scratch
                    tracker.reset()
                    tracker.initialize_jobs(all_job_ids)
                elif tracker.encountered_error:
                    # Some jobs failed; resets only failed jobs to SCHEDULED, keeping completed jobs
                    self._reset_failed_jobs(tracker=tracker)

            # Determines the stage to resume from based on job statuses
            self._pipeline_stage = self._determine_resume_stage(tracker=tracker)
        else:
            # Tracker does not exist or is empty; initializes all jobs from scratch
            tracker.initialize_jobs(all_job_ids)

        # Pushes the initialized/reconciled tracker to the server. This is the only push operation performed by
        # the pipeline; further updates are made by the jobs themselves.
        self.server.push(local_path=self.local_tracker_path, remote_path=self.remote_tracker_path)
        self._tracker_initialized = True

    def _reconcile_and_abort_running_jobs(self, tracker: ProcessingTracker) -> None:
        """Reconciles pipeline's job states stored in the tracker file with SLURM and aborts any active SLURM jobs.

        For all jobs that have a SLURM ID, queries the SLURM manager to check their actual status. Jobs that are still
        pending, running, or in an unknown state are aborted to allow a clean restart. Jobs that have completed or
        failed are updated accordingly.

        Args:
            tracker: The ProcessingTracker instance to reconcile.
        """
        for job_id, job_state in tracker.jobs.items():
            if job_state.slurm_job_id is None:
                # Job has no SLURM ID; resets to SCHEDULED if it was marked as RUNNING
                if job_state.status == ProcessingStatus.RUNNING:
                    job_state.status = ProcessingStatus.SCHEDULED
                continue

            # Queries SLURM for the actual job status
            slurm_status = self.server.get_job_status(slurm_job_id=job_state.slurm_job_id)
            mapped_status = _SLURM_TO_TRACKER_STATUS.get(slurm_status)

            if mapped_status == ProcessingStatus.SUCCEEDED:
                # Job completed successfully; updates the tracker
                tracker.complete_job(job_id)

            elif mapped_status == ProcessingStatus.FAILED:
                # Job failed; updates the tracker
                tracker.fail_job(job_id)

            else:
                # Aborts the job to allow a clean restart
                self.server.abort_job(slurm_job_id=job_state.slurm_job_id)

                # Resets the job to SCHEDULED so it can be resubmitted
                job_state.status = ProcessingStatus.SCHEDULED
                job_state.slurm_job_id = None

    @staticmethod
    def _reset_failed_jobs(tracker: ProcessingTracker) -> None:
        """Resets failed jobs in the tracker to SCHEDULED status while preserving completed jobs.

        This method iterates through all jobs in the tracker and resets any job marked as FAILED back to SCHEDULED,
        allowing them to be resubmitted. Jobs marked as SUCCEEDED are left unchanged.

        Args:
            tracker: The ProcessingTracker instance containing the jobs to reset.
        """
        for job_state in tracker.jobs.values():
            if job_state.status == ProcessingStatus.FAILED:
                job_state.status = ProcessingStatus.SCHEDULED
                job_state.slurm_job_id = None

    def _determine_resume_stage(self, tracker: ProcessingTracker) -> int:
        """Determines which processing stage to resume the pipeline from based on the managed job statuses.

        Notes:
            This method allows resuming partially completed pipelines while preserving the sequential dependencies
            between processing stages.

        Args:
            tracker: The ProcessingTracker instance to examine.

        Returns:
            The processing stage number to resume the pipeline from. Returns 0 to start from the first stage.
        """
        for stage in sorted(self.jobs.keys()):
            stage_job_ids = self._job_ids[stage]

            for job_id in stage_job_ids:
                # If the job is not marked as SUCCEEDED, resumes from this stage
                if tracker.get_job_status(job_id) != ProcessingStatus.SUCCEEDED:
                    # Returns stage - 1 to make the returned stage work with how stages are tracked during
                    # the runtime cycle.
                    return stage - 1

        # All jobs are scheduled (fresh start after reset); starts from stage 0
        return 0

    def _submit_jobs(self) -> None:
        """Submits the processing jobs for the currently active processing stage to the remote compute server.

        Notes:
            Jobs that are already marked as SUCCEEDED in the tracker are skipped. The jobs themselves update the
            tracker when they start running and when they complete. This method only submits jobs to SLURM.
        """
        stage_jobs = self.jobs[self._pipeline_stage]
        stage_job_ids = self._job_ids[self._pipeline_stage]
        tracker = ProcessingTracker(file_path=self.local_tracker_path)

        for (job, _), job_id in zip(stage_jobs, stage_job_ids, strict=True):
            if tracker.get_job_status(job_id) == ProcessingStatus.SUCCEEDED:
                # Job already completed; skips submission
                continue

            # Submits the job to the server.
            self.server.submit_job(job=job, verbose=False)

    def _finalize_pipeline_failure(self) -> None:
        """Handles pipeline finalization when one or more processing jobs fail."""
        sh.rmtree(self.local_tracker_path.parent, ignore_errors=True)
        self.pipeline_status = ProcessingStatus.FAILED

    def _finalize_pipeline_success(self) -> None:
        """Handles pipeline finalization when all jobs complete successfully."""
        sh.rmtree(self.local_tracker_path.parent, ignore_errors=True)
        self.pipeline_status = ProcessingStatus.SUCCEEDED

        # If the pipeline was configured to remove logs after completing successfully, removes the runtime log for
        # each job submitted as part of this pipeline from the remote server.
        if not self.keep_job_logs:
            for stage_jobs in self.jobs.values():
                for _, directory in stage_jobs:
                    self.server.remove(remote_path=directory, recursive=True, is_dir=True)

    def _finalize_pipeline_aborted(self) -> None:
        """Handles pipeline finalization when the pipeline is in an unexpected state, indicating that it was aborted
        by SLURM or the user.
        """
        sh.rmtree(self.local_tracker_path.parent, ignore_errors=True)
        self.pipeline_status = ProcessingStatus.FAILED
        # Removes the tracker file from the server to prevent deadlocking further runtimes
        with contextlib.suppress(Exception):
            self.server.remove(remote_path=self.remote_tracker_path, is_dir=False)

    @property
    def is_running(self) -> bool:
        """Returns True if the pipeline is currently running."""
        return self.pipeline_status == ProcessingStatus.RUNNING

    @property
    def status(self) -> ProcessingStatus:
        """Returns the current status of the pipeline packaged into a ProcessingStatus instance."""
        return ProcessingStatus(self.pipeline_status)
