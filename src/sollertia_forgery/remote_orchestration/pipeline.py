"""Provides assets for running complex data processing pipelines on remote compute servers.

Notes:
    A processing pipeline represents a higher unit of abstraction relative to the processing job, often leveraging
    multiple sequential or parallel jobs to process the data.
"""

from __future__ import annotations

import shutil as sh
from typing import TYPE_CHECKING
import contextlib
from dataclasses import field, dataclass

from ataraxis_base_utilities import console, ensure_directory_exists
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from ..server import Server, JobStatus
from ..pipelines import ProcessingPipelines
from ..shared_assets import delay_timer

if TYPE_CHECKING:
    from pathlib import Path

    from sollertia_shared_assets import SessionTypes, AcquisitionSystems

    # noinspection PyUnusedImports
    from ..server import Job
    from ..managing import ProjectManifest

# Type alias for the jobs' dictionary to improve readability. Each stage maps to a tuple of (Job, remote log directory,
# tracker job ID) triples. The tracker job ID is the identifier the remote worker uses to update the shared tracker, so
# the orchestrator stores the same value here to reconcile tracker state during resume.
JobsDict = dict[int, tuple[tuple["Job", "Path", str], ...]]


# Maps SLURM's JobStatus values to tracker's ProcessingStatus values. PENDING/RUNNING/UNKNOWN map to None, indicating
# that the job has not yet reached a terminal state.
_SLURM_TO_TRACKER_STATUS: dict[JobStatus, ProcessingStatus | None] = {
    JobStatus.PENDING: None,
    JobStatus.RUNNING: None,
    JobStatus.COMPLETED: ProcessingStatus.SUCCEEDED,
    JobStatus.FAILED: ProcessingStatus.FAILED,
    JobStatus.CANCELLED: ProcessingStatus.FAILED,
    JobStatus.TIMEOUT: ProcessingStatus.FAILED,
    JobStatus.NODE_FAIL: ProcessingStatus.FAILED,
    JobStatus.OUT_OF_MEMORY: ProcessingStatus.FAILED,
    JobStatus.UNKNOWN: None,
}


@dataclass()
class ProcessingPipeline:
    """Defines a data processing pipeline to be executed by a remote compute server over one or more processing stages.

    This class provides the interface for constructing and executing remote data processing pipelines. After
    instantiation, the runtime_cycle() method has to be called cyclically until the pipeline is complete. The class
    handles all interactions with the remote compute server necessary to run the managed pipeline and verify the
    runtime outcome.

    Notes:
        Each pipeline is executed as a series of one or more stages, with each stage using one or more parallel jobs.
        The orchestrator (this class) holds the multi-stage execution graph in memory and sequences the stages by
        polling the SLURM job scheduler for each submitted job's status. The remote processing scripts generate and
        update their own processing tracker, resolving their own SLURM job IDs. The orchestrator only synchronizes the
        tracker to the local machine at the end of the runtime (to resolve the final outcome) or when resuming a
        previously interrupted pipeline.

        This class supports resuming previously interrupted pipelines. When a tracker file already exists on the
        server, the pipeline reconciles the stored job states with SLURM (aborting any jobs left running by an
        interrupted orchestrator) and resumes execution from the first stage that contains an unfinished job.
    """

    pipeline: str | ProcessingPipelines
    """Stores the name of the processing pipeline managed by this instance."""
    server: Server
    """Stores the Server instance that interfaces with the remote compute server running the pipeline."""
    data_path: Path
    """Stores the path to the data directory being processed by the tracked pipeline."""
    jobs: JobsDict
    """Stores the dictionary that maps the pipeline's processing stage integer-codes to tuples of (Job, remote log
    directory, tracker job ID) triples to be submitted to the server as part of executing that stage."""
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
    """Determines whether to keep the logs for the jobs executed as part of the pipeline or (default) to remove them
    after the pipeline successfully ends its runtime. If the pipeline fails to complete its runtime, the logs are kept
    regardless of this setting."""
    rerun_completed_jobs: bool = False
    """Determines whether to rerun all jobs regardless of their state recorded in an existing tracker. When True, the
    pipeline resubmits every stage from the start. When False (default), the pipeline resumes from the first stage that
    contains an unfinished job, skipping jobs that already succeeded."""
    pipeline_status: ProcessingStatus | int = ProcessingStatus.RUNNING
    """Stores the current status of the managed pipeline."""
    _pipeline_stage: int = 0
    """Stores the current (1-based) stage of the tracked pipeline. A value of 0 indicates that the pipeline has not
    yet started executing."""
    _started: bool = False
    """Tracks whether the first runtime cycle (which performs resume reconciliation) has been carried out."""
    _resume_succeeded: set[str] = field(default_factory=set)
    """Stores the tracker job IDs that already succeeded in a previous run, used to skip resubmitting completed jobs
    when resuming an interrupted pipeline."""

    def __post_init__(self) -> None:
        """Carries out the necessary setup tasks to support pipeline execution."""
        ensure_directory_exists(self.local_tracker_path)  # Ensures that the local temporary directory exists.

    def runtime_cycle(self) -> None:
        """Advances the tracked pipeline by one cycle, submitting jobs or resolving the outcome as appropriate.

        This method is the main entry point for all interactions with the processing pipeline managed by this instance.
        The runtime manager process should call this method repeatedly (cyclically) until the 'is_running' property of
        the instance returns False.

        Notes:
            While the 'is_running' property can be used to determine whether the pipeline is still running, to resolve
            the final status of the pipeline (success or failure), the manager process should access the 'status'
            instance property.
        """
        # The first cycle reconciles any existing remote tracker (to support resuming an interrupted pipeline) and
        # submits the first stage that still has unfinished work.
        if not self._started:
            self._started = True
            resume_stage = self._resume_stage()
            if resume_stage > max(self.jobs):
                # Every job already succeeded during a previous run; resolves the outcome without resubmitting.
                self._finalize(success=self._tracker_reports_success())
                return
            self._pipeline_stage = resume_stage
            self._submit_stage()
            return

        # Subsequent cycles poll the SLURM scheduler to determine whether the current stage has finished.
        outcome = self._poll_current_stage()
        if outcome is None:
            return  # The current stage still has pending or running jobs.

        # If any job in the current stage did not complete successfully, aborts the remaining jobs and fails the
        # pipeline.
        if outcome == ProcessingStatus.FAILED:
            self._abort_current_stage()
            self._finalize(success=False)
            return

        # The current stage completed successfully. Advances to the next stage if one exists, otherwise resolves the
        # final outcome from the worker-written tracker.
        next_stage = self._pipeline_stage + 1
        if next_stage in self.jobs:
            self._pipeline_stage = next_stage
            self._submit_stage()
        else:
            self._finalize(success=self._tracker_reports_success())

    def _resume_stage(self) -> int:
        """Reconciles any existing remote tracker with SLURM and resolves the stage to resume execution from.

        Returns:
            The 1-based stage number to start execution from. Returns 1 for a fresh run (no prior tracker), or a value
            greater than the highest stage number when every job already succeeded in a previous run.
        """
        # Attempts to pull a tracker file written by a previous (possibly interrupted) run.
        try:
            self.server.pull(remote_path=self.remote_tracker_path, local_path=self.local_tracker_path)
        except FileNotFoundError:
            return 1  # No prior tracker exists; starts from the first stage.

        tracker = ProcessingTracker.from_yaml(file_path=self.local_tracker_path)
        if not tracker.jobs:
            return 1

        # Aborts any jobs the tracker still reports as RUNNING. These belong to an orchestrator that was interrupted
        # mid-runtime; aborting them prevents duplicate execution when their stage is resubmitted.
        self._abort_stale_running_jobs(tracker=tracker)

        # When a full rerun is requested, ignores the prior progress and resubmits every stage from the start.
        if self.rerun_completed_jobs:
            return 1

        # Caches the job IDs that already succeeded so they can be skipped during submission and polling.
        self._resume_succeeded = {
            job_id for job_id, state in tracker.jobs.items() if state.status == ProcessingStatus.SUCCEEDED
        }

        # Resumes from the first stage that contains a job which has not yet succeeded.
        for stage in sorted(self.jobs):
            if any(job_id not in self._resume_succeeded for _, _, job_id in self.jobs[stage]):
                return stage

        # Every job already succeeded; signals the caller to resolve the outcome without resubmitting.
        return max(self.jobs) + 1

    def _abort_stale_running_jobs(self, tracker: ProcessingTracker) -> None:
        """Aborts SLURM jobs that the tracker reports as RUNNING, using the executor IDs recorded by the workers.

        Args:
            tracker: The ProcessingTracker instance loaded from the remote server.
        """
        for state in tracker.jobs.values():
            if state.status != ProcessingStatus.RUNNING or state.executor_id is None:
                continue
            try:
                slurm_job_id = int(state.executor_id)
            except ValueError:
                continue  # The executor ID is a local process ID, not a SLURM job ID; nothing to abort remotely.
            with contextlib.suppress(Exception):
                self.server.abort_job(slurm_job_id=slurm_job_id)

    def _submit_stage(self) -> None:
        """Submits the jobs for the currently active processing stage to the remote compute server.

        Notes:
            Jobs that already succeeded in a previous run (tracked in _resume_succeeded) are skipped. The remote
            workers generate and update the shared processing tracker themselves; this method only submits jobs to
            SLURM.
        """
        for job, _, job_id in self.jobs[self._pipeline_stage]:
            if job_id in self._resume_succeeded:
                continue  # The job already completed successfully in a previous run; does not resubmit.
            self.server.submit_job(job=job, verbose=False)

    def _poll_current_stage(self) -> ProcessingStatus | None:
        """Polls SLURM for the status of every submitted job in the current stage.

        Returns:
            ProcessingStatus.SUCCEEDED when all submitted jobs in the stage completed successfully, ProcessingStatus.
            FAILED when any job did not complete successfully, or None when the stage still has pending or running jobs.
        """
        in_progress = False
        for job, _, job_id in self.jobs[self._pipeline_stage]:
            # Skips jobs that were not resubmitted because they already succeeded in a previous run.
            if job_id in self._resume_succeeded or job.job_id is None:
                continue

            mapped_status = _SLURM_TO_TRACKER_STATUS.get(self.server.get_job_status(slurm_job_id=int(job.job_id)))
            if mapped_status == ProcessingStatus.FAILED:
                return ProcessingStatus.FAILED  # Fails fast on the first job that did not complete successfully.
            if mapped_status is None:
                in_progress = True

        return None if in_progress else ProcessingStatus.SUCCEEDED

    def _abort_current_stage(self) -> None:
        """Aborts any still-running SLURM jobs submitted for the current stage."""
        for job, _, job_id in self.jobs[self._pipeline_stage]:
            if job_id in self._resume_succeeded or job.job_id is None:
                continue
            with contextlib.suppress(Exception):
                self.server.abort_job(slurm_job_id=int(job.job_id))

    def _tracker_reports_success(self) -> bool:
        """Pulls the worker-written tracker from the server and returns whether it reports the pipeline as complete.

        Returns:
            True when the tracker exists and reports all of its jobs as succeeded, False otherwise.
        """
        try:
            self.server.pull(remote_path=self.remote_tracker_path, local_path=self.local_tracker_path)
        except FileNotFoundError:
            return False  # The workers never produced a tracker; treats the pipeline as failed.
        return ProcessingTracker.from_yaml(file_path=self.local_tracker_path).complete

    def _finalize(self, *, success: bool) -> None:
        """Resolves the pipeline's final status and cleans up local and (on success) remote artifacts.

        Args:
            success: Determines whether the pipeline is finalized as succeeded or failed.
        """
        self.pipeline_status = ProcessingStatus.SUCCEEDED if success else ProcessingStatus.FAILED

        # Removes the local tracker working directory regardless of the outcome.
        sh.rmtree(self.local_tracker_path.parent, ignore_errors=True)

        # On success, optionally removes the remote job log directories. Logs are always kept when the pipeline fails.
        if success and not self.keep_job_logs:
            for stage_jobs in self.jobs.values():
                for _, directory, _ in stage_jobs:
                    with contextlib.suppress(Exception):
                        self.server.remove(remote_path=directory, recursive=True, is_dir=True)

    @property
    def is_running(self) -> bool:
        """Returns True if the pipeline is currently running."""
        return self.pipeline_status == ProcessingStatus.RUNNING

    @property
    def status(self) -> ProcessingStatus:
        """Returns the current status of the pipeline packaged into a ProcessingStatus instance."""
        return ProcessingStatus(self.pipeline_status)


def check_session_eligibility(
    manifest: ProjectManifest,
    session: str,
    pipeline: str | ProcessingPipelines,
    server: Server,
    supported_systems: set[str | AcquisitionSystems],
    supported_sessions: set[str | SessionTypes],
    *,
    allow_reprocessing: bool = False,
    configuration_path: Path | None = None,
) -> str | None:
    """Checks whether the target session meets the eligibility criteria for being processed with the specified pipeline.

    This function aggregates common eligibility checks to streamline the process for all supported processing pipelines.

    Args:
        manifest: The ProjectManifest instance that stores the metadata for the project under which the session was
            conducted.
        session: The unique identifier of the session to be processed.
        pipeline: The processing pipeline with which to process the session's data. Must be one of the
            ProcessingPipelines values.
        server: The Server instance that communicates with the remote compute server used to execute the pipeline.
        supported_systems: The data acquisition systems that support this type of processing.
        supported_sessions: The session types that support this type of processing.
        allow_reprocessing: Determines whether to reprocess the session if it has already been processed with this
            pipeline.
        configuration_path: The path to the pipeline's configuration file on the remote server. Required for the
            single-day cindra, video, and multi-day cindra pipelines. If provided, the function verifies the file
            exists on the remote server.

    Returns:
        None if the session is eligible for processing. Otherwise, returns a string describing why the session
        was excluded from processing.
    """
    # Parses the target session data from the manifest file.
    session_data = manifest.get_session_data(session=session)
    session_type = session_data["type"][0]
    session_system = session_data["system"][0]
    complete = session_data["complete"][0]
    integrity = bool(session_data["integrity"][0])

    # Ensures that the pipeline's name is stored as a ProcessingPipelines instance.
    pipeline = ProcessingPipelines(pipeline)

    # Determines whether the session has already been processed using the specified pipeline and whether the pipeline
    # requires a configuration file or prior processing steps.
    requires_configuration = False
    requires_integrity = True
    requires_cindra = False
    if pipeline == ProcessingPipelines.CHECKSUM:
        processed = integrity
        requires_integrity = False  # Checksum pipeline does not require prior integrity verification
    elif pipeline == ProcessingPipelines.BEHAVIOR:
        processed = bool(session_data["behavior"][0])
    elif pipeline == ProcessingPipelines.VIDEO:
        processed = bool(session_data["video"][0])
        requires_configuration = True
    elif pipeline == ProcessingPipelines.CINDRA_SINGLE_RECORDING:
        processed = bool(session_data["cindra"][0])
        requires_configuration = True
    elif pipeline == ProcessingPipelines.CINDRA_MULTI_RECORDING:
        # Multiday processing requires cindra to be completed first; uses tracker-based reprocessing check
        processed = False  # Determined by the tracker check below
        requires_configuration = True
        requires_cindra = True
    elif pipeline == ProcessingPipelines.FORGING:
        # Forging pipeline performs internal checks for available data and adjusts its runtime accordingly
        processed = False  # Determined by the tracker check below
    else:
        return f"The pipeline '{pipeline}' is not supported. Use one of: {list(ProcessingPipelines)}."

    # If the session was acquired using a data acquisition system that does not support this type of processing,
    # skips processing the session.
    if session_system not in supported_systems:
        return (
            f"The session was acquired using the acquisition system '{session_system},' "
            f"which does not support {pipeline} processing."
        )

    # If the session type is not one of the supported types, skips processing the session.
    if session_type not in supported_sessions:
        return f"The session is of type '{session_type},' which does not support {pipeline} processing."

    # Prevents processing incomplete sessions.
    if not complete:
        return "The session is marked as 'incomplete,' which excludes it from unsupervised data processing."

    # For all pipelines except CHECKSUM, the session must have passed the integrity verification pipeline.
    if requires_integrity and not integrity:
        return (
            "The session has not been processed with the integrity verification pipeline. "
            "Run the CHECKSUM pipeline first to verify the session's data integrity."
        )

    # For the multi-day cindra pipeline, the session must have been processed with the single-day cindra pipeline first.
    if requires_cindra and not bool(session_data["cindra"][0]):
        return (
            "The session has not been processed with the single-day cindra pipeline. "
            "Run the single-day cindra pipeline first to extract calcium fluorescence data."
        )

    # If the session has already been processed and reprocessing is not allowed, skips processing the session.
    if processed and not allow_reprocessing:
        return "The session has already been processed with this pipeline and reprocessing is disabled."

    # If the target processing pipeline requires a specific server-side configuration file, ensures that the file is
    # present at the expected remote server location.
    if requires_configuration:
        if configuration_path is None:
            return "The pipeline requires a configuration file, but no configuration path was provided."
        if not server.exists(remote_path=configuration_path):
            return f"The pipeline's configuration file does not exist on the remote server at: {configuration_path}."

    # The session is eligible for processing with this pipeline.
    return None


def execute_pipelines(
    pipelines: tuple[ProcessingPipeline, ...],
    stage_name: str,
    batch_size: int | None = 1,
    poll_delay: int = 10,
) -> tuple[int, int]:
    """Executes the input processing pipelines as sequential batches.

    This function provides a standardized interface for executing ProcessingPipeline instances in configurable batch
    sizes. Batch execution allows controlling the degree of parallelism to balance the throughput against the I/O load
    on the remote compute server.

    Args:
        pipelines: The ProcessingPipeline instances that define the pipelines to execute.
        stage_name: The name of the current processing stage, used for the progress bar labeling.
        batch_size: The maximum number of pipelines to execute concurrently within each batch. A batch size of 1
            executes pipelines sequentially. A batch size of None submits and executes all pipelines at once.
        poll_delay: The delay (in seconds) between polling the server for job status updates.

    Returns:
        A tuple of two integers: (successful_count, failed_count).
    """
    successful_count = 0
    failed_count = 0

    # Tracks which pipelines have been counted using their index
    counted_indices: set[int] = set()

    # Splits the overall sequence of pipelines into batches. If batch_size is None, all pipelines are executed at once.
    indexed_pipelines = list(enumerate(pipelines))
    effective_batch_size = len(pipelines) if batch_size is None else batch_size
    batches = [
        indexed_pipelines[i : i + effective_batch_size] for i in range(0, len(indexed_pipelines), effective_batch_size)
    ]

    with console.progress(
        total=len(pipelines), description=f"Executing {stage_name} pipelines", unit="pipeline"
    ) as pbar:
        for batch in batches:
            batch_complete = False

            # Processes the current batch until all batch pipelines complete
            while not batch_complete:
                batch_complete = True

                for idx, pipeline in batch:
                    # Checks if the pipeline is still running
                    if pipeline.is_running:
                        pipeline.runtime_cycle()
                        batch_complete = False

                    # If the pipeline completed and is not yet counted, updates the counters
                    if idx not in counted_indices:
                        if pipeline.pipeline_status == ProcessingStatus.FAILED:
                            failed_count += 1
                            counted_indices.add(idx)
                            pbar.update()
                        elif pipeline.pipeline_status == ProcessingStatus.SUCCEEDED:
                            successful_count += 1
                            counted_indices.add(idx)
                            pbar.update()

                # Delays between pipeline resolution cycles to avoid overwhelming the communication line
                if not batch_complete:
                    delay_timer.delay(delay=poll_delay, allow_sleep=True, block=False)

    return successful_count, failed_count
