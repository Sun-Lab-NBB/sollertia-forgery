"""This module provides tools used to run complex data processing pipelines on remote compute servers. A processing
pipeline represents a higher unit of abstraction relative to the Job class, often leveraging multiple sequential or
parallel jobs to process the data.
"""

from random import randint
import shutil as sh
from pathlib import Path
from dataclasses import dataclass

from xxhash import xxh3_64
from ataraxis_base_utilities import console, ensure_directory_exists
from ataraxis_time.time_helpers import get_timestamp
from sl_shared_assets import ProcessingPipelines, ProcessingStatus, ProcessingTracker

from .job import Job
from .server import Server


@dataclass()
class ProcessingPipeline:
    """Provides an interface to construct and execute data processing pipelines on the target remote compute server.

    This class functions as an interface for all data processing pipelines running on Sun lab compute servers. It is
    pipeline-type-agnostic and works for all data processing pipelines used in the lab. After instantiation, the class
    automatically handles all interactions with the server necessary to run the remote processing pipeline and
    verify the runtime outcome via the runtime_cycle() method that has to be called cyclically until the pipeline is
    complete.

    Notes:
        Each pipeline is executed as a series of one or more stages with each stage using one or more parallel jobs.
        Therefore, each pipeline can be seen as an execution graph that sequentially submits batches of jobs to the
        remote server. The processing graph for each pipeline is fully resolved at the instantiation of this class, so
        each instance contains the necessary data to run the entire processing pipeline.

        The minimum self-contained unit of the processing pipeline is a single job. Since jobs can depend on the output
        of other jobs, they are organized into stages based on the dependency graph between jobs. Combined with cluster
        management software, such as SLURM, this class can efficiently execute processing pipelines on scalable compute
        clusters.
    """

    pipeline_type: ProcessingPipelines
    """Stores the name of the processing pipeline managed by this instance. Primarily, this is used to identify the 
    pipeline to the user in terminal messages and logs."""
    server: Server
    """Store the reference to the Server object used to interface with the remote server running the pipeline."""
    manager_id: int
    """The unique identifier for the manager process that constructs and manages the runtime of the tracked pipeline."""
    jobs: dict[int, tuple[tuple[Job, Path], ...]]
    """Stores the dictionary that maps the pipeline processing stage integer-codes to two-element tuples. Each tuple
    stores the Job object and the path to its remote working directory to be submitted to the server as part of that 
    executing that stage."""
    remote_tracker_path: Path
    """Stores the path to the pipeline's processing tracker .yaml file stored on the remote compute server."""
    local_tracker_path: Path
    """Stores the path to the pipeline's processing tracker .yaml file on the local machine. The remote file is 
    pulled to this location when the instance verifies the outcome of the tracked processing pipeline."""
    session: str
    """Stores the ID of the session whose data is being processed by the tracked pipeline."""
    animal: str
    """Stores the ID of the animal whose data is being processed by the tracked pipeline."""
    project: str
    """Stores the name of the project whose data is being processed by the tracked pipeline."""
    keep_job_logs: bool = False
    """Determines whether to keep the logs for the jobs making up the pipeline execution graph or (default) to remove 
    them after pipeline successfully ends its runtime. If the pipeline fails to complete its runtime, the logs are kept 
    regardless of this setting."""
    pipeline_status: ProcessingStatus | int = ProcessingStatus.RUNNING
    """Stores the current status of the tracked remote pipeline. This field is updated each time runtime_cycle() 
    instance method is called."""
    _pipeline_stage: int = 0
    """Stores the current stage of the tracked pipeline. This field is monotonically incremented by the runtime_cycle()
    method to sequentially submit batches of jobs to the server in a processing-stage-driven fashion."""

    def __post_init__(self) -> None:
        """Carries out the necessary filesystem setup tasks to support pipeline execution."""
        # Ensures that the input processing tracker file name is supported.
        if self.pipeline_type not in tuple(ProcessingPipelines):
            message = (
                f"Unsupported processing pipeline type encountered when instantiating a ProcessingPipeline "
                f"instance: {self.pipeline_type}. Currently, only the following pipeline types are "
                f"supported: {', '.join(tuple(ProcessingPipelines))}."
            )
            console.error(message=message, error=ValueError)

        ensure_directory_exists(self.local_tracker_path)  # Ensures that the local temporary directory exists

    def runtime_cycle(self) -> None:
        """Checks the current status of the tracked pipeline and, if necessary, submits additional batches of jobs to
        the remote server to progress the pipeline.

        This method is the main entry point for all interactions with the processing pipeline managed by this instance.
        It checks the current state of the pipeline, advances the pipeline's processing stage, and submits the necessary
        jobs to the remote server. The runtime manager process should call this method repeatedly (cyclically) to run
        the pipeline until the 'is_running' property of the instance returns True.

        Notes:
            While the 'is_running' property can be used to determine whether the pipeline is still running, to resolve
            the final status of the pipeline (success or failure), the manager process should access the
            'status' instance property.
        """
        # This clause is executed the first time the method is called for the newly initialized pipeline tracker
        # instance. It submits the first batch of processing jobs (first stage) to the remote server. For one-stage
        # pipelines, this is the only time when pipeline jobs are submitted to the server.
        if self._pipeline_stage == 0:
            self._pipeline_stage += 1
            self._submit_jobs()

        # Waits until all jobs submitted to the server as part of the current processing stage are completed before
        # advancing further.
        for job, _ in self.jobs[self._pipeline_stage]:  # Ignores working directories as part of this iteration.
            if not self.server.job_complete(job=job):
                return

        # If all jobs for the current processing stage have completed, checks the pipeline's processing tracker file to
        # determine if all jobs completed successfully.
        self.server.pull_file(remote_file_path=self.remote_tracker_path, local_file_path=self.local_tracker_path)
        tracker = ProcessingTracker(self.local_tracker_path)

        # If the stage failed due to encountering an error, removes the local tracker copy and marks the pipeline
        # as 'failed'. It is expected that the pipeline state is then handed by the manager process to notify the
        # user about the runtime failure.
        if tracker.encountered_error:
            sh.rmtree(self.local_tracker_path.parent)  # Removes local temporary data
            self.pipeline_status = ProcessingStatus.FAILED  # Updates the processing status to 'failed'

        # If this was the last processing stage, the tracker indicates that the processing has been completed. In this
        # case, initializes the shutdown sequence:
        elif tracker.is_complete:
            sh.rmtree(self.local_tracker_path.parent)  # Removes local temporary data
            self.pipeline_status = ProcessingStatus.SUCCEEDED  # Updates the job status to 'succeeded'

            # If the pipeline was configured to remove logs after completing successfully, removes the runtime log for
            # each job submitted as part of this pipeline from the remote server.
            if not self.keep_job_logs:
                for stage_jobs in self.jobs.values():
                    for _, directory in stage_jobs:  # Ignores job objects as part of this iteration.
                        self.server.remove(remote_path=directory, recursive=True, is_dir=True)

        # If the processing is not complete (according to the tracker), this indicates that the pipeline has more
        # stages to execute. In this case, increments the processing stage tracker and submits the next batch of jobs
        # to the server.
        elif tracker.is_running:
            self._pipeline_stage += 1

            # If the incremented stage is not a valid stage, the pipeline has actually been aborted and the tracker file
            # does not properly reflect this state. Sets the internal state tracker appropriately and resets (removes)
            # the tracker file from the server to prevent deadlocking further runtimes
            if self._pipeline_stage not in self.jobs.keys():
                sh.rmtree(self.local_tracker_path.parent)  # Removes local temporary data
                self.pipeline_status = ProcessingStatus.ABORTED
                self.server.remove(remote_path=self.remote_tracker_path, is_dir=False)
            else:
                # Otherwise, submits the next batch of jobs to the server.
                self._submit_jobs()

        # The final and the rarest state: the pipeline was aborted before it finished the runtime. Generally, this state
        # should not be encountered during most runtimes.
        else:
            sh.rmtree(self.local_tracker_path.parent)  # Removes local temporary data
            self.pipeline_status = ProcessingStatus.ABORTED

    def _submit_jobs(self) -> None:
        """This worker method submits the processing jobs for the currently active processing stage to the remote
        server.

        It is used internally by the runtime_cycle() method to iteratively execute all stages of the managed processing
        pipeline on the remote server.
        """
        for job, _ in self.jobs[self._pipeline_stage]:
            self.server.submit_job(job=job, verbose=False)  # Silences terminal printouts

    @property
    def is_running(self) -> bool:
        """Returns True if the pipeline is currently running, False otherwise."""
        if self.pipeline_status == ProcessingStatus.RUNNING:
            return True
        return False

    @property
    def status(self) -> ProcessingStatus:
        """Returns the current status of the pipeline packaged into a ProcessingStatus instance."""
        return ProcessingStatus(self.pipeline_status)


def generate_manager_id() -> int:
    """Generates and returns a unique integer value that can be used to identify the manager process that calls
    this function.

    The identifier is generated based on the current timestamp, accurate to microseconds, and a random number between 1
    and 9999999999999. This ensures that the identifier is unique for each function call. The generated identifier
    string is converted to a unique integer value using the xxHash-64 algorithm before it is returned to the caller.

    Notes:
        This function should be used to generate manager process identifiers for working with ProcessingTracker
        instances from sl-shared-assets version 4.0.0 and above.
    """
    timestamp = get_timestamp()
    random_number = randint(1, 9999999999999)
    manager_id = f"{timestamp}_{random_number}"
    id_hash = xxh3_64()
    id_hash.update(manager_id)
    return id_hash.intdigest()
