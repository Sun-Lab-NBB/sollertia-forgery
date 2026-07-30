"""Provides API for submitting jobs to SLURM-managed servers, monitoring job status, and managing remote data."""

from __future__ import annotations

from enum import StrEnum
import stat
import shlex
from typing import TYPE_CHECKING, Self
from pathlib import Path
import tempfile
from dataclasses import dataclass

import paramiko
from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import LogLevel, console

if TYPE_CHECKING:
    from types import TracebackType
    from collections.abc import Sequence

    from paramiko.client import SSHClient
    from paramiko.sftp_client import SFTPClient

    from .job import Job
    from .server_configuration import ServerConfiguration

_BLOCKED_QUEUE_REASON: str = "DependencyNeverSatisfied"
"""The reason SLURM's queue reports for a pending job whose dependency can no longer be satisfied."""


@dataclass(frozen=True)
class CommandResult:
    """Stores the result of executing a command on the remote server.

    Attributes:
        stdout: The standard output from the command.
        stderr: The standard error output from the command.
        return_code: The exit code of the command (0 indicates success).
    """

    stdout: str
    stderr: str
    return_code: int


class JobStatus(StrEnum):
    """Defines the set of status codes returned by SLURM for managed jobs."""

    PENDING = "PENDING"
    """The job is queued and waiting for resources."""
    RUNNING = "RUNNING"
    """The job is currently executing."""
    COMPLETED = "COMPLETED"
    """The job finished successfully."""
    FAILED = "FAILED"
    """The job terminated with a non-zero exit code."""
    CANCELLED = "CANCELLED"
    """The job was cancelled by the user or administrator."""
    TIMEOUT = "TIMEOUT"
    """The job exceeded its time limit."""
    NODE_FAIL = "NODE_FAIL"
    """The job terminated due to node failure."""
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    """The job was terminated for exceeding memory limits."""
    BLOCKED = "BLOCKED"
    """The job is queued behind a dependency that can no longer be satisfied, so it will never run. Resolved from the
    queue's reason field rather than from accounting, which still reports such a job as pending."""
    UNKNOWN = "UNKNOWN"
    """The job status could not be determined."""


TERMINAL_JOB_STATUSES: frozenset[JobStatus] = frozenset(
    {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.TIMEOUT,
        JobStatus.NODE_FAIL,
        JobStatus.OUT_OF_MEMORY,
        JobStatus.BLOCKED,
    }
)
"""The statuses a job never leaves, which is what tells a caller a polled submission has settled.

Notes:
    ``UNKNOWN`` is absent, since accounting reports it for a submission it has not yet registered as well as for one
    it can no longer resolve.
"""


def _parse_job_status(state: str) -> JobStatus:
    """Resolves one accounting state string into a JobStatus member.

    Notes:
        SLURM decorates some states with a trailing marker or an attribution clause, reporting a cancelled job as
        'CANCELLED by 1234' and a truncated state as 'CANCELLED+'. Both name the same state, so the decoration is
        stripped before the state is matched.

    Args:
        state: The state string accounting reported for the allocation.

    Returns:
        The matching status, or ``UNKNOWN`` when the state names something this enumeration does not cover.
    """
    try:
        return JobStatus(state)
    except ValueError:
        undecorated = state.split(" ", maxsplit=1)[0]
        cleaned = "".join(character for character in undecorated if character.isalpha() or character == "_")
        try:
            return JobStatus(cleaned)
        except ValueError:
            return JobStatus.UNKNOWN


class Server:
    """Establishes and maintains a bidirectional interface that allows working with a remote compute server.

    This class provides the central API that allows submitting SLURM-managed jobs to the server and monitoring their
    execution status. Additionally, it also provides the API for managing the data stored on the remote compute server
    via the SFTP protocol.

    Notes:
        This class assumes that the target server has the SLURM job manager installed and accessible to the user whose
        credentials are used to connect to the server as part of class initialization.

    Args:
        configuration: The ServerConfiguration instance that contains the server hostname and access credentials.

    Attributes:
        _open: Tracks whether the connection to the server is open.
        _client: Stores the SSHClient instance used to interface with the server.
        _sftp: Stores the SFTPClient instance used for file transfer operations.
        _configuration: Stores the ServerConfiguration instance used to configure the server connection.
    """

    def __init__(self, configuration: ServerConfiguration) -> None:
        # Tracker used to prevent __del__ from calling close() for a partially initialized class.
        self._open: bool = False

        # Stores the server configuration
        self._configuration: ServerConfiguration = configuration

        # Initializes a timer class to optionally delay loop cycling below
        timer = PrecisionTimer(precision=TimerPrecisions.SECOND)

        # Establishes the SSH connection to the specified processing server. At most, attempts to connect to the server
        # 30 times before terminating with an error
        attempt = 0
        _maximum_connection_attempts = 30
        while True:
            console.echo(
                message=f"Connecting to {self._configuration.host} (attempt {attempt}/30)...", level=LogLevel.INFO
            )
            try:
                self._client: SSHClient = paramiko.SSHClient()
                self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                self._client.connect(
                    hostname=self._configuration.host,
                    username=self._configuration.username,
                    password=self._configuration.password,
                )
                console.echo(message=f"Connected to {self._configuration.host}", level=LogLevel.SUCCESS)

                # Initializes the SFTP client using the established SSH connection. This client is reused for all
                # file transfer operations during the lifetime of the Server instance.
                self._sftp: SFTPClient = self._client.open_sftp()

                self._open = True
                break
            except paramiko.AuthenticationException:
                message = (
                    f"Authentication failed when connecting to {self._configuration.host} using "
                    f"{self._configuration.username} user."
                )
                console.error(message, PermissionError)
            except Exception:
                if attempt == _maximum_connection_attempts:
                    message = f"Could not connect to {self._configuration.host} after 30 attempts. Aborting runtime."
                    console.error(message, ConnectionError)

                console.echo(
                    message=f"Could not SSH into {self._configuration.host}, retrying after a 2-second delay...",
                    level=LogLevel.WARNING,
                )
                attempt += 1
                timer.delay(delay=2, allow_sleep=True, block=False)

    def __del__(self) -> None:
        """If the instance is connected to the server, terminates the connection before the instance is destroyed."""
        self.close()

    def __enter__(self) -> Self:
        """Returns the connected instance so it can be used as a context manager."""
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Closes the connection when the context manager exits, however the block ended."""
        self.close()

    def submit_job(self, job: Job, *, verbose: bool = True) -> Job:
        """Submits the input job to the managed remote compute server via the SLURM job manager.

        This method is the entry point for all headless jobs that are executed on the remote compute server.

        Args:
            job: The Job instance that defines the job to be executed.
            verbose: Determines whether to notify the user about non-error states of the submission process.

        Returns:
            The job object whose 'job_id' attribute had been replaced with the SLURM-assigned job ID.

        Raises:
            RuntimeError: If the job cannot be submitted to the server for any reason.
        """
        if verbose:
            console.echo(message=f"Submitting '{job.job_name}' job to the remote server {self.host}...")

        # If the Job object already has a job ID, this indicates that the job has already been submitted to the server.
        # In this case returns it to the caller with no further modifications.
        if job.job_id is not None:
            console.echo(
                message=f"The '{job.job_name}' job has already been submitted to the server.",
                level=LogLevel.WARNING,
            )
            return job

        # Generates a temporary shell script on the local machine. Uses tempfile to automatically remove the
        # local script as soon as it is uploaded to the server.
        with tempfile.TemporaryDirectory() as temp_dir:
            local_script_path = Path(temp_dir).joinpath(f"{job.job_name}.sh")
            fixed_script_content = job.command_script

            # Creates a temporary script file locally and dumps translated command data into the file
            with local_script_path.open("w") as f:
                f.write(fixed_script_content)

            # Uploads the command script to the server using the persistent SFTP client
            self._sftp.put(localpath=str(local_script_path), remotepath=job.remote_script_path)

        # Makes the server-side script executable. The exit status is awaited, because a submission issued on a second
        # channel would otherwise race the permission change on the first.
        script_path = shlex.quote(job.remote_script_path)
        chmod_result = self.execute_command(command=f"chmod +x {script_path}")
        if chmod_result.return_code != 0:
            message = (
                f"Failed to make the '{job.job_name}' job script executable on the remote compute server. "
                f"{chmod_result.stderr.strip()}"
            )
            console.error(message, RuntimeError)

        # Submits the job to SLURM with sbatch and verifies submission state
        submission = self.execute_command(command=f"sbatch {script_path}")
        job_output = submission.stdout.strip()

        # If batch_job is not in the output received from SLURM in response to issuing the submission command, raises an
        # error.
        if "Submitted batch job" not in job_output:
            message = (
                f"Failed to submit the '{job.job_name}' job to the remote compute server. "
                f"{submission.stderr.strip() or job_output}"
            )
            console.error(message, RuntimeError)
            raise RuntimeError(message)  # pragma: no cover - console.error() is NoReturn but ruff cannot infer this

        # Otherwise, extracts the job id assigned to the job by SLURM from the response and writes it to the processed
        # Job object
        job_id = job_output.split()[-1]
        job.job_id = job_id

        if verbose:
            console.echo(message=f"{job.job_name} job: Submitted to {self.host}.", level=LogLevel.SUCCESS)

        # Returns the updated job object
        return job

    def abort_job(self, slurm_job_id: int) -> None:
        """Aborts the job with the specified SLURM-assigned ID if it is currently running or pending on the server.

        Args:
            slurm_job_id: The SLURM-assigned job ID to abort.
        """
        if self.get_job_status(slurm_job_id=slurm_job_id) in (JobStatus.PENDING, JobStatus.RUNNING):
            self.execute_command(command=f"scancel {slurm_job_id}")

    def abort_jobs(self, slurm_job_ids: Sequence[str]) -> None:
        """Aborts every named allocation that is still queued or running on the server.

        Args:
            slurm_job_ids: The SLURM-assigned job IDs to abort.
        """
        if not slurm_job_ids:
            return
        self.execute_command(command=f"scancel {' '.join(shlex.quote(str(job)) for job in slurm_job_ids)}")

    def get_job_status(self, slurm_job_id: int) -> JobStatus:
        """Queries the managed server's SLURM manager for the runtime status of the job with the specified
        SLURM-assigned ID.

        Notes:
            This method uses the 'sacct' command to determine the current state of the job, returning the actual status
            (e.g., PENDING, RUNNING, COMPLETED, FAILED) assigned by the SLURM manager.

        Args:
            slurm_job_id: The SLURM-assigned job ID for which to query the runtime status.

        Returns:
            The current status of the job as a JobStatus enumeration value.
        """
        return self.get_job_statuses(slurm_job_ids=(str(slurm_job_id),))[str(slurm_job_id)]

    def get_job_statuses(self, slurm_job_ids: Sequence[str]) -> dict[str, JobStatus]:
        """Queries the runtime status of every named allocation in one accounting call.

        Notes:
            Accounting reports each allocation alongside its steps, and only the allocation rows are read.

            A pending allocation whose dependency can no longer be satisfied is reported as blocked. Accounting still
            calls that job pending, so the queue's reason field is the only source of that distinction.

        Args:
            slurm_job_ids: The SLURM-assigned job IDs to query.

        Returns:
            A dictionary mapping every requested job ID to its status. An allocation accounting does not know reports
            as ``UNKNOWN``.
        """
        requested = [str(job_id) for job_id in slurm_job_ids]
        if not requested:
            return {}

        statuses: dict[str, JobStatus] = dict.fromkeys(requested, JobStatus.UNKNOWN)
        result = self.execute_command(
            command=f"sacct -j {','.join(requested)} --format=JobID,State --noheader --parsable2"
        )
        for line in result.stdout.splitlines():
            fields = line.split("|")
            expected_fields = 2
            if len(fields) < expected_fields:
                continue
            job_id = fields[0].strip()
            # Step rows carry a suffixed identifier ('12345.batch'), and describe part of the allocation rather than
            # the allocation itself.
            if "." in job_id or job_id not in statuses:
                continue
            statuses[job_id] = _parse_job_status(state=fields[1].strip())

        pending = [job_id for job_id, status in statuses.items() if status is JobStatus.PENDING]
        if pending:
            for job_id in self.get_blocked_job_ids():
                if job_id in statuses:
                    statuses[job_id] = JobStatus.BLOCKED

        return statuses

    def get_blocked_job_ids(self) -> set[str]:
        """Returns the identifiers of this user's queued allocations whose dependencies can no longer be satisfied.

        Notes:
            Queries the user's whole queue, since naming an allocation the queue no longer holds makes the command
            report an error for it.

        Returns:
            The SLURM-assigned job IDs the queue reports as permanently blocked.
        """
        result = self.execute_command(command=f'squeue -h -u {shlex.quote(self.user)} -o "%i|%r"')
        blocked: set[str] = set()
        for line in result.stdout.splitlines():
            fields = line.split("|")
            expected_fields = 2
            if len(fields) < expected_fields:
                continue
            if fields[1].strip() == _BLOCKED_QUEUE_REASON:
                blocked.add(fields[0].strip())
        return blocked

    def pull(self, local_path: Path, remote_path: Path) -> None:
        """Downloads a file or directory from the remote server to the local machine.

        This method automatically detects whether the remote path points to a file or directory and handles the
        transfer accordingly. For directories, all contents are recursively downloaded.

        Args:
            local_path: The path on the local machine where the file or directory will be saved.
            remote_path: The path to the file or directory on the remote server to download.

        Raises:
            FileNotFoundError: If the remote path does not exist on the server.
        """
        # Checks if the remote path exists and determines if it is a file or directory
        try:
            remote_stat = self._sftp.stat(str(remote_path))
        except FileNotFoundError:
            message = f"The remote path {remote_path} does not exist on the server."
            console.error(message, FileNotFoundError)
            raise FileNotFoundError(message) from None

        # Determines if the remote path is a directory or file and handles accordingly
        if stat.S_ISDIR(remote_stat.st_mode):
            self._pull_directory(local_path, remote_path)
        else:
            # Ensures the parent directory exists locally
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self._sftp.get(localpath=str(local_path), remotepath=str(remote_path))

    def _pull_directory(self, local_path: Path, remote_path: Path) -> None:
        """Recursively downloads a directory from the remote server.

        This is an internal helper method used by pull() to handle directory transfers.

        Args:
            local_path: The local directory path where contents will be saved.
            remote_path: The remote directory path to download.
        """
        # Creates the local directory if it doesn't exist
        local_path.mkdir(parents=True, exist_ok=True)

        # Gets the list of items in the remote directory
        remote_items = self._sftp.listdir_attr(str(remote_path))

        for item in remote_items:
            remote_item_path = remote_path / item.filename
            local_item_path = local_path / item.filename

            # Checks if the item is a directory
            if stat.S_ISDIR(item.st_mode):
                # Recursively pulls the subdirectory
                self._pull_directory(local_item_path, remote_item_path)
            else:
                # Downloads the individual file
                self._sftp.get(localpath=str(local_item_path), remotepath=str(remote_item_path))

    def push(self, local_path: Path, remote_path: Path) -> None:
        """Uploads a file or directory from the local machine to the remote server.

        This method automatically detects whether the local path points to a file or directory and handles the
        transfer accordingly. For directories, all contents are recursively uploaded.

        Args:
            local_path: The path to the file or directory on the local machine to upload.
            remote_path: The path on the remote server where the file or directory will be saved.

        Raises:
            FileNotFoundError: If the local path does not exist.
        """
        if not local_path.exists():
            message = f"The local path {local_path} does not exist."
            console.error(message, FileNotFoundError)
            raise FileNotFoundError(message)

        if local_path.is_dir():
            self._push_directory(local_path, remote_path)
        else:
            # Ensures the parent directory exists on the remote server
            self._create_directory(remote_path.parent, parents=True)
            self._sftp.put(localpath=str(local_path), remotepath=str(remote_path))

    def _push_directory(self, local_path: Path, remote_path: Path) -> None:
        """Recursively uploads a directory to the remote server.

        This is an internal helper method used by push() to handle directory transfers.

        Args:
            local_path: The local directory path to upload.
            remote_path: The remote directory path where contents will be saved.
        """
        # Creates the remote directory
        self._create_directory(remote_path, parents=True)

        # Iterates through all items in the local directory
        for local_item_path in local_path.iterdir():
            remote_item_path = remote_path / local_item_path.name

            if local_item_path.is_dir():
                # Recursively pushes subdirectory
                self._push_directory(local_item_path, remote_item_path)
            else:
                # Uploads the individual file
                self._sftp.put(localpath=str(local_item_path), remotepath=str(remote_item_path))

    def create(self, remote_path: Path, *, is_dir: bool = True, parents: bool = True) -> None:
        """Creates a file or directory on the remote server.

        Args:
            remote_path: The absolute path to the file or directory to create on the remote server.
            is_dir: If True, creates a directory. If False, creates an empty file.
            parents: If True and is_dir is True, creates parent directories if they are missing. If False and parents
                do not exist, raises a FileNotFoundError. This parameter is ignored when creating files (parents are
                always created for files).

        Notes:
            This method silently succeeds if the target already exists.
        """
        if is_dir:
            self._create_directory(remote_path, parents=parents)
        else:
            # For files, always ensure parent directories exist
            self._create_directory(remote_path.parent, parents=True)

            # Creates an empty file if it doesn't exist
            if not self.exists(remote_path):
                # Opens the file in 'write' mode and immediately closes it to create an empty file
                with self._sftp.open(str(remote_path), "w"):
                    pass

    def _create_directory(self, remote_path: Path, *, parents: bool = True) -> None:
        """Creates a directory on the remote server.

        This is an internal helper method used by create() and other methods that need to create directories.

        Notes:
            Creating a nested path is delegated to the shell, which resolves the whole chain in one round trip. Walking
            the chain over the file-transfer protocol instead costs one query per level, which a batch creating a
            directory per job pays many times over.

        Args:
            remote_path: The absolute path to the directory to create on the remote server.
            parents: If True, creates parent directories if they are missing.
        """
        remote_path_str = str(remote_path)

        if parents:
            result = self.execute_command(command=f"mkdir -p {shlex.quote(remote_path_str)}")
            if result.return_code != 0:
                message = (
                    f"Unable to create the directory {remote_path_str} on the remote compute server. "
                    f"{result.stderr.strip()}"
                )
                console.error(message, RuntimeError)
        else:
            # Only creates the final directory
            try:
                # Checks if the directory already exists
                self._sftp.stat(remote_path_str)
            except FileNotFoundError:
                # Creates the directory if it does not exist
                self._sftp.mkdir(remote_path_str)

    def remove(self, remote_path: Path, *, is_dir: bool, recursive: bool = False) -> None:
        """Removes a file or directory from the remote server.

        Args:
            remote_path: The path to the file or directory on the remote server to be removed.
            is_dir: Determines whether the input path represents a directory or a file.
            recursive: If True and is_dir is True, recursively deletes all contents of the directory
                before removing it. If False, only removes empty directories (standard rmdir behavior).
        """
        if is_dir:
            if recursive:
                # Recursively deletes all contents first and then removes the top-level (now empty) directory
                self._recursive_remove(remote_path)
            else:
                # Only removes empty directories
                self._sftp.rmdir(path=str(remote_path))
        else:
            self._sftp.unlink(path=str(remote_path))

    def _recursive_remove(self, remote_path: Path) -> None:
        """Recursively removes a directory and all its contents from the remote server.

        This is an internal helper method used by remove() to handle recursive directory deletion.

        Args:
            remote_path: The path to the remote directory to recursively remove.
        """
        try:
            # Lists all items in the directory
            items = self._sftp.listdir_attr(str(remote_path))

            for item in items:
                item_path = remote_path / item.filename

                # Checks if the item is a directory
                if stat.S_ISDIR(item.st_mode):
                    # Recursively removes subdirectories
                    self._recursive_remove(item_path)
                else:
                    # Removes files
                    self._sftp.unlink(str(item_path))

            # After all contents are removed, removes the empty directory
            self._sftp.rmdir(str(remote_path))

        except Exception as e:
            console.echo(
                message=f"Unable to remove the specified directory {remote_path}: {e!s}", level=LogLevel.WARNING
            )

    def exists(self, remote_path: Path) -> bool:
        """Returns True if the target file or directory exists on the remote server.

        Args:
            remote_path: The path to check on the remote server.

        Returns:
            True if the path exists, False otherwise.
        """
        try:
            self._sftp.stat(str(remote_path))
        except FileNotFoundError:
            return False
        else:
            return True

    def is_directory(self, remote_path: Path) -> bool:
        """Returns True if the target path is a directory on the remote server.

        Args:
            remote_path: The path to check on the remote server.

        Returns:
            True if the path exists and is a directory, False otherwise.
        """
        try:
            file_stat = self._sftp.stat(str(remote_path))
            return stat.S_ISDIR(file_stat.st_mode)
        except FileNotFoundError:
            return False

    def list_directory(self, remote_path: Path) -> list[str]:
        """Lists the contents of a directory on the remote server.

        Args:
            remote_path: The path to the directory on the remote server.

        Returns:
            A list of filenames (not full paths) in the directory.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        return self._sftp.listdir(str(remote_path))

    def execute_command(self, command: str) -> CommandResult:
        """Executes the specified command on the remote server and returns the result.

        Args:
            command: The shell command to execute on the remote server.

        Returns:
            A CommandResult instance containing stdout, stderr, and the return code of the executed command.
        """
        _, stdout, stderr = self._client.exec_command(command)
        return CommandResult(
            stdout=stdout.read().decode(),
            stderr=stderr.read().decode(),
            return_code=stdout.channel.recv_exit_status(),
        )

    def close(self) -> None:
        """Closes the SFTP and SSH connections to the server."""
        # Prevents closing already closed connections
        if self._open:
            self._sftp.close()
            self._client.close()
            self._open = False

    @property
    def root(self) -> Path:
        """Returns the absolute path to the single root directory of the remote compute server that stores all
        Sollertia data (raw and processed) accessible through this instance.
        """
        return Path(self._configuration.root)

    @property
    def host(self) -> str:
        """Returns the hostname or IP address of the server accessible through this class."""
        return self._configuration.host

    @property
    def user(self) -> str:
        """Returns the username used to authenticate with the server."""
        return self._configuration.username

    @property
    def environment(self) -> str:
        """Returns the name of the shared conda environment, on the server, in which the ``slf`` CLI and all of its
        processing dependencies are installed. Every remote compute job activates this environment before invoking the
        ``slf`` CLI.
        """
        return self._configuration.environment

    @property
    def cindra_configurations_directory(self) -> Path:
        """Returns the absolute path to the cindra configuration directory under the server's data root."""
        return self.root.joinpath("cindra_configurations")

    @property
    def dlc_projects_directory(self) -> Path:
        """Returns the absolute path to the DeepLabCut project directory under the server's data root."""
        return self.root.joinpath("deeplabcut_projects")
