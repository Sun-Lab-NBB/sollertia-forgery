"""Provides API for submitting jobs to SLURM-managed servers, monitoring job status, and managing remote data."""

from __future__ import annotations

from enum import StrEnum
import stat
from typing import TYPE_CHECKING
from pathlib import Path
import tempfile
from dataclasses import dataclass

import paramiko
from ataraxis_time import PrecisionTimer, TimerPrecisions, TimestampFormats, get_timestamp
from ataraxis_base_utilities import LogLevel, console

if TYPE_CHECKING:
    from paramiko.client import SSHClient
    from paramiko.sftp_client import SFTPClient

    from .job import Job
    from .server_configuration import ServerConfiguration


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


def get_remote_job_work_directory(server: Server, job_name: str, pipeline_name: str, *, base_path: Path) -> Path:
    """Resolves and creates the remote compute server log directory for the specified job.

    Args:
        server: The Server instance that interfaces with the remote compute server used to execute the job.
        job_name: The name of the job to be executed.
        pipeline_name: The name of the pipeline to which this job belongs.
        base_path: The data directory under which to nest the job's log directory. Each pipeline passes the
            processed session, dataset, or project path so that job logs are always stored under the data
            they operate on, scoped to a 'logs' subdirectory of that directory.

    Returns:
        The path to the job's log directory on the remote compute server.
    """
    # Resolves the log directory name using a timestamp (accurate to minutes) and the job's name. Job logs are
    # nested under a 'logs' subdirectory of the data directory the job operates on.
    timestamp = "-".join(get_timestamp(output_format=TimestampFormats.STRING).split("-")[:5])
    working_directory = base_path.joinpath("logs", f"{pipeline_name}", f"{job_name}", f"{timestamp}")

    # Creates the log directory on the remote server.
    server.create(remote_path=working_directory, is_dir=True, parents=True)

    return working_directory


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
    UNKNOWN = "UNKNOWN"
    """The job status could not be determined."""


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

        # Makes the server-side script executable
        self._client.exec_command(f"chmod +x {job.remote_script_path}")

        # Submits the job to SLURM with sbatch and verifies submission state
        job_output = self._client.exec_command(f"sbatch {job.remote_script_path}")[1].read().strip().decode()

        # If batch_job is not in the output received from SLURM in response to issuing the submission command, raises an
        # error.
        if "Submitted batch job" not in job_output:
            message = f"Failed to submit the '{job.job_name}' job to the remote compute server."
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
            self._client.exec_command(f"scancel {slurm_job_id}")

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
        # Uses the 'sacct' command with a specific format to get the job's state. The '--parsable2' flag provides clean
        # output. Queries both the main job and any job steps (.batch, .extern), taking the primary job status.
        result = (
            self._client.exec_command(f"sacct -j {slurm_job_id} --format=State --noheader --parsable2")[1]
            .read()
            .decode()
            .strip()
        )

        # The output may contain multiple lines (for job steps). The first line contains the main job status.
        if result:
            statuses = result.split("\n")
            if statuses:
                status_str = statuses[0].strip()
                # Attempts to match the status string to a JobStatus enum value
                try:
                    return JobStatus(status_str)
                except ValueError:
                    # SLURM may return statuses with suffixes (e.g., "CANCELLED+"). Strips non-alpha characters
                    # and retries.
                    cleaned = "".join(c for c in status_str if c.isalpha() or c == "_")
                    try:
                        return JobStatus(cleaned)
                    except ValueError:
                        return JobStatus.UNKNOWN

        return JobStatus.UNKNOWN

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

        Args:
            remote_path: The absolute path to the directory to create on the remote server.
            parents: If True, creates parent directories if they are missing.
        """
        remote_path_str = str(remote_path)

        if parents:
            # Creates parent directories if needed by splitting the path into parts and creating each level
            path_parts = Path(remote_path_str).parts
            current_path = ""

            for part in path_parts:
                # Skips empty path parts
                if not part:
                    continue

                # Builds the full path by concatenating the current path and the part
                current_path = str(Path(current_path) / part) if current_path else part

                try:
                    # Checks if the directory exists by trying to 'stat' it
                    self._sftp.stat(current_path)
                except FileNotFoundError:
                    # If the directory does not exist, creates it
                    self._sftp.mkdir(current_path)
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
    def cindra_configurations_directory(self) -> Path:
        """Returns the absolute path to the cindra configuration directory under the server's data root."""
        return self.root.joinpath("cindra_configurations")

    @property
    def dlc_projects_directory(self) -> Path:
        """Returns the absolute path to the DeepLabCut project directory under the server's data root."""
        return self.root.joinpath("deeplabcut_projects")
