"""This module provides the API for submitting jobs to the SLURM-managed compute servers and monitoring their
runtime status, and managing the data stored on the remote compute servers.
"""

from enum import StrEnum
import stat
import select
import socket
from typing import TYPE_CHECKING
from pathlib import Path
from secrets import randbelow
import tempfile
import threading
import contextlib
from dataclasses import dataclass

import paramiko
from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import LogLevel, console
from ataraxis_time.time_helpers import TimestampFormats, get_timestamp

from .job import Job, JupyterJob

if TYPE_CHECKING:
    from paramiko.client import SSHClient
    from sl_shared_assets import ServerConfiguration
    from paramiko.sftp_client import SFTPClient


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


def get_remote_job_work_directory(server: Server, job_name: str, pipeline_name: str) -> Path:
    """Resolves and creates the remote compute server working directory for the specified job.

    Args:
        server: The Server instance that interfaces with the remote compute server used to execute the job.
        job_name: The name of the job to be executed.
        pipeline_name: The name of the pipeline to which this job belongs.

    Returns:
        The path to the job's working directory on the remote compute server.
    """
    # Resolves working directory name using timestamp (accurate to minutes) and the job's name.
    timestamp = "-".join(get_timestamp(output_format=TimestampFormats.STRING).split("-")[:5])
    working_directory = Path(server.user_working_root).joinpath(
        "job_logs", f"{pipeline_name}", f"{job_name}", f"{timestamp}"
    )

    # Creates the working directory on the remote server.
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


class _SSHTunnel:
    """Manages an SSH tunnel for local port forwarding using paramiko's direct-tcpip channel.

    This class creates a local socket server that listens for incoming connections and forwards them through an
    SSH channel to a remote destination. It is used to enable localhost access to services running on remote
    compute nodes.

    Args:
        ssh_client: The paramiko SSHClient instance to use for creating the tunnel.
        local_port: The local port to listen on for incoming connections.
        remote_host: The hostname of the remote destination (e.g., compute node).
        remote_port: The port on the remote destination to forward traffic to.

    Attributes:
        _ssh_client: The SSH client used for the tunnel.
        _local_port: The local listening port.
        _remote_host: The remote destination hostname.
        _remote_port: The remote destination port.
        _server_socket: The local socket server accepting connections.
        _running: Flag indicating whether the tunnel is active.
        _accept_thread: The thread running the connection accept loop.
    """

    def __init__(
        self,
        ssh_client: SSHClient,
        local_port: int,
        remote_host: str,
        remote_port: int,
    ) -> None:
        self._ssh_client = ssh_client
        self._local_port = local_port
        self._remote_host = remote_host
        self._remote_port = remote_port
        self._server_socket: socket.socket | None = None
        self._running = False
        self._accept_thread: threading.Thread | None = None

    def __del__(self) -> None:
        """Ensures graceful resource deallocation when the tunnel instance is garbage collected."""
        self.stop()

    def start(self) -> None:
        """Starts the SSH tunnel by creating a local socket server and spawning the 'accept' loop thread."""
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.settimeout(1.0)  # Allows periodic checking of _running flag
        self._server_socket.bind(("127.0.0.1", self._local_port))
        self._server_socket.listen(5)
        self._running = True

        # Starts the 'accept' loop in a daemon thread
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def stop(self) -> None:
        """Stops the SSH tunnel and closes all connections."""
        self._running = False
        if self._server_socket:
            with contextlib.suppress(OSError):
                self._server_socket.close()

    def _accept_loop(self) -> None:
        """Accepts incoming connections on the local socket and spawns forwarding threads for each connection."""
        while self._running:
            try:
                client_socket, addr = self._server_socket.accept()

                # Opens a direct-tcpip channel to the remote destination
                transport = self._ssh_client.get_transport()
                if transport is None:
                    client_socket.close()
                    continue

                try:
                    channel = transport.open_channel("direct-tcpip", (self._remote_host, self._remote_port), addr)
                except Exception:
                    client_socket.close()
                    continue

                # Starts bidirectional forwarding in a separate thread
                forward_thread = threading.Thread(
                    target=self._forward_tunnel, args=(client_socket, channel), daemon=True
                )
                forward_thread.start()

            except TimeoutError:
                # Timeout allows periodic checking of the _running flag
                continue
            except OSError:
                # Socket was closed
                if self._running:
                    continue
                break

    def _forward_tunnel(self, client_socket: socket.socket, channel: paramiko.Channel) -> None:
        """Bidirectionally forwards the data between the local client socket and the SSH channel.

        Args:
            client_socket: The local socket connected to the client application.
            channel: The paramiko channel connected to the remote destination.
        """
        try:
            while self._running:
                # Uses select to wait for data on either the socket or the channel
                r, _, _ = select.select([client_socket, channel], [], [], 0.5)

                if client_socket in r:
                    data = client_socket.recv(4096)
                    if len(data) == 0:
                        break
                    channel.send(data)

                if channel in r:
                    data = channel.recv(4096)
                    if len(data) == 0:
                        break
                    client_socket.send(data)

        except OSError:
            pass
        finally:
            with contextlib.suppress(OSError):
                channel.close()
            with contextlib.suppress(OSError):
                client_socket.close()


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
                raise PermissionError(message) from None  # Fallback to appease mypy, should not be reachable
            except Exception:
                if attempt == _maximum_connection_attempts:
                    message = f"Could not connect to {self._configuration.host} after 30 attempts. Aborting runtime."
                    console.error(message, ConnectionError)
                    raise ConnectionError(message) from None  # Fallback to appease mypy, should not be reachable

                console.echo(
                    message=f"Could not SSH into {self._configuration.host}, retrying after a 2-second delay...",
                    level=LogLevel.WARNING,
                )
                attempt += 1
                timer.delay(delay=2, allow_sleep=True, block=False)

    def __del__(self) -> None:
        """If the instance is connected to the server, terminates the connection before the instance is destroyed."""
        self.close()

    def launch_jupyter_server(
        self,
        job_name: str,
        conda_environment: str,
        notebook_directory: Path,
        cpu_threads: int = 2,
        ram: int = 32,
        time: int = 240,
        port: int = 0,
        jupyter_arguments: str = "",
    ) -> JupyterJob:
        """Launches a remote Jupyter notebook session on the target remote compute server.

        Args:
            job_name: The descriptive name of the Jupyter SLURM job to be created.
            conda_environment: The name of the conda environment to activate on the server before running the job logic.
                For Jupyter jobs, the environment must include the 'notebook' and 'jupyterlab' packages.
            port: The connection port number for the Jupyter server. If set to 0 (default), a random port number between
                8888 and 9999 is assigned to this connection to reduce the possibility of colliding with other
                user sessions.
            notebook_directory: The root directory where to run the Jupyter notebook. During runtime, the notebook
                only has access to items stored under this directory.
            cpu_threads: The number of CPU threads to allocate to the Jupyter server.
            ram: The amount of RAM, in GB, to allocate to the Jupyter server.
            time: The maximum Jupyter server uptime, in minutes.
            jupyter_arguments: The additional arguments to pass to the jupyter notebook initialization command.

        Returns:
            The JupyterJob instance containing information about the completed session.

        Raises:
            TimeoutError: If the Jupyter server doesn't start within 120 seconds of being submitted.
            RuntimeError: If the job submission fails for any reason.
        """
        # Resolves the job's working directory
        working_directory = get_remote_job_work_directory(server=self, job_name=job_name, pipeline_name="JUPYTER")

        # If necessary, generates and sets port to a random value between 8888 and 9999.
        if port == 0:
            port = 8888 + randbelow(1112)  # Range: 8888-9999

        job = JupyterJob(
            job_name=job_name,
            output_log=working_directory.joinpath("stdout.txt"),
            error_log=working_directory.joinpath("stderr.txt"),
            working_directory=working_directory,
            conda_environment=conda_environment,
            notebook_directory=notebook_directory,
            port=port,
            cpu_threads=cpu_threads,
            ram=ram,
            time=time,
            jupyter_arguments=jupyter_arguments,
        )

        # Submits the job to the server and waits for connection info
        job = self.submit_job(job=job)  # type: ignore[assignment]

        # At this point, submit_job should populate connection_info for JupyterJob
        if job.connection_info is None:
            message = f"Failed to retrieve connection information for Jupyter session {job.job_name}."
            console.error(message, RuntimeError)
            raise RuntimeError(message)

        # Creates and starts the SSH tunnel to enable localhost access to the Jupyter server
        tunnel = _SSHTunnel(
            ssh_client=self._client,
            local_port=job.connection_info.port,
            remote_host=job.connection_info.compute_node,
            remote_port=job.connection_info.port,
        )

        try:
            tunnel.start()
            console.echo(message="SSH tunnel: Established.", level=LogLevel.SUCCESS)

            # Prints connection information for the user
            console.echo(message=f"Jupyter server running on the compute node: {job.connection_info.compute_node}")
            console.echo(message=f"Local access port: {job.connection_info.port}")
            console.echo(message=f"Access URL: {job.connection_info.localhost_url}", level=LogLevel.INFO)

            # Blocks until the user presses Enter
            console.echo(message="Enter anything to terminate the interactive Jupyter session...")
            input()

        except KeyboardInterrupt:
            # Handles Ctrl+C gracefully
            console.echo(
                message=(
                    f"Keyboard interrupt signal: Detected. Terminating the interactive Jupyter session "
                    f"{job.job_name}..."
                ),
                level=LogLevel.WARNING,
            )

        finally:
            # Cleanup: stops the tunnel and aborts the SLURM job
            console.echo(message=f"Terminating the interactive Jupyter session {job.job_name}...")
            tunnel.stop()

            if job.job_id is not None:
                self.abort_job(slurm_job_id=int(job.job_id))

        return job

    def submit_job(self, job: Job | JupyterJob, *, verbose: bool = True) -> Job | JupyterJob:
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

            # Fallback to appease mypy, should not be reachable
            raise RuntimeError(message)

        # Otherwise, extracts the job id assigned to the job by SLURM from the response and writes it to the processed
        # Job object
        job_id = job_output.split()[-1]
        job.job_id = job_id

        # Special processing for Jupyter jobs: waits for and parses connection information
        if isinstance(job, JupyterJob):
            # Transfers host and user information to the JupyterJob object
            job.host = self.host
            job.user = self.user

            # Initializes a timer class to optionally delay loop cycling below
            timer = PrecisionTimer(precision=TimerPrecisions.SECOND)

            timer.reset()
            _wait_time = 120  # 2 minutes
            while timer.elapsed < _wait_time:  # Waits for at most 2 minutes before terminating with an error
                # Checks if the connection info file exists
                try:
                    # Pulls the connection info file
                    local_info_file = Path(tempfile.gettempdir()) / f"{job.job_name}_connection.txt"
                    self.pull(local_path=local_info_file, remote_path=job.connection_info_file)

                    # Parses connection data from the file and caches it inside Job class attributes
                    job.parse_connection_data(local_info_file)

                    # Removes the local file copy after it is parsed
                    local_info_file.unlink(missing_ok=True)

                    # Also removes the remote copy once the runtime is over
                    self.remove(remote_path=job.connection_info_file, is_dir=False)

                    # Breaks the waiting loop
                    break

                except Exception:
                    # The file doesn't exist yet or job initialization failed. Checks if the job has already
                    # terminated, indicating a startup error.
                    if job.job_id is not None:
                        status = self.get_job_status(slurm_job_id=int(job.job_id))
                        if status not in (JobStatus.PENDING, JobStatus.RUNNING):
                            message = (
                                f"Remote jupyter session job {job.job_name} with id {job.job_id} encountered a "
                                f"startup error and was terminated prematurely."
                            )
                            console.error(message, RuntimeError)

                timer.delay(delay=5, allow_sleep=True, block=False)  # Waits for 5 seconds before checking again
            else:
                # Aborts the job if the server is busy running other jobs
                self.abort_job(slurm_job_id=int(job.job_id))

                # Only raises the timeout error if the while loop is not broken in 120 seconds
                message = (
                    f"Remote jupyter session job {job.job_name} with id {job.job_id} did not start within 120 seconds "
                    f"from being submitted. Since all jupyter jobs are intended to be interactive and the server is "
                    f"busy running other jobs, this job has been cancelled."
                )
                console.error(message, TimeoutError)
                raise TimeoutError(message)  # Fallback to appease mypy

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
    def shared_storage_root(self) -> Path:
        """Returns the absolute path to the shared storage volume directory of the remote compute server accessible
        through this instance.
        """
        return Path(self._configuration.shared_storage_root)

    @property
    def shared_working_root(self) -> Path:
        """Returns the absolute path to the shared working volume directory of the remote compute server accessible
        through this instance.
        """
        return Path(self._configuration.shared_working_root)

    @property
    def user_data_root(self) -> Path:
        """Returns the absolute path to the storage volume directory used to store user's data on the remote compute
        server accessible through this instance.
        """
        return Path(self._configuration.user_data_root)

    @property
    def user_working_root(self) -> Path:
        """Returns the absolute path to the working volume directory used to store user's data on the remote compute
        server accessible through this instance.
        """
        return Path(self._configuration.user_working_root)

    @property
    def host(self) -> str:
        """Returns the hostname or IP address of the server accessible through this class."""
        return self._configuration.host

    @property
    def user(self) -> str:
        """Returns the username used to authenticate with the server."""
        return self._configuration.username

    @property
    def suite2p_configurations_directory(self) -> Path:
        """Returns the absolute path to the user's sl-suite2p configuration directory."""
        return self.user_working_root.joinpath("suite2p_configurations")

    @property
    def dlc_projects_directory(self) -> Path:
        """Returns the absolute path to the user's DeepLabCut project directory."""
        return self.user_working_root.joinpath("deeplabcut_projects")
