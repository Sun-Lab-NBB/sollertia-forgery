"""Provides the API for submitting jobs to SLURM-managed servers, monitoring job status, and managing remote data."""

from __future__ import annotations

from enum import StrEnum
import stat
import shlex
from typing import TYPE_CHECKING, Self
from pathlib import Path
import tempfile
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from natsort import natsorted
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

_MAXIMUM_CONNECTION_RETRIES: int = 30
"""The number of times a Server instance retries the SSH handshake after the initial attempt fails, before it reports
the server unreachable."""

_EXPECTED_FIELD_COUNT: int = 2
"""The number of pipe-separated fields a parsable accounting or queue row carries."""

_REPORTED_ERROR_CHARACTERS: int = 2000
"""The number of characters of a failed command's error output that an error message carries. A command that cannot
read many directories reports one line per directory, which is worth naming but not worth printing whole."""


@dataclass(frozen=True, slots=True)
class _CommandResult:
    """Stores the result of executing a command on the remote server."""

    stdout: str
    """The standard output the command produced."""
    stderr: str
    """The standard error output the command produced."""
    return_code: int
    """The exit code of the command, where zero indicates success."""


class JobStatus(StrEnum):
    """Defines the set of status codes this library resolves for managed jobs.

    These are the states SLURM reports, plus the ``BLOCKED``, ``UNKNOWN``, and ``UNRESOLVED`` states resolved locally.
    """

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
    BOOT_FAIL = "BOOT_FAIL"
    """The job terminated because a node allocated to it failed to boot."""
    DEADLINE = "DEADLINE"
    """The job was terminated on reaching the deadline its partition enforces."""
    PREEMPTED = "PREEMPTED"
    """The job was terminated to release its resources to a higher-priority allocation."""
    REVOKED = "REVOKED"
    """The job's allocation was revoked, which a federated scheduler does once a sibling cluster starts the job."""
    BLOCKED = "BLOCKED"
    """The job is queued behind a dependency that can no longer be satisfied, so it will never run. Resolved from the
    queue's reason field rather than from accounting, which still reports such a job as pending."""
    UNKNOWN = "UNKNOWN"
    """Accounting returned a row for the allocation whose state string this enumeration does not model, which covers
    every live state beyond PENDING and RUNNING, such as SUSPENDED, CONFIGURING, or COMPLETING. A row exists, so the
    scheduler still holds the allocation and this state is never terminal."""
    UNRESOLVED = "UNRESOLVED"
    """Accounting was queried successfully and returned no row for the allocation, which covers a submission it has not
    registered yet as well as one it has purged. Nothing observable separates those two, so this state is never
    terminal and a batch an allocation holds it on is retired by an explicit caller rather than on a timer."""


TERMINAL_JOB_STATUSES: frozenset[JobStatus] = frozenset(
    {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.TIMEOUT,
        JobStatus.NODE_FAIL,
        JobStatus.OUT_OF_MEMORY,
        JobStatus.BOOT_FAIL,
        JobStatus.DEADLINE,
        JobStatus.PREEMPTED,
        JobStatus.REVOKED,
        JobStatus.BLOCKED,
    }
)
"""The statuses a job never leaves. Reaching one tells a caller that a polled submission has settled.

Notes:
    ``UNKNOWN`` and ``UNRESOLVED`` are both absent. A row this enumeration cannot parse still proves that the
    scheduler holds the allocation, and a successful query that returned no row for it covers a submission accounting
    has yet to register as well as one it has purged. Nothing observable separates those two, so an allocation that
    stays unresolvable never settles its batch on its own. The remote status read classifies such a batch as stalled
    and names the allocations, and the caller retires it explicitly.
"""


class Server:
    """Establishes and maintains a bidirectional interface that allows working with a remote compute server.

    Submits SLURM-managed jobs to the server, monitors their execution status, and manages the data stored on the
    server over the SFTP protocol.

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

        self._configuration: ServerConfiguration = configuration

        timer = PrecisionTimer(precision=TimerPrecisions.SECOND)

        attempt = 0
        while True:
            console.echo(
                message=(
                    f"Connecting to {self._configuration.host} (attempt {attempt}/{_MAXIMUM_CONNECTION_RETRIES})..."
                ),
                level=LogLevel.INFO,
            )
            # Built into a local until both handles are open, because an attempt that authenticates and then fails
            # leaves a live transport thread behind. Binding it to the instance first would hide that transport from
            # close(), whose guard only clears once this loop has succeeded.
            client = paramiko.SSHClient()
            try:
                # The compute server is named by the operator's own configuration file, and prompting for an unknown
                # host key would hang a headless job, so an unrecognized key is accepted.
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # noqa: S507
                client.connect(
                    hostname=self._configuration.host,
                    username=self._configuration.username,
                    password=self._configuration.password,
                )
                console.echo(message=f"Connected to {self._configuration.host}", level=LogLevel.SUCCESS)

                # The client is reused for every file transfer operation during the Server instance's lifetime.
                sftp: SFTPClient = client.open_sftp()
            except paramiko.AuthenticationException:
                client.close()
                message = (
                    f"Authentication failed when connecting to {self._configuration.host} using "
                    f"{self._configuration.username} user."
                )
                console.error(message=message, error=PermissionError)
            except Exception:
                client.close()
                if attempt == _MAXIMUM_CONNECTION_RETRIES:
                    message = (
                        f"Could not connect to {self._configuration.host} after {_MAXIMUM_CONNECTION_RETRIES} "
                        f"retries. Aborting runtime."
                    )
                    console.error(message=message, error=ConnectionError)

                console.echo(
                    message=f"Could not SSH into {self._configuration.host}, retrying after a 2-second delay...",
                    level=LogLevel.WARNING,
                )
                attempt += 1
                timer.delay(delay=2, allow_sleep=True, block=False)
            else:
                self._client: SSHClient = client
                self._sftp: SFTPClient = sftp
                self._open = True
                break

    def __del__(self) -> None:
        """Terminates an open connection to the server before the instance is destroyed."""
        self.close()

    def __repr__(self) -> str:
        """Returns a string representation of the Server instance."""
        return f"Server(host={self.host}, user={self.user}, open={self._open})"

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

        Args:
            job: The Job instance that defines the job to be executed.
            verbose: Determines whether to emit the submission start and success messages. The warning issued for a
                job that was already submitted is emitted regardless.

        Returns:
            The job object whose 'job_id' attribute had been replaced with the SLURM-assigned job ID.

        Raises:
            RuntimeError: If the job cannot be submitted to the server for any reason.
        """
        if verbose:
            console.echo(message=f"Submitting '{job.job_name}' job to the remote server {self.host}...")

        if job.job_id is not None:
            console.echo(
                message=f"The '{job.job_name}' job has already been submitted to the server.",
                level=LogLevel.WARNING,
            )
            return job

        # Generates a temporary shell script on the local machine. Uses tempfile to automatically remove the
        # local script as soon as it is uploaded to the server.
        with tempfile.TemporaryDirectory() as temporary_directory:
            local_script_path = Path(temporary_directory).joinpath(f"{job.job_name}.sh")
            script_content = job.command_script

            with local_script_path.open("w") as script_file:
                script_file.write(script_content)

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
            console.error(message=message, error=RuntimeError)

        submission = self.execute_command(command=f"sbatch {script_path}")
        job_output = submission.stdout.strip()

        if "Submitted batch job" not in job_output:
            message = (
                f"Failed to submit the '{job.job_name}' job to the remote compute server. "
                f"{submission.stderr.strip() or job_output}"
            )
            console.error(message=message, error=RuntimeError)

        # SLURM reports the assigned identifier as the last token of the acknowledgement line.
        job_id = job_output.split()[-1]
        job.job_id = job_id

        if verbose:
            console.echo(message=f"{job.job_name} job: Submitted to {self.host}.", level=LogLevel.SUCCESS)

        return job

    def abort_job(self, slurm_job_id: str) -> None:
        """Aborts the job with the specified SLURM-assigned ID if it is currently running or pending on the server.

        Args:
            slurm_job_id: The SLURM-assigned job ID to abort.

        Raises:
            RuntimeError: If the accounting query that reads the allocation's state fails.
        """
        if self.get_job_status(slurm_job_id=slurm_job_id) in (JobStatus.PENDING, JobStatus.RUNNING):
            self.execute_command(command=f"scancel {slurm_job_id}")

    def abort_jobs(self, slurm_job_ids: Sequence[str]) -> None:
        """Aborts every named allocation that is still queued or running on the server.

        Notes:
            The scheduler answers a cancellation naming an allocation it has already finished with as a success, so
            naming a settled allocation is not a failure here. A non-zero exit therefore reports that the cancellation
            did not reach the scheduler at all, and it is raised the way every other command in this class raises,
            because a caller that resets a job's tracker behind this call depends on the cancellation having been
            issued.

        Args:
            slurm_job_ids: The SLURM-assigned job IDs to abort.

        Raises:
            RuntimeError: If the cancellation fails.
        """
        if not slurm_job_ids:
            return
        command = f"scancel {' '.join(shlex.quote(str(job)) for job in slurm_job_ids)}"
        result = self.execute_command(command=command)
        if result.return_code != 0:
            message = (
                f"Unable to cancel the named allocations on the remote compute server with '{command}'. "
                f"{result.stderr.strip()[:_REPORTED_ERROR_CHARACTERS]}"
            )
            console.error(message=message, error=RuntimeError)

    def get_job_status(self, slurm_job_id: str) -> JobStatus:
        """Queries the managed server's SLURM manager for the runtime status of the job with the specified
        SLURM-assigned ID.

        Notes:
            Uses the 'sacct' command to determine the current state of the job, returning the actual status
            (e.g., PENDING, RUNNING, COMPLETED, FAILED) assigned by the SLURM manager. A pending allocation is
            additionally checked against the queue's reason field, so a job whose dependency can no longer be satisfied
            is reported as BLOCKED.

        Args:
            slurm_job_id: The SLURM-assigned job ID for which to query the runtime status.

        Returns:
            The current status of the job as a JobStatus enumeration value.

        Raises:
            RuntimeError: If the accounting query fails.
        """
        return self.get_job_statuses(slurm_job_ids=(slurm_job_id,))[slurm_job_id]

    def get_job_statuses(self, slurm_job_ids: Sequence[str]) -> dict[str, JobStatus]:
        """Queries the runtime status of every named allocation in one accounting call.

        Notes:
            Accounting reports each allocation alongside its steps, and only the allocation rows are read.

            A pending allocation whose dependency can no longer be satisfied is reported as blocked. Accounting still
            calls that job pending, so the queue's reason field is the only source of that distinction.

            A query that fails reports no state at all rather than a state per identifier. Accounting that cannot
            answer writes nothing to standard output, which is indistinguishable from an answer that holds no row for
            any of the requested allocations, so reporting that answer would write off every live allocation at once.

        Args:
            slurm_job_ids: The SLURM-assigned job IDs to query.

        Returns:
            A dictionary mapping every requested job ID to its status. An allocation for which the successful query
            returned no row reports as ``UNRESOLVED``.

        Raises:
            RuntimeError: If the accounting query fails.
        """
        requested = [str(job_id) for job_id in slurm_job_ids]
        if not requested:
            return {}

        # An identifier the answer holds no row for keeps this seed, so 'no row' stays distinguishable from a row
        # whose state this stack does not model.
        statuses: dict[str, JobStatus] = dict.fromkeys(requested, JobStatus.UNRESOLVED)
        command = f"sacct -j {','.join(requested)} --format=JobID,State --noheader --parsable2"
        result = self.execute_command(command=command)
        if result.return_code != 0:
            message = (
                f"Unable to read the state of the requested allocations from the remote compute server with "
                f"'{command}'. {result.stderr.strip()[:_REPORTED_ERROR_CHARACTERS]}"
            )
            console.error(message=message, error=RuntimeError)

        for line in result.stdout.splitlines():
            fields = line.split("|")
            if len(fields) < _EXPECTED_FIELD_COUNT:
                continue
            job_id = fields[0].strip()
            # Step rows carry a suffixed identifier ('12345.batch'), and describe part of the allocation rather than
            # the allocation itself.
            if "." in job_id or job_id not in statuses:
                continue
            statuses[job_id] = _parse_job_status(state=fields[1].strip())

        if any(status is JobStatus.PENDING for status in statuses.values()):
            for job_id in self.get_blocked_job_ids():
                if job_id in statuses:
                    statuses[job_id] = JobStatus.BLOCKED

        return statuses

    def get_blocked_job_ids(self) -> set[str]:
        """Returns the identifiers of this user's queued allocations whose dependencies can no longer be satisfied.

        Notes:
            Queries the user's whole queue, since naming an allocation that the queue no longer holds makes the
            command report an error for it.

        Returns:
            The SLURM-assigned job IDs the queue reports as permanently blocked.
        """
        result = self.execute_command(command=f'squeue -h -u {shlex.quote(self.user)} -o "%i|%r"')
        rows = (line.split("|") for line in result.stdout.splitlines())
        return {
            fields[0].strip()
            for fields in rows
            if len(fields) >= _EXPECTED_FIELD_COUNT and fields[1].strip() == _BLOCKED_QUEUE_REASON
        }

    def get_queued_job_ids(self) -> set[str]:
        """Returns the identifiers of every allocation of this user that the scheduler's queue currently holds.

        Notes:
            The queue records what the scheduler holds right now, while accounting records what it has committed. The
            controller queues an allocation before slurmdbd commits a row for it, so accounting alone cannot tell a
            freshly queued allocation from a purged one. Reading the queue is what separates those two.

            Queries the user's whole queue, since naming an allocation that the queue no longer holds makes the
            command report an error for it.

            A user holding nothing answers with an empty set, which is the truthful reading of an empty queue. A
            command that fails raises instead, because a failure writes nothing to standard output and answering that
            as an empty queue would report every outstanding allocation as one the scheduler no longer holds.

        Returns:
            The SLURM-assigned job IDs the queue holds.

        Raises:
            RuntimeError: If the queue query fails.
        """
        command = f'squeue -h -u {shlex.quote(self.user)} -o "%i"'
        result = self.execute_command(command=command)
        if result.return_code != 0:
            message = (
                f"Unable to read the allocations the remote compute server's queue holds with '{command}'. "
                f"{result.stderr.strip()[:_REPORTED_ERROR_CHARACTERS]}"
            )
            console.error(message=message, error=RuntimeError)

        return {identifier for line in result.stdout.splitlines() if (identifier := line.strip())}

    def pull(self, local_path: Path, remote_path: Path) -> None:
        """Downloads a file or directory from the remote server to the local machine.

        Detects whether the remote path points to a file or directory and handles the transfer
        accordingly. For directories, all contents are recursively downloaded.

        Args:
            local_path: The path on the local machine where the file or directory will be saved.
            remote_path: The path to the file or directory on the remote server to download.

        Raises:
            FileNotFoundError: If the remote path does not exist on the server.
        """
        try:
            remote_stat = self._sftp.stat(path=str(remote_path))
        except FileNotFoundError:
            message = f"The remote path {remote_path} does not exist on the server."
            console.error(message=message, error=FileNotFoundError)

        if stat.S_ISDIR(remote_stat.st_mode):
            self._pull_directory(local_path=local_path, remote_path=remote_path)
        else:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self._sftp.get(localpath=str(local_path), remotepath=str(remote_path))

    def push(self, local_path: Path, remote_path: Path) -> None:
        """Uploads a file or directory from the local machine to the remote server.

        Detects whether the local path points to a file or directory and handles the transfer
        accordingly. For directories, all contents are recursively uploaded.

        Args:
            local_path: The path to the file or directory on the local machine to upload.
            remote_path: The path on the remote server where the file or directory will be saved.

        Raises:
            FileNotFoundError: If the local path does not exist.
        """
        if not local_path.exists():
            message = f"The local path {local_path} does not exist."
            console.error(message=message, error=FileNotFoundError)

        if local_path.is_dir():
            self._push_directory(local_path=local_path, remote_path=remote_path)
        else:
            self._create_directory(remote_path=remote_path.parent, parents=True)
            self._sftp.put(localpath=str(local_path), remotepath=str(remote_path))

    def create(self, remote_path: Path, *, is_dir: bool = True, parents: bool = True) -> None:
        """Creates a file or directory on the remote server.

        Notes:
            An existing target is left as it is, so a repeated call succeeds.

        Args:
            remote_path: The absolute path to the file or directory to create on the remote server.
            is_dir: Determines whether the created target is a directory rather than an empty file.
            parents: Determines whether missing parent directories are created alongside a requested directory.
                Parents are always created for a file.

        Raises:
            RuntimeError: If the remote directory creation command fails.
            FileNotFoundError: If ``parents`` is False and the target's parent directory does not exist.
        """
        if is_dir:
            self._create_directory(remote_path=remote_path, parents=parents)
        else:
            self._create_directory(remote_path=remote_path.parent, parents=True)

            if not self.exists(remote_path=remote_path):
                # Opening the path for writing and immediately closing it leaves an empty file behind.
                with self._sftp.open(filename=str(remote_path), mode="w"):
                    pass

    def remove(self, remote_path: Path, *, is_dir: bool, recursive: bool = False) -> None:
        """Removes a file or directory from the remote server.

        Args:
            remote_path: The path to the file or directory on the remote server to be removed.
            is_dir: Determines whether the input path represents a directory or a file.
            recursive: Determines whether the directory's contents are deleted before the directory itself.
                A non-recursive removal succeeds only for an empty directory.
        """
        if is_dir:
            if recursive:
                self._recursive_remove(remote_path=remote_path)
            else:
                self._sftp.rmdir(path=str(remote_path))
        else:
            self._sftp.unlink(path=str(remote_path))

    def exists(self, remote_path: Path) -> bool:
        """Returns True if the target file or directory exists on the remote server.

        Args:
            remote_path: The path to check on the remote server.

        Returns:
            True if the path exists, False otherwise.
        """
        try:
            self._sftp.stat(path=str(remote_path))
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
            file_stat = self._sftp.stat(path=str(remote_path))
        except FileNotFoundError:
            return False
        else:
            return stat.S_ISDIR(file_stat.st_mode)

    def list_directory(self, remote_path: Path) -> list[str]:
        """Lists the contents of a directory on the remote server.

        Args:
            remote_path: The path to the directory on the remote server.

        Returns:
            A list of filenames (not full paths) in the directory.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        return self._sftp.listdir(path=str(remote_path))

    def find_paths(
        self, remote_path: Path, names: Sequence[str], *, minimum_depth: int, maximum_depth: int
    ) -> list[Path]:
        """Returns every path under the target directory whose final component matches one of the given names.

        Notes:
            The search runs as one shell command rather than as a walk over the file-transfer protocol, which trades
            one query per directory and per candidate for a single round trip. Symbolic links are followed, and a link
            whose target does not resolve is reported as absent, so the answer matches the one that exists() gives
            for the same path.

        Args:
            remote_path: The absolute path to the directory to search on the remote server.
            names: The final path components to match. Each is matched as a shell name pattern, so a name carrying a
                glob metacharacter matches as a pattern rather than as a literal.
            minimum_depth: The lowest depth, counted in path components below the searched directory, at which a match
                is reported.
            maximum_depth: The highest depth, counted in path components below the searched directory, at which a match
                is reported. The search never descends past this depth.

        Returns:
            The absolute paths of every match, in natural sort order.

        Raises:
            FileNotFoundError: If the searched path is not a directory on the remote server.
            RuntimeError: If the search command failed, which leaves it having covered only part of the tree, or if
                the search reported an entry that does not sit under the searched directory.
        """
        if not self.is_directory(remote_path=remote_path):
            message = (
                f"Unable to search {remote_path} on the remote compute server. The server holds no directory at that "
                f"path."
            )
            console.error(message=message, error=FileNotFoundError)

        name_tests: list[str] = []
        for name in names:
            if name_tests:
                name_tests.append("-o")
            name_tests.extend(("-name", name))

        # '-L' follows symbolic links, matching the file-transfer queries this search replaces, and '! -type l' then
        # discards the links '-L' could not resolve, which those queries report as absent.
        command = shlex.join(
            [
                "find",
                "-L",
                str(remote_path),
                "-mindepth",
                str(minimum_depth),
                "-maxdepth",
                str(maximum_depth),
                "(",
                *name_tests,
                ")",
                "!",
                "-type",
                "l",
                "-print0",
            ]
        )

        result = self.execute_command(command=command)
        if result.return_code != 0:
            message = (
                f"Unable to search {remote_path} on the remote compute server. The search reached only part of the "
                f"tree, so its answer would omit paths the server holds. "
                f"{result.stderr.strip()[:_REPORTED_ERROR_CHARACTERS]}"
            )
            console.error(message=message, error=RuntimeError)

        # Records are separated rather than terminated by the split, so the trailing separator yields one empty entry.
        matches: list[Path] = []
        for record in result.stdout.split("\0"):
            if not record:
                continue
            match = Path(record)
            # The search echoes the directory it was given at the head of every record, so a record that does not
            # carry it is output that the search did not produce, and the answer containing it cannot be trusted.
            if not match.is_relative_to(remote_path):
                message = (
                    f"Unable to search {remote_path} on the remote compute server. The search reported the entry "
                    f"'{record[:_REPORTED_ERROR_CHARACTERS]}', which does not sit under the searched directory, so "
                    f"its answer carries output another program wrote."
                )
                console.error(message=message, error=RuntimeError)
            matches.append(match)
        return natsorted(matches)

    def execute_command(self, command: str) -> _CommandResult:
        """Executes the specified command on the remote server and returns the result.

        Notes:
            Both streams are drained concurrently. They share the channel's flow-control window, so draining either
            to its end before the other lets a command fill that window with the unread stream and block forever.
            This call then waits on output the command cannot finish writing.

        Args:
            command: The shell command to execute on the remote server.

        Returns:
            A _CommandResult instance containing stdout, stderr, and the return code of the executed command.
        """
        _, stdout, stderr = self._client.exec_command(command=command)
        with ThreadPoolExecutor(max_workers=1) as reader:
            pending_errors = reader.submit(stderr.read)
            output = stdout.read()
            errors = pending_errors.result()

        return _CommandResult(
            stdout=output.decode(),
            stderr=errors.decode(),
            return_code=stdout.channel.recv_exit_status(),
        )

    def close(self) -> None:
        """Closes the SFTP and SSH connections to the server."""
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
        """Returns the name of the shared conda environment, on the server, that every remote compute job activates
        before invoking the ``slf`` CLI.
        """
        return self._configuration.environment

    def _pull_directory(self, local_path: Path, remote_path: Path) -> None:
        """Recursively downloads a directory from the remote server.

        Args:
            local_path: The local directory path where contents will be saved.
            remote_path: The remote directory path to download.
        """
        local_path.mkdir(parents=True, exist_ok=True)

        remote_items = self._sftp.listdir_attr(path=str(remote_path))

        for item in remote_items:
            remote_item_path = remote_path / item.filename
            local_item_path = local_path / item.filename

            if stat.S_ISDIR(item.st_mode):
                self._pull_directory(local_path=local_item_path, remote_path=remote_item_path)
            else:
                self._sftp.get(localpath=str(local_item_path), remotepath=str(remote_item_path))

    def _push_directory(self, local_path: Path, remote_path: Path) -> None:
        """Recursively uploads a directory to the remote server.

        Args:
            local_path: The local directory path to upload.
            remote_path: The remote directory path where contents will be saved.
        """
        self._create_directory(remote_path=remote_path, parents=True)

        for local_item_path in local_path.iterdir():
            remote_item_path = remote_path / local_item_path.name

            if local_item_path.is_dir():
                self._push_directory(local_path=local_item_path, remote_path=remote_item_path)
            else:
                self._sftp.put(localpath=str(local_item_path), remotepath=str(remote_item_path))

    def _create_directory(self, remote_path: Path, *, parents: bool = True) -> None:
        """Creates a directory on the remote server.

        Notes:
            Creating a nested path is delegated to the shell, which resolves the whole chain in one round trip. Walking
            the chain over the file-transfer protocol instead costs one query per level, which a batch creating a
            directory per job pays many times over.

        Args:
            remote_path: The absolute path to the directory to create on the remote server.
            parents: Determines whether missing parent directories are created alongside the requested directory.
        """
        remote_path_str = str(remote_path)

        if parents:
            result = self.execute_command(command=f"mkdir -p {shlex.quote(remote_path_str)}")
            if result.return_code != 0:
                message = (
                    f"Unable to create the directory {remote_path_str} on the remote compute server. "
                    f"{result.stderr.strip()}"
                )
                console.error(message=message, error=RuntimeError)
        else:
            try:
                self._sftp.stat(path=remote_path_str)
            except FileNotFoundError:
                self._sftp.mkdir(path=remote_path_str)

    def _recursive_remove(self, remote_path: Path) -> None:
        """Recursively removes a directory and all its contents from the remote server.

        Args:
            remote_path: The path to the remote directory to recursively remove.
        """
        try:
            items = self._sftp.listdir_attr(path=str(remote_path))

            for item in items:
                item_path = remote_path / item.filename

                if stat.S_ISDIR(item.st_mode):
                    self._recursive_remove(remote_path=item_path)
                else:
                    self._sftp.unlink(path=str(item_path))

            self._sftp.rmdir(path=str(remote_path))

        except Exception as error:
            console.echo(
                message=f"Unable to remove the specified directory {remote_path}: {error!s}", level=LogLevel.WARNING
            )


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
