"""Provides the Job class for SLURM-managed jobs on remote compute servers."""

from __future__ import annotations

from typing import TYPE_CHECKING
import datetime

if TYPE_CHECKING:
    from pathlib import Path


class Job:
    """Defines a non-interactive SLURM-managed job to be executed on the remote compute server.

    This class provides the API for constructing and managing the non-interactive jobs running on remote compute
    servers.

    Notes:
        Instances of this class should be submitted to an initialized Server instance's submit_job() method to be
        executed on the remote compute server.

    Args:
        job_name: The descriptive name of the SLURM job to be created.
        output_log: The absolute path to the .txt file on the compute server to use for storing the messages sent by
            the job to the 'stdout' pipe.
        error_log: The absolute path to the .txt file on the compute server to use for storing the messages sent by
            the job to the 'stderr' pipe.
        working_directory: The absolute path to the compute server's directory where to store the temporary job's files.
        conda_environment: The name of the mamba / conda environment to activate on the server before running the job.
        cpu_threads: The number of CPU threads to use for the job.
        ram: The amount of RAM to allocate for the job, in Gigabytes.
        time: The maximum period of time to run the job, in minutes.

    Attributes:
        remote_script_path: The path to the job's script file on the remote compute server.
        job_id: The unique job identifier assigned by the SLURM manager to this job when it is accepted for execution.
        job_name: The descriptive name of the SLURM job.
        _command: The _SlurmScript instance used to assemble the job before it is translated into a shell script.
    """

    def __init__(
        self,
        job_name: str,
        output_log: Path,
        error_log: Path,
        working_directory: Path,
        conda_environment: str,
        cpu_threads: int = 10,
        ram: int = 10,
        time: int = 60,
    ) -> None:
        # Resolves the paths to the remote (server-side) .sh script file. This is the path where the job script
        # will be stored on the server, once it is transferred by the Server class instance.
        self.remote_script_path = str(working_directory.joinpath(f"{job_name}.sh"))

        # Defines additional arguments used by the Server class that executed the job.
        self.job_id: str | None = None  # This is set by the Server that submits the job.
        self.job_name: str = job_name  # Also stores the job name to support more informative terminal prints

        # Builds the slurm command object filled with configuration information
        self._command: _SlurmScript = _SlurmScript(
            cpus_per_task=cpu_threads,
            job_name=job_name,
            output=str(output_log),
            error=str(error_log),
            memory=f"{ram}G",
            time=datetime.timedelta(minutes=time),
        )

        # Conda shell initialization commands
        self._command.add_command("eval $(conda shell.bash hook)")
        self._command.add_command("conda init bash")

        # Activates the target conda environment for the command.
        self._command.add_command(f"source activate {conda_environment}")  # Need to use old syntax for our server.

    def __repr__(self) -> str:
        """Returns the string representation of the Job instance."""
        return f"Job(name={self.job_name}, id={self.job_id})"

    def add_command(self, command: str) -> None:
        """Adds the input command string to the end of the job's command sequence.

        Notes:
            The instance generates a preamble section that configures the job's SLURM and Conda environments during
            class initialization. Do not submit additional SLURM or Conda commands via this method, as this may produce
            unexpected behavior.

        Args:
            command: The command string to append to the job's command sequence, e.g.: 'python main.py --input 1'.
        """
        self._command.add_command(command)

    @property
    def command_script(self) -> str:
        """Translates the managed job into a shell-script-writable string.

        Notes:
            This method is used by the Server class to translate the job into the format that can be submitted to and
            executed by the remote compute server. Do not call this method directly.
        """
        # Appends the command to clean up (remove) the temporary script file after processing runtime is over
        self._command.add_command(f"rm -f {self.remote_script_path}")

        # Translates the command to string format and returns the finalized script content to the caller.
        return self._command.render()


class _SlurmScript:
    """Builds a SLURM batch script from resource parameters and shell commands.

    Args:
        cpus_per_task: The number of CPU threads allocated to the job.
        job_name: The descriptive name of the SLURM job.
        output: The absolute path to the stdout log file on the compute server.
        error: The absolute path to the stderr log file on the compute server.
        memory: The memory allocation string in SLURM format, e.g. ``"10G"``.
        time: The maximum wall-time for the job.

    Attributes:
        _directives: The list of ``#SBATCH`` directive lines for the script header.
        _commands: The list of shell command lines appended to the script body.
    """

    __slots__ = ("_commands", "_directives")

    def __init__(
        self,
        cpus_per_task: int,
        job_name: str,
        output: str,
        error: str,
        memory: str,
        time: datetime.timedelta,
    ) -> None:
        self._directives: list[str] = [
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --output={output}",
            f"#SBATCH --error={error}",
            f"#SBATCH --mem={memory}",
            f"#SBATCH --time={self._format_time(time)}",
        ]
        self._commands: list[str] = []

    @staticmethod
    def _format_time(time_delta: datetime.timedelta) -> str:
        """Formats a timedelta as a SLURM-compatible time string (``[D-]HH:MM:SS``)."""
        total_seconds = int(time_delta.total_seconds())
        days, remainder = divmod(total_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        if days > 0:
            return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def add_command(self, command: str) -> None:
        """Appends a shell command to the script body.

        Args:
            command: The shell command string to append.
        """
        self._commands.append(command)

    def render(self) -> str:
        """Renders the complete batch script as a string."""
        lines = ["#!/bin/bash"]
        lines.extend(self._directives)
        lines.append("")
        lines.extend(self._commands)
        return "\n".join(lines) + "\n"
