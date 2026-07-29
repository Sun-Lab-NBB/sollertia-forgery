"""Provides the Job class for SLURM-managed jobs on remote compute servers."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING
import datetime

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Sequence


class Job:
    """Defines a non-interactive SLURM-managed job to be executed on the remote compute server.

    This class provides the API for constructing and managing the non-interactive jobs running on remote compute
    servers.

    Notes:
        Instances of this class should be submitted to an initialized Server instance's submit_job() method to be
        executed on the remote compute server.

        A job that names dependencies runs only after every named allocation completes successfully, and is canceled
        outright once any of them can no longer do so. Sequencing a pipeline this way lets the submitting process exit
        as soon as the whole graph is queued.

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
        dependencies: The SLURM-assigned identifiers of the allocations that must complete successfully before this
            job runs. Leave empty for a job that waits on nothing.

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
        dependencies: Sequence[str] = (),
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
            dependencies=tuple(dependencies),
            cleanup_path=self.remote_script_path,
        )

        # Conda shell initialization commands
        self._command.add_preamble("eval $(conda shell.bash hook)")
        self._command.add_preamble("conda init bash")

        # Activates the target conda environment for the command.
        self._command.add_preamble(f"source activate {conda_environment}")  # Need to use old syntax for our server.

    def __repr__(self) -> str:
        """Returns the string representation of the Job instance."""
        return f"Job(name={self.job_name}, id={self.job_id})"

    def add_command(self, command: str) -> None:
        """Adds the input command string to the end of the job's command sequence.

        Notes:
            The instance generates a preamble section that configures the job's SLURM and Conda environments during
            class initialization. Do not submit additional SLURM or Conda commands via this method, as this may produce
            unexpected behavior.

            Commands added through this method run under shell error checking, so the job exits with the status of the
            first command that fails. The scheduler reads that status, which is what lets a dependent job be sequenced
            behind this one.

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

            Rendering does not modify the job, so a script rendered twice is identical both times.
        """
        return self._command.render()


class _SlurmScript:
    """Builds a SLURM batch script from resource parameters and shell commands.

    Notes:
        The rendered script removes itself through an exit trap rather than through a trailing command, so the script's
        exit status stays the status of the work it ran. A cleanup command placed last would mask every failure behind
        its own success, which would report every job as completed and let a dependent job run on absent input.

    Args:
        cpus_per_task: The number of CPU threads allocated to the job.
        job_name: The descriptive name of the SLURM job.
        output: The absolute path to the stdout log file on the compute server.
        error: The absolute path to the stderr log file on the compute server.
        memory: The memory allocation string in SLURM format, e.g. ``"10G"``.
        time: The maximum wall-time for the job.
        dependencies: The SLURM-assigned identifiers this job waits for, which are rendered as one ``afterok``
            directive.
        cleanup_path: The absolute path to the script file itself, removed when the job exits.

    Attributes:
        _directives: The list of ``#SBATCH`` directive lines for the script header.
        _preamble: The list of environment setup lines that run before error checking is enabled.
        _commands: The list of shell command lines appended to the script body.
        _cleanup_path: The path removed by the script's exit trap.
    """

    __slots__ = ("_cleanup_path", "_commands", "_directives", "_preamble")

    def __init__(
        self,
        cpus_per_task: int,
        job_name: str,
        output: str,
        error: str,
        memory: str,
        time: datetime.timedelta,
        dependencies: tuple[str, ...] = (),
        cleanup_path: str = "",
    ) -> None:
        self._directives: list[str] = [
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --output={output}",
            f"#SBATCH --error={error}",
            f"#SBATCH --mem={memory}",
            f"#SBATCH --time={self._format_time(time)}",
        ]
        if dependencies:
            self._directives.append(f"#SBATCH --dependency=afterok:{':'.join(dependencies)}")
            # A dependency that can never be satisfied leaves the dependent queued for good, so the scheduler is asked
            # to cancel it instead. That turns an unreachable stage into a terminal state a status query can report.
            self._directives.append("#SBATCH --kill-on-invalid-dep=yes")
        self._preamble: list[str] = []
        self._commands: list[str] = []
        self._cleanup_path: str = cleanup_path

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

    def add_preamble(self, command: str) -> None:
        """Appends an environment setup line that runs before error checking is enabled.

        Notes:
            Environment activation is kept outside error checking because a conda hook exports shell state whose exit
            status does not describe whether the environment is usable. The work itself runs under error checking, and
            an unusable environment fails there instead.

        Args:
            command: The setup line to append.
        """
        self._preamble.append(command)

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
        if self._cleanup_path:
            lines.append(f"trap 'rm -f {shlex.quote(self._cleanup_path)}' EXIT")
        lines.extend(self._preamble)
        lines.append("")
        # Enabled after the environment is activated, so the job's exit status is the status of its first failing
        # command rather than the status of whatever ran last.
        lines.append("set -eo pipefail")
        lines.extend(self._commands)
        return "\n".join(lines) + "\n"
