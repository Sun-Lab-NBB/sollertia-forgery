"""Provides the Job class for SLURM-managed jobs on remote compute servers."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

from ataraxis_time import TimeUnits, to_timedelta

if TYPE_CHECKING:
    from pathlib import Path
    import datetime
    from collections.abc import Sequence

_SECONDS_PER_DAY: int = 86400
"""The divisor extracting the day field of a SLURM wall-time string."""

_SECONDS_PER_HOUR: int = 3600
"""The divisor extracting the hour field of a SLURM wall-time string."""

_SECONDS_PER_MINUTE: int = 60
"""The divisor extracting the minute field of a SLURM wall-time string."""


class Job:
    """Defines a non-interactive SLURM-managed job to be executed on the remote compute server.

    Notes:
        A job that names dependencies runs only after every named allocation completes successfully, and is canceled
        outright once any of them can no longer do so. Sequencing a pipeline this way lets the submitting process exit
        as soon as the whole graph is queued.

    Args:
        job_name: The descriptive name of the SLURM job to be created.
        output_log: The absolute path to the .txt file on the compute server to use for storing the messages sent by
            the job to the 'stdout' pipe.
        error_log: The absolute path to the .txt file on the compute server to use for storing the messages sent by
            the job to the 'stderr' pipe.
        working_directory: The absolute path to the compute server's directory in which the temporary job files are
            stored.
        conda_environment: The name of the mamba / conda environment to activate on the server before running the job.
        cpu_threads: The number of CPU threads to use for the job.
        ram: The amount of RAM to allocate for the job, in gigabytes.
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
        # The Server instance transfers the script to this path, so the path is fixed at construction rather than
        # at submission.
        self.remote_script_path: str = str(working_directory.joinpath(f"{job_name}.sh"))

        self.job_id: str | None = None
        self.job_name: str = job_name  # Supports more informative terminal prints.

        self._command: _SlurmScript = _SlurmScript(
            cpus_per_task=cpu_threads,
            job_name=job_name,
            output=str(output_log),
            error=str(error_log),
            memory=f"{ram}G",
            time=to_timedelta(time=time, from_units=TimeUnits.MINUTE),
            dependencies=tuple(dependencies),
            cleanup_path=self.remote_script_path,
        )

        # Initializes the conda shell hooks required before an environment can be activated.
        self._command.add_preamble(command="eval $(conda shell.bash hook)")
        self._command.add_preamble(command="conda init bash")

        # Uses the 'source activate' form, which is the activation syntax the reference compute server's conda
        # installation supports.
        self._command.add_preamble(command=f"source activate {conda_environment}")

    def __repr__(self) -> str:
        """Returns a string representation of the Job instance."""
        return f"Job(name={self.job_name}, id={self.job_id})"

    def add_command(self, command: str) -> None:
        """Adds the input command string to the end of the job's command sequence.

        Notes:
            The instance generates the job's SLURM directive header and a Conda activation preamble during class
            initialization. Do not submit additional SLURM or Conda commands via this method, as this may produce
            unexpected behavior.

            Commands added through this method run under shell error checking, so the job exits with the status of the
            first command that fails. The scheduler reads that status and sequences a dependent job behind this one.

        Args:
            command: The command string to append to the job's command sequence, for example
                'python main.py --input 1'.
        """
        self._command.add_command(command=command)

    @property
    def command_script(self) -> str:
        """Returns the managed job rendered as a shell-script-writable string."""
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
        dependencies: The SLURM-assigned identifiers of the allocations that must complete before this job runs,
            rendered as one ``afterok`` directive.
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
            f"#SBATCH --time={self._format_time(time_delta=time)}",
        ]
        if dependencies:
            self._directives.append(f"#SBATCH --dependency=afterok:{':'.join(dependencies)}")
            # A dependency that can never be satisfied leaves the dependent queued for good, so the scheduler is asked
            # to cancel it instead. That turns an unreachable stage into a terminal state a status query can report.
            self._directives.append("#SBATCH --kill-on-invalid-dep=yes")
        self._preamble: list[str] = []
        self._commands: list[str] = []
        self._cleanup_path: str = cleanup_path

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
        """Renders the complete batch script as a string.

        Returns:
            The full script text, including the directive header, the exit trap, the preamble, and the command body.
        """
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

    @staticmethod
    def _format_time(time_delta: datetime.timedelta) -> str:
        """Formats a timedelta as a SLURM-compatible time string (``[D-]HH:MM:SS``).

        Args:
            time_delta: The wall-time to format.

        Returns:
            The formatted time string, which carries a day field only for a wall-time of at least one day.
        """
        total_seconds = int(time_delta.total_seconds())
        days, remainder = divmod(total_seconds, _SECONDS_PER_DAY)
        hours, remainder = divmod(remainder, _SECONDS_PER_HOUR)
        minutes, seconds = divmod(remainder, _SECONDS_PER_MINUTE)
        if days:
            return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
