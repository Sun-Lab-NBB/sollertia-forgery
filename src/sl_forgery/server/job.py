"""This module provides the Job and JupyterJob classes that serve as the starting point for all SLURM-managed jobs
executed on remote compute server(s).
"""

import re
from typing import TYPE_CHECKING
import datetime
from dataclasses import dataclass

# noinspection PyProtectedMember
from simple_slurm import Slurm
from ataraxis_base_utilities import console

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class _JupyterConnectionInfo:
    """Stores the data used to establish the connection with a Jupyter notebook server running under SLURM control on a
    remote compute server.

    This class is used to transfer the connection metadata collected on the remote server back to the local machine
    that requested the Jupyter server session.
    """

    compute_node: str
    """The hostname of the compute node where Jupyter is running."""

    port: int
    """The port number on which Jupyter is listening for communication."""

    token: str
    """The authentication token for connecting to the Jupyter server."""

    @property
    def localhost_url(self) -> str:
        """Returns the localhost URL that can be used to connect to the server from the host-machine.

        Note:
            To use the URL returned by this function, first set up an SSH tunnel to the server via the specific Jupyter
            communication port and the remote server access credentials.
        """
        return f"http://localhost:{self.port}/?token={self.token}"


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
        _command: The SLURM command object used to assemble the job before it is translated into a shell script.
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
        self._command: Slurm = Slurm(
            cpus_per_task=cpu_threads,
            job_name=job_name,
            output=str(output_log),
            error=str(error_log),
            mem=f"{ram}G",
            time=datetime.timedelta(minutes=time),
        )

        # Conda shell initialization commands
        self._command.add_cmd("eval $(conda shell.bash hook)")
        self._command.add_cmd("conda init bash")

        # Activates the target conda environment for the command.
        self._command.add_cmd(f"source activate {conda_environment}")  # Need to use old syntax for our server.

    def __repr__(self) -> str:
        """Returns the string representation of the instance."""
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
        self._command.add_cmd(command)

    @property
    def command_script(self) -> str:
        """Translates the managed job into a shell-script-writable string.

        Notes:
            This method is used by the Server class to translate the job into the format that can be submitted to and
            executed by the remote compute server. Do not call this method directly.
        """
        # Appends the command to clean up (remove) the temporary script file after processing runtime is over
        self._command.add_cmd(f"rm -f {self.remote_script_path}")

        # Translates the command to string format
        script_content = str(self._command)

        # Replaces escaped $ (/$) with $. This is essential, as without this correction, things like conda
        # initialization would not work as expected. Returns the finalized script content to the caller.
        return script_content.replace("\\$", "$")


class JupyterJob(Job):
    """Defines a SLURM-managed job that launches an interactive Jupyter notebook on the remote compute server.

    This class extends the functionality of the base Job class to support running interactive Jupyter notebook sessions
    on the remote compute server while benefitting from resource allocation management offered by SLURM.

    Notes:
        Jupyter notebook sessions directly compete for resources with headless data processing jobs.

    Args:
        job_name: The descriptive name of the SLURM job to be created.
        output_log: The absolute path to the .txt file on the compute server to use for storing the messages sent by
            the job to the 'stdout' pipe.
        error_log: The absolute path to the .txt file on the compute server to use for storing the messages sent by
            the job to the 'stderr' pipe.
        working_directory: The absolute path to the compute server's directory where to store the temporary job's files.
        conda_environment: The name of the mamba / conda environment to activate on the server before running the job.
            For Jupyter notebook jobs, the environment must contain the 'jupyterlab' and 'notebook' Python packages.
        cpu_threads: The number of CPU threads to use for the job.
        ram: The amount of RAM to allocate for the job, in Gigabytes.
        time: The maximum period of time to run the job, in minutes.
        port: The connection port to use for the Jupyter server communication.
        notebook_directory: The remote compute server's directory where to run the Jupyter notebook.
        jupyter_arguments: Stores additional arguments to pass to the jupyter notebook initialization command.

    Attributes:
        port: The communication port for the managed Jupyter server.
        notebook_dir: The absolute path to the directory compute server's directory where to run the Jupyter notebook.
        connection_info: TheJupyterConnectionInfo instance that stores the initialized Jupyter notebook session's
            connection data.
        host: The hostname of the remote server.
        user: The username used to connect with the remote server.
        connection_info_file: The absolute path to the file on the remote compute server that contains the connection
            information for the initialized Jupyter notebook session.
        _command: The SLURM command object used to assemble the job before it is translated into a shell script.
    """

    def __init__(
        self,
        job_name: str,
        output_log: Path,
        error_log: Path,
        working_directory: Path,
        conda_environment: str,
        notebook_directory: Path,
        port: int = 9999,  # Defaults to using port 9999
        cpu_threads: int = 2,  # Defaults to 2 CPU cores
        ram: int = 32,  # Defaults to 32 GB of RAM
        time: int = 120,  # Defaults to 2 hours of runtime (120 minutes)
        jupyter_arguments: str = "",
    ) -> None:
        # Initializes parent Job class
        super().__init__(
            job_name=job_name,
            output_log=output_log,
            error_log=error_log,
            working_directory=working_directory,
            conda_environment=conda_environment,
            cpu_threads=cpu_threads,
            ram=ram,
            time=time,
        )

        # Saves important jupyter configuration parameters to class attributes
        self.port = port
        self.notebook_dir = notebook_directory

        # Similar to job ID, these attributes initialize to None and are reconfigured as part of the job submission
        # process.
        self.connection_info: _JupyterConnectionInfo | None = None
        self.host: str | None = None
        self.user: str | None = None

        # Resolves the server-side path to the jupyter server connection info file.
        self.connection_info_file = working_directory.joinpath(f"{job_name}_connection.txt")

        # Builds Jupyter launch command.
        self._build_jupyter_command(jupyter_arguments)

    def _build_jupyter_command(self, jupyter_arguments: str) -> None:
        """Builds the command to launch the Jupyter notebook server on the remote compute server.

        Args:
            jupyter_arguments: Additional arguments to pass to the Jupyter notebook initialization command.
        """
        # Gets the hostname of the compute node and caches it in the connection data file. Also caches the port name.
        self.add_command(f'echo "COMPUTE_NODE: $(hostname)" > {self.connection_info_file}')
        self.add_command(f'echo "PORT: {self.port}" >> {self.connection_info_file}')

        # Generates a random access token for security and caches it in the connection data file.
        self.add_command("TOKEN=$(openssl rand -hex 24)")
        self.add_command(f'echo "TOKEN: $TOKEN" >> {self.connection_info_file}')

        # Builds Jupyter startup command.
        jupyter_cmd = [
            "jupyter lab",
            "--no-browser",
            f"--port={self.port}",
            "--ip=0.0.0.0",  # Listen on all interfaces
            "--ServerApp.allow_origin='*'",  # Allow connections from SSH tunnel
            "--ServerApp.allow_remote_access=True",  # Enable remote access
            "--ServerApp.disable_check_xsrf=True",  # Helps with proxy connections
            f"--ServerApp.root_dir={self.notebook_dir}",  # Root directory (not notebook-dir)
            "--IdentityProvider.token=$TOKEN",  # Token authentication
        ]

        # Adds any additional arguments.
        if jupyter_arguments:
            jupyter_cmd.append(jupyter_arguments)

        # Adds the resolved jupyter command to the list of job commands.
        jupyter_cmd_str = " ".join(jupyter_cmd)
        self.add_command(jupyter_cmd_str)

    def parse_connection_data(self, data_file: Path) -> None:
        """Parses the connection information file created by the managed job on the remote server.

        Notes:
            This method is used by the Server instance to finalize the remote Jupyter session's initialization by
            parsing the connection instructions from the temporary storage file created by the job running on the
            remote server. Do not call this method directly.

        Args:
            data_file: The path to the .txt file generated on the remote compute server that stores the Jupyter
                connection data to be parsed.
        """
        with data_file.open() as f:
            content = f.read()

        # Extracts information using regex
        compute_node_match = re.search(r"COMPUTE_NODE: (.+)", content)
        port_match = re.search(r"PORT: (\d+)", content)
        token_match = re.search(r"TOKEN: (.+)", content)

        if not all([compute_node_match, port_match, token_match]):
            message = (
                f"Could not parse the connection data file for the Jupyter notebook session with id {self.job_id}."
            )
            console.error(message, ValueError)

        # Stores extracted data inside the connection_info attribute as a JupyterConnectionInfo instance.
        self.connection_info = _JupyterConnectionInfo(
            compute_node=compute_node_match.group(1).strip(),
            port=int(port_match.group(1)),
            token=token_match.group(1).strip(),
        )
