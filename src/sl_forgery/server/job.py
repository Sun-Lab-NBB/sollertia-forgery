"""This module provides the core Job class, used as the starting point for all SLURM-managed jobs executed on remote
compute server(s). Specifically, the Job class encapsulates the SLURM configuration and specific logic of each job.
During runtime, the Server class interacts with input Job objects to manage their transfer and execution on the remote
compute servers.

Since version 3.0.0, this module also provides the specialized JupyterJob class used to launch remote Jupyter notebook
servers.
"""

import re
from pathlib import Path
import datetime
from dataclasses import dataclass

# noinspection PyProtectedMember
from simple_slurm import Slurm
from ataraxis_base_utilities import LogLevel, console


@dataclass
class _JupyterConnectionInfo:
    """Stores the data used to establish the connection with a Jupyter notebook server running under SLURM control on a
    remote Sun lab server.

    This class is used to transfer the connection metadata collected on the remote server back to the local machine
    that requested the server to be established.
    """

    compute_node: str
    """The hostname of the compute node where Jupyter is running."""

    port: int
    """The port number on which Jupyter is listening for communication. Usually, this is the default port 8888 or 9999.
    """

    token: str
    """The authentication token for the Jupyter server. This token is used to authenticate the user when establishing 
    communication with the Jupyter server."""

    @property
    def localhost_url(self) -> str:
        """Returns the localhost URL for connecting to the server.

        To use this URL, first set up an SSH tunnel to the server via the specific Jupyter communication port and the
        remote server access credentials.
        """
        return f"http://localhost:{self.port}/?token={self.token}"


class Job:
    """Aggregates the data of a single SLURM-managed job to be executed on the Sun lab's remote compute server.

    This class provides the API for constructing any server-side job in the Sun lab. Internally, it wraps an instance
    of a Slurm class to package the job data into the format expected by the SLURM job manager. All jobs managed by this
    class instance should be submitted to an initialized Server instance's submit_job() method to be executed on the
    server.

    Notes:
        The initialization method of the class contains the arguments for configuring the SLURM and Conda environments
        used by the job. Do not submit additional SLURM or Conda commands via the 'add_command' method, as this may
        produce unexpected behavior.

        Each job can be conceptualized as a sequence of shell instructions to execute on the remote compute server. For
        the lab, that means that the bulk of the command consists of calling various CLIs exposed by data processing or
        analysis pipelines, installed in the calling user's Conda environments on the server. The Job instance also
        contains commands for activating the target conda environment and, in some cases, doing other preparatory or
        cleanup work. The source code of a 'remote' job is typically identical to what a human operator would type in a
        'local' terminal to run the same job on their PC.

        A key feature of server-side jobs is that they are executed on virtual machines managed by SLURM. Since the
        server has a lot more compute and memory resources than likely needed by individual jobs, each job typically
        requests a subset of these resources. Upon being executed, SLURM creates an isolated environment with the
        requested resources and runs the job in that environment.

    Args:
        job_name: The descriptive name of the SLURM job to be created. Primarily, this name is used in terminal
            printouts to identify the job to human operators.
        output_log: The absolute path to the .txt file on the processing server, where to store the standard output
            data of the job.
        error_log: The absolute path to the .txt file on the processing server, where to store the standard error
            data of the job.
        working_directory: The absolute path to the directory where temporary job files will be stored. During runtime,
            classes from this library use that directory to store files such as the job's shell script. All such files
            are automatically removed from the directory at the end of a non-errors runtime.
        conda_environment: The name of the conda environment to activate on the server before running the job logic. The
            environment should contain the necessary Python packages and CLIs to support running the job's logic.
        cpus_to_use: The number of CPUs to use for the job.
        ram_gb: The amount of RAM to allocate for the job, in Gigabytes.
        time_limit: The maximum time limit for the job, in minutes. If the job is still running at the end of this time
            period, it will be forcibly terminated. It is highly advised to always set adequate maximum runtime limits
            to prevent jobs from hogging the server in case of runtime or algorithm errors.

    Attributes:
        remote_script_path: Stores the path to the script file relative to the root of the remote server that runs the
            command.
        job_id: Stores the unique job identifier assigned by the SLURM manager to this job when it is accepted for
            execution. This field is initialized to None and is overwritten by the Server class that submits the job.
        job_name: Stores the descriptive name of the SLURM job.
        _command: Stores the managed SLURM command object.
    """

    def __init__(
        self,
        job_name: str,
        output_log: Path,
        error_log: Path,
        working_directory: Path,
        conda_environment: str,
        cpus_to_use: int = 10,
        ram_gb: int = 10,
        time_limit: int = 60,
    ) -> None:
        # Resolves the paths to the remote (server-side) .sh script file. This is the path where the job script
        # will be stored on the server, once it is transferred by the Server class instance.
        self.remote_script_path = str(working_directory.joinpath(f"{job_name}.sh"))

        # Defines additional arguments used by the Server class that executed the job.
        self.job_id: str | None = None  # This is set by the Server that submits the job.
        self.job_name: str = job_name  # Also stores the job name to support more informative terminal prints

        # Builds the slurm command object filled with configuration information
        self._command: Slurm = Slurm(
            cpus_per_task=cpus_to_use,
            job_name=job_name,
            output=str(output_log),
            error=str(error_log),
            mem=f"{ram_gb}G",
            time=datetime.timedelta(minutes=time_limit),
        )

        # Conda shell initialization commands
        self._command.add_cmd("eval $(conda shell.bash hook)")
        self._command.add_cmd("conda init bash")

        # Activates the target conda environment for the command.
        self._command.add_cmd(f"source activate {conda_environment}")  # Need to use old syntax for our server.

    def __repr__(self) -> str:
        """Returns the string representation of the Job instance."""
        return f"Job(name={self.job_name}, id={self.job_id})"

    def add_command(self, command: str) -> None:
        """Adds the input command string to the end of the managed SLURM job command list.

        This method is a wrapper around simple-slurm's add_cmd() method. It is used to iteratively build the shell
        command sequence for the managed job.

        Args:
            command: The command string to add to the command list, e.g.: 'python main.py --input 1'.
        """
        self._command.add_cmd(command)

    @property
    def command_script(self) -> str:
        """Translates the managed job data into a shell-script-writable string and returns it to caller.

        This method is used by the Server class to translate the job into the format that can be submitted to and
        executed on the remote compute server. The returned string is safe to dump into a .sh (shell script) file and
        move to the remote compute server for execution.
        """
        # Appends the command to clean up (remove) the temporary script file after processing runtime is over
        self._command.add_cmd(f"rm -f {self.remote_script_path}")

        # Translates the command to string format
        script_content = str(self._command)

        # Replaces escaped $ (/$) with $. This is essential, as without this correction, things like conda
        # initialization would not work as expected.
        fixed_script_content = script_content.replace("\\$", "$")

        # Returns the script content to the caller as a string
        return fixed_script_content


class JupyterJob(Job):
    """Aggregates the data of a specialized job used to launch a Jupyter notebook server under SLURM's control.

    This class extends the base Job class to include specific configuration and commands for starting a Jupyter notebook
    server in a SLURM environment. Using this specialized job allows users to set up remote Jupyter servers while
    benefitting from SLURM's job scheduling and resource management policies.

    Notes:
        Jupyter servers directly compete for resources with headless data processing jobs. Therefore, it is important
        to minimize the resource footprint and the runtime of each Jupyter server, if possible.

    Args:
        job_name: The descriptive name of the Jupyter SLURM job to be created. Primarily, this name is used in terminal
            printouts to identify the job to human operators.
        output_log: The absolute path to the .txt file on the processing server, where to store the standard output
            data of the job.
        error_log: The absolute path to the .txt file on the processing server, where to store the standard error
            data of the job.
        working_directory: The absolute path to the directory where to store temporary job files.
        conda_environment: The name of the conda environment to activate on the server before running the job. The
            environment should contain the necessary Python packages and CLIs to support running the job's logic. For
            Jupyter jobs, this necessarily includes the Jupyter notebook and jupyterlab packages.
        port: The connection port to use for the Jupyter server.
        notebook_directory: The root directory where to run the Jupyter notebook. During runtime, the notebook will
            only have access to items stored under this directory. For most runtimes, this should be set to the user's
            root working directory.
        cpus_to_use: The number of CPUs to allocate to the Jupyter server.
        ram_gb: The amount of RAM, in GB, to allocate to the Jupyter server.
        time_limit: The maximum Jupyter server uptime, in minutes.
        jupyter_args: Stores additional arguments to pass to the jupyter notebook initialization command.

    Attributes:
        port: Stores the connection port for the managed Jupyter server.
        notebook_dir: Stores the absolute path to the directory used to run the Jupyter notebook, relative to the
            remote server root.
        connection_info: Stores the JupyterConnectionInfo instance after the Jupyter server is instantiated.
        host: Stores the hostname of the remote server.
        user: Stores the username used to connect with the remote server.
        connection_info_file: Stores the absolute path to the file that contains the connection information for the
            initialized Jupyter session, relative to the remote server root.
        _command: Stores the shell command for launching the Jupyter server.
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
        cpus_to_use: int = 2,  # Defaults to 2 CPU cores
        ram_gb: int = 32,  # Defaults to 32 GB of RAM
        time_limit: int = 120,  # Defaults to 2 hours of runtime (120 minutes)
        jupyter_args: str = "",
    ) -> None:
        # Initializes parent Job class
        super().__init__(
            job_name=job_name,
            output_log=output_log,
            error_log=error_log,
            working_directory=working_directory,
            conda_environment=conda_environment,
            cpus_to_use=cpus_to_use,
            ram_gb=ram_gb,
            time_limit=time_limit,
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
        self._build_jupyter_command(jupyter_args)

    def _build_jupyter_command(self, jupyter_args: str) -> None:
        """Builds the command to launch the Jupyter notebook server on the remote Sun lab server."""
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
        if jupyter_args:
            jupyter_cmd.append(jupyter_args)

        # Adds the resolved jupyter command to the list of job commands.
        jupyter_cmd_str = " ".join(jupyter_cmd)
        self.add_command(jupyter_cmd_str)

    def parse_connection_info(self, info_file: Path) -> None:
        """Parses the connection information file created by the Jupyter job on the remote server.

        This method is used to finalize the remote Jupyter session initialization by parsing the connection session
        instructions from the temporary storage file created by the remote Job running on the server. After this
        method's runtime, the print_connection_info() method can be used to print the connection information to the
        terminal.

        Args:
            info_file: The path to the .txt file generated by the remote server that stores the Jupyter connection
                information to be parsed.
        """
        with info_file.open() as f:
            content = f.read()

        # Extracts information using regex
        compute_node_match = re.search(r"COMPUTE_NODE: (.+)", content)
        port_match = re.search(r"PORT: (\d+)", content)
        token_match = re.search(r"TOKEN: (.+)", content)

        if not all([compute_node_match, port_match, token_match]):
            message = f"Could not parse connection information file for the Jupyter server job with id {self.job_id}."
            console.error(message, ValueError)

        # Stores extracted data inside the connection_info attribute as a JupyterConnectionInfo instance.
        self.connection_info = _JupyterConnectionInfo(
            compute_node=compute_node_match.group(1).strip(),  # type: ignore
            port=int(port_match.group(1)),  # type: ignore
            token=token_match.group(1).strip(),  # type: ignore
        )

    def print_connection_info(self) -> None:
        """Constructs and displays the command to set up the SSH tunnel to the server and the link to the localhost
        server view in the terminal.

        The SSH command should be used via a separate terminal or subprocess call to establish the secure SSH tunnel to
        the Jupyter server. Once the SSH tunnel is established, the printed localhost url can be used to view the
        server from the local machine's browser.
        """
        # If connection information is not available, there is nothing to print
        if self.connection_info is None:
            console.echo(
                message=(
                    f"No connection information is available for the job {self.job_name}, which indicates that the job "
                    f"has not been submitted to the server. Submit the job for execution to the remote Sun lab server "
                    f"to generate the connection information"
                ),
                level=LogLevel.WARNING,
            )
            return  # No connection information available, so does not proceed with printing.

        # Prints generic connection details to the terminal
        console.echo(f"Jupyter is running on: {self.connection_info.compute_node}")
        console.echo(f"Port: {self.connection_info.port}")
        console.echo(f"Token: {self.connection_info.token}")

        # Constructs and displays the SSH tunnel command and the localhost url for connecting to the server
        tunnel_cmd = (
            f"ssh -N -L {self.connection_info.port}:{self.connection_info.compute_node}:{self.connection_info.port} "
            f"{self.user}@{self.host}"
        )
        localhost_url = f"http://localhost:{self.connection_info.port}/?token={self.connection_info.token}"
        print("\nTo access locally, run this in a terminal:")
        print(tunnel_cmd)
        print(f"\nThen open: {localhost_url}")
