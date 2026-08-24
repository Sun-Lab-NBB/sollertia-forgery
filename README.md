# sollertia-forgery
STUB

## Usage

This section has been transferred from sollertia-shared-assets and requires verification before 1.0.0 release!

### Configuring Server Access

To access the remote compute server, first author the server configuration. The configuration is stored inside the
'server_configuration.yaml' file under the Sollertia platform working directory, and is created with the
`slf server configure` command. It records the username, the password, the host, the absolute path to the server's
data root, and the name of the shared conda environment every remote job activates.

### Running a Remote Batch

The remote path runs the same prepared jobs the local batch engine runs, so one job graph, one core table, and one
memory model serve both. Locality is a property of execution rather than of planning.

A run has three steps, each exposed as a Model Context Protocol tool and backed by the `slf` CLI:

1. **Prepare.** `prepare_batch_tool` with `host='remote'` refreshes the project's two tables on the server and pulls
   them. Planning (`slf plan`) reads each unit's acquisition data, registers the jobs that unit can actually run on
   its processing tracker, and records each job's cores, memory, and upstream jobs. State generation
   (`slf manifest create`, or `slf dataset-state` for a dataset) turns those trackers into a table. A job absent from
   that table is a job the unit cannot run, which is what lets the submitting host resolve a batch without opening
   anything on the server.
   The two tables join on the job identifier and become one descriptor per job, registered under a batch identifier.
2. **Submit.** `execute_jobs_tool` submits each job as its own SLURM allocation, sized from its own estimate, in
   dependency order. It takes no host of its own, because a batch runs where it was prepared. Each job names the
   allocations of the upstream jobs the batch holds through an `afterok` dependency, so the scheduler sequences the
   graph and nothing has to stay running locally for the batch to finish.
3. **Read.** `get_processing_status_tool` with `host='remote'` reports what the scheduler observed while a run is in
   flight. `generate_project_manifest_tool` and `generate_dataset_state_tool`, each with `host='remote'`, regenerate
   the project's manifest, job table, and dataset state on the server. Every read tool called with `host='remote'`
   then mirrors the project's artifacts into the working directory and reads them exactly as it reads a local
   project. The plan projection is mirrored as the server last wrote it, since replanning belongs to preparation.

Because the scheduler owns the run once it accepts the jobs, every accepted allocation is recorded in a submission
ledger at `<working directory>/remote_state/submission_ledger.yaml`, written under a file lock like every other shared
artifact this library keeps. That is what keeps concurrent batches all queryable, keeps a batch findable after this
process exits, and keeps the allocations already accepted recorded when the scheduler rejects a later job of the same
batch. A batch leaves the ledger once every allocation it holds has reached a state it never leaves and its closure
has snapshotted what its jobs recorded. A batch the query did not fully observe, and a batch whose closure failed,
both stay outstanding.

A job whose upstream stage the run can neither queue nor find already succeeded is reported as blocked rather than
submitted, which matches what a local batch does with the same job.

### Running Headless Jobs

A headless job is a job that does not require any user interaction during runtime. Currently, all headless jobs in the 
sollertia platform rely on pip-installable packages that expose a callable Command-Line Interface to carry out
some type of data processing. In this regard, **running a headless job is equivalent to calling a CLI command on your local 
machine**, except that the command is executed on a remote compute server. Therefore, the primary purpose of the API 
exposed by this library is to transfer the target command request to the remote server, execute it, and monitor the 
runtime status until it is complete.

For example, the [cindra package](https://github.com/Sun-Lab-NBB/cindra) maintained in the sollertia platform
processes 2-Photon data from experiment sessions. During data processing by the
[sollertia-forgery](https://github.com/Sun-Lab-NBB/sollertia-forgery) library, a remote job is sent to the server that
runs `slf process ... two-photon`, which drives cindra's stages in-process.

### Creating Jobs
All remote jobs are sent to the server in the form of an executable *shell* (.sh) script. The script is composed on the 
local machine that uses this library and transferred to a temporary server directory using Secure Shell File 
Transfer Protocol (SFTP). The server is then instructed to evaluate (run) the script using SLURM job manager, via a 
Secure Shell (SSH) session.

Broadly, each job consists of three major steps, which correspond to three major sections of the job shell script:
1. **Setting up the job environment**. Each job script starts with a SLURM job parameter block, which tells SLURM 
   what resources (CPUs, GPUs, RAM, etc.) the job requires. When resources become available, SLURM generates a virtual
   environment and runs the rest of the job script in that environment. This forms the basis for using the shared
   compute resources fairly, as SLURM balances resource allocation and the order of job execution for all users.
2. **Activating the target conda environment**. Currently, all jobs are assumed to use Python libraries to execute the 
   intended data processing. Similar to processing data locally, each job expects the remote server to provide a 
   Conda environment preconfigured with necessary assets (packages) to run the job. Therefore, each job contains a 
   section that activates the user-defined conda environment before running the rest of the job.
3. **Executing processing**. The final section is typically unique to each job and calls specific CLI commands or runs 
   specific Python modules. Since each job is submitted as a shell script, it can do anything a server shell can
   do. Therefore, despite python-centric approach to data processing in the sollertia platform, a remote job composed via this library 
   can execute ***any*** arbitrary command available to the user on the remove server.

Use the *Job* class exposed by this library to compose remote jobs. **Steps 1 and 2** of each job are configured when
initializing the Job instance, while **step 3** is added via the `add_command()` method of the Job class:
```python
from pathlib import Path
from sollertia_forgery.server import Job

# Instantiates a job. The resource arguments become the SBATCH directive block, and 'dependencies' names the
# allocations that must complete successfully before this job runs.
job = Job(
    job_name="0000-Session-motion_energy-1",
    output_log=Path("/server/root/processing_batches/batch01/0000-Session-motion_energy-1.out"),
    error_log=Path("/server/root/processing_batches/batch01/0000-Session-motion_energy-1.err"),
    working_directory=Path("/server/root/processing_batches/batch01"),
    conda_environment="slf_server",
    cpu_threads=16,
    ram=6,
    time=480,
    dependencies=("1000",),
)

# Adds the command the job runs. Commands added this way run under shell error checking, so the job exits with the
# status of the first command that fails.
job.add_command("slf process -sp /server/root/Project/Animal/Session -w 16 -np -id a1b2c3d4 video")
```

The rendered script removes itself through an exit trap rather than through a trailing command, so its exit status
stays the status of the work it ran. That is what a dependent allocation is sequenced against.

### Submitting and Monitoring Jobs

To submit a job, use a **Server** instance. It reads the server configuration authored above and supports the context
manager protocol, so the connection closes however the block ends:
```python
from sollertia_forgery.server import Server, JobStatus, TERMINAL_JOB_STATUSES, get_server_configuration

with Server(configuration=get_server_configuration()) as server:
    job = server.submit_job(job=job)

    # Queries every allocation of a batch in one accounting call, keyed by the identifier the scheduler assigned.
    statuses = server.get_job_statuses(slurm_job_ids=[job.job_id])
    if statuses[job.job_id] in TERMINAL_JOB_STATUSES:
        print(f"Job finished as {statuses[job.job_id]}.")
```

`get_job_statuses()` returns a `JobStatus` per allocation. Alongside the states accounting reports, it resolves
`BLOCKED` for a queued job whose dependency can no longer be satisfied, which accounting still calls pending.

**Note!** Composing jobs by hand is the low-level path. Prefer the remote batch tools described above, which size
every allocation from the data it will process and build the dependency graph from each pipeline's own job ordering.

**Critical!** Since running remote jobs is largely equivalent to executing them locally, all users are highly encouraged
to test their job scripts locally before deploying them server-side. If a script works on a local machine, it is likely
that the script would behave similarly and work on the server.

## Developers

This section provides installation, dependency, and build-system instructions for the developers that want to modify
the source code of this library.

### Installing the Project

***Note,*** this installation method requires **mamba version 2.3.2 or above**. Currently, all automation pipelines
require that mamba is installed through the [miniforge3](https://github.com/conda-forge/miniforge) installer.

1. Download this repository to the local machine using the preferred method, such as git-cloning.
2. If the downloaded distribution is stored as a compressed archive, unpack it using the appropriate decompression
   tool.
3. `cd` to the root directory of the prepared project distribution.
4. Install the core development dependencies into the ***base*** mamba environment via the
   `mamba install tox uv tox-uv` command.
5. Use the `tox -e create` command to create the project-specific development environment followed by `tox -e install`
   command to install the project into that environment as a library.

### Additional Dependencies

In addition to installing the project and all user dependencies, install the following dependencies:

1. [Python](https://www.python.org/downloads/) distributions, one for each version supported by the developed project.
   Currently, this library supports Python 3.14 only. It is recommended to use a tool like
   [pyenv](https://github.com/pyenv/pyenv) to install and manage the required versions.

### Development Automation

This project uses `tox` for development automation. The following tox environments are available:

| Environment    | Description                                                 |
|----------------|-------------------------------------------------------------|
| `lint`         | Runs ruff formatting, ruff linting, and mypy type checking  |
| `stubs`        | Generates py.typed marker and .pyi stub files               |
| `{py314}-test` | Runs the test suite via pytest and aggregates coverage data |
| `coverage`     | Aggregates test coverage and applies the coverage gate      |
| `docs`         | Builds the API documentation via Sphinx                     |
| `build`        | Builds sdist and wheel distributions                        |
| `upload`       | Uploads distributions to PyPI via twine                     |
| `deploy`       | Uploads the built documentation to the Netlify site         |
| `install`      | Builds and installs the project into its mamba environment  |
| `uninstall`    | Uninstalls the project from its mamba environment           |
| `create`       | Creates the project's mamba development environment         |
| `remove`       | Removes the project's mamba development environment         |
| `provision`    | Recreates the mamba environment from scratch                |
| `export`       | Exports the mamba environment as a .yml file                |
| `import`       | Creates or updates the mamba environment from a .yml file   |

Run any environment using `tox -e ENVIRONMENT`. For example, `tox -e lint`.

***Note,*** all pull requests for this project have to successfully complete the `tox` task before being merged. To
expedite the task's runtime, use the `tox --parallel` command to run some tasks in parallel.
