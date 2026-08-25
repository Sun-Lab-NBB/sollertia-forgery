# sollertia-forgery

Provides tools for processing and managing the data acquired using the Sollertia data acquisition platform.

![PyPI - Version](https://img.shields.io/pypi/v/sollertia-forgery)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sollertia-forgery)
[![uv](https://tinyurl.com/uvbadge)](https://github.com/astral-sh/uv)
[![Ruff](https://tinyurl.com/ruffbadge)](https://github.com/astral-sh/ruff)
![type-checked: mypy](https://img.shields.io/badge/type--checked-mypy-blue?style=flat-square&logo=python)
![PyPI - License](https://img.shields.io/pypi/l/sollertia-forgery)
![PyPI - Status](https://img.shields.io/pypi/status/sollertia-forgery)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/sollertia-forgery)

___

## Detailed Description

This library is part of the [Sollertia](https://github.com/Sun-Lab-NBB/sollertia) AI-assisted scientific data
acquisition and processing platform, built on the [Ataraxis](https://github.com/Sun-Lab-NBB/ataraxis) framework and
developed in the Sun (NeuroAI) lab at Cornell University. It processes the raw acquisition logs of a recorded session
into per-session data tables, verifies the integrity of the acquired data, and forges the processed sessions into
multi-session datasets.

The library plans each unit's jobs, sizes the cores and memory every job occupies, and dispatches the resulting batches
onto this machine's process pool or onto a SLURM compute server. It exposes that pipeline through the `slf` CLI and
through an MCP server that hands the same operations to AI agents.

___

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
  - [Source](#source)
  - [pip](#pip)
- [Usage](#usage)
  - [CLI Commands](#cli-commands)
  - [MCP Server](#mcp-server)
  - [Configuring Server Access](#configuring-server-access)
  - [Running a Remote Batch](#running-a-remote-batch)
  - [Running Headless Jobs](#running-headless-jobs)
  - [Creating Jobs](#creating-jobs)
  - [Submitting and Monitoring Jobs](#submitting-and-monitoring-jobs)
- [API Documentation](#api-documentation)
- [Developers](#developers)
  - [Installing the Project](#installing-the-project)
  - [Additional Dependencies](#additional-dependencies)
  - [Development Automation](#development-automation)
  - [AI-Assisted Development](#ai-assisted-development)
  - [Automation Troubleshooting](#automation-troubleshooting)
- [Versioning](#versioning)
- [Authors](#authors)
- [License](#license)
- [Acknowledgments](#acknowledgments)

___

## Dependencies

For users, all library dependencies are installed automatically by all supported installation methods. For developers,
see the [Developers](#developers) section for information on installing additional development dependencies.

___

## Installation

### Source

***Note,*** installation from source is ***highly discouraged*** for anyone who is not an active project developer.

1. Download this repository to the local machine using the preferred method, such as git-cloning. Use one of the
   [stable releases](https://github.com/Sun-Lab-NBB/sollertia-forgery/tags) that include precompiled binary and source
   code distribution (sdist) wheels.
2. If the downloaded distribution is stored as a compressed archive, unpack it using the appropriate decompression
   tool.
3. `cd` to the root directory of the prepared project distribution.
4. Run `pip install .` to install the project and its dependencies.

### pip

Use the following command to install the library and all of its dependencies via [pip](https://pip.pypa.io/en/stable/):
`pip install sollertia-forgery`

___

## Usage

### CLI Commands

This library provides the `slf` CLI that exposes the following commands:

| Command         | Description                                                                            |
|-----------------|----------------------------------------------------------------------------------------|
| `plan`          | Estimates and records the cores and memory every processing or forging job occupies    |
| `process`       | Runs the requested data extraction pipelines on a single session                       |
| `forge`         | Forges a dataset by assembling per-session data from a project's processed sessions    |
| `checksum`      | Resolves the data integrity checksum for the target session's 'raw_data' directory     |
| `manifest`      | Generates and inspects the project manifest that snapshots a project's state           |
| `dataset-state` | Snapshots each named dataset's forging job state into a shippable table                |
| `reset`         | Returns tracked jobs of the named units to the scheduled state                         |
| `clean`         | Removes a pipeline's output and processing tracker for the named units                 |
| `server`        | Interacts with the remote Sollertia compute server                                     |
| `mcp`           | Starts the agentic Model Context Protocol server using the requested transport         |
| `omp`           | Links the OpenMP runtime the Numba threading layer loads on macOS into a searched path |

Use `slf --help` or `slf SUBCOMMAND --help` for detailed usage information.

### MCP Server

This library provides an MCP server that exposes the planning, processing, forging, and project management pipelines
for AI agent integration.

#### Starting the Server

Start the MCP server using the CLI:

```bash
slf mcp
```

#### Available Tools

| Tool                              | Description                                                                     |
|-----------------------------------|---------------------------------------------------------------------------------|
| `define_forging_dataset_tool`     | Creates or extends a forged dataset hierarchy and its per-animal configurations |
| `generate_dataset_state_tool`     | Snapshots each named dataset's forging job state into a shippable feather file  |
| `read_dataset_state_tool`         | Reads a dataset's forging job state out of its stored snapshot                  |
| `list_project_datasets_tool`      | Lists the forged datasets stored under a project and which hold a given session |
| `generate_project_manifest_tool`  | Regenerates the target project's manifest and job artifacts                     |
| `read_project_manifest_tool`      | Reads a project's sessions out of its stored manifest                           |
| `read_project_jobs_tool`          | Reads a project's tracked jobs out of its stored job artifact                   |
| `get_manifest_status_tool`        | Reports the state of the project's last state-artifact generation               |
| `plan_session_jobs_tool`          | Records what every processing job of one or more sessions costs                 |
| `plan_dataset_jobs_tool`          | Records what every forging job of one or more datasets costs                    |
| `generate_project_plan_tool`      | Projects every plan cache under a project into one table at the project root    |
| `read_project_plan_tool`          | Reads the planned cores and memory of a project's jobs out of its projection    |
| `prepare_batch_tool`              | Resolves a pipeline's dispatchable jobs for one or more units                   |
| `inspect_job_resources_tool`      | Reports the cores and memory a pipeline's outstanding jobs need                 |
| `execute_jobs_tool`               | Dispatches prepared batches onto this machine's pool or the server's scheduler  |
| `get_processing_status_tool`      | Reports the live status of the active batch                                     |
| `cancel_processing_tool`          | Cancels the active local batch, or the outstanding remote allocations           |
| `reset_processing_jobs_tool`      | Resets tracked jobs to SCHEDULED across one or more units                       |
| `clean_processing_output_tool`    | Removes a pipeline's output and processing tracker for one or more units        |
| `read_server_configuration_tool`  | Loads the server configuration from the working directory, password masked      |
| `write_server_configuration_tool` | Creates or replaces the server configuration YAML in the working directory      |

#### Client Registration

MCP server registration and Claude Code skill assets for this library are distributed through the
[sollertia](https://github.com/Sun-Lab-NBB/sollertia) marketplace as part of the **forging** plugin. Install the plugin
from the marketplace to automatically register the MCP server with compatible clients and make all associated skills
available.

### Configuring Server Access

To access the remote compute server, first author the server configuration. The configuration is stored inside the
'server_configuration.yaml' file under the Sollertia platform working directory, and is created with the
`slf server configure` command. It records the username, the password, the host, the absolute path to the server's
data root, and the name of the shared conda environment every remote job activates.

### Running a Remote Batch

The remote path runs the same prepared jobs the local batch engine runs, so one job graph, one core table, and one
memory model serve both.

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

A job whose upstream stage the run can neither queue nor confirm to have already succeeded is reported as blocked
rather than submitted, which matches what a local batch does with the same job.

### Running Headless Jobs

A headless job is a job that does not require any user interaction during runtime. Currently, all headless jobs in
the Sollertia platform rely on pip-installable packages that expose a callable Command-Line Interface to carry out
some type of data processing. In this regard, **running a headless job is equivalent to calling a CLI command on
the local machine**, except that the command is executed on a remote compute server. Therefore, the primary purpose
of the API exposed by this library is to transfer the target command request to the remote server, execute it, and
monitor the runtime status until it is complete.

For example, the [cindra package](https://github.com/Sun-Lab-NBB/cindra) maintained in the Sollertia platform
processes 2-Photon data from experiment sessions. During data processing by the
[sollertia-forgery](https://github.com/Sun-Lab-NBB/sollertia-forgery) library, a remote job is sent to the server that
runs `slf process ... two-photon`, which drives cindra's stages in-process.

### Creating Jobs

All remote jobs are sent to the server in the form of an executable *shell* (.sh) script. The script is composed on
the local machine that uses this library and transferred to a temporary server directory using Secure Shell File
Transfer Protocol (SFTP). The server is then instructed to evaluate (run) the script using the SLURM job manager,
via a Secure Shell (SSH) session.

Broadly, each job consists of three major steps, which correspond to three major sections of the job shell script:

1. **Setting up the job environment**. Each job script starts with a SLURM job parameter block, which tells SLURM
   what resources (CPUs, GPUs, RAM, etc.) the job requires. When resources become available, SLURM generates a virtual
   environment and runs the rest of the job script in that environment. This forms the basis for using the shared
   compute resources fairly, as SLURM balances resource allocation and the order of job execution for all users.
2. **Activating the target conda environment**. Currently, all jobs are assumed to use Python libraries to execute
   the intended data processing. Similar to processing data locally, each job expects the remote server to provide
   a conda environment preconfigured with the necessary assets (packages) to run the job. Therefore, each job
   contains a section that activates the user-defined conda environment before running the rest of the job.
3. **Executing processing**. The final section is typically unique to each job and calls specific CLI commands or
   runs specific Python modules. Since each job is submitted as a shell script, it can do anything a server shell
   can do. Therefore, despite the Python-centric approach to data processing in the Sollertia platform, a remote
   job composed via this library can execute ***any*** arbitrary command available to the user on the remote
   server.

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

***Note,*** composing jobs by hand is the low-level path. Prefer the remote batch tools described above, which size
every allocation from the data it processes and build the dependency graph from each pipeline's own job ordering.

***Critical!*** Since running remote jobs is largely equivalent to executing them locally, all users are highly
encouraged to test their job scripts locally before deploying them server-side. If a script works on a local
machine, it is likely that the script behaves the same way on the server.

___

## API Documentation

See the [API documentation](https://sollertia-forgery-api-docs.netlify.app/) for the detailed description of the
methods and classes exposed by components of this library.

___

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

### AI-Assisted Development

Claude Code skills and other AI development assets for this project are distributed through two marketplaces:

- [sollertia](https://github.com/Sun-Lab-NBB/sollertia) marketplace: the **forging** plugin, which registers the
  `slf mcp` server with compatible MCP clients and provides the behavior processing skills covering session discovery,
  batch preparation and execution, output verification, and feather data querying.
- [ataraxis](https://github.com/Sun-Lab-NBB/ataraxis) marketplace: the **automation** plugin, which provides the shared
  development skills that enforce Sollertia platform coding conventions (Python style, README style, commit messages,
  pyproject.toml, tox configuration) and the general-purpose codebase exploration tools.

Install both plugins to make the full skill set available to compatible AI coding agents.

### Automation Troubleshooting

Many packages used in `tox` automation pipelines (uv, mypy, ruff) and `tox` itself may experience runtime failures. In
most cases, this is related to their caching behavior. If an unintelligible error is encountered with any of the
automation components, deleting the corresponding cache directories (`.tox`, `.ruff_cache`, `.mypy_cache`, etc.)
manually or via a CLI command typically resolves the issue.

___

## Versioning

This project uses [semantic versioning](https://semver.org/). See the
[tags on this repository](https://github.com/Sun-Lab-NBB/sollertia-forgery/tags) for the available project releases.

___

## Authors

- Ivan Kondratyev ([Inkaros](https://github.com/Inkaros))
- Kushaan Gupta ([kushaangupta](https://github.com/kushaangupta))
- Natalie Yeung

___

## License

This project is licensed under the Apache 2.0 License: see the [LICENSE](LICENSE) file for details.

___

## Acknowledgments

- All individuals who contributed to the development of this library, directly or indirectly.
- The creators of all other dependencies and projects listed in the [pyproject.toml](pyproject.toml) file.
