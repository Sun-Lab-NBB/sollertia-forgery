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
developed in the Sun (NeuroAI) lab at Cornell University. It is the processing counterpart to
[sollertia-experiment](https://github.com/Sun-Lab-NBB/sollertia-experiment), which acquires the sessions this library
reads. It processes the raw data of a recorded session into multiple per-session intermediate data tables and forges the
processed sessions into the multi-session datasets consumed by the downstream analysis assets.

Every pipeline runs the same way for every acquisition system. Each system contributes its own parsers and workers as
data, through the dispatch registries described in [Acquisition Systems](#acquisition-systems), so every command infers
its acquisition system from the session instead of taking one from the caller. The library plans each unit's jobs, sizes
the cores and memory used by every job, and dispatches the resulting batches onto this machine's process pool or onto a
SLURM compute server.

The documented path to using the library runs through AI agents. Every operation is exposed as a Model Context Protocol
tool, and the Claude Code skills described in [AI-Assisted Development](#ai-assisted-development) orchestrate those
tools. The `slf` CLI serves the same operations to a human operator and to the job scripts the compute server runs.

___

## Features

- Supports Windows, Linux, and macOS.
- Processes camera, microcontroller, acquisition-runtime, and two-photon imaging data for supported acquisition systems.
- Forges the processed sessions of a project into analysis-ready multi-session datasets.
- Sizes every job from the data it reads and dispatches it onto a local process pool or onto a SLURM scheduler.
- Exposes every planning, processing, forging, and management operation through an MCP server.
- Apache 2.0 License.

___

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
  - [Source](#source)
  - [pip](#pip)
- [Usage](#usage)
  - [Acquisition Systems](#acquisition-systems)
  - [Processing Pipelines](#processing-pipelines)
  - [Jobs, Trackers, and Batches](#jobs-trackers-and-batches)
  - [Processed Data Structure](#processed-data-structure)
  - [Forged Datasets](#forged-datasets)
  - [CLI Commands](#cli-commands)
  - [Configuring Server Access](#configuring-server-access)
  - [Running a Remote Batch](#running-a-remote-batch)
  - [Recovering from Interruptions](#recovering-from-interruptions)
- [API Documentation](#api-documentation)
- [AI-Assisted Development](#ai-assisted-development)
  - [MCP Server](#mcp-server)
  - [Skills](#skills)
- [Developers](#developers)
  - [Installing the Project](#installing-the-project)
  - [Additional Dependencies](#additional-dependencies)
  - [Development Automation](#development-automation)
  - [Adding a New Acquisition System](#adding-a-new-acquisition-system)
  - [Adding a New Session Type](#adding-a-new-session-type)
  - [Adding a New Processing Stage](#adding-a-new-processing-stage)
  - [Adding a New Processing Pipeline](#adding-a-new-processing-pipeline)
  - [Adding an MCP Tool](#adding-an-mcp-tool)
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

### Acquisition Systems

An acquisition system is the collection of hardware and software that records a session, and it is defined by
[sollertia-shared-assets](https://github.com/Sun-Lab-NBB/sollertia-shared-assets) as a member of the
`AcquisitionSystems` enumeration. Every session on disk names the system that recorded it, so the pipelines in this
library resolve the system from the session's metadata. The current reference system is Mesoscope-VR, which pairs a
2-Photon Random Access Mesoscope with a Unity virtual reality task.

The processing packages themselves are system-agnostic. A category package such as `video`, `microcontrollers`,
`runtime`, `two_photon`, or `forging` owns one pipeline that runs identically for every system, and everything a
specific system contributes lives in a per-system package such as `mesoscope_vr`. The two meet in the library's
`registries.py` module, which maps each `AcquisitionSystems` member to the parsers, resolvers, locators, and workers
donated by that member:

| Donation                       | What the system supplies                                                                   |
|--------------------------------|--------------------------------------------------------------------------------------------|
| Microcontroller parsers        | One parser per hardware module, keyed by the module's type and identifier                  |
| Microcontroller event codes    | The event codes each parsed module reads, which build the extraction filter                |
| Microcontroller eligibility    | The modules a given session configured for use                                             |
| Runtime binding                | The source identifier of the runtime log, paired with the parser that decodes its payloads |
| Video tracking                 | The pass that reads the session's pose predictions and writes its tracking outputs         |
| Two-photon data locator        | The raw imaging directory the two-photon pipeline hands to cindra                          |
| Cindra configuration resolvers | The single-recording and multi-recording configurations passed to cindra                   |
| Forging assembler              | The per-session worker that assembles the dataset row, plus its column meanings            |
| Forging admission policy       | The pipelines a session of each type completes before it joins a dataset                   |
| Multi-recording session types  | The session types the system tracks across recordings                                      |

A coverage check runs when `registries.py` is imported and raises a `RuntimeError` that names every donation missing
from a registered system, so a partially wired system fails at import time rather than midway through a batch. The
dependency runs one way, because a per-system package never imports an agnostic category package.

***Note,*** adding a system to the platform spans three repositories. The enumeration member and the session records
belong to sollertia-shared-assets, the acquisition runtime belongs to sollertia-experiment, and the donations belong
here. See [Adding a New Acquisition System](#adding-a-new-acquisition-system) for the steps this library requires.

### Processing Pipelines

Each pipeline reads one class of acquired data and writes its outputs beside the session. A pipeline is made of
ordered stages, and each stage contributes one or more independently schedulable jobs:

| Pipeline          | Stages                                                                                     | Unit    |
|-------------------|--------------------------------------------------------------------------------------------|---------|
| `checksum`        | `checksum_resolution`                                                                      | Session |
| `runtime`         | `runtime_processing`                                                                       | Session |
| `microcontroller` | `microcontroller_data_extraction`, `module_parsing`                                        | Session |
| `video`           | `camera_timestamp_extraction`, `camera_timestamp_rename`, `pose_tracking`, `motion_energy` | Session |
| `two_photon`      | `binarization`, `registration`, `processing`, `combination`                                | Session |
| `forging`         | `multiday_discovery`, `multiday_extraction`, `session_data_assembly`                       | Dataset |
| `manifest`        | `manifest_generation`                                                                      | Project |

The `microcontroller_data_extraction`, `camera_timestamp_extraction`, and two-photon stages are owned by this library's
upstream dependencies. This library resolves the inputs for those stages, calls their job bindings in-process, and
reuses each library's own exported job-name constant, so the identifiers recorded by a tracker stay aligned with the
library that produced the work. The remaining stages are implemented here.

A stage resolves its own job universe from the acquisition data. The video pipeline reads the camera manifest, so it
declares one timestamp job and one motion-energy job per registered camera regardless of which archives happen to be on
disk. The microcontroller pipeline declares one parse job for every module used by the processed session that has an
assigned parser. The forging pipeline declares one assembly job per session in the dataset. Because every pipeline
resolves its universe this way, a completed tracker already accounts for every source recorded by the session.

### Jobs, Trackers, and Batches

Work reaches a host as a **job**, and every pipeline models its jobs the same way.

1. **Plan.** `slf plan` reads a unit's acquisition data, registers every job that unit is able to run on that unit's
   processing tracker, and records each job's cores, memory, and upstream jobs into a per-unit `job_plan.yaml`. A unit
   runs only the jobs registered on its tracker.
2. **Prepare.** Preparation joins the tracker state to those records into one descriptor per job, and registers the
   result under a batch identifier. A job counts as blocked when preparation can neither queue its upstream stage nor
   confirm that the stage already succeeded.
3. **Execute.** Dispatch runs the prepared batch on one of two backends. The local engine admits jobs against a budget
   of cores and memory, then dispatches them onto a shared process pool in dependency order. The remote engine submits
   one SLURM allocation per job, each sized from its own estimate and sequenced through an `afterok` dependency.
4. **Close.** Closure snapshots the state recorded by a finished batch's jobs, while the batch is still tracked.

Each job holds one of four statuses on its tracker, and a rerun resolves only the work still outstanding:

| Status      | Meaning                                                 |
|-------------|---------------------------------------------------------|
| `SCHEDULED` | The job is registered and has not started               |
| `RUNNING`   | The job is executing, and the record names its executor |
| `SUCCEEDED` | The job completed and its output is on disk             |
| `FAILED`    | The job raised, and a reset returns it to `SCHEDULED`   |

Trackers are per-unit YAML files written under a file lock, so the state of a unit travels with the unit's data rather
than with the host that processed it. Project-level artifacts roll that state up for a submitting host that holds none
of the data:

| Artifact                               | Contents                                                      |
|----------------------------------------|---------------------------------------------------------------|
| `<project>/<project>_manifest.feather` | One row per session, snapshotting the project's state         |
| `<project>/<project>_jobs.feather`     | One row per tracked job of every per-session pipeline         |
| `<project>/<project>_plan.feather`     | Every plan cache under the project, collected into one table  |
| `<dataset>/dataset_state.feather`      | One dataset's forging job state, in a shippable form          |
| `<working directory>/remote_state/`    | This host's prepared batches and its remote submission ledger |

### Processed Data Structure

The acquisition side owns a session's `raw_data` directory, and this library writes only under `processed_data`:

```text
Session/
├── raw_data/                                     <- Written by sollertia-experiment
│   ├── ax_checksum.txt                           <- Verified and regenerated by the checksum pipeline
│   ├── checksum_processing_tracker.yaml
│   ├── behavior_data/                            <- The runtime and microcontroller DataLogger archives
│   └── camera_data/                              <- The camera recordings and their log archives
└── processed_data/
    ├── job_plan.yaml                             <- Every job's cores, memory, and upstream jobs
    ├── runtime_data/
    │   ├── runtime_processing_tracker.yaml
    │   └── ...                                   <- The system's parsed runtime state and trial tables
    ├── microcontroller_data/
    │   ├── microcontroller_processing_tracker.yaml
    │   ├── extraction_configuration.yaml         <- The per-controller filter the extraction stage runs
    │   ├── controller_{id}_module_{type}_{id}.feather
    │   └── ...                                   <- One parsed table per module, named by the system
    ├── video_data/
    │   ├── video_processing_tracker.yaml
    │   ├── camera_{source_id}_timestamps.feather <- One per camera, written by the extraction stage
    │   ├── {camera_name}_timestamps.feather      <- The manifest-named hardlink the rename stage publishes
    │   ├── {camera_name}_energy.feather          <- One per camera, written by the motion-energy stage
    │   └── ...                                   <- Whatever the system's tracking pass writes
    └── cindra/                                   <- Written by cindra's single-recording pipeline
        ├── single_recording_tracker.yaml
        └── ...                                   <- The registered stacks, detected ROIs, and traces
```

The checksum tracker sits under the acquired data, because that pipeline verifies the acquired data in place. Every
other pipeline records beside the output it produces. The elided entries are named by the acquisition system's donated
parsers and workers, so their filenames and column schemas belong to that system rather than to the pipeline. The
agnostic entries are stable across systems, and every camera table is positional, carrying one row per acquired frame.

### Forged Datasets

A dataset aggregates the processed sessions of one session type, recorded by one acquisition system, across many animals
of one project, and it lives under the project root beside the animal directories. Defining a dataset admits only the
sessions whose acquisition system reports a success for every pipeline required by their session type, and it bakes that
system's column meanings into the dataset at definition time:

```text
Project/
└── Dataset/                          <- A sibling of the project's animal directories
    ├── dataset.yaml                  <- The dataset marker, which names its project, session type, and system
    ├── data_descriptions.feather     <- Every column the system's assembler can emit, with its meaning
    ├── dataset_state.feather         <- The dataset's forging job state, written by 'slf dataset-state'
    ├── forging_tracker.yaml
    ├── job_plan.yaml
    └── Animal/
        ├── surgery_metadata.yaml
        ├── multi_recording_configuration.yaml   <- Written for an animal the system tracks across recordings
        └── Session/
            ├── data.feather          <- The assembled per-session data the analysis layer reads
            └── ...                   <- The session's acquisition snapshots, re-exported beside its data
```

The forging pipeline runs the cross-recording stages first, once per animal whose system resolves a multi-recording
configuration, then assembles one `data.feather` per session. Extending an existing dataset materializes only the
animals introduced by the call, together with the animals it names for recreation, so an animal already configured keeps
its existing plan. A large project therefore forges in passes, while part of its source data lives elsewhere.

### CLI Commands

This library provides the `slf` CLI. Every command infers the acquisition system from the data it opens, so no command
takes a system selector:

| Command                   | Description                                                                              |
|---------------------------|------------------------------------------------------------------------------------------|
| `plan session`            | Records what every processing job of each named session will cost                        |
| `plan dataset`            | Records what every forging job of each named dataset will cost                           |
| `plan project`            | Projects every plan cache under the project into one table at the project root           |
| `process video`           | Extracts camera frame timestamps, processes pose predictions, and measures motion energy |
| `process microcontroller` | Extracts the microcontroller log archives and parses them into behavior feathers         |
| `process runtime`         | Decodes the acquisition runtime log archive into the session's runtime behavior feathers |
| `process two-photon`      | Runs the single-recording two-photon (calcium-imaging) processing pipeline for a session |
| `forge`                   | Forges a dataset by assembling per-session data from a project's processed sessions      |
| `checksum`                | Resolves the data integrity checksum for the target session's 'raw_data' directory       |
| `manifest create`         | Creates the .feather file capturing the snapshot of the target project's state           |
| `manifest print`          | Prints the requested data from the project's manifest as a formatted table               |
| `dataset-state`           | Snapshots each named dataset's forging job state into a shippable table                  |
| `reset`                   | Returns tracked jobs of the named units to the scheduled state                           |
| `clean`                   | Removes a pipeline's output and processing tracker for the named units                   |
| `server configure`        | Creates the remote compute server configuration file in the working directory            |
| `server print`            | Displays the remote server's SLURM queue status or job data as a formatted table         |
| `server discover`         | Discovers and prints the sessions stored under the project's directory on the server     |
| `mcp`                     | Starts the agentic Model Context Protocol server using the requested transport           |
| `omp`                     | Links the OpenMP runtime Numba loads on macOS into a directory the loader searches       |

Use `slf --help` or `slf SUBCOMMAND --help` for detailed usage information.

Each `process` subcommand accepts a job identifier. Without one, it runs every job that remains outstanding for the
session on this host, and with one it runs exactly the job named by that identifier. That is how a scheduler drives
cross-job parallelism, by dispatching each identifier as its own allocation.

***Note,*** on macOS the Numba threading layer resolves its OpenMP runtime from the dynamic loader's default search
path alone. Run `slf omp` once per host to report what it would link, and `slf omp -y` through `sudo` to create the
link. Every pipeline that dispatches a parallel worker pool verifies the runtime before it starts a job.

### Configuring Server Access

To access the remote compute server, first author the server configuration. The `slf server configure` command creates
the configuration and stores it as the 'server_configuration.yaml' file, under the working directory of the Sollertia
platform. It records the username, the password, the host, the absolute path to the server's data root, and the name of
the shared conda environment that every remote job activates. That environment supplies the remote half of every job, so
it holds this library and the processing libraries it drives.

### Running a Remote Batch

The remote path runs the same prepared jobs that the local batch engine runs, so one job graph, one core table, and one
memory model serve both. A run has three steps, each exposed as a Model Context Protocol tool and backed by the `slf`
CLI:

1. **Prepare.** `prepare_batch_tool` with `host='remote'` refreshes the project's two tables on the server and pulls
   them. Planning reads each unit's acquisition data, writes the runnable jobs of that unit onto its processing tracker,
   and records each job's cores, memory, and upstream jobs. State generation turns those trackers into a table. A job
   absent from that table is a job that the unit is unable to run. The submitting host therefore resolves a batch from
   the table alone, without opening anything on the server. The two tables join on the job identifier and become one
   descriptor per job, registered under a batch identifier.
2. **Submit.** `execute_jobs_tool` renders each job as its own shell script, transfers it over SFTP, and submits it as
   its own SLURM allocation, sized from its own estimate, in dependency order. It takes no host of its own, because a
   batch runs where it was prepared. Each script carries an SBATCH directive block, activates the configured conda
   environment, and then runs the `slf` command for its job. Each job declares an `afterok` dependency on the
   allocations of the upstream jobs held by the batch, so the scheduler sequences the graph and the batch finishes with
   nothing running locally.
3. **Read.** `get_processing_status_tool` with `host='remote'` reports what the scheduler observed while a run is in
   flight. `generate_project_manifest_tool` and `generate_dataset_state_tool`, each with `host='remote'`, regenerate the
   project's manifest, job table, and dataset state on the server. Every read tool called with `host='remote'` then
   mirrors the project's artifacts into the working directory and reads them exactly as it reads a local project. The
   plan projection is mirrored as the server last wrote it, since replanning belongs to preparation.

Because the scheduler owns the run once it accepts the jobs, every accepted allocation is recorded in a submission
ledger at `<working directory>/remote_state/submission_ledger.yaml`, guarded by a file lock, as is every other shared
artifact this library keeps. The ledger keeps concurrent batches queryable, keeps a batch findable after this process
exits, and preserves the record of the accepted allocations when the scheduler rejects a later job of the same batch. A
batch leaves the ledger once every allocation it holds has reached a terminal state and its closure has snapshotted what
its jobs recorded. A batch that the query did not fully observe stays outstanding, and so does a batch whose closure
failed.

The run reports a job as blocked rather than submitted when it can neither queue that job's upstream stage nor confirm
that the stage already succeeded. A local batch treats the same job the same way.

***Critical!*** A remote job runs the same `slf` command that a local run would run, so test a pipeline on one session
locally before dispatching a project-wide batch. A command that works on the local machine behaves the same way on the
server.

### Recovering from Interruptions

Every pipeline records its progress on a per-unit tracker, so an interrupted run resumes rather than restarts. What to
run depends on what the interruption left behind:

| Situation                                              | Recovery                                                                                                          |
|--------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| A batch is still running and is no longer wanted       | `cancel_processing_tool` stops the local batch or the outstanding allocations                                     |
| Jobs report `FAILED` after a fixable cause             | `slf reset` or `reset_processing_jobs_tool` returns them to `SCHEDULED`                                           |
| A pipeline's output is suspect and must be rebuilt     | `slf clean` or `clean_processing_output_tool` removes the output and the tracker                                  |
| A run was interrupted, leaving jobs stuck in `RUNNING` | A local batch clears every job's record before it dispatches, and a remote batch adopts an allocation still alive |
| A remote batch settled while this host was offline     | The next status read closes the settled batch and retires it from the ledger                                      |

***Note,*** `slf clean` removes processed output. The checksum pipeline verifies the acquired data in place and owns no
output directory, so cleaning that pipeline removes its tracker alone. Cleaning discards work and is refused while a
local batch is running, so prefer a reset whenever the failure cause was external.

___

## API Documentation

See the [API documentation](https://sollertia-forgery-api-docs.netlify.app/) for the detailed description of the
methods and classes exposed by components of this library.

___

## AI-Assisted Development

The library is built for AI-assisted operation. It ships one MCP server, exposed through the `slf mcp` command, and a
set of Claude Code skills distributed through the [sollertia](https://github.com/Sun-Lab-NBB/sollertia) marketplace.
Human operators reach the library through the `slf` CLI documented above, and AI agents work through the MCP server
tools and the skills described here.

### MCP Server

The server exposes the planning, processing, forging, and project management pipelines. The sollertia-shared-assets,
ataraxis-video-system, ataraxis-communication-interface, and cindra libraries each serve their own assets through their
own MCP server, so this server leaves those tools to them.

Every tool names a filesystem path by what that path holds, and the same name means the same thing in every tool. A
`session_path` names one session's root directory, a `dataset_path` names one forged dataset's root, and a
`project_path` names a project root under the data root. Almost every tool takes a `host`, which is `local` for the
data on this machine and `remote` for the data on the compute server. The two server-configuration tools are always
local, and `execute_jobs_tool` takes no host at all, because a batch runs where it was prepared.

#### Starting the Server

Start the MCP server using the CLI:

```bash
slf mcp
```

The `-t/--transport` option selects the transport. The default `stdio` serves a local agent client, while `sse` and
`streamable-http` serve the same tools over the network to a client that reaches a processing host.

#### Available Tools

The dataset tools compose and report forged datasets:

| Tool                          | Description                                                                      |
|-------------------------------|----------------------------------------------------------------------------------|
| `define_forging_dataset_tool` | Creates or extends a forged dataset hierarchy and its per-animal configurations  |
| `generate_dataset_state_tool` | Snapshots each named dataset's forging job state into a shippable feather file   |
| `read_dataset_state_tool`     | Reads a dataset's forging job state out of its stored snapshot                   |
| `list_project_datasets_tool`  | Lists a project's forged datasets and reports which of them hold a given session |

The management tools read and regenerate a project's state artifacts:

| Tool                             | Description                                                                      |
|----------------------------------|----------------------------------------------------------------------------------|
| `generate_project_manifest_tool` | Regenerates the target project's manifest and job artifacts                      |
| `read_project_manifest_tool`     | Reads a project's sessions out of its stored manifest                            |
| `read_project_jobs_tool`         | Reads a project's tracked jobs out of its stored job artifact                    |
| `get_manifest_status_tool`       | Reports the outcome of the last run that generated the project's state artifacts |

The planning tools record the cost of every job and collect those records into one table:

| Tool                         | Description                                                                         |
|------------------------------|-------------------------------------------------------------------------------------|
| `plan_session_jobs_tool`     | Records the cost of every processing job in one or more sessions                    |
| `plan_dataset_jobs_tool`     | Records the cost of every forging job in one or more datasets                       |
| `generate_project_plan_tool` | Collects every plan cache under a project into one table at the project root        |
| `read_project_plan_tool`     | Reads the cores and memory planned for a project's jobs out of that collected table |

The processing tools resolve, dispatch, and recover a batch:

| Tool                           | Description                                                                        |
|--------------------------------|------------------------------------------------------------------------------------|
| `prepare_batch_tool`           | Resolves a pipeline's dispatchable jobs for one or more units                      |
| `inspect_job_resources_tool`   | Reports the cores and memory a pipeline's outstanding jobs require                 |
| `execute_jobs_tool`            | Dispatches prepared batches onto this machine's pool or the server's scheduler     |
| `get_processing_status_tool`   | Reports the live status of the active batch                                        |
| `cancel_processing_tool`       | Cancels the active local batch, or the outstanding remote allocations              |
| `reset_processing_jobs_tool`   | Resets tracked jobs to SCHEDULED across one or more units                          |
| `clean_processing_output_tool` | Removes a pipeline's output files and its processing tracker for one or more units |

The server tools author the remote compute server's credentials:

| Tool                              | Description                                                                |
|-----------------------------------|----------------------------------------------------------------------------|
| `read_server_configuration_tool`  | Loads the server configuration from the working directory, password masked |
| `write_server_configuration_tool` | Creates or replaces the server configuration YAML in the working directory |

### Skills

The **forging** plugin ships the skills that orchestrate the tools above. The skills are driven by AI agents rather than
invoked directly by operators, and they group into the workflows they serve:

| Skill                           | Purpose                                                                           |
|---------------------------------|-----------------------------------------------------------------------------------|
| `behavior-processing`           | Orchestrate batch behavior processing across confirmed sessions                   |
| `checksum-verification`         | Orchestrate batch checksum verification and regeneration                          |
| `dataset-definition`            | Compose forged datasets and report their forging job state                        |
| `dataset-forging`               | Orchestrate batch dataset forging across a dataset                                |
| `behavior-input-format`         | Document the raw artifacts the behavior-processing pipelines consume              |
| `dataset-forging-input-format`  | Document the inputs the forging pipeline reads                                    |
| `behavior-results`              | Document the behavior-processing outputs and how to verify them                   |
| `dataset-forging-results`       | Document the forged dataset outputs and how to verify them                        |
| `project-manifest`              | Document the project manifest and the tools that read and generate it             |
| `camera-timestamp-extraction`   | Document the manifest-driven stage that extracts camera timestamps                |
| `microcontroller-primitives`    | Document the agnostic microcontroller parsing primitives                          |
| `data-processing-design`        | Document the design pattern that pairs agnostic workers with per-system donations |
| `server-configuration`          | Author and modify the remote compute server configuration                         |
| `forging-mcp-environment-setup` | Diagnose and resolve MCP server connectivity issues                               |

#### Client Registration

The **forging** plugin of the [sollertia](https://github.com/Sun-Lab-NBB/sollertia) marketplace distributes this
library's Claude Code skills and the registration for its MCP server. Installing that plugin registers the MCP server
with compatible clients and makes every associated skill available.

Contributors additionally install the **automation** plugin from the [ataraxis](https://github.com/Sun-Lab-NBB/ataraxis)
marketplace. That plugin provides the skills that enforce the Sollertia coding conventions of this repository, together
with the codebase exploration and audit tools.

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
| `coverage`     | Aggregates test coverage and applies the 100% coverage gate |
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

### Adding a New Acquisition System

A new acquisition system reaches the agnostic pipelines by donating one entry to each registry in `registries.py`.
The Mesoscope-VR package is the reference for both the module split and the contents.

**Step 1: Add the enumeration member upstream**

`AcquisitionSystems` is owned by sollertia-shared-assets. Add the member there first, along with the system's hardware
state, experiment configuration, and raw-data dataclasses. In that library's `SYSTEM_SESSION_TYPES`, pair the member
with the session types the system records. Until the steps below are complete, importing any pipeline in this library
raises, because the coverage check refuses a partially wired system.

**Step 2: Create the system package**

Create a `<system>/` package alongside `mesoscope_vr/`, and export every donated symbol from its `__init__.py`. The
package supplies:

1. A parser for every microcontroller hardware module the system records, plus the accessor that returns the event codes
   read by each parser and the accessor that returns the modules a given session configured for use.
2. The identifier of the runtime log source, together with the parser that interprets the decoded runtime payloads.
3. The video-tracking function that reads the session's pose predictions and writes its tracking outputs. That function
   no-ops for a system that performs no tracking.
4. The locator that resolves the session's raw two-photon imaging directory, and the resolvers that build the
   single-recording and multi-recording cindra configurations.
5. The per-session forging assembler, the mapping from each emitted column to that column's meaning, the admission
   policy that names the pipelines required for each session type, and the session types the system tracks across
   recordings.

A system that produces none of a given data class still donates an entry for that class. A no-op tracking function and a
multi-recording resolver that returns `None` are the donations a system makes for a class it never produces, so the
coverage check stays a check on wiring rather than on capability.

***Critical!*** A system package must never import an agnostic category package. The category pipelines import
`registries.py` and `registries.py` imports every system package, so the reverse import is a circular import rather
than a style preference.

**Step 3: Register the donations**

In `registries.py`, import the new symbols and add the system's entry to every registry. Then import the library. The
coverage check raises a `RuntimeError` that names each registry still missing the system, each parseable module without
event codes, and each declared session type absent from the upstream pairing for that system. The error is the remaining
checklist.

Nothing under `orchestration/` or `interfaces/` changes. The CLI and the MCP tools resolve the system from the session
or dataset they open, so a fully wired system reaches every command already exposed.

**Step 4: Cover the new package**

Add the system's `automodule` block to `docs/source/api.rst`, add its tests under `tests/`, and add the new names to the
donor registry list the registry-coverage test checks. The suite gates on 100% statement coverage.

**Step 5: Update the sibling libraries**

Coordinate with sollertia-experiment, which acquires the sessions the new system records, and confirm that every
artifact a pipeline reads is an artifact the acquisition runtime writes. That library carries no import-time coverage
check of its own, so a system left unwired in it surfaces only when an operator runs its configure command.

### Adding a New Session Type

A session type is owned by sollertia-shared-assets and reaches this library through the recording system's package.

1. Add the `SessionTypes` member and its descriptor dataclass in sollertia-shared-assets. In `SYSTEM_SESSION_TYPES`,
   pair the member with every acquisition system that records it.
2. Add the type to the recording system's admission policy, which names the pipelines a session of that type completes
   before it joins a forged dataset. A type left out of the policy joins no dataset, which is the deliberate opt-out
   rather than an omission.
3. Route the type in the system's forging assembler and write the branch that assembles its data. An unrouted type
   raises when the forging pipeline reaches it.
4. Add the type to the system's multi-recording session types when the system tracks its animals across recordings, and
   widen the system's own multi-recording configuration resolver to match. The import check validates the declared types
   against the upstream pairing alone, so it never catches a resolver that still declines the new type.
5. Give every new dataset column both its column member and its description entry. The system's metadata module raises
   at import when a column carries no description, because a dataset publishes that mapping alongside its data.

### Adding a New Processing Stage

A stage inside an existing pipeline rides the routing that pipeline already has, so it needs no dispatch, CLI, or MCP
change.

1. Define the stage's job name in the pipeline module, or reuse the constant the upstream library exports when that
   library's job binding backs the stage, and export the name from the category package.
2. Emit the job from the pipeline's job discovery, into the universe every time and into the possible set when the
   session's data supports it. Choose the specifier convention next. A stage that runs once per source carries the
   source identifier, and a stage that runs once per session carries an empty specifier.
3. Order the job in the pipeline's prerequisite mapping, where an independent stage maps to no upstream job.
4. Execute the stage inside the pipeline entry point, under the branch that the job identifier selects.
5. Declare the cores one of its jobs occupies. The resolver refuses to admit a job type that declares none, and says so.
6. Add its sizing model and route the job type to that model. A job with no resolvable resource figures is a hard error
   rather than a job that runs at a default size.
7. Declare a concurrency ceiling only when the job type's throughput plateaus before its cores do, and a reservation
   only when the type should hand capacity back. The core and memory budgets alone bound a type named in neither.

### Adding a New Processing Pipeline

A pipeline that reads a new class of acquired data is agnostic, so it becomes a category package rather than a system
donation.

1. Add the tracker filename to `ProcessingTrackers` and the output directory to `ProcessedData` in
   sollertia-shared-assets, because a per-session pipeline records beside the output it produces.
2. Add the `ProcessingPipelines` member and, for a per-session pipeline, its tracker location. The session pipeline
   listing derives from that mapping, so the two never disagree about the pipelines a session carries.
3. Create the category package. Its `__init__.py` exports the pipeline entry point, every stage job name, and the job
   discovery and prerequisite callables the orchestration layer binds.
4. Add the pipeline to the batch pipeline set and give it a dispatch entry that supplies its unit loader, job discovery,
   batch worker, prerequisites, tracker path, output path, unit name, sizing pass, remote command renderer, and any
   priming hook. An import-time check fails on either half alone.
5. Give every job type the pipeline resolves its core allocation and its sizing model, following [Adding a New
   Processing Stage](#adding-a-new-processing-stage).
6. Add the `slf process` subcommand and the MCP surface, then add the pipeline to the admission policy of every system
   whose sessions complete it before forging.

### Adding an MCP Tool

Place the tool in the `*_tools.py` module under `src/sollertia_forgery/interfaces/` that owns its domain, decorate it
with `@mcp.tool()`, and return through the shared `ok_response` and `error_response` helpers. Document the response key
shape in the tool's `Returns` docstring section, since it is part of the public contract. The server discovers tool
modules by their filename suffix, so a new module needs no edit to the server itself. Add a new module to the coverage
omit list in `pyproject.toml`, because tool modules reach infrastructure that only a live MCP session supplies.

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
