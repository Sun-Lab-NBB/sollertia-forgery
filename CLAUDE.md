# Claude Code Instructions

## Session start behavior

At the beginning of each coding session, before making any code changes, you MUST build a comprehensive understanding of
the codebase by invoking the `automation:explore-codebase` skill.

## Style guide compliance

You MUST invoke the appropriate style skill before performing ANY of the following tasks:

| Task                                          | Skill to invoke              |
|-----------------------------------------------|------------------------------|
| Writing or modifying Python code              | `automation:python-style`    |
| Writing or modifying README files             | `automation:readme-style`    |
| Writing or modifying skill files or this file | `automation:skill-design`    |
| Writing or modifying pyproject.toml           | `automation:pyproject-style` |
| Writing or modifying tox.ini                  | `automation:tox-config`      |
| Writing or modifying Sphinx docs files        | `automation:api-docs`        |
| Creating or verifying project structure       | `automation:project-layout`  |
| Committing local changes                      | `automation:commit`          |

Each skill contains a verification checklist that you MUST complete before submitting any work.

## Cross-referenced library verification

The `sollertia-forgery` package depends on several `ataraxis-*` libraries, on `sollertia-shared-assets`, and on
`cindra`. These libraries may be stored locally in the same parent directory as this project, reachable as `../`
from the repository root.

**Before writing code that interacts with a cross-referenced library, you MUST:**

1. **Check for local version**: Look for the library in the parent directory (e.g., `../ataraxis-video-system/`,
   `../ataraxis-communication-interface/`, `../sollertia-shared-assets/`, `../cindra/`).

2. **Compare versions**: If a local copy exists, compare its version against the latest release or main branch on
   GitHub:
   - Read the local `pyproject.toml` to get the current version
   - Use `gh api repos/Sun-Lab-NBB/{repo-name}/releases/latest` to check the latest release
   - Alternatively, check the main branch version on GitHub

3. **Handle version mismatches**: If the local version differs from the latest release or main branch, notify the user
   with the following options:
   - **Use online version**: Fetch documentation and API details from the GitHub repository
   - **Update local copy**: The user will pull the latest changes locally before proceeding

4. **Proceed with correct source**: Use whichever version the user selects, and treat that version as the
   authoritative reference for API usage, patterns, and documentation.

## Available skills

**Forging plugin skills** (`sollertia` marketplace, `forging` plugin):

| Skill                                   | Description                                                               |
|-----------------------------------------|---------------------------------------------------------------------------|
| `forging:batch-processing`              | Orchestrates batch processing across all six batch pipelines              |
| `forging:cli-reference`                 | Documents every `slf` command, option, and its MCP counterpart            |
| `forging:data-processing-design`        | Documents the agnostic-worker and per-system-donation design pattern      |
| `forging:dataset-definition`            | Composes forged dataset hierarchies and reports their forging job state   |
| `forging:dataset-forging`               | Documents how the forging pipeline differs from the per-session pipelines |
| `forging:forging-mcp-environment-setup` | Diagnoses MCP connectivity and owns the shared response envelope          |
| `forging:job-planning`                  | Sizes every runnable job and records the planned cores and memory         |
| `forging:library-extension`             | Owns the extension path for systems, stages, pipelines, and MCP tools     |
| `forging:pipeline`                      | Orders the end-to-end processing lifecycle and its local-remote split     |
| `forging:processing-input-format`       | Documents the on-disk inputs each batch pipeline requires                 |
| `forging:processing-results`            | Documents what each pipeline writes and how to verify it                  |
| `forging:project-state`                 | Documents the session manifest and the job table published beside it      |
| `forging:remote-execution`              | Runs work on the configured SLURM compute server through `slf mcp`        |
| `forging:server-configuration`          | Authors the `ServerConfiguration` YAML authorizing SSH and SLURM access   |

**Mesoscope plugin skills** (`sollertia` marketplace, `mesoscope` plugin) document the Mesoscope-VR donations under
`src/sollertia_forgery/mesoscope_vr/`, through `mesoscope:mesoscope-vr-module-parsing`,
`mesoscope:mesoscope-vr-trial-decomposition`, `mesoscope:mesoscope-vr-fluorescence-alignment`,
`mesoscope:mesoscope-vr-imaging-configuration`, `mesoscope:mesoscope-vr-video-tracking`,
`mesoscope:mesoscope-vr-dataset-assembly`, and `mesoscope:mesoscope-vr-processing-schema`, which owns the
`BehaviorDataFiles`, `VideoDataFiles`, and `DatasetColumn` rosters in `mesoscope_vr/metadata.py`.

**Automation plugin skills** (`ataraxis` marketplace, `automation` plugin) provide the style guides listed above,
`automation:explore-codebase`, `automation:explore-dependencies`, the `automation:audit-*` family, `automation:pr`, and
`automation:release`.

## MCP server

The library exposes an MCP server through the `slf mcp` command. `interfaces/entry_points.py` selects the transport with
`-t/--transport`, defaulting to `stdio` and also accepting `sse` and `streamable-http` for a network client, and
disables the console on the `stdio` path, because the pipelines echo progress to the same stream that carries the
JSON-RPC messages. `interfaces/mcp_server.py` runs the server, and its import discovers every `*_tools.py` module under
`src/sollertia_forgery/interfaces/`, each of which registers its tools purely as an import side effect.

**When adding an MCP tool**, place it in the `*_tools.py` module that owns its domain and decorate it with `@mcp.tool()`
from `.mcp_instance`. Return through the `ok_response` and `error_response` helpers in `.responses`, and give the tool a
`Returns` section that names the response keys in prose. Add a new tool module to the `[tool.coverage.run] omit` list in
`pyproject.toml`, because tool modules reach infrastructure that only a live MCP session supplies.

## Downstream library integration

This library reads the records defined by `sollertia-shared-assets` and drives the pipelines published by
`ataraxis-video-system`, `ataraxis-communication-interface`, and `cindra`. It never references the downstream analysis
repository, which owns its own artifact names.

## Companion library synchronization

`sollertia-shared-assets` owns the vocabulary this library uses for dispatch, and `sollertia-experiment` records the
data this library processes, so an out-of-domain extension usually lands as one change in each of the three
repositories.

- `AcquisitionSystems`, `SessionTypes`, `SYSTEM_SESSION_TYPES`, `ProcessingTrackers`, `ProcessedData`, `SessionData`,
  and `DatasetData` all live in `sollertia-shared-assets`. A new acquisition system or session type gains its
  enumeration member there first through `assets:library-extension`, and this library never mints one locally.
- Adding an `AcquisitionSystems` member upstream breaks every import path that reaches `registries.py`, and that break
  holds until this library wires the new system. Those paths cover every processing pipeline. Read the RuntimeError
  raised by `_assert_registry_coverage` as the remaining extension checklist.
- `_FORGING_ADMISSION_REGISTRY` and `_MULTI_RECORDING_SESSION_TYPE_REGISTRY` validate their session types against
  `SYSTEM_SESSION_TYPES`, so each type this library assigns to a system must appear under that same system in the
  upstream registry.
- A new per-session pipeline needs an upstream tracker filename in `ProcessingTrackers` and an upstream output directory
  in `ProcessedData`, before `shared_assets/pipelines.py` is able to resolve the tracker location.
- `sollertia-experiment` has to acquire the data before anything here can process it, so confirm that the acquisition
  side creates sessions of a new session type and writes the artifacts a new pipeline reads. `experiment:pipeline` owns
  that acquisition lifecycle, `experiment:data-management` owns the preprocessing that materializes `raw_data`, and
  `mesoscope:mesoscope-vr-runtime` owns the Mesoscope-VR runtime that records the session.

## Distribution model

The package ships to PyPI as `sollertia-forgery` and installs the `slf` CLI. Its Claude Code skills ship separately,
through the [sollertia](https://github.com/Sun-Lab-NBB/sollertia) marketplace, in its `forging` and `mesoscope` plugins,
and the `forging` plugin alone carries the MCP server registration. An agent asked to add or change a skill edits that
repository rather than this one.

## Project context

This is **sollertia-forgery**, the data processing counterpart to `sollertia-experiment`. It turns the raw acquisition
logs of a recorded session into per-session data tables, verifies the integrity of the acquired data, and forges the
processed sessions into the multi-session datasets consumed by a downstream analysis repository.

### Key areas

| Path                                   | Contents                                                                 |
|----------------------------------------|--------------------------------------------------------------------------|
| `src/sollertia_forgery/registries.py`  | The acquisition-system dispatch registries and their coverage check      |
| `src/sollertia_forgery/mesoscope_vr/`  | The Mesoscope-VR system's donated parsers, resolvers, and workers        |
| `src/sollertia_forgery/orchestration/` | Planning, preparation, dispatch, execution, closure, and maintenance     |
| `src/sollertia_forgery/managing/`      | The checksum pipeline, the project manifest, and the job artifact        |
| `src/sollertia_forgery/server/`        | SLURM jobs, remote discovery, the SSH and SFTP transport, and its config |
| `src/sollertia_forgery/interfaces/`    | The `slf` Click CLI and the MCP tool modules                             |
| `src/sollertia_forgery/shared_assets/` | The agnostic substrate every category package draws on                   |

The `video/`, `microcontrollers/`, `runtime/`, `two_photon/`, and `forging/` category packages sit beside them, one
per pipeline.

### Architecture

The processing packages are **system-agnostic**. A category package such as `video/`, `microcontrollers/`, `runtime/`,
`two_photon/`, or `forging/` owns a pipeline that runs the same way for every acquisition system. Everything a specific
acquisition system contributes lives in a per-system package such as `mesoscope_vr/`, and reaches the agnostic
pipelines through `registries.py`.

`registries.py` maps each `AcquisitionSystems` member to the workers, parsers, resolvers, and locators donated by that
member, and runs an import-time coverage check that refuses a partially wired system. The dependency runs one way,
because a per-system package never imports an agnostic category package. A category pipeline imports `registries.py` and
`registries.py` imports every per-system package, so the reverse import is a genuine cycle rather than a style
preference.

Work reaches a host as a **job**, and every pipeline models its jobs the same way. `slf plan session` and
`slf plan dataset` read a unit's acquisition data, register on its processing tracker every job that unit is able to
run, and record each job's cores, memory, and upstream jobs. Preparation joins the tracker state to those records into
one descriptor per job under a batch identifier, and dispatch runs the batch either on this machine's process pool or as
one SLURM allocation per job. Each job holds a `ProcessingStatus` on the tracker, one of `SCHEDULED`, `RUNNING`,
`SUCCEEDED`, or `FAILED`. A rerun therefore resolves only the work still outstanding. The run reports a job as blocked
rather than dispatched when it can neither queue that job's upstream stage nor confirm that the stage already succeeded.

The public surface of the distribution is the `slf` CLI and the MCP server that CLI starts, so the top-level
`__init__.py` re-exports no library symbol and its `__all__` is empty. Adding a name to a public listing is a deliberate
API change rather than a convenience.

### Extension contracts

Every registry is private to `registries.py` and is reached through that module's `resolve_*` accessors, so a consuming
pipeline never indexes a registry directly. All eleven registries are the designed extension point, and a new
acquisition system supplies an entry in each.

| Registry                                 | Donation                                                         |
|------------------------------------------|------------------------------------------------------------------|
| `_MICROCONTROLLER_PARSER_REGISTRY`       | One `MicrocontrollerParser` per parsed hardware module           |
| `_MICROCONTROLLER_EVENT_CODE_REGISTRY`   | The event codes every parsed module reads                        |
| `_MICROCONTROLLER_ELIGIBILITY_REGISTRY`  | The modules a given session configured for use                   |
| `_RUNTIME_PARSER_REGISTRY`               | The source id of the runtime log, paired with its parser         |
| `_POSE_PREDICTION_REGISTRY`              | The locator for the externally-produced pose-prediction file     |
| `_VIDEO_TRACKING_REGISTRY`               | The system's whole video-tracking pass                           |
| `_TWO_PHOTON_DATA_REGISTRY`              | The locator for the raw two-photon imaging directory             |
| `_CINDRA_CONFIGURATION_REGISTRY`         | The single- and multi-recording cindra config resolvers          |
| `_FORGING_ASSEMBLY_REGISTRY`             | The per-session assembler and its column descriptions            |
| `_FORGING_ADMISSION_REGISTRY`            | The pipelines a session of each type completes to join a dataset |
| `_MULTI_RECORDING_SESSION_TYPE_REGISTRY` | The session types the system tracks across recordings            |

Every registry is keyed by `AcquisitionSystems`, except `_MICROCONTROLLER_PARSER_REGISTRY`, whose `(AcquisitionSystems,
module_type, module_id)` key lets a system register one parser per hardware module.

`_assert_registry_coverage()` runs when `registries.py` is imported and raises a `RuntimeError` that names the offending
members. It fires when a system is missing from any registry, when a parseable microcontroller module declares no event
codes, and when a declared session type is absent from the upstream `SYSTEM_SESSION_TYPES` pairing for that system.

**When adding an acquisition system**, add its package, register every donation in `registries.py`, and let the coverage
check tell you what is still unwired. Nothing under `orchestration/` or `interfaces/` changes, because every command
resolves the system from the session or dataset it opens.

**When adding a processing stage**, define or reuse its job name, emit it from the pipeline's job discovery, order it in
the prerequisite mapping, and give it both a `_JOB_CORE_ALLOCATIONS` entry and a sizing model in `footprints.py`. A job
type missing either one is a hard error rather than a job admitted at a default size.

**When adding a processing pipeline**, add its `ProcessingPipelines` member and tracker entry in
`shared_assets/pipelines.py`, its category package, and both its `BATCH_PIPELINES` membership and its
`PipelineDispatch` entry, which `_assert_dispatch_coverage` holds in step. Then add its CLI and MCP surfaces.

### Code standards

- Every category package `__init__.py` exports its pipeline entry point, every stage job name, and the job discovery and
  prerequisite callables the orchestration layer binds. A name reaches that list only when a package outside the
  defining one imports it, and a symbol that no other module reaches at all carries the underscore.
- A stage backed by a library `execute_job` binding reuses the job-name constant that library exports rather than a
  local string, so the tracker identifiers stay aligned with the library's own.
- The test suite covers 100% of the measured statements. Interface modules are excluded per module through the
  `[tool.coverage.run] omit` list rather than through a directory glob.
- A test that spawns a process pool or mutates process-wide state carries `@pytest.mark.xdist_group`, because the suite
  runs under `pytest-xdist` with `--dist loadgroup`.

### Development commands

| Command             | Effect                                                     |
|---------------------|------------------------------------------------------------|
| `tox -e lint`       | Runs ruff formatting, ruff linting, and mypy type checking |
| `tox -e py314-test` | Runs the test suite and writes its coverage data           |
| `tox -e coverage`   | Merges the coverage data and applies the 100% gate         |
| `tox -e docs`       | Builds the API documentation                               |
| `tox -e create`     | Creates the `slf_dev` mamba environment                    |
| `tox -e install`    | Installs the project into that environment                 |

Run `tox` with no argument to execute the full pipeline, and `tox --parallel` to overlap the tasks that allow it.
