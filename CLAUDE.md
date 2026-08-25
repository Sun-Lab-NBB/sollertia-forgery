# Claude Code Instructions

## Session start behavior

At the beginning of each coding session, before making any code changes, you MUST build a comprehensive understanding of
the codebase by invoking the `/explore-codebase` skill.

## Style guide compliance

You MUST invoke the appropriate style skill before performing ANY of the following tasks:

| Task                                          | Skill to invoke    |
|-----------------------------------------------|--------------------|
| Writing or modifying Python code              | `/python-style`    |
| Writing or modifying README files             | `/readme-style`    |
| Writing or modifying skill files or this file | `/skill-design`    |
| Writing or modifying pyproject.toml           | `/pyproject-style` |
| Writing or modifying tox.ini                  | `/tox-config`      |
| Writing or modifying Sphinx docs files        | `/api-docs`        |
| Creating or verifying project structure       | `/project-layout`  |
| Committing local changes                      | `/commit`          |

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

4. **Proceed with correct source**: Use whichever version the user selects as the authoritative reference for API usage,
   patterns, and documentation.

## Available skills

**Forging plugin skills** (`sollertia` marketplace, `forging` plugin):

| Skill                            | Description                                                            |
|----------------------------------|------------------------------------------------------------------------|
| `/behavior-input-format`         | Documents the raw artifacts the behavior-processing pipelines consume  |
| `/behavior-processing`           | Orchestrates batch behavior processing through the MCP server          |
| `/behavior-results`              | Documents the behavior-processing outputs and how to verify them       |
| `/camera-timestamp-extraction`   | Documents the manifest-driven camera timestamp extraction stage        |
| `/checksum-verification`         | Orchestrates batch checksum verification and regeneration              |
| `/data-processing-design`        | Documents the agnostic-worker and per-system-donation design pattern   |
| `/dataset-forging`               | Orchestrates batch dataset forging through the MCP server              |
| `/dataset-forging-input-format`  | Documents the inputs the forging pipeline reads                        |
| `/dataset-forging-results`       | Documents the forged dataset outputs and how to verify them            |
| `/datasets`                      | Discovers, reads, and writes the dataset-level records                 |
| `/forging-mcp-environment-setup` | Diagnoses and resolves MCP server connectivity issues                  |
| `/microcontroller-primitives`    | Documents the agnostic microcontroller parsing primitives              |
| `/project-manifest`              | Documents the project manifest and the tools that read and generate it |
| `/server-configuration`          | Authors and modifies the remote compute server configuration           |

**Automation plugin skills** (`ataraxis` marketplace, `automation` plugin) provide the style guides listed above,
`/explore-codebase`, `/explore-dependencies`, and the `/audit-*` family.

## MCP server

The library exposes an MCP server through the `slf mcp` command, defined in
`src/sollertia_forgery/interfaces/mcp_server.py`. Tool modules register their tools purely as an import side effect,
and the server discovers them by the `_tools` filename suffix under `src/sollertia_forgery/interfaces/`.

**When adding an MCP tool**, place it in the `*_tools.py` module that owns its domain, decorate it with `@mcp.tool()`
from `.mcp_instance`, and give it a `Returns` section naming the response keys in prose. Add the module to the
`[tool.coverage.run] omit` list in `pyproject.toml` if it is a new tool module, because tool modules reach
infrastructure only a live MCP session supplies.

## Downstream library integration

This library reads the records `sollertia-shared-assets` defines and drives the pipelines `ataraxis-video-system`,
`ataraxis-communication-interface`, and `cindra` publish. It never references the downstream analysis repository, which
owns its own artifact names.

## Distribution model

The package ships to PyPI as `sollertia-forgery` and installs the `slf` CLI. Its Claude Code skills and its MCP server
registration ship separately, through the `forging` plugin of the [sollertia](https://github.com/Sun-Lab-NBB/sollertia)
marketplace. An agent asked to add or change a skill edits that repository rather than this one.

## Project context

### Key areas

| Path                                   | Contents                                                           |
|----------------------------------------|--------------------------------------------------------------------|
| `src/sollertia_forgery/registries.py`  | The acquisition-system dispatch registries and their import checks |
| `src/sollertia_forgery/mesoscope_vr/`  | The Mesoscope-VR system's donated parsers, resolvers, and workers  |
| `src/sollertia_forgery/orchestration/` | Batch preparation, dispatch, execution, and closure on both hosts  |
| `src/sollertia_forgery/interfaces/`    | The `slf` Click CLI and the MCP tool modules                       |
| `src/sollertia_forgery/shared_assets/` | The agnostic substrate every category package draws on             |

### Architecture

The processing packages are **system-agnostic**. A category package such as `video/`, `microcontrollers/`, `runtime/`,
`two_photon/`, or `forging/` owns a pipeline that runs the same way for every acquisition system. Everything a specific
acquisition system contributes lives in a per-system package such as `mesoscope_vr/`, and reaches the agnostic
pipelines through `registries.py`.

`registries.py` maps each `AcquisitionSystems` member to the workers, parsers, resolvers, and locators that system
donates, and runs an import-time coverage check that refuses a partially wired system. The dependency runs one way,
because a per-system package never imports an agnostic category package.

**When adding an acquisition system**, add its package, register every donation in `registries.py`, and let the
coverage check tell you what is still unwired.

### Code standards

- Every category package `__init__.py` exports its pipeline entry point, every stage job name, and the job
  discovery and prerequisite callables the orchestration layer binds. A name reaches that list only when a package
  outside the defining one imports it, and a symbol no other module reaches at all carries the underscore.
- A stage backed by a library `execute_job` binding reuses that library's own exported job-name constant rather than a
  local string, so the tracker identifiers stay aligned with the library's.
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
