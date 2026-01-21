# Claude Code Instructions

## Session Start Behavior

At the beginning of each coding session, before making any code changes, you should build a comprehensive
understanding of the codebase by invoking the `/explore-codebase` skill.

This ensures you:
- Understand the project architecture before modifying code
- Follow existing patterns and conventions
- Don't introduce inconsistencies or break integrations

## Style Guide Requirements

You MUST invoke `/sun-lab-style` and read the appropriate guide before performing ANY of the following tasks:

| Task                              | Guide to Read      |
|-----------------------------------|--------------------|
| Writing or modifying Python code  | PYTHON_STYLE.md    |
| Writing or modifying README files | README_STYLE.md    |
| Writing git commit messages       | COMMIT_STYLE.md    |
| Writing or modifying skill files  | SKILL_STYLE.md     |

This is non-negotiable. The skill contains verification checklists that you MUST complete before submitting any work.
Failure to read the appropriate guide results in style violations.

## Cross-Referenced Library Verification

Sun Lab projects often depend on other `ataraxis-*` or `sl-*` libraries. These libraries may be stored locally in the
same parent directory as this project (`/home/cyberaxolotl/Desktop/GitHubRepos/`).

**Before writing code that interacts with a cross-referenced library, you MUST:**

1. **Check for local version**: Look for the library in the parent directory (e.g., `../sl-shared-assets/`,
   `../sl-experiment/`).

2. **Compare versions**: If a local copy exists, compare its version against the latest release or main branch on
   GitHub:
   - Read the local `pyproject.toml` to get the current version
   - Use `gh api repos/Sun-Lab-NBB/{repo-name}/releases/latest` to check the latest release
   - Alternatively, check the main branch version on GitHub

3. **Handle version mismatches**: If the local version differs from the latest release or main branch, notify the user
   with the following options:
   - **Use online version**: Fetch documentation and API details from the GitHub repository
   - **Update local copy**: The user will pull the latest changes locally before proceeding

4. **Proceed with correct source**: Use whichever version the user selects as the authoritative reference for API
   usage, patterns, and documentation.

**Why this matters**: Skills and documentation may reference outdated APIs. Always verify against the actual library
state to prevent integration errors.

## Available Skills

| Skill               | Description                                                      |
|---------------------|------------------------------------------------------------------|
| `/explore-codebase` | Perform in-depth codebase exploration at session start           |
| `/sun-lab-style`    | Apply Sun Lab coding conventions (REQUIRED for all code changes) |

## Project Context

This is **sl-forgery**, a Python library for scientific data processing in the Sun Lab at Cornell University. The
library processes raw data acquired by sl-experiment and produces analysis-ready datasets.

### Key Areas

| Directory                     | Purpose                                       |
|-------------------------------|-----------------------------------------------|
| `src/sl_forgery/`             | Main library source code                      |
| `src/sl_forgery/core/`        | Core data processing and forging logic        |
| `src/sl_forgery/utilities/`   | Shared utility functions                      |

### Architecture

- Data processing pipelines for raw experimental data
- Integration with sl-experiment output formats
- Dataset generation for downstream analysis

### Code Standards

- MyPy strict mode with full type annotations
- Google-style docstrings
- 120 character line limit
- See `/sun-lab-style` for complete conventions

### Workflow Guidance

**Modifying data processing pipelines:**

1. Review existing pipeline structure in the relevant module
2. Follow existing patterns for data loading, processing, and output
3. Use configuration dataclasses for pipeline parameters
4. Ensure integration with sl-experiment data formats

**Modifying sl-shared-assets (configuration dataclasses):**

Changes to system configuration require updates in `sl-shared-assets` (`../sl-shared-assets/`). Configuration
dataclasses define the structure of experiment configurations and processing parameters.
