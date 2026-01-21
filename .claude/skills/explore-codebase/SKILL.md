---
name: exploring-codebase
description: >-
  Performs in-depth codebase exploration at the start of a coding session. Builds comprehensive
  understanding of project structure, architecture, key components, and patterns. Use when starting
  a new session, when asked to understand or explore the codebase, when asked "what does this project
  do", when exploring unfamiliar code, or when the user asks about project structure or architecture.
---

# Codebase Exploration

Performs thorough codebase exploration to build deep understanding before coding work begins.

---

## Exploration Approach

Use the Task tool with `subagent_type: Explore` to investigate the codebase. Focus on understanding:

1. **Project purpose and structure** - README, documentation, directory layout
2. **Architecture** - Main components, how they interact, communication patterns
3. **Core code** - Key classes, data models, utilities
4. **Configuration** - How the project is configured and customized
5. **Dependencies** - External libraries and integrations
6. **Patterns and conventions** - Coding style, naming conventions, design patterns

Adapt exploration depth based on project size and complexity. For small projects, a quick overview
suffices. For large projects, explore systematically.

---

## Guiding Questions

Answer these questions during exploration:

### Architecture
- What is the main entry point or controller?
- How do components communicate (IPC, APIs, events)?
- What external systems does this integrate with?

### Patterns
- What naming conventions are used?
- What design patterns appear (factories, dataclasses, protocols)?
- How is configuration managed?

### Structure
- Where is the core business logic?
- Where are tests located?
- What build/tooling configuration exists?

---

## Output Format

Provide a structured summary including:

- Project purpose (1-2 sentences)
- Key components table
- Important files list with paths
- Notable patterns or conventions
- Any areas of complexity or concern

### Example Output

```markdown
## Project Purpose

Processes raw experimental data acquired by sl-experiment and produces analysis-ready datasets for the Sun Lab at
Cornell University.

## Key Components

| Component           | Location                    | Purpose                                     |
|---------------------|-----------------------------|---------------------------------------------|
| Core Processing     | src/sl_forgery/core/        | Data processing and forging pipelines       |
| Utilities           | src/sl_forgery/utilities/   | Shared utility functions                    |
| CLI Entry Points    | src/sl_forgery/cli/         | Command-line interfaces for data processing |

## Important Files

- `src/sl_forgery/core/pipeline.py` - Main data processing pipeline
- `src/sl_forgery/core/forger.py` - Dataset forging logic
- `pyproject.toml` - Project configuration and dependencies

## Notable Patterns

- Data processing pipelines with configurable stages
- Integration with sl-experiment output formats
- Configuration dataclasses from sl-shared-assets
- MyPy strict mode with full type annotations

## Areas of Concern

- Cross-library coordination with sl-experiment and sl-shared-assets
- Data format compatibility across library versions
```

---

## Usage

Invoke at session start to ensure full context before making changes. Prevents blind modifications
and ensures understanding of existing patterns.
