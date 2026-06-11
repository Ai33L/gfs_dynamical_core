# Specification: Modernize project configuration and linting

## Objectives
- Migrate legacy configuration (setup.py, setup.cfg) to a modern pyproject.toml.
- Integrate Ruff for fast, unified linting and formatting.
- Ensure the project remains installable and tests continue to pass.

## Requirements
- pyproject.toml must serve as the single source of truth for project metadata and dependencies.
- Ruff must be configured with rules appropriate for a scientific Python project.
- Maintain compatibility with existing build tools (Cython, GNU Make).

## Acceptance Criteria
- All metadata and dependencies moved to pyproject.toml.
- Ruff runs successfully across the codebase.
- Project can be installed from source.
- All existing tests pass.
