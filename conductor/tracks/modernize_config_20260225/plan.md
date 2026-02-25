# Implementation Plan: Modernize project configuration and linting

## Phase 1: Infrastructure Setup
- [x] Task: Audit current configuration files (setup.py, setup.cfg, requirements_dev.txt, tox.ini). dc9cfe9
- [x] Task: Create initial pyproject.toml with setuptools build-system. 8692d1e
- [x] Task: Configure Ruff in pyproject.toml. cf40e1a
- [ ] Task: Conductor - User Manual Verification 'Infrastructure Setup' (Protocol in workflow.md)

## Phase 2: Migration & Cleanup
- [ ] Task: Write tests to verify metadata extraction and package installation.
- [ ] Task: Implement metadata migration to pyproject.toml.
- [ ] Task: Implement dependency migration to pyproject.toml.
- [ ] Task: Conductor - User Manual Verification 'Migration & Cleanup' (Protocol in workflow.md)

## Phase 3: Linting & Validation
- [ ] Task: Write tests to verify Ruff configuration.
- [ ] Task: Run Ruff check and Ruff format on the entire codebase.
- [ ] Task: Fix linting/formatting issues reported by Ruff.
- [ ] Task: Verify final build and run all existing tests.
- [ ] Task: Conductor - User Manual Verification 'Linting & Validation' (Protocol in workflow.md)
