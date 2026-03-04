# Implementation Plan: Modernize project configuration and linting

## Phase 1: Infrastructure Setup [checkpoint: a2fb710]
- [x] Task: Audit current configuration files (setup.py, setup.cfg, requirements_dev.txt, tox.ini). dc9cfe9
- [x] Task: Create initial pyproject.toml with setuptools build-system. 8692d1e
- [x] Task: Configure Ruff in pyproject.toml. cf40e1a
- [x] Task: Conductor - User Manual Verification 'Infrastructure Setup' (Protocol in workflow.md) a2fb710

## Phase 2: Migration & Cleanup
- [x] Task: Write tests to verify metadata extraction and package installation. aa23ee9
- [x] Task: Implement metadata migration to pyproject.toml. 9f62ea3
- [x] Task: Implement dependency migration to pyproject.toml. 05a4562
- [ ] Task: Conductor - User Manual Verification 'Migration & Cleanup' (Protocol in workflow.md)

## Phase 3: Linting & Validation
- [ ] Task: Write tests to verify Ruff configuration.
- [ ] Task: Run Ruff check and Ruff format on the entire codebase.
- [ ] Task: Fix linting/formatting issues reported by Ruff.
- [ ] Task: Verify final build and run all existing tests.
- [ ] Task: Conductor - User Manual Verification 'Linting & Validation' (Protocol in workflow.md)
