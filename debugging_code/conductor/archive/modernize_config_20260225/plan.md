# Implementation Plan: Modernize project configuration and linting

## Phase 1: Infrastructure Setup [checkpoint: a2fb710]
- [x] Task: Audit current configuration files (setup.py, setup.cfg, requirements_dev.txt, tox.ini). dc9cfe9
- [x] Task: Create initial pyproject.toml with setuptools build-system. 8692d1e
- [x] Task: Configure Ruff in pyproject.toml. cf40e1a
- [x] Task: Conductor - User Manual Verification 'Infrastructure Setup' (Protocol in workflow.md) a2fb710

## Phase 2: Migration & Cleanup [checkpoint: 51f19f3]
- [x] Task: Write tests to verify metadata extraction and package installation. aa23ee9
- [x] Task: Implement metadata migration to pyproject.toml. 9f62ea3
- [x] Task: Implement dependency migration to pyproject.toml. 05a4562
- [x] Task: Conductor - User Manual Verification 'Migration & Cleanup' (Protocol in workflow.md) 51f19f3

## Phase 3: Linting & Validation [checkpoint: eed5a7d]
- [x] Task: Write tests to verify Ruff configuration. c32fb25
- [x] Task: Run Ruff check and Ruff format on the entire codebase. a85a062
- [x] Task: Fix linting/formatting issues reported by Ruff. a85a062
- [x] Task: Verify final build and run all existing tests. b25e0d5
- [x] Task: Conductor - User Manual Verification 'Linting & Validation' (Protocol in workflow.md) eed5a7d
