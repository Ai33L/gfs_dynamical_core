# Implementation Plan: Validation of JAX Port for GFS Dynamical Core

## Phase 1: Data Extraction Infrastructure
- [ ] Task: Implement a Fortran module for file-based array export (e.g., binary or `.npz` using a helper library).
- [ ] Task: Inject export hooks into `dyn_run.f90` and `pressure_data.f90` to capture inputs/outputs of each logical block.
- [ ] Task: Create a Python utility to load and structure this data into JAX-compatible formats.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Data Extraction Infrastructure' (Protocol in workflow.md)

## Phase 2: Systematic Module-wise Verification
- [ ] Task: Write Tests: Verify `compute_pressure_diagnostics` against Fortran ground truth.
- [ ] Task: Write Tests: Verify `compute_vertical_velocities` against Fortran ground truth.
- [ ] Task: Write Tests: Verify vertical advection and PGF logic against Fortran ground truth.
- [ ] Task: Write Tests: Verify full grid-space assembly against Fortran ground truth.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Systematic Module-wise Verification' (Protocol in workflow.md)

## Phase 3: Spectral Transform Synchronization
- [ ] Task: Develop a dedicated test to compare `SHTNS` (Fortran) and `S2FFT` (JAX) transforms for identical grid/spectral fields.
- [ ] Task: Adjust JAX normalization and coordinate mapping to match Fortran precisely.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Spectral Transform Synchronization' (Protocol in workflow.md)

## Phase 4: End-to-End Simulation Validation
- [ ] Task: Modify `examples/held_suarez.py` to allow switching between Fortran and JAX dynamical cores.
- [ ] Task: Execute a single-step comparison of all tendencies in Held-Suarez setup.
- [ ] Task: Run a multi-day Held-Suarez simulation and analyze numerical drift.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: End-to-End Simulation Validation' (Protocol in workflow.md)

## Phase 5: Final Verification & Reporting
- [ ] Task: Generate statistical comparisons (mean states, variance) for climate states.
- [ ] Task: Final report on numerical convergence and remaining discrepancies.
- [ ] Task: Conductor - User Manual Verification 'Phase 5: Final Verification & Reporting' (Protocol in workflow.md)
