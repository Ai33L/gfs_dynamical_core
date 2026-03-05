# Implementation Plan: JAX Port of GFS Dynamical Core

## Phase 1: Environment & Foundations
- [x] Task: Install dependencies (`jax`, `jaxlib`, `s2fft`, `flax`). 3382dfc
- [x] Task: Create module structure: `gfs_dynamical_core/jax/` with `states.py`, `transforms.py`, and `dynamics.py`. 115dda9
- [x] Task: Implement `SpectralState` and `GridState` immutable containers using `flax.struct` to support JAX transformations. d196a72
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Environment & Foundations' (Protocol in workflow.md)

## Phase 2: Spectral Transform Layer (S2FFT)
- [ ] Task: Implement forward and backward transforms using `S2FFT` for GFS grid configurations.
- [ ] Task: Write Tests: Verify identity transform (Spectral -> Grid -> Spectral) for arbitrary fields.
- [ ] Task: Implement batching for transforms to handle multiple vertical levels simultaneously.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Spectral Transform Layer (S2FFT)' (Protocol in workflow.md)

## Phase 3: Grid-Space Dynamics Porting
- [ ] Task: Port pressure diagnostics (`compute_pressure_diagnostics`) from Fortran to pure JAX.
- [ ] Task: Port vertical velocity and surface pressure tendency logic (`compute_vertical_velocities`).
- [ ] Task: Port vertical advection and pressure gradient force logic.
- [ ] Task: Port energy conversion and final tendency assembly.
- [ ] Task: Write Tests: Unit tests for each pure function using a combination of synthetic inputs and legacy Fortran intermediate outputs.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Grid-Space Dynamics Porting' (Protocol in workflow.md)

## Phase 4: Sympl Component Integration
- [ ] Task: Implement `GFSDynamicsJAX` in `gfs_dynamical_core/component_jax.py` inheriting from `sympl.Stepper`.
- [ ] Task: Implement `__call__` (or `step`) to orchestrate the mapping between Sympl dictionaries and JAX containers.
- [ ] Task: Implement input/output mapping for Sympl/CliMT dictionaries.
- [ ] Task: Write Tests: Verify component initialization and basic Sympl integration.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Sympl Component Integration' (Protocol in workflow.md)

## Phase 5: End-to-End Validation & Verification
- [ ] Task: Develop a validation script to compare `GFSDynamicsJAX` output tendencies against the legacy Fortran version.
- [ ] Task: Verify numerical accuracy (matching within specified tolerance) for a full dynamics step.
- [ ] Task: Verify `jax.jit` compatibility and differentiability of the entire component.
- [ ] Task: Conductor - User Manual Verification 'Phase 5: End-to-End Validation & Verification' (Protocol in workflow.md)
