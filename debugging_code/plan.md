# Implementation Plan: Achieving Algorithmic Equivalence between JAX and Fortran GFS

## Phase 1: Mathematical Corrections (Grid Space)
- [x] **Task 1.1: Fix Vertical Advection Boundary Sign**
  - Modify `compute_vertical_advection` in `gfs_dynamical_core/jax/dynamics.py`.
  - Correct the tendency calculation at the bottom boundary from `(data[-2] - data[-1])` to `(data[-1] - data[-2])` to match Fortran's `Bottom - Level Above`.
- [x] **Task 1.2: Correct Pressure Gradient Force (PGF) Coefficients**
  - Update `compute_pressure_gradient_force` in `gfs_dynamical_core/jax/dynamics.py`.
  - Re-derive and implement the exact mathematical forms for `cofa` and `cofb` as present in Fortran's `getpresgrad` (`bk * rlnp` term in `cofb` and proper term grouping in `cofa`).
- [x] **Task 1.3: Write Grid Space Tests**
  - Write test cases in `tests/test_jax_dynamics.py` specifically comparing the PGF coefficients and vertical advection tendencies against Fortran reference outputs.

## Phase 2: Vector Transforms and Spatial Gradients
- [x] **Task 2.1: Implement Vector Spherical Transforms**
  - Update `transforms.py` to replace mocked `u` and `v` arrays with proper vector transforms from vorticity and divergence using `s2fft`.
  - Implement the reverse transform to convert grid-space momentum fluxes (`u_flux`, `v_flux`) back into spectral vorticity and divergence tendencies.
- [x] **Task 2.2: Spectral Gradient Calculations**
  - Update `transforms.py` to calculate exact spatial gradients (`d_log_ps_d_lambda`, `d_log_ps_d_phi`, `d_t_d_lambda`, `d_t_d_phi`) via spectral derivation before transforming to grid space, replacing the currently mocked zero arrays.
- [x] **Task 2.3: Write Transform Tests**
  - Write test cases in `tests/test_jax_transforms.py` to verify that spectral-to-grid and grid-to-spectral transformations of vectors and spatial gradients exactly match the Fortran `shtns` library outputs.

## Phase 3: Advanced Tracer Advection
- [x] **Task 3.1: Thuburn Flux-Limited Scheme**
  - Create a new function `compute_vertical_advection_tracers` in `dynamics.py`.
  - Implement the Thuburn (1993) positive-definite flux-limited vertical advection scheme, mirroring the Fortran `getvadv_tracers` subroutine to prevent negative humidity and tracer values.
  - Wire this new advection function into the tracer tendency logic in `assemble_grid_tendencies`.
- [x] **Task 3.2: Write Tracer Advection Tests**
  - Write test cases in `tests/test_jax_dynamics.py` verifying that the Thuburn scheme produces identical tracer tendencies to the Fortran `getvadv_tracers` subroutine.

## Phase 4: Time Integration Machinery
- [x] **Task 4.1: IMEX Runge-Kutta Stepper**
  - Create a new `stepper.py` module in the JAX package.
  - Implement the 3-stage, 2nd-order additive (IMEX) Runge-Kutta scheme defined in Fortran's `semimp_data.f90` and `run.f90`.
- [x] **Task 4.2: Linear Diffusion Solver**
  - Implement the forward-implicit treatment of linear hyper-diffusion and Rayleigh damping as an update step within the time integrator.
- [x] **Task 4.3: Write Time Integration Tests**
  - Write test cases in a new `tests/test_jax_stepper.py` to simulate a single Runge-Kutta timestep and ensure the final advanced state matches the Fortran `advance` subroutine.

## Phase 5: Verification
- [x] **Task 5.1: Unit Test the Fixes**
  - Run the existing and newly added unit tests to confirm the mathematical corrections align exactly with Fortran outputs.
- [x] **Task 5.2: End-to-End Test**
  - Run a multi-step test (e.g., `test_jax_fortran_comparison.py`) to verify that the full time integration cycle remains stable and mathematically equivalent to the Fortran baseline across multiple steps.