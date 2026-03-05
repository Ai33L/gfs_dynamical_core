# Specification: JAX Port of GFS Dynamical Core

## Overview
This track aims to implement a fully functional JAX-based version of the GFS dynamical core. This involves porting the legacy Fortran logic to a pure functional Python architecture using JAX and S2FFT for spectral transforms, enabling a differentiable dynamical core capable of leveraging GPU/TPU acceleration.

## Functional Requirements
1. **Functional Core Implementation**:
    - Port `getdyntend` from `dyn_run.f90` to JAX, following the architecture in `JAX_PORTING_STRATEGY.md`.
    - Implement pure functions for grid-space diagnostics, vertical velocities, advection, and energy conversion.
    - Decouple state from logic by using explicit state containers (e.g., `SpectralState`, `GridState`).
2. **Spectral Transforms**:
    - Integrate `S2FFT` to handle forward and backward spectral transforms between spectral and grid space.
    - Batch transforms where possible for performance efficiency.
3. **Sympl Component Integration**:
    - Create a new Sympl component `GFSDynamicsJAX` in `gfs_dynamical_core/component_jax.py` (or similar).
    - Ensure it adheres to the existing Sympl/CliMT API, accepting and returning dictionaries of quantities.
4. **Validation**:
    - Implement an end-to-end comparison between the legacy Fortran-based component and the new JAX-based component.
    - Compare output tendencies (e.g., vorticity, divergence, temperature) for a single time step.

## Non-Functional Requirements
1. **Differentiability**: All ported code must be compatible with `jax.jit` and `jax.grad`.
2. **Device Agnosticism**: The code should be able to run on CPU, GPU, and TPU without modification (where JAX and S2FFT support it).
3. **Numerical Correctness**: JAX tendencies must match Fortran tendencies within a specified numerical tolerance (e.g., $10^{-6}$ for single precision or $10^{-12}$ for double).

## Acceptance Criteria
- [ ] A new `GFSDynamicsJAX` component is available in the package.
- [ ] The component can be initialized with standard GFS parameters.
- [ ] End-to-end tendencies match the legacy version for a given initial state (within tolerance).
- [ ] The core functions (diagnostics, advection) are successfully `jax.jit` compiled.

## Out of Scope
- Optimizing performance for specific hardware targets (initial focus is correctness).
- Porting sub-grid scale physics parameterizations (only the dynamical core is in scope).
- Bit-for-bit equivalence (minor numerical differences due to JAX/Fortran precision handling are expected).
