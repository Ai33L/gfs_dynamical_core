# Specification: Validation of JAX Port for GFS Dynamical Core

## Overview
This track focuses on the rigorous validation of the JAX implementation of the GFS dynamical core. The primary goal is to ensure that the JAX port produces numerically convergent results compared to the original Fortran implementation across all functional modules, culminating in nearly identical climate states in the Held-Suarez test.

## Functional Requirements
1. **Fortran Intermediate Data Hooks**:
    - Implement a systematic mechanism in the Fortran code (`dyn_run.f90`, etc.) to export intermediate tensors at each key step (e.g., pressure data, vertical velocities, advection tendencies, momentum fluxes).
    - Use a file-based export strategy (e.g., `.npz` format via a small Fortran utility or raw binary export with Python readers).
2. **Module-wise Verification Suite**:
    - Create a Python test suite that loads the exported Fortran intermediate data.
    - Validate each JAX pure function (`compute_pressure_diagnostics`, `compute_vertical_velocities`, etc.) against its corresponding Fortran output using real-world simulation data rather than analytical inputs.
3. **Spectral Transform Alignment**:
    - Ensure that the JAX-native `S2FFT` transforms are perfectly aligned with the Fortran `SHTNS` transforms, including normalization factors and grid configurations.
4. **End-to-End Climate Validation**:
    - Run the `examples/held_suarez.py` simulation using both the Fortran and JAX backends.
    - Compare full-step tendencies and long-term climate statistics (mean temperature, zonal wind profiles) between the two implementations.

## Non-Functional Requirements
1. **Numerical Convergence**: Results should match within a specified numerical tolerance (e.g., $10^{-10}$ for double precision) across individual modules.
2. **Reproducibility**: The validation suite should be automated and reproducible, allowing for regression testing as the JAX port is refined.
3. **Observability**: Provide clear diagnostic logs and visual comparisons (e.g., difference maps) for validation failures.

## Acceptance Criteria
- [ ] Intermediate data export hooks are functional in the Fortran build.
- [ ] A systematic test suite exists, verifying all JAX dynamics modules against Fortran data.
- [ ] `examples/held_suarez.py` runs successfully with the JAX backend and produces results nearly identical to the Fortran baseline.
- [ ] Statistical climate states from Held-Suarez are indistinguishable within expected numerical drift.

## Out of Scope
- Performance optimization of the JAX code (focus is purely on correctness).
- Porting remaining physics parameterizations beyond what is necessary for the Held-Suarez test.
- Bit-for-bit equivalence if differences are proven to be solely due to library-specific floating-point arithmetic (e.g., JAX vs Fortran intrinsics).
