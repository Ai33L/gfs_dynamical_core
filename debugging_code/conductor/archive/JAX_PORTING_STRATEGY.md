# Strategy for Modularizing `getdyntend` for JAX and S2FFT

To prepare `getdyntend` (in `dyn_run.f90`) for a JAX-based implementation using `S2FFT`, you need to transition from the current monolithic, stateful Fortran design to a functional, pure, and decoupled architecture. JAX (and specifically `jax.jit`) requires pure functions without side-effects (no global mutable state, no in-place array updates).

Here is a step-by-step strategy to modularize `getdyntend` to facilitate a smooth port to JAX:

## 1. Decouple State from Logic (Eliminate Module Variables)
Currently, `getdyntend` heavily relies on module-level variables (e.g., `ug`, `vg`, `vrtg` from `grid_data`, and `vrtspec` from `spectral_data`).
* **JAX Paradigm:** JAX requires explicit state passing. 
* **Action:** Group your variables into state containers (e.g., `SpectralState`, `GridState`, `DiagnosticState`). Your overarching function should look like: `next_state = step(current_state, constants)`. 
* **Fortran Prep:** Refactor `getdyntend` to accept all necessary spectral and grid fields as `intent(in)` or `intent(out)` arguments rather than importing them via `use grid_data` or `use spectral_data`. 

## 2. Isolate Spectral Transforms (S2FFT Boundary)
In a JAX/S2FFT implementation, you want to batch your spectral transforms together to leverage GPU/TPU parallelism effectively. `getdyntend` currently interleaves math and transforms.
* **Action:** Split `getdyntend` into three distinct, sequential phases:
  1. **`spectral_to_grid(spectral_state) -> grid_state`**: Move all `spectogrd`, `getuv`, and `getgrad` calls here. This pure function takes spectral coefficients (vorticity, divergence, temperature, tracers, $\ln(p_s)$) and returns grid-space values and their spatial derivatives.
  2. **`compute_grid_tendencies(grid_state, grid_derivatives) -> grid_tendencies`**: All non-linear dynamics, advection, and physics happen here purely in grid space.
  3. **`grid_to_spectral(grid_tendencies) -> spectral_tendencies`**: Move all `grdtospec` and `getvrtdivspec` calls here to transform the computed grid tendencies back to spectral space.

## 3. Modularize the Grid-Space Dynamics (The Core Physics)
The grid-space tendency calculations in `getdyntend` are bundled together. Break these down into smaller, testable pure functions. This will make it much easier to write and test the JAX equivalents:
* **`compute_pressure_diagnostics(lnps)`**: Extract the logic of `calc_pressdata`. It should take $\ln(p_s)$ and return 3D arrays of $p$, $\Delta p$, $\alpha$, etc.
* **`compute_vertical_velocities(...)`**: Isolate `getomega`. It should take horizontal velocities and divergence and return $\omega$ (`dlnpdtg`), $\dot{\eta}$ (`etadot`), and the surface pressure tendency (`dlnpsdt`).
* **`compute_pressure_gradient(...)`**: Isolate `getpresgrad`. It should return the $x$ and $y$ pressure gradient forces.
* **`compute_vertical_advection(...)`**: Standardize `getvadv`. Make a generic pure function `vadv(field, etadot)` that works across $u, v, T, q$.
* **`compute_energy_conversion(...)`**: Isolate the computation of `vadvq` (the $\kappa \omega T_v$ term).

## 4. Separate Tendency Assembly
Once the diagnostic variables are computed, isolate the assembly of the final grid tendencies before they are transformed back:
* **Temperature Tendency:** Assemble $-u \frac{\partial T}{\partial x} - v \frac{\partial T}{\partial y} - \text{vadv}_t + \text{conversion}$.
* **Momentum Fluxes:** Compute the flux terms ($U(\zeta + f) + \text{vadv}_v$, etc.) which are later passed to `getvrtdivspec`.
* **Tracer Tendencies:** Assemble the 3D advection of tracers.

## Example of the Target Functional Architecture (Python/JAX pseudo-code)

Once you modularize the Fortran code as described, the JAX translation becomes a direct 1-to-1 mapping of pure functions:

```python
import jax.numpy as jnp
from jax import jit
import s2fft

@jit
def get_dyn_tend(spectral_state, constants):
    # 1. Spectral to Grid (Using S2FFT)
    grid_state, grid_grads = spectral_to_grid(spectral_state, s2fft_plan)
    
    # 2. Grid-Space Diagnostics
    press_diag = compute_pressure_diagnostics(grid_state.lnps, constants)
    omega, etadot, dlnps_dt = compute_vertical_velocities(grid_state, grid_grads, press_diag)
    
    # 3. Non-Linear Forcing & Advection
    pg_force_x, pg_force_y = compute_pressure_gradient(grid_state, grid_grads, press_diag)
    vadv_u, vadv_v, vadv_t, vadv_q = compute_vertical_advection(grid_state, etadot)
    energy_conv = compute_energy_conversion(omega, grid_state.T_v, grid_state.q)
    
    # 4. Assemble Grid Tendencies
    grid_tendencies = assemble_tendencies(
        grid_state, grid_grads, pg_force_x, pg_force_y, 
        vadv_u, vadv_v, vadv_t, vadv_q, energy_conv, dlnps_dt
    )
    
    # 5. Grid to Spectral Tendencies (Using S2FFT)
    spectral_tendencies = grid_to_spectral(grid_tendencies, s2fft_plan)
    
    # Add implicit/linear terms directly in spectral space (e.g. hyper-diffusion)
    spectral_tendencies = apply_linear_physics(spectral_tendencies, constants)
    
    return spectral_tendencies
```

## Next Immediate Steps for the Fortran Code
If you want to start refactoring the Fortran repository today:
1. Define a `derived type` containing all grid variables and another for spectral variables to replace the `grid_data` and `spectral_data` module includes.
2. Move `calc_pressdata`, `getomega`, `getpresgrad`, and `getvadv` out of relying on implicit module state, forcing them to take inputs and return outputs explicitly.
3. Group the `call spectogrd` and `call getgrad` block at the top into its own subroutine `state_to_grid()`.
