# JAX Dynamical Core — Component-Driven API (Fortran Parity)

**Date:** 2026-06-12
**Status:** Approved design, implementation deferred (not a current priority)
**Branch:** jax-port

## Summary

Give the JAX dynamical core (`GFSDynamicsJAX`) the same component-driven
stepper API as the Fortran core (`GFSDynamicalCore`): it accepts a list of
`TendencyComponent` objects, runs them internally each step, converts their
tendencies, and steps the state forward — with **full Fortran parity**,
including the virtual-temperature correction, surface-pressure tendency, and
the sympl tracer pack/unpack pathway (`register_tracer` / `uses_tracers`).

The existing lower-level API (manual `set_physics_tendencies` driving the pure
jnp pipeline) is preserved for differentiability, and the inner pipeline is
additionally exposed as a clean standalone pure function.

## Motivation

Today the two cores have divergent APIs. The Fortran core:

```python
dycore = GFSDynamicalCore(tendency_component_list=[hs])
state = climt.get_default_state([dycore], grid_state=grid)
# loop:
_, out = dycore(state, timestep); state.update(out)
```

The JAX core forces the *driver* to do the coupling by hand
(`debugging_code/heldsuarez_run.py` lines 79–92): reconstruct 3-D
`air_pressure` from `surface_air_pressure`, call `hs(state)`, extract u/v/T
tendencies, and push them through the imperative `set_physics_tendencies`
side-channel before each `dycore(state, timestep)`.

The goal is for the JAX core to be a drop-in replacement for the Fortran core
at the driver level, while keeping a differentiable path for gradient-based
work (e.g. ML physics emulators).

## Current State (reference)

- `gfs_dynamical_core/component_jax.py` — `GFSDynamicsJAX(Stepper)`.
  - Single-tracer handling: reads `state["specific_humidity"]`, builds
    `tracers=jnp.stack([q], axis=0)` (shape `(1, n_lev, n_lat, n_lon)`).
  - `set_physics_tendencies(u_tend, v_tend, t_tend)` stores grid jnp
    tendencies; `_apply_physics_tendencies` converts u/v → vort/div and T to
    spectral and applies a time-split increment after the dynamics step.
  - Outputs: air_temperature, eastward_wind, northward_wind,
    surface_air_pressure, specific_humidity. **No** `air_pressure`.
- `gfs_dynamical_core/component.py` — `GFSDynamicalCore(TendencyStepper)`.
  - `uses_tracers = True`, `tracer_dims = ("tracer","mid_levels","lat","lon")`,
    `prepend_tracers = (("specific_humidity","kg/kg"),)`.
  - `__call__` packs tracers, runs the component composite, unit-converts
    tendencies, calls `array_call(..., prognostic_tendencies=...)`, unpacks
    tracers.
  - `array_call` builds virtual-temperature tendency, lnps tendency, tracer
    tendencies and hands them to the Fortran stepper.
  - Outputs `air_pressure` and `air_pressure_on_interface_levels`.
- `gfs_dynamical_core/jax/stepper.py` — `advance(...)` is the jitted pure
  spectral step.

## Design

### 1. Class structure & API

`GFSDynamicsJAX` converts from `Stepper` to `TendencyStepper`, following the
pattern `GFSDynamicalCore` already establishes in this codebase.

```python
class GFSDynamicsJAX(TendencyStepper):
    uses_tracers = True
    tracer_dims = ("tracer", "mid_levels", "lat", "lon")
    prepend_tracers = (("specific_humidity", "kg/kg"),)  # humidity = tracer 0

    @property
    def spectral_names(self):
        return ("eastward_wind", "northward_wind", "air_temperature",
                "surface_air_pressure") + get_tracer_names()

    def __init__(self, tendency_component_list=None, adiabatic=False,
                 zero_negative_moisture=True, **kwargs):
        ...
```

Construction mirrors Fortran:

- Wrap `tendency_component_list` in `ImplicitTendencyComponentComposite`.
- Run the same bad-diagnostics validation: a component may not emit any
  `spectral_names` or any quantity already diagnosed by the core as a
  *diagnostic* output (raise `GFSError`).
- Merge the components' `input_/output_/diagnostic_properties` into the core's
  via the existing `get_valid_properties` helper (factored out of
  `component.py` for reuse, or imported).
- sympl's `uses_tracers` machinery provides `_tracer_packer`.

**Backward compatibility (hard requirement):** `GFSDynamicsJAX()` with no
arguments must produce results bit-for-bit identical to today's dynamics-only
behavior. All existing `tests/test_jax_*.py` must pass unchanged.

### 2. Pressure outputs

Add to `output_properties`:

| Quantity | Dims | Source |
|---|---|---|
| `air_pressure` | `mid_levels, lat, lon` | `compute_pressure_diagnostics(...).prs` |
| `air_pressure_on_interface_levels` | `interface_levels, lat, lon` | `.pk` |

Computed each step from the new surface pressure and returned as numpy
(matching the Fortran output contract). Internal components then receive a
consistent 3-D pressure field, and driver lines 80–87 (manual reconstruction)
are deleted.

Note BTT vs. TTB ordering: the JAX pipeline is bottom-to-top; climt/Fortran
expects interface pressure top-to-bottom. The interface-level output is
flipped on the way out to match the climt convention, mirroring
`component.py`'s `[::-1, :, :]` handling.

### 3. Tendency handling — full Fortran parity

`__call__` (modeled on `GFSDynamicalCore.__call__`):

1. Pack tracers into `raw_state["tracers"]` of shape
   `(n_tracers, n_lev, n_lat, n_lon)` via `_tracer_packer.pack`.
2. Run the component composite → `(tendencies, diagnostics)`.
3. Unit-convert each tendency that matches an input property to
   `"<unit> s^-1"`.
4. Call `array_call(raw_state, timestep, prognostic_tendencies=tendencies)`.
5. Unpack tracers from the output; restore data arrays with properties.

`array_call` builds the full tendency set before applying the time-split
increment:

- **Virtual-temperature tendency** (Fortran `component.py` lines 500–504):
  ```
  virtual_temp_tend = T_tend * (1 + fvirt*q) + fvirt * t_virt * q_tend
  fvirt = (1 - Rd/Rv) / (Rd/Rv)
  t_virt = air_temperature * (1 + fvirt*q)
  ```
- **Surface-pressure tendency:** `lnps_tend = ps_tend / ps`.
- **Tracer tendencies:** all packed tracers, in tracer order.

`_apply_physics_tendencies` extends from u/v/T-only to also apply the
**log-surface-pressure** and **per-tracer** spectral increments (today those
two are passed through untouched). Time-split semantics are preserved:
tendencies are computed from the state at the *start* of the step and applied
as `x += dt * tendency` *after* the dynamics step (matches Fortran
`run.f90`).

`zero_negative_moisture` reproduces Fortran's post-step clipping of tracer 0
(`set_negatives_to_zero`).

### 4. Multi-tracer generalization

The internal jnp pipeline currently hardcodes one tracer. It generalizes to
`n_tracers` taken from the packed array:

- `SpectralState.tracers` already carries a leading tracer axis; the dynamics
  and transforms must `vmap` cleanly over `n_tracers > 1`.
- **Verification step (implementation):** confirm
  `compute_vertical_advection_tracers` and the spectral transforms handle an
  arbitrary tracer count. If any assume a single tracer, generalize them.

### 5. Differentiable pure function

Extract the tendency → spectral increment + advance into a module-level pure
jnp function in `stepper.py`:

```python
def advance_with_tendencies(spec_state, phys_tends, phis_grads,
                            dyn_config, trans_config, stepper_config,
                            latitudes, gauss_weights=None, pdryini=None,
                            spec_tends=None) -> SpectralState:
    ...
```

The function advances the dynamics one step and then applies a **single
time-split increment assembled from two independent tendency containers**,
either of which may be `None`:

- **`phys_tends`** — a `PhysicsTendencies` flax-struct dataclass carrying
  *grid-space* jnp tendencies for u, v, virtual-T, lnps, and tracers. These are
  transformed to spectral coefficients internally (u, v → vorticity/divergence;
  the rest via `s2_forward`), then truncated. This is the path physical /
  learned column physics uses (Held–Suarez, the convection scheme, an NN
  residual).
- **`spec_tends`** — an optional `SpectralTendencies` flax-struct dataclass
  (the one already defined in `states.py`: `d_vorticity_d_t`,
  `d_divergence_d_t`, `d_temperature_d_t`, `d_log_surface_pressure_d_t`,
  `d_tracers_d_t`) carrying tendencies **already in spectral space**. These are
  added directly to the increment with no grid→spectral transform. This is the
  path stochastic / spectral-native components use — notably **SKEB**, which
  generates a kinetic-energy-backscatter perturbation to the *vorticity*
  tendency natively in spectral space (where its AR(1) pattern and the
  reality-symmetry guard live), and trainable **SPPT**, whose AR(1) pattern is
  spectral.

The two increments are summed field-by-field before the `x += dt * tendency`
update, so a caller may supply grid tendencies, spectral tendencies, both, or
neither. Container-level `None` is a trace-time structural choice (JIT-safe);
within a supplied container, unused fields are zero arrays so the whole path
stays `jit`/`grad`/`vmap`-able with no sympl/numpy in it. `_apply_physics_
tendencies` and the component path both become thin wrappers over the
`phys_tends` route; gradient users and stochastic components call the function
directly, using whichever container fits their tendency's natural space.

**Rationale for the dual container (logged):** injecting a spectral-space
perturbation directly avoids a spurious grid→spectral round-trip and its
truncation error, and it is the natural interface for the stochastic-physics
component family (SKEB, SPPT) and for adjoint / singular-vector work, all of
which live in spectral space. Reusing the existing `SpectralTendencies` struct
keeps the surface minimal.

### 6. Driver simplification

`debugging_code/heldsuarez_run.py` jax branch collapses to mirror the fortran
branch:

```python
from gfs_dynamical_core.component_jax import GFSDynamicsJAX
dycore = GFSDynamicsJAX(tendency_component_list=[hs])
state = climt.get_default_state([dycore], grid_state=grid)
# loop body:
_, out = dycore(state, timestep)
state.update(out)
state["time"] += timestep
```

No `hs(state)` call, no manual `air_pressure` rebuild, no
`set_physics_tendencies` in the loop. The `MODE`-specific branches in the loop
body collapse into the single shared path.

## Testing

- **Backward compatibility:** existing `tests/test_jax_*.py` pass unchanged.
- **Multi-tracer round-trip:** pack/unpack of humidity + one passive tracer
  survives a step with correct shapes and values.
- **Component vs. manual equivalence:** the component path with a
  tracer-tendency-producing component yields output identical to the manual
  `set_physics_tendencies` path given the same tendencies.
- **Parity of derived tendencies:** for one Held-Suarez step, the spectral
  increment from virtual-T and lnps tendencies matches an independent
  recomputation.
- **Differentiability:** `advance_with_tendencies` is `jax.grad`-able,
  including tracer-tendency inputs (smoke test on a small grid).
- **Integration:** a few Held-Suarez steps run end-to-end; `air_pressure` is
  present in the output and consistent with `surface_air_pressure`.

## Out of Scope / Non-Goals

- No change to the spectral dynamics numerics themselves.
- No new physics components — only the coupling pathway.
- No performance work beyond keeping the existing JIT behavior.

## Risks & Open Verification Points

1. **Multi-tracer vmap** in dynamics/transforms (see §4) — must be confirmed,
   may require generalizing single-tracer assumptions.
2. **Virtual-temperature convention** — the JAX dynamics' internal temperature
   treatment must be checked so the virtual-T tendency mapping (§3) is applied
   consistently with how the pipeline already handles moisture.
3. **TendencyStepper `__call__` surface** — the JAX core defines its own
   `__call__` (as Fortran does) rather than relying on a sympl base
   implementation; the property-merging and tracer-packing must follow the
   Fortran reference exactly to avoid subtle state-handling divergence.
