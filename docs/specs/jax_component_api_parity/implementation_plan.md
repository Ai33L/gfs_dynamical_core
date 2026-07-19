# JAX Component-Driven API (Fortran Parity) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `GFSDynamicsJAX` the Fortran core's component-driven `TendencyStepper` API — accept a `tendency_component_list`, run components internally, and step with full parity (virtual-T, lnps, and tracer pack/unpack tendencies) — while preserving and exposing a differentiable jnp pipeline.

**Architecture:** Convert `GFSDynamicsJAX` from `Stepper` to `TendencyStepper`, reusing sympl's tracer-packing and the existing `GFSDynamicalCore` patterns. A single module-level pure jnp function (`advance_with_tendencies`) becomes the shared core for both the component path (numpy tendencies → jnp) and the manual `set_physics_tendencies` path (jnp directly). The core also outputs 3-D pressure so internal components see consistent state.

**Tech Stack:** Python, JAX (x64, CPU), flax.struct, sympl/climt, s2fft, pytest.

> **Companion spec:** `docs/specs/jax_component_api_parity/design.md`. Read it first.

> **Reference implementation:** `gfs_dynamical_core/component.py` (the Fortran
> `GFSDynamicalCore`) is the parity target. Mirror its `__init__`, `__call__`,
> tracer handling, and tendency construction closely.

---

## File Structure

- `gfs_dynamical_core/component_jax.py` — **modify**. `GFSDynamicsJAX` becomes a
  `TendencyStepper`; gains tracer machinery, pressure outputs, full tendency
  construction, and thin wrappers over the pure core.
- `gfs_dynamical_core/jax/stepper.py` — **modify**. Add the `GridTendencies`-style
  physics-tendency struct and the pure `advance_with_tendencies` function.
- `gfs_dynamical_core/_property_utils.py` — **create**. Hold `get_valid_properties`
  (moved from `component.py`) so both cores share it without a circular import.
- `gfs_dynamical_core/component.py` — **modify**. Import `get_valid_properties`
  from the new shared module (no behavior change).
- `tests/test_jax_component_api.py` — **create**. New behavior: component path,
  multi-tracer, pressure output, equivalence, differentiability.
- `debugging_code/heldsuarez_run.py` — **modify**. Collapse the jax branch to
  the shared component-driven path.

---

## Task 1: Extract shared `get_valid_properties` helper

**Files:**
- Create: `gfs_dynamical_core/_property_utils.py`
- Modify: `gfs_dynamical_core/component.py:29-45` (remove local def, import instead)
- Test: `tests/test_jax_component_api.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_jax_component_api.py
def test_get_valid_properties_filters_known_quantities():
    from gfs_dynamical_core._property_utils import get_valid_properties
    gfs_props = {"air_temperature": {"dims": ["mid_levels", "lat", "lon"]}}
    prognostic = {
        "air_temperature": {"dims": ["mid_levels", "lat", "lon"], "units": "K"},
        "my_forcing": {"dims": ["mid_levels", "lat", "lon"], "units": "K s^-1"},
    }
    result = get_valid_properties(gfs_props, prognostic, "input")
    assert "air_temperature" not in result   # already known to the core
    assert "my_forcing" in result            # extra quantity surfaced
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_jax_component_api.py::test_get_valid_properties_filters_known_quantities -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gfs_dynamical_core._property_utils'`

- [ ] **Step 3: Create the shared module**

```python
# gfs_dynamical_core/_property_utils.py
class GFSError(Exception):
    pass


def get_valid_properties(gfs_properties, prognostic_properties, property_type):
    return_dict = {}
    for name, properties in prognostic_properties.items():
        if name not in gfs_properties:
            return_dict[name] = properties
        elif "dims" in prognostic_properties.keys():
            extra_dims = set(prognostic_properties["dims"]).difference(
                ["*"] + list(gfs_properties[name]["dims"])
            )
            if len(extra_dims) != 0:
                raise GFSError(
                    "Cannot handle TendencyComponent with {} {} "
                    "that has extra dimensions {} not used by GFS".format(
                        property_type, name, extra_dims
                    )
                )
    return return_dict
```

- [ ] **Step 4: Update `component.py` to import from the shared module**

In `gfs_dynamical_core/component.py`, delete the local `GFSError` class
(lines 25-26) and the local `get_valid_properties` def (lines 29-45). Add near
the top imports:

```python
from ._property_utils import GFSError, get_valid_properties
```

- [ ] **Step 5: Run tests to verify both pass**

Run: `pytest tests/test_jax_component_api.py::test_get_valid_properties_filters_known_quantities tests/test_components.py -v`
Expected: PASS (new test passes; Fortran-core tests unaffected). If the
compiled Fortran extension is unavailable in the environment, `test_components.py`
may skip — that is acceptable; the new test must pass.

- [ ] **Step 6: Commit**

```bash
git add gfs_dynamical_core/_property_utils.py gfs_dynamical_core/component.py tests/test_jax_component_api.py
git commit -m "refactor: extract get_valid_properties into shared module"
```

---

## Task 2: Pure physics-tendency struct + `advance_with_tendencies`

**Files:**
- Modify: `gfs_dynamical_core/jax/stepper.py`
- Test: `tests/test_jax_component_api.py`

This task introduces the differentiable core. `PhysicsTendencies` carries
grid-space jnp tendencies; `advance_with_tendencies` runs the dynamics step
then applies a time-split spectral increment built from those tendencies.

**Dual tendency path (design §5):** `advance_with_tendencies` accepts BOTH a
grid-space `PhysicsTendencies` container (`phys_tends`, transformed to spectral
internally) AND an optional spectral-space `SpectralTendencies` container
(`spec_tends`, added directly with no transform). Either may be `None`. The
spectral path is the injection point for stochastic / spectral-native
components — notably SKEB (kinetic-energy backscatter on the vorticity
tendency) and trainable SPPT — which generate their perturbations natively in
spectral space. `SpectralTendencies` already exists in `states.py`; reuse it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_jax_component_api.py (add to file)
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"
import jax
from jax import config as _jaxcfg
_jaxcfg.update("jax_enable_x64", True)
import jax.numpy as jnp
from gfs_dynamical_core.jax.stepper import (
    StepperConfig, advance, advance_with_tendencies, PhysicsTendencies,
)
from gfs_dynamical_core.jax.dynamics import DynamicsConfig
from gfs_dynamical_core.jax.states import SpectralState
from gfs_dynamical_core.jax.transforms import (
    TransformConfig, get_gaussian_latitudes,
)


def _mock_dyn_config(n_lev):
    ak = jnp.zeros(n_lev + 1)
    bk = jnp.linspace(1, 0, n_lev + 1)  # BTT: surface bk=1, TOA bk=0
    dbk = bk[:-1] - bk[1:]
    ck = ak[:-1] * bk[1:] - ak[1:] * bk[:-1]
    return DynamicsConfig(
        ak=ak, bk=bk, ck=ck, dbk=dbk, rk=0.286, toa_pressure=0.0,
        radius=6.371e6, omega=7.292e-5, g=9.81, rd=287.0, rv=461.0,
        cp=1004.0, cvap=1810.0,
    )


def _mock_state(n_lev, L, n_tracers=1):
    z = lambda: jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128)
    lnps = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128).at[0, L - 1].set(1.0)
    return SpectralState(
        vorticity=z(), divergence=z(), temperature=z(),
        log_surface_pressure=lnps,
        tracers=jnp.zeros((n_tracers, n_lev, L, 2 * L - 1), dtype=jnp.complex128),
    )


def test_advance_with_zero_tendencies_matches_advance():
    L, n_lev = 8, 10
    dyn = _mock_dyn_config(n_lev)
    trans = TransformConfig(L=L, radius=1.0)
    sc = StepperConfig(dt=10.0, explicit=True)
    lat = get_gaussian_latitudes(L)
    n_lat, n_lon = trans.n_lat, trans.n_lon
    phis = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))
    state = _mock_state(n_lev, L)

    base = advance(state, phis, dyn, trans, sc, lat)
    zero_tends = PhysicsTendencies(
        u=jnp.zeros((n_lev, n_lat, n_lon)),
        v=jnp.zeros((n_lev, n_lat, n_lon)),
        virtual_temperature=jnp.zeros((n_lev, n_lat, n_lon)),
        log_surface_pressure=jnp.zeros((n_lat, n_lon)),
        tracers=jnp.zeros((1, n_lev, n_lat, n_lon)),
    )
    with_t = advance_with_tendencies(state, zero_tends, phis, dyn, trans, sc, lat)
    assert jnp.allclose(base.vorticity, with_t.vorticity)
    assert jnp.allclose(base.divergence, with_t.divergence)
    assert jnp.allclose(base.temperature, with_t.temperature)
    assert jnp.allclose(base.log_surface_pressure, with_t.log_surface_pressure)
    assert jnp.allclose(base.tracers, with_t.tracers)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_jax_component_api.py::test_advance_with_zero_tendencies_matches_advance -v`
Expected: FAIL with `ImportError: cannot import name 'advance_with_tendencies'`
(also `PhysicsTendencies`).

- [ ] **Step 3: Add `PhysicsTendencies` and `advance_with_tendencies`**

In `gfs_dynamical_core/jax/stepper.py`, add the struct near the other
flax-struct configs:

```python
@struct.dataclass
class PhysicsTendencies:
    """Grid-space (bottom-to-top) physics tendencies, SI units.

    u, v               : m s^-2
    virtual_temperature: K s^-1   (already includes the moisture correction)
    log_surface_pressure: s^-1    (i.e. (1/ps) * dps/dt)
    tracers            : (n_tracers, n_lev, n_lat, n_lon) kg/kg s^-1
    """
    u: jnp.ndarray
    v: jnp.ndarray
    virtual_temperature: jnp.ndarray
    log_surface_pressure: jnp.ndarray
    tracers: jnp.ndarray
```

Add the conversion + advance function. It reuses the spectral conversion logic
currently inlined in `component_jax._apply_physics_tendencies`, generalized to
lnps and tracers:

```python
def _physics_tendencies_to_spectral(tends, trans_config, dyn_config):
    """Convert grid PhysicsTendencies to spectral increments (per-field)."""
    from .transforms import (
        s2_forward, enforce_triangular_truncation,
    )
    L = trans_config.L
    sampling = trans_config.sampling
    radius = dyn_config.radius
    T = trans_config.truncation

    l_arr = jnp.arange(L)
    l_factor = jnp.sqrt(l_arr * (l_arr + 1))

    def uv_to_vrtdiv(u, v):
        F1 = s2_forward(-v + 1j * u, L, sampling, spin=1)
        Fm1 = s2_forward(v + 1j * u, L, sampling, spin=-1)
        rp = l_factor[:, None] * F1 / radius
        rm = l_factor[:, None] * Fm1 / radius
        return (rp - rm) / 2j, (rp + rm) / 2  # vort, div

    vort_t, div_t = jax.vmap(uv_to_vrtdiv)(tends.u, tends.v)
    temp_t = jax.vmap(lambda f: s2_forward(f, L, sampling))(
        tends.virtual_temperature
    )
    lnps_t = s2_forward(tends.log_surface_pressure, L, sampling)
    tracer_t = jax.vmap(
        lambda field: jax.vmap(lambda f: s2_forward(f, L, sampling))(field)
    )(tends.tracers)

    vort_t = enforce_triangular_truncation(vort_t, L, T)
    div_t = enforce_triangular_truncation(div_t, L, T)
    temp_t = enforce_triangular_truncation(temp_t, L, T)
    lnps_t = enforce_triangular_truncation(lnps_t, L, T)
    tracer_t = jax.vmap(
        lambda field: enforce_triangular_truncation(field, L, T)
    )(tracer_t)
    return vort_t, div_t, temp_t, lnps_t, tracer_t


def advance_with_tendencies(
    spec_state, phys_tends, phis_grads, dyn_config, trans_config,
    stepper_config, latitudes, gauss_weights=None, pdryini=None,
    spec_tends=None,
):
    """Differentiable: dynamics step + time-split physics increment.

    Pure jnp; jit/grad/vmap-able. Applies a single time-split increment
    assembled from two independent tendency containers, either of which may
    be ``None``:

    * ``phys_tends`` : grid-space ``PhysicsTendencies`` — transformed to
      spectral internally (u, v -> vort/div; rest via ``s2_forward``).
    * ``spec_tends`` : spectral-space ``SpectralTendencies`` — added directly,
      no transform. This is the SKEB / SPPT injection point.

    With both ``None`` this is a dynamics-only step identical to ``advance``.
    """
    spec_new = advance(
        spec_state, phis_grads, dyn_config, trans_config, stepper_config,
        latitudes, gauss_weights, pdryini,
    )
    if phys_tends is None and spec_tends is None:
        return spec_new
    dt = stepper_config.dt

    # Accumulate the spectral increment from the grid path (if any)...
    vort_i = jnp.zeros_like(spec_new.vorticity)
    div_i = jnp.zeros_like(spec_new.divergence)
    temp_i = jnp.zeros_like(spec_new.temperature)
    lnps_i = jnp.zeros_like(spec_new.log_surface_pressure)
    tracer_i = jnp.zeros_like(spec_new.tracers)

    if phys_tends is not None:
        vort_t, div_t, temp_t, lnps_t, tracer_t = _physics_tendencies_to_spectral(
            phys_tends, trans_config, dyn_config
        )
        vort_i += vort_t
        div_i += div_t
        temp_i += temp_t
        lnps_i += lnps_t
        tracer_i += tracer_t

    # ...and add the spectral path directly (no grid->spectral transform).
    if spec_tends is not None:
        vort_i += spec_tends.d_vorticity_d_t
        div_i += spec_tends.d_divergence_d_t
        temp_i += spec_tends.d_temperature_d_t
        lnps_i += spec_tends.d_log_surface_pressure_d_t
        tracer_i += spec_tends.d_tracers_d_t

    return SpectralState(
        vorticity=spec_new.vorticity + dt * vort_i,
        divergence=spec_new.divergence + dt * div_i,
        temperature=spec_new.temperature + dt * temp_i,
        log_surface_pressure=spec_new.log_surface_pressure + dt * lnps_i,
        tracers=spec_new.tracers + dt * tracer_i,
    )
```

Confirm `SpectralState` and `SpectralTendencies` are already imported at the
top of `stepper.py` (they are:
`from .states import SpectralState, SpectralTendencies`). The caller builds
`spec_tends` from the existing `SpectralTendencies` struct; unused fields are
zero arrays of the matching spectral shape.

- [ ] **Step 4b: Add a spectral-tendency injection + composition test**

```python
# tests/test_jax_component_api.py (add to file)
from gfs_dynamical_core.jax.states import SpectralTendencies


def _zero_spec_tends(n_lev, L, n_tracers=1):
    z = lambda: jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128)
    return SpectralTendencies(
        d_vorticity_d_t=z(), d_divergence_d_t=z(), d_temperature_d_t=z(),
        d_log_surface_pressure_d_t=jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128),
        d_tracers_d_t=jnp.zeros((n_tracers, n_lev, L, 2 * L - 1), dtype=jnp.complex128),
    )


def test_spectral_vorticity_tendency_applied_directly():
    """A spectral vorticity tendency is added as spec_new.vorticity + dt*tend,
    with no grid round-trip (the SKEB injection path)."""
    L, n_lev = 8, 10
    dyn = _mock_dyn_config(n_lev)
    trans = TransformConfig(L=L, radius=1.0)
    sc = StepperConfig(dt=10.0, explicit=True)
    lat = get_gaussian_latitudes(L)
    n_lat, n_lon = trans.n_lat, trans.n_lon
    phis = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))
    state = _mock_state(n_lev, L)

    base = advance(state, phis, dyn, trans, sc, lat)
    dvort = _zero_spec_tends(n_lev, L)
    bump = jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128).at[:, 2, L].set(0.3)
    dvort = dvort.replace(d_vorticity_d_t=bump)

    out = advance_with_tendencies(
        state, None, phis, dyn, trans, sc, lat, spec_tends=dvort
    )
    assert jnp.allclose(out.vorticity, base.vorticity + sc.dt * bump)
    assert jnp.allclose(out.divergence, base.divergence)


def test_grid_and_spectral_paths_compose_additively():
    """phys_tends + spec_tends == sum of each applied alone (minus one base)."""
    L, n_lev = 8, 10
    dyn = _mock_dyn_config(n_lev)
    trans = TransformConfig(L=L, radius=1.0)
    sc = StepperConfig(dt=10.0, explicit=True)
    lat = get_gaussian_latitudes(L)
    n_lat, n_lon = trans.n_lat, trans.n_lon
    phis = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))
    state = _mock_state(n_lev, L)

    grid_t = PhysicsTendencies(
        u=0.01 * jnp.ones((n_lev, n_lat, n_lon)),
        v=jnp.zeros((n_lev, n_lat, n_lon)),
        virtual_temperature=jnp.zeros((n_lev, n_lat, n_lon)),
        log_surface_pressure=jnp.zeros((n_lat, n_lon)),
        tracers=jnp.zeros((1, n_lev, n_lat, n_lon)),
    )
    spec_t = _zero_spec_tends(n_lev, L).replace(
        d_temperature_d_t=jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128)
        .at[:, 1, L].set(0.2)
    )

    base = advance(state, phis, dyn, trans, sc, lat)
    only_grid = advance_with_tendencies(state, grid_t, phis, dyn, trans, sc, lat)
    only_spec = advance_with_tendencies(
        state, None, phis, dyn, trans, sc, lat, spec_tends=spec_t
    )
    both = advance_with_tendencies(
        state, grid_t, phis, dyn, trans, sc, lat, spec_tends=spec_t
    )
    # increments are additive about the common dynamics step
    for f in ("vorticity", "divergence", "temperature",
              "log_surface_pressure", "tracers"):
        expected = (getattr(only_grid, f) + getattr(only_spec, f)
                    - getattr(base, f))
        assert jnp.allclose(getattr(both, f), expected)
```

Run both: `pytest tests/test_jax_component_api.py -k "spectral or compose" -v`
Expected: PASS. (Task 3 adds the `jax.grad`-through-`spec_tends`
differentiability check.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_jax_component_api.py::test_advance_with_zero_tendencies_matches_advance -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gfs_dynamical_core/jax/stepper.py tests/test_jax_component_api.py
git commit -m "feat(jax): add pure advance_with_tendencies + PhysicsTendencies"
```

---

## Task 3: Differentiability smoke test for the pure core

**Files:**
- Test: `tests/test_jax_component_api.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_jax_component_api.py (add)
def test_advance_with_tendencies_is_differentiable():
    L, n_lev = 8, 10
    dyn = _mock_dyn_config(n_lev)
    trans = TransformConfig(L=L, radius=1.0)
    sc = StepperConfig(dt=10.0, explicit=True)
    lat = get_gaussian_latitudes(L)
    n_lat, n_lon = trans.n_lat, trans.n_lon
    phis = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))
    state = _mock_state(n_lev, L)

    def loss(scale):
        tends = PhysicsTendencies(
            u=scale * jnp.ones((n_lev, n_lat, n_lon)),
            v=jnp.zeros((n_lev, n_lat, n_lon)),
            virtual_temperature=jnp.zeros((n_lev, n_lat, n_lon)),
            log_surface_pressure=jnp.zeros((n_lat, n_lon)),
            tracers=jnp.zeros((1, n_lev, n_lat, n_lon)),
        )
        out = advance_with_tendencies(state, tends, phis, dyn, trans, sc, lat)
        return jnp.sum(jnp.abs(out.divergence) ** 2)

    g = jax.grad(loss)(1.0)
    assert jnp.isfinite(g)


def test_advance_with_spectral_tendencies_is_differentiable():
    """Gradients must flow through the spectral (spec_tends) injection path —
    this is the path SKEB/SPPT train through."""
    L, n_lev = 8, 10
    dyn = _mock_dyn_config(n_lev)
    trans = TransformConfig(L=L, radius=1.0)
    sc = StepperConfig(dt=10.0, explicit=True)
    lat = get_gaussian_latitudes(L)
    n_lat, n_lon = trans.n_lat, trans.n_lon
    phis = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))
    state = _mock_state(n_lev, L)

    def loss(amp):
        bump = jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128).at[:, 2, L].set(1.0)
        spec_t = _zero_spec_tends(n_lev, L).replace(d_vorticity_d_t=amp * bump)
        out = advance_with_tendencies(
            state, None, phis, dyn, trans, sc, lat, spec_tends=spec_t
        )
        return jnp.sum(jnp.abs(out.vorticity) ** 2)

    g = jax.grad(loss)(1.0)
    assert jnp.isfinite(g)
```

- [ ] **Step 2: Run test to verify it fails, then passes**

Run: `pytest tests/test_jax_component_api.py::test_advance_with_tendencies_is_differentiable -v`
Expected: PASS immediately (the function from Task 2 is already differentiable).
If it FAILS with a tracing/grad error, fix the offending non-differentiable
op in `advance_with_tendencies` / `_physics_tendencies_to_spectral` (e.g. a
stray `jax.device_get` or numpy call) before continuing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_jax_component_api.py
git commit -m "test(jax): differentiability smoke test for advance_with_tendencies"
```

---

## Task 4: Multi-tracer support in the dynamics/transform pipeline

**Files:**
- Modify (if needed): `gfs_dynamical_core/jax/dynamics.py`,
  `gfs_dynamical_core/jax/transforms.py`
- Test: `tests/test_jax_component_api.py`

This task confirms (and fixes if necessary) that `n_tracers > 1` flows through
`advance` cleanly. The mock state already supports `n_tracers`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_jax_component_api.py (add)
def test_advance_handles_two_tracers():
    L, n_lev = 8, 10
    dyn = _mock_dyn_config(n_lev)
    trans = TransformConfig(L=L, radius=1.0)
    sc = StepperConfig(dt=10.0, explicit=True)
    lat = get_gaussian_latitudes(L)
    n_lat, n_lon = trans.n_lat, trans.n_lon
    phis = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))
    state = _mock_state(n_lev, L, n_tracers=2)

    out = advance(state, phis, dyn, trans, sc, lat)
    assert out.tracers.shape == (2, n_lev, L, 2 * L - 1)
    assert not jnp.isnan(out.tracers).any()
```

- [ ] **Step 2: Run test to verify it passes or fails**

Run: `pytest tests/test_jax_component_api.py::test_advance_handles_two_tracers -v`
Expected: PASS if the pipeline already vmaps over the tracer axis. If it FAILS
(shape error or NaN), proceed to Step 3.

- [ ] **Step 3: Generalize single-tracer assumptions (only if Step 2 failed)**

Inspect `get_spectral_tendencies` / `full_dynamics_step` in
`gfs_dynamical_core/jax/dynamics.py` and the tracer transform calls in
`gfs_dynamical_core/jax/transforms.py`. Replace any indexing that assumes a
single tracer (e.g. `tracers[0]`) with a `jax.vmap` over the leading tracer
axis. Re-run until the test passes. Do **not** change the single-tracer
numerical result (existing `tests/test_jax_dynamics.py` must still pass).

- [ ] **Step 4: Run regression + new test**

Run: `pytest tests/test_jax_dynamics.py tests/test_jax_component_api.py::test_advance_handles_two_tracers -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gfs_dynamical_core/jax tests/test_jax_component_api.py
git commit -m "feat(jax): support multiple tracers through the dynamics pipeline"
```

---

## Task 5: Convert `GFSDynamicsJAX` to `TendencyStepper` with tracer machinery

**Files:**
- Modify: `gfs_dynamical_core/component_jax.py`
- Test: `tests/test_jax_component_api.py`

This is the central change. Keep the existing `array_call`/jnp pipeline working
(backward compatibility) while adding the component-driven `__call__`.

- [ ] **Step 1: Write the failing test (construction + property merge)**

```python
# tests/test_jax_component_api.py (add)
import climt


def test_jax_core_accepts_component_list_and_merges_properties():
    from gfs_dynamical_core.component_jax import GFSDynamicsJAX
    hs = climt.HeldSuarez()
    dycore = GFSDynamicsJAX(tendency_component_list=[hs])
    # air_pressure now an output (needed by internal components)
    assert "air_pressure" in dycore.output_properties
    assert "air_pressure_on_interface_levels" in dycore.output_properties
    # constructed without error and is a TendencyStepper
    from sympl import TendencyStepper
    assert isinstance(dycore, TendencyStepper)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_jax_component_api.py::test_jax_core_accepts_component_list_and_merges_properties -v`
Expected: FAIL — `GFSDynamicsJAX` is still a `Stepper`, has no
`tendency_component_list` arg, and `air_pressure` is not an output.

- [ ] **Step 3: Rework the class header, imports, and `__init__`**

In `gfs_dynamical_core/component_jax.py`, change imports:

```python
from sympl import (
    TendencyStepper, get_constant, get_tracer_names,
    ImplicitTendencyComponentComposite,
    get_numpy_arrays_with_properties,
    restore_data_arrays_with_properties,
)
from ._property_utils import GFSError, get_valid_properties
```

Change the class definition and add tracer attributes + `spectral_names`:

```python
class GFSDynamicsJAX(TendencyStepper):
    """JAX implementation of the GFS dynamical core (component-driven)."""

    uses_tracers = True
    tracer_dims = ("tracer", "mid_levels", "lat", "lon")
    prepend_tracers = (("specific_humidity", "kg/kg"),)

    # base (dynamics-only) property dicts; merged with component props in __init__
    _gfs_input_properties = dict(input_properties)   # see note below
    _gfs_output_properties = {
        "air_temperature": {"units": "K", "dims": ["mid_levels", "lat", "lon"]},
        "eastward_wind": {"units": "m s^-1", "dims": ["mid_levels", "lat", "lon"]},
        "northward_wind": {"units": "m s^-1", "dims": ["mid_levels", "lat", "lon"]},
        "surface_air_pressure": {"units": "Pa", "dims": ["lat", "lon"]},
        "specific_humidity": {"units": "kg kg^-1", "dims": ["mid_levels", "lat", "lon"]},
        "air_pressure": {"units": "Pa", "dims": ["mid_levels", "lat", "lon"]},
        "air_pressure_on_interface_levels": {
            "units": "Pa", "dims": ["interface_levels", "lat", "lon"]},
    }
    _gfs_diagnostic_properties = {}

    @property
    def spectral_names(self):
        return ("eastward_wind", "northward_wind", "air_temperature",
                "surface_air_pressure") + get_tracer_names()
```

> Note: the existing module-level `input_properties`/`output_properties` dicts
> become the `_gfs_*` base dicts. Move the current `input_properties` dict body
> into `_gfs_input_properties` (it already lists the a_coord/b_coord and all
> prognostic inputs). Add `air_pressure` to `_gfs_input_properties` as well
> (the core both consumes and produces it, like Fortran).

Rewrite `__init__` to mirror Fortran:

```python
    def __init__(self, tendency_component_list=None, adiabatic=False,
                 zero_negative_moisture=True, **kwargs):
        tendency_component_list = tendency_component_list or []
        self._tendency_component = ImplicitTendencyComponentComposite(
            *tendency_component_list
        )
        bad = set(self._tendency_component.diagnostic_properties.keys()
                  ).intersection(self.spectral_names)
        if bad:
            raise GFSError(
                "Components may not emit {} as diagnostics; these are stepped "
                "spectrally.".format(bad))

        self.input_properties = dict(self._gfs_input_properties)
        self.output_properties = dict(self._gfs_output_properties)
        self.diagnostic_properties = dict(self._gfs_diagnostic_properties)

        super().__init__(**kwargs)

        self.input_properties.update(get_valid_properties(
            self._gfs_input_properties,
            self._tendency_component.input_properties, "input"))
        self.output_properties.update(get_valid_properties(
            self._gfs_output_properties,
            self._tendency_component.tendency_properties, "output"))
        self.diagnostic_properties.update(get_valid_properties(
            self._gfs_diagnostic_properties,
            self._tendency_component.diagnostic_properties, "diagnostic"))

        self.adiabatic = adiabatic
        self._zero_negative_moisture = zero_negative_moisture
        # ---- existing jnp-pipeline state (unchanged) ----
        self.dyn_config = None
        self.trans_config = None
        self.stepper_config = None
        self._phis_grads = None
        self._latitudes = None
        self._gauss_weights = None
        self._pdryini = None
        self._jit_advance = jax.jit(advance)
        self._spec_state = None
        self._phys_tendencies = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_jax_component_api.py::test_jax_core_accepts_component_list_and_merges_properties -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add gfs_dynamical_core/component_jax.py tests/test_jax_component_api.py
git commit -m "feat(jax): GFSDynamicsJAX is now a TendencyStepper with tracer setup"
```

---

## Task 6: Pressure outputs in `array_call`

**Files:**
- Modify: `gfs_dynamical_core/component_jax.py`
- Test: `tests/test_jax_component_api.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_jax_component_api.py (add)
from datetime import timedelta


def _grid_state(L=16, n_lev=10):
    grid = climt.get_grid(nx=2 * L - 1, ny=L, nz=n_lev)
    from gfs_dynamical_core.component_jax import GFSDynamicsJAX
    dycore = GFSDynamicsJAX()
    state = climt.get_default_state([dycore], grid_state=grid)
    return dycore, state


def test_array_call_outputs_air_pressure():
    dycore, state = _grid_state()
    _, out = dycore(state, timestep=timedelta(minutes=10))
    assert "air_pressure" in out
    assert "air_pressure_on_interface_levels" in out
    ap = out["air_pressure"]
    assert ap.shape[0] == state["air_temperature"].shape[0]  # mid_levels
    assert (ap.values > 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_jax_component_api.py::test_array_call_outputs_air_pressure -v`
Expected: FAIL — `air_pressure` not in output.

- [ ] **Step 3: Compute and return pressure in `array_call`**

In `array_call`, after `spec_final` is computed and `grid_final` is obtained,
compute pressure diagnostics from the new surface pressure and add to the
returned dict:

```python
from .jax.dynamics import compute_pressure_diagnostics

pd = compute_pressure_diagnostics(grid_final.log_surface_pressure, self.dyn_config)
# pd.prs: (n_lev, n_lat, n_lon) layer-mean pressure (bottom-to-top)
# pd.pk : (n_lev+1, n_lat, n_lon) interface pressure (bottom-to-top)
```

Extend the returned output dict:

```python
return {}, {
    "air_temperature": to_numpy(grid_final.temperature),
    "eastward_wind": to_numpy(grid_final.u),
    "northward_wind": to_numpy(grid_final.v),
    "surface_air_pressure": to_numpy(jnp.exp(grid_final.log_surface_pressure)),
    "specific_humidity": to_numpy(grid_final.tracers[0]),
    "air_pressure": to_numpy(pd.prs),
    # climt expects interface pressure top-to-bottom; pipeline is bottom-to-top
    "air_pressure_on_interface_levels": to_numpy(pd.pk[::-1]),
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_jax_component_api.py::test_array_call_outputs_air_pressure -v`
Expected: PASS

- [ ] **Step 5: Run backward-compat regression**

Run: `pytest tests/test_jax_component.py tests/test_jax_stepper.py -v`
Expected: PASS (dynamics-only behavior unchanged).

- [ ] **Step 6: Commit**

```bash
git add gfs_dynamical_core/component_jax.py tests/test_jax_component_api.py
git commit -m "feat(jax): output air_pressure and interface pressure"
```

---

## Task 7: Component-driven `__call__` with full tendency construction

**Files:**
- Modify: `gfs_dynamical_core/component_jax.py`
- Test: `tests/test_jax_component_api.py`

Add the Fortran-style `__call__` and the tendency-building path in `array_call`.

- [ ] **Step 1: Write the failing test (equivalence)**

```python
# tests/test_jax_component_api.py (add)
import numpy as np


def test_component_path_matches_manual_tendencies():
    L, n_lev = 16, 10
    grid = climt.get_grid(nx=2 * L - 1, ny=L, nz=n_lev)
    hs = climt.HeldSuarez()
    from gfs_dynamical_core.component_jax import GFSDynamicsJAX
    from gfs_dynamical_core.jax.dynamics import compute_pressure_diagnostics
    import jax.numpy as jnp

    ts = timedelta(minutes=10)

    # --- component path ---
    dyn_a = GFSDynamicsJAX(tendency_component_list=[hs])
    state_a = climt.get_default_state([dyn_a], grid_state=grid)
    _, out_a = dyn_a(state_a, timestep=ts)

    # --- manual path: same forcing pushed via set_physics_tendencies ---
    dyn_b = GFSDynamicsJAX()  # no components
    state_b = climt.get_default_state([dyn_b], grid_state=grid)
    # provide 3-D pressure for hs on the manual path
    # (kick the lazy config init with one zero-tendency call)
    _ = dyn_b  # configs build on first array_call
    # Build pressure consistent with current ps for hs:
    # (mirror the legacy driver lines 80-87)
    # NB: requires one call to populate dyn_b.dyn_config; do a dynamics-only
    # step on a *copy* is overkill — instead compute via a fresh config.
    # Simplest: run the component path's first step forcing through manually.
    tend, _ = hs(state_b)
    u_t = tend["eastward_wind"].transpose("mid_levels", "lat", "lon").values
    v_t = tend["northward_wind"].transpose("mid_levels", "lat", "lon").values
    t_t = tend["air_temperature"].transpose("mid_levels", "lat", "lon").values
    dyn_b.set_physics_tendencies(u_t, v_t, t_t)
    _, out_b = dyn_b(state_b, timestep=ts)

    # HS produces only u/v/T tendencies (no moisture), so virtual-T parity
    # reduces to the plain-T path and the two must agree closely.
    np.testing.assert_allclose(
        out_a["eastward_wind"].values, out_b["eastward_wind"].values,
        rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        out_a["air_temperature"].values, out_b["air_temperature"].values,
        rtol=1e-6, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_jax_component_api.py::test_component_path_matches_manual_tendencies -v`
Expected: FAIL — `GFSDynamicsJAX` has no component-driven `__call__`; the
component path does not apply any forcing yet.

- [ ] **Step 3: Add `__call__` and tendency construction**

Add `fvirt` constant handling to `__init__` (after constants are needed) or
compute lazily in `array_call`:

```python
def _fvirt(self):
    rd = get_constant("gas_constant_of_dry_air", "J kg^-1 K^-1")
    rv = get_constant("gas_constant_of_vapor_phase", "J kg^-1 K^-1")
    return (1 - rd / rv) / (rd / rv)
```

Add the component-driven `__call__` (mirrors `component.py:301-393`,
simplified — no air_pressure assertions, JAX does its own stepping):

```python
def __call__(self, state, timestep):
    raw_state = get_numpy_arrays_with_properties(state, self.input_properties)
    raw_state["tracers"] = self._tracer_packer.pack(state)
    raw_state["time"] = state["time"]

    tendencies, diagnostics = self._tendency_component(state, timestep)
    for name, value in tendencies.items():
        if name in self.input_properties:
            tendencies[name] = value.to_units(
                self.input_properties[name]["units"] + " s^-1")

    raw_diag, raw_new = self.array_call(
        raw_state, timestep, prognostic_tendencies=tendencies)

    new_state = self._tracer_packer.unpack(raw_new.pop("tracers"), state)
    new_state.update(restore_data_arrays_with_properties(
        raw_new, self.output_properties, state, self.input_properties,
        ignore_missing=True))
    diagnostics.update(restore_data_arrays_with_properties(
        raw_diag, self.diagnostic_properties, state, self.input_properties,
        ignore_missing=True))
    for key in state.keys():
        if key not in new_state:
            new_state[key] = state[key]
    return diagnostics, new_state
```

Update `array_call` signature and convert component tendencies into the jnp
physics-tendency channel (reusing `_phys_tendencies`). At the top of
`array_call`, before the spectral advance:

```python
def array_call(self, state, timestep, prognostic_tendencies=None):
    prognostic_tendencies = prognostic_tendencies or {}
    ...
    # after configs/spec_orig are ready, build physics tendencies from
    # the component output (grid space, bottom-to-top):
    if prognostic_tendencies:
        nlev, nlat, nlon = temp.shape
        def tend_or_zero(name, shape):
            if name in prognostic_tendencies:
                arr = prognostic_tendencies[name].to_units(
                    self.input_properties[name]["units"] + " s^-1"
                ).transpose(*self.input_properties[name]["dims"]).values
                return jnp.asarray(np.ascontiguousarray(arr))
            return jnp.zeros(shape)

        u_t = tend_or_zero("eastward_wind", (nlev, nlat, nlon))
        v_t = tend_or_zero("northward_wind", (nlev, nlat, nlon))
        t_t = tend_or_zero("air_temperature", (nlev, nlat, nlon))
        ps_t = tend_or_zero("surface_air_pressure", (nlat, nlon))
        q = jnp.asarray(state["specific_humidity"]) \
            if "specific_humidity" in state else jnp.asarray(state["tracers"][0])
        fvirt = self._fvirt()
        t_virt = jnp.asarray(temp) * (1 + fvirt * q)
        # tracer tendencies, packed order
        n_tracers = state["tracers"].shape[0]
        tracer_t = jnp.stack([
            tend_or_zero(name, (nlev, nlat, nlon))
            for name in self._tracer_packer.tracer_names
        ], axis=0) if n_tracers else jnp.zeros((0, nlev, nlat, nlon))
        q_t = tracer_t[0] if n_tracers else jnp.zeros((nlev, nlat, nlon))
        virtual_temp_tend = t_t * (1 + fvirt * q) + fvirt * t_virt * q_t
        ps = jnp.asarray(state["surface_air_pressure"])
        lnps_tend = ps_t / ps
        from .jax.stepper import PhysicsTendencies
        self._phys_tendencies = PhysicsTendencies(
            u=u_t, v=v_t, virtual_temperature=virtual_temp_tend,
            log_surface_pressure=lnps_tend, tracers=tracer_t,
        )
```

Replace the existing `_apply_physics_tendencies` call with the unified
`advance_with_tendencies` path. Where the code currently does:

```python
spec_final = self._jit_advance(spec_orig, ...)
if self._phys_tendencies is not None:
    spec_final = self._apply_physics_tendencies(spec_final, dt)
```

change to use the pure function (jit it once in `__init__`:
`self._jit_advance_t = jax.jit(advance_with_tendencies)`):

```python
if self._phys_tendencies is not None:
    spec_final = self._jit_advance_t(
        spec_orig, self._phys_tendencies, self._phis_grads,
        self.dyn_config, self.trans_config, self.stepper_config,
        self._latitudes, self._gauss_weights, self._pdryini)
else:
    spec_final = self._jit_advance(
        spec_orig, self._phis_grads, self.dyn_config, self.trans_config,
        self.stepper_config, self._latitudes, self._gauss_weights,
        self._pdryini)
```

Keep the legacy `set_physics_tendencies(u, v, t)` method but have it construct a
`PhysicsTendencies` with zero lnps/tracers and `virtual_temperature=t_tend`
(so the manual path still works). Delete the now-unused
`_apply_physics_tendencies` method (its logic moved to
`_physics_tendencies_to_spectral` in `stepper.py`).

Add at top of file: `import numpy as np` (already present) and
`from .jax.stepper import advance, advance_with_tendencies, PhysicsTendencies`.

- [ ] **Step 4: Run the equivalence test**

Run: `pytest tests/test_jax_component_api.py::test_component_path_matches_manual_tendencies -v`
Expected: PASS

- [ ] **Step 5: Run full regression**

Run: `pytest tests/test_jax_component.py tests/test_jax_stepper.py tests/test_jax_component_api.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add gfs_dynamical_core/component_jax.py tests/test_jax_component_api.py
git commit -m "feat(jax): component-driven __call__ with full tendency parity"
```

---

## Task 8: Multi-tracer pack/unpack through a full step

**Files:**
- Test: `tests/test_jax_component_api.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_jax_component_api.py (add)
def test_multi_tracer_roundtrip_through_step():
    from sympl import register_tracer
    register_tracer("test_tracer", "kg/kg")
    try:
        L, n_lev = 16, 10
        grid = climt.get_grid(nx=2 * L - 1, ny=L, nz=n_lev)
        from gfs_dynamical_core.component_jax import GFSDynamicsJAX
        dycore = GFSDynamicsJAX()
        state = climt.get_default_state([dycore], grid_state=grid)
        assert "test_tracer" in state
        _, out = dycore(state, timestep=timedelta(minutes=10))
        assert "test_tracer" in out
        assert out["test_tracer"].shape == state["test_tracer"].shape
        assert not np.isnan(out["test_tracer"].values).any()
    finally:
        # avoid leaking the tracer registration into other tests
        from sympl._core.tracers import _tracer_unit_dict
        _tracer_unit_dict.pop("test_tracer", None)
```

> If the private `_tracer_unit_dict` cleanup path is unavailable in the
> installed sympl version, replace the `finally` body with a `pytest` fixture
> that snapshots/restores `get_tracer_names()` instead. The functional
> assertion (round-trip shape + no NaN) is the point.

- [ ] **Step 2: Run test**

Run: `pytest tests/test_jax_component_api.py::test_multi_tracer_roundtrip_through_step -v`
Expected: PASS (Task 4 generalized the pipeline; Task 5 set up packing).
If FAIL on shape/packing, revisit Task 4/Task 5 tracer handling.

- [ ] **Step 3: Commit**

```bash
git add tests/test_jax_component_api.py
git commit -m "test(jax): multi-tracer pack/unpack round-trip through a step"
```

---

## Task 9: Simplify the Held-Suarez driver

**Files:**
- Modify: `debugging_code/heldsuarez_run.py`

- [ ] **Step 1: Collapse the jax branch to the shared path**

Replace the `MODE`-specific construction (lines 49-60) so both modes build a
component-driven core:

```python
hs = climt.HeldSuarez()

if MODE == "fortran":
    from gfs_dynamical_core import GFSDynamicalCore
    dycore = GFSDynamicalCore(tendency_component_list=[hs])
else:
    from gfs_dynamical_core.component_jax import GFSDynamicsJAX
    dycore = GFSDynamicsJAX(tendency_component_list=[hs])

state = climt.get_default_state([dycore], grid_state=grid)
```

Remove the per-step jax block (lines 79-92: the `_dyn_cfg`/pressure
reconstruction, `hs(state)`, `set_physics_tendencies`). The loop body becomes
mode-agnostic:

```python
for i in range(1, N_STEPS + 1):
    _, out = dycore(state, timestep=timestep)
    state.update(out)
    state["time"] += timestep
    ...
```

Delete the now-unused imports (`compute_pressure_diagnostics`, `jnp`) from the
jax branch and the `_dyn_cfg = None` initialization.

- [ ] **Step 2: Smoke-run a few steps (manual / optional)**

Run (in the climt env, if the Fortran extension + JAX are available):
`python debugging_code/heldsuarez_run.py jax 0.05 0.0`
Expected: prints a day-0 line, exits 0, writes `hs_zonal_mean_jax.npz`. This is
a manual smoke check; it is not part of the pytest suite.

- [ ] **Step 3: Commit**

```bash
git add debugging_code/heldsuarez_run.py
git commit -m "refactor(driver): use component-driven JAX core in heldsuarez_run"
```

---

## Task 10: Full regression sweep + final commit

**Files:** none (verification)

- [ ] **Step 1: Run the JAX test suite**

Run: `pytest tests/test_jax_states.py tests/test_jax_transforms.py tests/test_jax_dynamics.py tests/test_jax_stepper.py tests/test_jax_component.py tests/test_jax_component_api.py -v`
Expected: all PASS.

- [ ] **Step 2: Run flake8 as CI does**

Run: `flake8 --ignore=E501,E226,W503,W504,W605 gfs_dynamical_core/ --count`
Expected: `0`.

- [ ] **Step 3: Final commit if anything was touched**

```bash
git add -A
git commit -m "chore: finalize JAX component-driven API parity"
```

---

## Self-Review Notes (coverage map)

| Spec section | Task(s) |
|---|---|
| §1 Class structure / tracer machinery | 1, 5 |
| §2 Pressure outputs | 6 |
| §3 Full tendency parity (virtual-T, lnps, tracers) | 7 |
| §4 Multi-tracer generalization | 4, 8 |
| §5 Differentiable pure function | 2, 3 |
| §6 Driver simplification | 9 |
| Testing (all bullets) | 2,3,4,6,7,8,10 |
| Backward compatibility | 5,6,10 |

**Known verification points carried from the spec** (resolve during execution,
not deferrable):
- Task 4 confirms multi-tracer vmap; generalize if the pipeline assumes one tracer.
- Task 7 must check the JAX dynamics temperature convention so the virtual-T
  tendency is applied consistently (the HS equivalence test in Task 7 guards this
  for the moisture-free case; add a moist-component check if a moist forcing is
  introduced later).
