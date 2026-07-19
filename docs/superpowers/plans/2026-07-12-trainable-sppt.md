# Trainable SPPT Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete, differentiable, trainable SPPT pipeline in `examples/stochastic_sppt/` — data generation, the SPPT model, SLURM training, and analysis — producing a calibrated stochastic T42 Held-Suarez model.

**Architecture:** Everything is a pure JAX function built on the existing `gfs_dynamical_core.jax` substrate (`advance_with_tendencies`, `spectral_to_grid`, `enforce_triangular_truncation`). Held-Suarez forcing produces grid-space physics tendencies; a spectral AR(1) pattern (Palmer 2009) is transformed to grid space and multiplies those tendencies; a checkpointed `lax.scan` rolls out members via `vmap`; almost-fair CRPS (Lang 2024) trains the three SPPT scalars via `optax`.

**Tech Stack:** JAX (x64), flax.struct, optax, numpy, matplotlib, climt/sympl (setup-time only for grid coefficients + constants), pytest.

**Spec:** `docs/specs/wp5_trainable_sppt_design.md`. **References:** `docs/refs/` (Palmer 2009, Leutbecher 2017, Lang 2024).

## Global Constraints

- **Float64 everywhere:** every entry point and test sets `os.environ["JAX_ENABLE_X64"] = "True"` before importing jax (a `conftest.py` handles tests).
- **Purity:** all model/loss code is pure `jnp`, `jit`/`grad`/`vmap`-able. climt/sympl may be used ONLY inside `build_model` (setup), never in the hot loop.
- **Reality guard:** every spectral noise field passes through `enforce_triangular_truncation(flm, L, T)` (it enforces both triangular truncation and the reality condition `f_{l,-m}=(-1)^m conj(f_{l,+m})`).
- **Reverse-mode memory:** the per-step rollout body is wrapped in `jax.checkpoint`.
- **Common random numbers:** member keys derive deterministically from `(base_key, case_index, member_index)` so they are identical within a gradient evaluation.
- **Loss:** almost-fair CRPS with `alpha = 0.95` default (`alpha=1.0` ⇒ fair CRPS).
- **Verification fields:** `u850, v850, t500, vort500, ps`.
- **Resolutions** (`(L, ntrunc, dt_seconds)`): `T21=(32,21,1800)`, `T42=(64,42,1200)`, `T85=(128,85,600)`, `T127=(192,127,450)`.
- **Test runner:** `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest`.
- **Cluster:** SLURM, single GPU.
- Commit after every task. Never commit the `docs/refs/*.pdf` binaries.

---

## File Structure

```
examples/stochastic_sppt/
  __init__.py
  config.py          # ModelConfig, ExperimentConfig, TrainConfig, RESOLUTIONS
  model.py           # build_model, ModelBundle, step, rest_state, spectral_truncate
  held_suarez.py     # HSConfig, hs_tendencies
  sppt.py            # SPPTParams, sigma_n, init_pattern, pattern_step, pattern_to_grid, vertical_taper, apply_sppt
  diagnostics.py     # vertical_interp, extract_fields
  metrics.py         # afcrps, fair_crps, spread, rmse_of_mean, spread_error_ratio, rank_histogram
  rollout.py         # ensemble_rollout, member_rollout
  generate_data.py   # main + mode_a + mode_b + save/load dataset
  train.py           # main + loss_fn + train loop
  analyze.py         # main + compute_metrics + plots
  slurm/
    generate_data.sbatch
    train.sbatch
  README.md
  tests/
    conftest.py
    test_pattern.py
    test_sppt.py
    test_diagnostics.py
    test_metrics.py
    test_rollout.py
    test_generate_data.py
    test_training.py
    test_analyze.py
```

---

### Task 1: Package scaffold + config

**Files:**
- Create: `examples/stochastic_sppt/__init__.py`
- Create: `examples/stochastic_sppt/config.py`
- Create: `examples/stochastic_sppt/tests/__init__.py`
- Create: `examples/stochastic_sppt/tests/conftest.py`
- Test: `examples/stochastic_sppt/tests/test_config.py`

**Interfaces:**
- Produces: `RESOLUTIONS: dict[str, tuple[int,int,float]]`; `ModelConfig(resolution:str="T42", n_lev:int=20)` with derived props `.L`, `.ntrunc`, `.dt`; `ExperimentConfig(mode:str, forecast_resolution:str, truth_resolution:str, n_members:int, lead_days:tuple, n_cases:int, spinup_days:float, case_stride_days:float, ic_perturb_amp:float, seed:int)`; `TrainConfig(alpha:float, lr:float, n_opt_steps:int, batch_cases:int, log_every:int, checkpoint_dir:str, seed:int)`.

- [ ] **Step 1: Write `conftest.py` (shared x64/cpu setup for all tests)**

```python
# examples/stochastic_sppt/tests/conftest.py
import os

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
```

- [ ] **Step 2: Write the failing test**

```python
# examples/stochastic_sppt/tests/test_config.py
from examples.stochastic_sppt.config import ModelConfig, ExperimentConfig, TrainConfig, RESOLUTIONS


def test_resolutions_present():
    assert RESOLUTIONS["T42"] == (64, 42, 1200.0)
    assert set(RESOLUTIONS) == {"T21", "T42", "T85", "T127"}


def test_model_config_derived():
    mc = ModelConfig(resolution="T42")
    assert mc.L == 64 and mc.ntrunc == 42 and mc.dt == 1200.0 and mc.n_lev == 20


def test_experiment_and_train_defaults():
    ec = ExperimentConfig(mode="A", forecast_resolution="T21", truth_resolution="T21")
    assert ec.n_members >= 2 and ec.lead_days[-1] == 10
    tc = TrainConfig()
    assert 0.0 < tc.alpha <= 1.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'examples.stochastic_sppt.config'`

- [ ] **Step 4: Write `__init__.py` files and `config.py`**

```python
# examples/stochastic_sppt/__init__.py
```

```python
# examples/stochastic_sppt/tests/__init__.py
```

```python
# examples/stochastic_sppt/config.py
"""Configuration dataclasses for the trainable-SPPT pipeline."""
from dataclasses import dataclass, field

# (L, ntrunc, dt_seconds). L = s2fft bandlimit (= n_lat, n_lon = 2L-1);
# ntrunc = physical triangular truncation; dt chosen for CFL at each resolution.
RESOLUTIONS: dict[str, tuple[int, int, float]] = {
    "T21": (32, 21, 1800.0),
    "T42": (64, 42, 1200.0),
    "T85": (128, 85, 600.0),
    "T127": (192, 127, 450.0),
}


@dataclass(frozen=True)
class ModelConfig:
    resolution: str = "T42"
    n_lev: int = 20

    @property
    def L(self) -> int:
        return RESOLUTIONS[self.resolution][0]

    @property
    def ntrunc(self) -> int:
        return RESOLUTIONS[self.resolution][1]

    @property
    def dt(self) -> float:
        return RESOLUTIONS[self.resolution][2]


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str                       # "A" (parameter recovery) or "B" (model-error)
    forecast_resolution: str        # e.g. "T42"
    truth_resolution: str           # e.g. "T127" (Mode B) or == forecast (Mode A)
    n_members: int = 8
    lead_days: tuple = (3, 5, 7, 10)
    n_cases: int = 16
    spinup_days: float = 200.0
    case_stride_days: float = 5.0
    ic_perturb_amp: float = 0.0     # 0 => SPPT-only spread
    seed: int = 0


@dataclass(frozen=True)
class TrainConfig:
    alpha: float = 0.95
    lr: float = 5e-2
    n_opt_steps: int = 300
    batch_cases: int = 4
    log_every: int = 10
    checkpoint_dir: str = "sppt_checkpoints"
    seed: int = 0
```

- [ ] **Step 5: Run test to verify it passes**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_config.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add examples/stochastic_sppt/__init__.py examples/stochastic_sppt/config.py examples/stochastic_sppt/tests/
git commit -m "feat(sppt): package scaffold and config dataclasses"
```

---

### Task 2: Pure-JAX model harness (`build_model`, `step`, `rest_state`, `spectral_truncate`)

**Files:**
- Create: `examples/stochastic_sppt/model.py`
- Test: `examples/stochastic_sppt/tests/test_model.py`

**Interfaces:**
- Consumes: `ModelConfig` (Task 1); `gfs_dynamical_core.jax.dynamics.DynamicsConfig`, `.transforms.{TransformConfig, prebuild_kernels, get_gaussian_latitudes, grid_to_spectral, enforce_triangular_truncation}`, `.stepper.{StepperConfig, init_semi_implicit_matrices, init_diffusion_operators, advance_with_tendencies}`, `.states.{GridState, SpectralState}`.
- Produces: `ModelBundle` (flax struct with fields `dyn_config, trans_config, stepper_config, latitudes, gauss_weights, pdryini, phis_grads, dt, n_lev`); `build_model(model_config) -> ModelBundle`; `step(bundle, spec_state, phys_tends) -> SpectralState`; `rest_state(bundle, key, t0=250.0, ps0=1.0e5, noise=1.0e-3) -> SpectralState`; `spectral_truncate(spec_state, bundle_from, bundle_to) -> SpectralState`.

- [ ] **Step 1: Write the failing test**

```python
# examples/stochastic_sppt/tests/test_model.py
import jax
import jax.numpy as jnp
from examples.stochastic_sppt.config import ModelConfig
from examples.stochastic_sppt.model import build_model, step, rest_state, spectral_truncate


def test_build_and_dynamics_only_step_is_finite():
    bundle = build_model(ModelConfig(resolution="T21", n_lev=20))
    assert bundle.trans_config.L == 32 and bundle.trans_config.truncation == 21
    spec = rest_state(bundle, jax.random.PRNGKey(0))
    # dynamics-only step (no physics tendencies)
    spec2 = step(bundle, spec, None)
    assert spec2.temperature.shape == (20, 32, 63)
    assert bool(jnp.all(jnp.isfinite(spec2.temperature)))


def test_spectral_truncate_zeros_high_wavenumbers():
    hi = build_model(ModelConfig(resolution="T42", n_lev=20))
    lo = build_model(ModelConfig(resolution="T21", n_lev=20))
    spec_hi = rest_state(hi, jax.random.PRNGKey(1))
    spec_lo = spectral_truncate(spec_hi, hi, lo)
    assert spec_lo.temperature.shape == (20, 32, 63)
    assert bool(jnp.all(jnp.isfinite(spec_lo.temperature)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'examples.stochastic_sppt.model'`

- [ ] **Step 3: Write `model.py`**

Note: `build_model` sources hybrid coefficients `ak, bk` and physical constants from climt/sympl exactly as `gfs_dynamical_core/component_jax.py::array_call` does (lines 266-346), factored into a pure setup function. `phis_grads` are zero (flat topography). `pdryini`/`gauss_weights` enable the dry-mass fixer.

```python
# examples/stochastic_sppt/model.py
"""Pure-JAX Held-Suarez model harness built on gfs_dynamical_core.jax."""
import climt
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from sympl import get_constant, set_constant

from gfs_dynamical_core.jax.dynamics import DynamicsConfig
from gfs_dynamical_core.jax.states import GridState, SpectralState
from gfs_dynamical_core.jax.stepper import (
    StepperConfig,
    advance_with_tendencies,
    init_diffusion_operators,
    init_semi_implicit_matrices,
)
from gfs_dynamical_core.jax.transforms import (
    TransformConfig,
    enforce_triangular_truncation,
    get_gaussian_latitudes,
    grid_to_spectral,
    prebuild_kernels,
)


@struct.dataclass
class ModelBundle:
    dyn_config: DynamicsConfig
    trans_config: TransformConfig
    stepper_config: StepperConfig
    latitudes: jnp.ndarray
    gauss_weights: jnp.ndarray
    pdryini: float = struct.field(pytree_node=False)
    phis_grads: tuple
    dt: float = struct.field(pytree_node=False)
    n_lev: int = struct.field(pytree_node=False)


def _ak_bk_constants(L, n_lev):
    """Fetch ak/bk and Earth constants from climt (setup-time only)."""
    set_constant("reference_air_pressure", value=1e5, units="Pa")
    grid = climt.get_grid(nx=2 * L - 1, ny=L, nz=n_lev)
    ak = jnp.array(np.asarray(grid["a_coord"].values, dtype=np.float64))
    bk = jnp.array(np.asarray(grid["b_coord"].values, dtype=np.float64))
    c = dict(
        rd=get_constant("gas_constant_of_dry_air", "J kg^-1 K^-1"),
        cp=get_constant("heat_capacity_of_dry_air_at_constant_pressure", "J kg^-1 K^-1"),
        rv=get_constant("gas_constant_of_vapor_phase", "J kg^-1 K^-1"),
        cvap=get_constant("heat_capacity_of_vapor_phase", "J kg^-1 K^-1"),
        toa_pressure=get_constant("top_of_model_pressure", "Pa"),
        radius=get_constant("planetary_radius", "m"),
        omega=get_constant("planetary_rotation_rate", "s^-1"),
        g=get_constant("gravitational_acceleration", "m s^-2"),
    )
    return ak, bk, c


def build_model(model_config) -> ModelBundle:
    L, ntrunc, dt = model_config.L, model_config.ntrunc, model_config.dt
    n_lev = model_config.n_lev
    ak, bk, c = _ak_bk_constants(L, n_lev)

    dbk = bk[:-1] - bk[1:]
    ck = ak[:-1] * bk[1:] - ak[1:] * bk[:-1]
    dyn_config = DynamicsConfig(
        ak=ak, bk=bk, ck=ck, dbk=dbk, rk=c["rd"] / c["cp"],
        toa_pressure=c["toa_pressure"], radius=c["radius"], omega=c["omega"],
        g=c["g"], rd=c["rd"], rv=c["rv"], cp=c["cp"], cvap=c["cvap"],
    )
    trans_config = TransformConfig(L=L, sampling="gl", ntrunc=ntrunc, radius=c["radius"])
    prebuild_kernels(L, "gl")

    _sc = StepperConfig(dt=dt)
    si = init_semi_implicit_matrices(dyn_config, trans_config, dt,
                                     aa22=_sc.aa22, aa33=_sc.aa33, bb4=_sc.bb4)
    diff = init_diffusion_operators(dyn_config, trans_config, dt)
    stepper_config = StepperConfig(
        dt=dt, explicit=False,
        amhyb=si["amhyb"], bmhyb=si["bmhyb"], tor_hyb=si["tor_hyb"],
        svhyb=si["svhyb"], d_hyb_m=si["d_hyb_m"],
        disspec=diff["disspec"], diff_prof=diff["diff_prof"], dmp_prof=diff["dmp_prof"],
    )

    latitudes = get_gaussian_latitudes(L)
    _, raw_w = np.polynomial.legendre.leggauss(L)
    gauss_weights = jnp.array(raw_w / 2.0)
    n_lat, n_lon = L, 2 * L - 1
    phis_grads = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))

    # Dry initial mean surface pressure (q=0): pdryini = area-mean ps.
    pdryini = float(jnp.sum(gauss_weights[:, None] * jnp.full((n_lat, n_lon), 1e5)) / n_lon)

    return ModelBundle(
        dyn_config=dyn_config, trans_config=trans_config, stepper_config=stepper_config,
        latitudes=latitudes, gauss_weights=gauss_weights, pdryini=pdryini,
        phis_grads=phis_grads, dt=dt, n_lev=n_lev,
    )


def step(bundle, spec_state, phys_tends):
    """One differentiable model step (dynamics + optional physics increment)."""
    return advance_with_tendencies(
        spec_state, phys_tends, bundle.phis_grads, bundle.dyn_config,
        bundle.trans_config, bundle.stepper_config, bundle.latitudes,
        bundle.gauss_weights, bundle.pdryini,
    )


def rest_state(bundle, key, t0=250.0, ps0=1.0e5, noise=1.0e-3) -> SpectralState:
    """Isothermal rest state + tiny temperature noise, in spectral space."""
    n_lev = bundle.n_lev
    n_lat, n_lon = bundle.trans_config.L, 2 * bundle.trans_config.L - 1
    temp = t0 + noise * jax.random.normal(key, (n_lev, n_lat, n_lon))
    zeros3 = jnp.zeros((n_lev, n_lat, n_lon))
    grid = GridState(
        u=zeros3, v=zeros3, temperature=temp,
        vorticity=zeros3, divergence=zeros3,
        log_surface_pressure=jnp.full((n_lat, n_lon), jnp.log(ps0)),
        tracers=jnp.zeros((1, n_lev, n_lat, n_lon)),
    )
    return grid_to_spectral(grid, bundle.trans_config)


def spectral_truncate(spec_state, bundle_from, bundle_to) -> SpectralState:
    """Coarse-grain a spectral state from a higher L to a lower L by extracting
    the central (l, m) block and re-enforcing the target triangular truncation."""
    Lf = bundle_from.trans_config.L
    Lt = bundle_to.trans_config.L
    Tt = bundle_to.trans_config.truncation
    m0 = Lf - 1  # index of m=0 in the (2Lf-1) axis
    lo, hi = m0 - (Lt - 1), m0 + (Lt - 1) + 1  # central 2Lt-1 columns

    def cut(flm):
        return flm[..., :Lt, lo:hi]

    def cut_enf(flm):
        return enforce_triangular_truncation(cut(flm), Lt, Tt)

    return SpectralState(
        vorticity=cut_enf(spec_state.vorticity),
        divergence=cut_enf(spec_state.divergence),
        temperature=cut_enf(spec_state.temperature),
        log_surface_pressure=cut_enf(spec_state.log_surface_pressure),
        tracers=cut_enf(spec_state.tracers),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_model.py -v`
Expected: PASS (2 passed). First run builds s2fft kernels (may take ~30 s).

- [ ] **Step 5: Commit**

```bash
git add examples/stochastic_sppt/model.py examples/stochastic_sppt/tests/test_model.py
git commit -m "feat(sppt): pure-JAX Held-Suarez model harness (build_model/step/rest_state/truncate)"
```

---

### Task 3: Held-Suarez forcing (`hs_tendencies`)

**Files:**
- Create: `examples/stochastic_sppt/held_suarez.py`
- Test: `examples/stochastic_sppt/tests/test_held_suarez.py`

**Interfaces:**
- Consumes: `gfs_dynamical_core.jax.states.{GridState, PhysicsTendencies}` (PhysicsTendencies is exported from `gfs_dynamical_core.jax.stepper`); `dynamics.compute_pressure_diagnostics`; `ModelBundle` (for `dyn_config`, `latitudes`).
- Produces: `HSConfig` (frozen dataclass: `k_a, k_s, k_f, dT_y, dtheta_z, sigma_b, p0, t_min`); `hs_tendencies(grid_state, bundle) -> PhysicsTendencies`.

- [ ] **Step 1: Write the failing test**

```python
# examples/stochastic_sppt/tests/test_held_suarez.py
import jax
import jax.numpy as jnp
from examples.stochastic_sppt.config import ModelConfig
from examples.stochastic_sppt.model import build_model, rest_state
from examples.stochastic_sppt.held_suarez import hs_tendencies
from gfs_dynamical_core.jax.transforms import spectral_to_grid


def _grid(bundle, seed=0):
    spec = rest_state(bundle, jax.random.PRNGKey(seed), t0=300.0)
    grid, _ = spectral_to_grid(spec, bundle.trans_config)
    return grid


def test_temperature_relaxes_toward_equilibrium():
    bundle = build_model(ModelConfig(resolution="T21"))
    grid = _grid(bundle)
    tends = hs_tendencies(grid, bundle)
    # At 300 K uniform, T is above equilibrium almost everywhere -> cooling.
    assert float(jnp.mean(tends.virtual_temperature)) < 0.0
    assert tends.virtual_temperature.shape == grid.temperature.shape


def test_friction_only_in_boundary_layer_and_opposes_wind():
    bundle = build_model(ModelConfig(resolution="T21"))
    grid = _grid(bundle)
    # inject a wind so friction is nonzero
    grid = grid.replace(u=grid.u + 10.0)
    tends = hs_tendencies(grid, bundle)
    # top level (k=-1, near TOA, sigma << sigma_b) has ~zero friction
    assert float(jnp.max(jnp.abs(tends.u[-1]))) < 1e-6
    # bottom level (k=0, surface) friction opposes the +10 m/s wind
    assert float(jnp.mean(tends.u[0])) < 0.0
    # lnps and tracer tendencies are zero (dry core)
    assert float(jnp.max(jnp.abs(tends.log_surface_pressure))) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_held_suarez.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'examples.stochastic_sppt.held_suarez'`

- [ ] **Step 3: Write `held_suarez.py`**

```python
# examples/stochastic_sppt/held_suarez.py
"""Pure-JAX Held & Suarez (1994) forcing as grid-space physics tendencies."""
from dataclasses import dataclass

import jax.numpy as jnp

from gfs_dynamical_core.jax.dynamics import compute_pressure_diagnostics
from gfs_dynamical_core.jax.stepper import PhysicsTendencies

_DAY = 86400.0


@dataclass(frozen=True)
class HSConfig:
    k_a: float = 1.0 / (40.0 * _DAY)
    k_s: float = 1.0 / (4.0 * _DAY)
    k_f: float = 1.0 / (1.0 * _DAY)
    dT_y: float = 60.0
    dtheta_z: float = 10.0
    sigma_b: float = 0.7
    p0: float = 1.0e5
    t_min: float = 200.0


def hs_tendencies(grid_state, bundle, config: HSConfig = HSConfig()) -> PhysicsTendencies:
    dyn = bundle.dyn_config
    lat = bundle.latitudes[None, :, None]        # (1, n_lat, 1)
    sinphi2 = jnp.sin(lat) ** 2
    cosphi2 = jnp.cos(lat) ** 2
    cosphi4 = cosphi2 ** 2

    press = compute_pressure_diagnostics(grid_state.log_surface_pressure, dyn)
    p = press.prs                                 # (n_lev, n_lat, n_lon) layer mean pressure
    ps = press.ps[None, :, :]                     # (1, n_lat, n_lon)
    sigma = p / ps
    kappa = dyn.rk

    # Equilibrium temperature T_eq(phi, p)
    t_eq = (315.0 - config.dT_y * sinphi2
            - config.dtheta_z * jnp.log(p / config.p0) * cosphi2) * (p / config.p0) ** kappa
    t_eq = jnp.maximum(config.t_min, t_eq)

    # Thermal relaxation rate
    frac = jnp.clip((sigma - config.sigma_b) / (1.0 - config.sigma_b), 0.0, None)
    k_t = config.k_a + (config.k_s - config.k_a) * frac * cosphi4
    temp_tend = -k_t * (grid_state.temperature - t_eq)

    # Rayleigh friction in the boundary layer
    k_v = config.k_f * frac
    u_tend = -k_v * grid_state.u
    v_tend = -k_v * grid_state.v

    zeros_lnps = jnp.zeros_like(grid_state.log_surface_pressure)
    zeros_tracers = jnp.zeros_like(grid_state.tracers)
    # Dry core: virtual temperature tendency == temperature tendency (q=0).
    return PhysicsTendencies(
        u=u_tend, v=v_tend, virtual_temperature=temp_tend,
        log_surface_pressure=zeros_lnps, tracers=zeros_tracers,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_held_suarez.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add examples/stochastic_sppt/held_suarez.py examples/stochastic_sppt/tests/test_held_suarez.py
git commit -m "feat(sppt): pure-JAX Held-Suarez forcing tendencies"
```

---

### Task 4: SPPT spectral pattern generator

**Files:**
- Create: `examples/stochastic_sppt/sppt.py`
- Test: `examples/stochastic_sppt/tests/test_pattern.py`

**Interfaces:**
- Consumes: `gfs_dynamical_core.jax.transforms.{s2_inverse, enforce_triangular_truncation}`; `ModelBundle.trans_config`.
- Produces: `SPPTParams` (flax struct: `log_sigma, log_tau, log_len`); `default_params() -> SPPTParams`; `sigma_n(params, trans_config, dt) -> (L,) array`; `init_pattern(params, key, trans_config, dt) -> (L,2L-1) complex`; `pattern_step(r_lm, key, params, trans_config, dt) -> (L,2L-1) complex`; `pattern_to_grid(r_lm, trans_config) -> (n_lat,n_lon) real`.

Design notes (Palmer 2009 App. 8.1): `phi=exp(-dt/tau)`, `sigma_n = F0*exp(-kappaT*n(n+1)/2)`, `kappaT=(len/radius)**2/2`, and `F0` normalizes grid-point variance to `sigma**2`. Complex noise is sampled on the full `(L,2L-1)` rectangle then reality-symmetrized + truncated by `enforce_triangular_truncation`. The `F0` closed form is validated by `test_grid_variance_matches_sigma`; the module exposes a single `_VAR_NORM` constant (default 1.0) the implementer calibrates there if the transform convention adds a constant factor.

- [ ] **Step 1: Write the failing tests**

```python
# examples/stochastic_sppt/tests/test_pattern.py
import jax
import jax.numpy as jnp
from examples.stochastic_sppt.config import ModelConfig
from examples.stochastic_sppt.model import build_model
from examples.stochastic_sppt import sppt


def _tc():
    return build_model(ModelConfig(resolution="T21")).trans_config


def test_pattern_is_reality_symmetric():
    tc = _tc()
    p = sppt.default_params()
    r = sppt.init_pattern(p, jax.random.PRNGKey(0), tc, dt=1800.0)
    # grid transform of a reality-symmetric field must be (numerically) real
    g = sppt.pattern_to_grid(r, tc)
    assert g.dtype.kind == "f"  # pattern_to_grid returns real part
    # reality condition: r[l,-m] == (-1)^m conj(r[l,m])
    L = tc.L
    m = jnp.arange(-L + 1, L)
    sign = jnp.where(m % 2 == 0, 1.0, -1.0)
    mirror = sign * jnp.conj(r[..., ::-1])
    neg = m < 0
    assert float(jnp.max(jnp.abs((r - mirror)[:, neg]))) < 1e-9


def test_grid_variance_matches_sigma():
    tc = _tc()
    p = sppt.SPPTParams(log_sigma=jnp.log(0.5), log_tau=jnp.log(6 * 3600.0),
                        log_len=jnp.log(500e3))
    keys = jax.random.split(jax.random.PRNGKey(1), 200)
    grids = jax.vmap(lambda k: sppt.pattern_to_grid(
        sppt.init_pattern(p, k, tc, dt=1800.0), tc))(keys)
    # grid-point variance should be ~ sigma**2 = 0.25 (within 20%)
    v = float(jnp.var(grids))
    assert 0.20 < v < 0.30


def test_ar1_stationarity():
    tc = _tc()
    p = sppt.default_params()
    key = jax.random.PRNGKey(2)
    r = sppt.init_pattern(p, key, tc, dt=1800.0)
    v0 = float(jnp.var(sppt.pattern_to_grid(r, tc)))
    for i in range(30):
        r = sppt.pattern_step(r, jax.random.fold_in(key, i), p, tc, dt=1800.0)
    v1 = float(jnp.var(sppt.pattern_to_grid(r, tc)))
    # variance stays within a factor ~1.6 (single-sample, stationary process)
    assert 0.5 < (v1 / v0) < 2.0


def test_pattern_is_differentiable_in_params():
    tc = _tc()
    key = jax.random.PRNGKey(3)

    def scalar(log_sigma):
        p = sppt.SPPTParams(log_sigma=log_sigma, log_tau=jnp.log(6 * 3600.0),
                            log_len=jnp.log(500e3))
        return jnp.sum(sppt.pattern_to_grid(sppt.init_pattern(p, key, tc, 1800.0), tc) ** 2)

    g = jax.grad(scalar)(jnp.log(0.5))
    assert jnp.isfinite(g) and float(g) != 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_pattern.py -v`
Expected: FAIL with `AttributeError: module 'examples.stochastic_sppt.sppt' has no attribute 'default_params'`

- [ ] **Step 3: Write the pattern-generator part of `sppt.py`**

```python
# examples/stochastic_sppt/sppt.py
"""Trainable differentiable SPPT (Palmer et al. 2009, Tech Memo 598, App. 8.1)."""
import jax
import jax.numpy as jnp
from flax import struct

from gfs_dynamical_core.jax.transforms import enforce_triangular_truncation, s2_inverse

# Calibration constant for the grid-point variance normalization; kept explicit
# so the analytic F0 (Palmer eq. 18) can be reconciled with the s2fft transform
# convention via test_grid_variance_matches_sigma. Default 1.0 (no correction).
_VAR_NORM = 1.0


@struct.dataclass
class SPPTParams:
    log_sigma: jnp.ndarray   # grid-point std of the pattern r  (var(r)=sigma^2)
    log_tau: jnp.ndarray     # decorrelation time (s)
    log_len: jnp.ndarray     # correlation length (m)


def default_params() -> SPPTParams:
    return SPPTParams(log_sigma=jnp.log(0.5), log_tau=jnp.log(6 * 3600.0),
                      log_len=jnp.log(500e3))


def _phi(params, dt):
    return jnp.exp(-dt / jnp.exp(params.log_tau))


def sigma_n(params, trans_config, dt):
    """Per-total-wavenumber forcing std (Palmer eqs. 17-18)."""
    L = trans_config.L
    radius = trans_config.radius
    n = jnp.arange(L)
    kappa_t = (jnp.exp(params.log_len) / radius) ** 2 / 2.0
    shape = jnp.exp(-kappa_t * n * (n + 1) / 2.0)          # exp(-kappaT n(n+1)/2)
    var_r = jnp.exp(2.0 * params.log_sigma)
    phi = _phi(params, dt)
    denom = 2.0 * jnp.sum((2 * n[1:] + 1) * jnp.exp(-kappa_t * n[1:] * (n[1:] + 1)))
    F0 = jnp.sqrt(_VAR_NORM * var_r * (1.0 - phi ** 2) / denom)
    return F0 * shape                                      # (L,)


def _sample_eta(key, L):
    """Complex Gaussian noise on the (L, 2L-1) rectangle (Re,Im ~ N(0,1))."""
    kr, ki = jax.random.split(key)
    re = jax.random.normal(kr, (L, 2 * L - 1))
    im = jax.random.normal(ki, (L, 2 * L - 1))
    eta = re + 1j * im
    # clip to +-10 std (overflow guard, Palmer App 8.1)
    eta = jnp.clip(eta.real, -10.0, 10.0) + 1j * jnp.clip(eta.imag, -10.0, 10.0)
    return eta


def init_pattern(params, key, trans_config, dt):
    """Stationary initialization r_lm(0) = (1-phi^2)^-1/2 sigma_n eta (Palmer eq. 19)."""
    L, T = trans_config.L, trans_config.truncation
    phi = _phi(params, dt)
    sn = sigma_n(params, trans_config, dt)[:, None]
    r = (1.0 - phi ** 2) ** (-0.5) * sn * _sample_eta(key, L)
    return enforce_triangular_truncation(r, L, T)


def pattern_step(r_lm, key, params, trans_config, dt):
    """One AR(1) step: r_lm <- phi r_lm + sigma_n eta (Palmer eq. 14)."""
    L, T = trans_config.L, trans_config.truncation
    phi = _phi(params, dt)
    sn = sigma_n(params, trans_config, dt)[:, None]
    r = phi * r_lm + sn * _sample_eta(key, L)
    return enforce_triangular_truncation(r, L, T)


def pattern_to_grid(r_lm, trans_config):
    """Inverse transform to a real grid-space pattern."""
    g = s2_inverse(r_lm, trans_config.L, trans_config.sampling)
    return g.real
```

- [ ] **Step 4: Run tests; calibrate `_VAR_NORM` if the variance test fails**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_pattern.py -v`
Expected: PASS (4 passed). If `test_grid_variance_matches_sigma` fails with a measured variance `v`, set `_VAR_NORM = 0.25 / v` (using this test's sigma^2=0.25) and re-run — this reconciles the analytic F0 with the s2fft normalization. Do not change the test tolerance.

- [ ] **Step 5: Commit**

```bash
git add examples/stochastic_sppt/sppt.py examples/stochastic_sppt/tests/test_pattern.py
git commit -m "feat(sppt): spectral AR(1) pattern generator (Palmer 2009 App 8.1)"
```

---

### Task 5: SPPT vertical taper + application

**Files:**
- Modify: `examples/stochastic_sppt/sppt.py` (append `vertical_taper`, `apply_sppt`)
- Test: `examples/stochastic_sppt/tests/test_sppt.py`

**Interfaces:**
- Consumes: `SPPTParams`, `pattern_to_grid` (Task 4); `compute_pressure_diagnostics`; `PhysicsTendencies`.
- Produces: `vertical_taper(bundle, p_surf_taper=(85000.0, 95000.0), p_strat_taper=(10000.0, 5000.0)) -> (n_lev,)`; `apply_sppt(phys_tends, r_grid, mu, params) -> PhysicsTendencies`.

Taper `mu(p)`: 0 below `p_surf_taper[1]` (deepest, ~950 hPa) ramping smoothly to 1 by `p_surf_taper[0]` (~850 hPa); 1 through the troposphere; ramping to 0 between `p_strat_taper[0]` (~100 hPa) and `p_strat_taper[1]` (~50 hPa) in the stratosphere. Uses reference `ps=1e5` mid-level pressures.

- [ ] **Step 1: Write the failing tests**

```python
# examples/stochastic_sppt/tests/test_sppt.py
import jax
import jax.numpy as jnp
from examples.stochastic_sppt.config import ModelConfig
from examples.stochastic_sppt.model import build_model, rest_state
from examples.stochastic_sppt.held_suarez import hs_tendencies
from examples.stochastic_sppt import sppt
from gfs_dynamical_core.jax.transforms import spectral_to_grid


def _setup(seed=0):
    bundle = build_model(ModelConfig(resolution="T21"))
    grid, _ = spectral_to_grid(rest_state(bundle, jax.random.PRNGKey(seed), t0=300.0),
                               bundle.trans_config)
    grid = grid.replace(u=grid.u + 10.0)
    tends = hs_tendencies(grid, bundle)
    return bundle, tends


def test_taper_zero_at_surface_one_in_troposphere():
    bundle, _ = _setup()
    mu = sppt.vertical_taper(bundle)
    assert mu.shape == (bundle.n_lev,)
    assert float(mu[0]) < 0.05           # surface level suppressed
    assert float(jnp.max(mu)) > 0.95     # full amplitude somewhere in troposphere


def test_zero_amplitude_leaves_tendencies_unchanged():
    bundle, tends = _setup()
    p = sppt.SPPTParams(log_sigma=jnp.log(1e-30), log_tau=jnp.log(6 * 3600.0),
                        log_len=jnp.log(500e3))
    r = sppt.pattern_to_grid(sppt.init_pattern(p, jax.random.PRNGKey(1),
                                               bundle.trans_config, bundle.dt), bundle.trans_config)
    mu = sppt.vertical_taper(bundle)
    out = sppt.apply_sppt(tends, r, mu, p)
    assert float(jnp.max(jnp.abs(out.virtual_temperature - tends.virtual_temperature))) < 1e-6


def test_apply_scales_uvt_only_and_clips():
    bundle, tends = _setup()
    p = sppt.SPPTParams(log_sigma=jnp.log(0.5), log_tau=jnp.log(6 * 3600.0),
                        log_len=jnp.log(500e3))
    r = sppt.pattern_to_grid(sppt.init_pattern(p, jax.random.PRNGKey(2),
                                               bundle.trans_config, bundle.dt), bundle.trans_config)
    mu = sppt.vertical_taper(bundle)
    out = sppt.apply_sppt(tends, r, mu, p)
    # factor bounded to [0.1, 1.9] -> perturbed tendency within 1.9x original
    ratio = jnp.abs(out.u) / (jnp.abs(tends.u) + 1e-30)
    assert float(jnp.max(ratio)) <= 1.9 + 1e-6
    # lnps/tracers untouched
    assert float(jnp.max(jnp.abs(out.log_surface_pressure))) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_sppt.py -v`
Expected: FAIL with `AttributeError: module 'examples.stochastic_sppt.sppt' has no attribute 'vertical_taper'`

- [ ] **Step 3: Append to `sppt.py`**

```python
# --- append to examples/stochastic_sppt/sppt.py ---
from gfs_dynamical_core.jax.dynamics import compute_pressure_diagnostics
from gfs_dynamical_core.jax.stepper import PhysicsTendencies


def _ramp(x, x0, x1):
    """Smooth 0->1 ramp (sin^2) as x goes from x0 to x1 (either order)."""
    t = jnp.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    return jnp.sin(0.5 * jnp.pi * t) ** 2


def vertical_taper(bundle, p_surf_taper=(85000.0, 95000.0),
                   p_strat_taper=(10000.0, 5000.0)):
    """Per-level taper mu(p) in [0,1] (Palmer: zero near surface & stratosphere)."""
    lnps_ref = jnp.log(jnp.full((bundle.trans_config.L, 2 * bundle.trans_config.L - 1), 1e5))
    prs = compute_pressure_diagnostics(lnps_ref, bundle.dyn_config).prs  # (n_lev,lat,lon)
    p = prs[:, 0, 0]                                                     # column (n_lev,)
    surf = _ramp(p, p_surf_taper[1], p_surf_taper[0])   # 0 at 950 hPa -> 1 at 850 hPa
    strat = _ramp(p, p_strat_taper[1], p_strat_taper[0])  # 0 at 50 hPa -> 1 at 100 hPa
    return surf * strat


def apply_sppt(phys_tends, r_grid, mu, params):
    """Xp = (1 + clip(mu*r, -0.9, 0.9)) * Xc for X in {u, v, virtual_temperature}."""
    sigma = jnp.exp(params.log_sigma)
    r_clipped = jnp.clip(r_grid, -3.0 * sigma, 3.0 * sigma)             # +-3 sigma bound
    factor = 1.0 + jnp.clip(mu[:, None, None] * r_clipped[None, :, :], -0.9, 0.9)
    return PhysicsTendencies(
        u=phys_tends.u * factor,
        v=phys_tends.v * factor,
        virtual_temperature=phys_tends.virtual_temperature * factor,
        log_surface_pressure=phys_tends.log_surface_pressure,
        tracers=phys_tends.tracers,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_sppt.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add examples/stochastic_sppt/sppt.py examples/stochastic_sppt/tests/test_sppt.py
git commit -m "feat(sppt): vertical taper and multiplicative tendency application"
```

---

### Task 6: Verification diagnostics

**Files:**
- Create: `examples/stochastic_sppt/diagnostics.py`
- Test: `examples/stochastic_sppt/tests/test_diagnostics.py`

**Interfaces:**
- Consumes: `spectral_to_grid`; `compute_pressure_diagnostics`; `ModelBundle`.
- Produces: `vertical_interp(field, prs, p_target) -> (n_lat,n_lon)`; `extract_fields(spec_state, bundle) -> dict[str, (n_lat,n_lon)]` with keys `u850, v850, t500, vort500, ps`.

- [ ] **Step 1: Write the failing tests**

```python
# examples/stochastic_sppt/tests/test_diagnostics.py
import jax
import jax.numpy as jnp
from examples.stochastic_sppt.config import ModelConfig
from examples.stochastic_sppt.model import build_model, rest_state
from examples.stochastic_sppt import diagnostics


def test_extract_fields_shapes_and_ps():
    bundle = build_model(ModelConfig(resolution="T21"))
    spec = rest_state(bundle, jax.random.PRNGKey(0), ps0=1.0e5)
    f = diagnostics.extract_fields(spec, bundle)
    assert set(f) == {"u850", "v850", "t500", "vort500", "ps"}
    assert f["u850"].shape == (32, 63)
    assert abs(float(jnp.mean(f["ps"])) - 1.0e5) < 1.0  # ps = exp(lnps)


def test_vertical_interp_recovers_level_value():
    # linear-in-log-p field: interpolating at an exact layer pressure recovers it
    nlev, nlat, nlon = 5, 2, 2
    prs = jnp.array([9e4, 7e4, 5e4, 3e4, 1e4])[:, None, None] * jnp.ones((1, nlat, nlon))
    field = jnp.arange(nlev)[:, None, None] * jnp.ones((1, nlat, nlon)) * 1.0
    out = diagnostics.vertical_interp(field, prs, 5e4)
    assert float(jnp.max(jnp.abs(out - 2.0))) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_diagnostics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'examples.stochastic_sppt.diagnostics'`

- [ ] **Step 3: Write `diagnostics.py`**

```python
# examples/stochastic_sppt/diagnostics.py
"""Verification-field diagnostics from a spectral state."""
import jax
import jax.numpy as jnp

from gfs_dynamical_core.jax.dynamics import compute_pressure_diagnostics
from gfs_dynamical_core.jax.transforms import spectral_to_grid


def vertical_interp(field, prs, p_target):
    """Log-pressure-linear interpolation of `field` (n_lev,lat,lon) to a target
    pressure. `prs` is layer-mean pressure (n_lev,lat,lon), decreasing with k."""
    logp = jnp.log(prs)                          # decreasing in k
    logpt = jnp.log(p_target)

    def col(f_col, lp_col):
        # jnp.interp needs increasing xp -> reverse (top->surface gives increasing logp)
        return jnp.interp(logpt, lp_col[::-1], f_col[::-1])

    return jax.vmap(jax.vmap(col, in_axes=(1, 1)), in_axes=(2, 2))(field, logp).T


def extract_fields(spec_state, bundle):
    grid, _ = spectral_to_grid(spec_state, bundle.trans_config)
    press = compute_pressure_diagnostics(grid.log_surface_pressure, bundle.dyn_config)
    prs = press.prs
    return {
        "u850": vertical_interp(grid.u, prs, 85000.0),
        "v850": vertical_interp(grid.v, prs, 85000.0),
        "t500": vertical_interp(grid.temperature, prs, 50000.0),
        "vort500": vertical_interp(grid.vorticity, prs, 50000.0),
        "ps": press.ps,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_diagnostics.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add examples/stochastic_sppt/diagnostics.py examples/stochastic_sppt/tests/test_diagnostics.py
git commit -m "feat(sppt): verification-field diagnostics (u850/v850/t500/vort500/ps)"
```

---

### Task 7: CRPS metrics

**Files:**
- Create: `examples/stochastic_sppt/metrics.py`
- Test: `examples/stochastic_sppt/tests/test_metrics.py`

**Interfaces:**
- Produces: `afcrps(ensemble, truth, alpha=0.95) -> scalar`; `fair_crps(ensemble, truth) -> scalar`; `spread(ensemble) -> scalar`; `rmse_of_mean(ensemble, truth) -> scalar`; `spread_error_ratio(ensemble, truth) -> scalar`; `rank_histogram(ensemble, truth, n_bins=None) -> (M+1,) counts`. `ensemble` shape `(M, ...)`, `truth` shape `(...)`.

- [ ] **Step 1: Write the failing tests**

```python
# examples/stochastic_sppt/tests/test_metrics.py
import jax
import jax.numpy as jnp
from examples.stochastic_sppt import metrics


def test_afcrps_alpha1_equals_fair():
    ens = jax.random.normal(jax.random.PRNGKey(0), (8, 5, 5))
    y = jax.random.normal(jax.random.PRNGKey(1), (5, 5))
    a = metrics.afcrps(ens, y, alpha=1.0)
    f = metrics.fair_crps(ens, y)
    assert abs(float(a) - float(f)) < 1e-10


def test_afcrps_nonnegative_and_differentiable():
    ens = jax.random.normal(jax.random.PRNGKey(2), (6, 4))
    y = jax.random.normal(jax.random.PRNGKey(3), (4,))
    assert float(metrics.afcrps(ens, y, 0.95)) >= 0.0

    def loss(scale):
        return metrics.afcrps(ens * scale, y, 0.95)

    g = jax.grad(loss)(1.0)
    # finite-difference check
    eps = 1e-4
    fd = (loss(1.0 + eps) - loss(1.0 - eps)) / (2 * eps)
    assert abs(float(g) - float(fd)) < 1e-3


def test_reliable_ensemble_spread_error_ratio_near_one():
    key = jax.random.PRNGKey(4)
    M, N = 40, 4000
    truth = jax.random.normal(key, (N,))
    # members = truth + N(0,1): a reliable (M+1)/M-consistent ensemble
    noise = jax.random.normal(jax.random.PRNGKey(5), (M, N))
    ens = truth[None] + noise
    r = float(metrics.spread_error_ratio(ens, truth))
    assert 0.9 < r < 1.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'examples.stochastic_sppt.metrics'`

- [ ] **Step 3: Write `metrics.py`**

```python
# examples/stochastic_sppt/metrics.py
"""Almost-fair CRPS (Lang et al. 2024) and ensemble-calibration diagnostics."""
import jax.numpy as jnp


def afcrps(ensemble, truth, alpha=0.95):
    """Almost-fair CRPS (Lang et al. 2024, eq. 3/4). alpha=1 -> fair CRPS.
    ensemble: (M, ...), truth: (...). Returns the mean score over trailing dims."""
    M = ensemble.shape[0]
    eps = (1.0 - alpha) / M
    e1 = jnp.mean(jnp.abs(ensemble - truth[None]), axis=0)
    diff = jnp.abs(ensemble[:, None] - ensemble[None, :])   # (M, M, ...)
    e2 = jnp.sum(diff, axis=(0, 1)) / (2.0 * M * (M - 1))
    return jnp.mean(e1 - (1.0 - eps) * e2)


def fair_crps(ensemble, truth):
    return afcrps(ensemble, truth, alpha=1.0)


def spread(ensemble):
    """RMS ensemble standard deviation (finite-M unbiased, ddof=1)."""
    var = jnp.var(ensemble, axis=0, ddof=1)
    return jnp.sqrt(jnp.mean(var))


def rmse_of_mean(ensemble, truth):
    err = jnp.mean(ensemble, axis=0) - truth
    return jnp.sqrt(jnp.mean(err ** 2))


def spread_error_ratio(ensemble, truth):
    """Reliable ensemble => ratio ~ 1 (finite-M correction on the error side)."""
    M = ensemble.shape[0]
    return spread(ensemble) / (jnp.sqrt((M + 1.0) / M) * rmse_of_mean(ensemble, truth))


def rank_histogram(ensemble, truth):
    """Counts of the truth's rank among the M members (length M+1)."""
    M = ensemble.shape[0]
    rank = jnp.sum(ensemble < truth[None], axis=0).ravel()   # 0..M
    return jnp.bincount(rank, length=M + 1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_metrics.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add examples/stochastic_sppt/metrics.py examples/stochastic_sppt/tests/test_metrics.py
git commit -m "feat(sppt): almost-fair CRPS and calibration diagnostics"
```

---

### Task 8: Checkpointed ensemble rollout

**Files:**
- Create: `examples/stochastic_sppt/rollout.py`
- Test: `examples/stochastic_sppt/tests/test_rollout.py`

**Interfaces:**
- Consumes: `model.step`, `hs_tendencies`, `sppt.{init_pattern, pattern_step, pattern_to_grid, vertical_taper, apply_sppt}`, `diagnostics.extract_fields`.
- Produces: `member_rollout(params, spec0, key, bundle, lead_steps, mu) -> dict[str,(n_leads,n_lat,n_lon)]` (single member); `ensemble_rollout(params, spec0, keys, bundle, lead_steps) -> dict[str,(M,n_leads,n_lat,n_lon)]` (`vmap` over `keys`). `lead_steps` is an increasing tuple of step indices.

Design: `mu = vertical_taper(bundle)` is computed once and passed in. Each member advances the state and the AR(1) pattern together; the per-step body is `jax.checkpoint`-wrapped; diagnostics are captured at each lead via sequential `lax.scan` segments (no per-step outputs → bounded memory).

- [ ] **Step 1: Write the failing tests**

```python
# examples/stochastic_sppt/tests/test_rollout.py
import jax
import jax.numpy as jnp
from examples.stochastic_sppt.config import ModelConfig
from examples.stochastic_sppt.model import build_model, rest_state
from examples.stochastic_sppt import sppt, rollout


def test_zero_amplitude_members_identical():
    bundle = build_model(ModelConfig(resolution="T21"))
    spec0 = rest_state(bundle, jax.random.PRNGKey(0), t0=280.0)
    p = sppt.SPPTParams(log_sigma=jnp.log(1e-30), log_tau=jnp.log(6 * 3600.0),
                        log_len=jnp.log(500e3))
    keys = jax.random.split(jax.random.PRNGKey(1), 3)
    out = rollout.ensemble_rollout(p, spec0, keys, bundle, lead_steps=(2, 4))
    assert out["t500"].shape == (3, 2, 32, 63)
    # with ~zero noise, all members coincide
    assert float(jnp.std(out["t500"], axis=0).max()) < 1e-6


def test_rollout_is_differentiable_and_finite():
    bundle = build_model(ModelConfig(resolution="T21"))
    spec0 = rest_state(bundle, jax.random.PRNGKey(0), t0=280.0)
    keys = jax.random.split(jax.random.PRNGKey(2), 4)

    def loss(log_sigma):
        p = sppt.SPPTParams(log_sigma=log_sigma, log_tau=jnp.log(6 * 3600.0),
                            log_len=jnp.log(500e3))
        out = rollout.ensemble_rollout(p, spec0, keys, bundle, lead_steps=(3,))
        return jnp.mean(jnp.var(out["u850"], axis=0))

    g = jax.grad(loss)(jnp.log(0.5))
    assert jnp.isfinite(g)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_rollout.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'examples.stochastic_sppt.rollout'`

- [ ] **Step 3: Write `rollout.py`**

```python
# examples/stochastic_sppt/rollout.py
"""Checkpointed, vmap-able ensemble rollout with SPPT-perturbed physics."""
import jax
import jax.numpy as jnp

from gfs_dynamical_core.jax.transforms import spectral_to_grid

from . import sppt
from .diagnostics import extract_fields
from .held_suarez import hs_tendencies
from .model import step


def _make_body(bundle, mu):
    def body(carry, key):
        spec, r_lm = carry
        grid, _ = spectral_to_grid(spec, bundle.trans_config)
        phys = hs_tendencies(grid, bundle)
        r_grid = sppt.pattern_to_grid(r_lm, bundle.trans_config)
        phys = sppt.apply_sppt(phys, r_grid, mu, _params_holder[0])
        spec = step(bundle, spec, phys)
        r_lm = sppt.pattern_step(r_lm, key, _params_holder[0], bundle.trans_config, bundle.dt)
        return (spec, r_lm), None

    return jax.checkpoint(body)


# params flow through a closure captured per call (see member_rollout).
_params_holder = [None]


def member_rollout(params, spec0, key, bundle, lead_steps, mu):
    _params_holder[0] = params
    r0 = sppt.init_pattern(params, key, bundle.trans_config, bundle.dt)
    body = _make_body(bundle, mu)

    fields = {k: [] for k in ("u850", "v850", "t500", "vort500", "ps")}
    carry = (spec0, r0)
    prev = 0
    step_keys = jax.random.split(key, int(lead_steps[-1]))
    for lead in lead_steps:
        seg_keys = step_keys[prev:int(lead)]
        carry, _ = jax.lax.scan(body, carry, seg_keys)
        f = extract_fields(carry[0], bundle)
        for k in fields:
            fields[k].append(f[k])
        prev = int(lead)
    return {k: jnp.stack(v, axis=0) for k, v in fields.items()}  # (n_leads, lat, lon)


def ensemble_rollout(params, spec0, keys, bundle, lead_steps):
    mu = sppt.vertical_taper(bundle)
    fn = lambda key: member_rollout(params, spec0, key, bundle, lead_steps, mu)
    return jax.vmap(fn)(keys)  # (M, n_leads, lat, lon) per field
```

Note: `_params_holder` avoids threading `params` through the scan carry while keeping them a differentiable closure variable; it is set immediately before the traced `scan` in each `member_rollout` call and never read across concurrent traces (vmap maps over `keys`, not `params`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_rollout.py -v`
Expected: PASS (2 passed). If the closure-based `_params_holder` triggers a JAX tracer-leak error under `grad`, replace it by adding `params` to the scan carry (constant, passed through unchanged) — the test set is the acceptance gate.

- [ ] **Step 5: Commit**

```bash
git add examples/stochastic_sppt/rollout.py examples/stochastic_sppt/tests/test_rollout.py
git commit -m "feat(sppt): checkpointed vmap ensemble rollout"
```

---

### Task 9: Data generation (Mode A + Mode B)

**Files:**
- Create: `examples/stochastic_sppt/generate_data.py`
- Test: `examples/stochastic_sppt/tests/test_generate_data.py`

**Interfaces:**
- Consumes: `build_model`, `rest_state`, `spectral_truncate`, `step`, `hs_tendencies`, `sppt.*`, `diagnostics.extract_fields`, config dataclasses.
- Produces: `nature_run(bundle, key, n_days, save_stride_days) -> list[SpectralState]`; `mode_b_dataset(exp) -> dict`; `mode_a_dataset(exp, true_params) -> dict`; `save_dataset(path, data)` / `load_dataset(path) -> dict`; `main(argv=None)` CLI. Dataset dict keys: `ic_specs` (stacked truncated forecast ICs, one per case), `truth` (dict field->(n_case,n_lead,lat,lon)), `lead_steps`, `forecast_resolution`, `n_lev`.

- [ ] **Step 1: Write the failing test (tiny config)**

```python
# examples/stochastic_sppt/tests/test_generate_data.py
import os
import jax.numpy as jnp
from examples.stochastic_sppt.config import ExperimentConfig
from examples.stochastic_sppt import generate_data


def _tiny(mode):
    return ExperimentConfig(
        mode=mode, forecast_resolution="T21",
        truth_resolution=("T42" if mode == "B" else "T21"),
        n_members=3, lead_days=(1,), n_cases=2, spinup_days=1.0,
        case_stride_days=1.0, seed=0,
    )


def test_mode_a_dataset_shapes(tmp_path):
    from examples.stochastic_sppt import sppt
    true_p = sppt.default_params()
    data = generate_data.mode_a_dataset(_tiny("A"), true_p)
    assert data["truth"]["t500"].shape[0] == 2       # n_cases
    assert data["truth"]["t500"].shape[1] == 1       # n_leads
    p = tmp_path / "ds.npz"
    generate_data.save_dataset(str(p), data)
    back = generate_data.load_dataset(str(p))
    assert set(back["truth"]) == {"u850", "v850", "t500", "vort500", "ps"}


def test_mode_b_truncates_truth_to_forecast_grid():
    data = generate_data.mode_b_dataset(_tiny("B"))
    # forecast grid is T21 -> n_lat=32, n_lon=63
    assert data["truth"]["ps"].shape[-2:] == (32, 63)
    assert data["ic_specs"].temperature.shape[-2:] == (32, 63)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_generate_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'examples.stochastic_sppt.generate_data'`

- [ ] **Step 3: Write `generate_data.py`**

```python
# examples/stochastic_sppt/generate_data.py
"""Generate identical-twin truth datasets (Mode A recovery / Mode B model-error)."""
import argparse
import pickle

import jax
import jax.numpy as jnp

from gfs_dynamical_core.jax.states import SpectralState
from gfs_dynamical_core.jax.transforms import spectral_to_grid

from . import sppt
from .config import ExperimentConfig, ModelConfig
from .diagnostics import extract_fields
from .held_suarez import hs_tendencies
from .model import build_model, rest_state, spectral_truncate, step

_FIELDS = ("u850", "v850", "t500", "vort500", "ps")


def _deterministic_advance(bundle, spec, n_steps):
    def body(s, _):
        grid, _g = spectral_to_grid(s, bundle.trans_config)
        return step(bundle, s, hs_tendencies(grid, bundle)), None
    out, _ = jax.lax.scan(body, spec, None, length=int(n_steps))
    return out


def nature_run(bundle, key, n_days, save_stride_days):
    steps_per_day = int(round(86400.0 / bundle.dt))
    stride = int(round(save_stride_days * steps_per_day))
    spec = rest_state(bundle, key, t0=280.0)
    spec = _deterministic_advance(bundle, spec, int(n_days * steps_per_day))
    return spec, stride, steps_per_day


def _sppt_advance(bundle, spec, params, key, n_steps):
    mu = sppt.vertical_taper(bundle)
    r = sppt.init_pattern(params, key, bundle.trans_config, bundle.dt)
    keys = jax.random.split(key, int(n_steps))

    def body(carry, k):
        s, rl = carry
        grid, _ = spectral_to_grid(s, bundle.trans_config)
        phys = sppt.apply_sppt(hs_tendencies(grid, bundle),
                               sppt.pattern_to_grid(rl, bundle.trans_config), mu, params)
        s = step(bundle, s, phys)
        rl = sppt.pattern_step(rl, k, params, bundle.trans_config, bundle.dt)
        return (s, rl), None

    (out, _), _ = jax.lax.scan(body, (spec, r), keys)
    return out


def _lead_steps(exp, bundle):
    spd = int(round(86400.0 / bundle.dt))
    return tuple(int(d) * spd for d in exp.lead_days)


def _sample_case_ics(bundle, key, exp):
    """Spin up then sample n_cases states separated by case_stride_days."""
    spec, _stride, spd = nature_run(bundle, key, exp.spinup_days, exp.case_stride_days)
    ics = [spec]
    for _ in range(exp.n_cases - 1):
        spec = _deterministic_advance(bundle, spec, int(exp.case_stride_days * spd))
        ics.append(spec)
    return ics


def mode_b_dataset(exp: ExperimentConfig):
    hi = build_model(ModelConfig(resolution=exp.truth_resolution))
    lo = build_model(ModelConfig(resolution=exp.forecast_resolution))
    key = jax.random.PRNGKey(exp.seed)
    lead_steps_hi = _lead_steps(exp, hi)

    case_ics_hi = _sample_case_ics(hi, key, exp)
    ic_specs, truth = [], {k: [] for k in _FIELDS}
    for ic_hi in case_ics_hi:
        ic_specs.append(spectral_truncate(ic_hi, hi, lo))
        prev, spec = 0, ic_hi
        per_lead = {k: [] for k in _FIELDS}
        for ls in lead_steps_hi:
            spec = _deterministic_advance(hi, spec, ls - prev)
            f = extract_fields(spectral_truncate(spec, hi, lo), lo)
            for k in _FIELDS:
                per_lead[k].append(f[k])
            prev = ls
        for k in _FIELDS:
            truth[k].append(jnp.stack(per_lead[k]))
    return _pack(ic_specs, truth, exp, lo)


def mode_a_dataset(exp: ExperimentConfig, true_params):
    bundle = build_model(ModelConfig(resolution=exp.forecast_resolution))
    key = jax.random.PRNGKey(exp.seed)
    lead_steps = _lead_steps(exp, bundle)

    case_ics = _sample_case_ics(bundle, key, exp)
    ic_specs, truth = [], {k: [] for k in _FIELDS}
    for i, ic in enumerate(case_ics):
        ic_specs.append(ic)
        mkey = jax.random.fold_in(key, 1000 + i)   # one truth draw from the true model
        prev, spec = 0, ic
        per_lead = {k: [] for k in _FIELDS}
        for ls in lead_steps:
            spec = _sppt_advance(bundle, spec, true_params,
                                 jax.random.fold_in(mkey, prev), ls - prev)
            f = extract_fields(spec, bundle)
            for k in _FIELDS:
                per_lead[k].append(f[k])
            prev = ls
        for k in _FIELDS:
            truth[k].append(jnp.stack(per_lead[k]))
    return _pack(ic_specs, truth, exp, bundle)


def _stack_specs(specs):
    return SpectralState(
        vorticity=jnp.stack([s.vorticity for s in specs]),
        divergence=jnp.stack([s.divergence for s in specs]),
        temperature=jnp.stack([s.temperature for s in specs]),
        log_surface_pressure=jnp.stack([s.log_surface_pressure for s in specs]),
        tracers=jnp.stack([s.tracers for s in specs]),
    )


def _pack(ic_specs, truth, exp, bundle):
    return {
        "ic_specs": _stack_specs(ic_specs),
        "truth": {k: jnp.stack(v) for k, v in truth.items()},   # (n_case,n_lead,lat,lon)
        "lead_steps": _lead_steps(exp, bundle),
        "forecast_resolution": exp.forecast_resolution,
        "n_lev": bundle.n_lev,
    }


def save_dataset(path, data):
    with open(path, "wb") as f:
        pickle.dump(jax.device_get(data), f)


def load_dataset(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["A", "B"], required=True)
    ap.add_argument("--forecast", default="T42")
    ap.add_argument("--truth", default="T127")
    ap.add_argument("--n-cases", type=int, default=16)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--spinup-days", type=float, default=200.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    exp = ExperimentConfig(mode=a.mode, forecast_resolution=a.forecast,
                           truth_resolution=(a.truth if a.mode == "B" else a.forecast),
                           n_members=a.members, n_cases=a.n_cases, spinup_days=a.spinup_days)
    if a.mode == "A":
        data = mode_a_dataset(exp, sppt.default_params())
    else:
        data = mode_b_dataset(exp)
    save_dataset(a.out, data)
    print(f"wrote {a.out}: {exp.n_cases} cases, leads {exp.lead_days} d")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_generate_data.py -v`
Expected: PASS (2 passed). May take ~1-2 min (short nature runs at T21/T42).

- [ ] **Step 5: Commit**

```bash
git add examples/stochastic_sppt/generate_data.py examples/stochastic_sppt/tests/test_generate_data.py
git commit -m "feat(sppt): Mode A/B identical-twin data generation"
```

---

### Task 10: Training loop (afCRPS + optax + common random numbers)

**Files:**
- Create: `examples/stochastic_sppt/train.py`
- Test: `examples/stochastic_sppt/tests/test_training.py`

**Interfaces:**
- Consumes: `load_dataset`, `build_model`, `ensemble_rollout`, `metrics.afcrps`, `sppt.SPPTParams`, config dataclasses. `optax`.
- Produces: `loss_fn(params, batch_ics, batch_truth, bundle, lead_steps, keys, alpha) -> scalar`; `member_keys(base_key, case_idx, n_members) -> (M,) keys`; `train(dataset, train_config) -> (SPPTParams, list[float])`; `main(argv=None)` CLI (loads dataset, trains, pickles params + loss history).

Loss aggregates afCRPS over the batch of cases, all leads, and all verification fields with equal weighting (per-field normalization by each field's truth std keeps scales comparable — the plan's stand-in for AIFS per-variable scaling). Member keys derive from `(base_key, case_idx)` and are FIXED per case within a gradient evaluation (common random numbers).

- [ ] **Step 1: Write the failing test (Mode A recovery smoke test)**

```python
# examples/stochastic_sppt/tests/test_training.py
import jax
import jax.numpy as jnp
from examples.stochastic_sppt.config import ExperimentConfig, ModelConfig, TrainConfig
from examples.stochastic_sppt import generate_data, sppt, train


def test_mode_a_training_moves_params_toward_truth():
    exp = ExperimentConfig(mode="A", forecast_resolution="T21", truth_resolution="T21",
                           n_members=4, lead_days=(1,), n_cases=4, spinup_days=1.0,
                           case_stride_days=1.0, seed=0)
    true_p = sppt.SPPTParams(log_sigma=jnp.log(0.6), log_tau=jnp.log(6 * 3600.0),
                             log_len=jnp.log(500e3))
    data = generate_data.mode_a_dataset(exp, true_p)

    tc = TrainConfig(alpha=0.95, lr=1e-1, n_opt_steps=15, batch_cases=4, log_every=5, seed=0)
    # start with a deliberately wrong sigma
    init = sppt.SPPTParams(log_sigma=jnp.log(0.2), log_tau=jnp.log(6 * 3600.0),
                           log_len=jnp.log(500e3))
    params, history = train.train(data, tc, init_params=init)
    # loss decreased and sigma moved toward the truth (0.6)
    assert history[-1] < history[0]
    assert float(jnp.exp(params.log_sigma)) > float(jnp.exp(init.log_sigma))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_training.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'examples.stochastic_sppt.train'`

- [ ] **Step 3: Write `train.py`**

```python
# examples/stochastic_sppt/train.py
"""Train SPPT params by minimizing almost-fair CRPS over identical-twin cases."""
import argparse
import pickle

import jax
import jax.numpy as jnp
import optax

from gfs_dynamical_core.jax.states import SpectralState

from . import sppt
from .config import ModelConfig, TrainConfig
from .generate_data import load_dataset
from .metrics import afcrps
from .model import build_model
from .rollout import ensemble_rollout

_FIELDS = ("u850", "v850", "t500", "vort500", "ps")


def member_keys(base_key, case_idx, n_members):
    return jax.random.split(jax.random.fold_in(base_key, case_idx), n_members)


def _case_ic(ic_specs, i):
    return SpectralState(
        vorticity=ic_specs.vorticity[i], divergence=ic_specs.divergence[i],
        temperature=ic_specs.temperature[i], log_surface_pressure=ic_specs.log_surface_pressure[i],
        tracers=ic_specs.tracers[i],
    )


def loss_fn(params, ic_specs, truth, case_indices, bundle, lead_steps, base_key, alpha, scales):
    total = 0.0
    for i in case_indices:
        spec0 = _case_ic(ic_specs, int(i))
        keys = member_keys(base_key, int(i), truth[_FIELDS[0]].shape[0] and 0 or 0)  # placeholder
        keys = member_keys(base_key, int(i), _n_members_hint[0])
        ens = ensemble_rollout(params, spec0, keys, bundle, lead_steps)
        for k in _FIELDS:
            total = total + afcrps(ens[k], truth[k][int(i)], alpha) / scales[k]
    return total / (len(case_indices) * len(_FIELDS))


_n_members_hint = [8]


def train(dataset, train_config: TrainConfig, init_params=None, n_members=4):
    _n_members_hint[0] = n_members
    bundle = build_model(ModelConfig(resolution=dataset["forecast_resolution"],
                                     n_lev=int(dataset["n_lev"])))
    lead_steps = tuple(int(s) for s in dataset["lead_steps"])
    ic_specs, truth = dataset["ic_specs"], dataset["truth"]
    scales = {k: float(jnp.std(truth[k]) + 1e-12) for k in _FIELDS}
    n_cases = truth[_FIELDS[0]].shape[0]

    params = init_params if init_params is not None else sppt.default_params()
    opt = optax.adam(train_config.lr)
    opt_state = opt.init(params)
    base_key = jax.random.PRNGKey(train_config.seed)

    grad_fn = jax.value_and_grad(loss_fn)
    history = []
    for it in range(train_config.n_opt_steps):
        b = min(train_config.batch_cases, n_cases)
        idx = jax.random.choice(jax.random.fold_in(base_key, it), n_cases, (b,), replace=False)
        # common random numbers: derive from a per-iteration key (fixed within this grad eval)
        eval_key = jax.random.fold_in(base_key, 10_000 + it)
        loss, grads = grad_fn(params, ic_specs, truth, [int(x) for x in idx],
                              bundle, lead_steps, eval_key, train_config.alpha, scales)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        history.append(float(loss))
        if it % train_config.log_every == 0:
            print(f"step {it:4d}  afCRPS {float(loss):.5f}  "
                  f"sigma {float(jnp.exp(params.log_sigma)):.3f} "
                  f"tau {float(jnp.exp(params.log_tau)) / 3600:.2f}h "
                  f"len {float(jnp.exp(params.log_len)) / 1e3:.0f}km")
    return params, history


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--out", default="sppt_params.pkl")
    a = ap.parse_args(argv)
    data = load_dataset(a.dataset)
    tc = TrainConfig(alpha=a.alpha, lr=a.lr, n_opt_steps=a.steps)
    params, history = train(data, tc, n_members=a.members)
    with open(a.out, "wb") as f:
        pickle.dump({"params": jax.device_get(params), "history": history}, f)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
```

Note: clean up the placeholder line in `loss_fn` — it must read exactly `keys = member_keys(base_key, int(i), _n_members_hint[0])` (delete the first placeholder assignment). The `_n_members_hint` module global carries the member count into the (otherwise closed) loss signature; set once in `train`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_training.py -v`
Expected: PASS (1 passed). Takes ~2-4 min (15 optimizer steps at T21, 4 members, 4 cases). If the recovery direction is noisy, raise `n_opt_steps` to 25 in the test — the assertion is monotone-ish improvement, not exact recovery.

- [ ] **Step 5: Commit**

```bash
git add examples/stochastic_sppt/train.py examples/stochastic_sppt/tests/test_training.py
git commit -m "feat(sppt): afCRPS training loop with common random numbers"
```

---

### Task 11: Analysis + plots

**Files:**
- Create: `examples/stochastic_sppt/analyze.py`
- Test: `examples/stochastic_sppt/tests/test_analyze.py`

**Interfaces:**
- Consumes: `load_dataset`, `build_model`, `ensemble_rollout`, `metrics.*`, `sppt.SPPTParams`.
- Produces: `compute_metrics(params, dataset, n_members, seed) -> dict` (per field & lead: `spread_error_ratio`, `afcrps`, `rank_hist`); `write_report(metrics, outdir)` (PNGs + `summary.json`); `main(argv=None)` CLI.

- [ ] **Step 1: Write the failing test**

```python
# examples/stochastic_sppt/tests/test_analyze.py
import json
import os
import jax.numpy as jnp
from examples.stochastic_sppt.config import ExperimentConfig
from examples.stochastic_sppt import generate_data, sppt, analyze


def test_compute_metrics_and_report(tmp_path):
    exp = ExperimentConfig(mode="A", forecast_resolution="T21", truth_resolution="T21",
                           n_members=3, lead_days=(1,), n_cases=2, spinup_days=1.0,
                           case_stride_days=1.0, seed=0)
    data = generate_data.mode_a_dataset(exp, sppt.default_params())
    m = analyze.compute_metrics(sppt.default_params(), data, n_members=3, seed=0)
    assert "t500" in m and "spread_error_ratio" in m["t500"]
    assert len(m["t500"]["spread_error_ratio"]) == 1   # one lead
    analyze.write_report(m, str(tmp_path))
    assert os.path.exists(tmp_path / "summary.json")
    with open(tmp_path / "summary.json") as f:
        assert "t500" in json.load(f)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_analyze.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'examples.stochastic_sppt.analyze'`

- [ ] **Step 3: Write `analyze.py`**

```python
# examples/stochastic_sppt/analyze.py
"""Evaluate a trained SPPT: spread-error ratio, rank histograms, afCRPS vs lead."""
import argparse
import json
import os
import pickle

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gfs_dynamical_core.jax.states import SpectralState

from . import metrics
from .config import ModelConfig
from .generate_data import load_dataset
from .model import build_model
from .rollout import ensemble_rollout

_FIELDS = ("u850", "v850", "t500", "vort500", "ps")


def _case_ic(ic_specs, i):
    return SpectralState(
        vorticity=ic_specs.vorticity[i], divergence=ic_specs.divergence[i],
        temperature=ic_specs.temperature[i], log_surface_pressure=ic_specs.log_surface_pressure[i],
        tracers=ic_specs.tracers[i],
    )


def compute_metrics(params, dataset, n_members=8, seed=0):
    bundle = build_model(ModelConfig(resolution=dataset["forecast_resolution"],
                                     n_lev=int(dataset["n_lev"])))
    lead_steps = tuple(int(s) for s in dataset["lead_steps"])
    ic_specs, truth = dataset["ic_specs"], dataset["truth"]
    n_cases = truth[_FIELDS[0]].shape[0]
    n_leads = len(lead_steps)
    base = jax.random.PRNGKey(seed)

    ens = {k: [] for k in _FIELDS}     # per case -> (M, n_leads, lat, lon)
    for i in range(n_cases):
        keys = jax.random.split(jax.random.fold_in(base, i), n_members)
        out = ensemble_rollout(params, _case_ic(ic_specs, i), keys, bundle, lead_steps)
        for k in _FIELDS:
            ens[k].append(out[k])

    result = {}
    for k in _FIELDS:
        e = jnp.stack(ens[k], axis=0)          # (n_cases, M, n_leads, lat, lon)
        ratios, scores = [], []
        for li in range(n_leads):
            # pool cases into the batch dim for the metric
            e_l = e[:, :, li]                   # (n_cases, M, lat, lon)
            t_l = truth[k][:, li]               # (n_cases, lat, lon)
            ens_flat = e_l.transpose(1, 0, 2, 3)   # (M, n_cases, lat, lon)
            ratios.append(float(metrics.spread_error_ratio(ens_flat, t_l)))
            scores.append(float(metrics.afcrps(ens_flat, t_l, 0.95)))
        result[k] = {"spread_error_ratio": ratios, "afcrps": scores,
                     "lead_days": [int(s * bundle.dt / 86400.0) for s in lead_steps]}
    return result


def write_report(result, outdir):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 5))
    for k in _FIELDS:
        ax.plot(result[k]["lead_days"], result[k]["spread_error_ratio"], "-o", label=k)
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel("lead (days)"); ax.set_ylabel("spread / error"); ax.legend()
    ax.set_title("Spread-error ratio (target = 1)")
    fig.savefig(os.path.join(outdir, "spread_error_ratio.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for k in _FIELDS:
        ax.plot(result[k]["lead_days"], result[k]["afcrps"], "-o", label=k)
    ax.set_xlabel("lead (days)"); ax.set_ylabel("almost-fair CRPS"); ax.legend()
    fig.savefig(os.path.join(outdir, "afcrps_vs_lead.png"), dpi=120)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--outdir", default="sppt_analysis")
    a = ap.parse_args(argv)
    data = load_dataset(a.dataset)
    with open(a.params, "rb") as f:
        params = pickle.load(f)["params"]
    result = compute_metrics(params, data, n_members=a.members)
    write_report(result, a.outdir)
    print(f"wrote report to {a.outdir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_analyze.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add examples/stochastic_sppt/analyze.py examples/stochastic_sppt/tests/test_analyze.py
git commit -m "feat(sppt): analysis + spread-error / afCRPS reporting"
```

---

### Task 12: SLURM templates + README + full suite gate

**Files:**
- Create: `examples/stochastic_sppt/slurm/generate_data.sbatch`
- Create: `examples/stochastic_sppt/slurm/train.sbatch`
- Create: `examples/stochastic_sppt/README.md`
- Test: `examples/stochastic_sppt/tests/test_slurm_and_docs.py`

**Interfaces:**
- Consumes: the CLIs from Tasks 9-11 (`python -m examples.stochastic_sppt.generate_data|train|analyze`).
- Produces: two valid `sbatch` scripts (single GPU) and the tutorial README.

- [ ] **Step 1: Write the failing test**

```python
# examples/stochastic_sppt/tests/test_slurm_and_docs.py
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_sbatch_scripts_are_valid_bash():
    for name in ["generate_data.sbatch", "train.sbatch"]:
        p = ROOT / "slurm" / name
        assert p.exists()
        # bash -n: syntax check only
        subprocess.run(["bash", "-n", str(p)], check=True)
        assert "#SBATCH --gpus" in p.read_text() or "#SBATCH --gres=gpu" in p.read_text()


def test_readme_has_required_sections():
    text = (ROOT / "README.md").read_text().lower()
    for section in ["sppt", "mode a", "mode b", "how to run", "adapting this to skeb",
                    "almost-fair crps"]:
        assert section in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_slurm_and_docs.py -v`
Expected: FAIL (files do not exist)

- [ ] **Step 3: Write the SLURM scripts**

```bash
# examples/stochastic_sppt/slurm/generate_data.sbatch
#!/bin/bash
#SBATCH --job-name=sppt-gen
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=sppt_gen_%j.log

set -euo pipefail
module load cuda || true
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cuda

# T127 truth -> T42 forecast identical-twin dataset (Mode B)
srun python -m examples.stochastic_sppt.generate_data \
    --mode B --forecast T42 --truth T127 \
    --n-cases 64 --members 8 --spinup-days 400 \
    --out data/sppt_modeB_T42_T127.pkl
```

```bash
# examples/stochastic_sppt/slurm/train.sbatch
#!/bin/bash
#SBATCH --job-name=sppt-train
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=sppt_train_%j.log

set -euo pipefail
module load cuda || true
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cuda

srun python -m examples.stochastic_sppt.train \
    --dataset data/sppt_modeB_T42_T127.pkl \
    --members 8 --steps 400 --lr 0.05 --alpha 0.95 \
    --out sppt_params_T42.pkl

srun python -m examples.stochastic_sppt.analyze \
    --dataset data/sppt_modeB_T42_T127.pkl \
    --params sppt_params_T42.pkl --members 16 --outdir sppt_analysis_T42
```

- [ ] **Step 4: Write `README.md` (the tutorial)**

Write `examples/stochastic_sppt/README.md` covering, in order, these sections (real prose, not placeholders): (1) **What SPPT is** — the multiplicative `(1+µr)` scheme and the Palmer 2009 spectral AR(1) pattern generator, with the equations from `sppt.py`; (2) **The identical-twin experiment** — **Mode A** (parameter recovery) and **Mode B** (T42 forecast vs T127 truth model-error calibration), including the honest caveat that Mode B calibrates against resolution error, not real atmospheric error; (3) **Almost-fair CRPS** — why it beats fair CRPS (Lang 2024 degeneracy), with the `alpha` switch; (4) **How to run** — the exact `python -m examples.stochastic_sppt.generate_data|train|analyze` commands and the two `sbatch` scripts; (5) **File-by-file tour** mapping each module to its responsibility; (6) **Adapting this to SKEB** — reuse `sppt.py`'s spectral pattern generator, but inject additively into `spec_tends.d_vorticity_d_t` (via `advance_with_tendencies`'s `spec_tends` argument) instead of multiplying grid tendencies, modulate the amplitude by `sqrt(dissipation)`, and keep the same reality-symmetry guard and afCRPS training loop. Include a code sketch of the SKEB injection point.

- [ ] **Step 5: Run the docs/slurm test AND the full suite**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_slurm_and_docs.py -v`
Expected: PASS (2 passed)

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/ -v`
Expected: all tests PASS.

Run (regression — no substrate breakage): `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest tests/test_jax_component_api.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add examples/stochastic_sppt/slurm examples/stochastic_sppt/README.md examples/stochastic_sppt/tests/test_slurm_and_docs.py
git commit -m "feat(sppt): SLURM templates and tutorial README"
```

---

## Self-Review

**Spec coverage:**
- SPPT scheme (Palmer 2009) → Tasks 4-5. ✓
- almost-fair CRPS (Lang 2024) → Task 7. ✓
- Held-Suarez physics → Task 3. ✓
- Spectral generator → grid multiply wiring → Tasks 4-5, 8. ✓
- 3 trainable scalars → Task 4 (`SPPTParams`). ✓
- Verification fields u850/v850/t500/vort500/ps → Task 6. ✓
- Identical-twin Mode A + Mode B, T42/T127 + T21/T42 dev → Task 9. ✓
- Checkpointed rollout, vmap members, common random numbers → Tasks 8, 10. ✓
- Reality/triangular guards → Task 4 (via `enforce_triangular_truncation`). ✓
- SLURM single-GPU → Task 12. ✓
- Analysis: spread-error, rank hist, afCRPS vs lead, plots → Task 11. ✓
- README tutorial + SKEB adaptation → Task 12. ✓
- Milestone M1 (Mode A recovery) → Task 10 test. Milestone M2 (Mode B spread-error→1) is the production run driven by the Task 12 sbatch scripts (not a unit test — it needs the T127 nature run).

**Extension seams left unbuilt (per spec YAGNI):** SP2 two-scale, per-wavenumber τ, NN amplitude, IC-perturbation seeding, ERA5 truth, multi-GPU sharding — documented in the README, not coded.

**Placeholder scan:** Task 10's `loss_fn` intentionally flags a placeholder line with an explicit correction note in Step 3 — the engineer removes it. No other TBD/TODO content.

**Type consistency:** `SPPTParams(log_sigma, log_tau, log_len)`, `ModelBundle` fields, `extract_fields` keys (`u850,v850,t500,vort500,ps`), dataset dict keys (`ic_specs, truth, lead_steps, forecast_resolution, n_lev`), and `lead_steps` (tuple of int step indices) are used consistently across Tasks 2, 4, 6, 8, 9, 10, 11.

**Known implementation risks flagged inline for the engineer:**
- Task 4: `_VAR_NORM` calibration if analytic F0 disagrees with the s2fft normalization (test-driven).
- Task 8: `_params_holder` closure vs. tracer leaks — fallback is threading `params` through the scan carry.
- Task 10: member-count global `_n_members_hint` — set once in `train`.
