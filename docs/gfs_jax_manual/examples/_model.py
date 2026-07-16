"""Self-contained Held-Suarez model built from the gfs_dynamical_core.jax core.

This is the reference "how to build a model" for the manual: it wires the four
core modules (states, transforms, dynamics, stepper) into a single callable
`step(...)`, using climt only to fetch the vertical coordinate (ak/bk) and Earth
constants at setup time. Everything returned is a pure-JAX pytree, so the model
composes with jit / vmap / grad.

All example scripts import from here so the demonstrations stay short.
"""
import os

# Float64 is mandatory (spectral dynamics are stiff); on Apple Silicon JAX would
# otherwise pick the Metal backend, which cannot do float64. Pin both at import.
os.environ["JAX_ENABLE_X64"] = "True"
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import climt
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from sympl import get_constant, set_constant

jax.config.update("jax_enable_x64", True)

from gfs_dynamical_core.jax.dynamics import DynamicsConfig, compute_pressure_diagnostics
from gfs_dynamical_core.jax.states import GridState
from gfs_dynamical_core.jax.stepper import (
    PhysicsTendencies,
    StepperConfig,
    advance_with_tendencies,
    init_diffusion_operators,
    init_semi_implicit_matrices,
)
from gfs_dynamical_core.jax.transforms import (
    TransformConfig,
    get_gaussian_latitudes,
    grid_to_spectral,
    prebuild_kernels,
    spectral_to_grid,
)

_DAY = 86400.0

# (L, ntrunc, dt). T21 is the light, fast resolution used by every example here.
RESOLUTIONS = {"T21": (32, 21, 1800.0), "T42": (64, 42, 1200.0)}


@struct.dataclass
class ModelBundle:
    """Everything a step needs, as one pytree (array leaves are differentiable;
    scalar/config leaves are static pytree aux)."""
    dyn_config: DynamicsConfig
    trans_config: TransformConfig
    stepper_config: StepperConfig
    latitudes: jnp.ndarray
    gauss_weights: jnp.ndarray
    phis_grads: tuple
    dt: float = struct.field(pytree_node=False)
    n_lev: int = struct.field(pytree_node=False)


def build_model(resolution="T21", n_lev=20) -> ModelBundle:
    """Assemble a Held-Suarez model bundle at the requested resolution."""
    L, ntrunc, dt = RESOLUTIONS[resolution]

    # --- vertical coordinate and Earth constants from climt (setup-time only) ---
    set_constant("reference_air_pressure", value=1e5, units="Pa")
    grid = climt.get_grid(nx=2 * L - 1, ny=L, nz=n_lev)
    ak = jnp.array(np.asarray(grid[
        "atmosphere_hybrid_sigma_pressure_a_coordinate_on_interface_levels"].values,
        dtype=np.float64))
    bk = jnp.array(np.asarray(grid[
        "atmosphere_hybrid_sigma_pressure_b_coordinate_on_interface_levels"].values,
        dtype=np.float64))
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

    dyn_config = DynamicsConfig(
        ak=ak, bk=bk, ck=ak[:-1] * bk[1:] - ak[1:] * bk[:-1], dbk=bk[:-1] - bk[1:],
        rk=c["rd"] / c["cp"], toa_pressure=c["toa_pressure"], radius=c["radius"],
        omega=c["omega"], g=c["g"], rd=c["rd"], rv=c["rv"], cp=c["cp"], cvap=c["cvap"],
    )
    trans_config = TransformConfig(L=L, sampling="gl", ntrunc=ntrunc, radius=c["radius"])
    prebuild_kernels(L, "gl")  # eager kernel build; jit only ever reads the cache

    sc = StepperConfig(dt=dt)
    si = init_semi_implicit_matrices(dyn_config, trans_config, dt,
                                     aa22=sc.aa22, aa33=sc.aa33, bb4=sc.bb4)
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
    return ModelBundle(dyn_config=dyn_config, trans_config=trans_config,
                       stepper_config=stepper_config, latitudes=latitudes,
                       gauss_weights=gauss_weights, phis_grads=phis_grads,
                       dt=dt, n_lev=n_lev)


# --------------------------------------------------------------------------
# Held-Suarez (1994) forcing: the deterministic "physics" P_X the core steps.
# --------------------------------------------------------------------------
def hs_tendencies(grid_state, bundle) -> PhysicsTendencies:
    dyn = bundle.dyn_config
    lat = bundle.latitudes[None, :, None]
    sinphi2, cosphi2 = jnp.sin(lat) ** 2, jnp.cos(lat) ** 2
    cosphi4 = cosphi2 ** 2
    lnps = grid_state.log_surface_pressure.real
    temperature = grid_state.temperature.real
    press = compute_pressure_diagnostics(lnps, dyn)
    p, ps = press.prs, press.ps[None, :, :]
    sigma, kappa = p / ps, dyn.rk

    t_eq = jnp.maximum(200.0, (315.0 - 60.0 * sinphi2
           - 10.0 * jnp.log(p / 1e5) * cosphi2) * (p / 1e5) ** kappa)
    frac = jnp.clip((sigma - 0.7) / (1.0 - 0.7), 0.0, None)
    k_t = 1.0 / (40 * _DAY) + (1.0 / (4 * _DAY) - 1.0 / (40 * _DAY)) * frac * cosphi4
    k_v = (1.0 / _DAY) * frac
    return PhysicsTendencies(
        u=-k_v * grid_state.u, v=-k_v * grid_state.v,
        virtual_temperature=-k_t * (temperature - t_eq),
        log_surface_pressure=jnp.zeros_like(grid_state.log_surface_pressure),
        tracers=jnp.zeros_like(grid_state.tracers))


def step(bundle, spec_state):
    """One deterministic Held-Suarez model step (dynamics + HS forcing)."""
    grid, _ = spectral_to_grid(spec_state, bundle.trans_config)
    phys = hs_tendencies(grid, bundle)
    return advance_with_tendencies(
        spec_state, phys, bundle.phis_grads, bundle.dyn_config, bundle.trans_config,
        bundle.stepper_config, bundle.latitudes, bundle.gauss_weights, None)


def rest_state(bundle, key, t0=250.0, ps0=1.0e5, noise=1.0e-3):
    """Isothermal rest state + tiny temperature noise, in spectral space.

    The (dry) core still advects one tracer. We seed it with a *tiny,
    vertically- and meridionally-varying, strictly positive* field rather than
    zeros: an identically-zero tracer makes the positive-definite vertical-
    advection flux limiter evaluate 0/0 (guarded to ~1e-308), which is fine for
    the value but makes FORWARD-mode derivatives (jacfwd / jvp) divide tangents
    by that tiny number and blow up to NaN. A small non-uniform q keeps the
    limiter well-conditioned; at ~1e-4 kg/kg it is physically inert in HS.
    """
    n_lev = bundle.n_lev
    n_lat, n_lon = bundle.trans_config.L, 2 * bundle.trans_config.L - 1
    temp = t0 + noise * jax.random.normal(key, (n_lev, n_lat, n_lon))
    z3 = jnp.zeros((n_lev, n_lat, n_lon))
    lat = bundle.latitudes[:, None]                          # (n_lat, 1)
    k = jnp.arange(n_lev)[:, None, None]
    q0 = 1e-4 * (1.0 + 0.3 * k / max(n_lev - 1, 1)) * (1.0 + 0.2 * jnp.cos(lat) ** 2)
    q0 = jnp.broadcast_to(q0, (n_lev, n_lat, n_lon))
    grid = GridState(u=z3, v=z3, temperature=temp, vorticity=z3, divergence=z3,
                     log_surface_pressure=jnp.full((n_lat, n_lon), jnp.log(ps0)),
                     tracers=q0[None])
    return grid_to_spectral(grid, bundle.trans_config)


def global_mean(field, gauss_weights):
    """Area-weighted global mean of a (..., n_lat, n_lon) grid field."""
    n_lon = field.shape[-1]
    return jnp.sum(gauss_weights[:, None] * field, axis=(-2, -1)) / n_lon
