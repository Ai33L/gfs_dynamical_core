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

    # spectral_to_grid leaves a negligible numerical-noise imaginary part
    # (~1e-19) on real physical fields from the round-trip SHT; the substrate
    # itself takes `.real` before feeding log_surface_pressure into
    # compute_pressure_diagnostics (see stepper.py's dry-mass-fixer call), so
    # we follow the same precedent here for both log_surface_pressure and
    # temperature.
    lnps = grid_state.log_surface_pressure.real
    temperature = grid_state.temperature.real

    press = compute_pressure_diagnostics(lnps, dyn)
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
    temp_tend = -k_t * (temperature - t_eq)

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
