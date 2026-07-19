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
    # spectral_to_grid leaves a negligible numerical-noise imaginary part
    # (~1e-19) on real physical fields from the round-trip SHT for the scalar
    # transforms (temperature, vorticity, log_surface_pressure); u/v are
    # already real (derived via .real/.imag of the spin-1 transform). Follow
    # the same `.real` precedent used in stepper.py / held_suarez.py before
    # feeding log_surface_pressure into compute_pressure_diagnostics and
    # before interpolating temperature/vorticity.
    lnps = grid.log_surface_pressure.real
    temperature = grid.temperature.real
    vorticity = grid.vorticity.real
    press = compute_pressure_diagnostics(lnps, bundle.dyn_config)
    prs = press.prs
    return {
        "u850": vertical_interp(grid.u, prs, 85000.0),
        "v850": vertical_interp(grid.v, prs, 85000.0),
        "t500": vertical_interp(temperature, prs, 50000.0),
        "vort500": vertical_interp(vorticity, prs, 50000.0),
        "ps": press.ps,
    }
