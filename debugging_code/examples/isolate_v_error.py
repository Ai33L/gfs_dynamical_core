"""
Term isolation: disable individual dynamics terms in JAX and compare
v-wind against Fortran to identify which term causes the ~2% error.
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

from datetime import timedelta
from functools import partial

import climt
import jax
import jax.numpy as jnp
import numpy as np
from sympl import set_constant

from gfs_dynamical_core import GFSDynamicalCore
from gfs_dynamical_core.component_jax import GFSDynamicsJAX
from gfs_dynamical_core.jax.dynamics import (
    DynamicsConfig,
    compute_energy_conversion,
    compute_pressure_diagnostics,
    compute_pressure_gradient_force,
    compute_vertical_advection,
    compute_vertical_velocities,
    get_spectral_tendencies,
)
from gfs_dynamical_core.jax.states import GridState, GridGradients, SpectralState, SpectralTendencies
from gfs_dynamical_core.jax.transforms import (
    TransformConfig,
    get_gaussian_latitudes,
    grid_to_spectral,
    grid_to_spectral_tendencies,
    spectral_to_grid,
)

L = 64
N_LON = 2 * L - 1
N_LAT = L
N_LEV = 20

set_constant("reference_air_pressure", value=1e5, units="Pa")


def assemble_grid_tendencies_selective(
    grid_state, grid_grads, vvels, press_diag, pgf, energy_conv,
    config, latitudes,
    use_coriolis=True, use_pgf=True, use_vadv=True, use_energy_conv=True,
):
    """assemble_grid_tendencies with toggles for each term."""
    u, v = grid_state.u, grid_state.v
    vort = grid_state.vorticity
    pgf_x, pgf_y = pgf

    if use_vadv:
        vadv_u = compute_vertical_advection(u, vvels.etadot, press_diag.dp)
        vadv_v = compute_vertical_advection(v, vvels.etadot, press_diag.dp)
        vadv_t = compute_vertical_advection(
            grid_state.temperature, vvels.etadot, press_diag.dp
        )
    else:
        vadv_u = jnp.zeros_like(u)
        vadv_v = jnp.zeros_like(v)
        vadv_t = jnp.zeros_like(grid_state.temperature)

    f = 2.0 * config.omega * jnp.sin(latitudes)
    abs_vort = vort + f[None, :, None] if use_coriolis else vort

    eff_pgf_x = pgf_x if use_pgf else jnp.zeros_like(pgf_x)
    eff_pgf_y = pgf_y if use_pgf else jnp.zeros_like(pgf_y)
    eff_ec = energy_conv if use_energy_conv else jnp.zeros_like(energy_conv)

    u_flux = u * abs_vort + (vadv_v - eff_pgf_y)
    v_flux = v * abs_vort - (vadv_u - eff_pgf_x)

    temp_tend = (
        -u * grid_grads.d_t_d_lambda - v * grid_grads.d_t_d_phi
        - vadv_t + eff_ec
    )

    from gfs_dynamical_core.jax.dynamics import compute_vertical_advection_tracers, GridTendencies

    def compute_tracer_tend(q, dq_dlambda, dq_dphi):
        vadv_q = compute_vertical_advection_tracers(q, vvels.etadot, press_diag.dp)
        return -u * dq_dlambda - v * dq_dphi - vadv_q

    tracer_tends = jax.vmap(compute_tracer_tend)(
        grid_state.tracers,
        grid_grads.d_tracers_d_lambda,
        grid_grads.d_tracers_d_phi,
    )
    ke = 0.5 * (u**2 + v**2)
    return GridTendencies(
        u_flux=u_flux,
        v_flux=v_flux,
        temp_tend=temp_tend,
        log_ps_tend=vvels.d_log_ps_d_t,
        tracer_tends=tracer_tends,
        kinetic_energy=ke,
    )


def run_jax_one_step_selective(dycore_j, state_j, timestep,
                                use_coriolis=True, use_pgf=True,
                                use_vadv=True, use_energy_conv=True):
    """Run one JAX step with selective term disabling."""
    import s2fft
    from gfs_dynamical_core.jax.stepper import advance

    u = jnp.array(state_j["eastward_wind"])
    v = jnp.array(state_j["northward_wind"])
    temp = jnp.array(state_j["air_temperature"])
    ps = jnp.array(state_j["surface_air_pressure"])
    q = jnp.array(state_j["specific_humidity"])

    dyn_config = dycore_j.dyn_config
    trans_config = dycore_j.trans_config
    stepper_config = dycore_j.stepper_config
    latitudes = dycore_j._latitudes
    phis_grads = dycore_j._phis_grads

    # Build spectral state
    if dycore_j._spec_state is not None:
        spec_state = dycore_j._spec_state
    else:
        grid_orig = GridState(
            u=u, v=v, temperature=temp,
            vorticity=jnp.zeros_like(u),
            divergence=jnp.zeros_like(u),
            log_surface_pressure=jnp.log(ps),
            tracers=jnp.stack([q], axis=0),
        )
        spec_state = grid_to_spectral(grid_orig, trans_config)

    # Compute tendencies with selective terms
    grid_state, grid_grads = spectral_to_grid(spec_state, trans_config)
    press_diag = compute_pressure_diagnostics(
        grid_state.log_surface_pressure, dyn_config
    )
    vvels = compute_vertical_velocities(grid_state, grid_grads, press_diag, dyn_config)
    pgf = compute_pressure_gradient_force(
        grid_state.temperature, grid_grads, press_diag, dyn_config, phis_grads
    )
    energy_conv = compute_energy_conversion(
        vvels.omega, grid_state.temperature, grid_state.tracers[0], dyn_config
    )

    grid_tends = assemble_grid_tendencies_selective(
        grid_state, grid_grads, vvels, press_diag, pgf, energy_conv,
        dyn_config, latitudes,
        use_coriolis=use_coriolis, use_pgf=use_pgf,
        use_vadv=use_vadv, use_energy_conv=use_energy_conv,
    )

    # Print individual term magnitudes for v_flux
    print(f"    Individual term magnitudes (v_flux components):")
    vort = grid_state.vorticity
    f = 2.0 * dyn_config.omega * jnp.sin(latitudes)
    abs_vort = vort + f[None, :, None]
    vadv_u = compute_vertical_advection(grid_state.u, vvels.etadot, press_diag.dp)
    pgf_x, pgf_y = pgf

    term_coriolis = grid_state.v * abs_vort
    term_vadv = -vadv_u
    term_pgf = pgf_x
    print(f"      v*abs_vort max: {float(jnp.abs(term_coriolis).max()):.6e}")
    print(f"      -vadv_u    max: {float(jnp.abs(term_vadv).max()):.6e}")
    print(f"      pgf_x      max: {float(jnp.abs(term_pgf).max()):.6e}")
    print(f"      v_flux     max: {float(jnp.abs(grid_tends.v_flux).max()):.6e}")

    print(f"    Individual term magnitudes (u_flux components):")
    term_u_cor = grid_state.u * abs_vort
    vadv_v = compute_vertical_advection(grid_state.v, vvels.etadot, press_diag.dp)
    print(f"      u*abs_vort max: {float(jnp.abs(term_u_cor).max()):.6e}")
    print(f"      vadv_v     max: {float(jnp.abs(vadv_v).max()):.6e}")
    print(f"      -pgf_y     max: {float(jnp.abs(pgf_y).max()):.6e}")
    print(f"      u_flux     max: {float(jnp.abs(grid_tends.u_flux).max()):.6e}")

    return grid_tends


def main():
    grid = climt.get_grid(nx=N_LON, ny=N_LAT, nz=N_LEV)
    timestep = timedelta(minutes=5)
    dcmip = climt.DcmipInitialConditions(add_perturbation=False)

    # Fortran reference
    print("=== Fortran reference (1 step) ===")
    dycore_f = GFSDynamicalCore()
    state_f = climt.get_default_state([dycore_f], grid_state=grid)
    state_f.update(dcmip(state_f))
    _, out_f = dycore_f(state_f, timestep=timestep)

    v_f = np.asarray(out_f["northward_wind"])
    u_f = np.asarray(out_f["eastward_wind"])
    print(f"  Fortran v max: {np.abs(v_f).max():.6e}")
    print(f"  Fortran u max: {np.abs(u_f).max():.6e}")

    # JAX baseline (all terms)
    print("\n=== JAX baseline (all terms, 1 step) ===")
    dycore_j = GFSDynamicsJAX()
    state_j = climt.get_default_state([dycore_j], grid_state=grid)
    state_j.update(dcmip(state_j))

    # First, do a normal step to initialize configs
    _, out_j = dycore_j(state_j, timestep=timestep)

    v_j = np.asarray(out_j["northward_wind"])
    if np.iscomplexobj(v_j):
        v_j = v_j.real
    u_j = np.asarray(out_j["eastward_wind"])
    if np.iscomplexobj(u_j):
        u_j = u_j.real

    v_diff = np.abs(v_f - v_j)
    u_diff = np.abs(u_f - u_j)
    print(f"  JAX v max:     {np.abs(v_j).max():.6e}")
    print(f"  v diff max:    {v_diff.max():.6e}")
    print(f"  v rel err:     {v_diff.max() / np.abs(v_f).max():.6e}")
    print(f"  u diff max:    {u_diff.max():.6e}")
    print(f"  u rel err:     {u_diff.max() / np.abs(u_f).max():.6e}")

    # Now compute term magnitudes from the initial state
    print("\n=== Term magnitudes from initial state ===")
    grid_tends = run_jax_one_step_selective(dycore_j, state_j, timestep)

    # Term isolation tests
    configs = [
        ("no_coriolis", dict(use_coriolis=False)),
        ("no_pgf",      dict(use_pgf=False)),
        ("no_vadv",     dict(use_vadv=False)),
        ("no_energy",   dict(use_energy_conv=False)),
    ]

    print(f"\n{'='*80}")
    print("TERM ISOLATION: disable one term at a time and compare v error")
    print(f"{'='*80}")

    for name, kwargs in configs:
        # Reset JAX dycore
        dycore_j2 = GFSDynamicsJAX()
        state_j2 = climt.get_default_state([dycore_j2], grid_state=grid)
        state_j2.update(dcmip(state_j2))
        # Do a normal step first to initialize configs
        _, _ = dycore_j2(state_j2, timestep=timestep)
        # Reset state to initial
        state_j2 = climt.get_default_state([dycore_j2], grid_state=grid)
        state_j2.update(dcmip(state_j2))

        # Now compute the modified tendencies from initial state
        print(f"\n--- {name} ---")
        grid_tends_mod = run_jax_one_step_selective(
            dycore_j2, state_j2, timestep, **kwargs
        )

        # Compare the modified grid tendencies with the baseline
        # Focus on v_flux and u_flux differences
        print(f"    v_flux change from baseline: {float(jnp.abs(grid_tends_mod.v_flux - grid_tends.v_flux).max()):.6e}")
        print(f"    u_flux change from baseline: {float(jnp.abs(grid_tends_mod.u_flux - grid_tends.u_flux).max()):.6e}")


if __name__ == "__main__":
    main()
