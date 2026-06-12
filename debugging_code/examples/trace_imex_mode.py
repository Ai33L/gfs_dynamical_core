"""
Trace the IMEX stepper for a single spectral mode (l=1, m=0) through
all three RK stages, printing every intermediate value.

This replicates the IMEX logic from stepper.py manually to see exactly
where the v-error enters.
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

from datetime import timedelta

import climt
import jax.numpy as jnp
import numpy as np
from sympl import set_constant

from gfs_dynamical_core.component_jax import GFSDynamicsJAX
from gfs_dynamical_core.jax.dynamics import get_spectral_tendencies
from gfs_dynamical_core.jax.states import GridState, SpectralState
from gfs_dynamical_core.jax.transforms import grid_to_spectral, spectral_to_grid

L = 64
N_LON = 2 * L - 1
N_LAT = L
N_LEV = 20

set_constant("reference_air_pressure", value=1e5, units="Pa")


def extract_mode(arr, l, m):
    """Extract spectral coefficient at (l, m) from s2fft rectangular layout.
    m index in s2fft: column = L - 1 + m (so m=0 -> column L-1)."""
    col = L - 1 + m
    if arr.ndim == 2:  # (L, 2L-1) -> scalar spectral field
        return complex(arr[l, col])
    elif arr.ndim == 3:  # (n_lev, L, 2L-1)
        return np.array(arr[:, l, col], dtype=complex)
    else:
        raise ValueError(f"Unexpected ndim: {arr.ndim}")


def print_mode(name, val):
    """Print a spectral mode value (scalar or per-level)."""
    if isinstance(val, (complex, np.complexfloating)):
        print(f"    {name}: {val.real:+.10e} {val.imag:+.10e}j")
    elif isinstance(val, np.ndarray):
        # Per-level: show a few representative levels
        for k in [0, 5, 10, 15, 19]:
            if k < len(val):
                print(f"    {name}[k={k:2d}]: {val[k].real:+.10e} {val[k].imag:+.10e}j")


def main():
    grid = climt.get_grid(nx=N_LON, ny=N_LAT, nz=N_LEV)
    timestep = timedelta(minutes=5)
    dt = timestep.total_seconds()
    dcmip = climt.DcmipInitialConditions(add_perturbation=False)

    # Initialize JAX dycore
    dycore_j = GFSDynamicsJAX()
    state_j = climt.get_default_state([dycore_j], grid_state=grid)
    state_j.update(dcmip(state_j))
    _, _ = dycore_j(state_j, timestep=timestep)  # init configs

    sc = dycore_j.stepper_config
    dc = dycore_j.dyn_config
    tc = dycore_j.trans_config
    lats = dycore_j._latitudes
    phis_grads = dycore_j._phis_grads

    # Build initial spectral state
    u0 = jnp.array(state_j["eastward_wind"])
    v0 = jnp.array(state_j["northward_wind"])
    temp0 = jnp.array(state_j["air_temperature"])
    ps0 = jnp.array(state_j["surface_air_pressure"])
    q0 = jnp.array(state_j["specific_humidity"])
    grid_orig = GridState(
        u=u0, v=v0, temperature=temp0,
        vorticity=jnp.zeros_like(u0), divergence=jnp.zeros_like(u0),
        log_surface_pressure=jnp.log(ps0), tracers=jnp.stack([q0], axis=0),
    )
    spec0 = grid_to_spectral(grid_orig, tc)

    # Mode to trace
    l_trace, m_trace = 1, 0

    print(f"Tracing mode l={l_trace}, m={m_trace}")
    print(f"dt = {dt} s")
    print(f"IMEX coefficients: a21={sc.a21}, aa21={sc.aa21}, aa22={sc.aa22}")
    print(f"                   a31={sc.a31}, a32={sc.a32}, aa31={sc.aa31}, aa32={sc.aa32}, aa33={sc.aa33}")

    # lap for this mode
    lap_l = -l_trace * (l_trace + 1.0)
    print(f"lap(l={l_trace}) = {lap_l}")

    # ── Initial state ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("INITIAL STATE")
    print(f"{'='*60}")
    vort0_mode = extract_mode(np.array(spec0.vorticity), l_trace, m_trace)
    div0_mode = extract_mode(np.array(spec0.divergence), l_trace, m_trace)
    temp0_mode = extract_mode(np.array(spec0.temperature), l_trace, m_trace)
    lnps0_mode = extract_mode(np.array(spec0.log_surface_pressure), l_trace, m_trace)
    print_mode("vort0", vort0_mode)
    print_mode("div0", div0_mode)
    print_mode("temp0", temp0_mode)
    print_mode("lnps0", lnps0_mode)

    # ── Compute full tendencies from dynamics ────────────────────────────
    tends0 = get_spectral_tendencies(spec0, phis_grads, dc, tc, lats)

    print(f"\n{'='*60}")
    print("FULL TENDENCIES (from get_spectral_tendencies)")
    print(f"{'='*60}")
    dvort0 = extract_mode(np.array(tends0.d_vorticity_d_t), l_trace, m_trace)
    ddiv0 = extract_mode(np.array(tends0.d_divergence_d_t), l_trace, m_trace)
    dtemp0 = extract_mode(np.array(tends0.d_temperature_d_t), l_trace, m_trace)
    dlnps0 = extract_mode(np.array(tends0.d_log_surface_pressure_d_t), l_trace, m_trace)
    print_mode("dvort/dt", dvort0)
    print_mode("ddiv/dt", ddiv0)
    print_mode("dtemp/dt", dtemp0)
    print_mode("dlnps/dt", dlnps0)

    # ── Compute linear tendencies ────────────────────────────────────────
    amhyb = np.array(sc.amhyb)
    bmhyb = np.array(sc.bmhyb)
    tor_hyb = np.array(sc.tor_hyb)
    svhyb = np.array(sc.svhyb)

    # Linear div tendency: -lap * (amhyb @ temp + tor_hyb * lnps)
    temp_term = amhyb @ temp0_mode  # (n_lev,) complex
    lnps_term = tor_hyb * lnps0_mode  # (n_lev,) complex
    ddivdtlin0 = -lap_l * (temp_term + lnps_term)

    # Linear temp tendency: -bmhyb @ div
    dtvdtlin0 = -bmhyb @ div0_mode

    # Linear lnps tendency: -svhyb . div
    dlnpsdtlin0 = -np.dot(svhyb, div0_mode)

    print(f"\n{'='*60}")
    print("LINEAR TENDENCIES (from semi-implicit matrices)")
    print(f"{'='*60}")
    print_mode("ddivdtlin0", ddivdtlin0)
    print_mode("dtvdtlin0", dtvdtlin0)
    print_mode("dlnpsdtlin0", dlnpsdtlin0)

    # ── Nonlinear tendencies ─────────────────────────────────────────────
    ddiv_nl0 = ddiv0 - ddivdtlin0
    dtemp_nl0 = dtemp0 - dtvdtlin0
    dlnps_nl0 = dlnps0 - dlnpsdtlin0

    print(f"\n{'='*60}")
    print("NONLINEAR TENDENCIES (full - linear)")
    print(f"{'='*60}")
    print_mode("ddiv_nl0", ddiv_nl0)
    print_mode("dtemp_nl0", dtemp_nl0)
    print_mode("dlnps_nl0", dlnps_nl0)

    # Check: for balanced state, nl ≈ -linear
    print(f"\n  Check: ddiv_nl0 + ddivdtlin0 (= full tendency, should be ~0):")
    print_mode("  ddiv_full", ddiv_nl0 + ddivdtlin0)

    # ── IMEX Stage 1: explicit update ────────────────────────────────────
    print(f"\n{'='*60}")
    print("STAGE 1: Explicit update")
    print(f"{'='*60}")

    div_expl1 = div0_mode + dt * (sc.a21 * ddiv_nl0 + sc.aa21 * ddivdtlin0)
    temp_expl1 = temp0_mode + dt * (sc.a21 * dtemp_nl0 + sc.aa21 * dtvdtlin0)
    lnps_expl1 = lnps0_mode + dt * (sc.a21 * dlnps_nl0 + sc.aa21 * dlnpsdtlin0)

    print_mode("div_expl1", div_expl1)
    print_mode("temp_expl1", temp_expl1)
    print_mode("lnps_expl1", lnps_expl1)

    # For balanced state where full ≈ 0: nl ≈ -lin
    # div_expl1 ≈ div0 + dt * (-a21 + aa21) * lin = div0 + dt * (0.635 - 1.0) * lin
    print(f"\n  Expected contribution from linear tendency imbalance:")
    print(f"  dt * (aa21 - a21) = {dt * (sc.aa21 - sc.a21):.6f}")
    print_mode("  (aa21-a21)*dt*lin", dt * (sc.aa21 - sc.a21) * ddivdtlin0)

    # ── IMEX Stage 1: implicit solve ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("STAGE 1: Implicit solve")
    print(f"{'='*60}")

    # rhs = div_expl - aa22 * dt * lap * (amhyb @ temp_expl + tor_hyb * lnps_expl)
    rhs_temp = amhyb @ temp_expl1 + tor_hyb * lnps_expl1
    rhs = div_expl1 - sc.aa22 * dt * lap_l * rhs_temp
    print_mode("rhs", rhs)

    # div1 = d_hyb_m @ rhs
    d_mat = np.array(sc.d_hyb_m)[0, l_trace]  # stage 0, degree l_trace
    div1 = d_mat @ rhs
    print_mode("div1 (after solve)", div1)

    # Back-substitution
    temp1 = temp_expl1 - sc.aa22 * dt * (bmhyb @ div1)
    lnps1 = lnps_expl1 - sc.aa22 * dt * np.dot(svhyb, div1)
    print_mode("temp1 (after backsub)", temp1)
    print_mode("lnps1 (after backsub)", lnps1)

    # ── What v would this produce? ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("STAGE 1: Implied v from divergence")
    print(f"{'='*60}")

    import s2fft

    # Build full spectral state for stage 1
    vort1_mode = vort0_mode + sc.a21 * dt * dvort0  # vorticity is explicit
    print_mode("vort1", vort1_mode)
    print_mode("div1", div1)

    # Reconstruct the full spectral arrays for inverse transform
    vort1_full = np.array(spec0.vorticity)
    div1_full = np.array(spec0.divergence)
    temp1_full = np.array(spec0.temperature)
    lnps1_full = np.array(spec0.log_surface_pressure)

    # Update only the traced mode
    col = L - 1 + m_trace
    vort1_full[:, l_trace, col] = vort1_mode
    div1_full[:, l_trace, col] = div1

    # Do the full inverse to get v
    state1_spec = SpectralState(
        vorticity=jnp.array(vort1_full),
        divergence=jnp.array(div1_full),
        temperature=jnp.array(temp1_full),
        log_surface_pressure=jnp.array(lnps1_full),
        tracers=spec0.tracers,
    )
    grid1, _ = spectral_to_grid(state1_spec, tc)
    v1 = np.array(grid1.v)
    if np.iscomplexobj(v1):
        v1 = v1.real

    print(f"  v from stage1 spectral state:")
    print(f"    max|v| = {np.abs(v1).max():.6e}")
    print(f"    v zonal mean max = {np.abs(v1.mean(axis=-1)).max():.6e}")

    # For reference: what does the full IMEX stepper give?
    print(f"\n{'='*60}")
    print("COMPARISON: Full stepper vs Fortran")
    print(f"{'='*60}")

    from gfs_dynamical_core import GFSDynamicalCore
    dycore_f = GFSDynamicalCore()
    state_f = climt.get_default_state([dycore_f], grid_state=grid)
    state_f.update(dcmip(state_f))
    _, out_f = dycore_f(state_f, timestep=timestep)

    # Reset JAX and run
    state_j2 = climt.get_default_state([dycore_j], grid_state=grid)
    state_j2.update(dcmip(state_j2))
    dycore_j._spec_state = None  # force re-init
    _, out_j = dycore_j(state_j2, timestep=timestep)

    v_f = np.asarray(out_f["northward_wind"])
    v_j = np.asarray(out_j["northward_wind"])
    if np.iscomplexobj(v_j): v_j = v_j.real

    print(f"  After 1 full step:")
    print(f"    v_F max: {np.abs(v_f).max():.6e}")
    print(f"    v_J max: {np.abs(v_j).max():.6e}")
    print(f"    diff:    {np.abs(v_f - v_j).max():.6e}")
    print(f"    rel:     {np.abs(v_f - v_j).max() / np.abs(v_f).max():.6e}")

    # Spectral decomposition of Fortran v to see its l=1,m=0 content
    for k in [0, 10, 19]:
        v_f_spec = np.array(s2fft.forward_jax(jnp.array(v_f[k]), L, sampling="gl"))
        v_j_spec = np.array(s2fft.forward_jax(jnp.array(v_j[k]), L, sampling="gl"))
        vf_10 = v_f_spec[l_trace, col]
        vj_10 = v_j_spec[l_trace, col]
        print(f"\n    v spectral (l={l_trace},m={m_trace}) at lev {k}:")
        print(f"      Fortran: {vf_10.real:+.10e} {vf_10.imag:+.10e}j")
        print(f"      JAX:     {vj_10.real:+.10e} {vj_10.imag:+.10e}j")
        ratio = vj_10.real / vf_10.real if abs(vf_10.real) > 1e-20 else float('nan')
        print(f"      ratio (real): {ratio:.10f}")


if __name__ == "__main__":
    main()
