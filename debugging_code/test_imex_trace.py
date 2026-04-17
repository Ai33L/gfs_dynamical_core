"""
Trace IMEX stage 1 intermediates to find where divergence differs.

The 2.2% div error between Fortran and JAX is confirmed in spectral space.
This test decomposes the IMEX stage 1 to identify which intermediate
quantity first shows the discrepancy.

Key insight: for the balanced JW06 state (no perturbation),
total tendency ≈ 0, so NL ≈ -L. The IMEX gives:
  div_expl = dt * (a21 * NL + aa21 * L) = dt * L * (aa21 - a21)
So any difference in L (linear tendency) or NL directly maps to div.
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np
import jax
import jax.numpy as jnp
import s2fft
import climt
from sympl import set_constant
from datetime import timedelta

from gfs_dynamical_core import GFSDynamicalCore
from gfs_dynamical_core.component_jax import GFSDynamicsJAX
from gfs_dynamical_core.jax.dynamics import get_spectral_tendencies, full_dynamics_step, compute_pressure_diagnostics
from gfs_dynamical_core.jax.transforms import (
    TransformConfig, grid_to_spectral, spectral_to_grid,
    grid_to_spectral_tendencies, get_gaussian_latitudes,
)
from gfs_dynamical_core.jax.states import GridState, SpectralState
from gfs_dynamical_core.jax.stepper import StepperConfig, init_semi_implicit_matrices, init_diffusion_operators

# ── Setup ──
L = 64
N_LON = 2 * L - 1
N_LAT = L
N_LEV = 20
set_constant("reference_air_pressure", value=1e5, units="Pa")
grid = climt.get_grid(nx=N_LON, ny=N_LAT, nz=N_LEV)
dt_td = timedelta(minutes=5)
dt = dt_td.total_seconds()

# JW06 ICs
dcmip = climt.DcmipInitialConditions(add_perturbation=False)

# Initialize JAX dycore (to get configs)
dycore_j = GFSDynamicsJAX()
state_j = climt.get_default_state([dycore_j], grid_state=grid)
out = dcmip(state_j)
state_j.update(out)

# Step once to initialize configs
diag_j, out_j = dycore_j(state_j, timestep=dt_td)

# Now get the initial spectral state (before the step)
# We need to redo grid_to_spectral from the ICs
state_j2 = climt.get_default_state([dycore_j], grid_state=grid)
out = dcmip(state_j2)
state_j2.update(out)

u = jnp.array(state_j2["eastward_wind"])
v = jnp.array(state_j2["northward_wind"])
temp = jnp.array(state_j2["air_temperature"])
ps = jnp.array(state_j2["surface_air_pressure"])
q = jnp.array(state_j2["specific_humidity"])
phis = jnp.array(state_j2["surface_geopotential"])

trans_config = dycore_j.trans_config
dyn_config = dycore_j.dyn_config
stepper_config = dycore_j.stepper_config

grid_orig = GridState(
    u=u, v=v, temperature=temp,
    vorticity=jnp.zeros_like(u),
    divergence=jnp.zeros_like(u),
    log_surface_pressure=jnp.log(ps),
    tracers=jnp.stack([q], axis=0),
)
spec_orig = grid_to_spectral(grid_orig, trans_config)
latitudes = get_gaussian_latitudes(L)
phis_grads = dycore_j._phis_grads

# ── Compute total tendencies ──
print("=== Total spectral tendencies from initial state ===")
tends = get_spectral_tendencies(spec_orig, phis_grads, dyn_config, trans_config, latitudes)

div_tend_total = np.array(tends.d_divergence_d_t)
temp_tend_total = np.array(tends.d_temperature_d_t)
lnps_tend_total = np.array(tends.d_log_surface_pressure_d_t)
vort_tend_total = np.array(tends.d_vorticity_d_t)

print(f"  div  tendency energy: {np.sum(np.abs(div_tend_total)**2):.6e}")
print(f"  temp tendency energy: {np.sum(np.abs(temp_tend_total)**2):.6e}")
print(f"  lnps tendency energy: {np.sum(np.abs(lnps_tend_total)**2):.6e}")
print(f"  vort tendency energy: {np.sum(np.abs(vort_tend_total)**2):.6e}")

# ── Compute linear tendencies ──
print("\n=== Linear tendencies from initial state ===")
l_arr = jnp.arange(L)
lap = -l_arr * (l_arr + 1.0)

# Linear div tendency: ddivdtlin = -lap * (amhyb @ T + tor_hyb * lnps)
temp_term = jnp.einsum("ij,j...->i...", stepper_config.amhyb, spec_orig.temperature)
lnps_term = stepper_config.tor_hyb[:, None, None] * spec_orig.log_surface_pressure[None, :, :]
ddivdtlin = -lap[None, :, None] * (temp_term + lnps_term)

# Linear temp tendency: dtvdtlin = -bmhyb @ div
dtvdtlin = -jnp.einsum("ij,j...->i...", stepper_config.bmhyb, spec_orig.divergence)

# Linear lnps tendency: dlnpsdtlin = -svhyb . div
dlnpsdtlin = -jnp.einsum("i,i...->...", stepper_config.svhyb, spec_orig.divergence)

ddivdtlin_np = np.array(ddivdtlin)
dtvdtlin_np = np.array(dtvdtlin)
dlnpsdtlin_np = np.array(dlnpsdtlin)

print(f"  linear div  tendency energy: {np.sum(np.abs(ddivdtlin_np)**2):.6e}")
print(f"  linear temp tendency energy: {np.sum(np.abs(dtvdtlin_np)**2):.6e}")
print(f"  linear lnps tendency energy: {np.sum(np.abs(dlnpsdtlin_np)**2):.6e}")

# div0 = 0, so linear temp and lnps tendencies should be zero
print(f"  (div0=0, so linear temp/lnps should be ~0)")

# ── Nonlinear tendencies = total - linear ──
print("\n=== Nonlinear tendencies (total - linear) ===")
ddivdt_nl = div_tend_total - ddivdtlin_np
dtvdt_nl = temp_tend_total - dtvdtlin_np
dlnpsdt_nl = lnps_tend_total - dlnpsdtlin_np

print(f"  NL div  tendency energy: {np.sum(np.abs(ddivdt_nl)**2):.6e}")
print(f"  NL temp tendency energy: {np.sum(np.abs(dtvdt_nl)**2):.6e}")
print(f"  NL lnps tendency energy: {np.sum(np.abs(dlnpsdt_nl)**2):.6e}")

# ── IMEX Stage 1 step-by-step ──
print("\n=== IMEX Stage 1 (manual computation) ===")
a21 = stepper_config.a21    # 1.0
aa21 = stepper_config.aa21  # 0.635
aa22 = stepper_config.aa22  # 0.365

print(f"  a21={a21}, aa21={aa21}, aa22={aa22}")

# Explicit predictor for div, temp, lnps
div_expl = np.array(spec_orig.divergence) + dt * (a21 * ddivdt_nl + aa21 * ddivdtlin_np)
temp_expl = np.array(spec_orig.temperature) + dt * (a21 * dtvdt_nl + aa21 * dtvdtlin_np)
lnps_expl = np.array(spec_orig.log_surface_pressure) + dt * (a21 * dlnpsdt_nl + aa21 * dlnpsdtlin_np)

print(f"\n  div_expl energy:  {np.sum(np.abs(div_expl)**2):.6e}")
print(f"  Since div0=0 and dtvdtlin=0, dlnpsdtlin=0:")
print(f"    div_expl = dt*(a21*NL_div + aa21*L_div)")
print(f"            = dt*(a21*(total_div - L_div) + aa21*L_div)")
print(f"            = dt*(a21*total_div + (aa21-a21)*L_div)")
print(f"    where total_div ≈ 0, so:")
print(f"    div_expl ≈ dt*(aa21-a21)*L_div = {dt*(aa21-a21):.1f} * L_div")

# Show the m=0 div_expl profile
m0 = L - 1  # m=0 column index in s2fft
print(f"\n  div_expl m=0 profile at level 10:")
for l in range(10):
    val = div_expl[10, l, m0]
    lin = ddivdtlin_np[10, l, m0]
    tot = div_tend_total[10, l, m0]
    nl = ddivdt_nl[10, l, m0]
    print(f"    l={l:2d}: div_expl={val.real:+.6e}, L_div={lin.real:+.6e}, "
          f"total_div={tot.real:+.6e}, NL_div={nl.real:+.6e}")

# ── Now compare with Fortran ──
# Run Fortran dycore from the same ICs
print("\n=== Running Fortran dycore for comparison ===")
dycore_f = GFSDynamicalCore()
state_f = climt.get_default_state([dycore_f], grid_state=grid)
out = dcmip(state_f)
state_f.update(out)
diag_f, out_f = dycore_f(state_f, timestep=dt_td)
state_f.update(out_f)

# Forward-transform Fortran output to spectral
u_f = np.asarray(state_f["eastward_wind"])
v_f = np.asarray(state_f["northward_wind"])
T_f = np.asarray(state_f["air_temperature"])
ps_f = np.asarray(state_f["surface_air_pressure"])
q_f = np.asarray(state_f["specific_humidity"])

gs_f = GridState(
    u=jnp.array(u_f), v=jnp.array(v_f), temperature=jnp.array(T_f),
    vorticity=jnp.zeros_like(jnp.array(u_f)),
    divergence=jnp.zeros_like(jnp.array(u_f)),
    log_surface_pressure=jnp.log(jnp.array(ps_f)),
    tracers=jnp.stack([jnp.array(q_f)], axis=0),
)
spec_f = grid_to_spectral(gs_f, trans_config)

# The spectral div after one full step (3 IMEX stages + diffusion)
div_f_final = np.array(spec_f.divergence)

# JAX's final spectral div (cached)
state_j3 = climt.get_default_state([dycore_j], grid_state=grid)
out = dcmip(state_j3)
state_j3.update(out)
# Reset the cached state to force fresh computation
dycore_j._spec_state = None
diag_j3, out_j3 = dycore_j(state_j3, timestep=dt_td)
div_j_final = np.array(dycore_j._spec_state.divergence)

print(f"\n  Final div (Fortran) m=0 energy: {np.sum(np.abs(div_f_final[:,:,m0])**2):.6e}")
print(f"  Final div (JAX)    m=0 energy: {np.sum(np.abs(div_j_final[:,:,m0])**2):.6e}")

# Per-l comparison of m=0 divergence
print(f"\n  Final spectral divergence m=0, level 10:")
print(f"  {'l':>4s}  {'div_F':>14s}  {'div_J':>14s}  {'ratio':>10s}  {'diff':>14s}")
for l in range(15):
    df = div_f_final[10, l, m0].real
    dj = div_j_final[10, l, m0].real
    ratio = df/dj if abs(dj) > 1e-30 else float('nan')
    diff = df - dj
    print(f"  {l:4d}  {df:+14.6e}  {dj:+14.6e}  {ratio:10.6f}  {diff:+14.6e}")

# ── Check: does the linear tendency alone explain the difference? ──
print("\n=== Does the linear tendency explain the Fortran divergence? ===")
# If Fortran computes the SAME linear tendency, then:
# div_F_stage1 ≈ dt*(aa21-a21)*L_div  (same as JAX)
# After 3 stages, the accumulation differs only if total tendencies differ.
#
# For a balanced state where total ≈ 0:
# The linear tendency L = -lap * (amhyb @ T + tor_hyb * lnps)
# is computed from the SPECTRAL T and lnps. If these are the same (they should be),
# L should be the same. So the div should be the same.
#
# But it's NOT. So either:
# (1) The total tendency differs (dynamics equations differ)
# (2) T or lnps spectral representation differs
# (3) Something else in the stepper logic

# Check option 2: compare initial spectral T and lnps
print("\n  Initial spectral state comparison (T, lnps):")
# Get Fortran's initial spectral state
state_f_ic = climt.get_default_state([dycore_f], grid_state=grid)
out = dcmip(state_f_ic)
state_f_ic.update(out)

gs_f_ic = GridState(
    u=jnp.array(np.asarray(state_f_ic["eastward_wind"])),
    v=jnp.array(np.asarray(state_f_ic["northward_wind"])),
    temperature=jnp.array(np.asarray(state_f_ic["air_temperature"])),
    vorticity=jnp.zeros_like(jnp.array(np.asarray(state_f_ic["eastward_wind"]))),
    divergence=jnp.zeros_like(jnp.array(np.asarray(state_f_ic["eastward_wind"]))),
    log_surface_pressure=jnp.log(jnp.array(np.asarray(state_f_ic["surface_air_pressure"]))),
    tracers=jnp.stack([jnp.array(np.asarray(state_f_ic["specific_humidity"]))], axis=0),
)
spec_f_ic = grid_to_spectral(gs_f_ic, trans_config)

T_spec_diff = np.abs(np.array(spec_f_ic.temperature) - np.array(spec_orig.temperature))
lnps_spec_diff = np.abs(np.array(spec_f_ic.log_surface_pressure) - np.array(spec_orig.log_surface_pressure))

print(f"  T spectral max diff:    {np.max(T_spec_diff):.6e}")
print(f"  lnps spectral max diff: {np.max(lnps_spec_diff):.6e}")

# If ICs are identical, then the total tendency must differ.
# Let's check: what total div tendency would Fortran need to produce its output?
# From the IMEX formula (simplified for balanced state):
# div_final ≈ some_coeff * L_div  (since total ≈ 0 in all 3 stages)
# The coefficient depends on the full 3-stage accumulation.
# But we can check: is div_F / div_J ≈ constant across all l,k?
print("\n=== Ratio div_F / div_J across modes and levels ===")
print("  Level-averaged ratio for significant m=0 modes:")
for l in [2, 4, 6, 8, 10]:
    ratios = []
    for k in range(N_LEV):
        df = div_f_final[k, l, m0].real
        dj = div_j_final[k, l, m0].real
        if abs(dj) > 1e-25:
            ratios.append(df/dj)
    if ratios:
        ratios = np.array(ratios)
        print(f"  l={l:2d}: mean ratio = {np.mean(ratios):.6f}, std = {np.std(ratios):.6f}, "
              f"range = [{np.min(ratios):.6f}, {np.max(ratios):.6f}]")

print("\nDone.")
