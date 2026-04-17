"""
Diagnose whether the 2% v error comes from:
  (A) Different spectral divergence between Fortran & JAX (stepper/tendency bug)
  (B) Same divergence but different v (transform bug in div→v for m=0)

Method:
  1. Run both dycores 1 step from identical JW06 ICs
  2. Forward-transform BOTH grid outputs using s2fft to recover spectral (vort, div)
  3. Compare the recovered spectral div → if they differ, it's (A); if same, it's (B)
  4. Also access JAX's cached spectral state directly for cross-check
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
from gfs_dynamical_core.jax.transforms import grid_to_spectral, spectral_to_grid, TransformConfig
from gfs_dynamical_core.jax.states import GridState, SpectralState

# ── Setup ──
L = 64
N_LON = 2 * L - 1
N_LAT = L
N_LEV = 20
set_constant("reference_air_pressure", value=1e5, units="Pa")
grid = climt.get_grid(nx=N_LON, ny=N_LAT, nz=N_LEV)
timestep = timedelta(minutes=5)

# JW06 ICs, no perturbation
dcmip = climt.DcmipInitialConditions(add_perturbation=False)

# Fortran
dycore_f = GFSDynamicalCore()
state_f = climt.get_default_state([dycore_f], grid_state=grid)
out = dcmip(state_f)
state_f.update(out)

# JAX
dycore_j = GFSDynamicsJAX()
state_j = climt.get_default_state([dycore_j], grid_state=grid)
out = dcmip(state_j)
state_j.update(out)

# Step both once
diag_f, out_f = dycore_f(state_f, timestep=timestep)
state_f.update(out_f)

diag_j, out_j = dycore_j(state_j, timestep=timestep)
state_j.update(out_j)

# ── Grid comparison ──
u_f = np.asarray(state_f["eastward_wind"])
v_f = np.asarray(state_f["northward_wind"])
u_j = np.asarray(state_j["eastward_wind"])
v_j = np.asarray(state_j["northward_wind"])

print("=== Grid v comparison ===")
v_diff = np.abs(v_f - v_j)
print(f"  max|v_F| = {np.max(np.abs(v_f)):.6e}")
print(f"  max|v_J| = {np.max(np.abs(v_j)):.6e}")
print(f"  max|diff| = {np.max(v_diff):.6e}")
print(f"  rel_err = {np.max(v_diff) / max(np.max(np.abs(v_f)), 1e-30):.6e}")

# Zonal mean decomposition
v_f_zm = v_f.mean(axis=-1)  # average over longitude
v_j_zm = v_j.mean(axis=-1)
v_f_eddy = v_f - v_f_zm[..., None]
v_j_eddy = v_j - v_j_zm[..., None]
print(f"\n  Zonal mean diff max = {np.max(np.abs(v_f_zm - v_j_zm)):.6e}")
print(f"  Eddy diff max       = {np.max(np.abs(v_f_eddy - v_j_eddy)):.6e}")

# ── Forward-transform both grid outputs to spectral using s2fft ──
print("\n=== Forward-transforming both grid outputs via s2fft ===")
trans_config = TransformConfig(L=L, sampling="gl", radius=6.371e6)

def grid_to_spec(u, v, temp, ps, q):
    """Forward transform grid fields to spectral."""
    gs = GridState(
        u=jnp.array(u),
        v=jnp.array(v),
        temperature=jnp.array(temp),
        vorticity=jnp.zeros_like(jnp.array(u)),
        divergence=jnp.zeros_like(jnp.array(u)),
        log_surface_pressure=jnp.log(jnp.array(ps)),
        tracers=jnp.stack([jnp.array(q)], axis=0),
    )
    return grid_to_spectral(gs, trans_config)

T_f = np.asarray(state_f["air_temperature"])
ps_f = np.asarray(state_f["surface_air_pressure"])
q_f = np.asarray(state_f["specific_humidity"])
T_j = np.asarray(state_j["air_temperature"])
ps_j = np.asarray(state_j["surface_air_pressure"])
q_j = np.asarray(state_j["specific_humidity"])

spec_from_fortran = grid_to_spec(u_f, v_f, T_f, ps_f, q_f)
spec_from_jax_grid = grid_to_spec(u_j, v_j, T_j, ps_j, q_j)

# Also get JAX's cached spectral state directly
spec_cached = dycore_j._spec_state

print("\n=== Spectral divergence comparison ===")
div_from_f = np.array(spec_from_fortran.divergence)
div_from_j_grid = np.array(spec_from_jax_grid.divergence)
div_cached = np.array(spec_cached.divergence)

# Compare div recovered from Fortran grid vs div recovered from JAX grid
div_fj_diff = np.abs(div_from_f - div_from_j_grid)
div_fj_max = max(np.max(np.abs(div_from_f)), np.max(np.abs(div_from_j_grid)), 1e-30)
print(f"  div(from F grid) vs div(from J grid): max|diff| = {np.max(div_fj_diff):.6e}, rel = {np.max(div_fj_diff)/div_fj_max:.6e}")

# Compare div recovered from JAX grid vs JAX's cached spectral div
div_cache_diff = np.abs(div_from_j_grid - div_cached)
print(f"  div(from J grid) vs div(cached):      max|diff| = {np.max(div_cache_diff):.6e}, rel = {np.max(div_cache_diff)/div_fj_max:.6e}")

# Compare div recovered from Fortran grid vs JAX's cached spectral div
div_fc_diff = np.abs(div_from_f - div_cached)
print(f"  div(from F grid) vs div(cached):      max|diff| = {np.max(div_fc_diff):.6e}, rel = {np.max(div_fc_diff)/div_fj_max:.6e}")

# ── Zonal mean (m=0) of spectral div ──
print("\n=== Spectral divergence m=0 comparison ===")
# In s2fft MW/GL format, m=0 is the middle column: index L-1
m0_idx = L - 1
div_from_f_m0 = div_from_f[:, :, m0_idx]
div_from_j_m0 = div_from_j_grid[:, :, m0_idx]
div_cached_m0 = div_cached[:, :, m0_idx]

print(f"  div_F(m=0) vs div_J(m=0): max|diff| = {np.max(np.abs(div_from_f_m0 - div_from_j_m0)):.6e}")
print(f"  div_F(m=0) vs div_cached(m=0): max|diff| = {np.max(np.abs(div_from_f_m0 - div_cached_m0)):.6e}")
print(f"  div_J(m=0) vs div_cached(m=0): max|diff| = {np.max(np.abs(div_from_j_m0 - div_cached_m0)):.6e}")

# Show the actual m=0 div profiles at a few levels
print("\n  m=0 divergence profile (level 10):")
print(f"    l   |  div_from_F      |  div_from_J_grid |  div_cached      |  F-cached")
for l in range(min(10, L)):
    df = div_from_f_m0[10, l]
    dj = div_from_j_m0[10, l]
    dc = div_cached_m0[10, l]
    print(f"    {l:3d} |  {df.real:+.6e}  |  {dj.real:+.6e}  |  {dc.real:+.6e}  |  {(df-dc).real:+.6e}")

# ── Vorticity comparison too ──
print("\n=== Spectral vorticity comparison ===")
vort_from_f = np.array(spec_from_fortran.vorticity)
vort_from_j = np.array(spec_from_jax_grid.vorticity)
vort_cached = np.array(spec_cached.vorticity)
vort_max = max(np.max(np.abs(vort_from_f)), 1e-30)

print(f"  vort(from F) vs vort(cached): max|diff| = {np.max(np.abs(vort_from_f - vort_cached)):.6e}, rel = {np.max(np.abs(vort_from_f - vort_cached))/vort_max:.6e}")
print(f"  vort(from F) vs vort(from J): max|diff| = {np.max(np.abs(vort_from_f - vort_from_j)):.6e}, rel = {np.max(np.abs(vort_from_f - vort_from_j))/vort_max:.6e}")

# ── KEY TEST: Reconstruct v from Fortran's spectral state using s2fft ──
print("\n=== KEY TEST: Reconstruct v from Fortran-derived spectral state ===")
# If we take Fortran's grid output, forward-transform to spectral, then
# inverse-transform back, do we get Fortran's v back?
# (This tests the s2fft roundtrip on Fortran's actual data)
grid_from_f_spec, _ = spectral_to_grid(spec_from_fortran, trans_config)
v_roundtrip = np.array(grid_from_f_spec.v)

print(f"  v_F (original Fortran grid):     max = {np.max(np.abs(v_f)):.6e}")
print(f"  v_F_roundtrip (F→s2fft→s2fft):   max = {np.max(np.abs(v_roundtrip)):.6e}")
print(f"  diff (v_F - v_F_roundtrip):      max = {np.max(np.abs(v_f - v_roundtrip)):.6e}")
print(f"  → This is the s2fft roundtrip error on Fortran's data")

# Now reconstruct v from JAX's cached spectral state
grid_from_cached, _ = spectral_to_grid(spec_cached, trans_config)
v_from_cached = np.array(grid_from_cached.v)

print(f"\n  v_J (JAX grid output):           max = {np.max(np.abs(v_j)):.6e}")
print(f"  v_from_cached (cached→s2fft):    max = {np.max(np.abs(v_from_cached)):.6e}")
print(f"  diff (v_J - v_from_cached):      max = {np.max(np.abs(v_j - v_from_cached)):.6e}")
print(f"  → This checks JAX internal consistency")

# THE DECISIVE TEST: reconstruct v from Fortran's spectral div using s2fft
# vs Fortran's actual grid v
print(f"\n  v_F - v_from_F_spectral:         max = {np.max(np.abs(v_f - v_roundtrip)):.6e}")
print(f"  v_F - v_from_cached_spectral:    max = {np.max(np.abs(v_f - v_from_cached)):.6e}")
print(f"  v_roundtrip - v_from_cached:     max = {np.max(np.abs(v_roundtrip - v_from_cached)):.6e}")

# Zonal mean of the roundtrip test
v_rt_zm = v_roundtrip.mean(axis=-1)
v_fc_zm = v_from_cached.mean(axis=-1)
print(f"\n  Zonal mean tests:")
print(f"    v_F_zm - v_roundtrip_zm:     max = {np.max(np.abs(v_f_zm - v_rt_zm)):.6e}")
print(f"    v_F_zm - v_from_cached_zm:   max = {np.max(np.abs(v_f_zm - v_fc_zm)):.6e}")
print(f"    v_roundtrip_zm - v_cached_zm: max = {np.max(np.abs(v_rt_zm - v_fc_zm)):.6e}")

print("\n=== INTERPRETATION ===")
rt_err = np.max(np.abs(v_f - v_roundtrip))
fj_err = np.max(np.abs(v_f - v_j))
spec_div_err = np.max(np.abs(div_from_f - div_cached)) / div_fj_max

if rt_err > 0.5 * fj_err:
    print("  The s2fft roundtrip error on Fortran's data is comparable to the F-J diff.")
    print("  → The transform itself introduces error on this data pattern.")
    print("  → Likely a TRANSFORM issue (hypothesis B)")
elif spec_div_err > 0.01:
    print(f"  Spectral div relative error = {spec_div_err:.4e} (>1%)")
    print("  → Fortran and JAX produce DIFFERENT spectral divergence")
    print("  → Likely a STEPPER/TENDENCY issue (hypothesis A)")
else:
    print(f"  Spectral div relative error = {spec_div_err:.4e} (<1%)")
    print(f"  s2fft roundtrip error = {rt_err:.6e}")
    print(f"  F-J grid error = {fj_err:.6e}")
    print("  → Need further analysis to determine source")

print("\nDone.")
