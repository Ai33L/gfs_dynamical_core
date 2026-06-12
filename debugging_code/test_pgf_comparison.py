"""Compare Fortran PGF dump against JAX PGF computation."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'

import numpy as np
import jax.numpy as jnp
from datetime import timedelta
import climt
from sympl import set_constant
from gfs_dynamical_core import GFSDynamicalCore
from gfs_dynamical_core.component_jax import GFSDynamicsJAX

L = 64; N_LON = 2*L-1; N_LAT = L; N_LEV = 20
set_constant("reference_air_pressure", value=1e5, units="Pa")
grid = climt.get_grid(nx=N_LON, ny=N_LAT, nz=N_LEV)
dcmip = climt.DcmipInitialConditions(add_perturbation=False)
dt = timedelta(minutes=5)

# Run Fortran 1 step
dycore_f = GFSDynamicalCore()
state_f = climt.get_default_state([dycore_f], grid_state=grid)
state_f.update(dcmip(state_f))
dycore_f(state_f, dt)
print("Fortran done")

# Read dump
with open('debug_data/fortran_pgf.bin', 'rb') as f:
    nlons, nlats, nlevs = np.fromfile(f, dtype=np.int32, count=3)
    print(f"Grid: {nlons}x{nlats}x{nlevs}")
    s3 = (nlons, nlats, nlevs); s2 = (nlons, nlats)
    prsgx_f = np.fromfile(f, np.float64, np.prod(s3)).reshape(s3, order='F')
    prsgy_f = np.fromfile(f, np.float64, np.prod(s3)).reshape(s3, order='F')
    dTdx_f  = np.fromfile(f, np.float64, np.prod(s3)).reshape(s3, order='F')
    dTdy_f  = np.fromfile(f, np.float64, np.prod(s3)).reshape(s3, order='F')
    dlnpsdx_f = np.fromfile(f, np.float64, np.prod(s2)).reshape(s2, order='F')
    dlnpsdy_f = np.fromfile(f, np.float64, np.prod(s2)).reshape(s2, order='F')
    remaining = f.read()
    print(f"Remaining bytes: {len(remaining)} (should be 0)")

# Run JAX 1 step
dycore_j = GFSDynamicsJAX()
state_j = climt.get_default_state([dycore_j], grid_state=grid)
state_j.update(dcmip(state_j))
dycore_j(state_j, dt)
print("JAX done")

# Compute JAX PGF from initial spectral state
from gfs_dynamical_core.jax.dynamics import compute_pressure_diagnostics, compute_pressure_gradient_force
from gfs_dynamical_core.jax.transforms import spectral_to_grid, grid_to_spectral
from gfs_dynamical_core.jax.states import GridState
import s2fft

tc = dycore_j.trans_config
dc = dycore_j.dyn_config

spec = dycore_j._spec_state
if spec is None:
    T = jnp.array(state_j['air_temperature'])
    lnps = jnp.log(jnp.array(state_j['surface_air_pressure']))
    gs = GridState(
        u=jnp.array(state_j['eastward_wind']),
        v=jnp.array(state_j['northward_wind']),
        temperature=T, log_surface_pressure=lnps,
        tracers=jnp.array(state_j['specific_humidity'])[None],
        vorticity=jnp.zeros_like(T), divergence=jnp.zeros_like(T),
        surface_geopotential=jnp.array(state_j['surface_geopotential']),
    )
    spec = grid_to_spectral(gs, tc)

grid_state, grid_grads = spectral_to_grid(spec, tc)
press_diag = compute_pressure_diagnostics(grid_state.log_surface_pressure, dc)

# Surface geopotential gradients
phis = jnp.array(state_j['surface_geopotential'])
phis_lm = s2fft.forward_jax(phis, tc.L, sampling=tc.sampling)
l_arr = jnp.arange(tc.L)
l_factor = jnp.sqrt(l_arr * (l_arr + 1))
F1 = -l_factor[:, None] * phis_lm
fs1 = s2fft.inverse_jax(F1, tc.L, spin=1, sampling=tc.sampling)
dphisdx = fs1.imag / tc.radius
dphisdy = -fs1.real / tc.radius

pgfx_j, pgfy_j = compute_pressure_gradient_force(
    grid_state.temperature, grid_grads, press_diag, dc, (dphisdx, dphisdy))
pgfx_j = np.array(pgfx_j); pgfy_j = np.array(pgfy_j)

# Compare - Fortran is (nlon, nlat, nlev) BTU, JAX is (nlev, nlat, nlon) BTU
# Transpose Fortran to (nlev, nlat, nlon)
pf_y = prsgy_f.transpose(2, 1, 0)
pf_x = prsgx_f.transpose(2, 1, 0)
df_Tdy = dTdy_f.transpose(2, 1, 0)
df_Tdx = dTdx_f.transpose(2, 1, 0)
df_lnpsdx = dlnpsdx_f.T
df_lnpsdy = dlnpsdy_f.T

dTdy_j = np.array(grid_grads.d_t_d_phi)
dTdx_j = np.array(grid_grads.d_t_d_lambda)
dlnpsdy_j = np.array(grid_grads.d_log_ps_d_phi)
dlnpsdx_j = np.array(grid_grads.d_log_ps_d_lambda)

def compare(name, fort, jax):
    # Use zonal mean if lon dims differ
    if fort.shape != jax.shape:
        fort = fort.mean(axis=-1)
        jax = jax.mean(axis=-1)
        name += " (zm)"
    d = np.abs(fort - jax)
    mx = max(np.abs(fort).max(), 1e-30)
    print(f"{name:25s}: max|F|={np.abs(fort).max():.4e}  max|diff|={d.max():.4e}  rel={d.max()/mx:.4e}")

print(f"\n{'='*70}")
print("PGF OUTPUT COMPARISON")
compare("pgf_y", pf_y, pgfy_j)
compare("pgf_x", pf_x, pgfx_j)
print("\nGRADIENT INPUT COMPARISON")
compare("dTdy", df_Tdy, dTdy_j)
compare("dTdx", df_Tdx, dTdx_j)
compare("dlnpsdy", df_lnpsdy, dlnpsdy_j)
compare("dlnpsdx", df_lnpsdx, dlnpsdx_j)
