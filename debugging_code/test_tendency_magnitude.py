"""
Quick check: compute JAX div tendency two ways and compare magnitudes.
Way 1: From Fortran dump spectral state (via packed_to_s2fft)
Way 2: From climt/DCMIP initialization (same as compare_imex_intermediates.py)
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np
import jax.numpy as jnp
from datetime import timedelta
import climt
import s2fft

L = 64
N_LON = 2 * L - 1
N_LAT = L
N_LEV = 20
NTRUNC = 40
NDIMSPEC = (NTRUNC + 1) * (NTRUNC + 2) // 2

from gfs_dynamical_core.component_jax import GFSDynamicsJAX
from gfs_dynamical_core.jax.transforms import TransformConfig, get_gaussian_latitudes, spectral_to_grid, grid_to_spectral
from gfs_dynamical_core.jax.dynamics import DynamicsConfig, get_spectral_tendencies, full_dynamics_step
from gfs_dynamical_core.jax.transforms import grid_to_spectral_tendencies
from gfs_dynamical_core.jax.states import SpectralState, GridState
from sympl import get_constant

# ── Way 2: climt/DCMIP initialization ──────────────────────────────────
print("=== Way 2: climt/DCMIP initialization ===")
grid = climt.get_grid(nx=N_LON, ny=N_LAT, nz=N_LEV)
dycore_j = GFSDynamicsJAX()
state_j = climt.get_default_state([dycore_j], grid_state=grid)
dcmip = climt.DcmipInitialConditions(add_perturbation=False)
out = dcmip(state_j)
state_j.update(out)

u = jnp.array(state_j["eastward_wind"])
v = jnp.array(state_j["northward_wind"])
temp = jnp.array(state_j["air_temperature"])
ps = jnp.array(state_j["surface_air_pressure"])
q = jnp.array(state_j["specific_humidity"])
phis = jnp.array(state_j["surface_geopotential"])

ak = jnp.array(state_j["atmosphere_hybrid_sigma_pressure_a_coordinate_on_interface_levels"])
bk = jnp.array(state_j["atmosphere_hybrid_sigma_pressure_b_coordinate_on_interface_levels"])
dbk = bk[:-1] - bk[1:]
ck = ak[:-1] * bk[1:] - ak[1:] * bk[:-1]

dc = DynamicsConfig(
    ak=ak, bk=bk, ck=ck, dbk=dbk,
    rk=get_constant("gas_constant_of_dry_air", "J kg^-1 K^-1")
       / get_constant("heat_capacity_of_dry_air_at_constant_pressure", "J kg^-1 K^-1"),
    toa_pressure=0.0,
    radius=get_constant("planetary_radius", "m"),
    omega=get_constant("planetary_rotation_rate", "s^-1"),
    g=get_constant("gravitational_acceleration", "m s^-2"),
    rd=get_constant("gas_constant_of_dry_air", "J kg^-1 K^-1"),
    rv=get_constant("gas_constant_of_vapor_phase", "J kg^-1 K^-1"),
    cp=get_constant("heat_capacity_of_dry_air_at_constant_pressure", "J kg^-1 K^-1"),
    cvap=get_constant("heat_capacity_of_vapor_phase", "J kg^-1 K^-1"),
)
tc = TransformConfig(L=L, sampling="gl", radius=dc.radius)
latitudes = get_gaussian_latitudes(L)

# Build grid state (climt uses (x, y, z) = (lon, lat, lev) with Fortran ordering)
print(f"  u shape from climt: {u.shape}")
print(f"  ps shape from climt: {ps.shape}")

# Climt may return (nlevs, nlats, nlons) or (nlons, nlats, nlevs) depending on version
# Let's detect and handle both
if u.ndim == 3 and u.shape == (N_LEV, N_LAT, N_LON):
    u_grid = u
    v_grid = v
    t_grid = temp
    q_grid = q
elif u.ndim == 3 and u.shape == (N_LON, N_LAT, N_LEV):
    u_grid = jnp.transpose(u, (2, 1, 0))
    v_grid = jnp.transpose(v, (2, 1, 0))
    t_grid = jnp.transpose(temp, (2, 1, 0))
    q_grid = jnp.transpose(q, (2, 1, 0))
else:
    raise ValueError(f"Unexpected u shape: {u.shape}")

if ps.ndim == 2:
    ps_grid = ps  # already (nlats, nlons) or (nlons, nlats)
    if ps.shape == (N_LON, N_LAT):
        ps_grid = ps.T
elif ps.ndim == 3:
    ps_grid = ps[:, :, 0].T
else:
    raise ValueError(f"Unexpected ps shape: {ps.shape}")

lnps_grid = jnp.log(ps_grid)
if phis.ndim == 2:
    phis_grid = phis if phis.shape == (N_LAT, N_LON) else phis.T
else:
    phis_grid = phis[:, :, 0].T

print(f"  u_grid max: {float(jnp.abs(u_grid).max()):.4e}")
print(f"  T_grid max: {float(jnp.abs(t_grid).max()):.4e}")

# Convert to spectral using JAX forward transform
grid_state_climt = GridState(
    u=u_grid, v=v_grid, temperature=t_grid,
    vorticity=jnp.zeros_like(u_grid),  # placeholder, will be recomputed
    divergence=jnp.zeros_like(u_grid),
    log_surface_pressure=lnps_grid,
    tracers=q_grid[None, :, :, :],
)
spec_state_climt = grid_to_spectral(grid_state_climt, tc)

# Now do full roundtrip: spectral -> grid -> tendencies -> spectral
grid_state_2, grid_grads_2 = spectral_to_grid(spec_state_climt, tc)
phis_grads_2 = (jnp.zeros((N_LAT, N_LON)), jnp.zeros((N_LAT, N_LON)))

# Compute spectral tendencies
spec_tends_2 = get_spectral_tendencies(
    spec_state_climt, phis_grads_2, dc, tc, latitudes
)

ddivdt_2 = np.array(spec_tends_2.d_divergence_d_t)
m0_idx = L - 1
print(f"\n  d_div/dt (climt init):")
print(f"    max = {np.abs(ddivdt_2).max():.4e}")
for l_val in [2, 4, 6, 8]:
    print(f"    l={l_val}, m=0: max|val| = {np.abs(ddivdt_2[:, l_val, m0_idx]).max():.4e}")

# Also convert to packed format for direct comparison with Fortran
def s2fft_to_packed(arr_2d, ntrunc):
    """Convert s2fft 2D to SHTNS packed m-first format."""
    arr = np.array(arr_2d)
    ndimspec = (ntrunc + 1) * (ntrunc + 2) // 2
    if arr.ndim == 3:
        nlevs = arr.shape[0]
        L_loc = arr.shape[1]
        out = np.zeros((ndimspec, nlevs), dtype=arr.dtype)
        idx = 0
        for m in range(ntrunc + 1):
            for l in range(m, ntrunc + 1):
                out[idx, :] = arr[:, l, (L_loc - 1) + m]
                idx += 1
        return out

ddivdt_packed = s2fft_to_packed(ddivdt_2, NTRUNC)
print(f"\n  d_div/dt in packed format:")
print(f"    max = {np.abs(ddivdt_packed).max():.4e}")
# Mode 2 (m=0, l=2) is at packed index 2
print(f"    packed idx 2 (l=2,m=0): max = {np.abs(ddivdt_packed[2, :]).max():.4e}")

# Check initial spectral state
print(f"\n  Initial spectral state (climt):")
vrt_climt = np.array(spec_state_climt.vorticity)
print(f"    vrt max = {np.abs(vrt_climt).max():.4e}")
print(f"    vrt l=2,m=0 max = {np.abs(vrt_climt[:, 2, m0_idx]).max():.4e}")

# ── Now read Fortran dump and compare spectral states ──────────────────
print("\n=== Comparing spectral states ===")

def packed_to_s2fft(packed, ntrunc, L):
    if packed.ndim == 2:
        nlevs = packed.shape[1]
        out = np.zeros((nlevs, L, 2 * L - 1), dtype=packed.dtype)
        idx = 0
        for m in range(ntrunc + 1):
            for l in range(m, ntrunc + 1):
                out[:, l, (L - 1) + m] = packed[idx, :]
                if m > 0:
                    out[:, l, (L - 1) - m] = np.conj(packed[idx, :]) * (-1)**m
                idx += 1
        return out

dump_path = "debug_data/fortran_step_1_stage_0.bin"
if os.path.exists(dump_path):
    gsz = N_LON * N_LAT
    with open(dump_path, "rb") as f:
        f.seek(gsz * N_LEV * 8 * 3 + gsz * 8 + gsz * N_LEV * 8)  # skip grid fields
        vrtspec_f = np.frombuffer(f.read(NDIMSPEC * N_LEV * 16),
                                   dtype=np.complex128).reshape((NDIMSPEC, N_LEV), order="F").copy()

    vrt_dump = packed_to_s2fft(vrtspec_f, NTRUNC, L)
    print(f"  Fortran dump vrt max = {np.abs(vrt_dump).max():.4e}")
    print(f"  Fortran dump vrt l=2,m=0 max = {np.abs(vrt_dump[:, 2, m0_idx]).max():.4e}")

    diff_vrt = np.abs(vrt_climt - vrt_dump)
    print(f"  vrt diff (climt vs dump): max = {diff_vrt.max():.4e}, rel = {diff_vrt.max() / np.abs(vrt_climt).max():.4e}")

    # Now compute tendency from dump spectral state
    spec_state_dump = SpectralState(
        vorticity=jnp.array(vrt_dump),
        divergence=jnp.array(packed_to_s2fft(
            np.frombuffer(open(dump_path, "rb").read()[
                gsz*N_LEV*8*3 + gsz*8 + gsz*N_LEV*8 + NDIMSPEC*N_LEV*16:
                gsz*N_LEV*8*3 + gsz*8 + gsz*N_LEV*8 + NDIMSPEC*N_LEV*16*2
            ], dtype=np.complex128).reshape((NDIMSPEC, N_LEV), order="F").copy(),
            NTRUNC, L)),
        temperature=jnp.array(packed_to_s2fft(
            np.frombuffer(open(dump_path, "rb").read()[
                gsz*N_LEV*8*3 + gsz*8 + gsz*N_LEV*8 + NDIMSPEC*N_LEV*16*2:
                gsz*N_LEV*8*3 + gsz*8 + gsz*N_LEV*8 + NDIMSPEC*N_LEV*16*3
            ], dtype=np.complex128).reshape((NDIMSPEC, N_LEV), order="F").copy(),
            NTRUNC, L)),
        log_surface_pressure=jnp.array(packed_to_s2fft(
            np.frombuffer(open(dump_path, "rb").read()[
                gsz*N_LEV*8*3 + gsz*8 + gsz*N_LEV*8 + NDIMSPEC*N_LEV*16*3:
                gsz*N_LEV*8*3 + gsz*8 + gsz*N_LEV*8 + NDIMSPEC*N_LEV*16*3 + NDIMSPEC*16
            ], dtype=np.complex128).copy(),
            NTRUNC, L).squeeze()),
        tracers=jnp.zeros((1, N_LEV, L, 2*L-1)),
    )

    spec_tends_dump = get_spectral_tendencies(
        spec_state_dump, phis_grads_2, dc, tc, latitudes
    )
    ddivdt_dump = np.array(spec_tends_dump.d_divergence_d_t)
    print(f"\n  d_div/dt (from dump):")
    print(f"    max = {np.abs(ddivdt_dump).max():.4e}")
    for l_val in [2, 4, 6, 8]:
        print(f"    l={l_val}, m=0: max|val| = {np.abs(ddivdt_dump[:, l_val, m0_idx]).max():.4e}")

    # Compare
    diff_tend = np.abs(ddivdt_2 - ddivdt_dump)
    print(f"\n  d_div/dt comparison (climt vs dump):")
    print(f"    max|diff| = {diff_tend.max():.4e}")
    print(f"    rel = {diff_tend.max() / np.abs(ddivdt_2).max():.4e}")
else:
    print(f"  {dump_path} not found, skipping dump comparison")

print("\nDone.")
