"""Compare JAX inverse transform vs Fortran grid output from SAME spectral coefficients.

This answers: do SHTNS and s2fft produce the same grid fields from the same spectral input?
If not, the inverse vector transform is the root cause.
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np
import jax.numpy as jnp
import climt
import s2fft
from sympl import set_constant
from pathlib import Path
from gfs_dynamical_core import GFSDynamicalCore
from gfs_dynamical_core.component_jax import GFSDynamicsJAX
from gfs_dynamical_core.jax.states import GridState, SpectralState
from gfs_dynamical_core.jax.transforms import (
    TransformConfig, spectral_to_grid, grid_to_spectral
)
from datetime import timedelta

set_constant("reference_air_pressure", value=1e5, units="Pa")

L = 64
NLONS = 2 * L - 1
NLATS = L
NLEVS = 20
NTRAC = 1
NTRUNC = int(NLONS / 3 - 2)
NDIMSPEC = (NTRUNC + 1) * (NTRUNC + 2) // 2

def shtns_packed_to_s2fft(packed_coeffs, L, ntrunc):
    """Convert SHTNS packed (m>=0 triangular) to s2fft 2D (L, 2L-1) format.

    Applies (-1)^m CS phase correction since SHTNS uses NO_CS_PHASE
    and s2fft includes it.
    """
    flm = np.zeros((L, 2 * L - 1), dtype=np.complex128)
    idx = 0
    for m in range(ntrunc + 1):
        for l in range(m, ntrunc + 1):
            if l < L:
                j_pos = m + L - 1      # positive m index in s2fft
                j_neg = -m + L - 1     # negative m index in s2fft
                coeff = packed_coeffs[idx]
                cs_factor = (-1) ** m
                flm[l, j_pos] = cs_factor * coeff
                if m > 0:
                    # Conjugate symmetry for real fields (with CS phase)
                    flm[l, j_neg] = cs_factor * (-1)**m * np.conj(coeff)
            idx += 1
    return flm

def load_fortran_dump(path):
    data = Path(path).read_bytes()
    offset = 0
    def read_array(shape, dtype=np.float64):
        nonlocal offset
        n = int(np.prod(shape)) * np.dtype(dtype).itemsize
        arr = np.frombuffer(data[offset:offset+n], dtype=dtype)
        offset += n
        return arr.reshape(shape[::-1]).T

    result = {}
    result["ug"] = read_array((NLONS, NLATS, NLEVS))
    result["vg"] = read_array((NLONS, NLATS, NLEVS))
    result["virtempg"] = read_array((NLONS, NLATS, NLEVS))
    result["lnpsg"] = read_array((NLONS, NLATS))
    result["tracerg"] = read_array((NLONS, NLATS, NLEVS, NTRAC))
    result["vrtspec"] = read_array((NDIMSPEC, NLEVS), dtype=np.complex128)
    result["divspec"] = read_array((NDIMSPEC, NLEVS), dtype=np.complex128)
    result["virtempspec"] = read_array((NDIMSPEC, NLEVS), dtype=np.complex128)
    result["lnpsspec"] = read_array((NDIMSPEC,), dtype=np.complex128)
    assert offset == len(data)

    # Reorder grid fields to (lev, lat, lon), flip lat to N->S
    for name in ["ug", "vg", "virtempg"]:
        result[name] = result[name].transpose(2, 1, 0)[:, ::-1, :]
    result["lnpsg"] = result["lnpsg"].T[::-1, :]
    for name in ["vrtspec", "divspec", "virtempspec"]:
        result[name] = result[name].T
    return result

# --- Setup and run Fortran to get debug dump ---
grid = climt.get_grid(nx=NLONS, ny=NLATS, nz=NLEVS)
dycore_f = GFSDynamicalCore()
state_f = climt.get_default_state([dycore_f], grid_state=grid)
dcmip = climt.DcmipInitialConditions(add_perturbation=True)
state_f.update(dcmip(state_f))

# Save true IC
u_true = np.array(state_f["eastward_wind"].values)
v_true = np.array(state_f["northward_wind"].values)

timestep = timedelta(minutes=5)
dycore_f(state_f, timestep=timestep)

dump_path = Path("debug_data/fortran_step_1_stage_0.bin")
if not dump_path.exists():
    print("ERROR: No Fortran dump found")
    exit(1)

f_dump = load_fortran_dump(dump_path)

# --- Convert Fortran spectral to s2fft format ---
vrt_s2fft = np.stack([shtns_packed_to_s2fft(f_dump["vrtspec"][k], L, NTRUNC) for k in range(NLEVS)])
div_s2fft = np.stack([shtns_packed_to_s2fft(f_dump["divspec"][k], L, NTRUNC) for k in range(NLEVS)])
tmp_s2fft = np.stack([shtns_packed_to_s2fft(f_dump["virtempspec"][k], L, NTRUNC) for k in range(NLEVS)])
lnps_s2fft = shtns_packed_to_s2fft(f_dump["lnpsspec"], L, NTRUNC)

# Build SpectralState from Fortran coefficients (in s2fft format)
spec_from_fortran = SpectralState(
    vorticity=jnp.array(vrt_s2fft),
    divergence=jnp.array(div_s2fft),
    temperature=jnp.array(tmp_s2fft),
    log_surface_pressure=jnp.array(lnps_s2fft),
    tracers=jnp.zeros((1, NLEVS, L, 2*L-1), dtype=jnp.complex128),
)

# Apply JAX inverse transform
config = TransformConfig(L=L, sampling="gl", radius=6371000.0)
grid_from_jax, _ = spectral_to_grid(spec_from_fortran, config)

u_jax = np.array(grid_from_jax.u.real)
v_jax = np.array(grid_from_jax.v.real)
t_jax = np.array(grid_from_jax.temperature.real)

# Fortran grid output
u_fortran = f_dump["ug"]
v_fortran = f_dump["vg"]
t_fortran = f_dump["virtempg"]

print("=== Comparing JAX inverse transform vs Fortran grid output ===")
print("(Both using Fortran's spectral coefficients as input)\n")

for name, jax_arr, fort_arr in [("u (eastward)", u_jax, u_fortran),
                                  ("v (northward)", v_jax, v_fortran),
                                  ("T (temperature)", t_jax, t_fortran)]:
    diff = np.abs(jax_arr - fort_arr)
    max_diff = diff.max()
    max_val = max(np.abs(jax_arr).max(), np.abs(fort_arr).max())
    rel_err = max_diff / max_val if max_val > 0 else 0

    print(f"{name:20s}: max|diff|={max_diff:.4e}, max|val|={max_val:.4e}, rel_err={rel_err:.4e}")

    # Check zonal mean vs eddy decomposition
    zm_diff = np.abs(jax_arr.mean(axis=-1) - fort_arr.mean(axis=-1)).max()
    eddy_j = jax_arr - jax_arr.mean(axis=-1, keepdims=True)
    eddy_f = fort_arr - fort_arr.mean(axis=-1, keepdims=True)
    eddy_diff = np.abs(eddy_j - eddy_f).max()
    print(f"  {'zonal mean diff':20s}: {zm_diff:.4e}")
    print(f"  {'eddy diff':20s}: {eddy_diff:.4e}")

# Also check: does Fortran ug look like u_east or V_theta?
print("\n=== Sanity check: are Fortran ug/vg physical winds? ===")
print(f"Fortran ug vs true u_east: max|diff|={np.abs(u_fortran - u_true).max():.4e}")
print(f"Fortran vg vs true v_north: max|diff|={np.abs(v_fortran - v_true).max():.4e}")
print(f"Fortran ug vs true v_north: max|diff|={np.abs(u_fortran - v_true).max():.4e}")
print(f"Fortran vg vs true u_east: max|diff|={np.abs(v_fortran - u_true).max():.4e}")

# Check if Fortran's ug might be -v_north (i.e., V_theta)
print(f"Fortran ug vs -true v_north: max|diff|={np.abs(u_fortran + v_true).max():.4e}")
print(f"Fortran vg vs -true u_east: max|diff|={np.abs(v_fortran + u_true).max():.4e}")
