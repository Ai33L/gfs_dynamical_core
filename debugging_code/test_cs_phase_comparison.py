"""
Quick diagnostic: does the ~200% div error in compare_full_run vanish
when we account for the Condon-Shortley phase convention difference?

s2fft includes CS phase by default; SHTNS was initialized with SHT_NO_CS_PHASE.
For real fields: f_lm^{SHTNS} = (-1)^m * f_lm^{s2fft}

This script reads the step-1 Fortran dump and the JAX spectral state,
compares WITH and WITHOUT the (-1)^m correction.
"""
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

L = 64
N_LON = 2 * L - 1
N_LAT = L
N_LEV = 20
DT_MIN = 5
NTRUNC = 40

set_constant("reference_air_pressure", value=1e5, units="Pa")
grid = climt.get_grid(nx=N_LON, ny=N_LAT, nz=N_LEV)
dcmip = climt.DcmipInitialConditions(add_perturbation=True)
dt = timedelta(minutes=DT_MIN)


def s2fft_to_packed_3d(arr_3d, ntrunc):
    nlevs, L_loc, _ = arr_3d.shape
    ndimspec = (ntrunc + 1) * (ntrunc + 2) // 2
    out = np.zeros((ndimspec, nlevs), dtype=arr_3d.dtype)
    idx = 0
    for m in range(ntrunc + 1):
        for l in range(m, ntrunc + 1):
            out[idx, :] = arr_3d[:, l, L_loc - 1 + m]
            idx += 1
    return out


def s2fft_to_packed(arr_2d, ntrunc):
    L_loc = arr_2d.shape[0]
    ndimspec = (ntrunc + 1) * (ntrunc + 2) // 2
    out = np.zeros(ndimspec, dtype=arr_2d.dtype)
    idx = 0
    for m in range(ntrunc + 1):
        for l in range(m, ntrunc + 1):
            out[idx] = arr_2d[l, L_loc - 1 + m]
            idx += 1
    return out


def build_cs_correction(ntrunc):
    """Build (-1)^m array in packed ordering."""
    ndimspec = (ntrunc + 1) * (ntrunc + 2) // 2
    cs = np.ones(ndimspec, dtype=np.float64)
    idx = 0
    for m in range(ntrunc + 1):
        for l in range(m, ntrunc + 1):
            cs[idx] = (-1.0) ** m
            idx += 1
    return cs


def read_fortran_dump(step):
    path = f'debug_data/fortran_step_{step:05d}.bin'
    with open(path, 'rb') as f:
        ndimspec, nlevs, step_num = np.fromfile(f, np.int32, 3)
        def read_c2d():
            return np.fromfile(f, np.complex128, ndimspec * nlevs).reshape(
                (ndimspec, nlevs), order='F')
        def read_c1d():
            return np.fromfile(f, np.complex128, ndimspec)
        data = {}
        data['vrtspec'] = read_c2d()
        data['divspec'] = read_c2d()
        data['virtempspec'] = read_c2d()
        data['lnpsspec'] = read_c1d()
    return data


def compare(name, f_packed, j_packed, cs_corr=None):
    """Compare with and without CS correction."""
    diff_raw = np.abs(f_packed - j_packed)
    maxf = max(np.abs(f_packed).max(), 1e-30)
    maxj = max(np.abs(j_packed).max(), 1e-30)
    rel_raw = diff_raw.max() / max(maxf, maxj)

    if cs_corr is not None:
        if j_packed.ndim == 2:
            j_corrected = j_packed * cs_corr[:, None]
        else:
            j_corrected = j_packed * cs_corr
        diff_corr = np.abs(f_packed - j_corrected)
        rel_corr = diff_corr.max() / max(maxf, maxj)
    else:
        rel_corr = rel_raw

    print(f"  {name:12s}: raw_rel={rel_raw:.3e}  cs_corrected_rel={rel_corr:.3e}  "
          f"max|F|={maxf:.3e}  max|J|={maxj:.3e}")


# --- Run one step each ---
print("Initializing Fortran dycore...")
dycore_f = GFSDynamicalCore()
state_f = climt.get_default_state([dycore_f], grid_state=grid)
state_f.update(dcmip(state_f))

print("Initializing JAX dycore...")
dycore_j = GFSDynamicsJAX()
state_j = climt.get_default_state([dycore_j], grid_state=grid)
state_j.update(dcmip(state_j))

print("Stepping Fortran...")
diag_f, out_f = dycore_f(state_f, timestep=dt)
state_f.update(out_f)

print("Stepping JAX...")
diag_j, out_j = dycore_j(state_j, timestep=dt)
state_j.update(out_j)

# --- Read Fortran dump and JAX state ---
fdata = read_fortran_dump(1)
jspec = dycore_j._spec_state

j_vrt = s2fft_to_packed_3d(np.array(jspec.vorticity), NTRUNC)
j_div = s2fft_to_packed_3d(np.array(jspec.divergence), NTRUNC)
j_temp = s2fft_to_packed_3d(np.array(jspec.temperature), NTRUNC)
j_lnps = s2fft_to_packed(np.array(jspec.log_surface_pressure), NTRUNC)

cs = build_cs_correction(NTRUNC)

print("\n=== Step 1 comparison: raw vs CS-corrected ===")
compare("vrtspec", fdata['vrtspec'], j_vrt, cs)
compare("divspec", fdata['divspec'], j_div, cs)
compare("virtempspec", fdata['virtempspec'], j_temp, cs)
compare("lnpsspec", fdata['lnpsspec'], j_lnps, cs)

# --- Per-m breakdown for divspec ---
print("\n=== Divspec per-m breakdown (step 1) ===")
print(f"  {'m':>3}  {'max|F|':>12}  {'max|J|':>12}  {'raw_rel':>12}  {'cs_rel':>12}")
idx = 0
for m in range(min(NTRUNC + 1, 10)):
    n_modes = NTRUNC + 1 - m
    f_slice = fdata['divspec'][idx:idx+n_modes, :]
    j_slice = j_div[idx:idx+n_modes, :]
    j_corr = j_slice * ((-1.0)**m)

    maxf = np.abs(f_slice).max()
    maxj = np.abs(j_slice).max()
    maxref = max(maxf, maxj, 1e-30)
    raw_rel = np.abs(f_slice - j_slice).max() / maxref
    cs_rel = np.abs(f_slice - j_corr).max() / maxref

    print(f"  {m:3d}  {maxf:12.3e}  {maxj:12.3e}  {raw_rel:12.3e}  {cs_rel:12.3e}")
    idx += n_modes
