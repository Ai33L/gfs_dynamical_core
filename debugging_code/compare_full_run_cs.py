"""
Full-run comparison with Condon-Shortley phase correction.
s2fft includes CS phase; SHTNS does not. Correction: multiply s2fft
coefficients by (-1)^m before comparing with SHTNS packed format.
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
N_STEPS = 2500
DT_MIN = 5
NTRUNC = 40

set_constant("reference_air_pressure", value=1e5, units="Pa")
grid = climt.get_grid(nx=N_LON, ny=N_LAT, nz=N_LEV)
dcmip = climt.DcmipInitialConditions(add_perturbation=True)
dt = timedelta(minutes=DT_MIN)


def build_cs_packed(ntrunc):
    """(-1)^m in SHTNS packed ordering."""
    ndimspec = (ntrunc + 1) * (ntrunc + 2) // 2
    cs = np.ones(ndimspec, dtype=np.float64)
    idx = 0
    for m in range(ntrunc + 1):
        for l in range(m, ntrunc + 1):
            cs[idx] = (-1.0) ** m
            idx += 1
    return cs


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


def read_fortran_dump(step):
    path = f'debug_data/fortran_step_{step:05d}.bin'
    if not os.path.exists(path):
        return None
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


def compare_packed(f_packed, j_packed, cs_corr):
    """Compare with CS correction. Returns raw and corrected relative errors."""
    if j_packed.ndim == 2:
        j_corr = j_packed * cs_corr[:, None]
    else:
        j_corr = j_packed * cs_corr

    maxf = max(np.abs(f_packed).max(), 1e-30)
    rel_corr = np.abs(f_packed - j_corr).max() / maxf
    return rel_corr


def main():
    cs = build_cs_packed(NTRUNC)

    print(f"Grid: {N_LON}x{N_LAT}x{N_LEV}, L={L}, ntrunc={NTRUNC}")
    print(f"Running {N_STEPS} steps ({N_STEPS * DT_MIN / 60:.1f} hours)")
    print("CS phase correction: ENABLED\n")

    dycore_f = GFSDynamicalCore()
    state_f = climt.get_default_state([dycore_f], grid_state=grid)
    state_f.update(dcmip(state_f))

    dycore_j = GFSDynamicsJAX()
    state_j = climt.get_default_state([dycore_j], grid_state=grid)
    state_j.update(dcmip(state_j))

    print(f"{'step':>5} {'hour':>6} | {'vrt':>10} {'div':>10} {'T':>10} {'lnps':>10} | "
          f"{'ps_F':>10} {'ps_J':>10} {'dps':>10}")
    print("-" * 110)

    for step in range(1, N_STEPS + 1):
        diag_f, out_f = dycore_f(state_f, timestep=dt)
        state_f.update(out_f)
        state_f['time'] += dt

        diag_j, out_j = dycore_j(state_j, timestep=dt)
        state_j.update(out_j)
        state_j['time'] += dt

        fdata = read_fortran_dump(step)
        if fdata is None:
            continue

        jspec = dycore_j._spec_state
        if jspec is None:
            continue

        j_vrt = s2fft_to_packed_3d(np.array(jspec.vorticity), NTRUNC)
        j_div = s2fft_to_packed_3d(np.array(jspec.divergence), NTRUNC)
        j_temp = s2fft_to_packed_3d(np.array(jspec.temperature), NTRUNC)
        j_lnps = s2fft_to_packed(np.array(jspec.log_surface_pressure), NTRUNC)

        rel_vrt = compare_packed(fdata['vrtspec'], j_vrt, cs)
        rel_div = compare_packed(fdata['divspec'], j_div, cs)
        rel_T = compare_packed(fdata['virtempspec'], j_temp, cs)
        rel_lnps = compare_packed(fdata['lnpsspec'], j_lnps, cs)

        ps_f = np.asarray(state_f['surface_air_pressure'])
        ps_j = np.asarray(state_j['surface_air_pressure'])
        if np.iscomplexobj(ps_j):
            ps_j = ps_j.real

        hour = step * DT_MIN / 60.0
        dps = (ps_j.min() - ps_f.min()) / 100  # hPa difference in min ps

        if step <= 10 or step % 10 == 0 or (step > 100 and step % 100 == 0):
            print(f"{step:5d} {hour:6.1f} | "
                  f"{rel_vrt:10.3e} {rel_div:10.3e} {rel_T:10.3e} {rel_lnps:10.3e} | "
                  f"{ps_f.min()/100:10.2f} {ps_j.min()/100:10.2f} {dps:10.4f}")

        max_rel = max(rel_vrt, rel_div, rel_T, rel_lnps)
        if max_rel > 0.01 and step > 1:
            print(f"  *** ALERT: CS-corrected rel_err > 1% at step {step}! "
                  f"vrt={rel_vrt:.3e} div={rel_div:.3e} T={rel_T:.3e} lnps={rel_lnps:.3e}")

        if ps_j.min() / 100 < 900 or ps_j.max() / 100 > 1100:
            print(f"\n*** JAX BLOW-UP at step {step} (hour {hour:.1f})")
            break
        if ps_f.min() / 100 < 900 or ps_f.max() / 100 > 1100:
            print(f"\n*** FORTRAN BLOW-UP at step {step} (hour {hour:.1f})")
            break


if __name__ == '__main__':
    main()
