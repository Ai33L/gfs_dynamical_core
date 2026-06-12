"""
Per-level error analysis: which vertical levels diverge first?

Re-runs both dycores side-by-side, comparing spectral state per level
at regular intervals. Prints a heatmap-style table of relative errors
by level and timestep.
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

# Steps at which to print per-level breakdown
REPORT_STEPS = [1, 10, 50, 100, 500, 1000, 1200, 1400, 1500, 1600,
                1700, 1800, 1900, 2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400]

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


def per_level_rel_err(f_packed, j_packed, cs_corr):
    """Return per-level relative error (CS-corrected). Shape: (nlevs,)"""
    j_corr = j_packed * cs_corr[:, None]
    diff = np.abs(f_packed - j_corr)
    fmax = np.maximum(np.abs(f_packed).max(axis=0), 1e-30)  # per-level max
    return diff.max(axis=0) / fmax


def report_per_level(step, hour, fdata, jspec, cs):
    """Print per-level error breakdown for vrt, div, T."""
    j_vrt = s2fft_to_packed_3d(np.array(jspec.vorticity), NTRUNC)
    j_div = s2fft_to_packed_3d(np.array(jspec.divergence), NTRUNC)
    j_temp = s2fft_to_packed_3d(np.array(jspec.temperature), NTRUNC)

    vrt_err = per_level_rel_err(fdata['vrtspec'], j_vrt, cs)
    div_err = per_level_rel_err(fdata['divspec'], j_div, cs)
    T_err = per_level_rel_err(fdata['virtempspec'], j_temp, cs)

    print(f"\n=== Step {step} (hour {hour:.1f}, day {hour/24:.2f}) ===")
    print(f"{'lev':>4} | {'vrt_rel':>10} {'div_rel':>10} {'T_rel':>10} | {'max_field':>10}")
    print("-" * 60)

    # Also track which level has max error for each field
    worst_vrt = np.argmax(vrt_err)
    worst_div = np.argmax(div_err)
    worst_T = np.argmax(T_err)

    for k in range(N_LEV):
        marker = ""
        if k == worst_vrt:
            marker += " <-vrt"
        if k == worst_div:
            marker += " <-div"
        if k == worst_T:
            marker += " <-T"
        # max|F| across all spectral coefficients at this level (vrt as reference)
        maxf = np.abs(fdata['vrtspec'][:, k]).max()
        print(f"{k:4d} | {vrt_err[k]:10.3e} {div_err[k]:10.3e} {T_err[k]:10.3e} | {maxf:10.3e}{marker}")

    print(f"  worst vrt: lev {worst_vrt} ({vrt_err[worst_vrt]:.3e})")
    print(f"  worst div: lev {worst_div} ({div_err[worst_div]:.3e})")
    print(f"  worst T:   lev {worst_T} ({T_err[worst_T]:.3e})")

    return vrt_err, div_err, T_err


def main():
    cs = build_cs_packed(NTRUNC)

    print(f"Grid: {N_LON}x{N_LAT}x{N_LEV}, L={L}, ntrunc={NTRUNC}")
    print(f"Running {N_STEPS} steps ({N_STEPS * DT_MIN / 60:.1f} hours)")
    print(f"Reporting per-level errors at steps: {REPORT_STEPS}\n")

    dycore_f = GFSDynamicalCore()
    state_f = climt.get_default_state([dycore_f], grid_state=grid)
    state_f.update(dcmip(state_f))

    dycore_j = GFSDynamicsJAX()
    state_j = climt.get_default_state([dycore_j], grid_state=grid)
    state_j.update(dcmip(state_j))

    # Compact progress line for non-report steps
    print(f"{'step':>5} {'hour':>6} | {'vrt':>10} {'div':>10} {'T':>10} | "
          f"{'ps_F':>10} {'ps_J':>10} {'dps':>10}")
    print("-" * 90)

    all_results = {}

    for step in range(1, N_STEPS + 1):
        diag_f, out_f = dycore_f(state_f, timestep=dt)
        state_f.update(out_f)
        state_f['time'] += dt

        diag_j, out_j = dycore_j(state_j, timestep=dt)
        state_j.update(out_j)
        state_j['time'] += dt

        fdata = read_fortran_dump(step)
        jspec = dycore_j._spec_state
        if fdata is None or jspec is None:
            continue

        # Quick aggregate check
        j_vrt = s2fft_to_packed_3d(np.array(jspec.vorticity), NTRUNC)
        j_temp = s2fft_to_packed_3d(np.array(jspec.temperature), NTRUNC)
        j_vrt_corr = j_vrt * cs[:, None]
        j_temp_corr = j_temp * cs[:, None]

        maxf_v = max(np.abs(fdata['vrtspec']).max(), 1e-30)
        maxf_d = max(np.abs(fdata['divspec']).max(), 1e-30)
        maxf_t = max(np.abs(fdata['virtempspec']).max(), 1e-30)
        rel_vrt = np.abs(fdata['vrtspec'] - j_vrt_corr).max() / maxf_v
        j_div = s2fft_to_packed_3d(np.array(jspec.divergence), NTRUNC)
        j_div_corr = j_div * cs[:, None]
        rel_div = np.abs(fdata['divspec'] - j_div_corr).max() / maxf_d
        rel_T = np.abs(fdata['virtempspec'] - j_temp_corr).max() / maxf_t

        ps_f = np.asarray(state_f['surface_air_pressure'])
        ps_j = np.asarray(state_j['surface_air_pressure'])
        if np.iscomplexobj(ps_j):
            ps_j = ps_j.real
        hour = step * DT_MIN / 60.0
        dps = (ps_j.min() - ps_f.min()) / 100

        # Print compact line every 100 steps (or early on)
        if step <= 10 or step % 100 == 0:
            print(f"{step:5d} {hour:6.1f} | {rel_vrt:10.3e} {rel_div:10.3e} {rel_T:10.3e} | "
                  f"{ps_f.min()/100:10.2f} {ps_j.min()/100:10.2f} {dps:10.4f}")

        # Per-level report at selected steps
        if step in REPORT_STEPS:
            vrt_err, div_err, T_err = report_per_level(step, hour, fdata, jspec, cs)
            all_results[step] = (vrt_err, div_err, T_err)

        # Blow-up detection
        if ps_j.min() / 100 < 900 or ps_j.max() / 100 > 1100:
            print(f"\n*** JAX BLOW-UP at step {step} (hour {hour:.1f})")
            # Do a final per-level report
            if step not in all_results:
                report_per_level(step, hour, fdata, jspec, cs)
            break
        if ps_f.min() / 100 < 900 or ps_f.max() / 100 > 1100:
            print(f"\n*** FORTRAN BLOW-UP at step {step} (hour {hour:.1f})")
            break

    # Summary: for each report step, show which level had worst vrt error
    print("\n\n=== SUMMARY: worst level per field over time ===")
    print(f"{'step':>5} {'hour':>6} | {'worst_vrt_lev':>14} {'worst_vrt_err':>14} | "
          f"{'worst_div_lev':>14} {'worst_div_err':>14} | "
          f"{'worst_T_lev':>12} {'worst_T_err':>12}")
    print("-" * 110)
    for step in sorted(all_results.keys()):
        vrt_err, div_err, T_err = all_results[step]
        hour = step * DT_MIN / 60.0
        wv = np.argmax(vrt_err)
        wd = np.argmax(div_err)
        wt = np.argmax(T_err)
        print(f"{step:5d} {hour:6.1f} | {wv:14d} {vrt_err[wv]:14.3e} | "
              f"{wd:14d} {div_err[wd]:14.3e} | "
              f"{wt:12d} {T_err[wt]:12.3e}")


if __name__ == '__main__':
    main()
