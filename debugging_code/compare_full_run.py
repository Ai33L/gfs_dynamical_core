"""
Full-run comparison: Fortran vs JAX spectral state & tendencies at every step.
Reads Fortran binary dumps (one per step), runs JAX side-by-side,
and reports when/where the codes diverge.
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

# ── Config ──
L = 64
N_LON = 2 * L - 1
N_LAT = L
N_LEV = 20
N_STEPS = 2500  # ~8.7 days at 5 min dt
DT_MIN = 5
NTRUNC = 40

set_constant("reference_air_pressure", value=1e5, units="Pa")
grid = climt.get_grid(nx=N_LON, ny=N_LAT, nz=N_LEV)
dcmip = climt.DcmipInitialConditions(add_perturbation=True)
dt = timedelta(minutes=DT_MIN)


def packed_to_s2fft(packed, ntrunc, L):
    """Convert SHTNS m-first packed (ndimspec,) to s2fft (L, 2L-1).
    Only fills m>=0 side. Caller must add conjugate symmetry if needed."""
    out = np.zeros((L, 2 * L - 1), dtype=packed.dtype)
    idx = 0
    for m in range(ntrunc + 1):
        for l in range(m, ntrunc + 1):
            out[l, L - 1 + m] = packed[idx]
            idx += 1
    return out


def packed_to_s2fft_3d(packed_2d, ntrunc, L):
    """Convert (ndimspec, nlevs) packed to (nlevs, L, 2L-1)."""
    ndimspec, nlevs = packed_2d.shape
    out = np.zeros((nlevs, L, 2 * L - 1), dtype=packed_2d.dtype)
    idx = 0
    for m in range(ntrunc + 1):
        for l in range(m, ntrunc + 1):
            out[:, l, L - 1 + m] = packed_2d[idx, :]
            idx += 1
    return out


def s2fft_to_packed(arr_2d, ntrunc):
    """Convert s2fft (L, 2L-1) to SHTNS packed (ndimspec,)."""
    L_loc = arr_2d.shape[0]
    ndimspec = (ntrunc + 1) * (ntrunc + 2) // 2
    out = np.zeros(ndimspec, dtype=arr_2d.dtype)
    idx = 0
    for m in range(ntrunc + 1):
        for l in range(m, ntrunc + 1):
            out[idx] = arr_2d[l, L_loc - 1 + m]
            idx += 1
    return out


def s2fft_to_packed_3d(arr_3d, ntrunc):
    """Convert (nlevs, L, 2L-1) to (ndimspec, nlevs)."""
    nlevs, L_loc, _ = arr_3d.shape
    ndimspec = (ntrunc + 1) * (ntrunc + 2) // 2
    out = np.zeros((ndimspec, nlevs), dtype=arr_3d.dtype)
    idx = 0
    for m in range(ntrunc + 1):
        for l in range(m, ntrunc + 1):
            out[idx, :] = arr_3d[:, l, L_loc - 1 + m]
            idx += 1
    return out


def read_fortran_dump(step):
    """Read Fortran binary dump for a given step."""
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
        data['dvrtdt'] = read_c2d()
        data['ddivdt'] = read_c2d()
        data['dtempdt'] = read_c2d()
        data['dlnpsdt'] = read_c1d()
        data['ndimspec'] = ndimspec
        data['nlevs'] = nlevs
    return data


def spectral_energy(packed_2d, ntrunc):
    """Compute spectral energy (sum |f_lm|^2) per level."""
    ndimspec, nlevs = packed_2d.shape
    return np.sum(np.abs(packed_2d) ** 2, axis=0)


def compare_packed(name, f_packed, j_packed, step):
    """Compare two packed arrays. Returns max relative error."""
    diff = np.abs(f_packed - j_packed)
    maxf = max(np.abs(f_packed).max(), 1e-30)
    maxd = diff.max()
    rel = maxd / maxf
    return rel


def main():
    print(f"Grid: {N_LON}x{N_LAT}x{N_LEV}, L={L}, ntrunc={NTRUNC}")
    print(f"Running {N_STEPS} steps ({N_STEPS * DT_MIN / 60:.1f} hours)")
    print()

    # ── Init Fortran ──
    dycore_f = GFSDynamicalCore()
    state_f = climt.get_default_state([dycore_f], grid_state=grid)
    state_f.update(dcmip(state_f))

    # ── Init JAX ──
    dycore_j = GFSDynamicsJAX()
    state_j = climt.get_default_state([dycore_j], grid_state=grid)
    state_j.update(dcmip(state_j))

    # Header
    print(f"{'step':>5} {'hour':>6} | {'vrt':>10} {'div':>10} {'T':>10} {'lnps':>10} | "
          f"{'dvrt_dt':>10} {'ddiv_dt':>10} {'dT_dt':>10} {'dlnps_dt':>10} | "
          f"{'ps_F':>10} {'ps_J':>10}")
    print("-" * 140)

    for step in range(1, N_STEPS + 1):
        # Step Fortran
        diag_f, out_f = dycore_f(state_f, timestep=dt)
        state_f.update(out_f)
        state_f['time'] += dt

        # Step JAX
        diag_j, out_j = dycore_j(state_j, timestep=dt)
        state_j.update(out_j)
        state_j['time'] += dt

        # Read Fortran dump
        fdata = read_fortran_dump(step)
        if fdata is None:
            print(f"{step:5d}: Fortran dump not found, skipping comparison")
            continue

        # Get JAX spectral state
        jspec = dycore_j._spec_state
        if jspec is None:
            continue

        # Convert JAX to packed format for comparison
        j_vrt = s2fft_to_packed_3d(np.array(jspec.vorticity), NTRUNC)
        j_div = s2fft_to_packed_3d(np.array(jspec.divergence), NTRUNC)
        j_temp = s2fft_to_packed_3d(np.array(jspec.temperature), NTRUNC)
        j_lnps = s2fft_to_packed(np.array(jspec.log_surface_pressure), NTRUNC)

        # Compare spectral state
        rel_vrt = compare_packed("vrt", fdata['vrtspec'], j_vrt, step)
        rel_div = compare_packed("div", fdata['divspec'], j_div, step)
        rel_T = compare_packed("T", fdata['virtempspec'], j_temp, step)
        rel_lnps = compare_packed("lnps", fdata['lnpsspec'], j_lnps, step)

        # Compare tendencies
        # For JAX tendencies, we'd need to instrument the JAX stepper too.
        # For now, compare just the Fortran tendency magnitudes.
        rel_dvrt = np.abs(fdata['dvrtdt']).max()
        rel_ddiv = np.abs(fdata['ddivdt']).max()
        rel_dT = np.abs(fdata['dtempdt']).max()
        rel_dlnps = np.abs(fdata['dlnpsdt']).max()

        # Surface pressure range
        ps_f = np.asarray(state_f['surface_air_pressure'])
        ps_j = np.asarray(state_j['surface_air_pressure'])
        if np.iscomplexobj(ps_j):
            ps_j = ps_j.real

        hour = step * DT_MIN / 60.0

        # Print at logarithmic intervals: every step for first 10,
        # then every 10, then every 100
        if step <= 10 or step % 10 == 0 or (step > 100 and step % 100 == 0):
            print(f"{step:5d} {hour:6.1f} | "
                  f"{rel_vrt:10.3e} {rel_div:10.3e} {rel_T:10.3e} {rel_lnps:10.3e} | "
                  f"{rel_dvrt:10.3e} {rel_ddiv:10.3e} {rel_dT:10.3e} {rel_dlnps:10.3e} | "
                  f"{ps_f.min()/100:10.2f} {ps_j.min()/100:10.2f}")

        # Alert if relative error exceeds threshold
        max_rel = max(rel_vrt, rel_div, rel_T, rel_lnps)
        if max_rel > 0.01 and step > 1:
            print(f"  *** ALERT: rel_err > 1% at step {step}! "
                  f"vrt={rel_vrt:.3e} div={rel_div:.3e} T={rel_T:.3e} lnps={rel_lnps:.3e}")

        # Stop if blow-up detected
        if ps_j.min() / 100 < 900 or ps_j.max() / 100 > 1100:
            print(f"\n*** JAX BLOW-UP at step {step} (hour {hour:.1f})")
            break
        if ps_f.min() / 100 < 900 or ps_f.max() / 100 > 1100:
            print(f"\n*** FORTRAN BLOW-UP at step {step} (hour {hour:.1f})")
            break


if __name__ == '__main__':
    main()
