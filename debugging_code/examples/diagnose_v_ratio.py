"""
Diagnose the v-wind error structure: is it a constant scaling factor?
Also test the inverse vector transform directly by comparing the
spectral-to-grid reconstruction of u,v between Fortran and JAX
from the SAME spectral state.
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

from datetime import timedelta

import climt
import jax.numpy as jnp
import numpy as np
from sympl import set_constant

from gfs_dynamical_core import GFSDynamicalCore
from gfs_dynamical_core.component_jax import GFSDynamicsJAX
from gfs_dynamical_core.jax.states import GridState
from gfs_dynamical_core.jax.transforms import grid_to_spectral, spectral_to_grid

L = 64
N_LON = 2 * L - 1
N_LAT = L
N_LEV = 20

set_constant("reference_air_pressure", value=1e5, units="Pa")


def main():
    grid = climt.get_grid(nx=N_LON, ny=N_LAT, nz=N_LEV)
    timestep = timedelta(minutes=5)
    dcmip = climt.DcmipInitialConditions(add_perturbation=False)

    # Run both for 5 steps
    dycore_f = GFSDynamicalCore()
    state_f = climt.get_default_state([dycore_f], grid_state=grid)
    state_f.update(dcmip(state_f))

    dycore_j = GFSDynamicsJAX()
    state_j = climt.get_default_state([dycore_j], grid_state=grid)
    state_j.update(dcmip(state_j))

    for step in range(5):
        _, out_f = dycore_f(state_f, timestep=timestep)
        state_f.update(out_f)
        state_f["time"] += timestep

        _, out_j = dycore_j(state_j, timestep=timestep)
        state_j.update(out_j)
        state_j["time"] += timestep

    v_f = np.asarray(out_f["northward_wind"])
    v_j = np.asarray(out_j["northward_wind"])
    u_f = np.asarray(out_f["eastward_wind"])
    u_j = np.asarray(out_j["eastward_wind"])
    if np.iscomplexobj(v_j): v_j = v_j.real
    if np.iscomplexobj(u_j): u_j = u_j.real

    # ── 1. Ratio analysis: is v_jax/v_fortran constant? ─────────────────
    print("=" * 80)
    print("RATIO ANALYSIS: v_jax / v_fortran after 5 steps")
    print("=" * 80)

    # Only look where |v| is not too small (avoid division noise)
    threshold = np.abs(v_f).max() * 0.01
    mask = np.abs(v_f) > threshold

    if mask.sum() > 0:
        ratios = v_j[mask] / v_f[mask]
        print(f"  Points above threshold: {mask.sum()}")
        print(f"  Ratio mean:   {ratios.mean():.10f}")
        print(f"  Ratio std:    {ratios.std():.10f}")
        print(f"  Ratio min:    {ratios.min():.10f}")
        print(f"  Ratio max:    {ratios.max():.10f}")
        print(f"  (1 = perfect match, deviation from 1 = systematic scaling)")

    # Per-level ratio
    print(f"\n  Per-level ratio (zonal mean v_jax / v_fortran):")
    for k in range(N_LEV):
        zm_f = v_f[k].mean(axis=-1)  # zonal mean at each lat
        zm_j = v_j[k].mean(axis=-1)
        mask_k = np.abs(zm_f) > np.abs(zm_f).max() * 0.01
        if mask_k.sum() > 0:
            r = zm_j[mask_k] / zm_f[mask_k]
            print(f"    lev {k:2d}: ratio = {r.mean():.10f} ± {r.std():.2e}  (n={mask_k.sum()})")

    # ── 2. Error structure: zonal mean vs eddy ───────────────────────────
    print(f"\n{'='*80}")
    print("ERROR DECOMPOSITION: zonal mean vs eddy")
    print("=" * 80)

    v_diff = v_j - v_f
    # Zonal mean of the difference
    v_diff_zm = v_diff.mean(axis=-1)  # (lev, lat)
    # Eddy part of the difference
    v_diff_eddy = v_diff - v_diff_zm[:, :, None]

    print(f"  Total v diff max:      {np.abs(v_diff).max():.6e}")
    print(f"  Zonal mean diff max:   {np.abs(v_diff_zm).max():.6e}")
    print(f"  Eddy diff max:         {np.abs(v_diff_eddy).max():.6e}")
    print(f"  Fraction in zonal mean: {np.abs(v_diff_zm).max() / np.abs(v_diff).max():.4f}")

    # ── 3. Same analysis for u ───────────────────────────────────────────
    print(f"\n{'='*80}")
    print("U-WIND COMPARISON (for reference)")
    print("=" * 80)
    u_diff = u_j - u_f
    u_diff_zm = u_diff.mean(axis=-1)
    u_diff_eddy = u_diff - u_diff_zm[:, :, None]
    print(f"  Total u diff max:      {np.abs(u_diff).max():.6e}")
    print(f"  Zonal mean diff max:   {np.abs(u_diff_zm).max():.6e}")
    print(f"  Eddy diff max:         {np.abs(u_diff_eddy).max():.6e}")

    # ── 4. Spectral analysis of the v difference ─────────────────────────
    print(f"\n{'='*80}")
    print("SPECTRAL ANALYSIS of v difference")
    print("=" * 80)

    import s2fft
    # Convert the v difference to spectral space to see which modes are affected
    for k in [0, 5, 10, 15, 19]:
        v_diff_k = jnp.array(v_diff[k])
        v_diff_spec = s2fft.forward_jax(v_diff_k, L, sampling="gl")
        v_f_spec = s2fft.forward_jax(jnp.array(v_f[k]), L, sampling="gl")

        # Energy by total wavenumber l
        energy_diff = np.zeros(L)
        energy_f = np.zeros(L)
        v_diff_spec_np = np.array(v_diff_spec)
        v_f_spec_np = np.array(v_f_spec)
        for l in range(L):
            energy_diff[l] = np.sum(np.abs(v_diff_spec_np[l])**2)
            energy_f[l] = np.sum(np.abs(v_f_spec_np[l])**2)

        # Find dominant error modes
        top_l = np.argsort(energy_diff)[::-1][:5]
        print(f"\n  Level {k}: top 5 error modes (by l):")
        for l in top_l:
            ratio = energy_diff[l] / energy_f[l] if energy_f[l] > 0 else 0
            print(f"    l={l:3d}: err_energy={energy_diff[l]:.6e}  sig_energy={energy_f[l]:.6e}  ratio={ratio:.4f}")

        # m=0 vs m>0
        m0_err = np.abs(v_diff_spec_np[:, L-1])**2  # m=0 column
        m_nonzero_err = energy_diff - m0_err
        print(f"    m=0 total error:   {m0_err.sum():.6e}")
        print(f"    m!=0 total error:  {m_nonzero_err.sum():.6e}")

    # ── 5. Latitude profile of v difference ──────────────────────────────
    print(f"\n{'='*80}")
    print("LATITUDE PROFILE of zonal-mean v difference (level 13, peak error)")
    print("=" * 80)

    k = 13
    zm_vf = v_f[k].mean(axis=-1)
    zm_vj = v_j[k].mean(axis=-1)
    zm_diff = zm_vj - zm_vf

    lats = np.linspace(90, -90, N_LAT)  # approximate GL latitudes
    print(f"  {'lat':>6s}  {'v_F':>12s}  {'v_J':>12s}  {'diff':>12s}  {'ratio':>10s}")
    for i in range(0, N_LAT, 4):
        r = zm_vj[i] / zm_vf[i] if abs(zm_vf[i]) > 1e-10 else float('nan')
        print(f"  {lats[i]:6.1f}  {zm_vf[i]:12.6e}  {zm_vj[i]:12.6e}  {zm_diff[i]:12.6e}  {r:10.6f}")


if __name__ == "__main__":
    main()
