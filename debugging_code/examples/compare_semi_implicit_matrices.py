"""
Numerically compare the semi-implicit matrices between Fortran and JAX.

Since we can't easily extract Fortran's matrices directly, we:
1. Compute JAX matrices using init_semi_implicit_matrices
2. Recompute using the EXACT Fortran algorithm (top-to-bottom, 1-based indexing)
   with the same ak/bk that Fortran receives
3. Compare element-by-element

If these differ, the bug is in the matrix setup.
If they match, the bug is in how the matrices are used.
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

from datetime import timedelta

import climt
import jax.numpy as jnp
import numpy as np
from sympl import get_constant, set_constant

from gfs_dynamical_core import GFSDynamicalCore
from gfs_dynamical_core.component_jax import GFSDynamicsJAX

L = 64
N_LON = 2 * L - 1
N_LAT = L
N_LEV = 20
NTRUNC = 40

set_constant("reference_air_pressure", value=1e5, units="Pa")


def fortran_style_matrices(ak_ttb, bk_ttb, rd, kappa, rerth, dt, nlevs, ntrunc):
    """
    Compute semi-implicit matrices using the EXACT Fortran algorithm
    from semimp_data.f90, using top-to-bottom ak/bk.
    Returns matrices in Fortran's output ordering (already flipped to BTU).
    """
    ref_temp = 300.0
    ref_press = 800.0e2

    tref = np.full(nlevs, ref_temp)
    pkref = np.empty(nlevs + 1)
    for k in range(nlevs + 1):  # 0-based, but same as Fortran 1-based
        pkref[k] = ak_ttb[k] + bk_ttb[k] * ref_press
    dpkref = np.empty(nlevs)
    for k in range(nlevs):
        dpkref[k] = pkref[k + 1] - pkref[k]
    alfaref = np.empty(nlevs)
    alfaref[0] = np.log(2.0)  # top layer
    for k in range(1, nlevs):
        alfaref[k] = 1.0 - (pkref[k] / dpkref[k]) * np.log(pkref[k + 1] / pkref[k])

    # yecm (upper triangular + diagonal)
    yecm = np.zeros((nlevs, nlevs))
    for irow in range(nlevs):
        yecm[irow, irow] = alfaref[irow] * rd
        for icol in range(irow + 1, nlevs):
            yecm[irow, icol] = rd * np.log(pkref[icol + 1] / pkref[icol])

    # tecm (lower triangular + diagonal)
    tecm = np.zeros((nlevs, nlevs))
    for irow in range(nlevs):
        tecm[irow, irow] = kappa * tref[irow] * alfaref[irow]
        for icol in range(irow):
            tecm[irow, icol] = (
                kappa * tref[irow] * dpkref[icol] / dpkref[irow]
            ) * np.log(pkref[irow + 1] / pkref[irow])

    vecm = dpkref / ref_press

    # Flip to bottom-to-top (Fortran lines 230-235)
    amhyb = np.zeros((nlevs, nlevs))
    bmhyb = np.zeros((nlevs, nlevs))
    svhyb = np.zeros(nlevs)
    for j in range(nlevs):
        svhyb[j] = vecm[nlevs - 1 - j]
        for k in range(nlevs):
            amhyb[k, j] = yecm[nlevs - 1 - k, nlevs - 1 - j]
            bmhyb[k, j] = tecm[nlevs - 1 - k, nlevs - 1 - j]

    amhyb = amhyb / rerth**2
    tor_hyb = rd * tref / rerth**2

    # ym
    ym = np.outer(tor_hyb, svhyb) + amhyb @ bmhyb

    # d_hyb_m
    aa22, aa33, bb4 = 0.365, 0.1825, 0.35
    consts = [aa22, aa33, bb4]
    d_hyb_m = np.zeros((3, ntrunc + 1, nlevs, nlevs))
    rim = np.eye(nlevs)
    for stage_idx, const in enumerate(consts):
        for nn in range(ntrunc + 1):
            n = nn
            rnn1 = n * (n + 1.0)
            mat = rim + (const * dt) ** 2 * rnn1 * ym
            d_hyb_m[stage_idx, nn] = np.linalg.inv(mat)

    return dict(
        amhyb=amhyb, bmhyb=bmhyb, tor_hyb=tor_hyb,
        svhyb=svhyb, d_hyb_m=d_hyb_m,
        # Also return intermediates for debugging
        pkref=pkref, dpkref=dpkref, alfaref=alfaref,
        yecm=yecm, tecm=tecm, vecm=vecm, ym=ym,
    )


def main():
    grid = climt.get_grid(nx=N_LON, ny=N_LAT, nz=N_LEV)
    timestep = timedelta(minutes=5)
    dt = timestep.total_seconds()
    dcmip = climt.DcmipInitialConditions(add_perturbation=False)

    # Initialize JAX dycore to get its matrices
    dycore_j = GFSDynamicsJAX()
    state_j = climt.get_default_state([dycore_j], grid_state=grid)
    state_j.update(dcmip(state_j))
    _, _ = dycore_j(state_j, timestep=timestep)

    sc = dycore_j.stepper_config
    dc = dycore_j.dyn_config

    # Get the ak/bk that Fortran would receive (reversed from climt's BTU)
    ak_btu = np.array(state_j["atmosphere_hybrid_sigma_pressure_a_coordinate_on_interface_levels"])
    bk_btu = np.array(state_j["atmosphere_hybrid_sigma_pressure_b_coordinate_on_interface_levels"])
    ak_ttb = ak_btu[::-1]  # This is what component.py line 273-274 passes to Fortran
    bk_ttb = bk_btu[::-1]

    rd = float(dc.rd)
    kappa = float(dc.rk)
    rerth = float(dc.radius)

    print(f"ak_btu[0] (surface) = {ak_btu[0]:.6f}, bk_btu[0] = {bk_btu[0]:.6f}")
    print(f"ak_btu[-1] (TOA)    = {ak_btu[-1]:.6f}, bk_btu[-1] = {bk_btu[-1]:.6f}")
    print(f"ak_ttb[0] (TOA)     = {ak_ttb[0]:.6f}, bk_ttb[0] = {bk_ttb[0]:.6f}")
    print(f"ak_ttb[-1] (surface)= {ak_ttb[-1]:.6f}, bk_ttb[-1] = {bk_ttb[-1]:.6f}")
    print(f"rd = {rd}, kappa = {kappa}, rerth = {rerth}, dt = {dt}")

    # Compute "Fortran-style" matrices
    f_mats = fortran_style_matrices(ak_ttb, bk_ttb, rd, kappa, rerth, dt, N_LEV, NTRUNC)

    # Compare with JAX matrices
    print(f"\n{'='*80}")
    print("MATRIX COMPARISON")
    print(f"{'='*80}")

    for name in ["amhyb", "bmhyb", "tor_hyb", "svhyb"]:
        f_arr = f_mats[name]
        j_arr = np.array(getattr(sc, name))
        diff = np.abs(f_arr - j_arr)
        max_f = np.abs(f_arr).max()
        max_diff = diff.max()
        rel = max_diff / max_f if max_f > 0 else 0
        print(f"  {name:10s}: max|F|={max_f:.6e}  max|diff|={max_diff:.6e}  rel={rel:.6e}")

    # d_hyb_m comparison (JAX has shape (3, L, nlev, nlev), Fortran-style has (3, ntrunc+1, nlev, nlev))
    # JAX's d_hyb_m covers l=0..L-1, Fortran's covers n=0..ntrunc
    j_dhm = np.array(sc.d_hyb_m)
    f_dhm = f_mats["d_hyb_m"]
    for stage in range(3):
        for nn in range(min(NTRUNC + 1, L)):
            diff = np.abs(j_dhm[stage, nn] - f_dhm[stage, nn]).max()
            if diff > 1e-14:
                rel = diff / np.abs(f_dhm[stage, nn]).max()
                print(f"  d_hyb_m[stage={stage}, l={nn}]: max|diff|={diff:.6e}  rel={rel:.6e}")
    # Summary
    for stage in range(3):
        max_diff = 0
        for nn in range(min(NTRUNC + 1, L)):
            d = np.abs(j_dhm[stage, nn] - f_dhm[stage, nn]).max()
            max_diff = max(max_diff, d)
        print(f"  d_hyb_m stage {stage}: max|diff| across all l = {max_diff:.6e}")

    # Also compare the intermediate quantities
    print(f"\n--- Intermediate quantities ---")

    # Recompute JAX's intermediates manually from its ak/bk
    from gfs_dynamical_core.jax.stepper import _REF_TEMP, _REF_PRESS
    ak_jax_ttb = np.array(dc.ak)[::-1]  # JAX stores BTU, flip to TTB
    bk_jax_ttb = np.array(dc.bk)[::-1]

    print(f"  ak_ttb match: {np.allclose(ak_ttb, ak_jax_ttb)}")
    print(f"  bk_ttb match: {np.allclose(bk_ttb, bk_jax_ttb)}")
    print(f"  ak_ttb diff:  {np.abs(ak_ttb - ak_jax_ttb).max():.6e}")
    print(f"  bk_ttb diff:  {np.abs(bk_ttb - bk_jax_ttb).max():.6e}")

    # pkref comparison
    j_mats = fortran_style_matrices(ak_jax_ttb, bk_jax_ttb, rd, kappa, rerth, dt, N_LEV, NTRUNC)
    print(f"  pkref diff:   {np.abs(f_mats['pkref'] - j_mats['pkref']).max():.6e}")
    print(f"  dpkref diff:  {np.abs(f_mats['dpkref'] - j_mats['dpkref']).max():.6e}")
    print(f"  alfaref diff: {np.abs(f_mats['alfaref'] - j_mats['alfaref']).max():.6e}")
    print(f"  yecm diff:    {np.abs(f_mats['yecm'] - j_mats['yecm']).max():.6e}")
    print(f"  tecm diff:    {np.abs(f_mats['tecm'] - j_mats['tecm']).max():.6e}")
    print(f"  ym diff:      {np.abs(f_mats['ym'] - j_mats['ym']).max():.6e}")


if __name__ == "__main__":
    main()
