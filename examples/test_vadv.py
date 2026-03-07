"""
Concrete numerical test of compute_vertical_advection against the Fortran getvadv.

We set up a tiny 3-level, 1-lat, 1-lon problem so every index can be checked by hand.

Fortran getvadv conventions:
  - datag(nlons, nlats, nlevs)  : bottom-to-top  (k=1 surface, k=nlevs TOA)
  - etadot(nlons, nlats, nlevs+1): TOP-to-bottom   (k=1 TOA boundary, k=nlevs+1 surface boundary)
  - dpk(nlons, nlats, nlevs)    : TOP-to-bottom   (k=1 TOA layer, k=nlevs surface layer)
  - vadv(nlons, nlats, nlevs)   : bottom-to-top

JAX compute_vertical_advection conventions:
  - data(n_lev, n_lat, n_lon)   : bottom-to-top  (k=0 surface, k=-1 TOA)
  - etadot(n_lev+1, n_lat, n_lon): bottom-to-top (k=0 surface boundary, k=-1 TOA boundary)
  - dp(n_lev, n_lat, n_lon)     : bottom-to-top  (k=0 surface layer, k=-1 TOA layer)
"""

import numpy as np


def fortran_getvadv(datag, etadot, dpk, nlevs):
    """
    Direct Python translation of the Fortran getvadv subroutine.

    Uses FORTRAN 1-BASED indexing internally (with helper lambdas) to avoid
    off-by-one errors.  All input/output arrays are 0-based Python arrays.

    Input conventions (matching Fortran):
      datag[k]  : k=0..nlevs-1, bottom-to-top (0=surface, nlevs-1=TOA)
      etadot[k] : k=0..nlevs,   top-to-bottom (0=TOA boundary, nlevs=surface boundary)
      dpk[k]    : k=0..nlevs-1, top-to-bottom (0=TOA layer, nlevs-1=surface layer)

    Returns vadv[k] : k=0..nlevs-1, bottom-to-top
    """
    vadv = np.zeros(nlevs)

    # Fortran 1-based accessors (F1(1) == Python [0], etc.)
    F_datag = lambda i: datag[i - 1]  # 1-based BTU
    F_etadot = lambda i: etadot[i - 1]  # 1-based TTB
    F_dpk = lambda i: dpk[i - 1]  # 1-based TTB

    def F_vadv_set(i, val):
        vadv[i - 1] = val

    # vadv(:,:,nlevs) = (0.5/dpk(:,:,1))*etadot(:,:,2)*(datag(:,:,nlevs-1)-datag(:,:,nlevs))
    F_vadv_set(
        nlevs,
        (0.5 / F_dpk(1)) * F_etadot(2) * (F_datag(nlevs - 1) - F_datag(nlevs)),
    )

    # vadv(:,:,1) = (0.5/dpk(:,:,nlevs))*etadot(:,:,nlevs)*(datag(:,:,1)-datag(:,:,2))
    F_vadv_set(
        1,
        (0.5 / F_dpk(nlevs)) * F_etadot(nlevs) * (F_datag(1) - F_datag(2)),
    )

    # do k=2,nlevs-1
    for k in range(2, nlevs):  # k is Fortran 1-based
        # vadv(:,:,nlevs+1-k) = (0.5/dpk(:,:,k)) * (
        #   etadot(:,:,k+1)*(datag(:,:,nlevs-k) - datag(:,:,nlevs+1-k)) +
        #   etadot(:,:,k)  *(datag(:,:,nlevs+1-k) - datag(:,:,nlevs+2-k)))
        F_vadv_set(
            nlevs + 1 - k,
            (0.5 / F_dpk(k))
            * (
                F_etadot(k + 1) * (F_datag(nlevs - k) - F_datag(nlevs + 1 - k))
                + F_etadot(k) * (F_datag(nlevs + 1 - k) - F_datag(nlevs + 2 - k))
            ),
        )

    return vadv


def jax_compute_vertical_advection_ORIGINAL(data, etadot, dp):
    """
    ORIGINAL JAX code (before any fix).
    data: (n_lev,) BTU,  etadot: (n_lev+1,) BTU,  dp: (n_lev,) BTU
    """
    vadv_bot = (0.5 / dp[0]) * etadot[1] * (data[1] - data[0])
    vadv_top = (0.5 / dp[-1]) * etadot[-2] * (data[-1] - data[-2])

    vadv_mid = (0.5 / dp[1:-1]) * (
        etadot[2:-1] * (data[2:] - data[1:-1]) + etadot[1:-2] * (data[1:-1] - data[:-2])
    )
    return np.concatenate([[vadv_bot], vadv_mid, [vadv_top]])


def jax_compute_vertical_advection_SIGNFIX(data, etadot, dp):
    """
    With the boundary sign fix only (not the middle layers).
    """
    vadv_bot = (0.5 / dp[0]) * etadot[1] * (data[0] - data[1])
    vadv_top = (0.5 / dp[-1]) * etadot[-2] * (data[-2] - data[-1])

    vadv_mid = (0.5 / dp[1:-1]) * (
        etadot[2:-1] * (data[2:] - data[1:-1]) + etadot[1:-2] * (data[1:-1] - data[:-2])
    )
    return np.concatenate([[vadv_bot], vadv_mid, [vadv_top]])


def jax_compute_vertical_advection_CORRECTED(data, etadot, dp):
    """
    Corrected: swap etadot-to-layer pairing and data difference directions.

    Fortran all-BTU formula (layer j, 0-based):
      vadv[j] = (0.5/dp[j]) * (
          etadot[j]   * (data[j-1] - data[j])
        + etadot[j+1] * (data[j] - data[j+1]))
    """
    vadv_bot = (0.5 / dp[0]) * etadot[1] * (data[0] - data[1])
    vadv_top = (0.5 / dp[-1]) * etadot[-2] * (data[-2] - data[-1])

    vadv_mid = (0.5 / dp[1:-1]) * (
        etadot[1:-2] * (data[:-2] - data[1:-1]) + etadot[2:-1] * (data[1:-1] - data[2:])
    )
    return np.concatenate([[vadv_bot], vadv_mid, [vadv_top]])


def convert_etadot_ttb_to_btu(etadot_ttb):
    """Flip etadot from top-to-bottom to bottom-to-top."""
    return etadot_ttb[::-1]


def convert_dpk_ttb_to_btu(dpk_ttb):
    """Flip dpk from top-to-bottom to bottom-to-top."""
    return dpk_ttb[::-1]


def main():
    nlevs = 5

    # datag: BTU  (k=0 surface, k=nlevs-1 TOA)
    datag = np.array([300.0, 290.0, 270.0, 240.0, 200.0])

    # dpk: TTB  (k=0 TOA layer, k=nlevs-1 surface layer)
    dpk_ttb = np.array([50.0, 100.0, 150.0, 200.0, 250.0])

    # etadot: TTB (k=0 TOA boundary=0, k=nlevs surface boundary=0)
    # Boundaries are zero, interior interfaces have some values
    etadot_ttb = np.array([0.0, -0.02, -0.05, -0.03, -0.01, 0.0])

    print("=" * 70)
    print("INPUT DATA")
    print("=" * 70)
    print(f"nlevs = {nlevs}")
    print(f"datag (BTU, surface->TOA) = {datag}")
    print(f"dpk   (TTB, TOA->surface) = {dpk_ttb}")
    print(f"etadot(TTB, TOA->surface) = {etadot_ttb}")
    print()

    # --- Fortran reference ---
    vadv_fortran = fortran_getvadv(datag, etadot_ttb, dpk_ttb, nlevs)

    # --- Convert to JAX BTU conventions ---
    etadot_btu = convert_etadot_ttb_to_btu(etadot_ttb)
    dp_btu = convert_dpk_ttb_to_btu(dpk_ttb)

    print(f"etadot(BTU, surface->TOA) = {etadot_btu}")
    print(f"dp    (BTU, surface->TOA) = {dp_btu}")
    print()

    # --- JAX original ---
    vadv_jax_orig = jax_compute_vertical_advection_ORIGINAL(datag, etadot_btu, dp_btu)

    # --- JAX sign fix (boundary only) ---
    vadv_jax_signfix = jax_compute_vertical_advection_SIGNFIX(datag, etadot_btu, dp_btu)

    # --- JAX corrected (full fix: swap etadot-layer pairing + data diff direction) ---
    vadv_jax_corrected = jax_compute_vertical_advection_CORRECTED(
        datag, etadot_btu, dp_btu
    )

    print("=" * 70)
    print("RESULTS (all BTU: k=0 surface, k=nlevs-1 TOA)")
    print("=" * 70)
    print(
        f"{'Level':>6s}  {'Fortran':>12s}  {'JAX orig':>12s}  {'JAX bndfix':>12s}  {'JAX corrected':>14s}  {'corr err':>12s}"
    )
    print("-" * 80)
    for k in range(nlevs):
        label = "sfc" if k == 0 else ("TOA" if k == nlevs - 1 else f"mid-{k}")
        err_corr = vadv_jax_corrected[k] - vadv_fortran[k]
        print(
            f"{label:>6s}  {vadv_fortran[k]:12.6e}  {vadv_jax_orig[k]:12.6e}  "
            f"{vadv_jax_signfix[k]:12.6e}  {vadv_jax_corrected[k]:14.6e}  {err_corr:12.6e}"
        )

    print()
    max_err_orig = np.max(np.abs(vadv_jax_orig - vadv_fortran))
    max_err_signfix = np.max(np.abs(vadv_jax_signfix - vadv_fortran))
    max_err_corrected = np.max(np.abs(vadv_jax_corrected - vadv_fortran))
    print(f"Max abs error (original):   {max_err_orig:.6e}")
    print(f"Max abs error (bnd fix):    {max_err_signfix:.6e}")
    print(f"Max abs error (corrected):  {max_err_corrected:.6e}")

    if max_err_corrected < 1e-15:
        print("\n>>> CORRECTED JAX code MATCHES Fortran <<<")
    elif max_err_orig < 1e-15:
        print("\n>>> ORIGINAL JAX code MATCHES Fortran <<<")
    else:
        print("\n>>> No version matches Fortran exactly <<<")

    # ---- Detailed per-level trace of Fortran getvadv ----
    print()
    print("=" * 70)
    print("DETAILED FORTRAN TRACE (1-based Fortran indices)")
    print("=" * 70)
    print()

    # Fortran TOA layer: vadv(nlevs)
    print(f"vadv(nlevs={nlevs}) [TOA in BTU]:")
    print(f"  = (0.5/dpk(1)) * etadot(2) * (datag(nlevs-1) - datag(nlevs))")
    print(
        f"  = (0.5/{dpk_ttb[0]}) * {etadot_ttb[1]} * ({datag[nlevs - 2]} - {datag[nlevs - 1]})"
    )
    print(f"  = {vadv_fortran[nlevs - 1]}")
    print()

    # Fortran surface layer: vadv(1)
    print(f"vadv(1) [surface in BTU]:")
    print(f"  = (0.5/dpk(nlevs)) * etadot(nlevs) * (datag(1) - datag(2))")
    print(
        f"  = (0.5/{dpk_ttb[nlevs - 1]}) * {etadot_ttb[nlevs - 1]} * ({datag[0]} - {datag[1]})"
    )
    print(f"  = {vadv_fortran[0]}")
    print()

    # Fortran middle layers
    for k in range(2, nlevs):
        j_btu = nlevs + 1 - k  # 1-based BTU index
        print(f"vadv({j_btu}) [middle, Fortran TTB k={k}]:")
        print(f"  = (0.5/dpk({k})) * (")
        print(f"      etadot({k + 1}) * (datag({nlevs - k}) - datag({nlevs + 1 - k}))")
        print(
            f"    + etadot({k})   * (datag({nlevs + 1 - k}) - datag({nlevs + 2 - k})))"
        )
        e1 = etadot_ttb[k]  # etadot(k+1) 1-based -> [k] 0-based
        e2 = etadot_ttb[k - 1]  # etadot(k)   1-based -> [k-1] 0-based
        d1a = datag[nlevs - k - 1]  # datag(nlevs-k) 1-based -> [nlevs-k-1] 0-based
        d1b = datag[nlevs - k]  # datag(nlevs+1-k) 1-based -> [nlevs-k] 0-based
        d2b = datag[
            nlevs - k + 1
        ]  # datag(nlevs+2-k) 1-based -> [nlevs-k+1] 0-based... wait
        # Actually: datag(nlevs+2-k) 1-based -> [nlevs+2-k-1] = [nlevs+1-k] 0-based
        d2b_correct = datag[nlevs + 1 - k]
        print(f"  values: dpk={dpk_ttb[k - 1]}, etadot({k + 1})={e1}, etadot({k})={e2}")
        print(
            f"          datag({nlevs - k})={d1a}, datag({nlevs + 1 - k})={d1b}, datag({nlevs + 2 - k})={d2b_correct}"
        )
        print(f"  = {vadv_fortran[j_btu - 1]}")
        print()

    # ---- Now trace JAX computation ----
    print("=" * 70)
    print("DETAILED JAX TRACE (0-based BTU indices)")
    print("=" * 70)
    print()
    print("etadot_btu:", etadot_btu)
    print("dp_btu:    ", dp_btu)
    print("datag:     ", datag)
    print()

    for k in range(nlevs):
        label = "sfc" if k == 0 else ("TOA" if k == nlevs - 1 else f"mid-{k}")
        print(f"JAX vadv[{k}] ({label}):")
        if k == 0:
            print(f"  = (0.5/dp[0]) * etadot[1] * (data[1] - data[0])")
            print(
                f"  = (0.5/{dp_btu[0]}) * {etadot_btu[1]} * ({datag[1]} - {datag[0]})"
            )
        elif k == nlevs - 1:
            print(f"  = (0.5/dp[-1]) * etadot[-2] * (data[-1] - data[-2])")
            print(
                f"  = (0.5/{dp_btu[-1]}) * {etadot_btu[-2]} * ({datag[-1]} - {datag[-2]})"
            )
        else:
            print(
                f"  = (0.5/dp[{k}]) * (etadot[{k + 1}]*(data[{k + 1}]-data[{k}]) + etadot[{k}]*(data[{k}]-data[{k - 1}]))"
            )
            print(
                f"  = (0.5/{dp_btu[k]}) * ({etadot_btu[k + 1]}*({datag[k + 1]}-{datag[k]}) + {etadot_btu[k]}*({datag[k]}-{datag[k - 1]}))"
            )
        print(f"  = {vadv_jax_orig[k]}")
        print()

    # --- Global sign analysis ---
    print("=" * 70)
    print("SIGN ANALYSIS")
    print("=" * 70)
    ratio = np.where(np.abs(vadv_fortran) > 1e-20, vadv_jax_orig / vadv_fortran, 0.0)
    for k in range(nlevs):
        label = "sfc" if k == 0 else ("TOA" if k == nlevs - 1 else f"mid-{k}")
        print(f"  {label}: JAX/Fortran = {ratio[k]:.4f}")

    print()
    if np.allclose(ratio[np.abs(vadv_fortran) > 1e-20], -1.0):
        print(">>> ALL levels have ratio = -1.0 => GLOBAL SIGN FLIP <<<")
        print(">>> The etadot used by JAX vadv likely has opposite sign to Fortran <<<")
    else:
        print(f">>> Ratios are not all -1.  Investigate further. <<<")

    # ---- Now also test whether the middle formula, extended to boundaries
    #      with etadot=0, matches Fortran ----
    print()
    print("=" * 70)
    print("CHECKING: does middle formula + etadot=0 BCs reproduce Fortran?")
    print("=" * 70)

    # Middle formula for ALL levels (etadot[0]=etadot[-1]=0 handles boundaries):
    vadv_mid_all = np.zeros(nlevs)
    for k in range(nlevs):
        # In BTU:
        # etadot_below = etadot_btu[k]   (interface at bottom of layer k)
        # etadot_above = etadot_btu[k+1]  (interface at top of layer k)
        # data_below = data[k-1] if k>0 else N/A (etadot=0 kills it)
        # data_above = data[k+1] if k<nlevs-1 else N/A
        term_above = (
            etadot_btu[k + 1] * (datag[k + 1] - datag[k]) if k < nlevs - 1 else 0.0
        )
        term_below = etadot_btu[k] * (datag[k] - datag[k - 1]) if k > 0 else 0.0
        vadv_mid_all[k] = (0.5 / dp_btu[k]) * (term_above + term_below)

    print(f"{'Level':>6s}  {'Fortran':>12s}  {'mid-all':>12s}  {'error':>12s}")
    print("-" * 50)
    for k in range(nlevs):
        label = "sfc" if k == 0 else ("TOA" if k == nlevs - 1 else f"mid-{k}")
        err = vadv_mid_all[k] - vadv_fortran[k]
        print(
            f"{label:>6s}  {vadv_fortran[k]:12.6e}  {vadv_mid_all[k]:12.6e}  {err:12.6e}"
        )

    max_err_mid = np.max(np.abs(vadv_mid_all - vadv_fortran))
    print(f"\nMax abs error (mid-formula extended): {max_err_mid:.6e}")
    if max_err_mid < 1e-15:
        print(">>> Middle formula extended to boundaries MATCHES Fortran <<<")
    else:
        print(">>> Middle formula extended does NOT match Fortran <<<")


if __name__ == "__main__":
    main()
