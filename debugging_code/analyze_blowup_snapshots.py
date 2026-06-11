"""Postmortem of the JAX blow-up using compare_long snapshots.

For each snapshot: JAX spectral vorticity energy by degree l and by
level, compared with the Fortran-trajectory state (via grid fields).
Identifies where (l, level, latitude) the JAX solution departs.
"""

import sys

import numpy as np

SNAP_DIR = "/tmp/gfs_compare_scratch/snapshots"
L = 64


def spectral_energy_by_l(flm):
    """flm: (n_lev, L, 2L-1) -> energy (n_lev, L) summed over m."""
    return np.sum(np.abs(flm) ** 2, axis=-1)


def main():
    steps = [int(s) for s in sys.argv[1:]] or [1800, 2000, 2126]
    for step in steps:
        d = np.load(f"{SNAP_DIR}/snap_{step:05d}.npz")
        vort = d["j_spec_vort"]  # (n_lev, L, 2L-1) complex64
        div = d["j_spec_div"]

        ev = spectral_energy_by_l(vort)  # (n_lev, L)
        ed = spectral_energy_by_l(div)

        print(f"\n=== step {step} ===")
        # Energy by l-band, summed over levels
        bands = [(0, 10), (10, 20), (20, 30), (30, 38), (38, 41), (41, 64)]
        print("  vort energy by l-band (sum levels):")
        tot_v = ev.sum()
        for lo, hi in bands:
            e = ev[:, lo:hi].sum()
            print(f"    l in [{lo:2d},{hi:2d}): {e:10.3e}  ({e / tot_v:8.2%})")
        print("  div energy by l-band (sum levels):")
        tot_d = ed.sum()
        for lo, hi in bands:
            e = ed[:, lo:hi].sum()
            print(f"    l in [{lo:2d},{hi:2d}): {e:10.3e}  ({e / tot_d:8.2%})")

        # Energy by level (sum l>=30 — small scales), bottom-to-top
        print("  small-scale (l>=30) vort energy by level (k=0 surface):")
        es = ev[:, 30:].sum(axis=1)
        for k in range(ev.shape[0]):
            bar = "#" * int(60 * es[k] / max(es.max(), 1e-300))
            print(f"    k={k:2d}: {es[k]:10.3e} {bar}")

        # Grid-space u difference (J vs F) by latitude band, near-surface
        fu = d["f_u"].astype(np.float64)
        ju = d["j_u"].astype(np.float64)
        diff = np.abs(fu - ju).max(axis=(0, 2))  # (n_lat,)
        imax = int(np.argmax(diff))
        print(f"  max |uF-uJ| = {np.abs(fu - ju).max():.3e} at lat index {imax}/{len(diff) - 1}")
        print(f"  |u|max F = {np.abs(fu).max():.2f}, J = {np.abs(ju).max():.2f}")


if __name__ == "__main__":
    main()
