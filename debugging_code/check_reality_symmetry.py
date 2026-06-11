"""Measure reality-symmetry violation in JAX cached spectral states.

For a real field in s2fft convention (CS phase included):
    f_{l,-m} = (-1)^m * conj(f_{l,+m})
Violation => the grid field has a nonzero imaginary part that the
running JAX component never removes (to_numpy takes .real only at the
sympl boundary; the internal cached state keeps it).
"""

import numpy as np

SNAP_DIR = "/tmp/gfs_compare_scratch/snapshots"
L = 64
STEPS = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]


def asymmetry(flm):
    """flm: (n_lev, L, 2L-1). Return max and rms of the violation, and field scale."""
    m = np.arange(1, L)
    pos = flm[:, :, L - 1 + m]  # f_{l,+m}
    neg = flm[:, :, L - 1 - m]  # f_{l,-m}
    viol = neg - ((-1.0) ** m)[None, None, :] * np.conj(pos)
    scale = np.abs(flm).max()
    return np.abs(viol).max(), np.sqrt((np.abs(viol) ** 2).mean()), scale


print(f"{'step':>6} | {'vort viol max':>13} {'vort scale':>11} {'ratio':>9} |"
      f" {'div viol max':>13} {'div scale':>11} {'ratio':>9}")
for step in STEPS:
    d = np.load(f"{SNAP_DIR}/snap_{step:05d}.npz")
    vmax, vrms, vsc = asymmetry(d["j_spec_vort"].astype(np.complex128))
    dmax, drms, dsc = asymmetry(d["j_spec_div"].astype(np.complex128))
    print(f"{step:>6} | {vmax:13.4e} {vsc:11.3e} {vmax / vsc:9.2e} |"
          f" {dmax:13.4e} {dsc:11.3e} {dmax / dsc:9.2e}")
