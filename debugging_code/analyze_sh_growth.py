"""Track the growth of the JAX-Fortran difference by hemisphere and its
zonal wavenumber structure in the SH, across compare_long snapshots."""

import numpy as np

SNAP_DIR = "/tmp/gfs_compare_scratch/snapshots"
STEPS = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

# climt/s2fft rows are N->S: index 0 = ~+87.9N, 63 = ~-87.9S
cos_t, _ = np.polynomial.legendre.leggauss(64)
lats = 90.0 - np.degrees(np.flip(np.arccos(cos_t)))  # N->S in degrees...
# lat of row i: 90 - theta_deg
NH = slice(0, 32)
SH = slice(32, 64)

print(f"{'step':>6} {'day':>5} | {'max|du| NH':>12} {'max|du| SH':>12} | {'maxlat':>7}")
prev_sh = None
for step in STEPS:
    d = np.load(f"{SNAP_DIR}/snap_{step:05d}.npz")
    du = np.abs(d["f_u"].astype(np.float64) - d["j_u"].astype(np.float64))
    nh = du[:, NH, :].max()
    sh = du[:, SH, :].max()
    ilat = np.unravel_index(np.argmax(du), du.shape)[1]
    growth = ""
    if prev_sh and prev_sh > 0 and sh > 0:
        efold_steps = 200 / np.log(sh / prev_sh) if sh != prev_sh else np.inf
        growth = f"  SH e-fold: {efold_steps:7.1f} steps"
    prev_sh = sh
    print(
        f"{step:>6} {step * 5 / 60 / 24:>5.2f} | {nh:12.4e} {sh:12.4e} |"
        f" {lats[ilat]:+6.1f}{growth}"
    )

# Zonal wavenumber structure of the SH difference at step 1400 and 1800
for step in [1000, 1400, 1800]:
    d = np.load(f"{SNAP_DIR}/snap_{step:05d}.npz")
    du = d["f_u"].astype(np.float64) - d["j_u"].astype(np.float64)
    # SH rows only, level with max diff
    k = np.unravel_index(np.argmax(np.abs(du)), du.shape)[0]
    sh_du = du[k, SH, :]  # (32, 127)
    spec = np.abs(np.fft.rfft(sh_du, axis=1)).sum(axis=0)  # (64,)
    top = np.argsort(spec)[::-1][:8]
    print(f"\nstep {step}, level k={k}: dominant zonal wavenumbers of SH du:")
    for m in top:
        print(f"  m={m:3d}: {spec[m]:.4e}")
# Also: JAX SH eddy (deviation from zonal mean) amplitude vs Fortran's
print("\nSH eddy u amplitude (deviation from zonal mean), J vs F:")
for step in STEPS:
    d = np.load(f"{SNAP_DIR}/snap_{step:05d}.npz")
    fu = d["f_u"].astype(np.float64)[:, SH, :]
    ju = d["j_u"].astype(np.float64)[:, SH, :]
    f_eddy = np.abs(fu - fu.mean(axis=2, keepdims=True)).max()
    j_eddy = np.abs(ju - ju.mean(axis=2, keepdims=True)).max()
    print(f"  step {step:5d}: F_eddy={f_eddy:10.4e}  J_eddy={j_eddy:10.4e}")
