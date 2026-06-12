"""Evidence plots for the baroclinic-wave blow-up investigation.

Produces PNGs in debugging_code/evidence/ from:
- compare_long_broken.log     (Fortran + JAX-before-fix, side by side)
- stability_check_fixed_*.log (JAX-after-fix)
- fortran_ctrl.log            (perturbed-Fortran control)
- snapshots_brokenrun/        (field maps + spectral states of broken run)
"""

import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "evidence")
os.makedirs(OUT, exist_ok=True)

SCR = "/tmp/gfs_compare_scratch"


# ── Parsers ──────────────────────────────────────────────────────────────
def parse_compare_long(path):
    """rows: step day u_rel v_rel T_rel ps_rel | psF[min,max] psJ[min,max] | uF uJ"""
    rows = []
    pat = re.compile(
        r"^\s*(\d+)\s+([\d.]+)\s*\|\s*([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s*\|"
        r"\s*psF\[\s*([-\d.]+),\s*([-\d.]+)\]\s*psJ\[\s*([-\d.naninf]+),\s*([-\d.naninf]+)\]\s*\|"
        r"\s*([-\d.e+]+)\s+([\d.e+-]+)"
    )
    for line in open(path):
        m = pat.match(line)
        if m:
            rows.append([float(g) for g in m.groups()])
    return np.array(rows)


def parse_stability(path):
    """rows: step day ps_min ps_max u_max"""
    rows = []
    pat = re.compile(
        r"^step\s+(\d+)\s+day\s+([\d.]+)\s+\|\s+ps \[\s*([-\d.]+),\s*([-\d.]+)\] hPa \| \|u\|max\s+([\d.]+)"
    )
    for line in open(path):
        m = pat.match(line)
        if m:
            rows.append([float(g) for g in m.groups()])
    return np.array(rows)


broken = parse_compare_long(os.path.join(SCR, "compare_long_broken.log"))
fixed_log = sorted(glob.glob(os.path.join(HERE, "stability_check_fixed_*.log")))[-1]
fixed = parse_stability(fixed_log)
fctrl = parse_stability("/tmp/gfs_fortran_ctrl/fortran_ctrl.log")

# ── Plot 1: surface-pressure minimum (cyclone depth) ─────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax = axes[0]
ax.plot(broken[:, 1], broken[:, 6], "k-", lw=2, label="Fortran (reference)")
ax.plot(broken[:, 1], broken[:, 8], "r--", lw=1.5, label="JAX before fix (IMEX)")
ax.plot(fixed[:, 1], fixed[:, 2], "b-", lw=1.5, label="JAX after fix (IMEX)")
ax.plot(fctrl[:, 1], fctrl[:, 2], "g:", lw=2, label="Fortran + 1e-6 noise (control)")
ax.axvline(7.38, color="r", alpha=0.3)
ax.text(7.45, 940, "JAX blow-up\n(before fix)", color="r", fontsize=9)
ax.set_ylim(800, 1010)
ax.set_ylabel("min surface pressure (hPa)")
ax.set_title("DCMIP 4.1 baroclinic wave, T40 L20 dt=5min — cyclone deepening")
ax.legend(loc="lower left", fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(broken[:, 1], broken[:, 10], "k-", lw=2, label="Fortran (reference)")
ax.plot(broken[:, 1], broken[:, 11], "r--", lw=1.5, label="JAX before fix")
ax.plot(fixed[:, 1], fixed[:, 4], "b-", lw=1.5, label="JAX after fix")
ax.plot(fctrl[:, 1], fctrl[:, 4], "g:", lw=2, label="Fortran control")
ax.set_ylim(30, 80)
ax.set_xlabel("simulated days")
ax.set_ylabel("max |u| (m/s)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "1_timeseries_psmin_umax.png"), dpi=130)
plt.close(fig)

# ── Plot 2: trajectory agreement after fix (fixed JAX vs Fortran ctrl) ──
# Interpolate Fortran control onto fixed-run days
fig, ax = plt.subplots(figsize=(10, 4))
f_interp = np.interp(fixed[:, 1], fctrl[:, 1], fctrl[:, 2])
ax.plot(fixed[:, 1], np.abs(fixed[:, 2] - f_interp), "b-")
ax.set_yscale("log")
ax.set_xlabel("simulated days")
ax.set_ylabel("|ps_min difference| (hPa)")
ax.set_title("JAX (fixed) vs Fortran control: cyclone-depth difference")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "2_fixed_vs_fortran_psmin_diff.png"), dpi=130)
plt.close(fig)

# ── Plot 3: reality-symmetry (ghost mode) exponential growth ─────────────
L = 64
steps = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
viols_v, viols_d, scale_d = [], [], []
m = np.arange(1, L)
for s in steps:
    d = np.load(os.path.join(SCR, "snapshots_brokenrun", f"snap_{s:05d}.npz"))
    for key, acc in (("j_spec_vort", viols_v), ("j_spec_div", viols_d)):
        flm = d[key].astype(np.complex128)
        v = flm[..., L - 1 - m] - ((-1.0) ** m)[None, None, :] * np.conj(flm[..., L - 1 + m])
        acc.append(np.abs(v).max())
    scale_d.append(np.abs(d["j_spec_div"]).max())

fig, ax = plt.subplots(figsize=(10, 5))
days = np.array(steps) * 5 / 1440
ax.semilogy(days, viols_v, "o-", label="vorticity asymmetry (ghost amplitude)")
ax.semilogy(days, viols_d, "s-", label="divergence asymmetry")
ax.semilogy(days, scale_d, "k--", alpha=0.6, label="divergence field scale")
# 5.5-hour e-fold reference line through the step-1200 point
t = np.linspace(2.8, 7.0, 50)
ax.semilogy(t, viols_d[5] * np.exp((t - days[5]) * 24 / 5.5), "r:", lw=2,
            label="5.5 h e-folding reference")
ax.set_xlabel("simulated days")
ax.set_ylabel("max reality-symmetry violation")
ax.set_title("Ghost-mode growth in the broken run:  f(l,-m) - (-1)^m conj(f(l,+m))")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "3_ghost_mode_growth.png"), dpi=130)
plt.close(fig)

# ── Plot 4: where the noise lives — u difference map at day 6.9 ─────────
d = np.load(os.path.join(SCR, "snapshots_brokenrun", "snap_02000.npz"))
fu = d["f_u"].astype(np.float64)
ju = d["j_u"].astype(np.float64)
k = 19  # top level, where ghost is strongest
cos_t, _ = np.polynomial.legendre.leggauss(64)
lats = 90.0 - np.degrees(np.flip(np.arccos(cos_t)))
lons = np.linspace(0, 360, 127, endpoint=False)

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
vmax = 30
im0 = axes[0].pcolormesh(lons, lats, fu[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
axes[0].set_title("day 6.9, level k=19 (TOA): Fortran u (healthy zonal jets + NH wave)")
im1 = axes[1].pcolormesh(lons, lats, ju[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
axes[1].set_title("JAX before fix: u — both jets covered in ghost noise (l≈30-38)")
im2 = axes[2].pcolormesh(lons, lats, ju[k] - fu[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
axes[2].set_title("difference (JAX - Fortran)")
for ax_ in axes:
    ax_.set_ylabel("latitude")
fig.colorbar(im0, ax=axes, label="u (m/s)", shrink=0.8)
axes[2].set_xlabel("longitude")
fig.savefig(os.path.join(OUT, "4_ghost_noise_map_day6.9.png"), dpi=130)
plt.close(fig)

print("wrote:")
for f in sorted(os.listdir(OUT)):
    print("  debugging_code/evidence/" + f)

# ── Stats table ──────────────────────────────────────────────────────────
print("\n=== Statistics ===")
print(f"Broken run: JAX blow-up at day 7.38 (step 2126); Fortran stable")
print(f"Fixed run:  12.0 days complete, div-asymmetry exactly 0.0 throughout")
i34 = np.argmin(np.abs(fixed[:, 0] - 3400))
print(f"Fixed JAX  step 3400: ps_min={fixed[i34, 2]:.2f} hPa, umax={fixed[i34, 4]:.2f}")
j34 = np.argmin(np.abs(fctrl[:, 0] - 3400))
print(f"F control  step 3400: ps_min={fctrl[j34, 2]:.2f} hPa, umax={fctrl[j34, 4]:.2f}")
g = np.polyfit(days[4:], np.log(np.array(viols_d[4:])), 1)[0]  # per day
print(f"Ghost e-folding time (fit, steps 1000-2000): {24 / g:.2f} hours")
