import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/Users/joymonteiro/github/gfs_dynamical_core/debugging_code/evidence"
f = np.load("/tmp/gfs_hs_fortran/hs_zonal_mean_fortran.npz")
j = np.load("/tmp/gfs_hs_jax/hs_zonal_mean_jax.npz")
print("samples:", int(f["n"]), int(j["n"]))

cos_t, _ = np.polynomial.legendre.leggauss(64)
lats = 90.0 - np.degrees(np.flip(np.arccos(cos_t)))

uF = f["u_sum"] / f["n"]; uJ = j["u_sum"] / j["n"]
TF = f["T_sum"] / f["n"]; TJ = j["T_sum"] / j["n"]
pF = (f["p_sum"] / f["n"]).mean(axis=1) / 100.0  # (nlev,) hPa
pJ = (j["p_sum"] / j["n"]).mean(axis=1) / 100.0

fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
levels_u = np.arange(-32, 33, 4)
levels_T = np.arange(180, 311, 10)

for col, (fld_F, fld_J, levels, cmap, name) in enumerate([
        (uF, uJ, levels_u, "RdBu_r", "zonal-mean u (m/s)"),
        (TF, TJ, levels_T, "viridis", "zonal-mean T (K)")]):
    ax = axes[0][col]
    c = ax.contourf(lats, pF, fld_F, levels=levels, cmap=cmap, extend="both")
    ax.contour(lats, pF, fld_F, levels=levels, colors="k", linewidths=0.4)
    ax.set_title(f"Fortran: {name}")
    plt.colorbar(c, ax=ax)
    ax = axes[1][col]
    c = ax.contourf(lats, pJ, fld_J, levels=levels, cmap=cmap, extend="both")
    ax.contour(lats, pJ, fld_J, levels=levels, colors="k", linewidths=0.4)
    ax.set_title(f"JAX: {name}")
    plt.colorbar(c, ax=ax)

# difference panels
du = uJ - uF; dT = TJ - TF
ax = axes[0][2]
c = ax.contourf(lats, pF, du, levels=np.linspace(-3, 3, 13), cmap="RdBu_r", extend="both")
ax.set_title(f"u diff (J-F), max|du|={np.abs(du).max():.2f} m/s")
plt.colorbar(c, ax=ax)
ax = axes[1][2]
c = ax.contourf(lats, pF, dT, levels=np.linspace(-2, 2, 13), cmap="RdBu_r", extend="both")
ax.set_title(f"T diff (J-F), max|dT|={np.abs(dT).max():.2f} K")
plt.colorbar(c, ax=ax)

for ax in axes.flat:
    ax.invert_yaxis() if ax.get_ylim()[0] < ax.get_ylim()[1] else None
    ax.set_xlabel("latitude")
axes[0][0].set_ylabel("pressure (hPa)"); axes[1][0].set_ylabel("pressure (hPa)")
for ax in axes.flat:
    ax.set_ylim(1000, 0)
fig.suptitle("Held-Suarez climatology, days 200-400 (800 samples), T40 L20 dt=10min — Fortran (explicit) vs JAX (IMEX)", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/6_heldsuarez_zonal_mean.png", dpi=130)
print(f"u: max|J-F| = {np.abs(du).max():.2f} m/s   (jet max F={uF.max():.1f}, J={uJ.max():.1f})")
print(f"T: max|J-F| = {np.abs(dT).max():.2f} K")
