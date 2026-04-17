#!/usr/bin/env python
"""
jw06_diagnostics.py
===================
Diagnostic suite from Jablonowski & Williamson (2006) Q.J.R.Meteorol.Soc. 132,
applied to the GFS Fortran and JAX dynamical cores.

Section 4 — Steady-state test (add_perturbation=False):
  jw06_steady_errors.png   l2 zonal-symmetry (Eq.14) and zonal-mean drift (Eq.15)

Section 5 — Baroclinic wave test (add_perturbation=True):
  jw06_wave_snapshots.png  Surface pressure + 850 hPa T at days 4, 6, 8, 9, 10
  jw06_vorticity.png       850 hPa relative vorticity at days 7 and 9
  jw06_ps_l2_diff.png      l2(ps_JAX − ps_Fortran) over time (Eq.16)

Usage
-----
  python jw06_diagnostics.py [--no-fortran]

Pass --no-fortran to skip the Fortran core (plots that need it are omitted).
"""

import os
import sys
import argparse

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import climt
from sympl import set_constant

# ── physical constants / grid ─────────────────────────────────────────────────
set_constant("reference_air_pressure", value=1e5, units="Pa")

EARTH_RADIUS = 6.371229e6   # m
L     = 64
N_LON = 2 * L - 1           # 127
N_LAT = L                   # 64
N_LEV = 20

# ── run parameters ────────────────────────────────────────────────────────────
DAYS_STEADY  = 30    # Section 4 (paper Fig 3/4 uses 30 days)
DAYS_WAVE    = 10    # Section 5
DT_JAX_MIN   = 10   # JAX timestep  [min]
DT_FORT_MIN  = 5    # Fortran timestep [min]

SNAP_DAYS = [4, 6, 8, 9, 10]   # snapshot days (paper Figs 5-7)
VORT_DAYS = [7, 9]             # vorticity snapshot days  (paper Fig 8)


# ═════════════════════════════════════════════════════════════════════════════
# Helper: l2 norms (Eqs. 14, 15, 16 in the paper)
# ═════════════════════════════════════════════════════════════════════════════

def _gauss_weights_from_lats(n_lat):
    """Gaussian quadrature weights for a GL grid of n_lat latitudes.
    Returned weights sum to 2 (standard leggauss convention)."""
    _, w = np.polynomial.legendre.leggauss(n_lat)
    return w   # sum = 2


def l2_zonal_asymmetry(u, gauss_w, dbk):
    """Eq. 14 — l2(u − ū):  RMS of the departure from the zonal mean.

    u       : (n_lev, n_lat, n_lon)  m s^-1
    gauss_w : (n_lat,)               Gaussian weights, sum=2
    dbk     : (n_lev,)               layer thickness in eta/sigma, sum≈1
    Returns a scalar in m s^-1.
    """
    u_bar  = u.mean(axis=2)                            # (n_lev, n_lat)
    anom2  = (u - u_bar[:, :, None]) ** 2             # (n_lev, n_lat, n_lon)
    num    = np.einsum("kjl,k,j->", anom2, dbk, gauss_w) / N_LON
    den    = np.dot(dbk, gauss_w.sum() * np.ones(N_LEV))   # sum_k dbk * sum_j w_j
    # equivalently: den = dbk.sum() * gauss_w.sum()
    den    = dbk.sum() * gauss_w.sum()
    return float(np.sqrt(num / den))


def l2_zonal_mean_drift(u, u0, gauss_w, dbk):
    """Eq. 15 — l2(ū(t) − ū(0)):  RMS drift of the zonal mean from t=0.

    u, u0   : (n_lev, n_lat, n_lon)
    gauss_w : (n_lat,)
    dbk     : (n_lev,)
    Returns a scalar in m s^-1.
    """
    u_bar  = u.mean(axis=2)    # (n_lev, n_lat)
    u0_bar = u0.mean(axis=2)
    diff2  = (u_bar - u0_bar) ** 2
    num    = np.einsum("kj,k,j->", diff2, dbk, gauss_w)
    den    = dbk.sum() * gauss_w.sum()
    return float(np.sqrt(num / den))


def l2_ps_diff(ps1, ps2, gauss_w):
    """Eq. 16 — l2(ps1 − ps2):  area-weighted RMS surface-pressure difference.

    ps1, ps2 : (n_lat, n_lon)  Pa
    gauss_w  : (n_lat,)
    Returns a scalar in hPa.
    """
    diff2 = (ps1 - ps2) ** 2             # (n_lat, n_lon)
    num   = (diff2 * gauss_w[:, None]).sum() / N_LON
    den   = gauss_w.sum()
    return float(np.sqrt(num / den)) / 100.0   # Pa → hPa


# ═════════════════════════════════════════════════════════════════════════════
# Helper: pressure grid and level-interpolation
# ═════════════════════════════════════════════════════════════════════════════

def pressure_3d(ps, ak, bk):
    """Compute mid-level pressure on the 3-D grid.

    ps   : (n_lat, n_lon)  Pa
    ak   : (n_lev+1,)      interface A-coeff  — BTU (index 0 = surface)
    bk   : (n_lev+1,)      interface B-coeff  — BTU
    Returns p : (n_lev, n_lat, n_lon), BTU — p[0] > p[-1]
    """
    ak_mid = 0.5 * (ak[:-1] + ak[1:])   # (n_lev,)
    bk_mid = 0.5 * (bk[:-1] + bk[1:])
    return ak_mid[:, None, None] + bk_mid[:, None, None] * ps[None, :, :]


def interp_to_level(field, p3d, p_target):
    """Linear interpolation of 'field' to pressure level p_target.

    field, p3d : (n_lev, n_lat, n_lon) — BTU (p3d[0] > p3d[-1])
    p_target   : scalar [Pa]
    Returns     : (n_lat, n_lon)
    """
    n_lev, n_lat, n_lon = field.shape
    # Number of levels with p >= p_target (i.e. below or at the target)
    n_below = (p3d >= p_target).sum(axis=0)          # (n_lat, n_lon)
    k_lo    = np.clip(n_below - 1, 0, n_lev - 2)    # lower bounding level index
    k_hi    = k_lo + 1

    j_idx, i_idx = np.mgrid[0:n_lat, 0:n_lon]
    p_lo = p3d[k_lo, j_idx, i_idx]
    p_hi = p3d[k_hi, j_idx, i_idx]
    f_lo = field[k_lo, j_idx, i_idx]
    f_hi = field[k_hi, j_idx, i_idx]

    denom = np.log(np.maximum(p_hi, 1.0)) - np.log(np.maximum(p_lo, 1.0))
    alpha = np.where(
        np.abs(denom) > 1e-10,
        (np.log(p_target) - np.log(np.maximum(p_lo, 1.0))) / denom,
        0.5,
    )
    alpha = np.clip(alpha, 0.0, 1.0)
    return f_lo + alpha * (f_hi - f_lo)


# ═════════════════════════════════════════════════════════════════════════════
# Helper: relative vorticity
# ═════════════════════════════════════════════════════════════════════════════

def relative_vorticity_850(u_3d, v_3d, p3d, lat_deg, lon_deg):
    """850 hPa relative vorticity from u, v on the Gaussian grid.

    ζ = (1 / (a cos φ)) [∂v/∂λ  −  ∂(u cos φ)/∂φ]

    u_3d, v_3d : (n_lev, n_lat, n_lon)  m s^-1
    lat_deg    : (n_lat,)  degrees
    lon_deg    : (n_lon,)  degrees
    Returns     : (n_lat, n_lon)  s^-1
    """
    u850 = interp_to_level(u_3d, p3d, 850e2)
    v850 = interp_to_level(v_3d, p3d, 850e2)

    lat = np.deg2rad(lat_deg)         # (n_lat,)
    lon = np.deg2rad(lon_deg)         # (n_lon,)
    cos_lat = np.cos(lat)             # (n_lat,)

    dlon = lon[1] - lon[0]
    dv_dlam = np.gradient(v850, axis=1) / dlon          # ∂v/∂λ

    u_cos = u850 * cos_lat[:, None]
    du_cos_dphi = np.gradient(u_cos, lat, axis=0)       # ∂(u cos φ)/∂φ

    safe_cos = np.where(np.abs(cos_lat) < 1e-10, 1e-10, cos_lat)
    return (dv_dlam - du_cos_dphi) / (EARTH_RADIUS * safe_cos[:, None])


# ═════════════════════════════════════════════════════════════════════════════
# Dycore runner
# ═════════════════════════════════════════════════════════════════════════════

def _snapshot(state):
    """Extract plain numpy arrays from a sympl state dict."""
    return {
        "surface_air_pressure": np.array(state["surface_air_pressure"].values),
        "air_temperature":      np.array(state["air_temperature"].values),
        "eastward_wind":        np.array(state["eastward_wind"].values),
        "northward_wind":       np.array(state["northward_wind"].values),
        "latitude":             np.array(state["latitude"].values[:, 0]),
        "longitude":            np.array(state["longitude"].values[0, :]),
        "a_coord": np.array(
            state[
                "atmosphere_hybrid_sigma_pressure_a_coordinate_on_interface_levels"
            ].values
        ),
        "b_coord": np.array(
            state[
                "atmosphere_hybrid_sigma_pressure_b_coordinate_on_interface_levels"
            ].values
        ),
    }


def run_dycore(dycore_class, n_days, dt_min, add_perturbation, label,
               dycore_kwargs=None, snap_days=None):
    """Run a dycore for n_days and return daily snapshots.

    Parameters
    ----------
    snap_days : set or None
        Days on which to save a full state snapshot (default: every day).

    Returns
    -------
    daily : dict  {day_int: snapshot_dict}
    """
    if dycore_kwargs is None:
        dycore_kwargs = {}
    if snap_days is None:
        snap_days = set(range(n_days + 1))

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {n_days} days | dt={dt_min} min | perturbation={add_perturbation}")
    print(f"{'='*60}")

    grid   = climt.get_grid(nx=N_LON, ny=N_LAT, nz=N_LEV)
    dycore = dycore_class(**dycore_kwargs)
    dcmip  = climt.DcmipInitialConditions(add_perturbation=add_perturbation)
    state  = climt.get_default_state([dycore], grid_state=grid)
    state.update(dcmip(state))

    dt             = timedelta(minutes=dt_min)
    steps_per_day  = int(24 * 60 / dt_min)

    daily = {}
    if 0 in snap_days:
        daily[0] = _snapshot(state)

    for day in range(1, n_days + 1):
        for _ in range(steps_per_day):
            _, output = dycore(state, timestep=dt)
            state.update(output)
            state["time"] += dt

        ps  = np.array(state["surface_air_pressure"].values)
        ps_min, ps_max = ps.min() / 100, ps.max() / 100
        print(f"  Day {day:2d}: PS {ps_min:.1f} – {ps_max:.1f} hPa")

        if day in snap_days:
            daily[day] = _snapshot(state)

        if ps_max > 2e5 or np.isnan(ps_max):
            print(f"  UNSTABLE — stopping at day {day}.")
            break

    return daily


# ═════════════════════════════════════════════════════════════════════════════
# Section 4 — Steady-state error norms
# ═════════════════════════════════════════════════════════════════════════════

def run_steady_state(dycore_class, label, dycore_kwargs=None):
    """Run without perturbation; compute Eqs 14 & 15 daily."""
    daily = run_dycore(
        dycore_class, DAYS_STEADY, DT_JAX_MIN if "JAX" in label else DT_FORT_MIN,
        add_perturbation=False, label=label,
        dycore_kwargs=dycore_kwargs,
        snap_days=set(range(DAYS_STEADY + 1)),
    )

    d0   = daily[0]
    ak   = d0["a_coord"]
    bk   = d0["b_coord"]
    dbk  = bk[:-1] - bk[1:]   # layer thickness (BTU, positive)
    gw   = _gauss_weights_from_lats(N_LAT)
    u0   = d0["eastward_wind"]

    days_out, asym_l2, drift_l2 = [], [], []
    for day in sorted(daily.keys()):
        snap = daily[day]
        u    = snap["eastward_wind"]
        days_out.append(day)
        asym_l2.append(l2_zonal_asymmetry(u, gw, dbk))
        drift_l2.append(l2_zonal_mean_drift(u, u0, gw, dbk))

    return days_out, asym_l2, drift_l2


def plot_steady_errors(results):
    """Plot Eqs 14 & 15 for all models.

    results : list of (label, days, asym_l2, drift_l2)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    styles = {"JAX": dict(color="C0", lw=2), "Fortran": dict(color="C1", lw=2, ls="--")}

    for label, days, asym, drift in results:
        kw = styles.get("JAX" if "JAX" in label else "Fortran", {})
        axes[0].semilogy(days, asym,  label=label, **kw)
        axes[1].semilogy(days, drift, label=label, **kw)

    for ax, title, ylabel in zip(
        axes,
        [r"$\ell_2(u - \bar{u})$  (zonal asymmetry, Eq.14)",
         r"$\ell_2(\bar{u}(t) - \bar{u}(0))$  (zonal-mean drift, Eq.15)"],
        [r"$\ell_2$ norm  (m s$^{-1}$)", r"$\ell_2$ norm  (m s$^{-1}$)"],
    ):
        ax.set_xlabel("Model day")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", ls=":")

    fig.suptitle("Steady-state test  (J&W 2006, Sect. 4)", fontsize=13)
    fig.tight_layout()
    fig.savefig("jw06_steady_errors.png", dpi=150)
    plt.close(fig)
    print("Saved jw06_steady_errors.png")


# ═════════════════════════════════════════════════════════════════════════════
# Section 5 — Baroclinic wave snapshots
# ═════════════════════════════════════════════════════════════════════════════

def run_wave(dycore_class, label, dycore_kwargs=None):
    """Run with perturbation; save snapshots at SNAP_DAYS ∪ VORT_DAYS."""
    all_days = set(range(DAYS_WAVE + 1)) | set(SNAP_DAYS) | set(VORT_DAYS)
    dt_min   = DT_JAX_MIN if "JAX" in label else DT_FORT_MIN
    return run_dycore(
        dycore_class, DAYS_WAVE, dt_min,
        add_perturbation=True, label=label,
        dycore_kwargs=dycore_kwargs,
        snap_days=all_days,
    )


def _ps_and_t850(snap):
    """Extract surface pressure (hPa) and 850 hPa temperature from a snapshot."""
    ak  = snap["a_coord"]
    bk  = snap["b_coord"]
    ps  = snap["surface_air_pressure"]
    t   = snap["air_temperature"]
    p3d = pressure_3d(ps, ak, bk)
    t850 = interp_to_level(t, p3d, 850e2)
    return ps / 100.0, t850


def plot_wave_snapshots(results, label="JAX"):
    """Paper Fig 5 / 7 style: surface pressure (left) + 850 hPa T (right)
    at each of SNAP_DAYS.

    results : dict {day: snapshot}
    """
    days_available = [d for d in SNAP_DAYS if d in results]
    n_rows = len(days_available)
    if n_rows == 0:
        print("No snapshot days available — skipping wave snapshots.")
        return

    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    snap0 = results[0]
    lon   = snap0["longitude"]
    lat   = snap0["latitude"]

    # Restrict to NH (lat >= 0) and lon 45–360 like the paper
    lat_mask = lat >= 0
    lon_mask = lon >= 45
    lon_plot = lon[lon_mask]
    lat_plot = lat[lat_mask]

    t_levels   = np.linspace(220, 310, 19)

    for row, day in enumerate(days_available):
        snap       = results[day]
        ps_hpa, t850 = _ps_and_t850(snap)

        ps_plot  = ps_hpa[np.ix_(lat_mask, lon_mask)]
        t_plot   = t850[np.ix_(lat_mask, lon_mask)]

        # Surface pressure — levels adapt to the data range each day
        ps_lo = np.floor(ps_plot.min() / 4) * 4
        ps_hi = np.ceil(ps_plot.max() / 4) * 4
        ps_levels = np.linspace(ps_lo, ps_hi, 17)

        # Surface pressure
        ax = axes[row, 0]
        cf = ax.contourf(lon_plot, lat_plot, ps_plot,
                         levels=ps_levels, cmap="RdBu_r", extend="both")
        ax.contour(lon_plot, lat_plot, ps_plot, levels=ps_levels,
                   colors="k", linewidths=0.3, alpha=0.4)
        plt.colorbar(cf, ax=ax, label="hPa", shrink=0.85)
        ax.set_title(f"Day {day} — Surface pressure")
        ax.set_ylabel("Latitude")
        if row == n_rows - 1:
            ax.set_xlabel("Longitude")

        # 850 hPa temperature
        ax = axes[row, 1]
        cf = ax.contourf(lon_plot, lat_plot, t_plot,
                         levels=t_levels, cmap="RdBu_r", extend="both")
        ax.contour(lon_plot, lat_plot, t_plot, levels=t_levels,
                   colors="k", linewidths=0.3, alpha=0.4)
        plt.colorbar(cf, ax=ax, label="K", shrink=0.85)
        ax.set_title(f"Day {day} — 850 hPa temperature")
        if row == n_rows - 1:
            ax.set_xlabel("Longitude")

    fig.suptitle(f"Baroclinic wave evolution — {label}  (J&W 2006, Sect. 5)",
                 fontsize=13)
    fig.tight_layout()
    fname = f"jw06_wave_snapshots_{label.replace(' ', '_')}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


def plot_vorticity(results_dict, label_key_pairs):
    """Paper Fig 8 style: 850 hPa relative vorticity at VORT_DAYS.

    label_key_pairs : list of (plot_label, results_dict)
    """
    days_available = [d for d in VORT_DAYS if any(d in r for _, r in label_key_pairs)]
    if not days_available:
        return

    n_models = len(label_key_pairs)
    n_days   = len(days_available)
    fig, axes = plt.subplots(n_days, n_models,
                             figsize=(7 * n_models, 4.5 * n_days),
                             squeeze=False)

    for col, (lbl, results) in enumerate(label_key_pairs):
        snap0  = results.get(0) or results[min(results.keys())]
        ak, bk = snap0["a_coord"], snap0["b_coord"]
        lat    = snap0["latitude"]
        lon    = snap0["longitude"]

        # Focus on NH mid-latitudes lon 90–270 like the paper
        lat_mask = (lat >= 25) & (lat <= 75)
        lon_mask = (lon >= 90) & (lon <= 270)
        lon_plot = lon[lon_mask]
        lat_plot = lat[lat_mask]

        for row, day in enumerate(days_available):
            if day not in results:
                axes[row, col].set_visible(False)
                continue

            snap = results[day]
            ps   = snap["surface_air_pressure"]
            p3d  = pressure_3d(ps, ak, bk)
            vort = relative_vorticity_850(
                snap["eastward_wind"], snap["northward_wind"], p3d, lat, lon
            )
            vort_plot = vort[np.ix_(lat_mask, lon_mask)] * 1e5   # → 10^-5 s^-1

            # Colour scale matches paper (Fig 8)
            vmax = 6 if day == 7 else 40
            vmin = -vmax // 3

            ax  = axes[row, col]
            cf  = ax.contourf(lon_plot, lat_plot, vort_plot,
                              levels=np.linspace(vmin, vmax, 17),
                              cmap="RdBu_r", extend="both")
            ax.contour(lon_plot, lat_plot, vort_plot,
                       levels=[0], colors="k", linewidths=0.8)
            plt.colorbar(cf, ax=ax, label=r"$10^{-5}$ s$^{-1}$", shrink=0.85)
            ax.set_title(f"Day {day} — {lbl}")
            ax.set_ylabel("Latitude")
            ax.set_xlabel("Longitude")

    fig.suptitle("850 hPa relative vorticity  (J&W 2006, Fig. 8)", fontsize=13)
    fig.tight_layout()
    fig.savefig("jw06_vorticity.png", dpi=150)
    plt.close(fig)
    print("Saved jw06_vorticity.png")


# ═════════════════════════════════════════════════════════════════════════════
# Section 5(e) — l2 surface-pressure difference norm (Eq. 16)
# ═════════════════════════════════════════════════════════════════════════════

def plot_ps_l2_diff(daily_jax, daily_fort):
    """Eq. 16: l2(ps_JAX − ps_Fortran) over time."""
    gw   = _gauss_weights_from_lats(N_LAT)
    days, norms = [], []
    for day in sorted(set(daily_jax.keys()) & set(daily_fort.keys())):
        ps_j = daily_jax[day]["surface_air_pressure"]
        ps_f = daily_fort[day]["surface_air_pressure"]
        days.append(day)
        norms.append(l2_ps_diff(ps_j, ps_f, gw))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(days, norms, "o-", color="C0", lw=2)
    ax.set_xlabel("Model day")
    ax.set_ylabel(r"$\ell_2(p_{s,\mathrm{JAX}} - p_{s,\mathrm{Fortran}})$  (hPa)")
    ax.set_title(r"Surface-pressure $\ell_2$ difference norm  (J&W 2006, Eq. 16)")
    ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    fig.savefig("jw06_ps_l2_diff.png", dpi=150)
    plt.close(fig)
    print("Saved jw06_ps_l2_diff.png")
    return days, norms


# ═════════════════════════════════════════════════════════════════════════════
# Additional: surface pressure minimum tracking (paper text Sect. 5a)
# ═════════════════════════════════════════════════════════════════════════════

def plot_ps_min_tracking(results_dict_list):
    """Plot global surface-pressure minimum over time (qualitative stability check)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = [dict(color=f"C{i}", lw=2, ls=ls)
              for i, ls in enumerate(["-", "--", ":", "-."])]
    for (lbl, daily), sty in zip(results_dict_list, styles):
        days  = sorted(daily.keys())
        ps_min = [daily[d]["surface_air_pressure"].min() / 100 for d in days]
        ax.plot(days, ps_min, label=lbl, **sty)

    ax.set_xlabel("Model day")
    ax.set_ylabel("Min surface pressure (hPa)")
    ax.set_title("Surface-pressure minimum (deepest low pressure)")
    ax.legend()
    ax.grid(True, ls=":")
    fig.tight_layout()
    fig.savefig("jw06_ps_min.png", dpi=150)
    plt.close(fig)
    print("Saved jw06_ps_min.png")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-fortran", action="store_true",
                        help="Skip the Fortran dycore (default: try to use it)")
    args = parser.parse_args()

    # Lazy imports — keep top-level clean
    from gfs_dynamical_core.component_jax import GFSDynamicsJAX

    use_fortran = not args.no_fortran
    if use_fortran:
        try:
            from gfs_dynamical_core import GFSDynamicalCore
        except ImportError:
            print("Fortran dycore not available — running JAX only.")
            use_fortran = False

    # ── Section 4: Steady-state test ─────────────────────────────────────────
    print("\n" + "━" * 60)
    print("SECTION 4 — Steady-state test (no perturbation)")
    print("━" * 60)

    steady_results = []

    days_j, asym_j, drift_j = run_steady_state(GFSDynamicsJAX, "JAX")
    steady_results.append(("JAX", days_j, asym_j, drift_j))

    if use_fortran:
        days_f, asym_f, drift_f = run_steady_state(GFSDynamicalCore, "Fortran")
        steady_results.append(("Fortran", days_f, asym_f, drift_f))

    plot_steady_errors(steady_results)

    # ── Section 5: Baroclinic wave test ──────────────────────────────────────
    print("\n" + "━" * 60)
    print("SECTION 5 — Baroclinic wave test (with perturbation)")
    print("━" * 60)

    daily_jax = run_wave(GFSDynamicsJAX, "JAX")
    plot_wave_snapshots(daily_jax, label="JAX")

    vort_pairs = [("JAX", daily_jax)]

    if use_fortran:
        daily_fort = run_wave(GFSDynamicalCore, "Fortran")
        plot_wave_snapshots(daily_fort, label="Fortran")
        vort_pairs.append(("Fortran", daily_fort))

        # Eq. 16: l2 difference norm
        print("\nComputing l2 surface-pressure difference norms (Eq. 16) …")
        plot_ps_l2_diff(daily_jax, daily_fort)

    plot_vorticity(None, vort_pairs)

    # Surface pressure minimum tracking
    ps_min_inputs = [("JAX", daily_jax)]
    if use_fortran:
        ps_min_inputs.append(("Fortran", daily_fort))
    plot_ps_min_tracking(ps_min_inputs)

    print("\nDone. Figures written to the current directory.")


if __name__ == "__main__":
    main()
