"""
Direct 1-step comparison between Fortran and JAX dynamical cores.

Takes one 5-minute timestep from identical DCMIP baroclinic wave initial
conditions and compares every output field.
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

# ── Setup ────────────────────────────────────────────────────────────────
set_constant("reference_air_pressure", value=1e5, units="Pa")

L = 64
n_lon = 2 * L - 1  # 127
n_lat = L  # 64
n_lev = 20
timestep = timedelta(minutes=5)

print(f"Grid: {n_lon}x{n_lat}x{n_lev}  (L={L}),  dt={timestep}")
grid = climt.get_grid(nx=n_lon, ny=n_lat, nz=n_lev)

# ── Initialize both dycores ─────────────────────────────────────────────
dycore_f = GFSDynamicalCore()
dycore_j = GFSDynamicsJAX()
dcmip = climt.DcmipInitialConditions(add_perturbation=True)

# ── Build two independent copies of the same initial state ───────────────
state_f = climt.get_default_state([dycore_f], grid_state=grid)
out_f = dcmip(state_f)
state_f.update(out_f)

state_j = climt.get_default_state([dycore_j], grid_state=grid)
out_j = dcmip(state_j)
state_j.update(out_j)

# ── Verify initial conditions are identical ──────────────────────────────
fields = [
    "eastward_wind",
    "northward_wind",
    "air_temperature",
    "surface_air_pressure",
    "specific_humidity",
]

print("\n=== Initial condition comparison ===")
for name in fields:
    f_arr = np.asarray(state_f[name])
    j_arr = np.asarray(state_j[name])
    diff = np.abs(f_arr - j_arr).max()
    print(f"  {name:30s}  max|diff|={diff:.2e}")

# ── Take one step with each ─────────────────────────────────────────────
print("\n--- Taking 1 Fortran step ---")
diag_f, out_f = dycore_f(state_f, timestep=timestep)
print("--- Taking 1 JAX step ---")
diag_j, out_j = dycore_j(state_j, timestep=timestep)

# ── Compare outputs ─────────────────────────────────────────────────────
print("\n=== Output comparison after 1 step ===")
print(
    f"{'field':>30s}  {'max|F|':>12s}  {'max|J|':>12s}  {'max|diff|':>12s}  {'max|diff|/max|F|':>16s}"
)
print("-" * 100)

results = {}
for name in fields:
    f_arr = np.asarray(out_f[name])
    j_arr = np.asarray(out_j[name])
    if np.iscomplexobj(j_arr):
        j_arr = j_arr.real
    diff = np.abs(f_arr - j_arr)
    max_f = np.abs(f_arr).max()
    max_j = np.abs(j_arr).max()
    max_diff = diff.max()
    rel = max_diff / max_f if max_f > 0 else 0.0
    results[name] = {
        "f_arr": f_arr,
        "j_arr": j_arr,
        "diff": diff,
        "max_diff": max_diff,
        "rel": rel,
    }
    print(f"  {name:30s}  {max_f:12.6e}  {max_j:12.6e}  {max_diff:12.6e}  {rel:16.6e}")

# ── Detailed field-by-field analysis ─────────────────────────────────────
print("\n=== Detailed per-level analysis ===")
for name in ["eastward_wind", "northward_wind", "air_temperature"]:
    f_arr = results[name]["f_arr"]
    j_arr = results[name]["j_arr"]
    diff = results[name]["diff"]
    print(f"\n  {name}:")
    if f_arr.ndim == 3:
        for k in range(f_arr.shape[0]):
            md = diff[k].max()
            mf = np.abs(f_arr[k]).max()
            rel = md / mf if mf > 0 else 0.0
            flag = " <<<" if rel > 1e-3 else ""
            print(
                f"    level {k:2d}: max|F|={mf:12.6e}  max|diff|={md:12.6e}  rel={rel:12.6e}{flag}"
            )
    else:
        print(f"    max|diff|={diff.max():.6e}")

# ── Change analysis: compare Δ(field) = output - input ──────────────────
print("\n=== Change (tendency) comparison:  Δfield = output - input ===")
print(
    f"{'field':>30s}  {'max|ΔF_fort|':>14s}  {'max|ΔF_jax|':>14s}  {'ratio J/F':>12s}  {'max|Δdiff|':>14s}"
)
print("-" * 100)

for name in fields:
    f_in = np.asarray(state_f[name])
    j_in = np.asarray(state_j[name])
    f_out = results[name]["f_arr"]
    j_out = results[name]["j_arr"]

    delta_f = f_out - f_in
    delta_j = j_out - j_in

    max_df = np.abs(delta_f).max()
    max_dj = np.abs(delta_j).max()
    ratio = max_dj / max_df if max_df > 0 else float("inf")
    delta_diff = np.abs(delta_f - delta_j).max()

    print(
        f"  {name:30s}  {max_df:14.6e}  {max_dj:14.6e}  {ratio:12.4f}  {delta_diff:14.6e}"
    )

# ── Zonal mean analysis ─────────────────────────────────────────────────
print("\n=== Zonal-mean tendency comparison ===")
for name in ["eastward_wind", "northward_wind", "air_temperature"]:
    f_in = np.asarray(state_f[name])
    f_out = results[name]["f_arr"]
    j_out = results[name]["j_arr"]

    delta_f_zm = (f_out - f_in).mean(axis=-1)
    delta_j_zm = (j_out - f_in).mean(axis=-1)

    print(f"\n  {name}  (zonal mean Δ):")
    print(f"    Fortran max|zm Δ|: {np.abs(delta_f_zm).max():.6e}")
    print(f"    JAX     max|zm Δ|: {np.abs(delta_j_zm).max():.6e}")
    ratio = np.abs(delta_j_zm).max() / max(np.abs(delta_f_zm).max(), 1e-30)
    print(f"    ratio:              {ratio:.4f}")

# ── Surface pressure tendency ────────────────────────────────────────────
print("\n=== Surface pressure tendency ===")
ps_f_in = np.asarray(state_f["surface_air_pressure"])
ps_f_out = results["surface_air_pressure"]["f_arr"]
ps_j_out = results["surface_air_pressure"]["j_arr"]

dps_f = ps_f_out - ps_f_in
dps_j = ps_j_out - ps_f_in

print(f"  Fortran Δps:  min={dps_f.min():.4f}  max={dps_f.max():.4f}  Pa")
print(f"  JAX     Δps:  min={dps_j.min():.4f}  max={dps_j.max():.4f}  Pa")
print(f"  Fortran Δps rms: {np.sqrt((dps_f**2).mean()):.4f} Pa")
print(f"  JAX     Δps rms: {np.sqrt((dps_j**2).mean()):.4f} Pa")
print(
    f"  ratio rms(JAX)/rms(F): {np.sqrt((dps_j**2).mean()) / max(np.sqrt((dps_f**2).mean()), 1e-30):.4f}"
)

# ── Run a few more steps to see divergence rate ─────────────────────────
print("\n=== Multi-step divergence tracking (10 steps) ===")
print(
    f"{'step':>4s}  {'PS_F min':>10s}  {'PS_F max':>10s}  {'PS_J min':>10s}  {'PS_J max':>10s}  {'max|u_diff|':>12s}  {'max|T_diff|':>12s}"
)
print("-" * 85)

state_f.update(out_f)
state_f["time"] += timestep
state_j.update(out_j)
state_j["time"] += timestep

for step in range(2, 12):
    diag_f2, out_f2 = dycore_f(state_f, timestep=timestep)
    diag_j2, out_j2 = dycore_j(state_j, timestep=timestep)

    state_f.update(out_f2)
    state_f["time"] += timestep
    state_j.update(out_j2)
    state_j["time"] += timestep

    ps_f = np.asarray(state_f["surface_air_pressure"])
    ps_j = np.asarray(state_j["surface_air_pressure"])
    if np.iscomplexobj(ps_j):
        ps_j = ps_j.real

    u_f = np.asarray(state_f["eastward_wind"])
    u_j = np.asarray(state_j["eastward_wind"])
    if np.iscomplexobj(u_j):
        u_j = u_j.real
    t_f = np.asarray(state_f["air_temperature"])
    t_j = np.asarray(state_j["air_temperature"])
    if np.iscomplexobj(t_j):
        t_j = t_j.real

    print(
        f"  {step:3d}  "
        f"{ps_f.min() / 100:10.2f}  {ps_f.max() / 100:10.2f}  "
        f"{ps_j.min() / 100:10.2f}  {ps_j.max() / 100:10.2f}  "
        f"{np.abs(u_f - u_j).max():12.6e}  "
        f"{np.abs(t_f - t_j).max():12.6e}"
    )

    if np.isnan(ps_j).any() or ps_j.max() / 100 > 1200:
        print("  JAX BLOWUP detected, stopping.")
        break

print("\nDone.")
