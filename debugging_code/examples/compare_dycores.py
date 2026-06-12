"""
Side-by-side comparison of Fortran and JAX dynamical cores.

Runs both from identical JW06 ICs (NO perturbation) and compares
the sympl output state directly in memory after each timestep.
No binary dumps — just direct numpy array comparison.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

from datetime import timedelta

import climt
import numpy as np
from sympl import set_constant

from gfs_dynamical_core import GFSDynamicalCore
from gfs_dynamical_core.component_jax import GFSDynamicsJAX

# ── Grid parameters ──────────────────────────────────────────────────────
L = 64
N_LON = 2 * L - 1  # 127
N_LAT = L  # 64
N_LEV = 20

set_constant("reference_air_pressure", value=1e5, units="Pa")

FIELDS = [
    ("eastward_wind", "u", "m/s"),
    ("northward_wind", "v", "m/s"),
    ("air_temperature", "T", "K"),
    ("surface_air_pressure", "ps", "Pa"),
    ("specific_humidity", "q", "kg/kg"),
]


def compare_states(state_f, state_j, step, dt_minutes):
    """Compare Fortran and JAX output states field by field."""
    hour = step * dt_minutes / 60.0
    print(f"\n{'='*80}")
    print(f"Step {step:3d}  (hour {hour:.1f})")
    print(f"{'='*80}")
    print(
        f"  {'field':6s}  {'max|F|':>12s}  {'max|J|':>12s}  "
        f"{'max|diff|':>12s}  {'rel_err':>12s}  {'rms_diff':>12s}"
    )

    all_ok = True
    for key, short, unit in FIELDS:
        f_arr = np.asarray(state_f[key])
        j_arr = np.asarray(state_j[key])
        if np.iscomplexobj(j_arr):
            j_arr = j_arr.real

        diff = np.abs(f_arr - j_arr)
        max_f = np.abs(f_arr).max()
        max_j = np.abs(j_arr).max()
        max_diff = diff.max()
        rms_diff = np.sqrt((diff**2).mean())
        rel = max_diff / max_f if max_f > 0 else 0.0

        flag = " <<<" if rel > 1e-3 else ""
        if rel > 1e-3:
            all_ok = False

        print(
            f"  {short:6s}  {max_f:12.6e}  {max_j:12.6e}  "
            f"{max_diff:12.6e}  {rel:12.6e}  {rms_diff:12.6e}{flag}"
        )

    # Also show tendency (change from IC) comparison for key fields
    return all_ok


def main():
    print(f"Grid: {N_LON}x{N_LAT}x{N_LEV}  L={L}")
    grid = climt.get_grid(nx=N_LON, ny=N_LAT, nz=N_LEV)
    timestep = timedelta(minutes=5)

    # ── JW06 initial conditions, NO perturbation ─────────────────────────
    dcmip = climt.DcmipInitialConditions(add_perturbation=False)

    # ── Initialize Fortran dycore ────────────────────────────────────────
    print("\nInitializing Fortran dycore...")
    dycore_f = GFSDynamicalCore()
    state_f = climt.get_default_state([dycore_f], grid_state=grid)
    out = dcmip(state_f)
    state_f.update(out)

    # ── Initialize JAX dycore ────────────────────────────────────────────
    print("Initializing JAX dycore...")
    dycore_j = GFSDynamicsJAX()
    state_j = climt.get_default_state([dycore_j], grid_state=grid)
    out = dcmip(state_j)
    state_j.update(out)

    # ── Verify ICs match ─────────────────────────────────────────────────
    print("\n--- Initial condition comparison ---")
    for key, short, unit in FIELDS:
        diff = np.abs(np.asarray(state_f[key]) - np.asarray(state_j[key])).max()
        print(f"  {short:6s}  max|diff| = {diff:.6e}")

    # ── Step loop ────────────────────────────────────────────────────────
    n_steps = 50  # ~4 hours at 5 min dt
    dt_minutes = 5

    print(f"\nRunning {n_steps} steps ({n_steps * dt_minutes / 60:.1f} hours)...")

    for step in range(1, n_steps + 1):
        diag_f, out_f = dycore_f(state_f, timestep=timestep)
        state_f.update(out_f)
        state_f["time"] += timestep

        diag_j, out_j = dycore_j(state_j, timestep=timestep)
        state_j.update(out_j)
        state_j["time"] += timestep

        # Compare at every step for first 5, then every 10
        if step <= 5 or step % 10 == 0:
            ok = compare_states(state_f, state_j, step, dt_minutes)
            if not ok:
                # Show per-level breakdown for the worst field
                for key, short, unit in FIELDS:
                    f_arr = np.asarray(state_f[key])
                    j_arr = np.asarray(state_j[key])
                    if np.iscomplexobj(j_arr):
                        j_arr = j_arr.real
                    if f_arr.ndim == 3:
                        rel = np.abs(f_arr - j_arr).max() / max(
                            np.abs(f_arr).max(), 1e-30
                        )
                        if rel > 1e-3:
                            print(f"\n  {short} per-level breakdown:")
                            for k in range(f_arr.shape[0]):
                                d = np.abs(f_arr[k] - j_arr[k]).max()
                                r = d / max(np.abs(f_arr[k]).max(), 1e-30)
                                flag = " <<<" if r > 1e-3 else ""
                                print(
                                    f"    lev {k:2d}: max|diff|={d:12.6e}  rel={r:12.6e}{flag}"
                                )

        # Bail if NaN
        ps_j = np.asarray(state_j["surface_air_pressure"])
        if np.isnan(ps_j).any() or ps_j.max() > 2e5:
            print(f"\n*** JAX blew up at step {step}! ***")
            break

        ps_f = np.asarray(state_f["surface_air_pressure"])
        if np.isnan(ps_f).any() or ps_f.max() > 2e5:
            print(f"\n*** Fortran blew up at step {step}! ***")
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
