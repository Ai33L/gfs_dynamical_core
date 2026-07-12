"""Held-Suarez (1994) climatology run for either dycore.

Usage: python heldsuarez_run.py {fortran|jax} [n_days] [spinup_days]

Both dycores use the IDENTICAL climt.HeldSuarez forcing component, coupled
the SAME way — as a tendency_component_list on a component-driven core:
- Fortran: GFSDynamicalCore(tendency_component_list=[hs])
- JAX:     GFSDynamicsJAX(tendency_component_list=[hs])
Each core runs the forcing internally and applies the time-split spectral
adjustment; the driver loop is mode-agnostic.

Accumulates zonal-mean u, T (and pressure) every 6 h after spinup and
writes hs_zonal_mean_{mode}.npz in the cwd.
"""

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

import glob
import time
from datetime import timedelta

import climt
import numpy as np
from sympl import set_constant

MODE = sys.argv[1]
N_DAYS = float(sys.argv[2]) if len(sys.argv) > 2 else 400.0
SPINUP_DAYS = float(sys.argv[3]) if len(sys.argv) > 3 else 200.0

L = 64
N_LEV = 20
DT_MIN = 10.0
STEPS_PER_DAY = int(24 * 60 / DT_MIN)
N_STEPS = int(N_DAYS * STEPS_PER_DAY)
SPINUP_STEPS = int(SPINUP_DAYS * STEPS_PER_DAY)
SAMPLE_EVERY = 36  # 6 h

set_constant("reference_air_pressure", value=1e5, units="Pa")
grid = climt.get_grid(nx=2 * L - 1, ny=L, nz=N_LEV)
timestep = timedelta(minutes=DT_MIN)

hs = climt.HeldSuarez()

if MODE == "fortran":
    from gfs_dynamical_core import GFSDynamicalCore

    dycore = GFSDynamicalCore(tendency_component_list=[hs])
else:
    from gfs_dynamical_core.component_jax import GFSDynamicsJAX

    dycore = GFSDynamicsJAX(tendency_component_list=[hs])

state = climt.get_default_state([dycore], grid_state=grid)

# Identical random thermal perturbation to break zonal symmetry (same seed).
rng = np.random.default_rng(7)
state["air_temperature"].values += 0.5 * rng.standard_normal(
    state["air_temperature"].values.shape
)

print(f"[{MODE}] Held-Suarez: {N_DAYS:.0f} days, dt={DT_MIN:.0f} min,"
      f" spinup {SPINUP_DAYS:.0f} d, {N_STEPS} steps", flush=True)

u_sum = np.zeros((N_LEV, L))
T_sum = np.zeros((N_LEV, L))
p_sum = np.zeros((N_LEV, L))
n_samples = 0

t0 = time.time()
for i in range(1, N_STEPS + 1):
    _, out = dycore(state, timestep=timestep)
    state.update(out)
    state["time"] += timestep

    if i % 100 == 0 and MODE == "fortran":
        for f in glob.glob("debug_data/fortran_step_*.bin"):
            os.remove(f)

    if i > SPINUP_STEPS and i % SAMPLE_EVERY == 0:
        u = np.asarray(state["eastward_wind"].values)
        T = np.asarray(state["air_temperature"].values)
        p = np.asarray(state["air_pressure"].values)
        u_sum += u.mean(axis=2)
        T_sum += T.mean(axis=2)
        p_sum += p.mean(axis=2)
        n_samples += 1

    if i % 1440 == 0:  # every 10 days
        ps = np.asarray(state["surface_air_pressure"].values)
        u = np.asarray(state["eastward_wind"].values)
        print(f"[{MODE}] day {i / STEPS_PER_DAY:6.1f} | ps [{ps.min() / 100:7.2f},"
              f" {ps.max() / 100:8.2f}] | |u|max {np.abs(u).max():6.1f} |"
              f" samples {n_samples} | {time.time() - t0:7.0f}s", flush=True)
        np.savez("hs_zonal_mean_%s.npz" % MODE, u_sum=u_sum, T_sum=T_sum,
                 p_sum=p_sum, n=n_samples, day=i / STEPS_PER_DAY)

    ps = np.asarray(state["surface_air_pressure"].values)
    u = np.asarray(state["eastward_wind"].values)
    if np.isnan(ps).any() or ps.max() > 2e5 or ps.min() < 3e4 or np.abs(u).max() > 500:
        print(f"[{MODE}] UNSTABLE at step {i} (day {i / STEPS_PER_DAY:.2f})", flush=True)
        sys.exit(1)

np.savez("hs_zonal_mean_%s.npz" % MODE, u_sum=u_sum, T_sum=T_sum,
         p_sum=p_sum, n=n_samples, day=N_DAYS)
print(f"[{MODE}] DONE: {n_samples} samples, {time.time() - t0:.0f}s", flush=True)
