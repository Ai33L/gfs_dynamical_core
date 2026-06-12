"""Stability check: run the JAX dycore on DCMIP 4.1 baroclinic wave.

Purpose: the toa_pressure fix (D1.1, 2026-04-19) was never tested in a
full run -- all blow-up runs in the investigation log predate it.
Previous behavior: blow-up at step 2060 (day 7.15) with dt=5min, default
diffusion. This script reruns identical settings and reports whether the
blow-up persists.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

import sys
import time
from datetime import timedelta

import climt
import numpy as np
from sympl import set_constant

from gfs_dynamical_core.component_jax import GFSDynamicsJAX

set_constant("reference_air_pressure", value=1e5, units="Pa")

L = 64
n_lon = 2 * L - 1
n_lat = L
n_lev = 20

print(f"Grid: {n_lon}x{n_lat}x{n_lev} (L={L})", flush=True)
grid = climt.get_grid(nx=n_lon, ny=n_lat, nz=n_lev)

dycore = GFSDynamicsJAX()
dcmip = climt.DcmipInitialConditions(add_perturbation=True)

my_state = climt.get_default_state([dycore], grid_state=grid)
out = dcmip(my_state)
my_state.update(out)

timestep = timedelta(minutes=5)
n_steps = 3456  # 12 days at dt=5min (blow-up previously at step 2060)
print(f"dt={timestep}, steps={n_steps} (12 days)", flush=True)

t0 = time.time()
for i in range(n_steps):
    diag, output = dycore(my_state, timestep=timestep)
    my_state.update(output)
    my_state["time"] += timestep

    ps = my_state["surface_air_pressure"].values
    u = my_state["eastward_wind"].values
    T = my_state["air_temperature"].values
    ps_min, ps_max = np.min(ps), np.max(ps)
    u_max = np.max(np.abs(u))
    T_min, T_max = np.min(T), np.max(T)

    if (i + 1) % 50 == 0 or i == 0:
        elapsed = time.time() - t0
        day = (i + 1) * 5 / 60 / 24
        # Reality-symmetry violation of the cached spectral state — the
        # tangent-linear "ghost mode" that caused the day-7/8 blow-up.
        spec = dycore._spec_state
        viol = 0.0
        if spec is not None:
            L_ = spec.divergence.shape[-2]
            m = np.arange(1, L_)
            dv = np.asarray(spec.divergence)
            v = dv[..., L_ - 1 - m] - ((-1.0) ** m) * np.conj(dv[..., L_ - 1 + m])
            viol = float(np.abs(v).max())
        print(
            f"step {i + 1:5d} day {day:5.2f} | ps [{ps_min / 100:8.2f},"
            f" {ps_max / 100:8.2f}] hPa | |u|max {u_max:7.2f} m/s |"
            f" T [{T_min:6.1f}, {T_max:6.1f}] K | div-asym {viol:8.1e} |"
            f" {elapsed:6.0f}s",
            flush=True,
        )

    if ps_max > 2e5 or ps_min < 4e4 or np.isnan(ps_max) or u_max > 400:
        print(
            f"UNSTABLE at step {i + 1} (day {(i + 1) * 5 / 60 / 24:.2f}):"
            f" ps [{ps_min / 100:.1f}, {ps_max / 100:.1f}] hPa,"
            f" |u|max {u_max:.1f} m/s",
            flush=True,
        )
        sys.exit(1)

print("STABLE: completed 12 days without blow-up.", flush=True)
