"""Cross-restart: initialize BOTH dycores from the JAX trajectory's
step-1800 state (day 6.25, SH eddies growing but |u| still sane) taken
from the compare_long snapshot (float32 grid fields; the implied ~1e-7
truncation noise is harmless per the perturbed-Fortran control).

Outcomes:
- Fortran stable, JAX blows: maps differ on JAX-trajectory states ->
  run the tendency comparison on this exact state to localize the term.
- Both blow: the JAX state already contains a doomed structure ->
  bisect to earlier snapshots.
- Both stable: the instability needs the JAX trajectory's full spectral
  state (snapshot grid fields lost something) -> rerun JAX with full
  state dumps.
"""

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

import time
from datetime import timedelta

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "tests"))

import climt
import jax.numpy as jnp
from sympl import set_constant

import gfs_dynamical_core.component_jax as cj
from gfs_dynamical_core import GFSDynamicalCore
from gfs_dynamical_core.jax.dynamics import compute_pressure_diagnostics
from tests.test_phase1_dynamics import _build_config_from_climt

SNAP = "/tmp/gfs_compare_scratch/snapshots/snap_01800.npz"
RESTART_STEP = 1800
L = 64
N_LEV = 20

# JAX explicit to match Fortran scheme
_orig_sc = cj.StepperConfig


def _patched_sc(*args, **kwargs):
    kwargs["explicit"] = True
    return _orig_sc(*args, **kwargs)


cj.StepperConfig = _patched_sc

set_constant("reference_air_pressure", value=1e5, units="Pa")

d = np.load(SNAP)
u_g = d["j_u"].astype(np.float64)
v_g = d["j_v"].astype(np.float64)
T_g = d["j_T"].astype(np.float64)
ps_g = d["j_ps"].astype(np.float64)

print(f"JAX step-{RESTART_STEP} state: |u|max={np.abs(u_g).max():.2f}"
      f"  ps [{ps_g.min() / 100:.2f},{ps_g.max() / 100:.2f}] hPa", flush=True)

grid = climt.get_grid(nx=2 * L - 1, ny=L, nz=N_LEV)
dcmip = climt.DcmipInitialConditions(add_perturbation=True)

_dyn_cfg = _build_config_from_climt(n_lev=N_LEV, n_lat=L, n_lon=2 * L - 1)
_pd = compute_pressure_diagnostics(jnp.log(jnp.asarray(ps_g)), _dyn_cfg)
p_mid = np.asarray(_pd.prs)
p_int = np.asarray(_pd.pk)


def make_state(dycore):
    state = climt.get_default_state([dycore], grid_state=grid)
    state.update(dcmip(state))
    state["eastward_wind"].values[:] = u_g
    state["northward_wind"].values[:] = v_g
    state["air_temperature"].values[:] = T_g
    state["surface_air_pressure"].values[:] = ps_g
    state["specific_humidity"].values[:] = 0.0
    state["air_pressure"].values[:] = p_mid
    state["air_pressure_on_interface_levels"].values[:] = p_int
    return state


dycore_f = GFSDynamicalCore()
state_f = make_state(dycore_f)
dycore_j = cj.GFSDynamicsJAX()
state_j = make_state(dycore_j)

timestep = timedelta(minutes=5)
n_steps = 700  # JAX originally blew ~326 steps after 1800
SH = slice(32, 64)

print(f"{'step':>5} {'day':>5} | {'max|du|':>10} | {'SHeddy_F':>10} {'SHeddy_J':>10} |"
      f" {'uFmax':>6} {'uJmax':>6} {'psFmin':>8} {'psJmin':>8}", flush=True)

t0 = time.time()
f_alive, j_alive = True, True
for i in range(1, n_steps + 1):
    if f_alive:
        _, out_f = dycore_f(state_f, timestep=timestep)
        state_f.update(out_f)
        state_f["time"] += timestep
    if j_alive:
        _, out_j = dycore_j(state_j, timestep=timestep)
        state_j.update(out_j)
        state_j["time"] += timestep

    if i % 10 == 0 or i <= 3:
        fu = np.asarray(state_f["eastward_wind"])
        ju = np.asarray(state_j["eastward_wind"])
        if np.iscomplexobj(ju):
            ju = ju.real
        fps = np.asarray(state_f["surface_air_pressure"])
        jps = np.asarray(state_j["surface_air_pressure"])
        if np.iscomplexobj(jps):
            jps = jps.real
        f_sh = fu[:, SH, :]
        j_sh = ju[:, SH, :]
        f_eddy = np.abs(f_sh - f_sh.mean(axis=2, keepdims=True)).max()
        j_eddy = np.abs(j_sh - j_sh.mean(axis=2, keepdims=True)).max()
        print(
            f"{i:>5} {RESTART_STEP * 5 / 1440 + i * 5 / 1440:>5.2f} |"
            f" {np.abs(fu - ju).max():>10.3e} |"
            f" {f_eddy:>10.3e} {j_eddy:>10.3e} |"
            f" {np.abs(fu).max():>6.1f} {np.abs(ju).max():>6.1f}"
            f" {fps.min() / 100:>8.2f} {jps.min() / 100:>8.2f} | {time.time() - t0:5.0f}s",
            flush=True,
        )
        if j_alive and (np.isnan(jps).any() or jps.max() > 2e5 or np.abs(ju).max() > 400):
            print(f"*** JAX UNSTABLE at restart step {i} ***", flush=True)
            j_alive = False
        if f_alive and (np.isnan(fps).any() or fps.max() > 2e5 or np.abs(fu).max() > 400):
            print(f"*** FORTRAN UNSTABLE at restart step {i} ***", flush=True)
            f_alive = False
        if not (f_alive or j_alive):
            break

print("done", flush=True)
