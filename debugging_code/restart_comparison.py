"""Restart experiment: initialize BOTH dycores from Fortran's healthy
step-2000 state (day 6.94, finite amplitude) and integrate side by side.

JAX runs EXPLICIT mode to match Fortran's scheme exactly, so identical
initial states + identical per-step maps => trajectories stay glued
(roundoff-level divergence only). If the JAX SH instability (m=11-15,
l=30-38, 5.5h e-fold) is real, divergence appears within ~100 steps.
"""

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

import time
from datetime import timedelta

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import climt
import jax.numpy as jnp
import s2fft
from sympl import set_constant

from test_finite_amplitude_tendencies import load_step, packed_levels_to_rect
from spectral_converter import shtns_packed_to_s2fft_rect

import gfs_dynamical_core.component_jax as cj
from gfs_dynamical_core import GFSDynamicalCore

# Patch JAX stepper to explicit mode (matches Fortran's default).
_orig_sc = cj.StepperConfig


def _patched_sc(*args, **kwargs):
    kwargs["explicit"] = True
    return _orig_sc(*args, **kwargs)


cj.StepperConfig = _patched_sc

L = 64
N_LEV = 20
RESTART_STEP = 2000
DUMP_DIR = "/tmp/gfs_compare_scratch/debug_data"

set_constant("reference_air_pressure", value=1e5, units="Pa")

# ── Synthesize grid fields from the Fortran spectral dump ────────────────
st = load_step(os.path.join(DUMP_DIR, f"fortran_step_{RESTART_STEP:05d}.bin"))

vort = jnp.asarray(packed_levels_to_rect(st["vrt"]))
div = jnp.asarray(packed_levels_to_rect(st["div"]))
temp_lm = jnp.asarray(packed_levels_to_rect(st["temp"]))
lnps_lm = jnp.asarray(shtns_packed_to_s2fft_rect(st["lnps"], 40, L))

l_arr = jnp.arange(L)
l_factor = jnp.sqrt(l_arr * (l_arr + 1))
inv_l = jnp.where(l_arr > 0, 1.0 / l_factor, 0.0)
radius = 6371000.0  # must match planetary_radius constant used by both


def synth(flm):
    return np.asarray(s2fft.inverse_jax(flm, L, sampling="gl").real)


u_g = np.empty((N_LEV, L, 2 * L - 1))
v_g = np.empty_like(u_g)
T_g = np.empty_like(u_g)
for k in range(N_LEV):
    F1 = inv_l[:, None] * (div[k] + 1j * vort[k]) * radius
    f_spin1 = s2fft.inverse_jax(F1, L, spin=1, sampling="gl")
    u_g[k] = np.asarray(f_spin1.imag)
    v_g[k] = np.asarray(-f_spin1.real)
    T_g[k] = synth(temp_lm[k])
ps_g = np.exp(synth(lnps_lm))

print(f"Restart state from Fortran step {RESTART_STEP}:")
print(f"  |u|max={np.abs(u_g).max():.2f}  T range [{T_g.min():.1f},{T_g.max():.1f}]"
      f"  ps range [{ps_g.min() / 100:.2f},{ps_g.max() / 100:.2f}] hPa", flush=True)

# ── Build states for both dycores ────────────────────────────────────────
grid = climt.get_grid(nx=2 * L - 1, ny=L, nz=N_LEV)
dcmip = climt.DcmipInitialConditions(add_perturbation=True)


from gfs_dynamical_core.jax.dynamics import compute_pressure_diagnostics

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tests.test_phase1_dynamics import _build_config_from_climt

_dyn_cfg = _build_config_from_climt(n_lev=N_LEV, n_lat=L, n_lon=2 * L - 1)
_pd = compute_pressure_diagnostics(jnp.log(jnp.asarray(ps_g)), _dyn_cfg)
p_mid = np.asarray(_pd.prs)  # (n_lev, lat, lon) BTU — matches Fortran's prs
p_int = np.asarray(_pd.pk)  # (n_lev+1, lat, lon) BTU — matches Fortran's pk


def make_state(dycore):
    state = climt.get_default_state([dycore], grid_state=grid)
    state.update(dcmip(state))  # sets surface_geopotential etc.
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

# ── Run side by side ─────────────────────────────────────────────────────
timestep = timedelta(minutes=5)
n_steps = 600
SH = slice(32, 64)

print(f"\nRunning {n_steps} steps from restart (JAX explicit)...", flush=True)
print(f"{'step':>5} {'day':>5} | {'max|du|':>10} {'max|dT|':>10} {'max|dps|':>10} |"
      f" {'SHeddy_F':>10} {'SHeddy_J':>10} | {'uFmax':>6} {'uJmax':>6}", flush=True)

t0 = time.time()
for i in range(1, n_steps + 1):
    _, out_f = dycore_f(state_f, timestep=timestep)
    state_f.update(out_f)
    state_f["time"] += timestep
    _, out_j = dycore_j(state_j, timestep=timestep)
    state_j.update(out_j)
    state_j["time"] += timestep

    if i % 10 == 0 or i <= 3:
        fu = np.asarray(state_f["eastward_wind"])
        ju = np.asarray(state_j["eastward_wind"])
        if np.iscomplexobj(ju):
            ju = ju.real
        fT = np.asarray(state_f["air_temperature"])
        jT = np.asarray(state_j["air_temperature"])
        if np.iscomplexobj(jT):
            jT = jT.real
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
            f" {np.abs(fu - ju).max():>10.3e} {np.abs(fT - jT).max():>10.3e}"
            f" {np.abs(fps - jps).max():>10.3e} |"
            f" {f_eddy:>10.3e} {j_eddy:>10.3e} |"
            f" {np.abs(fu).max():>6.1f} {np.abs(ju).max():>6.1f} | {time.time() - t0:5.0f}s",
            flush=True,
        )
        if np.isnan(jps).any() or jps.max() > 2e5 or np.abs(ju).max() > 400:
            print(f"JAX UNSTABLE at restart step {i}", flush=True)
            break
        if np.isnan(fps).any() or fps.max() > 2e5:
            print(f"FORTRAN UNSTABLE at restart step {i}", flush=True)
            break

print("done", flush=True)
