import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

from datetime import timedelta

import climt
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from sympl import set_constant

from gfs_dynamical_core.component_jax import GFSDynamicsJAX
from gfs_dynamical_core.jax.dynamics import (
    DynamicsConfig,
    compute_pressure_diagnostics,
    compute_vertical_velocities,
    full_dynamics_step,
)
from gfs_dynamical_core.jax.states import GridGradients, GridState
from gfs_dynamical_core.jax.transforms import (
    TransformConfig,
    grid_to_spectral,
    grid_to_spectral_tendencies,
    spectral_to_grid,
)

# ── Setup ────────────────────────────────────────────────────────────────
set_constant("reference_air_pressure", value=1e5, units="Pa")

L = 64
n_lon = 2 * L - 1
n_lat = L
n_lev = 20

print(f"Grid: {n_lon}x{n_lat}x{n_lev}  (L={L})")
grid = climt.get_grid(nx=n_lon, ny=n_lat, nz=n_lev)

dycore = GFSDynamicsJAX()
dcmip = climt.DcmipInitialConditions(add_perturbation=True)

my_state = climt.get_default_state([dycore], grid_state=grid)
out = dcmip(my_state)
my_state.update(out)

# ── Take a single timestep to force initialisation of internal configs ──
timestep = timedelta(minutes=5)
diag, output = dycore(my_state, timestep=timestep)

# Grab configs that were lazily built inside the component
dyn_config = dycore.dyn_config
trans_config = dycore.trans_config
stepper_config = dycore.stepper_config
latitudes = dycore._latitudes
phis_grads = dycore._phis_grads

# ── Build the initial grid / spectral state manually ─────────────────────
u0 = jnp.array(my_state["eastward_wind"])
v0 = jnp.array(my_state["northward_wind"])
temp0 = jnp.array(my_state["air_temperature"])
ps0 = jnp.array(my_state["surface_air_pressure"])
q0 = jnp.array(my_state["specific_humidity"])

grid_orig = GridState(
    u=u0,
    v=v0,
    temperature=temp0,
    vorticity=jnp.zeros_like(u0),
    divergence=jnp.zeros_like(u0),
    log_surface_pressure=jnp.log(ps0),
    tracers=jnp.stack([q0], axis=0),
)

spec0 = grid_to_spectral(grid_orig, trans_config)
grid0_raw, grads0_raw = spectral_to_grid(spec0, trans_config)

# Strip imaginary parts from the spectral roundtrip (scalar inverse transforms
# return tiny imaginary residuals ~1e-13 that propagate into dynamics and cause
# float() conversion errors).
grid0 = GridState(
    u=grid0_raw.u.real,
    v=grid0_raw.v.real,
    temperature=grid0_raw.temperature.real,
    vorticity=grid0_raw.vorticity.real,
    divergence=grid0_raw.divergence.real,
    log_surface_pressure=grid0_raw.log_surface_pressure.real,
    tracers=grid0_raw.tracers.real,
)
grads0 = GridGradients(
    d_log_ps_d_lambda=grads0_raw.d_log_ps_d_lambda.real,
    d_log_ps_d_phi=grads0_raw.d_log_ps_d_phi.real,
    d_t_d_lambda=grads0_raw.d_t_d_lambda.real,
    d_t_d_phi=grads0_raw.d_t_d_phi.real,
    d_tracers_d_lambda=grads0_raw.d_tracers_d_lambda.real,
    d_tracers_d_phi=grads0_raw.d_tracers_d_phi.real,
)

# ── Diagnose initial state ───────────────────────────────────────────────
print("\n=== Initial state (after spectral roundtrip) ===")
print(f"  u     min/max: {float(grid0.u.min()):.6e}  {float(grid0.u.max()):.6e}")
print(f"  v     min/max: {float(grid0.v.min()):.6e}  {float(grid0.v.max()):.6e}")
print(
    f"  T     min/max: {float(grid0.temperature.min()):.4f}  {float(grid0.temperature.max()):.4f}"
)
print(
    f"  lnps  min/max: {float(grid0.log_surface_pressure.min()):.6f}  {float(grid0.log_surface_pressure.max()):.6f}"
)
print(
    f"  vort  min/max: {float(grid0.vorticity.min()):.6e}  {float(grid0.vorticity.max()):.6e}"
)
print(
    f"  div   min/max: {float(grid0.divergence.min()):.6e}  {float(grid0.divergence.max()):.6e}"
)
print(
    f"  q     min/max: {float(grid0.tracers.min()):.6e}  {float(grid0.tracers.max()):.6e}"
)

# ── Pressure diagnostics ────────────────────────────────────────────────
press_diag = compute_pressure_diagnostics(grid0.log_surface_pressure, dyn_config)
print(f"\n=== Pressure diagnostics ===")
print(
    f"  ps    min/max: {float(press_diag.ps.min()) / 100:.2f}  {float(press_diag.ps.max()) / 100:.2f} hPa"
)
print(
    f"  dp    min/max: {float(press_diag.dp.min()):.2f}  {float(press_diag.dp.max()):.2f} Pa"
)
print(f"  dp<0? {bool((press_diag.dp < 0).any())}")
print(
    f"  pk    min/max: {float(press_diag.pk.min()):.4f}  {float(press_diag.pk.max()):.4f}"
)
print(
    f"  alfa  min/max: {float(press_diag.alfa.min()):.6e}  {float(press_diag.alfa.max()):.6e}"
)
print(
    f"  rlnp  min/max: {float(press_diag.rlnp.min()):.6e}  {float(press_diag.rlnp.max()):.6e}"
)

# ── Vertical velocities ─────────────────────────────────────────────────
vvels = compute_vertical_velocities(grid0, grads0, press_diag, dyn_config)
print(f"\n=== Vertical velocities ===")
print(
    f"  omega   min/max: {float(vvels.omega.min().real):.6e}  {float(vvels.omega.max().real):.6e}"
)
print(
    f"  etadot  min/max: {float(vvels.etadot.min().real):.6e}  {float(vvels.etadot.max().real):.6e}"
)
print(
    f"  dlnpsdt min/max: {float(vvels.d_log_ps_d_t.min().real):.6e}  {float(vvels.d_log_ps_d_t.max().real):.6e}"
)
print(
    f"  etadot[0]  (sfc) min/max: {float(vvels.etadot[0].min().real):.6e}  {float(vvels.etadot[0].max().real):.6e}"
)
print(
    f"  etadot[-1] (toa) min/max: {float(vvels.etadot[-1].min().real):.6e}  {float(vvels.etadot[-1].max().real):.6e}"
)

# ── Full grid-space tendencies ──────────────────────────────────────────
grid_tends = full_dynamics_step(grid0, grads0, phis_grads, dyn_config, latitudes)
print(f"\n=== Grid-space tendencies ===")
print(
    f"  u_flux      min/max: {float(grid_tends.u_flux.min()):.6e}  {float(grid_tends.u_flux.max()):.6e}"
)
print(
    f"  v_flux      min/max: {float(grid_tends.v_flux.min()):.6e}  {float(grid_tends.v_flux.max()):.6e}"
)
print(
    f"  temp_tend   min/max: {float(grid_tends.temp_tend.min()):.6e}  {float(grid_tends.temp_tend.max()):.6e}"
)
print(
    f"  log_ps_tend min/max: {float(grid_tends.log_ps_tend.min()):.6e}  {float(grid_tends.log_ps_tend.max()):.6e}"
)
print(
    f"  KE          min/max: {float(grid_tends.kinetic_energy.min()):.6e}  {float(grid_tends.kinetic_energy.max()):.6e}"
)

# ── Spectral tendencies ─────────────────────────────────────────────────
spec_tends = grid_to_spectral_tendencies(grid_tends, trans_config)
print(f"\n=== Spectral tendencies ===")
print(
    f"  d_vort/dt  min/max: {float(spec_tends.d_vorticity_d_t.real.min()):.6e}  {float(spec_tends.d_vorticity_d_t.real.max()):.6e}"
)
print(
    f"  d_div/dt   min/max: {float(spec_tends.d_divergence_d_t.real.min()):.6e}  {float(spec_tends.d_divergence_d_t.real.max()):.6e}"
)
print(
    f"  d_T/dt     min/max: {float(spec_tends.d_temperature_d_t.real.min()):.6e}  {float(spec_tends.d_temperature_d_t.real.max()):.6e}"
)
print(
    f"  d_lnps/dt  min/max: {float(spec_tends.d_log_surface_pressure_d_t.real.min()):.6e}  {float(spec_tends.d_log_surface_pressure_d_t.real.max()):.6e}"
)

# ── Tendency magnitudes vs state magnitudes ──────────────────────────────
dt = timestep.total_seconds()
print(f"\n=== Δ(state) after one step (dt={dt}s) vs initial state ===")

# Convert spectral tendencies back to grid to see physical tendencies
vort_tend_grid = spec_tends.d_vorticity_d_t
div_tend_grid = spec_tends.d_divergence_d_t

import s2fft

# Reconstruct what the u,v change looks like
# Just check the output state from the actual step
u1 = jnp.array(output["eastward_wind"])
v1 = jnp.array(output["northward_wind"])
t1 = jnp.array(output["air_temperature"])
ps1 = jnp.array(output["surface_air_pressure"])

du = u1 - u0
dv = v1 - v0
dT = t1 - temp0
dps = ps1 - ps0

print(
    f"  Δu     min/max: {float(du.min()):.6e}  {float(du.max()):.6e}   (u0 range: {float(u0.max() - u0.min()):.2f})"
)
print(
    f"  Δv     min/max: {float(dv.min()):.6e}  {float(dv.max()):.6e}   (v0 range: {float(v0.max() - v0.min()):.2f})"
)
print(
    f"  ΔT     min/max: {float(dT.min()):.6e}  {float(dT.max()):.6e}   (T0 range: {float(temp0.max() - temp0.min()):.2f})"
)
print(
    f"  Δps    min/max: {float(dps.min()):.4f}  {float(dps.max()):.4f}  Pa   (ps0 range: {float(ps0.max() - ps0.min()):.2f})"
)

# ── Zonal-mean wind tendency (key diagnostic for vort/div swap) ──────────
lat_np = np.array(latitudes)
du_zonal = np.array(du.mean(axis=-1))  # (n_lev, n_lat)
dv_zonal = np.array(dv.mean(axis=-1))

print(f"\n=== Zonal-mean Δu/Δv (should be ~0 for axisym initial cond) ===")
print(f"  zonal-mean Δu  max abs: {np.abs(du_zonal).max():.6e}")
print(f"  zonal-mean Δv  max abs: {np.abs(dv_zonal).max():.6e}")

# ── Check energy: is total energy growing? ──────────────────────────────
# Crude global energy metric
gauss_w = dycore._gauss_weights
w = np.array(gauss_w)[:, None]  # (n_lat, 1)


def global_mean_ke(u, v, dp):
    """Column-integrated KE per unit area."""
    ke = 0.5 * (u**2 + v**2)
    ke_int = np.sum(np.array(ke * dp), axis=0)  # (n_lat, n_lon)
    return np.sum(w * np.array(ke_int)) / n_lon


def global_mean_ie(T, dp):
    """Column-integrated internal energy per unit area."""
    cp = float(dyn_config.cp)
    ie_int = np.sum(np.array(T * dp) * cp, axis=0)
    return np.sum(w * np.array(ie_int)) / n_lon


ke0 = global_mean_ke(u0, v0, press_diag.dp)
ie0 = global_mean_ie(temp0, press_diag.dp)

press1 = compute_pressure_diagnostics(jnp.log(ps1), dyn_config)
ke1 = global_mean_ke(u1, v1, press1.dp)
ie1 = global_mean_ie(t1, press1.dp)

print(f"\n=== Global energy (per unit area) ===")
print(f"  KE0:  {ke0:.6e}    KE1:  {ke1:.6e}    ΔKE/KE0: {(ke1 - ke0) / ke0:.6e}")
print(f"  IE0:  {ie0:.6e}    IE1:  {ie1:.6e}    ΔIE/IE0: {(ie1 - ie0) / ie0:.6e}")
print(
    f"  TE0:  {ke0 + ie0:.6e}    TE1:  {ke1 + ie1:.6e}    ΔTE/TE0: {((ke1 + ie1) - (ke0 + ie0)) / (ke0 + ie0):.6e}"
)

# ── Run a few steps and track PS extrema ─────────────────────────────────
print(f"\n=== Multi-step stability check (dt={int(dt)}s = {int(dt / 60)} min) ===")

state_run = dict(my_state)
ps_hist = []
ke_hist = []

for i in range(60):
    ps_arr = np.array(state_run["surface_air_pressure"])
    ps_min, ps_max = ps_arr.min() / 100, ps_arr.max() / 100
    ps_hist.append((ps_min, ps_max))

    if ps_max > 1200 or ps_min < 800 or np.isnan(ps_max):
        print(f"  Step {i:3d}: BLOWUP  PS range: {ps_min:.1f} - {ps_max:.1f} hPa")
        break

    if i % 6 == 0:
        # Compute KE
        u_r = jnp.array(state_run["eastward_wind"])
        v_r = jnp.array(state_run["northward_wind"])
        t_r = jnp.array(state_run["air_temperature"])
        ps_r = jnp.array(state_run["surface_air_pressure"])
        pd = compute_pressure_diagnostics(jnp.log(ps_r), dyn_config)
        ke_r = global_mean_ke(u_r, v_r, pd.dp)
        ie_r = global_mean_ie(t_r, pd.dp)
        ke_hist.append(ke_r + ie_r)
        print(
            f"  Step {i:3d} ({i * int(dt / 60):4d} min): "
            f"PS {ps_min:.2f} - {ps_max:.2f} hPa,  "
            f"TE={ke_r + ie_r:.6e}"
        )

    diag, out_r = dycore(state_run, timestep=timestep)
    state_run.update(out_r)
    state_run["time"] += timestep

# ── Final summary ────────────────────────────────────────────────────────
ps_mins = [p[0] for p in ps_hist]
ps_maxs = [p[1] for p in ps_hist]
steps = list(range(len(ps_hist)))
hours = [s * int(dt / 60) / 60 for s in steps]

print(f"\n=== PS range evolution ===")
print(f"  Start: {ps_mins[0]:.2f} - {ps_maxs[0]:.2f} hPa")
print(f"  End:   {ps_mins[-1]:.2f} - {ps_maxs[-1]:.2f} hPa")
print(
    f"  Spread growth: {(ps_maxs[-1] - ps_mins[-1]) - (ps_maxs[0] - ps_mins[0]):.2f} hPa over {hours[-1]:.1f} hours"
)
if len(ps_hist) > 1:
    spread_rate = ((ps_maxs[-1] - ps_mins[-1]) - (ps_maxs[0] - ps_mins[0])) / hours[-1]
    print(f"  Spread rate: {spread_rate:.2f} hPa/hour")

# ── Plots ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# PS range over time
ax = axes[0, 0]
ax.plot(hours, ps_mins, label="PS min")
ax.plot(hours, ps_maxs, label="PS max")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Surface pressure (hPa)")
ax.set_title("Surface pressure range")
ax.legend()
ax.grid(True)

# PS spread over time
ax = axes[0, 1]
spreads = [mx - mn for mn, mx in zip(ps_mins, ps_maxs)]
ax.plot(hours, spreads)
ax.set_xlabel("Time (hours)")
ax.set_ylabel("PS spread (hPa)")
ax.set_title("Surface pressure spread")
ax.grid(True)

# Δu after first step
ax = axes[1, 0]
c = ax.contourf(
    np.degrees(np.array(latitudes)),
    np.arange(n_lev),
    np.array(du.mean(axis=-1)),
    cmap="RdBu_r",
    levels=21,
)
fig.colorbar(c, ax=ax)
ax.set_xlabel("Latitude (deg)")
ax.set_ylabel("Level")
ax.set_title("Zonal-mean Δu after 1 step")
ax.invert_yaxis()

# ΔT after first step
ax = axes[1, 1]
c = ax.contourf(
    np.degrees(np.array(latitudes)),
    np.arange(n_lev),
    np.array(dT.mean(axis=-1)),
    cmap="RdBu_r",
    levels=21,
)
fig.colorbar(c, ax=ax)
ax.set_xlabel("Latitude (deg)")
ax.set_ylabel("Level")
ax.set_title("Zonal-mean ΔT after 1 step")
ax.invert_yaxis()

# Surface pressure at initial time
ax = axes[2, 0]
lons_deg = np.linspace(0, 360, n_lon, endpoint=False)
lats_deg = np.degrees(np.array(latitudes))
c = ax.contourf(lons_deg, lats_deg, np.array(ps0) / 100, levels=21, cmap="viridis")
fig.colorbar(c, ax=ax, label="hPa")
ax.set_title("Initial surface pressure")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Surface pressure after run
ax = axes[2, 1]
ps_final = np.array(state_run["surface_air_pressure"])
c = ax.contourf(lons_deg, lats_deg, ps_final / 100, levels=21, cmap="viridis")
fig.colorbar(c, ax=ax, label="hPa")
ax.set_title(f"Surface pressure after {hours[-1]:.1f} hours")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.savefig("baroclinic_wave_diag.png", dpi=150)
print("\nPlot saved to baroclinic_wave_diag.png")
