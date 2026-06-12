"""
Compare Fortran debug binary dumps with JAX model state stage-by-stage.

The Fortran debug dumps are written by dump_intermediate() in run.f90:
  - ug          (nlons, nlats, nlevs)      float64
  - vg          (nlons, nlats, nlevs)      float64
  - virtempg    (nlons, nlats, nlevs)      float64
  - lnpsg       (nlons, nlats)             float64
  - tracerg     (nlons, nlats, nlevs, ntrac) float64
  - vrtspec     (ndimspec, nlevs)          complex128
  - divspec     (ndimspec, nlevs)          complex128
  - virtempspec (ndimspec, nlevs)          complex128
  - lnpsspec    (ndimspec)                 complex128

Fortran grid arrays use (lon, lat, lev) ordering (column-major), bottom-to-top
for physics fields (k=1 = surface). SHTNS latitudes go south-to-north.

s2fft / JAX uses (lat, lon) with lat going north-to-south (GL sampling).

This script:
  1. Re-runs the Fortran for 1 step to regenerate fresh debug dumps
  2. Takes 1 JAX step from the same initial conditions
  3. Loads the Fortran stage-0 dump (= initial state after spec->grid)
  4. Loads the Fortran stage-1 dump (= state after RK stage 1)
  5. Compares grid fields and spectral fields against JAX equivalents
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

import struct as pystruct
from datetime import timedelta
from pathlib import Path

import climt
import jax
import jax.numpy as jnp
import numpy as np
import s2fft
from sympl import set_constant

from gfs_dynamical_core import GFSDynamicalCore
from gfs_dynamical_core.component_jax import GFSDynamicsJAX
from gfs_dynamical_core.jax.dynamics import (
    DynamicsConfig,
    compute_pressure_diagnostics,
    compute_vertical_velocities,
    full_dynamics_step,
)
from gfs_dynamical_core.jax.states import GridState, SpectralState
from gfs_dynamical_core.jax.stepper import advance
from gfs_dynamical_core.jax.transforms import (
    TransformConfig,
    get_gaussian_latitudes,
    grid_to_spectral,
    grid_to_spectral_tendencies,
    spectral_to_grid,
)

# ── Parameters matching the Fortran build ────────────────────────────────
L = 64
NLONS = 2 * L - 1  # 127
NLATS = L  # 64
NLEVS = 20
NTRAC = 1
NTRUNC = int(NLONS / 3 - 2)  # 40
NDIMSPEC = (NTRUNC + 1) * (NTRUNC + 2) // 2  # 861

set_constant("reference_air_pressure", value=1e5, units="Pa")


# ═══════════════════════════════════════════════════════════════════════════
# Loading Fortran debug dumps
# ═══════════════════════════════════════════════════════════════════════════


def load_fortran_dump(path):
    """Load a Fortran debug binary dump and return a dict of numpy arrays.

    The Fortran arrays are written in column-major order (lon fastest).
    We reshape and transpose to get (lev, lat, lon) / (lat, lon) convention.

    Fortran grid latitudes go south-to-north (SHTNS Gauss).
    s2fft GL latitudes go north-to-south.
    We flip the latitude axis so everything is in s2fft convention.
    """
    data = Path(path).read_bytes()
    offset = 0

    def read_array(shape, dtype=np.float64):
        nonlocal offset
        n = int(np.prod(shape)) * np.dtype(dtype).itemsize
        arr = np.frombuffer(data[offset : offset + n], dtype=dtype)
        offset += n
        # Fortran column-major: reverse shape for reshape, then transpose
        arr = arr.reshape(shape[::-1]).T  # now in (dim0, dim1, ...) C order
        return arr

    result = {}

    # Grid fields: Fortran shape (nlons, nlats, nlevs) -> we get (nlons, nlats, nlevs)
    result["ug"] = read_array((NLONS, NLATS, NLEVS))
    result["vg"] = read_array((NLONS, NLATS, NLEVS))
    result["virtempg"] = read_array((NLONS, NLATS, NLEVS))
    result["lnpsg"] = read_array((NLONS, NLATS))
    result["tracerg"] = read_array((NLONS, NLATS, NLEVS, NTRAC))

    # Spectral fields: complex128
    result["vrtspec"] = read_array((NDIMSPEC, NLEVS), dtype=np.complex128)
    result["divspec"] = read_array((NDIMSPEC, NLEVS), dtype=np.complex128)
    result["virtempspec"] = read_array((NDIMSPEC, NLEVS), dtype=np.complex128)
    result["lnpsspec"] = read_array((NDIMSPEC,), dtype=np.complex128)

    assert offset == len(data), f"Did not consume all data: {offset} != {len(data)}"

    # Reorder grid fields to (lev, lat, lon) with lat north-to-south
    # Fortran grid: (nlons, nlats, nlevs) with lat south-to-north
    # Want: (nlevs, nlats, nlons) with lat north-to-south
    for name in ["ug", "vg", "virtempg"]:
        arr = result[name]  # (nlons, nlats, nlevs)
        arr = arr.transpose(2, 1, 0)  # (nlevs, nlats, nlons)
        arr = arr[:, ::-1, :]  # flip lat to N->S
        result[name] = arr

    # lnpsg: (nlons, nlats) -> (nlats, nlons) with lat flip
    result["lnpsg"] = result["lnpsg"].T[::-1, :]

    # tracerg: (nlons, nlats, nlevs, ntrac) -> (ntrac, nlevs, nlats, nlons)
    arr = result["tracerg"]  # (nlons, nlats, nlevs, ntrac)
    arr = arr.transpose(3, 2, 1, 0)  # (ntrac, nlevs, nlats, nlons)
    arr = arr[:, :, ::-1, :]  # flip lat
    result["tracerg"] = arr

    # Spectral: (ndimspec, nlevs) -> (nlevs, ndimspec)
    for name in ["vrtspec", "divspec", "virtempspec"]:
        result[name] = result[name].T  # (nlevs, ndimspec)

    # lnpsspec stays (ndimspec,)

    return result


def compare_field(name, fortran, jax_arr, indent="  "):
    """Print comparison statistics for a single field."""
    if np.iscomplexobj(jax_arr):
        jax_arr = jax_arr.real
    fortran = np.asarray(fortran)
    jax_arr = np.asarray(jax_arr)

    if fortran.shape != jax_arr.shape:
        print(
            f"{indent}{name:25s}  SHAPE MISMATCH: F={fortran.shape} J={jax_arr.shape}"
        )
        return

    diff = np.abs(fortran - jax_arr)
    max_f = np.abs(fortran).max()
    max_j = np.abs(jax_arr).max()
    max_diff = diff.max()
    rms_diff = np.sqrt((diff**2).mean())
    rel = max_diff / max_f if max_f > 0 else 0.0
    flag = " <<<" if rel > 1e-3 else ""
    print(
        f"{indent}{name:25s}  max|F|={max_f:12.6e}  max|J|={max_j:12.6e}  "
        f"max|diff|={max_diff:12.6e}  rel={rel:12.6e}  rms={rms_diff:12.6e}{flag}"
    )


def compare_field_per_level(name, fortran, jax_arr, max_levels=None):
    """Print per-level comparison for 3D fields (lev, lat, lon)."""
    if np.iscomplexobj(jax_arr):
        jax_arr = jax_arr.real
    fortran = np.asarray(fortran)
    jax_arr = np.asarray(jax_arr)
    n_lev = fortran.shape[0]
    if max_levels is not None:
        n_lev = min(n_lev, max_levels)
    print(f"  {name} per level:")
    for k in range(n_lev):
        diff = np.abs(fortran[k] - jax_arr[k])
        max_f = np.abs(fortran[k]).max()
        max_diff = diff.max()
        rel = max_diff / max_f if max_f > 0 else 0.0
        flag = " <<<" if rel > 1e-3 else ""
        print(
            f"    lev {k:2d}: max|F|={max_f:12.6e}  max|diff|={max_diff:12.6e}  rel={rel:12.6e}{flag}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    print(f"Grid: {NLONS}x{NLATS}x{NLEVS}  L={L}  ntrunc={NTRUNC}  ndimspec={NDIMSPEC}")
    grid = climt.get_grid(nx=NLONS, ny=NLATS, nz=NLEVS)
    timestep = timedelta(minutes=5)
    dt = timestep.total_seconds()

    # ── Run Fortran for 1 step ───────────────────────────────────────────
    print("\n=== Running Fortran (1 step) to generate debug dumps ===")
    dycore_f = GFSDynamicalCore()
    dcmip = climt.DcmipInitialConditions(add_perturbation=True)

    state_f = climt.get_default_state([dycore_f], grid_state=grid)
    out_f = dcmip(state_f)
    state_f.update(out_f)

    diag_f, out_f = dycore_f(state_f, timestep=timestep)

    # ── Run JAX for 1 step ───────────────────────────────────────────────
    print("\n=== Running JAX (1 step) ===")
    dycore_j = GFSDynamicsJAX()

    state_j = climt.get_default_state([dycore_j], grid_state=grid)
    out_j = dcmip(state_j)
    state_j.update(out_j)

    diag_j, out_j = dycore_j(state_j, timestep=timestep)

    # Grab internal configs
    dyn_config = dycore_j.dyn_config
    trans_config = dycore_j.trans_config
    stepper_config = dycore_j.stepper_config
    latitudes = dycore_j._latitudes
    phis_grads = dycore_j._phis_grads

    # ── Load Fortran debug dumps ─────────────────────────────────────────
    dump_dir = Path("debug_data")
    dump_stage0 = dump_dir / "fortran_step_1_stage_0.bin"
    dump_stage1 = dump_dir / "fortran_step_1_stage_1.bin"
    dump_stage2 = dump_dir / "fortran_step_1_stage_2.bin"
    dump_stage3 = dump_dir / "fortran_step_1_stage_3.bin"

    for p in [dump_stage0, dump_stage1, dump_stage2, dump_stage3]:
        if not p.exists():
            print(
                f"  WARNING: {p} not found! The Fortran run above should have created it."
            )
            print("  Make sure debug_step_count threshold in run.f90 allows dumping.")
            return

    f_s0 = load_fortran_dump(dump_stage0)
    f_s1 = load_fortran_dump(dump_stage1)
    f_s2 = load_fortran_dump(dump_stage2)
    f_s3 = load_fortran_dump(dump_stage3)

    # ── Build JAX initial spectral state ─────────────────────────────────
    u0 = jnp.array(state_j["eastward_wind"])
    v0 = jnp.array(state_j["northward_wind"])
    temp0 = jnp.array(state_j["air_temperature"])
    ps0 = jnp.array(state_j["surface_air_pressure"])
    q0 = jnp.array(state_j["specific_humidity"])

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

    # Reconstruct grid from spectral (this is what the model actually uses)
    grid0_jax, grads0_jax = spectral_to_grid(spec0, trans_config)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 0: Initial state (after spec -> grid in getdyntend)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("STAGE 0: Initial state (spec -> grid)")
    print("=" * 80)

    # Note: Fortran's "stage 0" is dumped inside getdyntend after converting
    # spectral -> grid. The grid fields are what the Fortran dynamics sees.

    compare_field("u (eastward wind)", f_s0["ug"], grid0_jax.u)
    compare_field("v (northward wind)", f_s0["vg"], grid0_jax.v)
    compare_field("T (virtual temp)", f_s0["virtempg"], grid0_jax.temperature)
    compare_field("ln(ps)", f_s0["lnpsg"], grid0_jax.log_surface_pressure)
    compare_field("q (tracer)", f_s0["tracerg"], grid0_jax.tracers)

    # Detailed per-level for u and v
    compare_field_per_level("u", f_s0["ug"], grid0_jax.u)
    compare_field_per_level("v", f_s0["vg"], grid0_jax.v)

    # ── Compute JAX grid tendencies from initial state ───────────────────
    print("\n--- Computing JAX grid-space tendencies from stage-0 state ---")
    grid_tends_jax = full_dynamics_step(
        grid0_jax, grads0_jax, phis_grads, dyn_config, latitudes
    )

    # ── Compare vertical velocities ──────────────────────────────────────
    press_diag = compute_pressure_diagnostics(
        grid0_jax.log_surface_pressure, dyn_config
    )
    vvels = compute_vertical_velocities(grid0_jax, grads0_jax, press_diag, dyn_config)
    print(f"\n  Vertical velocities from JAX:")
    print(
        f"    omega   min/max: {float(vvels.omega.real.min()):.6e}  {float(vvels.omega.real.max()):.6e}"
    )
    print(
        f"    etadot  min/max: {float(vvels.etadot.real.min()):.6e}  {float(vvels.etadot.real.max()):.6e}"
    )
    print(
        f"    dlnpsdt min/max: {float(vvels.d_log_ps_d_t.real.min()):.6e}  {float(vvels.d_log_ps_d_t.real.max()):.6e}"
    )

    # ── Grid tendency magnitudes ─────────────────────────────────────────
    print(f"\n  Grid-space tendency magnitudes:")
    print(
        f"    u_flux      min/max: {float(grid_tends_jax.u_flux.real.min()):.6e}  {float(grid_tends_jax.u_flux.real.max()):.6e}"
    )
    print(
        f"    v_flux      min/max: {float(grid_tends_jax.v_flux.real.min()):.6e}  {float(grid_tends_jax.v_flux.real.max()):.6e}"
    )
    print(
        f"    temp_tend   min/max: {float(grid_tends_jax.temp_tend.real.min()):.6e}  {float(grid_tends_jax.temp_tend.real.max()):.6e}"
    )
    print(
        f"    log_ps_tend min/max: {float(grid_tends_jax.log_ps_tend.real.min()):.6e}  {float(grid_tends_jax.log_ps_tend.real.max()):.6e}"
    )
    print(
        f"    KE          min/max: {float(grid_tends_jax.kinetic_energy.real.min()):.6e}  {float(grid_tends_jax.kinetic_energy.real.max()):.6e}"
    )

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: After RK stage 1
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("STAGE 1: After RK stage 1 update")
    print("=" * 80)

    # JAX stage 1: We need to replicate the stepper logic manually.
    # The stepper computes spectral tendencies and does the explicit RK update.
    from gfs_dynamical_core.jax.dynamics import get_spectral_tendencies
    from gfs_dynamical_core.jax.stepper import StepperConfig

    spec_tends_0 = get_spectral_tendencies(
        spec0, phis_grads, dyn_config, trans_config, latitudes
    )

    # Explicit RK stage 1:
    sc = stepper_config
    vort1 = spec0.vorticity + sc.a21 * dt * spec_tends_0.d_vorticity_d_t
    div1 = spec0.divergence + sc.a21 * dt * spec_tends_0.d_divergence_d_t
    temp1 = spec0.temperature + sc.a21 * dt * spec_tends_0.d_temperature_d_t
    lnps1 = (
        spec0.log_surface_pressure
        + sc.a21 * dt * spec_tends_0.d_log_surface_pressure_d_t
    )
    tracers1 = spec0.tracers + sc.a21 * dt * spec_tends_0.d_tracers_d_t

    state1_jax = SpectralState(
        vorticity=vort1,
        divergence=div1,
        temperature=temp1,
        log_surface_pressure=lnps1,
        tracers=tracers1,
    )
    grid1_jax, _ = spectral_to_grid(state1_jax, trans_config)

    compare_field("u (eastward wind)", f_s1["ug"], grid1_jax.u)
    compare_field("v (northward wind)", f_s1["vg"], grid1_jax.v)
    compare_field("T (virtual temp)", f_s1["virtempg"], grid1_jax.temperature)
    compare_field("ln(ps)", f_s1["lnpsg"], grid1_jax.log_surface_pressure)

    compare_field_per_level("u", f_s1["ug"], grid1_jax.u)
    compare_field_per_level("v", f_s1["vg"], grid1_jax.v)
    compare_field_per_level("T", f_s1["virtempg"], grid1_jax.temperature)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: After RK stage 2
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("STAGE 2: After RK stage 2 update")
    print("=" * 80)

    spec_tends_1 = get_spectral_tendencies(
        state1_jax, phis_grads, dyn_config, trans_config, latitudes
    )

    vort2 = spec0.vorticity + dt * (
        sc.a31 * spec_tends_0.d_vorticity_d_t + sc.a32 * spec_tends_1.d_vorticity_d_t
    )
    div2 = spec0.divergence + dt * (
        sc.a31 * spec_tends_0.d_divergence_d_t + sc.a32 * spec_tends_1.d_divergence_d_t
    )
    temp2 = spec0.temperature + dt * (
        sc.a31 * spec_tends_0.d_temperature_d_t
        + sc.a32 * spec_tends_1.d_temperature_d_t
    )
    lnps2 = spec0.log_surface_pressure + dt * (
        sc.a31 * spec_tends_0.d_log_surface_pressure_d_t
        + sc.a32 * spec_tends_1.d_log_surface_pressure_d_t
    )
    tracers2 = spec0.tracers + dt * (
        sc.a31 * spec_tends_0.d_tracers_d_t + sc.a32 * spec_tends_1.d_tracers_d_t
    )

    state2_jax = SpectralState(
        vorticity=vort2,
        divergence=div2,
        temperature=temp2,
        log_surface_pressure=lnps2,
        tracers=tracers2,
    )
    grid2_jax, _ = spectral_to_grid(state2_jax, trans_config)

    compare_field("u (eastward wind)", f_s2["ug"], grid2_jax.u)
    compare_field("v (northward wind)", f_s2["vg"], grid2_jax.v)
    compare_field("T (virtual temp)", f_s2["virtempg"], grid2_jax.temperature)
    compare_field("ln(ps)", f_s2["lnpsg"], grid2_jax.log_surface_pressure)

    compare_field_per_level("v stage2", f_s2["vg"], grid2_jax.v)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 3: Final state after full timestep
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("STAGE 3: Final state (after diffusion)")
    print("=" * 80)

    # Compare the final output of the stepper
    grid_final_f_u = np.asarray(out_f["eastward_wind"])
    grid_final_f_v = np.asarray(out_f["northward_wind"])
    grid_final_f_t = np.asarray(out_f["air_temperature"])
    grid_final_f_ps = np.asarray(out_f["surface_air_pressure"])

    grid_final_j_u = np.asarray(out_j["eastward_wind"])
    grid_final_j_v = np.asarray(out_j["northward_wind"])
    grid_final_j_t = np.asarray(out_j["air_temperature"])
    grid_final_j_ps = np.asarray(out_j["surface_air_pressure"])

    if np.iscomplexobj(grid_final_j_u):
        grid_final_j_u = grid_final_j_u.real
    if np.iscomplexobj(grid_final_j_v):
        grid_final_j_v = grid_final_j_v.real
    if np.iscomplexobj(grid_final_j_t):
        grid_final_j_t = grid_final_j_t.real
    if np.iscomplexobj(grid_final_j_ps):
        grid_final_j_ps = grid_final_j_ps.real

    compare_field("u (final)", grid_final_f_u, grid_final_j_u)
    compare_field("v (final)", grid_final_f_v, grid_final_j_v)
    compare_field("T (final)", grid_final_f_t, grid_final_j_t)
    compare_field("ps (final)", grid_final_f_ps, grid_final_j_ps)

    # Also compare with the dump
    compare_field("u (dump vs output)", f_s3["ug"], grid_final_f_u)
    compare_field("v (dump vs output)", f_s3["vg"], grid_final_f_v)

    compare_field_per_level("v (final)", grid_final_f_v, grid_final_j_v)

    # ══════════════════════════════════════════════════════════════════════
    # Spectral tendency analysis
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SPECTRAL TENDENCY ANALYSIS")
    print("=" * 80)

    print("\n--- Stage 0 spectral tendencies (JAX) ---")
    # Show the spectral tendency magnitudes
    for name, tend in [
        ("d_vort/dt", spec_tends_0.d_vorticity_d_t),
        ("d_div/dt", spec_tends_0.d_divergence_d_t),
        ("d_T/dt", spec_tends_0.d_temperature_d_t),
        ("d_lnps/dt", spec_tends_0.d_log_surface_pressure_d_t),
    ]:
        arr = np.asarray(tend)
        print(
            f"  {name:15s}  max|real|={np.abs(arr.real).max():.6e}  max|imag|={np.abs(arr.imag).max():.6e}"
        )

    # ── Check the spectral Δ between stage 0 and stage 1 ────────────────
    print("\n--- Spectral state change: stage1 - stage0 ---")
    for name, s0, s1 in [
        ("vorticity", spec0.vorticity, vort1),
        ("divergence", spec0.divergence, div1),
        ("temperature", spec0.temperature, temp1),
        ("lnps", spec0.log_surface_pressure, lnps1),
    ]:
        d = np.asarray(s1 - s0)
        print(f"  {name:15s}  max|Δ|={np.abs(d).max():.6e}")

    # ══════════════════════════════════════════════════════════════════════
    # Summary diagnostics
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Key ratios
    du_f = np.asarray(out_f["eastward_wind"]) - np.asarray(state_j["eastward_wind"])
    du_j = grid_final_j_u - np.asarray(state_j["eastward_wind"])
    dv_f = np.asarray(out_f["northward_wind"]) - np.asarray(state_j["northward_wind"])
    dv_j = grid_final_j_v - np.asarray(state_j["northward_wind"])
    dt_f = np.asarray(out_f["air_temperature"]) - np.asarray(state_j["air_temperature"])
    dt_j = grid_final_j_t - np.asarray(state_j["air_temperature"])
    dps_f = np.asarray(out_f["surface_air_pressure"]) - np.asarray(
        state_j["surface_air_pressure"]
    )
    dps_j = grid_final_j_ps - np.asarray(state_j["surface_air_pressure"])

    print(f"\n  Tendency magnitude ratios (JAX / Fortran):")
    for name, df, dj in [
        ("Δu", du_f, du_j),
        ("Δv", dv_f, dv_j),
        ("ΔT", dt_f, dt_j),
        ("Δps", dps_f, dps_j),
    ]:
        max_f = np.abs(df).max()
        max_j = np.abs(dj).max()
        ratio = max_j / max_f if max_f > 0 else float("inf")
        print(
            f"    {name:5s}:  max|F|={max_f:12.6e}  max|J|={max_j:12.6e}  ratio={ratio:.4f}"
        )

    # Zonal mean tendencies
    print(f"\n  Zonal-mean tendency comparison:")
    for name, df, dj in [("Δu", du_f, du_j), ("Δv", dv_f, dv_j), ("ΔT", dt_f, dt_j)]:
        zm_f = np.abs(df.mean(axis=-1)).max()
        zm_j = np.abs(dj.mean(axis=-1)).max()
        ratio = zm_j / zm_f if zm_f > 0 else float("inf")
        print(
            f"    {name:5s} zonal mean:  F={zm_f:12.6e}  J={zm_j:12.6e}  ratio={ratio:.4f}"
        )


if __name__ == "__main__":
    main()
