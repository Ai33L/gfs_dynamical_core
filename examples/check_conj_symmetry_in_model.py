"""
Check conjugate symmetry of spectral data throughout actual model simulation.

The inverse vector transform (spectral→grid) using a single spin+1 transform
is mathematically correct IF the spectral coefficients satisfy conjugate symmetry:

    f_{l,-m} = (-1)^m * conj(f_{l,m})

This is the condition for the corresponding grid field to be real-valued.
If this symmetry breaks at any point, the inverse transform will produce
complex grid values whose imaginary parts get silently discarded (via .real),
corrupting the dynamics and causing blow-up.

This script:
  1. Initializes the DCMIP baroclinic wave
  2. Converts to spectral space (grid_to_spectral)
  3. Steps through the model
  4. At each step, checks conjugate symmetry of ALL spectral fields
  5. Reports where/when symmetry breaks

If symmetry breaks, this is the root cause of the blow-up — not the
inverse transform itself.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

from datetime import timedelta

import climt
import jax
import jax.numpy as jnp
import numpy as np
import s2fft
from sympl import set_constant

from gfs_dynamical_core.component_jax import GFSDynamicsJAX
from gfs_dynamical_core.jax.dynamics import DynamicsConfig, compute_pressure_diagnostics
from gfs_dynamical_core.jax.states import GridState, SpectralState
from gfs_dynamical_core.jax.stepper import (
    StepperConfig,
    advance,
    init_diffusion_operators,
    init_semi_implicit_matrices,
)
from gfs_dynamical_core.jax.transforms import (
    TransformConfig,
    get_gaussian_latitudes,
    grid_to_spectral,
    spectral_to_grid,
)

# ── Setup ────────────────────────────────────────────────────────────────
set_constant("reference_air_pressure", value=1e5, units="Pa")

L = 64
n_lon = 2 * L - 1  # 127
n_lat = L  # 64
n_lev = 20

print(f"Grid: {n_lon}x{n_lat}x{n_lev} (L={L})")
grid = climt.get_grid(nx=n_lon, ny=n_lat, nz=n_lev)
timestep = timedelta(minutes=5)
dt = timestep.total_seconds()


# ── Helper: check conjugate symmetry ────────────────────────────────────
def check_conjugate_symmetry(flm, name, L, verbose=False):
    """
    Check if f_{l,-m} = (-1)^m * conj(f_{l,m}) for all l, m>0.

    Returns (max_abs_error, max_rel_error, worst_l, worst_m).
    """
    flm_np = np.asarray(flm)
    max_abs = 0.0
    max_rel = 0.0
    worst_l = 0
    worst_m = 0

    for ll in range(L):
        for mm in range(1, min(ll + 1, L)):
            pos_idx = L - 1 + mm
            neg_idx = L - 1 - mm
            val_pos = flm_np[ll, pos_idx]
            val_neg = flm_np[ll, neg_idx]
            expected_neg = ((-1) ** mm) * val_pos.conjugate()
            err = abs(val_neg - expected_neg)
            scale = max(abs(val_pos), abs(val_neg), 1e-30)
            rel = err / scale

            if err > max_abs:
                max_abs = err
                worst_l = ll
                worst_m = mm
            if rel > max_rel:
                max_rel = rel

    if verbose and max_abs > 1e-10:
        print(
            f"    {name:30s}  max_abs={max_abs:.4e}  max_rel={max_rel:.4e}  "
            f"worst at (l={worst_l}, m={worst_m})"
        )

    return max_abs, max_rel, worst_l, worst_m


def check_spectral_state(spec_state, L, label="", verbose=True):
    """Check conjugate symmetry of all fields in a SpectralState."""
    results = {}

    # Scalar 2D fields
    for name, flm in [("log_surface_pressure", spec_state.log_surface_pressure)]:
        abs_err, rel_err, wl, wm = check_conjugate_symmetry(
            flm, name, L, verbose=verbose
        )
        results[name] = (abs_err, rel_err, wl, wm)

    # 3D fields (n_lev, L, 2L-1)
    for name, field3d in [
        ("vorticity", spec_state.vorticity),
        ("divergence", spec_state.divergence),
        ("temperature", spec_state.temperature),
    ]:
        field_np = np.asarray(field3d)
        worst_abs = 0.0
        worst_rel = 0.0
        worst_lev = 0
        worst_l = 0
        worst_m = 0
        for k in range(field_np.shape[0]):
            abs_err, rel_err, wl, wm = check_conjugate_symmetry(
                field_np[k], f"{name}[{k}]", L, verbose=False
            )
            if abs_err > worst_abs:
                worst_abs = abs_err
                worst_rel = rel_err
                worst_lev = k
                worst_l = wl
                worst_m = wm
        if verbose and worst_abs > 1e-10:
            print(
                f"    {name:30s}  max_abs={worst_abs:.4e}  max_rel={worst_rel:.4e}  "
                f"worst at lev={worst_lev}, (l={worst_l}, m={worst_m})"
            )
        results[name] = (worst_abs, worst_rel, worst_lev, worst_l, worst_m)

    # Tracers (n_tracers, n_lev, L, 2L-1)
    tracers_np = np.asarray(spec_state.tracers)
    for t in range(tracers_np.shape[0]):
        tname = f"tracer[{t}]"
        worst_abs = 0.0
        worst_rel = 0.0
        worst_lev = 0
        worst_l = 0
        worst_m = 0
        for k in range(tracers_np.shape[1]):
            abs_err, rel_err, wl, wm = check_conjugate_symmetry(
                tracers_np[t, k], f"{tname}[{k}]", L, verbose=False
            )
            if abs_err > worst_abs:
                worst_abs = abs_err
                worst_rel = rel_err
                worst_lev = k
                worst_l = wl
                worst_m = wm
        if verbose and worst_abs > 1e-10:
            print(
                f"    {tname:30s}  max_abs={worst_abs:.4e}  max_rel={worst_rel:.4e}  "
                f"worst at lev={worst_lev}, (l={worst_l}, m={worst_m})"
            )
        results[tname] = (worst_abs, worst_rel, worst_lev, worst_l, worst_m)

    return results


def check_grid_reality(grid_state, label=""):
    """Check that grid fields are real-valued (no spurious imaginary parts)."""
    issues = {}
    for name, field in [
        ("u", grid_state.u),
        ("v", grid_state.v),
        ("temperature", grid_state.temperature),
        ("log_surface_pressure", grid_state.log_surface_pressure),
        ("vorticity", grid_state.vorticity),
        ("divergence", grid_state.divergence),
    ]:
        arr = np.asarray(field)
        if np.iscomplexobj(arr):
            imag_max = np.abs(arr.imag).max()
            real_max = np.abs(arr.real).max()
            rel = imag_max / max(real_max, 1e-30)
            issues[name] = (imag_max, rel)
            if imag_max > 1e-10:
                print(
                    f"    {name:30s}  max|imag|={imag_max:.4e}  "
                    f"rel_to_real={rel:.4e}  {'⚠ COMPLEX' if rel > 1e-10 else 'OK'}"
                )
    return issues


# ═══════════════════════════════════════════════════════════════════════════
# Initialize model state
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Initializing DCMIP Baroclinic Wave ===")
dycore = GFSDynamicsJAX()
dcmip = climt.DcmipInitialConditions(add_perturbation=True)
my_state = climt.get_default_state([dycore], grid_state=grid)
out = dcmip(my_state)
my_state.update(out)

from sympl import get_constant

# Extract arrays and build JAX state manually so we can inspect spectral data
u = jnp.array(my_state["eastward_wind"].values)
v = jnp.array(my_state["northward_wind"].values)
temp = jnp.array(my_state["air_temperature"].values)
ps = jnp.array(my_state["surface_air_pressure"].values)
q = jnp.array(my_state["specific_humidity"].values)
phis = jnp.array(my_state["surface_geopotential"].values)
ak = jnp.array(
    my_state["atmosphere_hybrid_sigma_pressure_a_coordinate_on_interface_levels"].values
)
bk = jnp.array(
    my_state["atmosphere_hybrid_sigma_pressure_b_coordinate_on_interface_levels"].values
)

trans_config = TransformConfig(
    L=L, sampling="gl", radius=get_constant("planetary_radius", "m")
)
T = trans_config.truncation

dbk = bk[:-1] - bk[1:]
ck = ak[:-1] * bk[1:] - ak[1:] * bk[:-1]

dyn_config = DynamicsConfig(
    ak=ak,
    bk=bk,
    ck=ck,
    dbk=dbk,
    rk=get_constant("gas_constant_of_dry_air", "J kg^-1 K^-1")
    / get_constant("heat_capacity_of_dry_air_at_constant_pressure", "J kg^-1 K^-1"),
    toa_pressure=0.0,
    radius=get_constant("planetary_radius", "m"),
    omega=get_constant("planetary_rotation_rate", "s^-1"),
    g=get_constant("gravitational_acceleration", "m s^-2"),
    rd=get_constant("gas_constant_of_dry_air", "J kg^-1 K^-1"),
    rv=get_constant("gas_constant_of_vapor_phase", "J kg^-1 K^-1"),
    cp=get_constant("heat_capacity_of_dry_air_at_constant_pressure", "J kg^-1 K^-1"),
    cvap=get_constant("heat_capacity_of_vapor_phase", "J kg^-1 K^-1"),
)

# Build grid state
grid_orig = GridState(
    u=u,
    v=v,
    temperature=temp,
    vorticity=jnp.zeros_like(u),
    divergence=jnp.zeros_like(u),
    log_surface_pressure=jnp.log(ps),
    tracers=jnp.stack([q], axis=0),
)

# ═══════════════════════════════════════════════════════════════════════════
# CHECK 0: Initial grid → spectral
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("CHECK 0: Conjugate symmetry of initial spectral state (grid_to_spectral)")
print("=" * 72)

spec_orig = grid_to_spectral(grid_orig, trans_config)
results0 = check_spectral_state(spec_orig, L, label="initial")

all_ok = all(r[0] < 1e-10 if len(r) == 4 else r[0] < 1e-10 for r in results0.values())
if all_ok:
    print("  ✅ All fields have conjugate symmetry (max error < 1e-10)")
else:
    print("  ❌ Some fields LACK conjugate symmetry!")

# ═══════════════════════════════════════════════════════════════════════════
# CHECK 0b: Grid reality after spectral → grid round-trip
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("CHECK 0b: Grid reality after spec→grid inverse of initial state")
print("=" * 72)

grid_rt, _ = spectral_to_grid(spec_orig, trans_config)
issues0 = check_grid_reality(grid_rt, label="initial round-trip")
if not issues0 or all(v[0] < 1e-10 for v in issues0.values()):
    print("  ✅ All grid fields are real-valued")
else:
    print("  ⚠ Some grid fields have non-negligible imaginary parts!")

# ═══════════════════════════════════════════════════════════════════════════
# CHECK 0c: Compare original grid vs round-tripped grid
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("CHECK 0c: Grid → spectral → grid round-trip accuracy")
print("=" * 72)

for name, orig_field, rt_field in [
    ("u", grid_orig.u, grid_rt.u),
    ("v", grid_orig.v, grid_rt.v),
    ("temperature", grid_orig.temperature, grid_rt.temperature),
    (
        "log_surface_pressure",
        grid_orig.log_surface_pressure,
        grid_rt.log_surface_pressure,
    ),
]:
    orig_np = np.real(np.asarray(orig_field))
    rt_np = np.real(np.asarray(rt_field))
    diff = np.abs(orig_np - rt_np)
    max_diff = diff.max()
    max_val = max(np.abs(orig_np).max(), 1e-30)
    rel = max_diff / max_val
    ok = "OK" if rel < 1e-3 else "WARN" if rel < 0.1 else "FAIL"
    print(f"  {name:30s}  max|diff|={max_diff:.4e}  rel={rel:.4e}  [{ok}]")

# ═══════════════════════════════════════════════════════════════════════════
# Setup stepper
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("Setting up stepper configuration...")
print("=" * 72)

_default_sc = StepperConfig(dt=dt)
si_matrices = init_semi_implicit_matrices(
    dyn_config,
    trans_config,
    dt,
    aa22=_default_sc.aa22,
    aa33=_default_sc.aa33,
    bb4=_default_sc.bb4,
)
diff_ops = init_diffusion_operators(dyn_config, trans_config, dt)
stepper_config = StepperConfig(
    dt=dt,
    explicit=False,
    amhyb=si_matrices["amhyb"],
    bmhyb=si_matrices["bmhyb"],
    tor_hyb=si_matrices["tor_hyb"],
    svhyb=si_matrices["svhyb"],
    d_hyb_m=si_matrices["d_hyb_m"],
    disspec=diff_ops["disspec"],
    diff_prof=diff_ops["diff_prof"],
    dmp_prof=diff_ops["dmp_prof"],
)

latitudes = get_gaussian_latitudes(L)

# Gaussian weights
_, raw_weights = np.polynomial.legendre.leggauss(L)
gauss_weights = jnp.array(raw_weights / 2.0)

# Compute pdryini
lnps_grid = jnp.log(ps)
press_diag_init = compute_pressure_diagnostics(lnps_grid, dyn_config)
g = dyn_config.g
pwat_init = jnp.sum(q * press_diag_init.dp, axis=0) / g
w = gauss_weights[:, None]
pmean_init = float(jnp.sum(w * ps) / n_lon)
pwat_global_init = float(jnp.sum(w * pwat_init) / n_lon)
pdryini = pmean_init - g * pwat_global_init

# Compute phis gradients
sampling = trans_config.sampling
radius = dyn_config.radius
phis_lm = s2fft.forward_jax(phis, L, sampling=sampling)
l_arr = jnp.arange(L)
l_factor = jnp.sqrt(l_arr * (l_arr + 1))
F1_phis_lm = -l_factor[:, None] * phis_lm
f_phis_spin1 = s2fft.inverse_jax(F1_phis_lm, L, spin=1, sampling=sampling)
dphisdx = f_phis_spin1.imag / radius
dphisdy = -f_phis_spin1.real / radius
phis_grads = (dphisdx, dphisdy)

# ═══════════════════════════════════════════════════════════════════════════
# Time-stepping loop with conjugate symmetry checks
# ═══════════════════════════════════════════════════════════════════════════
N_STEPS = 30  # Enough to see blow-up (usually happens around step 24)
spec_state = spec_orig

print("\n" + "=" * 72)
print(f"Running {N_STEPS} steps, checking conjugate symmetry at each step")
print("=" * 72)

history = []

for step in range(N_STEPS):
    # Check conjugate symmetry BEFORE the step
    results = check_spectral_state(spec_state, L, label=f"step {step}", verbose=False)

    # Aggregate max errors
    max_abs_all = max(r[0] for r in results.values())

    # Also check vort/div specifically (these are the vector-transform outputs)
    vort_err = results.get("vorticity", (0,))[0]
    div_err = results.get("divergence", (0,))[0]
    temp_err = results.get("temperature", (0,))[0]
    lnps_err = results.get("log_surface_pressure", (0,))[0]

    # Check grid reality
    grid_state, _ = spectral_to_grid(spec_state, trans_config)
    u_imag = np.abs(np.imag(np.asarray(grid_state.u))).max()
    v_imag = np.abs(np.imag(np.asarray(grid_state.v))).max()

    # Surface pressure range
    lnps_real = np.real(np.asarray(grid_state.log_surface_pressure))
    ps_grid = np.exp(lnps_real)
    ps_min = ps_grid.min() / 100.0
    ps_max = ps_grid.max() / 100.0

    status = "OK" if max_abs_all < 1e-8 else "WARN" if max_abs_all < 1.0 else "FAIL"

    print(
        f"  Step {step:3d}  conj_sym_err={max_abs_all:.4e}  "
        f"vort={vort_err:.2e}  div={div_err:.2e}  "
        f"temp={temp_err:.2e}  lnps={lnps_err:.2e}  "
        f"|Im(u)|={u_imag:.2e}  |Im(v)|={v_imag:.2e}  "
        f"PS={ps_min:.1f}-{ps_max:.1f}hPa  [{status}]"
    )

    history.append(
        {
            "step": step,
            "max_conj_err": max_abs_all,
            "vort_err": vort_err,
            "div_err": div_err,
            "temp_err": temp_err,
            "lnps_err": lnps_err,
            "u_imag": u_imag,
            "v_imag": v_imag,
            "ps_min": ps_min,
            "ps_max": ps_max,
        }
    )

    # Check for blow-up
    if ps_max > 2000 or np.isnan(ps_max):
        print(f"\n  *** BLOW-UP at step {step}! ***")
        break

    # Advance one step
    try:
        spec_state = advance(
            spec_state,
            phis_grads,
            dyn_config,
            trans_config,
            stepper_config,
            latitudes,
            gauss_weights,
            pdryini,
        )
    except Exception as e:
        print(f"\n  *** ERROR at step {step}: {e} ***")
        break

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

if history:
    first_bad = None
    for h in history:
        if h["max_conj_err"] > 1e-8:
            first_bad = h
            break

    if first_bad is None:
        print("  ✅ Conjugate symmetry maintained throughout all steps.")
        print("     The inverse transform is NOT the cause of blow-up.")
        print("     Look elsewhere: vertical advection, pressure diagnostics,")
        print("     semi-implicit solve, diffusion, etc.")
    else:
        print(f"  ❌ Conjugate symmetry BREAKS at step {first_bad['step']}!")
        print(f"     Max conjugate error: {first_bad['max_conj_err']:.4e}")
        print(f"     vort={first_bad['vort_err']:.4e}  div={first_bad['div_err']:.4e}")
        print(
            f"     temp={first_bad['temp_err']:.4e}  lnps={first_bad['lnps_err']:.4e}"
        )
        print()
        print("     This means the spectral coefficients lose their")
        print("     reality condition, so the single-spin inverse transform")
        print("     produces complex grid values. Taking .real silently")
        print("     discards information → corruption → blow-up.")
        print()
        print("     ROOT CAUSE: find where conjugate symmetry is broken.")
        print("     Candidates:")
        print("       1. grid_to_spectral_tendencies (forward vector transform)")
        print("       2. Semi-implicit solve (d_hyb_m matrices)")
        print("       3. Diffusion operators (disspec)")
        print("       4. The linear tendency computation")
        print("       5. IMEX RK coefficient arithmetic")

    print()
    print("  PS range over time:")
    for h in history:
        bar = "#" * max(1, int((h["ps_max"] - h["ps_min"]) / 2))
        print(
            f"    step {h['step']:3d}: {h['ps_min']:.1f} - {h['ps_max']:.1f} hPa  {bar}"
        )
