"""
Test using spin+1 and spin-1 forward transforms to correctly extract
complex vort/div spectral coefficients.

THE BUG:
  In grid_to_spectral, we compute:
    f_spin1 = -v + i*u
    F1_lm = forward_spin1(f_spin1)
    D_plus_izeta = l_factor * F1_lm / R
    div = D_plus_izeta.real    # WRONG for m!=0
    vrt = D_plus_izeta.imag    # WRONG for m!=0

  When D_lm = a+bi and zeta_lm = c+di (both complex for m!=0):
    D + i*zeta = (a-d) + i*(b+c)
    .real = a-d  (NOT D_lm!)
    .imag = b+c  (NOT zeta_lm!)

THE FIX:
  Use BOTH spin+1 and spin-1 transforms:
    f_plus  = -v + i*u  -> forward spin+1 -> F1_lm  -> D + i*zeta
    f_minus =  v + i*u  -> forward spin-1 -> Fm1_lm -> D - i*zeta

  Then:
    D_lm    = (spin1_result + spinm1_result) / 2
    zeta_lm = (spin1_result - spinm1_result) / (2i)

  Similarly for the inverse (spectral_to_grid):
    Use spin+1 inverse for (D + i*zeta) to get (-v + i*u)  ... but this has
    the same problem in reverse. Instead:
    spin+1 inverse of D_lm coefficients gives one part
    spin+1 inverse of zeta_lm coefficients gives another part
    ... or simply do TWO spin-1 inverse transforms.

  Actually for the INVERSE the current code is fine IF div and vrt are stored
  correctly. The combination div + 1j*vort forms the correct complex spin-1
  coefficients, and the inverse spin-1 transform correctly produces u,v.
  The problem is only in the FORWARD direction (and in grid_to_spectral_tendencies).
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

import jax.numpy as jnp
import numpy as np
import s2fft

L = 64
NLONS = 2 * L - 1
NLATS = L
sampling = "gl"
radius = 6.371e6

l_arr = jnp.arange(L)
l_factor = jnp.sqrt(l_arr * (l_arr + 1))
inv_l_factor = jnp.where(l_arr > 0, 1.0 / l_factor, 0.0)

cos_theta, _ = np.polynomial.legendre.leggauss(L)
thetas = np.flip(np.arccos(cos_theta))
latitudes = np.pi / 2.0 - thetas
lat_grid = latitudes[:, None]
lon_grid = np.linspace(0, 2 * np.pi, NLONS, endpoint=False)[None, :]


def old_forward(u, v):
    """Old (buggy) forward transform: (u,v) -> (vrt_spec, div_spec)."""
    f_spin1 = -v + 1j * u
    F1_lm = s2fft.forward_jax(f_spin1, L, spin=1, sampling=sampling)
    D_plus_izeta = l_factor[:, None] * F1_lm / radius
    div_spec = D_plus_izeta.real
    vrt_spec = D_plus_izeta.imag
    return vrt_spec, div_spec


def new_forward(u, v):
    """New (fixed) forward transform using spin+1 and spin-1.

    spin+1 forward of (-v + i*u) gives F1_lm  where l_factor*F1/R  = D + i*zeta
    spin-1 forward of ( v + i*u) gives Fm1_lm where l_factor*Fm1/R = D - i*zeta

    Then:
      D    = (result_p + result_m) / 2
      zeta = (result_p - result_m) / (2i) = -i*(result_p - result_m)/2
    """
    f_plus = -v + 1j * u  # spin +1 input
    f_minus = v + 1j * u  # spin -1 input

    F1_lm = s2fft.forward_jax(f_plus, L, spin=1, sampling=sampling)
    Fm1_lm = s2fft.forward_jax(f_minus, L, spin=-1, sampling=sampling)

    result_p = l_factor[:, None] * F1_lm / radius  # D + i*zeta
    result_m = l_factor[:, None] * Fm1_lm / radius  # D - i*zeta

    div_spec = (result_p + result_m) / 2
    vrt_spec = (result_p - result_m) / (2j)  # divide by 2i

    return vrt_spec, div_spec


def inverse(vrt_spec, div_spec):
    """Inverse transform: (vrt_spec, div_spec) -> (u, v).

    This is the same for both old and new, since the inverse correctly
    handles complex coefficients.
    """
    F1_lm = inv_l_factor[:, None] * (div_spec + 1j * vrt_spec) * radius
    f_spin1 = s2fft.inverse_jax(F1_lm, L, spin=1, sampling=sampling)
    u = f_spin1.imag
    v = -f_spin1.real
    return u, v


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: Roundtrip with known complex spectral coefficients
# ═══════════════════════════════════════════════════════════════════════

print("=" * 80)
print("TEST 1: Roundtrip with known complex vrt/div spectral coefficients")
print("=" * 80)

# Create spectral coefficients with known complex values for m>=1
vrt_orig = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
div_orig = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)

# Set some complex coefficients for l=2,m=1
vrt_val = 1e-5 + 3e-6j
div_val = 2e-6 - 1e-6j
j_m1_pos = 1 + L - 1
j_m1_neg = -1 + L - 1

# s2fft CS phase: f_l^{-m} = (-1)^m * conj(f_l^m)
vrt_orig = vrt_orig.at[2, j_m1_pos].set(vrt_val)
vrt_orig = vrt_orig.at[2, j_m1_neg].set(-np.conj(vrt_val))  # (-1)^1
div_orig = div_orig.at[2, j_m1_pos].set(div_val)
div_orig = div_orig.at[2, j_m1_neg].set(-np.conj(div_val))

# Also set some m=0 (real) and m=2 coefficients
vrt_orig = vrt_orig.at[1, L - 1].set(5e-6)  # l=1,m=0 (real)
vrt_orig = vrt_orig.at[3, L - 1].set(-2e-6)  # l=3,m=0 (real)

vrt_val_m2 = 7e-7 + 2e-7j
j_m2_pos = 2 + L - 1
j_m2_neg = -2 + L - 1
vrt_orig = vrt_orig.at[3, j_m2_pos].set(vrt_val_m2)
vrt_orig = vrt_orig.at[3, j_m2_neg].set(np.conj(vrt_val_m2))  # (-1)^2 = +1

# Inverse: spectral -> grid
u_grid, v_grid = inverse(vrt_orig, div_orig)

# Forward (old): grid -> spectral
vrt_old, div_old = old_forward(u_grid, v_grid)

# Forward (new): grid -> spectral
vrt_new, div_new = new_forward(u_grid, v_grid)

# Compare
print("\n  vrt(l=2,m=1):")
print(f"    original:  {complex(vrt_orig[2, j_m1_pos])}")
print(f"    old:       {complex(vrt_old[2, j_m1_pos])}")
print(f"    new:       {complex(vrt_new[2, j_m1_pos])}")
print(
    f"    old diff:  {abs(complex(vrt_orig[2, j_m1_pos]) - complex(vrt_old[2, j_m1_pos])):.6e}"
)
print(
    f"    new diff:  {abs(complex(vrt_orig[2, j_m1_pos]) - complex(vrt_new[2, j_m1_pos])):.6e}"
)

print("\n  div(l=2,m=1):")
print(f"    original:  {complex(div_orig[2, j_m1_pos])}")
print(f"    old:       {complex(div_old[2, j_m1_pos])}")
print(f"    new:       {complex(div_new[2, j_m1_pos])}")
print(
    f"    old diff:  {abs(complex(div_orig[2, j_m1_pos]) - complex(div_old[2, j_m1_pos])):.6e}"
)
print(
    f"    new diff:  {abs(complex(div_orig[2, j_m1_pos]) - complex(div_new[2, j_m1_pos])):.6e}"
)

print("\n  vrt(l=1,m=0):")
print(f"    original:  {complex(vrt_orig[1, L - 1])}")
print(f"    old:       {complex(vrt_old[1, L - 1])}")
print(f"    new:       {complex(vrt_new[1, L - 1])}")
print(
    f"    old diff:  {abs(complex(vrt_orig[1, L - 1]) - complex(vrt_old[1, L - 1])):.6e}"
)
print(
    f"    new diff:  {abs(complex(vrt_orig[1, L - 1]) - complex(vrt_new[1, L - 1])):.6e}"
)

print("\n  vrt(l=3,m=2):")
print(f"    original:  {complex(vrt_orig[3, j_m2_pos])}")
print(f"    old:       {complex(vrt_old[3, j_m2_pos])}")
print(f"    new:       {complex(vrt_new[3, j_m2_pos])}")
print(
    f"    old diff:  {abs(complex(vrt_orig[3, j_m2_pos]) - complex(vrt_old[3, j_m2_pos])):.6e}"
)
print(
    f"    new diff:  {abs(complex(vrt_orig[3, j_m2_pos]) - complex(vrt_new[3, j_m2_pos])):.6e}"
)

# ═══════════════════════════════════════════════════════════════════════
# TEST 2: Full grid roundtrip with realistic velocity field
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("TEST 2: Full grid roundtrip with realistic velocity field")
print("=" * 80)

u_test = (
    10.0 * np.cos(lat_grid) * np.ones_like(lon_grid)
    + 5.0 * np.cos(lat_grid) * np.cos(lon_grid)
    + 2.0 * np.sin(2 * lat_grid) * np.sin(3 * lon_grid)
)
v_test = 0.5 * np.sin(lat_grid) * np.sin(lon_grid) + 0.3 * np.cos(lat_grid) * np.cos(
    2 * lon_grid
)

u_j = jnp.array(u_test)
v_j = jnp.array(v_test)

# Old roundtrip
vrt_o, div_o = old_forward(u_j, v_j)
u_o, v_o = inverse(vrt_o, div_o)

# New roundtrip
vrt_n, div_n = new_forward(u_j, v_j)
u_n, v_n = inverse(vrt_n, div_n)

u_o = np.asarray(u_o.real)
v_o = np.asarray(v_o.real)
u_n = np.asarray(u_n.real)
v_n = np.asarray(v_n.real)

# Truncation-limited input (smooth the input to match spectral truncation)
# For a fair comparison, first project u,v to spectral and back
u_ref, v_ref = inverse(*new_forward(u_j, v_j))
u_ref = np.asarray(u_ref.real)
v_ref = np.asarray(v_ref.real)

print(f"\n  Old method roundtrip:")
print(
    f"    u max|diff|: {np.abs(u_o - u_test).max():.6e}  (max|u|={np.abs(u_test).max():.4f})"
)
print(
    f"    v max|diff|: {np.abs(v_o - v_test).max():.6e}  (max|v|={np.abs(v_test).max():.4f})"
)

print(f"\n  New method roundtrip:")
print(f"    u max|diff|: {np.abs(u_n - u_test).max():.6e}")
print(f"    v max|diff|: {np.abs(v_n - v_test).max():.6e}")

print(
    f"\n  Improvement factor u: {np.abs(u_o - u_test).max() / max(np.abs(u_n - u_test).max(), 1e-30):.1f}x"
)
print(
    f"  Improvement factor v: {np.abs(v_o - v_test).max() / max(np.abs(v_n - v_test).max(), 1e-30):.1f}x"
)

# ═══════════════════════════════════════════════════════════════════════
# TEST 3: Verify against Fortran debug dump
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("TEST 3: Compare with Fortran debug dump (if available)")
print("=" * 80)

from datetime import timedelta
from pathlib import Path

import climt
from sympl import set_constant

from gfs_dynamical_core.component_jax import GFSDynamicsJAX
from gfs_dynamical_core.jax.states import GridState
from gfs_dynamical_core.jax.transforms import (
    TransformConfig,
    grid_to_spectral,
    spectral_to_grid,
)

set_constant("reference_air_pressure", value=1e5, units="Pa")
NLEVS = 20
NTRUNC = int(NLONS / 3 - 2)
NDIMSPEC = (NTRUNC + 1) * (NTRUNC + 2) // 2

grid = climt.get_grid(nx=NLONS, ny=NLATS, nz=NLEVS)
dcmip = climt.DcmipInitialConditions(add_perturbation=True)
from gfs_dynamical_core import GFSDynamicalCore

dycore_f = GFSDynamicalCore()
state_f = climt.get_default_state([dycore_f], grid_state=grid)
out_f = dcmip(state_f)
state_f.update(out_f)
# Take one step to generate debug dumps
diag_f, out_f_step = dycore_f(state_f, timestep=timedelta(minutes=5))

dycore_j = GFSDynamicsJAX()
state_j = climt.get_default_state([dycore_j], grid_state=grid)
out_j = dcmip(state_j)
state_j.update(out_j)
diag_j, out_j_step = dycore_j(state_j, timestep=timedelta(minutes=5))

trans_config = dycore_j.trans_config

u0 = jnp.array(state_j["eastward_wind"])
v0 = jnp.array(state_j["northward_wind"])
temp0 = jnp.array(state_j["air_temperature"])
ps0 = jnp.array(state_j["surface_air_pressure"])
q0 = jnp.array(state_j["specific_humidity"])

dump_path = Path("debug_data/fortran_step_1_stage_0.bin")
if dump_path.exists():
    data = dump_path.read_bytes()
    offset = 0

    def read_arr(shape, dtype=np.float64):
        global offset
        n = int(np.prod(shape)) * np.dtype(dtype).itemsize
        arr = np.frombuffer(data[offset : offset + n], dtype=dtype).copy()
        offset += n
        return arr.reshape(shape[::-1]).T

    f_ug = read_arr((NLONS, NLATS, NLEVS)).transpose(2, 1, 0)[:, ::-1, :]
    f_vg = read_arr((NLONS, NLATS, NLEVS)).transpose(2, 1, 0)[:, ::-1, :]
    read_arr((NLONS, NLATS, NLEVS))  # virtempg
    read_arr((NLONS, NLATS))  # lnpsg
    read_arr((NLONS, NLATS, NLEVS, 1))  # tracerg
    read_arr((NDIMSPEC, NLEVS), dtype=np.complex128)  # vrtspec
    read_arr((NDIMSPEC, NLEVS), dtype=np.complex128)  # divspec

    # Build JAX spectral state using OLD method (current code)
    grid_orig = GridState(
        u=u0,
        v=v0,
        temperature=temp0,
        vorticity=jnp.zeros_like(u0),
        divergence=jnp.zeros_like(u0),
        log_surface_pressure=jnp.log(ps0),
        tracers=jnp.stack([q0], axis=0),
    )
    spec_old = grid_to_spectral(grid_orig, trans_config)
    grid_old, _ = spectral_to_grid(spec_old, trans_config)
    u_old = np.asarray(grid_old.u.real)
    v_old = np.asarray(grid_old.v.real)

    # Build JAX spectral state using NEW method
    # We need to manually do the forward transform with the new method
    # for each level, then do the inverse
    n_lev = u0.shape[0]
    vrt_new_all = []
    div_new_all = []
    for k in range(n_lev):
        vrt_k, div_k = new_forward(u0[k], v0[k])
        from gfs_dynamical_core.jax.transforms import enforce_triangular_truncation

        T = trans_config.truncation
        vrt_k = enforce_triangular_truncation(vrt_k, L, T)
        div_k = enforce_triangular_truncation(div_k, L, T)
        vrt_new_all.append(vrt_k)
        div_new_all.append(div_k)

    vrt_new_spec = jnp.stack(vrt_new_all)
    div_new_spec = jnp.stack(div_new_all)

    # Inverse transform to get u, v
    def inverse_level(vrt, div):
        F1 = inv_l_factor[:, None] * (div + 1j * vrt) * radius
        f1 = s2fft.inverse_jax(F1, L, spin=1, sampling=sampling)
        return f1.imag, -f1.real

    import jax

    u_new_all, v_new_all = jax.vmap(inverse_level)(vrt_new_spec, div_new_spec)
    u_new = np.asarray(u_new_all.real)
    v_new = np.asarray(v_new_all.real)

    print(f"\n  Old method vs Fortran stage-0 dump:")
    print(
        f"    u max|diff|: {np.abs(u_old - f_ug).max():.6e}  (rel={np.abs(u_old - f_ug).max() / np.abs(f_ug).max():.6e})"
    )
    print(
        f"    v max|diff|: {np.abs(v_old - f_vg).max():.6e}  (rel={np.abs(v_old - f_vg).max() / max(np.abs(f_vg).max(), 1e-30):.6e})"
    )

    print(f"\n  New method vs Fortran stage-0 dump:")
    print(
        f"    u max|diff|: {np.abs(u_new - f_ug).max():.6e}  (rel={np.abs(u_new - f_ug).max() / np.abs(f_ug).max():.6e})"
    )
    print(
        f"    v max|diff|: {np.abs(v_new - f_vg).max():.6e}  (rel={np.abs(v_new - f_vg).max() / max(np.abs(f_vg).max(), 1e-30):.6e})"
    )

    u_improve = np.abs(u_old - f_ug).max() / max(np.abs(u_new - f_ug).max(), 1e-30)
    v_improve = np.abs(v_old - f_vg).max() / max(np.abs(v_new - f_vg).max(), 1e-30)
    print(f"\n  Improvement: u {u_improve:.1f}x,  v {v_improve:.1f}x")

    # Per-level comparison
    print(f"\n  Per-level u comparison (old vs new vs Fortran):")
    for k in [0, 5, 10, 13, 19]:
        old_err = np.abs(u_old[k] - f_ug[k]).max()
        new_err = np.abs(u_new[k] - f_ug[k]).max()
        print(
            f"    level {k:2d}: old={old_err:.6e}  new={new_err:.6e}  improve={old_err / max(new_err, 1e-30):.1f}x"
        )

    print(f"\n  Per-level v comparison (old vs new vs Fortran):")
    for k in [0, 5, 10, 13, 19]:
        old_err = np.abs(v_old[k] - f_vg[k]).max()
        new_err = np.abs(v_new[k] - f_vg[k]).max()
        print(
            f"    level {k:2d}: old={old_err:.6e}  new={new_err:.6e}  improve={old_err / max(new_err, 1e-30):.1f}x"
        )
else:
    print("  Fortran debug dump not found, skipping comparison.")

# ═══════════════════════════════════════════════════════════════════════
# TEST 4: Verify that the fix also works for tendencies
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("TEST 4: Tendency forward transform (grid_to_spectral_tendencies)")
print("=" * 80)

# The tendency transform has the same structure:
#   f_spin1 = -v_flux + i*u_flux
#   F1_lm = forward_spin1(f_spin1)
#   D_plus_izeta = l_factor * F1_lm / R
#   d_vort = -D_plus_izeta.real   (WRONG)
#   d_div  =  D_plus_izeta.imag   (WRONG)
#
# Fix:
#   f_plus  = -v_flux + i*u_flux  -> spin+1 forward -> result_p = D+izeta
#   f_minus =  v_flux + i*u_flux  -> spin-1 forward -> result_m = D-izeta
#   div_of_flux  = (result_p + result_m) / 2
#   curl_of_flux = (result_p - result_m) / (2i)
#   d_vort = -div_of_flux
#   d_div  =  curl_of_flux


def old_tendency_forward(u_flux, v_flux):
    """Old (buggy) tendency forward transform."""
    f_spin1 = -v_flux + 1j * u_flux
    F1_lm = s2fft.forward_jax(f_spin1, L, spin=1, sampling=sampling)
    D_plus_izeta = l_factor[:, None] * F1_lm / radius
    d_vort = -D_plus_izeta.real
    d_div = D_plus_izeta.imag
    return d_vort, d_div


def new_tendency_forward(u_flux, v_flux):
    """New (fixed) tendency forward transform."""
    f_plus = -v_flux + 1j * u_flux
    f_minus = v_flux + 1j * u_flux

    F1_lm = s2fft.forward_jax(f_plus, L, spin=1, sampling=sampling)
    Fm1_lm = s2fft.forward_jax(f_minus, L, spin=-1, sampling=sampling)

    result_p = l_factor[:, None] * F1_lm / radius
    result_m = l_factor[:, None] * Fm1_lm / radius

    div_of_flux = (result_p + result_m) / 2
    curl_of_flux = (result_p - result_m) / (2j)

    d_vort = -div_of_flux
    d_div = curl_of_flux
    return d_vort, d_div


# Test with a known flux field
u_flux = jnp.array(
    3.0 * np.cos(lat_grid) * np.sin(lon_grid) + 1.0 * np.sin(2 * lat_grid)
)
v_flux = jnp.array(2.0 * np.sin(lat_grid) * np.cos(2 * lon_grid))

# Get spectral tendencies
dvort_old, ddiv_old = old_tendency_forward(u_flux, v_flux)
dvort_new, ddiv_new = new_tendency_forward(u_flux, v_flux)

# Roundtrip test: convert tendency to u,v, then back
u_from_old, v_from_old = inverse(dvort_old, ddiv_old)
u_from_new, v_from_new = inverse(dvort_new, ddiv_new)

# The "correct" answer: new forward should give exact roundtrip
dvort_rt, ddiv_rt = new_tendency_forward(
    jnp.array(np.asarray(u_from_new.real)), jnp.array(np.asarray(v_from_new.real))
)

print(f"\n  Tendency roundtrip (dvort):")
print(f"    new method, l=2 m=1: {complex(dvort_new[2, L]):.6e}")
print(f"    after RT,   l=2 m=1: {complex(dvort_rt[2, L]):.6e}")
print(f"    diff: {abs(complex(dvort_new[2, L]) - complex(dvort_rt[2, L])):.6e}")

print(
    f"\n  Old method: dvort(l=2,m=1) has zero imag: {abs(complex(dvort_old[2, L]).imag) < 1e-25}"
)
print(
    f"  New method: dvort(l=2,m=1) has zero imag: {abs(complex(dvort_new[2, L]).imag) < 1e-25}"
)

# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
BUG: The grid_to_spectral and grid_to_spectral_tendencies functions combine
divergence and vorticity spectral coefficients into a single complex number
(D + i*zeta), then try to separate them via .real and .imag. This LOSES
INFORMATION for m!=0 because D_lm and zeta_lm are themselves complex.

FIX: Use BOTH spin+1 and spin-1 forward transforms:
  spin+1 forward of (-v + i*u) -> l_factor * F1/R  = D + i*zeta
  spin-1 forward of ( v + i*u) -> l_factor * Fm1/R = D - i*zeta

  D_lm    = (result_p + result_m) / 2
  zeta_lm = (result_p - result_m) / (2i)

This correctly recovers both the real and imaginary parts of the complex
spectral coefficients for ALL m values.

The inverse transform (spectral_to_grid) is UNAFFECTED because it correctly
constructs (div + i*vrt) and uses a single spin+1 inverse. The combination
of two complex arrays into one doesn't lose information when going TO grid
space (grid values are real, so the information is preserved).

FILES TO MODIFY:
  gfs_dynamical_core/jax/transforms.py:
    - grid_to_spectral: fix transform_level function
    - grid_to_spectral_tendencies: fix transform_vector_tendencies function
""")
