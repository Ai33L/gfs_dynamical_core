"""
Focused test comparing s2fft vs SHTNS vector (spin-1) spherical harmonic transforms.

This script diagnoses the discrepancy between the s2fft-based JAX implementation
and the SHTNS-based Fortran implementation for converting between:
  - (vorticity_spectral, divergence_spectral) <-> (u_grid, v_grid)

The key operations tested:
  1. getuv:          (vrtspec, divspec) -> (u, v)   [inverse / synthesis]
  2. getvrtdivspec:  (u, v) -> (vrtspec, divspec)   [forward / analysis]

We use the DCMIP baroclinic wave initial conditions to get a realistic test case
with known spectral coefficients from both implementations.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

from datetime import timedelta

import climt
import jax.numpy as jnp
import numpy as np
import s2fft
from sympl import set_constant

from gfs_dynamical_core import GFSDynamicalCore
from gfs_dynamical_core.component_jax import GFSDynamicsJAX
from gfs_dynamical_core.jax.states import GridState
from gfs_dynamical_core.jax.transforms import (
    TransformConfig,
    enforce_triangular_truncation,
    get_gaussian_latitudes,
    grid_to_spectral,
    spectral_to_grid,
)

# ── Parameters ───────────────────────────────────────────────────────────
L = 64
NLONS = 2 * L - 1  # 127
NLATS = L  # 64
NLEVS = 20
NTRAC = 1
NTRUNC = int(NLONS / 3 - 2)  # 40
NDIMSPEC = (NTRUNC + 1) * (NTRUNC + 2) // 2  # 861

set_constant("reference_air_pressure", value=1e5, units="Pa")

# ═════════════════════════════════════════════════════════════════════════
# Section 1: Get initial conditions from both implementations
# ═════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("Setting up initial conditions")
print("=" * 80)

grid = climt.get_grid(nx=NLONS, ny=NLATS, nz=NLEVS)
dcmip = climt.DcmipInitialConditions(add_perturbation=True)

# Fortran setup
dycore_f = GFSDynamicalCore()
state_f = climt.get_default_state([dycore_f], grid_state=grid)
out_f = dcmip(state_f)
state_f.update(out_f)

# JAX setup
dycore_j = GFSDynamicsJAX()
state_j = climt.get_default_state([dycore_j], grid_state=grid)
out_j = dcmip(state_j)
state_j.update(out_j)

# Take one Fortran step to trigger initialization and generate debug dumps
timestep = timedelta(minutes=5)
diag_f, out_f_step = dycore_f(state_f, timestep=timestep)

# Take one JAX step to trigger initialization
diag_j, out_j_step = dycore_j(state_j, timestep=timestep)

trans_config = dycore_j.trans_config
dyn_config = dycore_j.dyn_config

# ═════════════════════════════════════════════════════════════════════════
# Section 2: Load Fortran debug dump (stage 0 = initial state after spec->grid)
# ═════════════════════════════════════════════════════════════════════════

from pathlib import Path


def load_fortran_dump(path):
    """Load Fortran debug binary and return dict of arrays in JAX convention."""
    data = Path(path).read_bytes()
    offset = 0

    def read_array(shape, dtype=np.float64):
        nonlocal offset
        n = int(np.prod(shape)) * np.dtype(dtype).itemsize
        arr = np.frombuffer(data[offset : offset + n], dtype=dtype).copy()
        offset += n
        arr = arr.reshape(shape[::-1]).T
        return arr

    result = {}
    result["ug"] = read_array((NLONS, NLATS, NLEVS))
    result["vg"] = read_array((NLONS, NLATS, NLEVS))
    result["virtempg"] = read_array((NLONS, NLATS, NLEVS))
    result["lnpsg"] = read_array((NLONS, NLATS))
    result["tracerg"] = read_array((NLONS, NLATS, NLEVS, NTRAC))
    result["vrtspec"] = read_array((NDIMSPEC, NLEVS), dtype=np.complex128)
    result["divspec"] = read_array((NDIMSPEC, NLEVS), dtype=np.complex128)
    result["virtempspec"] = read_array((NDIMSPEC, NLEVS), dtype=np.complex128)
    result["lnpsspec"] = read_array((NDIMSPEC,), dtype=np.complex128)

    # Reorder to (lev, lat, lon) with lat N->S (s2fft convention)
    for name in ["ug", "vg", "virtempg"]:
        arr = result[name].transpose(2, 1, 0)  # (nlevs, nlats, nlons)
        arr = arr[:, ::-1, :]  # flip lat S->N to N->S
        result[name] = arr

    result["lnpsg"] = result["lnpsg"].T[::-1, :]

    arr = result["tracerg"].transpose(3, 2, 1, 0)
    arr = arr[:, :, ::-1, :]
    result["tracerg"] = arr

    for name in ["vrtspec", "divspec", "virtempspec"]:
        result[name] = result[name].T  # (nlevs, ndimspec)

    return result


dump_path = Path("debug_data/fortran_step_1_stage_0.bin")
if not dump_path.exists():
    print(f"ERROR: {dump_path} not found!")
    exit(1)

f_dump = load_fortran_dump(dump_path)

print(
    f"\nFortran dump loaded: u range = [{f_dump['ug'].min():.4f}, {f_dump['ug'].max():.4f}]"
)
print(
    f"                     v range = [{f_dump['vg'].min():.6f}, {f_dump['vg'].max():.6f}]"
)

# ═════════════════════════════════════════════════════════════════════════
# Section 3: Build JAX spectral state and reconstruct u, v
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("Section 3: JAX spectral -> grid vector transform")
print("=" * 80)

# Build the JAX spectral state from the initial grid fields
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
grid0_jax, _ = spectral_to_grid(spec0, trans_config)

# Compare u, v from JAX spectral roundtrip vs Fortran dump
print("\n--- JAX spectral roundtrip vs Fortran stage-0 dump ---")
u_jax = np.asarray(grid0_jax.u.real)
v_jax = np.asarray(grid0_jax.v.real)
u_fort = f_dump["ug"]
v_fort = f_dump["vg"]

print(
    f"  u: max|diff| = {np.abs(u_jax - u_fort).max():.6e}  (max|F| = {np.abs(u_fort).max():.6e})"
)
print(
    f"  v: max|diff| = {np.abs(v_jax - v_fort).max():.6e}  (max|F| = {np.abs(v_fort).max():.6e})"
)

# ═════════════════════════════════════════════════════════════════════════
# Section 4: Test with a simple analytical vector field
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("Section 4: Analytical test — solid body rotation u=cos(lat), v=0")
print("=" * 80)

latitudes = get_gaussian_latitudes(L)
lat_grid = np.array(latitudes)[:, None]  # (nlat, 1)
lon_grid = np.linspace(0, 2 * np.pi, NLONS, endpoint=False)[None, :]  # (1, nlon)

# Solid body rotation: u = u0*cos(lat), v = 0
u0_val = 10.0
u_analytic = u0_val * np.cos(lat_grid) * np.ones_like(lon_grid)  # (nlat, nlon)
v_analytic = np.zeros_like(u_analytic)

# Forward transform (u, v) -> (vrt_spec, div_spec) using JAX/s2fft
u_j = jnp.array(u_analytic)
v_j = jnp.array(v_analytic)

sampling = trans_config.sampling
radius = trans_config.radius
l_arr = jnp.arange(L)
l_factor = jnp.sqrt(l_arr * (l_arr + 1))
inv_l_factor = jnp.where(l_arr > 0, 1.0 / l_factor, 0.0)

# Forward: grid (u,v) -> spectral (vrt, div)
f_spin1_fwd = -v_j + 1j * u_j
F1_lm_fwd = s2fft.forward_jax(f_spin1_fwd, L, spin=1, sampling=sampling)
D_plus_izeta_fwd = l_factor[:, None] * F1_lm_fwd / radius
vrt_spec = D_plus_izeta_fwd.imag
div_spec = D_plus_izeta_fwd.real

print(
    f"  Solid body rotation: max|vrt_spec| = {np.abs(np.asarray(vrt_spec)).max():.6e}"
)
print(
    f"                       max|div_spec| = {np.abs(np.asarray(div_spec)).max():.6e}"
)
print(f"  (Expected: vrt ≠ 0 for solid body rotation, div ≈ 0)")

# Inverse: spectral (vrt, div) -> grid (u, v)
F1_lm_inv = inv_l_factor[:, None] * (div_spec + 1j * vrt_spec) * radius
f_spin1_inv = s2fft.inverse_jax(F1_lm_inv, L, spin=1, sampling=sampling)
u_recovered = f_spin1_inv.imag
v_recovered = -f_spin1_inv.real

u_rec = np.asarray(u_recovered.real)
v_rec = np.asarray(v_recovered.real)

print(f"\n  Roundtrip: max|u - u_orig| = {np.abs(u_rec - u_analytic).max():.6e}")
print(f"             max|v - v_orig| = {np.abs(v_rec - v_analytic).max():.6e}")

# ═════════════════════════════════════════════════════════════════════════
# Section 5: Test the Condon-Shortley phase issue
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("Section 5: Condon-Shortley phase analysis")
print("=" * 80)

# SHTNS is initialized with SHT_NO_CS_PHASE (see shtns.f90 line ~158)
# s2fft uses Condon-Shortley phase by default
# This means for m>0: Y_l^m(s2fft) = (-1)^m * Y_l^m(SHTNS)
#
# For scalar transforms, this is self-consistent (forward and inverse cancel).
# But for spin-1 transforms, the phase convention may differ.

# Let's check: take the Fortran's spectral vrt/div and reconstruct u,v using
# the JAX spin-1 inverse, and see if we get the Fortran's u,v.

# First, we need to convert Fortran's packed triangular spectral coefficients
# to s2fft's (L, 2L-1) layout.


def shtns_packed_to_s2fft(packed_coeffs, L, ntrunc):
    """
    Convert SHTNS packed triangular coefficients to s2fft (L, 2L-1) layout.

    SHTNS packed format (with orthonormal + no CS phase):
      Index ordering: (m=0,l=0..ntrunc), (m=1,l=1..ntrunc), ..., (m=ntrunc,l=ntrunc)
      Total: (ntrunc+1)*(ntrunc+2)/2 complex coefficients

    s2fft layout (L, 2L-1):
      Row l (0..L-1), column m (-L+1..L-1)
      Column index j = m + L - 1
      For m=0: j = L-1
      For m>0: j = m + L - 1
      For m<0: j = m + L - 1

    SHTNS stores only m>=0 coefficients. For real fields:
      f_l^{-m} = (-1)^m * conj(f_l^m)  (with CS phase)
    But SHTNS uses NO_CS_PHASE, so:
      f_l^{-m} = conj(f_l^m)
    """
    flm = np.zeros((L, 2 * L - 1), dtype=np.complex128)

    idx = 0
    for m in range(ntrunc + 1):
        for l in range(m, ntrunc + 1):
            if l < L:
                j_pos = m + L - 1  # positive m column
                j_neg = -m + L - 1  # negative m column

                coeff = packed_coeffs[idx]

                # s2fft uses CS phase, SHTNS does not.
                # s2fft: f_l^m (with CS) = (-1)^m * f_l^m (without CS)
                # So we need to multiply by (-1)^m to convert SHTNS -> s2fft
                cs_factor = (-1) ** m
                flm[l, j_pos] = cs_factor * coeff

                if m > 0:
                    # Negative m: s2fft convention for real fields
                    # With CS phase: f_l^{-m} = (-1)^m * conj(f_l^m)
                    # In s2fft convention: f_l^{-m} = (-1)^m * conj(s2fft_f_l^m)
                    #   = (-1)^m * conj((-1)^m * shtns_f_l^m)
                    #   = conj(shtns_f_l^m)
                    flm[l, j_neg] = np.conj(coeff)

            idx += 1

    assert idx == (ntrunc + 1) * (ntrunc + 2) // 2
    return flm


# Convert Fortran spectral coefficients to s2fft layout
print("\nConverting Fortran spectral coefficients to s2fft layout...")

# Pick one level to test (level 13 = max wind)
test_level = 13
vrt_fort_packed = f_dump["vrtspec"][test_level]
div_fort_packed = f_dump["divspec"][test_level]
temp_fort_packed = f_dump["virtempspec"][test_level]

vrt_fort_s2fft = shtns_packed_to_s2fft(vrt_fort_packed, L, NTRUNC)
div_fort_s2fft = shtns_packed_to_s2fft(div_fort_packed, L, NTRUNC)
temp_fort_s2fft = shtns_packed_to_s2fft(temp_fort_packed, L, NTRUNC)

# Test scalar reconstruction first: temperature
print(f"\n--- Scalar test: temperature at level {test_level} ---")
temp_grid_s2fft = np.asarray(
    s2fft.inverse_jax(jnp.array(temp_fort_s2fft), L, sampling=sampling)
)
temp_grid_fort = f_dump["virtempg"][test_level]
temp_diff = np.abs(temp_grid_s2fft.real - temp_grid_fort).max()
temp_rel = temp_diff / np.abs(temp_grid_fort).max()
print(
    f"  T from Fortran spec via s2fft: max|diff| = {temp_diff:.6e}  rel = {temp_rel:.6e}"
)

# Now test vector reconstruction: (vrt, div) -> (u, v)
print(f"\n--- Vector test: u,v at level {test_level} ---")
vrt_j = jnp.array(vrt_fort_s2fft)
div_j = jnp.array(div_fort_s2fft)

# Method A: same as spectral_to_grid
F1_lm_A = inv_l_factor[:, None] * (div_j + 1j * vrt_j) * radius
f_spin1_A = s2fft.inverse_jax(F1_lm_A, L, spin=1, sampling=sampling)
u_A = np.asarray(f_spin1_A.imag)
v_A = np.asarray(-f_spin1_A.real)

u_fort_lev = f_dump["ug"][test_level]
v_fort_lev = f_dump["vg"][test_level]

u_diff_A = np.abs(u_A - u_fort_lev).max()
v_diff_A = np.abs(v_A - v_fort_lev).max()
print(
    f"  Method A (standard):  u max|diff| = {u_diff_A:.6e}  (rel = {u_diff_A / np.abs(u_fort_lev).max():.6e})"
)
print(
    f"                        v max|diff| = {v_diff_A:.6e}  (rel = {v_diff_A / max(np.abs(v_fort_lev).max(), 1e-30):.6e})"
)


# Method B: try without CS phase correction (no (-1)^m)
def shtns_packed_to_s2fft_no_cs(packed_coeffs, L, ntrunc):
    """Same but without the (-1)^m CS phase correction."""
    flm = np.zeros((L, 2 * L - 1), dtype=np.complex128)
    idx = 0
    for m in range(ntrunc + 1):
        for l in range(m, ntrunc + 1):
            if l < L:
                j_pos = m + L - 1
                j_neg = -m + L - 1
                coeff = packed_coeffs[idx]
                flm[l, j_pos] = coeff
                if m > 0:
                    # Without CS: f_l^{-m} = (-1)^m * conj(f_l^m) for s2fft
                    flm[l, j_neg] = (-1) ** m * np.conj(coeff)
            idx += 1
    return flm


vrt_fort_s2fft_B = shtns_packed_to_s2fft_no_cs(vrt_fort_packed, L, NTRUNC)
div_fort_s2fft_B = shtns_packed_to_s2fft_no_cs(div_fort_packed, L, NTRUNC)

vrt_jB = jnp.array(vrt_fort_s2fft_B)
div_jB = jnp.array(div_fort_s2fft_B)
F1_lm_B = inv_l_factor[:, None] * (div_jB + 1j * vrt_jB) * radius
f_spin1_B = s2fft.inverse_jax(F1_lm_B, L, spin=1, sampling=sampling)
u_B = np.asarray(f_spin1_B.imag)
v_B = np.asarray(-f_spin1_B.real)

u_diff_B = np.abs(u_B - u_fort_lev).max()
v_diff_B = np.abs(v_B - v_fort_lev).max()
print(
    f"\n  Method B (no CS fix): u max|diff| = {u_diff_B:.6e}  (rel = {u_diff_B / np.abs(u_fort_lev).max():.6e})"
)
print(
    f"                        v max|diff| = {v_diff_B:.6e}  (rel = {v_diff_B / max(np.abs(v_fort_lev).max(), 1e-30):.6e})"
)

# Method C: try swapping real/imag in the spin-1 inverse
F1_lm_C = inv_l_factor[:, None] * (vrt_j + 1j * div_j) * radius  # swap vrt/div
f_spin1_C = s2fft.inverse_jax(F1_lm_C, L, spin=1, sampling=sampling)
u_C = np.asarray(f_spin1_C.imag)
v_C = np.asarray(-f_spin1_C.real)

u_diff_C = np.abs(u_C - u_fort_lev).max()
v_diff_C = np.abs(v_C - v_fort_lev).max()
print(
    f"\n  Method C (swap vrt/div): u max|diff| = {u_diff_C:.6e}  (rel = {u_diff_C / np.abs(u_fort_lev).max():.6e})"
)
print(f"                           v max|diff| = {v_diff_C:.6e}")

# Method D: try different sign conventions
F1_lm_D = inv_l_factor[:, None] * (div_j - 1j * vrt_j) * radius  # flip imag sign
f_spin1_D = s2fft.inverse_jax(F1_lm_D, L, spin=1, sampling=sampling)
u_D = np.asarray(f_spin1_D.imag)
v_D = np.asarray(-f_spin1_D.real)

u_diff_D = np.abs(u_D - u_fort_lev).max()
v_diff_D = np.abs(v_D - v_fort_lev).max()
print(
    f"\n  Method D (flip imag):  u max|diff| = {u_diff_D:.6e}  (rel = {u_diff_D / np.abs(u_fort_lev).max():.6e})"
)
print(f"                         v max|diff| = {v_diff_D:.6e}")

# Method E: try v = +real instead of -real
F1_lm_E = inv_l_factor[:, None] * (div_j + 1j * vrt_j) * radius
f_spin1_E = s2fft.inverse_jax(F1_lm_E, L, spin=1, sampling=sampling)
u_E = np.asarray(f_spin1_E.imag)
v_E = np.asarray(f_spin1_E.real)  # +real instead of -real

u_diff_E = np.abs(u_E - u_fort_lev).max()
v_diff_E = np.abs(v_E - v_fort_lev).max()
print(
    f"\n  Method E (v=+real):    u max|diff| = {u_diff_E:.6e}  (rel = {u_diff_E / np.abs(u_fort_lev).max():.6e})"
)
print(f"                         v max|diff| = {v_diff_E:.6e}")

# Method F: u = -imag
F1_lm_F = inv_l_factor[:, None] * (div_j + 1j * vrt_j) * radius
f_spin1_F = s2fft.inverse_jax(F1_lm_F, L, spin=1, sampling=sampling)
u_F = np.asarray(-f_spin1_F.imag)
v_F = np.asarray(-f_spin1_F.real)

u_diff_F = np.abs(u_F - u_fort_lev).max()
v_diff_F = np.abs(v_F - v_fort_lev).max()
print(
    f"\n  Method F (u=-imag):    u max|diff| = {u_diff_F:.6e}  (rel = {u_diff_F / np.abs(u_fort_lev).max():.6e})"
)
print(f"                         v max|diff| = {v_diff_F:.6e}")

# ═════════════════════════════════════════════════════════════════════════
# Section 6: Try all 8 sign combinations systematically
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("Section 6: Systematic search over sign/swap conventions")
print("=" * 80)

# The spin-1 inverse gives a complex field f = f_real + i*f_imag
# u and v are some combination of {±f_real, ±f_imag}
# Also try swapping div/vrt in the input

best_u_err = 1e30
best_v_err = 1e30
best_combo = None

print(f"\n{'combo':>45s}  {'u rel err':>12s}  {'v rel err':>12s}  {'total':>12s}")
print("-" * 95)

for cs_mode in ["with_cs", "no_cs"]:
    if cs_mode == "with_cs":
        vrt_s = jnp.array(shtns_packed_to_s2fft(vrt_fort_packed, L, NTRUNC))
        div_s = jnp.array(shtns_packed_to_s2fft(div_fort_packed, L, NTRUNC))
    else:
        vrt_s = jnp.array(shtns_packed_to_s2fft_no_cs(vrt_fort_packed, L, NTRUNC))
        div_s = jnp.array(shtns_packed_to_s2fft_no_cs(div_fort_packed, L, NTRUNC))

    for swap in [False, True]:
        if swap:
            a, b = vrt_s, div_s
        else:
            a, b = div_s, vrt_s

        for real_sign in [1, -1]:
            for imag_sign in [1, -1]:
                for uv_swap in [False, True]:
                    F1 = inv_l_factor[:, None] * (a + 1j * b) * radius
                    f1 = s2fft.inverse_jax(F1, L, spin=1, sampling=sampling)

                    comp_r = np.asarray(real_sign * f1.real)
                    comp_i = np.asarray(imag_sign * f1.imag)

                    if uv_swap:
                        u_test, v_test = comp_r, comp_i
                    else:
                        u_test, v_test = comp_i, comp_r

                    u_err = np.abs(u_test - u_fort_lev).max() / np.abs(u_fort_lev).max()
                    v_err = np.abs(v_test - v_fort_lev).max() / max(
                        np.abs(v_fort_lev).max(), 1e-30
                    )
                    total = u_err + v_err

                    label = (
                        f"{cs_mode} swap={swap} rs={real_sign:+d} is={imag_sign:+d}"
                        f" uv_swap={uv_swap}"
                    )

                    flag = ""
                    if total < best_u_err + best_v_err:
                        best_u_err = u_err
                        best_v_err = v_err
                        best_combo = label
                        flag = " <-- BEST"

                    if u_err < 0.01 or v_err < 0.01 or total < 1.0:
                        print(
                            f"  {label:>43s}  {u_err:12.6e}  {v_err:12.6e}  {total:12.6e}{flag}"
                        )

print(f"\n  BEST: {best_combo}")
print(f"        u rel err = {best_u_err:.6e},  v rel err = {best_v_err:.6e}")

# ═════════════════════════════════════════════════════════════════════════
# Section 7: Direct comparison of JAX spectral coefficients vs Fortran
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("Section 7: Compare JAX spectral coefficients vs Fortran packed coefficients")
print("=" * 80)

# The JAX grid_to_spectral produces vort and div in s2fft (L, 2L-1) layout.
# Let's compare these with the Fortran's packed coefficients converted to s2fft layout.

vrt_jax_lev = np.asarray(spec0.vorticity[test_level])  # (L, 2L-1) complex
div_jax_lev = np.asarray(spec0.divergence[test_level])

# Compare m=0 coefficients (these should not be affected by CS phase)
print(f"\n  Level {test_level}, m=0 coefficients comparison:")
print(
    f"  {'l':>3s}  {'vrt_F real':>14s}  {'vrt_J real':>14s}  {'ratio':>10s}  {'div_F real':>14s}  {'div_J real':>14s}  {'ratio':>10s}"
)
j_m0 = L - 1  # column index for m=0 in s2fft

for l in range(min(10, NTRUNC + 1)):
    vf = vrt_fort_s2fft[l, j_m0].real
    vj = vrt_jax_lev[l, j_m0].real
    vr = vj / vf if abs(vf) > 1e-30 else 0
    df = div_fort_s2fft[l, j_m0].real
    dj = div_jax_lev[l, j_m0].real
    dr = dj / df if abs(df) > 1e-30 else 0
    print(
        f"  {l:3d}  {vf:14.6e}  {vj:14.6e}  {vr:10.6f}  {df:14.6e}  {dj:14.6e}  {dr:10.6f}"
    )

# Compare m=1 coefficients (affected by CS phase)
print(f"\n  Level {test_level}, m=1 coefficients comparison:")
print(f"  {'l':>3s}  {'vrt_F':>28s}  {'vrt_J':>28s}  {'ratio_r':>10s}")
j_m1 = 1 + L - 1  # column index for m=+1

for l in range(1, min(10, NTRUNC + 1)):
    vf = vrt_fort_s2fft[l, j_m1]
    vj = vrt_jax_lev[l, j_m1]
    rr = vj.real / vf.real if abs(vf.real) > 1e-30 else 0
    ri = vj.imag / vf.imag if abs(vf.imag) > 1e-30 else 0
    print(
        f"  {l:3d}  {vf.real:+13.6e}{vf.imag:+13.6e}j  {vj.real:+13.6e}{vj.imag:+13.6e}j  r={rr:+8.4f} i={ri:+8.4f}"
    )

# ═════════════════════════════════════════════════════════════════════════
# Section 8: SHTNS getuv formula analysis
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("Section 8: SHTNS getuv formula analysis")
print("=" * 80)

# From the Fortran shtns.f90:
#   subroutine getuv(vrtspec, divspec, ugrid, vgrid, rsphere)
#     vrtspec_tmp = invlap * rsphere * vrtspec   ! invlap = 1/(-l(l+1))
#     divspec_tmp = invlap * rsphere * divspec
#     call shtns_sphtor_to_spat(vrtspec_tmp, divspec_tmp, ugrid, vgrid)
#
# So SHTNS gets: Slm = invlap*R*vrtspec = -R/(l(l+1)) * vrtspec = -R * vrtspec / (l(l+1))
#                Tlm = invlap*R*divspec  = -R/(l(l+1)) * divspec
#
# SHTNS_sphtor_to_spat(Slm, Tlm, Vtheta, Vphi) computes:
#   V_theta, V_phi from stream function Slm and velocity potential Tlm
#
# The output ugrid, vgrid are then:
#   ugrid = V_theta component (= southward? or northward?)
#   vgrid = V_phi component (= eastward)
#
# Wait — the Fortran calls ugrid the FIRST output and vgrid the SECOND.
# SHTNS convention: first output is theta component, second is phi component.
# But in GFS, ug = eastward wind (phi direction), vg = northward wind (lat direction).
#
# So either:
#   a) SHTNS theta component = eastward wind (unlikely, would be wrong)
#   b) The Fortran's variable naming is misleading
#   c) SHTNS uses a non-standard ordering
#
# Let's test empirically!

print("\nEmpirical test: which SHTNS output is which wind component?")
print("Using solid body rotation u_east=cos(lat), v_north=0")
print("")

# For solid body rotation u=cos(lat), v=0:
# vorticity = 2*u0*sin(lat)/R (vrt_l=1,m=0 is nonzero)
# divergence = 0

# The stream function psi satisfies: vrt = -lap(psi)/R^2 = l(l+1)/R^2 * psi
# So psi = R^2 * vrt / (l(l+1))
# And invlap*R*vrt = -R/(l(l+1)) * vrt = -psi/R

# From psi, the velocity components in spherical coords are:
# u_phi (eastward) = -(1/R) * d(psi)/d(theta)  [with SHTNS convention]
# u_theta (southward) = -(1/(R*sin(theta))) * d(psi)/d(phi)

# For axisymmetric psi (m=0 only), d(psi)/d(phi) = 0, so u_theta = 0 and
# u_phi = cos(lat) which is our eastward wind.

# In SHTNS, sphtor_to_spat produces (V_theta, V_phi).
# The Fortran maps: ugrid <- V_theta, vgrid <- V_phi
# But we know u_east = V_phi and v_south = V_theta (or v_north = -V_theta)

# This means: ugrid = V_theta = v_south, vgrid = V_phi = u_east
# The Fortran naming is MISLEADING: ug is NOT the eastward wind directly
# from SHTNS — it's the theta component!

# UNLESS SHTNS uses (Vphi, Vtheta) ordering... Let's check empirically.

# Actually, let me just check if the Fortran u,v at stage 0 make sense for
# the DCMIP test case. The baroclinic wave has strong eastward (zonal) wind
# and very weak meridional wind. Fortran ug has max ~35 m/s (which matches
# the jet stream), so ug IS the eastward wind. This means SHTNS puts the
# phi (eastward) component FIRST, contrary to what we might expect.

# Let's verify by checking the SHTNS documentation more carefully.
# Looking at shtns.h comments:
#   SH_to_spat_cplx(Qlm, Slm, Tlm, Vr, Vt, Vp)
# For the 2D sphtor version:
#   shtns_sphtor_to_spat(Qlm_s, Qlm_t, Vt, Vp)
# Output: Vt = THETA component, Vp = PHI component
#
# But in the Fortran wrapper, the call is:
#   call shtns_sphtor_to_spat(vrtspec_tmp, divspec_tmp, ugrid, vgrid)
# So ugrid = Vt (theta component), vgrid = Vp (phi component)
#
# For the DCMIP baroclinic wave: strong zonal (eastward = phi) wind
# Fortran: max(ug) = 35 m/s, max(vg) ~ 0.001 m/s
# If ugrid=Vt: theta comp should be ~0 for zonal flow  → CONTRADICTION
# If ugrid=Vp: phi comp should be ~35 m/s → but docs say Vt is first!
#
# Resolution: The Fortran SHTNS wrapper must be using PHI_CONTIGUOUS flag
# or SHTNS has a different ordering. Let's check the init:
#   call shtns_set_size(ntrunc, ntrunc, 1, SHT_ORTHONORMAL+SHT_NO_CS_PHASE)
#   call shtns_precompute(SHT_GAUSS_FLY, SHT_PHI_CONTIGUOUS, ...)
#
# SHT_PHI_CONTIGUOUS = 512: this affects memory layout, not math.
#
# Actually, the real answer might be simpler: SHTNS documentation for
# SHTqst_to_spat says it returns (Vt, Vp), but the ACTUAL C function
# signature might use a different convention. Let me check.
#
# From SHTNS source: for shtns_sphtor_to_spat(Slm, Tlm, Vt, Vp):
#   Vt = sum_lm {  dSlm/dtheta * Ylm / sin(theta) ... }
#   Vp = sum_lm { ... }
# The first output IS theta, second IS phi.
#
# So if ugrid gets theta and vgrid gets phi, and the jet is at max ~35 m/s
# in ug, then ug must actually be phi not theta. Something is off.
#
# WAIT: Perhaps the Fortran has (nlons, nlats) ordering and SHTNS is getting
# the arrays transposed due to Fortran's column-major ordering??
#
# SHTNS C function expects: array[phi_index * nlat + theta_index] when
# PHI_CONTIGUOUS. But Fortran array(nlons, nlats) in column-major is
# array[lon_index + nlons * lat_index], which when passed to C by reference
# looks like array[lon_index * nlats + lat_index] — wait, no. C sees
# Fortran's column-major as transposed.
#
# Fortran array(nlons, nlats): stored as a(1,1), a(2,1), ..., a(nlons,1), a(1,2), ...
# C sees this as: c[0][0], c[0][1], ..., c[0][nlons-1], c[1][0], ...
# i.e., c[lat][lon] — so c is (nlats, nlons).
#
# With SHT_PHI_CONTIGUOUS, SHTNS expects arrays with phi (lon) as the
# fastest-varying index. The Fortran layout with (nlons, nlats) gives
# lon as fastest — so it IS phi-contiguous. Good.
#
# So the question remains: why does ug (first output of sphtor_to_spat)
# contain the zonal wind?
#
# Let me re-read the SHTNS doc more carefully. The actual SHTNS convention:
# SHTsphtor_to_spat(Slm, Tlm, Vt, Vp) where:
#   Vt = -1/sin(theta) * d/dphi S + d/dtheta T
#   Vp =  d/dtheta S + 1/sin(theta) * d/dphi T
#
# Wait, that would mean for a purely zonal (m=0) stream function S:
#   Vt = 0 (no phi dependence)
#   Vp = dS/dtheta (= meridional gradient of stream function = zonal wind)
# So Vp IS the zonal wind. And it goes into vgrid (second output), not ugrid.
#
# This contradicts the observation that ug has max 35 m/s...
#
# UNLESS the Fortran is calling them "ug" and "vg" but using them swapped!
# Or there could be a sign flip. Let me just check what SHTNS actually
# produces numerically for a known case.

print("The analysis above suggests a possible swap between u and v")
print("in the SHTNS wrapper. Let's verify by direct numerical test.\n")

# ═════════════════════════════════════════════════════════════════════════
# Section 9: Pure numerical experiment — what does s2fft spin-1 produce?
# ═════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("Section 9: What does s2fft spin-1 inverse actually produce?")
print("=" * 80)

# Create a simple test: l=1, m=0 vorticity only (solid body rotation)
# Vorticity of solid body rotation with angular velocity Ω:
#   ζ = 2Ω (uniform), which projects onto l=0 mode
# Actually for u = u0*cos(lat), ζ = -1/(a cos(lat)) d/d(lat) (u*cos(lat))
# = -1/(a cos lat) d/d(lat) (u0 cos^2 lat) = 2 u0 sin(lat) / a
# This projects onto Y_1^0 ∝ cos(theta) = sin(lat)

# Let's just set vrt_lm with only l=1, m=0 nonzero and see what u,v we get

vrt_test = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
div_test = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)

# Set vrt(l=1, m=0) to some value
vrt_val = 1e-5
vrt_test = vrt_test.at[1, L - 1].set(vrt_val)  # l=1, m=0

# Inverse transform to get u, v via the JAX formula
F1_test = inv_l_factor[:, None] * (div_test + 1j * vrt_test) * radius
f_spin1_test = s2fft.inverse_jax(F1_test, L, spin=1, sampling=sampling)

u_test_std = np.asarray(f_spin1_test.imag)  # standard: u = imag
v_test_std = np.asarray(-f_spin1_test.real)  # standard: v = -real

print(f"\n  Input: vrt(l=1,m=0) = {vrt_val}")
print(f"  Theoretical: u ∝ cos(lat), v = 0")
print(f"\n  Standard (u=imag, v=-real):")
print(f"    u zonal mean at equator (lat~0): {u_test_std[NLATS // 2, :].mean():.6e}")
print(f"    u zonal mean at pole (lat~90):   {u_test_std[0, :].mean():.6e}")
print(
    f"    v zonal mean max abs:             {np.abs(v_test_std.mean(axis=1)).max():.6e}"
)

# Check if u has cos(lat) pattern
u_zm = u_test_std.mean(axis=1)
cos_lat = np.cos(np.array(latitudes))
if np.abs(u_zm).max() > 1e-15:
    u_normalized = u_zm / u_zm.max()
    cos_normalized = cos_lat / cos_lat.max()
    corr = np.corrcoef(u_normalized, cos_normalized)[0, 1]
    print(f"    u ~ cos(lat) correlation: {corr:.6f}")

print(f"\n  Alternative (u=-real, v=imag):")
u_alt = np.asarray(-f_spin1_test.real)
v_alt = np.asarray(f_spin1_test.imag)
print(f"    u zonal mean at equator: {u_alt[NLATS // 2, :].mean():.6e}")
print(f"    v zonal mean max abs:    {np.abs(v_alt.mean(axis=1)).max():.6e}")

# ═════════════════════════════════════════════════════════════════════════
# Section 10: What if the Fortran ug/vg labels are swapped?
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("Section 10: What if Fortran ug=phi wind and vg is theta?")
print("           Check by comparing JAX u with Fortran vg")
print("=" * 80)

# If SHTNS sphtor_to_spat returns (Vtheta, Vphi) and Fortran maps
# ugrid <- Vtheta, vgrid <- Vphi, then:
#   Fortran ug = V_theta (should be ~0 for zonal flow)
#   Fortran vg = V_phi = eastward wind

# But Fortran ug has max 35 m/s — so either:
# 1. SHTNS returns (Vphi, Vtheta), OR
# 2. The Fortran naming is correct (ug = eastward)

# The answer: SHTNS's sphtor_to_spat for SH_PHI_CONTIGUOUS layout
# may use a different ordering. But the empirical evidence says:
#   Fortran ug has the zonal wind → ug = eastward wind
#   This is consistent with how GFS uses these arrays everywhere.

# So the real question is: does s2fft's spin-1 transform match SHTNS?
# Let's compare the JAX u,v (from the JAX's own spectral coefficients)
# with the Fortran u,v (from Fortran's own spectral coefficients)
# and see if the spectral coefficients THEMSELVES differ.

print("\nComparing JAX vs Fortran spectral coefficients (vrt, div):")
print("  (Using the shtns_packed_to_s2fft conversion with CS phase correction)\n")

# Sum of absolute differences per m
for m_test in range(5):
    j_pos = m_test + L - 1
    vrt_f_col = vrt_fort_s2fft[:, j_pos]
    vrt_j_col = vrt_jax_lev[:, j_pos]
    div_f_col = div_fort_s2fft[:, j_pos]
    div_j_col = div_jax_lev[:, j_pos]

    vrt_diff = np.abs(vrt_f_col - vrt_j_col).max()
    div_diff = np.abs(div_f_col - div_j_col).max()
    vrt_max = max(np.abs(vrt_f_col).max(), np.abs(vrt_j_col).max())
    div_max = max(np.abs(div_f_col).max(), np.abs(div_j_col).max())
    vrt_rel = vrt_diff / vrt_max if vrt_max > 1e-30 else 0
    div_rel = div_diff / div_max if div_max > 1e-30 else 0

    print(
        f"  m={m_test}: vrt max|diff|={vrt_diff:.6e} (rel={vrt_rel:.6e})  "
        f"div max|diff|={div_diff:.6e} (rel={div_rel:.6e})"
    )

# Try the no-CS version too
print("\n  Without CS phase correction:")
vrt_fort_noCS = shtns_packed_to_s2fft_no_cs(vrt_fort_packed, L, NTRUNC)
div_fort_noCS = shtns_packed_to_s2fft_no_cs(div_fort_packed, L, NTRUNC)
for m_test in range(5):
    j_pos = m_test + L - 1
    vrt_diff = np.abs(vrt_fort_noCS[:, j_pos] - vrt_jax_lev[:, j_pos]).max()
    div_diff = np.abs(div_fort_noCS[:, j_pos] - div_jax_lev[:, j_pos]).max()
    vrt_max = max(
        np.abs(vrt_fort_noCS[:, j_pos]).max(), np.abs(vrt_jax_lev[:, j_pos]).max()
    )
    div_max = max(
        np.abs(div_fort_noCS[:, j_pos]).max(), np.abs(div_jax_lev[:, j_pos]).max()
    )
    vrt_rel = vrt_diff / vrt_max if vrt_max > 1e-30 else 0
    div_rel = div_diff / div_max if div_max > 1e-30 else 0
    print(
        f"  m={m_test}: vrt max|diff|={vrt_diff:.6e} (rel={vrt_rel:.6e})  "
        f"div max|diff|={div_diff:.6e} (rel={div_rel:.6e})"
    )

# ═════════════════════════════════════════════════════════════════════════
# Section 11: Check scalar transform consistency
# ═════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("Section 11: Scalar transform comparison (sanity check)")
print("=" * 80)

# Temperature spectral coefficients should match between JAX and Fortran
temp_jax_lev = np.asarray(spec0.temperature[test_level])  # s2fft (L, 2L-1)

print(f"\n  Temperature spectral coefficients at level {test_level}:")
for m_test in range(5):
    j_pos = m_test + L - 1
    f_col = temp_fort_s2fft[:, j_pos]
    j_col = temp_jax_lev[:, j_pos]
    diff = np.abs(f_col - j_col).max()
    mx = max(np.abs(f_col).max(), np.abs(j_col).max())
    rel = diff / mx if mx > 1e-30 else 0
    print(f"  m={m_test}: max|diff|={diff:.6e} (rel={rel:.6e})")

# No-CS version
print("\n  Temperature (no CS phase correction):")
temp_fort_noCS = shtns_packed_to_s2fft_no_cs(temp_fort_packed, L, NTRUNC)
for m_test in range(5):
    j_pos = m_test + L - 1
    diff = np.abs(temp_fort_noCS[:, j_pos] - temp_jax_lev[:, j_pos]).max()
    mx = max(
        np.abs(temp_fort_noCS[:, j_pos]).max(), np.abs(temp_jax_lev[:, j_pos]).max()
    )
    rel = diff / mx if mx > 1e-30 else 0
    print(f"  m={m_test}: max|diff|={diff:.6e} (rel={rel:.6e})")


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
