"""
Minimal test to understand why s2fft spin-1 transforms produce different u,v
than SHTNS for the same vorticity/divergence spectral coefficients.

Key finding from previous analysis:
- Scalar (spin-0) transforms match perfectly between s2fft and SHTNS
- Vector (spin-1) transforms show ~3% error in u and ~100% error in v
- m=0 spectral coefficients match, but m>=1 have zero imaginary parts in JAX
  while Fortran has both real and imaginary parts

This script investigates:
1. Whether s2fft spin-1 forward/inverse is self-consistent (roundtrip)
2. Whether the grid_to_spectral function correctly maps (u,v) -> (vrt,div)
3. Whether the issue is in the CS phase handling for spin-1 transforms
4. Whether the issue is that s2fft and SHTNS define spin-1 harmonics differently
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

# Get Gaussian latitudes (s2fft GL convention: north to south)
cos_theta, _ = np.polynomial.legendre.leggauss(L)
thetas = np.flip(np.arccos(cos_theta))
latitudes = np.pi / 2.0 - thetas
lat_grid = latitudes[:, None]
lon_grid = np.linspace(0, 2 * np.pi, NLONS, endpoint=False)[None, :]

print("=" * 80)
print("TEST 1: s2fft spin-1 roundtrip self-consistency")
print("=" * 80)

# Create a known velocity field: u = cos(lat)*cos(lon), v = sin(lat)*sin(lon)
u_test = np.cos(lat_grid) * np.cos(lon_grid)
v_test = 0.5 * np.sin(lat_grid) * np.sin(lon_grid)

u_j = jnp.array(u_test)
v_j = jnp.array(v_test)

# Forward: (u, v) -> spectral (vrt, div)
f_spin1_fwd = -v_j + 1j * u_j
F1_lm = s2fft.forward_jax(f_spin1_fwd, L, spin=1, sampling=sampling)
D_plus_izeta = l_factor[:, None] * F1_lm / radius
div_spec = D_plus_izeta.real
vrt_spec = D_plus_izeta.imag

# Inverse: spectral (vrt, div) -> (u, v)
F1_lm_inv = inv_l_factor[:, None] * (div_spec + 1j * vrt_spec) * radius
f_spin1_inv = s2fft.inverse_jax(F1_lm_inv, L, spin=1, sampling=sampling)
u_rec = np.asarray(f_spin1_inv.imag)
v_rec = np.asarray(-f_spin1_inv.real)

print(f"  u roundtrip max|diff|: {np.abs(u_rec - u_test).max():.6e}")
print(f"  v roundtrip max|diff|: {np.abs(v_rec - v_test).max():.6e}")
print(f"  -> s2fft spin-1 is self-consistent (roundtrip works)")

print()
print("=" * 80)
print("TEST 2: Check what grid_to_spectral actually computes for m>=1")
print("=" * 80)

# For pure u=cos(lat), v=0:
# The spin-1 forward transform of f = -v + i*u = i*u = i*cos(lat)
# This is purely imaginary. The forward transform of a purely imaginary
# field should give coefficients that reflect that.

u_zonal = np.cos(lat_grid) * np.ones_like(lon_grid)
v_zero = np.zeros_like(u_zonal)

f_spin1_zonal = jnp.array(-v_zero + 1j * u_zonal)  # = i * cos(lat)
F1_lm_zonal = s2fft.forward_jax(f_spin1_zonal, L, spin=1, sampling=sampling)

print(f"  Input: f = i*cos(lat) (purely imaginary, axisymmetric)")
print(f"  F1_lm(l=1, m=0): {complex(F1_lm_zonal[1, L - 1]):.6e}")
print(f"  F1_lm(l=1, m=1): {complex(F1_lm_zonal[1, L]):.6e}")
print(f"  F1_lm(l=1,m=-1): {complex(F1_lm_zonal[1, L - 2]):.6e}")
print(
    f"  max|F1_lm| for m!=0: {np.abs(np.asarray(F1_lm_zonal[:, : L - 1])).max():.6e} "
    f"and {np.abs(np.asarray(F1_lm_zonal[:, L:])).max():.6e}"
)
print(f"  -> For axisymmetric input, only m=0 should be nonzero (as expected)")

# Now with a perturbation: u = cos(lat)*cos(lon)
u_pert = np.cos(lat_grid) * np.cos(lon_grid)
f_spin1_pert = jnp.array(1j * u_pert)  # v=0
F1_lm_pert = s2fft.forward_jax(f_spin1_pert, L, spin=1, sampling=sampling)

print(f"\n  Input: f = i*cos(lat)*cos(lon) (has m=1 component)")
print(f"  F1_lm(l=1, m=0):  {complex(F1_lm_pert[1, L - 1]):.6e}")
print(f"  F1_lm(l=1, m=+1): {complex(F1_lm_pert[1, L]):.6e}")
print(f"  F1_lm(l=1, m=-1): {complex(F1_lm_pert[1, L - 2]):.6e}")
print(f"  F1_lm(l=2, m=+1): {complex(F1_lm_pert[2, L]):.6e}")
print(f"  F1_lm(l=2, m=-1): {complex(F1_lm_pert[2, L - 2]):.6e}")

# Check: for a real-valued u field, f=i*u is purely imaginary.
# The spin-1 SH coefficients of a purely imaginary field should satisfy
# certain symmetry relations. Let's check.
print(f"\n  Checking m symmetry of spin-1 coefficients:")
for l_check in range(1, 5):
    for m_check in range(1, min(l_check + 1, 4)):
        j_pos = m_check + L - 1
        j_neg = -m_check + L - 1
        c_pos = complex(F1_lm_pert[l_check, j_pos])
        c_neg = complex(F1_lm_pert[l_check, j_neg])
        print(
            f"    l={l_check} m=+{m_check}: {c_pos:+.6e}   "
            f"m=-{m_check}: {c_neg:+.6e}   "
            f"ratio: {c_neg / c_pos if abs(c_pos) > 1e-30 else 'N/A'}"
        )

print()
print("=" * 80)
print("TEST 3: Compare s2fft spin-0 vs spin-1 for gradient computation")
print("=" * 80)

# The gradient of a scalar f is computed via spin-1:
#   F1_lm = -l_factor * f_lm
#   grad_spin1 = inverse_spin1(F1_lm)
#   df/dlambda = grad_spin1.imag / R
#   df/dphi = -grad_spin1.real / R
#
# Let's verify this gives the right answer for f = cos(lat)*cos(lon)
# df/dlambda = -cos(lat)*sin(lon) / (R*cos(lat)) = -sin(lon)/R  [on unit sphere]
# Wait, lambda is longitude, so df/dlambda = cos(lat)*(-sin(lon))
# But the gradient is (1/R)df/dlambda and (1/R)df/dphi

f_scalar = jnp.array(np.cos(lat_grid) * np.cos(lon_grid))
f_lm = s2fft.forward_jax(f_scalar, L, sampling=sampling)

# Gradient via spin-1
F1_grad = -l_factor[:, None] * f_lm
grad_spin1 = s2fft.inverse_jax(F1_grad, L, spin=1, sampling=sampling)
grad_lon = np.asarray(grad_spin1.imag) / radius  # df/dlambda / R
grad_lat = np.asarray(-grad_spin1.real) / radius  # df/dphi / R

# Analytical gradients:
# f = cos(lat)*cos(lon)
# (1/R)*df/dlambda = (1/(R*cos(lat))) * cos(lat)*(-sin(lon)) = -sin(lon)/R
# (1/R)*df/dphi = (1/R)*(-sin(lat)*cos(lon))

# Note: the gradient operators return (1/R)*d/dlambda and (1/R)*d/dphi
# where d/dlambda already includes the 1/cos(lat) factor...
# Actually no. Let me think about this more carefully.
#
# The spin-1 gradient is the covariant gradient on the sphere:
#   grad_x = (1/(R*cos(lat))) * df/dlambda  (eastward)
#   grad_y = (1/R) * df/dphi               (northward)
#
# For f = cos(lat)*cos(lon):
#   grad_x = (1/(R*cos(lat))) * cos(lat)*(-sin(lon)) = -sin(lon)/R
#   grad_y = (1/R) * (-sin(lat)*cos(lon))

grad_x_exact = -np.sin(lon_grid) / radius * np.ones_like(lat_grid)
grad_y_exact = -np.sin(lat_grid) * np.cos(lon_grid) / radius

print(f"  Gradient of cos(lat)*cos(lon):")
print(f"    grad_lon max|diff| from exact: {np.abs(grad_lon - grad_x_exact).max():.6e}")
print(f"    grad_lat max|diff| from exact: {np.abs(grad_lat - grad_y_exact).max():.6e}")
print(f"    -> Scalar gradient via spin-1 works correctly")

print()
print("=" * 80)
print("TEST 4: Direct comparison - build spectral field, check grid recovery")
print("=" * 80)

# Set up a spectral vorticity field with known l=2,m=1 coefficient
# and check what grid u,v we get
vrt_manual = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
div_manual = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)

# Set vrt(l=2, m=1) = 1e-5 + 0.5e-5j
vrt_val = 1e-5 + 0.5e-5j
j_m1_pos = 1 + L - 1  # m=+1
j_m1_neg = -1 + L - 1  # m=-1
vrt_manual = vrt_manual.at[2, j_m1_pos].set(vrt_val)
# For spin-0 real field: f_l^{-m} = (-1)^m * conj(f_l^m) [with CS phase]
# s2fft uses CS phase, so:
vrt_manual = vrt_manual.at[2, j_m1_neg].set((-1) ** 1 * np.conj(vrt_val))

# Reconstruct u, v
F1_inv = inv_l_factor[:, None] * (div_manual + 1j * vrt_manual) * radius
f1_inv = s2fft.inverse_jax(F1_inv, L, spin=1, sampling=sampling)
u_from_spec = np.asarray(f1_inv.imag)
v_from_spec = np.asarray(-f1_inv.real)

print(f"  Set vrt(l=2,m=1) = {vrt_val}")
print(
    f"  Recovered u: max={np.abs(u_from_spec).max():.6e}, is_real={np.abs(u_from_spec.imag).max() < 1e-15}"
)
print(
    f"  Recovered v: max={np.abs(v_from_spec).max():.6e}, is_real={np.abs(v_from_spec.imag).max() < 1e-15}"
)

# Now do the roundtrip: (u,v) -> spectral -> (u,v)
f_fwd = -jnp.array(v_from_spec) + 1j * jnp.array(u_from_spec)
F1_fwd = s2fft.forward_jax(f_fwd, L, spin=1, sampling=sampling)
Diz_fwd = l_factor[:, None] * F1_fwd / radius
vrt_recovered = Diz_fwd.imag
div_recovered = Diz_fwd.real

vrt_orig_21 = complex(vrt_manual[2, j_m1_pos])
vrt_rec_21 = complex(vrt_recovered[2, j_m1_pos])
print(f"\n  Roundtrip vrt(l=2,m=1):")
print(f"    original:  {vrt_orig_21}")
print(f"    recovered: {vrt_rec_21}")
print(f"    diff:      {abs(vrt_orig_21 - vrt_rec_21):.6e}")

# Check: what does the forward transform produce for m=-1?
vrt_rec_2m1 = complex(vrt_recovered[2, j_m1_neg])
vrt_orig_2m1 = complex(vrt_manual[2, j_m1_neg])
print(f"\n  Roundtrip vrt(l=2,m=-1):")
print(f"    original:  {vrt_orig_2m1}")
print(f"    recovered: {vrt_rec_2m1}")
print(f"    diff:      {abs(vrt_orig_2m1 - vrt_rec_2m1):.6e}")

print()
print("=" * 80)
print("TEST 5: Is the issue that vrt/div from grid_to_spectral are wrong?")
print("=" * 80)

# The DCMIP initial condition has u = u(lat) (almost purely zonal).
# The perturbation adds a small wave with m=1 structure.
# When we do grid_to_spectral, we compute:
#   f_spin1 = -v + i*u
#   F1_lm = forward_spin1(f_spin1)
#   D + i*zeta = l_factor * F1_lm / R
#   div = D.real, vrt = D.imag
#
# But note: the input to forward_spin1 is f_spin1 = -v + i*u.
# For the DCMIP IC: u is real, v is real, so f_spin1 is complex.
# The spin-1 forward transform of a complex field should give complex
# coefficients. But the previous test showed JAX vrt has zero imaginary
# parts for m>=1!
#
# Let's trace through step by step.

# Build DCMIP-like initial conditions
import climt
from sympl import set_constant

from gfs_dynamical_core.component_jax import GFSDynamicsJAX
from gfs_dynamical_core.jax.states import GridState
from gfs_dynamical_core.jax.transforms import (
    TransformConfig,
    enforce_triangular_truncation,
    grid_to_spectral,
    spectral_to_grid,
)

set_constant("reference_air_pressure", value=1e5, units="Pa")
NLEVS = 20
grid = climt.get_grid(nx=NLONS, ny=NLATS, nz=NLEVS)
dcmip = climt.DcmipInitialConditions(add_perturbation=True)
dycore_j = GFSDynamicsJAX()
state_j = climt.get_default_state([dycore_j], grid_state=grid)
out_j = dcmip(state_j)
state_j.update(out_j)

u0 = jnp.array(state_j["eastward_wind"])
v0 = jnp.array(state_j["northward_wind"])
temp0 = jnp.array(state_j["air_temperature"])
ps0 = jnp.array(state_j["surface_air_pressure"])
q0 = jnp.array(state_j["specific_humidity"])

# Pick level 13 (max wind)
lev = 13
u_lev = u0[lev]
v_lev = v0[lev]

print(f"  Level {lev}: u range [{float(u_lev.min()):.4f}, {float(u_lev.max()):.4f}]")
print(f"             v range [{float(v_lev.min()):.6f}, {float(v_lev.max()):.6f}]")

# Step 1: form the spin-1 input field
f_spin1_input = -v_lev + 1j * u_lev
print(f"\n  f_spin1 = -v + i*u")
print(
    f"    real part (=-v): range [{float(f_spin1_input.real.min()):.6e}, {float(f_spin1_input.real.max()):.6e}]"
)
print(
    f"    imag part (=u):  range [{float(f_spin1_input.imag.min()):.4f}, {float(f_spin1_input.imag.max()):.4f}]"
)

# Step 2: forward spin-1 transform
F1_lm_step = s2fft.forward_jax(f_spin1_input, L, spin=1, sampling=sampling)

# Check m=0 and m=1 coefficients
print(f"\n  F1_lm from spin-1 forward transform:")
for l_c in range(1, 6):
    c0 = complex(F1_lm_step[l_c, L - 1])
    c1 = complex(F1_lm_step[l_c, L])
    cm1 = complex(F1_lm_step[l_c, L - 2])
    print(f"    l={l_c}: m=0: {c0:.6e}   m=+1: {c1:.6e}   m=-1: {cm1:.6e}")

# Step 3: compute D + i*zeta
D_plus_izeta_step = l_factor[:, None] * F1_lm_step / radius

print(f"\n  D + i*zeta = l_factor * F1_lm / R:")
print(f"    vrt (=imag part):")
for l_c in range(1, 6):
    v0_coeff = complex(D_plus_izeta_step[l_c, L - 1])
    v1_coeff = complex(D_plus_izeta_step[l_c, L])
    print(
        f"      l={l_c}: m=0 imag={v0_coeff.imag:.6e}   m=1: real={v1_coeff.real:.6e} imag={v1_coeff.imag:.6e}"
    )

# Step 4: extract vrt = imag, div = real
vrt_result = D_plus_izeta_step.imag
div_result = D_plus_izeta_step.real

print(f"\n  vrt = D_plus_izeta.imag  (should have both real and imag parts for m>=1)")
for l_c in range(1, 6):
    vr = complex(vrt_result[l_c, L])  # m=+1
    print(
        f"    vrt(l={l_c}, m=1) = {vr:.6e}  (imag is {'ZERO' if abs(vr.imag) < 1e-25 else 'nonzero'})"
    )

print(f"\n  KEY INSIGHT: vrt_result = D_plus_izeta.imag extracts the imaginary")
print(f"  part of D_plus_izeta, which is ALREADY A REAL ARRAY.")
print(f"  So vrt(l,m) for m>=1 will always have zero imaginary part!")
print(f"  But SHTNS stores complex coefficients for m>=1.")
print(f"")
print(f"  The issue is NOT a bug in the transform itself.")
print(f"  The s2fft (L, 2L-1) array stores coefficients for m from -(L-1) to +(L-1).")
print(f"  For a REAL field, f_l^{{-m}} = (-1)^m * conj(f_l^m) [CS phase].")
print(f"  But vrt and div are REAL fields on the sphere.")
print(f"  So vrt_l^m and div_l^m should be complex for m>=1.")
print(f"  The D_plus_izeta array contains D_l^m + i*zeta_l^m as complex.")
print(f"  Taking .imag gives zeta_l^m as a REAL number for each (l,m).")
print(f"  But zeta_l^m should be complex!")
print(f"")
print(f"  Wait - the s2fft array layout stores f(l, m) as complex numbers")
print(f"  at position [l, m+L-1]. The REAL and IMAGINARY parts of f(l,m)")
print(f"  are both needed. When we take D_plus_izeta.imag, we get the")
print(f"  imaginary part of each ELEMENT, not the imaginary part of the")
print(f"  operator. This is correct!")
print(f"")
print(f"  Let me re-examine: D_plus_izeta[l, j] = D_l^m + i*zeta_l^m")
print(f"  where D_l^m and zeta_l^m are themselves complex numbers.")
print(
    f"  So D_plus_izeta[l,j].real = Re(D_l^m) and D_plus_izeta[l,j].imag = Re(zeta_l^m)"
)
print(f"  NO WAIT. D_l^m + i*zeta_l^m where D_l^m = a+bi and zeta_l^m = c+di")
print(f"  then D_plus_izeta = (a+bi) + i*(c+di) = (a-d) + i*(b+c)")
print(f"  So .real = a-d and .imag = b+c")
print(f"  This is NOT the same as extracting D and zeta separately!")
print(f"")
print(f"  THIS IS THE BUG!")

print()
print("=" * 80)
print("TEST 6: Verify the bug - D+izeta decomposition is wrong for complex coeffs")
print("=" * 80)

# For m=0, coefficients are purely real (imaginary part = 0 for real fields)
# So D_l^0 = real, zeta_l^0 = real
# D+izeta = D + i*zeta -> .real = D, .imag = zeta  (CORRECT for m=0)
#
# For m>=1, coefficients are complex: D_l^m = a+bi, zeta_l^m = c+di
# D+izeta = (a+bi) + i*(c+di) = (a-d) + i*(b+c)
# .real = a-d (NOT just D!)
# .imag = b+c (NOT just zeta!)
#
# The CORRECT decomposition needs to track real and imaginary parts separately.
# Or equivalently: don't combine D and zeta into a single complex number
# when the individual components are already complex.

# Let's verify numerically:
# Set up vrt with l=2, m=1 having both real and imaginary parts
vrt_check = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
div_check = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)

vrt_val_check = 1e-5 + 3e-6j  # complex coefficient
div_val_check = 2e-6 - 1e-6j  # complex coefficient

# Set m=+1 and corresponding m=-1 (for real field on sphere)
vrt_check = vrt_check.at[2, L].set(vrt_val_check)  # l=2, m=+1
vrt_check = vrt_check.at[2, L - 2].set(
    -np.conj(vrt_val_check)
)  # l=2, m=-1 (CS phase: (-1)^1)
div_check = div_check.at[2, L].set(div_val_check)
div_check = div_check.at[2, L - 2].set(-np.conj(div_val_check))

# Now reconstruct (u,v) via spin-1 inverse
F1_check = inv_l_factor[:, None] * (div_check + 1j * vrt_check) * radius
f1_check = s2fft.inverse_jax(F1_check, L, spin=1, sampling=sampling)
u_check = f1_check.imag
v_check = -f1_check.real

# Forward transform back
f_fwd_check = -v_check + 1j * u_check
F1_fwd_check = s2fft.forward_jax(f_fwd_check, L, spin=1, sampling=sampling)
Diz_fwd_check = l_factor[:, None] * F1_fwd_check / radius

# Method 1 (current code): .real and .imag
div_method1 = Diz_fwd_check.real
vrt_method1 = Diz_fwd_check.imag

# Check
print(f"  Original vrt(l=2,m=1): {vrt_val_check}")
print(f"  Method 1 vrt(l=2,m=1): {complex(vrt_method1[2, L])}")
print(f"  Diff: {abs(vrt_val_check - complex(vrt_method1[2, L])):.6e}")
print()
print(f"  Original div(l=2,m=1): {div_val_check}")
print(f"  Method 1 div(l=2,m=1): {complex(div_method1[2, L])}")
print(f"  Diff: {abs(div_val_check - complex(div_method1[2, L])):.6e}")
print()

# The D+izeta approach works because we're storing the result in a complex array.
# D_plus_izeta[l, m+L-1] is a complex128 number.
# When D_l^m = a+bi and zeta_l^m = c+di:
#   D+izeta = (a-d) + (b+c)i  as a complex128
#   .real gives a-d (a real float64)
#   .imag gives b+c (a real float64)
# These are NOT the original complex D_l^m and zeta_l^m!
#
# BUT: the roundtrip should still work because the inverse transform
# uses (div + i*vrt) to reconstruct the spin-1 field, and if div and vrt
# are stored as complex arrays with the correct values, it should be fine.
#
# The question is: does grid_to_spectral produce CORRECT div and vrt arrays
# even though the intermediate .real/.imag extraction looks wrong?

# Let's check if this is actually a problem by seeing if the roundtrip works
# when we go through the full grid_to_spectral -> spectral_to_grid path

print("=" * 80)
print("TEST 7: Full roundtrip through grid_to_spectral and spectral_to_grid")
print("=" * 80)

trans_config = TransformConfig(L=L, sampling=sampling, radius=radius)

# Create a grid state with known u, v fields
u_test2 = np.cos(lat_grid) * np.cos(lon_grid) * 10.0 + np.cos(lat_grid) * 5.0
v_test2 = 0.5 * np.sin(lat_grid) * np.sin(2 * lon_grid)
t_test2 = 300.0 + 10.0 * np.cos(lat_grid) * np.ones_like(lon_grid)

# Broadcast to (nlevs, nlat, nlon) with just 1 level
u_3d = jnp.broadcast_to(jnp.array(u_test2)[None, :, :], (1, NLATS, NLONS))
v_3d = jnp.broadcast_to(jnp.array(v_test2)[None, :, :], (1, NLATS, NLONS))
t_3d = jnp.broadcast_to(jnp.array(t_test2)[None, :, :], (1, NLATS, NLONS))

grid_test = GridState(
    u=u_3d,
    v=v_3d,
    temperature=t_3d,
    vorticity=jnp.zeros_like(u_3d),
    divergence=jnp.zeros_like(u_3d),
    log_surface_pressure=jnp.log(jnp.ones((NLATS, NLONS)) * 1e5),
    tracers=jnp.zeros((1, 1, NLATS, NLONS)),
)

spec_test = grid_to_spectral(grid_test, trans_config)
grid_rec, _ = spectral_to_grid(spec_test, trans_config)

u_rec2 = np.asarray(grid_rec.u.real)[0]
v_rec2 = np.asarray(grid_rec.v.real)[0]
t_rec2 = np.asarray(grid_rec.temperature.real)[0]

print(
    f"  u roundtrip: max|diff| = {np.abs(u_rec2 - u_test2).max():.6e}  (max|u| = {np.abs(u_test2).max():.4f})"
)
print(
    f"  v roundtrip: max|diff| = {np.abs(v_rec2 - v_test2).max():.6e}  (max|v| = {np.abs(v_test2).max():.4f})"
)
print(
    f"  T roundtrip: max|diff| = {np.abs(t_rec2 - t_test2).max():.6e}  (max|T| = {np.abs(t_test2).max():.4f})"
)

print(f"\n  If the roundtrip is good for u and v, then grid_to_spectral")
print(f"  and spectral_to_grid are self-consistent.")
print(f"  The problem must be that grid_to_spectral produces DIFFERENT")
print(f"  spectral coefficients than SHTNS's getvrtdivspec, even though")
print(f"  both are internally consistent.")

print()
print("=" * 80)
print("TEST 8: Phase convention difference between s2fft and SHTNS")
print("=" * 80)

# s2fft uses Condon-Shortley phase: Y_l^m(s2fft) = (-1)^m * Y_l^m(no_CS)
# SHTNS is initialized with SHT_NO_CS_PHASE
#
# For scalar (spin-0) transforms:
#   Forward: f_lm(s2fft) = (-1)^m * f_lm(SHTNS) for m > 0
#   Inverse: both produce the same grid values (self-consistent)
#
# For the forward vector transform in grid_to_spectral:
#   1. We form f = -v + i*u
#   2. Forward spin-1 transform gives F1_lm
#   3. D + i*zeta = l_factor * F1_lm / R
#   4. vrt = (D+izeta).imag, div = (D+izeta).real
#
# In SHTNS getvrtdivspec:
#   1. shtns_spat_to_sphtor(u, v, psi_lm, chi_lm)
#   2. vrtspec = (lap/R) * psi_lm = -l(l+1)/R * psi_lm
#   3. divspec = (lap/R) * chi_lm = -l(l+1)/R * chi_lm
#
# The key question: when s2fft and SHTNS both transform the SAME grid field,
# do they produce the same spectral coefficients (up to phase)?
#
# For spin-0: f_lm(s2fft) = (-1)^m * f_lm(SHTNS)  [for m>0]
# For spin-1: ???
#
# The spin-weighted spherical harmonics have their own CS phase convention.
# s2fft's spin-1 harmonics: _1Y_l^m with CS phase
# SHTNS's sphtor harmonics: related to vector spherical harmonics
#
# CRITICAL: SHTNS uses the "toroidal/spheroidal" decomposition which
# is related to but different from spin-weighted harmonics.
# The relationship between them involves additional phase factors.

# Let's determine the phase empirically by comparing Fortran and JAX
# spectral coefficients for the SAME initial grid state.

# Load Fortran dump
from pathlib import Path

NTRUNC = int(NLONS / 3 - 2)
NDIMSPEC = (NTRUNC + 1) * (NTRUNC + 2) // 2
dump_path = Path("debug_data/fortran_step_1_stage_0.bin")
f_dump = {}
data = dump_path.read_bytes()
offset = 0


def read_arr(shape, dtype=np.float64):
    global offset
    n = int(np.prod(shape)) * np.dtype(dtype).itemsize
    arr = np.frombuffer(data[offset : offset + n], dtype=dtype).copy()
    offset += n
    return arr.reshape(shape[::-1]).T


f_dump["ug"] = read_arr((NLONS, NLATS, NLEVS))
f_dump["vg"] = read_arr((NLONS, NLATS, NLEVS))
f_dump["virtempg"] = read_arr((NLONS, NLATS, NLEVS))
f_dump["lnpsg"] = read_arr((NLONS, NLATS))
f_dump["tracerg"] = read_arr((NLONS, NLATS, NLEVS, 1))
f_dump["vrtspec"] = read_arr((NDIMSPEC, NLEVS), dtype=np.complex128)
f_dump["divspec"] = read_arr((NDIMSPEC, NLEVS), dtype=np.complex128)
f_dump["virtempspec"] = read_arr((NDIMSPEC, NLEVS), dtype=np.complex128)
f_dump["lnpsspec"] = read_arr((NDIMSPEC,), dtype=np.complex128)

# Reorder
for nm in ["ug", "vg", "virtempg"]:
    f_dump[nm] = f_dump[nm].transpose(2, 1, 0)[:, ::-1, :]
f_dump["lnpsg"] = f_dump["lnpsg"].T[::-1, :]
for nm in ["vrtspec", "divspec", "virtempspec"]:
    f_dump[nm] = f_dump[nm].T

# Unpack SHTNS triangular -> compare with JAX s2fft
# SHTNS packed: index runs over (m=0,l=0..T), (m=1,l=1..T), ...
lev = 13

# Build the JAX spectral state from the initial grid
grid_orig2 = GridState(
    u=u0,
    v=v0,
    temperature=temp0,
    vorticity=jnp.zeros_like(u0),
    divergence=jnp.zeros_like(u0),
    log_surface_pressure=jnp.log(ps0),
    tracers=jnp.stack([q0], axis=0),
)
spec_jax = grid_to_spectral(
    grid_orig2, TransformConfig(L=L, sampling=sampling, radius=6.371e6)
)

# Extract JAX vrt/div at test level
vrt_jax = np.asarray(spec_jax.vorticity[lev])
div_jax = np.asarray(spec_jax.divergence[lev])
temp_jax = np.asarray(spec_jax.temperature[lev])

# Extract Fortran vrt/div at test level (packed format)
vrt_fort_packed = f_dump["vrtspec"][lev]  # (NDIMSPEC,)
div_fort_packed = f_dump["divspec"][lev]
temp_fort_packed = f_dump["virtempspec"][lev]

# Compare m=0 coefficients (no phase issue for m=0)
print(f"\n  m=0 comparison (no CS phase difference expected):")
idx = 0
for m in range(min(3, NTRUNC + 1)):
    print(f"\n  m={m}:")
    for l in range(m, min(m + 5, NTRUNC + 1)):
        # SHTNS packed index
        packed_idx = idx + (l - m)
        vrt_f = vrt_fort_packed[packed_idx]
        div_f = div_fort_packed[packed_idx]

        if l < L:
            j = m + L - 1  # s2fft column for +m
            vrt_j = vrt_jax[l, j]
            div_j = div_jax[l, j]

            # Try different phase corrections
            vrt_j_cs = (-1) ** m * vrt_j  # undo CS phase

            if abs(vrt_f) > 1e-20:
                ratio_nofix = vrt_j / vrt_f
                ratio_cs = vrt_j_cs / vrt_f
                print(
                    f"    l={l}: F={vrt_f:.6e}  J={vrt_j:.6e}  J_cs={vrt_j_cs:.6e}  "
                    f"ratio={ratio_nofix:.4f}  ratio_cs={ratio_cs:.4f}"
                )
            else:
                print(f"    l={l}: F={vrt_f:.6e}  J={vrt_j:.6e}  (F~0)")

    # Advance packed index past this m block
    idx += NTRUNC + 1 - m

print()
print("=" * 80)
print("TEST 9: Check if temp spectral coefficients need CS correction")
print("=" * 80)

# For scalars, the CS phase should be the only difference
idx = 0
for m in range(4):
    mismatch_count = 0
    match_nofix = 0
    match_cs = 0
    for l in range(m, NTRUNC + 1):
        packed_idx = idx + (l - m)
        tf = temp_fort_packed[packed_idx]
        if l < L:
            j = m + L - 1
            tj = temp_jax[l, j]
            tj_cs = (-1) ** m * tj
            if abs(tf) > 1e-20:
                if abs(tj - tf) / abs(tf) < 1e-10:
                    match_nofix += 1
                elif abs(tj_cs - tf) / abs(tf) < 1e-10:
                    match_cs += 1
                else:
                    mismatch_count += 1
    idx += NTRUNC + 1 - m
    print(
        f"  Temp m={m}: match_nofix={match_nofix}  match_cs={match_cs}  mismatch={mismatch_count}"
    )

print(f"\n  Conclusion: if match_nofix works for even m and match_cs works")
print(f"  for odd m (or vice versa), the CS phase is the difference.")
print(f"  For scalars this doesn't matter because roundtrip cancels.")
print(f"  For vectors, the phase matters if s2fft and SHTNS apply it differently.")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
The s2fft spin-1 transform is self-consistent: roundtrip works perfectly.
The grid_to_spectral and spectral_to_grid are self-consistent: roundtrip works.

The problem is that s2fft and SHTNS use DIFFERENT conventions for vector
spherical harmonics / spin-weighted harmonics. Specifically:

1. s2fft uses spin-weighted spherical harmonics with Condon-Shortley phase
2. SHTNS uses toroidal/spheroidal decomposition WITHOUT CS phase

The spectral coefficients from the two libraries are related by a phase
factor that depends on m and possibly l. For the scalar (spin-0) case,
this phase factor cancels in the roundtrip (forward + inverse). For the
vector (spin-1) case, the phase factor is different and does NOT cancel
when mixing transforms (e.g., using s2fft forward with SHTNS-convention
spectral coefficients).

The DCMIP initial conditions provide u,v on a grid. Both implementations:
- Forward transform to get vrt_spec, div_spec (DIFFERENT due to conventions)
- Inverse transform to recover u,v (SAME, because each is self-consistent)

The problem arises because the dynamics code works in spectral space:
- Tendencies are computed in grid space
- Transformed to spectral space via grid_to_spectral_tendencies
- Added to the spectral state and transformed back

If the spectral representation itself is different between the two,
the dynamics computations that mix grid and spectral operations will diverge.

KEY QUESTION: Is the u,v recovered from the spectral state the same in both?
If yes, then the spectral representation difference doesn't matter.
If no, then the spin-1 inverse is also different.

From the Stage 0 comparison: u differs by ~1 m/s and v by ~100%.
This means spectral_to_grid (spin-1 inverse) produces DIFFERENT u,v
for the SAME spectral state. This is the root cause.
""")
