"""Minimal diagnostic: s2fft spin-1 transform for m=0 zonal wind.

Tests:
1. s2fft roundtrip (forward+inverse) for u=cos(lat), v=0
2. Analytic check: do the m=0 spectral coefficients match theory?
3. Single-mode test: set vrt(l,m=0)=1, invert, check profile
4. Cross-check: compare SHTNS-style formula with s2fft spin-1

No Fortran/SHTNS dependency required.
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

import jax
import jax.numpy as jnp
import numpy as np
import s2fft

L = 64
sampling = "gl"
radius = 6371000.0
NLATS, NLONS = L, 2 * L - 1

# Get GL colatitudes and latitudes
cos_theta_nodes, gl_weights = np.polynomial.legendre.leggauss(L)
thetas = np.flip(np.arccos(cos_theta_nodes))  # ascending colatitude (N->S)
latitudes = np.pi / 2 - thetas  # geographic latitude, N->S

l_arr = np.arange(L)
l_factor = np.sqrt(l_arr * (l_arr + 1))
inv_l_factor = np.where(l_arr > 0, 1.0 / l_factor, 0.0)

# ============================================================
# TEST 1: s2fft spin-1 roundtrip for zonal wind u=u0*cos(lat)
# ============================================================
print("=" * 70)
print("TEST 1: s2fft spin-1 vector roundtrip for u=cos(lat), v=0")
print("=" * 70)

u0 = 10.0
u_orig = u0 * np.cos(latitudes)[:, None] * np.ones((1, NLONS))
v_orig = np.zeros((NLATS, NLONS))

# Forward: (u, v) -> (vorticity, divergence) spectral
# Using both spin+1 and spin-1 as in transforms.py
f_plus = -v_orig + 1j * u_orig   # spin +1 input
f_minus = v_orig + 1j * u_orig   # spin -1 input

F1_lm = s2fft.forward_jax(jnp.array(f_plus), L, spin=1, sampling=sampling)
Fm1_lm = s2fft.forward_jax(jnp.array(f_minus), L, spin=-1, sampling=sampling)

result_p = l_factor[:, None] * np.array(F1_lm) / radius
result_m = l_factor[:, None] * np.array(Fm1_lm) / radius

flm_div = (result_p + result_m) / 2
flm_vort = (result_p - result_m) / (2j)

print(f"\nForward transform results:")
print(f"  max|vort_lm|  = {np.abs(flm_vort).max():.6e}")
print(f"  max|div_lm|   = {np.abs(flm_div).max():.6e}  (should be ~0)")

# Check m=0 vorticity coefficients
m0_idx = L - 1  # m=0 column in s2fft format
print(f"\n  Vorticity m=0 coefficients (first 10):")
for l in range(min(10, L)):
    val = flm_vort[l, m0_idx]
    print(f"    l={l:2d}: {val.real:+12.5e} {val.imag:+12.5e}j")

# Inverse: (vorticity, divergence) spectral -> (u, v) grid
F1_inv = inv_l_factor[:, None] * (flm_div + 1j * flm_vort) * radius
f_spin1 = s2fft.inverse_jax(jnp.array(F1_inv), L, spin=1, sampling=sampling)
u_rec = np.array(f_spin1.imag)
v_rec = np.array(-f_spin1.real)

# Compare roundtrip
u_diff = np.abs(u_rec - u_orig)
v_diff = np.abs(v_rec - v_orig)
print(f"\nRoundtrip errors:")
print(f"  max|u_rec - u_orig| = {u_diff.max():.6e}  (max|u| = {np.abs(u_orig).max():.2f})")
print(f"  max|v_rec - v_orig| = {v_diff.max():.6e}")

# Zonal mean comparison
u_zm_orig = u_orig.mean(axis=1)
u_zm_rec = u_rec.mean(axis=1)
zm_diff = np.abs(u_zm_rec - u_zm_orig)
print(f"  max|u_zm_rec - u_zm_orig| = {zm_diff.max():.6e}")

if u_zm_orig.max() > 0:
    ratio = u_zm_rec / np.where(np.abs(u_zm_orig) > 0.01, u_zm_orig, np.nan)
    valid = ~np.isnan(ratio)
    if valid.any():
        print(f"  u_zm ratio (rec/orig) range: [{np.nanmin(ratio):.6f}, {np.nanmax(ratio):.6f}]")

# ============================================================
# TEST 2: Single-mode test — set vrt(l=1,m=0)=known, invert
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: Single-mode inverse — vrt(l=1,m=0) only")
print("=" * 70)

# For solid body rotation u = u0*cos(lat):
# Vorticity ζ = 2*u0/R * sin(lat) = 2*u0/R * cos(theta)
# In orthonormal SH: ζ = ζ_10 * Y_1^0 where Y_1^0 = sqrt(3/4pi) * cos(theta)
# So ζ_10 = 2*u0/R * sqrt(4pi/3)

zeta_10 = 2.0 * u0 / radius * np.sqrt(4 * np.pi / 3)
print(f"\n  Analytic ζ_10 = {zeta_10:.10e}")
print(f"  s2fft ζ_10   = {flm_vort[1, m0_idx].real:.10e}")
print(f"  Ratio        = {flm_vort[1, m0_idx].real / zeta_10:.10f}")

# Now do inverse from a single vrt(1,0) coefficient
vrt_single = np.zeros((L, 2*L-1), dtype=np.complex128)
vrt_single[1, m0_idx] = zeta_10
div_single = np.zeros_like(vrt_single)

F1_single = inv_l_factor[:, None] * (div_single + 1j * vrt_single) * radius
f_spin1_single = s2fft.inverse_jax(jnp.array(F1_single), L, spin=1, sampling=sampling)
u_single = np.array(f_spin1_single.imag)
v_single = np.array(-f_spin1_single.real)

# Expected: u = u0 * cos(lat), v = 0
u_zm_single = u_single.mean(axis=1)
u_expected = u0 * np.cos(latitudes)

print(f"\n  Single-mode inverse results:")
print(f"  max|u_single| = {np.abs(u_single).max():.6f}")
print(f"  max|v_single| = {np.abs(v_single).max():.6e}  (should be ~0)")
print(f"  max|u_zm - u0*cos(lat)| = {np.abs(u_zm_single - u_expected).max():.6e}")

if np.abs(u_expected).max() > 0:
    ratio2 = u_zm_single / np.where(np.abs(u_expected) > 0.01, u_expected, np.nan)
    valid2 = ~np.isnan(ratio2)
    if valid2.any():
        print(f"  u_zm ratio range: [{np.nanmin(ratio2):.10f}, {np.nanmax(ratio2):.10f}]")

# ============================================================
# TEST 3: Multi-mode roundtrip — realistic JW06-like jet
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: Realistic JW06-like jet profile roundtrip")
print("=" * 70)

# Create a jet-like profile (narrower than solid body rotation)
lat_0 = np.pi / 4  # jet center at 45N
jet_width = np.pi / 6
u_jet = 35.0 * np.exp(-((latitudes - lat_0) / jet_width) ** 2)
u_jet_grid = u_jet[:, None] * np.ones((1, NLONS))
v_jet_grid = np.zeros_like(u_jet_grid)

# Forward
f_plus_j = -v_jet_grid + 1j * u_jet_grid
f_minus_j = v_jet_grid + 1j * u_jet_grid
F1_j = s2fft.forward_jax(jnp.array(f_plus_j), L, spin=1, sampling=sampling)
Fm1_j = s2fft.forward_jax(jnp.array(f_minus_j), L, spin=-1, sampling=sampling)

res_p_j = l_factor[:, None] * np.array(F1_j) / radius
res_m_j = l_factor[:, None] * np.array(Fm1_j) / radius

vort_j = (res_p_j - res_m_j) / (2j)
div_j = (res_p_j + res_m_j) / 2

# Truncate at T=40 (matching Fortran convention)
T = 40
for l in range(L):
    for m_idx in range(2*L-1):
        m = m_idx - (L-1)
        if l > T or abs(m) > T:
            vort_j[l, m_idx] = 0
            div_j[l, m_idx] = 0

# Inverse
F1_inv_j = inv_l_factor[:, None] * (div_j + 1j * vort_j) * radius
f_spin1_j = s2fft.inverse_jax(jnp.array(F1_inv_j), L, spin=1, sampling=sampling)
u_rec_j = np.array(f_spin1_j.imag)
v_rec_j = np.array(-f_spin1_j.real)

u_zm_jet_orig = u_jet_grid.mean(axis=1)
u_zm_jet_rec = u_rec_j.mean(axis=1)

# The roundtrip won't be exact because truncation removes high-l modes
# But the ratio should be constant if it's a normalization issue
print(f"\n  max|u_orig| = {np.abs(u_jet_grid).max():.2f}")
print(f"  max|u_rec|  = {np.abs(u_rec_j).max():.6f}")
print(f"  max|u_zm diff| = {np.abs(u_zm_jet_rec - u_zm_jet_orig).max():.6f}")

# Check if this is a UNIFORM scaling (normalization) or truncation-shaped
ratio_jet = u_zm_jet_rec / np.where(np.abs(u_zm_jet_orig) > 1.0, u_zm_jet_orig, np.nan)
valid_jet = ~np.isnan(ratio_jet) & (np.abs(u_zm_jet_orig) > 5.0)  # only where jet is strong
if valid_jet.any():
    r_vals = ratio_jet[valid_jet]
    print(f"  u_zm ratio in jet core: mean={r_vals.mean():.10f}, std={r_vals.std():.10f}")
    print(f"  ratio range: [{r_vals.min():.10f}, {r_vals.max():.10f}]")

# ============================================================
# TEST 4: Check if spin-1 forward uses different quadrature
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: Forward transform — manual integration vs s2fft")
print("=" * 70)

# The forward spin-1 transform should compute:
#   F1_lm = integral[ f(theta,phi) * conj(_1Y_l^m) dOmega ]
# For m=0 zonal wind, f_plus = i*u (since v=0):
#   f_plus = i * u0 * cos(lat) = i * u0 * sin(theta)
#
# We can check by manually integrating against the spin-1 basis

# s2fft forward result (already computed above for u=cos(lat)):
print(f"\n  s2fft forward spin+1 of (i*u0*cos(lat)) at m=0:")
for l in range(min(8, L)):
    val = F1_lm[l, m0_idx]
    print(f"    l={l:2d}: {val.real:+15.8e} {val.imag:+15.8e}j")

# Compare: if everything is consistent, F1_lm[l,m=0] should equal
# inv_l_factor[l] * i * vort_l0 * radius (from our computed vort)
print(f"\n  Cross-check: F1_lm vs inv_l_factor * (div + i*vort) * R at m=0:")
for l in range(1, min(8, L)):
    expected = inv_l_factor[l] * (flm_div[l, m0_idx] + 1j * flm_vort[l, m0_idx]) * radius
    # But F1_lm came from result_p = l_factor * F1_lm / R, so
    # result_p = D + i*zeta, and F1_lm = result_p * R / l_factor
    # The inverse formula uses inv_l_factor * (D + i*zeta) * R
    # which equals inv_l_factor * l_factor * F1_lm = F1_lm (for l>0)
    # So they should be identical.
    actual = F1_lm[l, m0_idx]
    print(f"    l={l:2d}: actual={actual.real:+12.5e}{actual.imag:+12.5e}j  "
          f"expected={expected.real:+12.5e}{expected.imag:+12.5e}j  "
          f"diff={abs(actual-expected):.2e}")

# ============================================================
# TEST 5: Direct comparison of _1Y_l^0 normalization
# ============================================================
print("\n" + "=" * 70)
print("TEST 5: Spin-1 basis function normalization at m=0")
print("=" * 70)

# Create delta function in spectral space: only (l, m=0) = 1
# and check what grid field we get
for l_test in [1, 2, 3, 5, 10]:
    flm_test = np.zeros((L, 2*L-1), dtype=np.complex128)
    flm_test[l_test, m0_idx] = 1.0

    # Spin-0 inverse: Y_l^0(theta)
    f0 = np.array(s2fft.inverse_jax(jnp.array(flm_test), L, spin=0, sampling=sampling))

    # Spin-1 inverse: _1Y_l^0(theta)
    f1 = np.array(s2fft.inverse_jax(jnp.array(flm_test), L, spin=1, sampling=sampling))

    # Expected: _1Y_l^0 = -(1/sqrt(l(l+1))) * dY_l^0/dtheta
    # Compute dY_l^0/dtheta numerically from the spin-0 result
    # Y_l^0 is constant in phi, so take zonal mean
    y_l0 = f0.mean(axis=1).real

    # Numerical derivative: d/dtheta Y_l^0
    # theta = colatitude (ascending), so use finite differences
    dtheta = np.diff(thetas)
    dy_dtheta_num = np.diff(y_l0) / dtheta
    theta_mid = (thetas[:-1] + thetas[1:]) / 2

    # The spin-1 result at the GL points
    s1_zm = f1.mean(axis=1)  # complex: real = -v component, imag = u component

    # For purely real spectral coefficients at m=0, _1Y_l^0 should be real
    # (since it involves d^l_{0,-1}(theta) which is real)
    print(f"\n  l={l_test:2d}: max|imag(_1Y_l0)| = {np.abs(s1_zm.imag).max():.2e}, "
          f"max|real(_1Y_l0)| = {np.abs(s1_zm.real).max():.6f}")

    # Compare: _1Y_l^0 should equal -(1/sqrt(l(l+1))) * dY_l^0/dtheta
    expected_s1 = -(1.0 / np.sqrt(l_test * (l_test + 1))) * dy_dtheta_num

    # Interpolate s2fft result to midpoints for comparison
    s1_at_mid = (s1_zm.real[:-1] + s1_zm.real[1:]) / 2

    ratio_s1 = s1_at_mid / np.where(np.abs(expected_s1) > 1e-10, expected_s1, np.nan)
    valid_r = ~np.isnan(ratio_s1) & (np.abs(expected_s1) > np.abs(expected_s1).max() * 0.1)
    if valid_r.any():
        r_mean = ratio_s1[valid_r].mean()
        r_std = ratio_s1[valid_r].std()
        print(f"         _1Y_l0 / (-(1/sqrt(l(l+1))) * dY_l0/dtheta): "
              f"mean={r_mean:.8f}, std={r_std:.2e}")
    else:
        print(f"         Could not compute ratio (too small)")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
