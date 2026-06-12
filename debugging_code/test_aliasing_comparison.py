"""
Compare aliasing properties of SHTNS vs s2fft for the quadratic KE term.

Hypothesis: the persistent ~20% divergence error between Fortran (SHTNS) and
JAX (s2fft) dycores could be caused by different aliasing of the quadratic
nonlinear term KE = 0.5*(u^2 + v^2).

This script:
  1. Creates a realistic JW06 u-wind field on an L=64 GL grid.
  2. Computes KE = 0.5*(u^2 + v^2) in grid space.
  3. Forward-transforms KE using BOTH shtns and s2fft.
  4. Compares the resulting spectral coefficients.
  5. Tests the roundtrip aliasing (forward -> inverse -> forward).
  6. Compares the KE Laplacian term that enters the divergence tendency.

Key config: L=64, N_LON(s2fft)=127, N_LON(shtns)=128, N_LAT=64, NTRUNC=40
"""
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np
import jax.numpy as jnp
import s2fft
import shtns

# =============================================================================
# Configuration
# =============================================================================
L = 64
NTRUNC = 40
SAMPLING = "gl"
R = 6.371e6

N_LAT_S2FFT = L           # 64
N_LON_S2FFT = 2 * L - 1   # 127

# =============================================================================
# 1. Setup SHTNS
# =============================================================================
sh = shtns.sht(NTRUNC, NTRUNC)
nlat_sh, nlon_sh = sh.set_grid(nlat=N_LAT_S2FFT, nphi=N_LON_S2FFT + 1,
                                flags=shtns.SHT_NO_CS_PHASE)
# SHTNS uses nphi = 128 (even) for FFT efficiency, s2fft uses 127.
# To make the comparison fair, we also try with nphi=127 if SHTNS allows it,
# but first try the native grids and interpolate if needed.

print(f"SHTNS grid: nlat={nlat_sh}, nlon={nlon_sh}")
print(f"s2fft grid: nlat={N_LAT_S2FFT}, nlon={N_LON_S2FFT}")

# Both use Gauss-Legendre quadrature, so latitude nodes should be identical.
# Check:
cos_theta_sh = sh.cos_theta  # SHTNS GL nodes
from numpy.polynomial.legendre import leggauss
nodes, weights = leggauss(L)
cos_theta_s2fft = -nodes  # s2fft convention: colatitude increases with index
lat_diff = np.max(np.abs(np.sort(cos_theta_sh) - np.sort(cos_theta_s2fft)))
print(f"Max |cos(theta)| difference between SHTNS and s2fft GL nodes: {lat_diff:.2e}")

# =============================================================================
# 2. Create a realistic JW06-like u-wind and v-wind
# =============================================================================
# Approximate JW06 zonal wind: u(lat) = u0 * cos(lat)^(3/2) * sin(2*lat)^2
# with a small perturbation in v for realism.
u0 = 35.0  # m/s

# s2fft GL latitudes (colatitude = arccos(cos_theta))
theta_s2fft = np.arccos(cos_theta_s2fft)  # colatitudes
lat_s2fft = np.pi / 2 - theta_s2fft       # geographic latitudes

# s2fft longitude grid
lon_s2fft = np.linspace(0, 2 * np.pi, N_LON_S2FFT, endpoint=False)
LON_s2fft, LAT_s2fft = np.meshgrid(lon_s2fft, lat_s2fft)

# JW06-like zonal wind profile
u_s2fft = u0 * np.cos(LAT_s2fft) ** 1.5 * np.sin(2 * LAT_s2fft) ** 2
# Add a wavenumber-4 perturbation
u_s2fft += 2.0 * np.cos(LAT_s2fft) * np.cos(4 * LON_s2fft) * np.exp(
    -((LAT_s2fft - np.pi / 4) ** 2) / (2 * (np.pi / 12) ** 2)
)
# Meridional wind: small perturbation
v_s2fft = 1.0 * np.sin(LON_s2fft) * np.cos(LAT_s2fft) * np.exp(
    -((LAT_s2fft - np.pi / 4) ** 2) / (2 * (np.pi / 12) ** 2)
)

# For SHTNS: resample to 128 longitudes (or use 127 if it worked)
# SHTNS expects (nlat, nlon) with phi-contiguous (lon varies fastest)
if nlon_sh == N_LON_S2FFT:
    u_sh = u_s2fft.copy()
    v_sh = v_s2fft.copy()
    print("SHTNS and s2fft have SAME longitude count -- direct comparison")
else:
    # Resample: create the field on the SHTNS lon grid
    lon_sh = np.linspace(0, 2 * np.pi, nlon_sh, endpoint=False)
    LON_sh, LAT_sh = np.meshgrid(lon_sh, lat_s2fft)  # same latitudes
    u_sh = u0 * np.cos(LAT_sh) ** 1.5 * np.sin(2 * LAT_sh) ** 2
    u_sh += 2.0 * np.cos(LAT_sh) * np.cos(4 * LON_sh) * np.exp(
        -((LAT_sh - np.pi / 4) ** 2) / (2 * (np.pi / 12) ** 2)
    )
    v_sh = 1.0 * np.sin(LON_sh) * np.cos(LAT_sh) * np.exp(
        -((LAT_sh - np.pi / 4) ** 2) / (2 * (np.pi / 12) ** 2)
    )
    print(f"SHTNS uses {nlon_sh} longitudes vs s2fft {N_LON_S2FFT} -- "
          "fields generated analytically on both grids")

# =============================================================================
# 3. Compute KE = 0.5*(u^2 + v^2) in grid space
# =============================================================================
ke_s2fft = 0.5 * (u_s2fft ** 2 + v_s2fft ** 2)
ke_sh = 0.5 * (u_sh ** 2 + v_sh ** 2)

print(f"\nKE grid statistics:")
print(f"  s2fft: mean={ke_s2fft.mean():.4f}, max={ke_s2fft.max():.4f}")
print(f"  SHTNS: mean={ke_sh.mean():.4f}, max={ke_sh.max():.4f}")

# =============================================================================
# 4. Forward transform KE using both libraries
# =============================================================================
# --- s2fft forward transform ---
ke_lm_s2fft = np.array(s2fft.forward_jax(jnp.array(ke_s2fft), L, sampling=SAMPLING))
print(f"\ns2fft KE spectral shape: {ke_lm_s2fft.shape}")

# --- SHTNS forward transform ---
ke_lm_sh = sh.analys(ke_sh)
print(f"SHTNS KE spectral shape: {ke_lm_sh.shape} (packed, nlm={sh.nlm})")

# =============================================================================
# 5. Convert SHTNS packed coefficients to s2fft (L, 2L-1) format for comparison
# =============================================================================
def shtns_to_s2fft_format(packed, sh, L, ntrunc):
    """Convert SHTNS packed spectral coefficients to s2fft (L, 2L-1) format.

    SHTNS uses NO_CS_PHASE.  s2fft includes CS phase by default, so we need
    to multiply by (-1)^m to match.

    s2fft layout: flm[l, L-1+m] for m in [-L+1, ..., L-1]
    """
    out = np.zeros((L, 2 * L - 1), dtype=complex)
    for l in range(min(ntrunc + 1, L)):
        for m in range(0, l + 1):
            val = packed[sh.idx(l, m)]
            # Apply CS phase correction: s2fft includes (-1)^m, SHTNS does not
            cs = (-1) ** m
            out[l, L - 1 + m] = cs * val
            if m > 0:
                # s2fft convention for negative m
                out[l, L - 1 - m] = cs * (-1) ** m * np.conj(val)
    return out


ke_lm_sh_s2fft = shtns_to_s2fft_format(ke_lm_sh, sh, L, NTRUNC)

# =============================================================================
# 6. Compare spectral coefficients
# =============================================================================
print("\n" + "=" * 80)
print("COMPARISON: s2fft vs SHTNS spectral KE coefficients")
print("=" * 80)

# m=0 comparison (these are real)
print(f"\n{'l':>4} {'s2fft(m=0)':>18} {'SHTNS(m=0)':>18} {'diff':>14} {'rel_diff':>14}")
print("-" * 72)
max_rel_diff_m0 = 0.0
for l in range(min(NTRUNC + 1, 20)):
    v_s2 = ke_lm_s2fft[l, L - 1].real
    v_sh = ke_lm_sh_s2fft[l, L - 1].real
    diff = v_s2 - v_sh
    ref = max(abs(v_s2), abs(v_sh))
    rel = abs(diff) / ref if ref > 1e-20 else 0.0
    max_rel_diff_m0 = max(max_rel_diff_m0, rel)
    print(f"{l:4d} {v_s2:18.8e} {v_sh:18.8e} {diff:14.6e} {rel:14.6e}")

# Overall comparison for all (l, m) within truncation
diffs_all = []
vals_all = []
for l in range(NTRUNC + 1):
    for m in range(-l, l + 1):
        idx = L - 1 + m
        v_s2 = ke_lm_s2fft[l, idx]
        v_sh = ke_lm_sh_s2fft[l, idx]
        diffs_all.append(abs(v_s2 - v_sh))
        vals_all.append(max(abs(v_s2), abs(v_sh)))

diffs_all = np.array(diffs_all)
vals_all = np.array(vals_all)
nonzero = vals_all > 1e-20
rel_diffs = diffs_all[nonzero] / vals_all[nonzero]

print(f"\nAll (l,m) with l,|m| <= {NTRUNC}:")
print(f"  max |diff|     = {diffs_all.max():.6e}")
print(f"  max |value|    = {vals_all.max():.6e}")
print(f"  max rel |diff| = {rel_diffs.max():.6e}")
print(f"  mean rel |diff|= {rel_diffs.mean():.6e}")
print(f"  median rel     = {np.median(rel_diffs):.6e}")

# =============================================================================
# 7. Compare by wavenumber band
# =============================================================================
print(f"\n{'l_range':>12} {'max_rel_diff':>14} {'mean_rel_diff':>14} {'n_coeffs':>10}")
print("-" * 54)
for l_lo, l_hi in [(0, 5), (5, 10), (10, 20), (20, 30), (30, NTRUNC + 1)]:
    band_diffs = []
    band_vals = []
    for l in range(l_lo, min(l_hi, NTRUNC + 1)):
        for m in range(-l, l + 1):
            idx = L - 1 + m
            v_s2 = ke_lm_s2fft[l, idx]
            v_sh = ke_lm_sh_s2fft[l, idx]
            d = abs(v_s2 - v_sh)
            v = max(abs(v_s2), abs(v_sh))
            if v > 1e-20:
                band_diffs.append(d / v)
                band_vals.append(v)
    if band_diffs:
        bd = np.array(band_diffs)
        print(f"  [{l_lo:2d},{l_hi:2d}) {bd.max():14.6e} {bd.mean():14.6e} {len(bd):10d}")

# =============================================================================
# 8. Roundtrip aliasing test (s2fft only)
#    If forward -> inverse -> forward changes the result, there's aliasing.
# =============================================================================
print("\n" + "=" * 80)
print("ROUNDTRIP ALIASING TEST (s2fft)")
print("=" * 80)

# Truncate to ntrunc before roundtrip
from gfs_dynamical_core.jax.transforms import enforce_triangular_truncation
ke_lm_trunc = np.array(enforce_triangular_truncation(
    jnp.array(ke_lm_s2fft), L, NTRUNC
))

# Inverse
ke_roundtrip_grid = np.array(
    s2fft.inverse_jax(jnp.array(ke_lm_trunc), L, sampling=SAMPLING)
)

# Forward again
ke_lm_roundtrip = np.array(
    s2fft.forward_jax(jnp.array(ke_roundtrip_grid), L, sampling=SAMPLING)
)
ke_lm_roundtrip_trunc = np.array(enforce_triangular_truncation(
    jnp.array(ke_lm_roundtrip), L, NTRUNC
))

# Compare
rt_diff = np.abs(ke_lm_trunc - ke_lm_roundtrip_trunc)
rt_vals = np.abs(ke_lm_trunc)
mask = rt_vals > 1e-20
if mask.any():
    rt_rel = rt_diff[mask] / rt_vals[mask]
    print(f"Roundtrip max |diff|     = {rt_diff.max():.6e}")
    print(f"Roundtrip max rel |diff| = {rt_rel.max():.6e}")
    print(f"Roundtrip mean rel |diff|= {rt_rel.mean():.6e}")
else:
    print("All coefficients are zero (unexpected).")

# Same test for SHTNS
ke_roundtrip_sh = sh.synth(ke_lm_sh)
ke_lm_sh_rt = sh.analys(ke_roundtrip_sh)
rt_diff_sh = np.abs(ke_lm_sh - ke_lm_sh_rt)
rt_vals_sh = np.abs(ke_lm_sh)
mask_sh = rt_vals_sh > 1e-20
if mask_sh.any():
    rt_rel_sh = rt_diff_sh[mask_sh] / rt_vals_sh[mask_sh]
    print(f"\nSHTNS roundtrip max rel |diff| = {rt_rel_sh.max():.6e}")
    print(f"SHTNS roundtrip mean rel |diff|= {rt_rel_sh.mean():.6e}")

# =============================================================================
# 9. KE Laplacian comparison  -l(l+1)/R^2 * KE_lm
# =============================================================================
print("\n" + "=" * 80)
print("KE LAPLACIAN COMPARISON (enters divergence tendency)")
print("=" * 80)

l_arr = np.arange(L)
lap_factor = -l_arr * (l_arr + 1) / R ** 2

lap_ke_s2fft = lap_factor[:, None] * ke_lm_s2fft
lap_ke_sh = lap_factor[:, None] * ke_lm_sh_s2fft

# Compare the Laplacian at m=0
print(f"\n{'l':>4} {'s2fft_lap(m=0)':>18} {'SHTNS_lap(m=0)':>18} {'rel_diff':>14}")
print("-" * 58)
for l in range(min(NTRUNC + 1, 20)):
    v_s2 = lap_ke_s2fft[l, L - 1].real
    v_sh = lap_ke_sh[l, L - 1].real
    ref = max(abs(v_s2), abs(v_sh))
    rel = abs(v_s2 - v_sh) / ref if ref > 1e-20 else 0.0
    print(f"{l:4d} {v_s2:18.8e} {v_sh:18.8e} {rel:14.6e}")

# Full Laplacian comparison
lap_diffs = []
lap_vals = []
for l in range(NTRUNC + 1):
    for m in range(-l, l + 1):
        idx = L - 1 + m
        v_s2 = lap_ke_s2fft[l, idx]
        v_sh = lap_ke_sh[l, idx]
        d = abs(v_s2 - v_sh)
        v = max(abs(v_s2), abs(v_sh))
        if v > 1e-20:
            lap_diffs.append(d / v)
            lap_vals.append(v)

lap_diffs = np.array(lap_diffs)
print(f"\nKE Laplacian (all l,m within truncation):")
print(f"  max rel diff  = {lap_diffs.max():.6e}")
print(f"  mean rel diff = {lap_diffs.mean():.6e}")
print(f"  median rel    = {np.median(lap_diffs):.6e}")

# =============================================================================
# 10. Test with SAME grid (both 127 longitudes) -- if possible
# =============================================================================
print("\n" + "=" * 80)
print("SAME-GRID TEST: Force SHTNS to also use 127 longitudes")
print("=" * 80)

try:
    sh2 = shtns.sht(NTRUNC, NTRUNC)
    nlat2, nlon2 = sh2.set_grid(nlat=N_LAT_S2FFT, nphi=N_LON_S2FFT,
                                 flags=shtns.SHT_NO_CS_PHASE)
    print(f"SHTNS accepted 127 longitudes: nlat={nlat2}, nlon={nlon2}")

    # Use the EXACT same grid-space KE field
    ke_lm_sh2 = sh2.analys(ke_s2fft)
    ke_lm_sh2_s2fft = shtns_to_s2fft_format(ke_lm_sh2, sh2, L, NTRUNC)

    # Compare m=0
    print(f"\n{'l':>4} {'s2fft(m=0)':>18} {'SHTNS127(m=0)':>18} {'diff':>14} {'rel_diff':>14}")
    print("-" * 72)
    for l in range(min(NTRUNC + 1, 20)):
        v_s2 = ke_lm_s2fft[l, L - 1].real
        v_sh = ke_lm_sh2_s2fft[l, L - 1].real
        diff = v_s2 - v_sh
        ref = max(abs(v_s2), abs(v_sh))
        rel = abs(diff) / ref if ref > 1e-20 else 0.0
        print(f"{l:4d} {v_s2:18.8e} {v_sh:18.8e} {diff:14.6e} {rel:14.6e}")

    # Full comparison
    diffs_same = []
    vals_same = []
    for l in range(NTRUNC + 1):
        for m in range(-l, l + 1):
            idx = L - 1 + m
            v_s2 = ke_lm_s2fft[l, idx]
            v_sh = ke_lm_sh2_s2fft[l, idx]
            d = abs(v_s2 - v_sh)
            v = max(abs(v_s2), abs(v_sh))
            if v > 1e-20:
                diffs_same.append(d / v)
    diffs_same = np.array(diffs_same)
    print(f"\nSame-grid (127 lon), all (l,m):")
    print(f"  max rel diff  = {diffs_same.max():.6e}")
    print(f"  mean rel diff = {diffs_same.mean():.6e}")
    print(f"  median rel    = {np.median(diffs_same):.6e}")

except Exception as e:
    print(f"Could not set SHTNS to 127 longitudes: {e}")
    print("Trying 126 longitudes (next even number)...")
    try:
        sh3 = shtns.sht(NTRUNC, NTRUNC)
        nlat3, nlon3 = sh3.set_grid(nlat=N_LAT_S2FFT, nphi=126,
                                     flags=shtns.SHT_NO_CS_PHASE)
        print(f"SHTNS accepted 126 longitudes: nlat={nlat3}, nlon={nlon3}")
    except Exception as e2:
        print(f"Also failed: {e2}")

# =============================================================================
# 11. Energy spectrum comparison: |KE_lm|^2 summed over m, as fn of l
# =============================================================================
print("\n" + "=" * 80)
print("ENERGY SPECTRUM: |KE_lm|^2 summed over m")
print("=" * 80)

spec_s2fft = np.zeros(L)
spec_sh = np.zeros(L)
for l in range(min(NTRUNC + 1, L)):
    for m in range(-l, l + 1):
        idx = L - 1 + m
        spec_s2fft[l] += abs(ke_lm_s2fft[l, idx]) ** 2
        spec_sh[l] += abs(ke_lm_sh_s2fft[l, idx]) ** 2

print(f"\n{'l':>4} {'s2fft_power':>16} {'SHTNS_power':>16} {'ratio':>10}")
print("-" * 50)
for l in range(min(NTRUNC + 1, 20)):
    p_s2 = spec_s2fft[l]
    p_sh = spec_sh[l]
    ratio = p_s2 / p_sh if p_sh > 1e-30 else float("nan")
    print(f"{l:4d} {p_s2:16.6e} {p_sh:16.6e} {ratio:10.6f}")

# Near-truncation wavenumbers
print(f"\nNear truncation (l={NTRUNC-5} to {NTRUNC}):")
for l in range(max(0, NTRUNC - 5), NTRUNC + 1):
    p_s2 = spec_s2fft[l]
    p_sh = spec_sh[l]
    ratio = p_s2 / p_sh if p_sh > 1e-30 else float("nan")
    print(f"{l:4d} {p_s2:16.6e} {p_sh:16.6e} {ratio:10.6f}")

# =============================================================================
# 12. High-wavenumber leakage: power ABOVE truncation in s2fft
# =============================================================================
print("\n" + "=" * 80)
print("HIGH-WAVENUMBER LEAKAGE: power in l > NTRUNC from s2fft")
print("=" * 80)

power_above = 0.0
power_total = 0.0
for l in range(L):
    for m in range(-min(l, L - 1), min(l, L - 1) + 1):
        idx = L - 1 + m
        if 0 <= idx < 2 * L - 1:
            p = abs(ke_lm_s2fft[l, idx]) ** 2
            power_total += p
            if l > NTRUNC or abs(m) > NTRUNC:
                power_above += p

print(f"Total spectral power (s2fft): {power_total:.6e}")
print(f"Power above truncation:       {power_above:.6e}")
print(f"Fraction above truncation:    {power_above / power_total:.6e}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Grid difference: SHTNS uses {nlon_sh} lons, s2fft uses {N_LON_S2FFT} lons")
print(f"  (Both use {N_LAT_S2FFT} GL latitudes with identical nodes)")
print(f"Max relative difference in KE spectral coefficients (m=0): {max_rel_diff_m0:.6e}")
print(f"Max relative difference in KE spectral coefficients (all): {rel_diffs.max():.6e}")
print(f"s2fft roundtrip aliasing (max rel diff): {rt_rel.max():.6e}")
if mask_sh.any():
    print(f"SHTNS roundtrip aliasing (max rel diff):  {rt_rel_sh.max():.6e}")
print(f"KE Laplacian max relative difference:     {lap_diffs.max():.6e}")
print(f"Spectral power fraction above NTRUNC:     {power_above / power_total:.6e}")
print()
if rel_diffs.max() > 0.01:
    print("CONCLUSION: Significant differences (>1%) between s2fft and SHTNS")
    print("  forward transforms of the SAME quadratic KE field.")
    print("  Aliasing differences ARE a likely contributor to the divergence error.")
elif rel_diffs.max() > 1e-6:
    print("CONCLUSION: Small but non-trivial differences detected.")
    print("  Aliasing may contribute at the ~0.01-1% level but is unlikely")
    print("  the primary cause of the ~20% divergence error.")
else:
    print("CONCLUSION: Forward transforms agree to machine precision.")
    print("  Aliasing differences are NOT the cause of the divergence error.")
