"""
Minimal test of vector spectral <-> grid round-trips using s2fft.

Tests:
1. Forward only: put a known (vort, div) into spectral, convert to (u,v) grid,
   convert back to spectral (vort, div). Check round-trip.
2. Inverse only: put a known (u,v) on the grid, convert to spectral (vort,div),
   convert back to (u,v). Check round-trip.
3. Single-mode tests: put energy in one (l,m) mode at a time.

This helps isolate whether the problem is in:
  - inverse (spectral -> grid): vort,div -> u,v
  - forward (grid -> spectral): u,v -> vort,div
  - or both
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

import jax
import jax.numpy as jnp
import numpy as np
import s2fft

L = 8
sampling = "gl"
n_lat, n_lon = L, 2 * L - 1
radius = 6371000.0

l_arr = jnp.arange(L)
l_factor = jnp.sqrt(l_arr * (l_arr + 1.0))
inv_l_factor = jnp.where(l_arr > 0, 1.0 / l_factor, 0.0)


# ── Current inverse: spectral (vort, div) -> grid (u, v) ────────────────
def current_inverse(vort_lm, div_lm):
    """Current code: single spin+1 inverse."""
    F1_lm = inv_l_factor[:, None] * (div_lm + 1j * vort_lm) * radius
    f_spin1 = s2fft.inverse_jax(F1_lm, L, spin=1, sampling=sampling)
    u = f_spin1.imag
    v = -f_spin1.real
    return u, v


def proposed_inverse(vort_lm, div_lm):
    """Proposed: use BOTH spin+1 and spin-1 inverse, analogous to forward fix."""
    # spin+1:  ₁f = -vθ + i vφ,  with ₁f_lm = (D + iζ) * R / √(l(l+1))
    # spin-1:  ₋₁f = vθ + i vφ,  with ₋₁f_lm = (D - iζ) * R / √(l(l+1))
    F1_lm = inv_l_factor[:, None] * (div_lm + 1j * vort_lm) * radius
    Fm1_lm = inv_l_factor[:, None] * (div_lm - 1j * vort_lm) * radius

    f_spin1 = s2fft.inverse_jax(F1_lm, L, spin=1, sampling=sampling)
    f_spinm1 = s2fft.inverse_jax(Fm1_lm, L, spin=-1, sampling=sampling)

    # ₁f = -vθ + i vφ
    # ₋₁f = vθ + i vφ
    # => vφ = Im(₁f + ₋₁f) / 2  (but both should give same Im for real fields)
    # => vθ = Re(₋₁f - ₁f) / 2
    #
    # Actually for real vector fields:
    #   u = vφ,  v = -vθ  (in the physics convention where u=eastward, v=northward)
    #   Wait — need to be careful about sign conventions.
    #
    # From ₁f = -vθ + i vφ:   vφ = Im(₁f),  vθ = -Re(₁f)
    # From ₋₁f = vθ + i vφ:   vφ = Im(₋₁f), vθ = Re(₋₁f)
    #
    # For a real vector field these should be consistent. Let's average:
    u = 0.5 * (f_spin1.imag + f_spinm1.imag)
    v = 0.5 * (-f_spin1.real + f_spinm1.real)  # v = -vθ... wait
    # Actually: v_northward = -v_theta (since theta increases southward)
    # But the SHTNS/Fortran convention: getuv returns (u, v) = (v_phi, v_theta)
    # where v_theta points southward.
    # Let me just try both and see which round-trips.

    # Option A: same as current code but averaged
    u_A = 0.5 * (f_spin1.imag + f_spinm1.imag)
    v_A = -0.5 * (f_spin1.real + f_spinm1.real)

    # Option B: using the difference
    u_B = 0.5 * (f_spin1.imag + f_spinm1.imag)
    v_B = 0.5 * (f_spinm1.real - f_spin1.real)

    return u_A, v_A, u_B, v_B


# ── Current forward: grid (u, v) -> spectral (vort, div) ────────────────
def current_forward(u, v):
    """Current code: both spin+1 and spin-1 forward."""
    f_plus = -v + 1j * u
    f_minus = v + 1j * u

    F1_lm = s2fft.forward_jax(f_plus, L, spin=1, sampling=sampling)
    Fm1_lm = s2fft.forward_jax(f_minus, L, spin=-1, sampling=sampling)

    result_p = l_factor[:, None] * F1_lm / radius
    result_m = l_factor[:, None] * Fm1_lm / radius

    div_lm = (result_p + result_m) / 2
    vort_lm = (result_p - result_m) / (2j)

    return vort_lm, div_lm


# ── Gradient computation (for testing) ───────────────────────────────────
def spectral_gradient(flm):
    """Compute gradient of scalar field using spin-1 inverse."""
    F1_lm = -l_factor[:, None] * flm
    f_spin1 = s2fft.inverse_jax(F1_lm, L, spin=1, sampling=sampling)
    grad_x = f_spin1.imag / radius
    grad_y = -f_spin1.real / radius
    return grad_x, grad_y


def compare(name, a, b, atol=1e-10):
    """Compare two arrays and print results."""
    a = np.asarray(a)
    b = np.asarray(b)
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        diff = np.abs(a - b)
    else:
        diff = np.abs(a - b)
    max_diff = diff.max()
    max_val = max(np.abs(a).max(), np.abs(b).max(), 1e-30)
    rel = max_diff / max_val
    status = "OK" if max_diff < atol else "FAIL"
    print(f"  {name:40s}  max|diff|={max_diff:.4e}  rel={rel:.4e}  [{status}]")
    return max_diff < atol


# ══════════════════════════════════════════════════════════════════════════
# TEST 1: spectral -> grid -> spectral round-trip
# ══════════════════════════════════════════════════════════════════════════
def test_roundtrip_spec_grid_spec():
    print("\n" + "=" * 70)
    print("TEST 1: spectral -> grid -> spectral (vort,div -> u,v -> vort,div)")
    print("=" * 70)

    # Put energy in a single mode
    for l0 in [1, 2, 3]:
        for m0 in [0, 1, -1]:
            if abs(m0) > l0:
                continue
            print(f"\n  Mode l={l0}, m={m0}:")

            vort_lm = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
            div_lm = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)

            # Put 1.0 in vorticity at (l0, m0)
            vort_lm = vort_lm.at[l0, L - 1 + m0].set(1.0 + 0.0j)

            # Inverse: spectral -> grid
            u, v = current_inverse(vort_lm, div_lm)

            # Forward: grid -> spectral
            vort_out, div_out = current_forward(u, v)

            compare(f"vort round-trip (l={l0},m={m0})", vort_lm, vort_out, atol=1e-8)
            compare(f"div round-trip (l={l0},m={m0})", div_lm, div_out, atol=1e-8)

            # Also test with divergence
            vort_lm2 = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
            div_lm2 = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
            div_lm2 = div_lm2.at[l0, L - 1 + m0].set(1.0 + 0.0j)

            u2, v2 = current_inverse(vort_lm2, div_lm2)
            vort_out2, div_out2 = current_forward(u2, v2)

            compare(f"div->vort leak (l={l0},m={m0})", vort_lm2, vort_out2, atol=1e-8)
            compare(f"div round-trip (l={l0},m={m0})", div_lm2, div_out2, atol=1e-8)


# ══════════════════════════════════════════════════════════════════════════
# TEST 2: Check u,v field magnitudes from pure vorticity
# ══════════════════════════════════════════════════════════════════════════
def test_uv_magnitudes():
    print("\n" + "=" * 70)
    print("TEST 2: u,v magnitudes from pure vorticity & pure divergence")
    print("=" * 70)

    for l0 in [1, 2, 4]:
        vort_lm = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
        div_lm = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
        vort_lm = vort_lm.at[l0, L - 1].set(1.0)  # m=0

        u, v = current_inverse(vort_lm, div_lm)
        print(f"\n  Pure vort l={l0}, m=0:")
        print(f"    max|u|={np.abs(u).max():.6e}  max|v|={np.abs(v).max():.6e}")
        print(f"    u should be nonzero (zonal flow from vorticity)")
        print(f"    v should be ~0 for m=0 axisymmetric vorticity")

        # Check if v is spuriously large
        u_A, v_A, u_B, v_B = proposed_inverse(vort_lm, div_lm)
        print(
            f"    proposed A: max|u|={np.abs(u_A).max():.6e}  max|v|={np.abs(v_A).max():.6e}"
        )
        print(
            f"    proposed B: max|u|={np.abs(u_B).max():.6e}  max|v|={np.abs(v_B).max():.6e}"
        )


# ══════════════════════════════════════════════════════════════════════════
# TEST 3: Compare current vs proposed inverse
# ══════════════════════════════════════════════════════════════════════════
def test_inverse_comparison():
    print("\n" + "=" * 70)
    print("TEST 3: Current vs proposed inverse transform")
    print("=" * 70)

    # Use a realistic-ish vorticity field: solid body rotation
    vort_lm = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
    # l=1, m=0 vorticity = solid body rotation
    vort_lm = vort_lm.at[1, L - 1].set(1.0)
    # Add some l=2 for asymmetry
    vort_lm = vort_lm.at[2, L - 1 + 1].set(0.3 + 0.2j)

    div_lm = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
    div_lm = div_lm.at[2, L - 1].set(0.5)

    u_cur, v_cur = current_inverse(vort_lm, div_lm)
    u_A, v_A, u_B, v_B = proposed_inverse(vort_lm, div_lm)

    print("\n  Current inverse:")
    print(f"    max|u|={np.abs(u_cur).max():.6e}  max|v|={np.abs(v_cur).max():.6e}")
    print(f"    u is real: {np.allclose(u_cur.imag, 0, atol=1e-12)}")
    print(f"    v is real: {np.allclose(v_cur.imag, 0, atol=1e-12)}")

    print("\n  Proposed A (average of Re/Im):")
    print(f"    max|u|={np.abs(u_A).max():.6e}  max|v|={np.abs(v_A).max():.6e}")
    print(f"    u is real: {np.allclose(np.imag(u_A), 0, atol=1e-12)}")
    print(f"    v is real: {np.allclose(np.imag(v_A), 0, atol=1e-12)}")

    print("\n  Proposed B (difference):")
    print(f"    max|u|={np.abs(u_B).max():.6e}  max|v|={np.abs(v_B).max():.6e}")

    # Do a round-trip with each
    print("\n  Round-trip test (forward after inverse):")
    vort_cur, div_cur = current_forward(np.real(u_cur), np.real(v_cur))
    vort_A, div_A = current_forward(np.real(u_A), np.real(v_A))
    vort_B, div_B = current_forward(np.real(u_B), np.real(v_B))

    compare("current: vort roundtrip", vort_lm, vort_cur, atol=1e-8)
    compare("current: div roundtrip", div_lm, div_cur, atol=1e-8)
    compare("proposed A: vort roundtrip", vort_lm, vort_A, atol=1e-8)
    compare("proposed A: div roundtrip", div_lm, div_A, atol=1e-8)
    compare("proposed B: vort roundtrip", vort_lm, vort_B, atol=1e-8)
    compare("proposed B: div roundtrip", div_lm, div_B, atol=1e-8)


# ══════════════════════════════════════════════════════════════════════════
# TEST 4: Check imaginary parts of u, v (should be zero for real fields)
# ══════════════════════════════════════════════════════════════════════════
def test_reality():
    print("\n" + "=" * 70)
    print("TEST 4: Reality check — u,v should be real for real vort/div")
    print("=" * 70)

    # Construct vort/div with proper conjugate symmetry for real fields:
    # f_{l,-m} = (-1)^m * conj(f_{l,m})
    vort_lm = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
    div_lm = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)

    # Set m=0 (real)
    vort_lm = vort_lm.at[1, L - 1].set(1.0)
    vort_lm = vort_lm.at[3, L - 1].set(0.5)

    # Set m=1 with conjugate symmetry
    vort_lm = vort_lm.at[2, L - 1 + 1].set(0.3 + 0.2j)
    vort_lm = vort_lm.at[2, L - 1 - 1].set((-1) ** 1 * (0.3 - 0.2j))

    # Set m=2 with conjugate symmetry
    vort_lm = vort_lm.at[3, L - 1 + 2].set(0.1 - 0.4j)
    vort_lm = vort_lm.at[3, L - 1 - 2].set((-1) ** 2 * (0.1 + 0.4j))

    div_lm = div_lm.at[2, L - 1].set(0.5)
    div_lm = div_lm.at[2, L - 1 + 1].set(0.2 + 0.1j)
    div_lm = div_lm.at[2, L - 1 - 1].set((-1) ** 1 * (0.2 - 0.1j))

    # Test scalar inverse (should give real result)
    grid_vort = s2fft.inverse_jax(vort_lm, L, sampling=sampling)
    print(f"\n  Scalar inverse of vort_lm:")
    print(f"    max|imag|={np.abs(np.imag(grid_vort)).max():.4e}  (should be ~0)")

    # Test spin+1 inverse
    F1_lm = inv_l_factor[:, None] * (div_lm + 1j * vort_lm) * radius
    f_spin1 = s2fft.inverse_jax(F1_lm, L, spin=1, sampling=sampling)
    u_cur = f_spin1.imag
    v_cur = -f_spin1.real

    print(f"\n  Current inverse (spin+1 only):")
    print(f"    f_spin1 max|val|={np.abs(f_spin1).max():.6e}")
    print(f"    u = Im(f_spin1):  max|u|={np.abs(u_cur).max():.6e}")
    print(f"    v = -Re(f_spin1): max|v|={np.abs(v_cur).max():.6e}")
    print(f"    u is real: max|Im(u)|={np.abs(np.imag(u_cur)).max():.4e}")
    print(f"    v is real: max|Im(v)|={np.abs(np.imag(v_cur)).max():.4e}")

    # Test with both spins
    Fm1_lm = inv_l_factor[:, None] * (div_lm - 1j * vort_lm) * radius
    f_spinm1 = s2fft.inverse_jax(Fm1_lm, L, spin=-1, sampling=sampling)

    print(f"\n  Spin-1 inverse:")
    print(f"    f_spinm1 max|val|={np.abs(f_spinm1).max():.6e}")

    # For a real vector field: ₋₁f = conj(₁f)
    print(f"\n  Consistency check: ₋₁f should = conj(₁f) for real fields:")
    print(
        f"    max|₋₁f - conj(₁f)|={np.abs(np.array(f_spinm1) - np.conj(np.array(f_spin1))).max():.4e}"
    )

    # If ₋₁f = conj(₁f), then:
    #   u = Im(₁f) = Im(₋₁f)  (both give same result)
    #   v = -Re(₁f) = Re(₋₁f)  (both give same result)
    # And the average/difference give nothing new.
    # BUT if s2fft doesn't enforce this symmetry, then we need both.

    u_avg = 0.5 * (np.array(f_spin1).imag + np.array(f_spinm1).imag)
    v_from_spin1 = -np.array(f_spin1).real
    v_from_spinm1 = np.array(f_spinm1).real
    v_avg_A = -0.5 * (np.array(f_spin1).real + np.array(f_spinm1).real)
    v_avg_B = 0.5 * (np.array(f_spinm1).real - np.array(f_spin1).real)

    print(f"\n  v from spin+1 only: max|v|={np.abs(v_from_spin1).max():.6e}")
    print(f"  v from spin-1 only: max|v|={np.abs(v_from_spinm1).max():.6e}")
    print(f"  v avg A (-avg Re):  max|v|={np.abs(v_avg_A).max():.6e}")
    print(f"  v avg B (diff Re):  max|v|={np.abs(v_avg_B).max():.6e}")
    print(f"  max|v_spin1 - v_spinm1|={np.abs(v_from_spin1 - v_from_spinm1).max():.4e}")

    # Round-trip
    print(f"\n  Round-trip test:")
    vort_out_cur, div_out_cur = current_forward(np.real(u_cur), np.real(v_cur))
    compare("current: vort", vort_lm, vort_out_cur, atol=1e-8)
    compare("current: div", div_lm, div_out_cur, atol=1e-8)

    # With the spin-1 v
    vort_out_m1, div_out_m1 = current_forward(np.real(u_cur), v_from_spinm1.real)
    compare("spin-1 v: vort", vort_lm, vort_out_m1, atol=1e-8)
    compare("spin-1 v: div", div_lm, div_out_m1, atol=1e-8)


# ══════════════════════════════════════════════════════════════════════════
# TEST 5: Verify s2fft spin conventions with a known analytic case
# ══════════════════════════════════════════════════════════════════════════
def test_s2fft_spin_conventions():
    print("\n" + "=" * 70)
    print("TEST 5: s2fft spin transform convention verification")
    print("=" * 70)

    # Y_1^0 = sqrt(3/4pi) * cos(theta)
    # The stream function psi = Y_1^0 gives solid-body rotation:
    #   u = -d(psi)/d(theta) * 1/sin(theta) ... no, let's just check numerically

    # Put 1.0 in vort at l=1, m=0
    vort_lm = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
    vort_lm = vort_lm.at[1, L - 1].set(1.0)
    div_lm = jnp.zeros_like(vort_lm)

    # Stream function: psi_lm = -vort_lm / (l(l+1)) * R^2
    # But the spin-1 field is: ₁f_lm = (D + iζ) * R / √(l(l+1))
    # For pure vorticity: ₁f_lm = i * ζ_lm * R / √(l(l+1))

    # Get grid values
    u_cur, v_cur = current_inverse(vort_lm, div_lm)

    # The Gauss-Legendre latitudes
    cos_theta, _ = np.polynomial.legendre.leggauss(L)
    thetas = np.flip(np.arccos(cos_theta))
    lats = np.pi / 2.0 - thetas  # geographic latitude, N->S

    print(f"\n  Pure vort (l=1, m=0) -> solid body rotation")
    print(f"  u should be constant * sin(lat), v should be 0")
    print(f"  Latitudes: {np.rad2deg(lats)[:4]}... (N->S)")
    print(f"  u at lon=0: {np.real(np.array(u_cur))[:, 0]}")
    print(f"  v at lon=0: {np.real(np.array(v_cur))[:, 0]}")
    print(f"  max|v|={np.abs(v_cur).max():.6e}  (should be 0)")
    print(f"  max|u|={np.abs(u_cur).max():.6e}")

    # Check u ~ sin(lat) pattern
    u_at_lon0 = np.real(np.array(u_cur))[:, 0]
    sin_lat = np.sin(lats)
    if np.abs(u_at_lon0).max() > 1e-10:
        ratio = u_at_lon0 / (sin_lat + 1e-30)
        print(f"  u/sin(lat) at lon=0: {ratio}")
        print(f"  (should be constant if u ~ sin(lat))")


# ══════════════════════════════════════════════════════════════════════════
# TEST 6: Direct comparison with SHTNS-style getuv
# ══════════════════════════════════════════════════════════════════════════
def test_shtns_equivalent():
    print("\n" + "=" * 70)
    print("TEST 6: SHTNS-equivalent getuv via scalar sphtor approach")
    print("=" * 70)
    print("  SHTNS getuv does: psi_lm = invlap * R * vort_lm")
    print("                    chi_lm = invlap * R * div_lm")
    print("                    (u, v) = sphtor_to_spat(psi_lm, chi_lm)")
    print("  where sphtor_to_spat computes the toroidal+poloidal vector field.")
    print()
    print("  In s2fft terms, SHTNS sphtor_to_spat(Ψ, Φ) computes:")
    print("    ₁f = ð(Ψ + iΦ)  and returns u = Im(₁f), v = Re(₁f)")
    print("  where ð is the spin-raising operator: ðY_l^m = √(l(l+1)) ₁Y_l^m")
    print()

    # Construct test data
    vort_lm = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
    div_lm = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)

    vort_lm = vort_lm.at[2, L - 1].set(1.0)
    vort_lm = vort_lm.at[3, L - 1 + 1].set(0.5 + 0.3j)
    vort_lm = vort_lm.at[3, L - 1 - 1].set(-1 * (0.5 - 0.3j))  # conjugate sym

    div_lm = div_lm.at[2, L - 1 + 1].set(0.4 - 0.2j)
    div_lm = div_lm.at[2, L - 1 - 1].set(-1 * (0.4 + 0.2j))

    # SHTNS approach: psi = invlap * R * vort, chi = invlap * R * div
    invlap = jnp.where(l_arr > 0, -1.0 / (l_arr * (l_arr + 1.0)), 0.0)
    psi_lm = invlap[:, None] * radius * vort_lm
    chi_lm = invlap[:, None] * radius * div_lm

    # ð(Ψ + iΦ) = √(l(l+1)) * ₁Y_l^m * (Ψ_lm + iΦ_lm)
    # So ₁f_lm = √(l(l+1)) * (Ψ_lm + iΦ_lm)
    F1_shtns = l_factor[:, None] * (psi_lm + 1j * chi_lm)
    f_spin1_shtns = s2fft.inverse_jax(F1_shtns, L, spin=1, sampling=sampling)

    # Now expand: √(l(l+1)) * (Ψ + iΦ) = √(l(l+1)) * invlap * R * (vort + i div)
    #           = √(l(l+1)) * (-1/(l(l+1))) * R * (vort + i div)
    #           = -R / √(l(l+1)) * (vort + i div)
    # Compare with current code:
    #   F1_lm = R / √(l(l+1)) * (div + i vort)
    # These differ by: current = R/√ll1 * (div + i vort), shtns = -R/√ll1 * (vort + i div)
    # Let's see: -R/√ll1 * (vort + i div) = R/√ll1 * (-vort - i div)
    # vs current: R/√ll1 * (div + i vort)
    # These are NOT the same!  current has (div + i*vort), shtns has -(vort + i*div) = -vort - i*div
    # Note: div + i*vort = i*(vort - i*div) = i*(vort + i*(-div))
    # And: -vort - i*div = -(vort + i*div)
    # So these are different by more than just a phase!

    F1_current = inv_l_factor[:, None] * (div_lm + 1j * vort_lm) * radius
    print(f"  F1_shtns[2, L-1]: {np.array(F1_shtns[2, L - 1]):.6f}")
    print(f"  F1_current[2, L-1]: {np.array(F1_current[2, L - 1]):.6f}")
    print(
        f"  ratio: {np.array(F1_shtns[2, L - 1]) / np.array(F1_current[2, L - 1]):.6f}"
    )
    print()

    f_spin1_current = s2fft.inverse_jax(F1_current, L, spin=1, sampling=sampling)

    # SHTNS returns u = Im(₁f), v = Re(₁f) (not v = -Re!)
    # Wait, actually SHTNS sphtor_to_spat convention may differ.
    # Let's just check all sign combos:
    u_shtns_A = np.array(f_spin1_shtns).imag
    v_shtns_A = np.array(f_spin1_shtns).real
    u_shtns_B = np.array(f_spin1_shtns).imag
    v_shtns_B = -np.array(f_spin1_shtns).real

    u_current = np.array(f_spin1_current).imag
    v_current = -np.array(f_spin1_current).real

    print(f"  SHTNS-equiv approach:")
    print(f"    u (Im): max={np.abs(u_shtns_A).max():.6e}")
    print(f"    v (+Re): max={np.abs(v_shtns_A).max():.6e}")
    print(f"    v (-Re): max={np.abs(v_shtns_B).max():.6e}")

    print(f"  Current approach:")
    print(f"    u (Im): max={np.abs(u_current).max():.6e}")
    print(f"    v (-Re): max={np.abs(v_current).max():.6e}")

    # Round-trip each
    print(f"\n  Round-trip (SHTNS-equiv, v=+Re):")
    vort_rt, div_rt = current_forward(u_shtns_A, v_shtns_A)
    compare("  vort", vort_lm, vort_rt, atol=1e-8)
    compare("  div", div_lm, div_rt, atol=1e-8)

    print(f"\n  Round-trip (SHTNS-equiv, v=-Re):")
    vort_rt, div_rt = current_forward(u_shtns_B, v_shtns_B)
    compare("  vort", vort_lm, vort_rt, atol=1e-8)
    compare("  div", div_lm, div_rt, atol=1e-8)

    print(f"\n  Round-trip (current code):")
    vort_rt, div_rt = current_forward(u_current, v_current)
    compare("  vort", vort_lm, vort_rt, atol=1e-8)
    compare("  div", div_lm, div_rt, atol=1e-8)

    # Try the SHTNS F1 with current forward
    print(f"\n  SHTNS F1 formula with current forward:")
    f_spin1_shtns_grid = s2fft.inverse_jax(F1_shtns, L, spin=1, sampling=sampling)
    # Try all 4 combos of sign for (u, v)
    for u_sign, v_sign, label in [
        (1, 1, "u=+Im, v=+Re"),
        (1, -1, "u=+Im, v=-Re"),
        (-1, 1, "u=-Im, v=+Re"),
        (-1, -1, "u=-Im, v=-Re"),
    ]:
        u_try = u_sign * np.array(f_spin1_shtns_grid).imag
        v_try = v_sign * np.array(f_spin1_shtns_grid).real
        vort_rt, div_rt = current_forward(u_try, v_try)
        v_ok = compare(f"    {label}: vort", vort_lm, vort_rt, atol=1e-8)
        d_ok = compare(f"    {label}: div", div_lm, div_rt, atol=1e-8)
        if v_ok and d_ok:
            print(f"    >>> {label} WORKS! <<<")


if __name__ == "__main__":
    test_roundtrip_spec_grid_spec()
    test_uv_magnitudes()
    test_inverse_comparison()
    test_reality()
    test_s2fft_spin_conventions()
    test_shtns_equivalent()
