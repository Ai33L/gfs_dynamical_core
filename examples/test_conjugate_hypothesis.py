"""
Diagnostic: does the vector round-trip failure for m≠0 come from
missing conjugate symmetry in the test coefficients?

Hypothesis
----------
The current inverse transform (spin+1 only) is actually correct for *real*
vector fields — i.e. spectral coefficients that satisfy

    f_{l,-m} = (-1)^m  conj(f_{l,m})

TEST 1 in test_vector_roundtrip.py set a single (l, m≠0) mode without its
conjugate partner, producing a *complex* grid field.  Taking only the real
part before feeding back into the forward transform destroys information,
giving the 0.5 error.

This script checks:
  A) Single mode WITHOUT conjugate  →  expect ~0.5 error (complex field)
  B) Single mode WITH conjugate     →  expect ~0 error   (real field)
  C) Mixed multi-mode WITH conjugate →  expect ~0 error

If B and C pass, the inverse transform is fine and the blow-up cause is
elsewhere.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

import jax.numpy as jnp
import numpy as np
import s2fft

L = 8
sampling = "gl"
radius = 6371000.0

l_arr = jnp.arange(L)
l_factor = jnp.sqrt(l_arr * (l_arr + 1.0))
inv_l_factor = jnp.where(l_arr > 0, 1.0 / l_factor, 0.0)


def inverse_current(vort_lm, div_lm):
    """spectral (vort, div) -> grid (u, v)  [current single-spin+1 code]"""
    F1_lm = inv_l_factor[:, None] * (div_lm + 1j * vort_lm) * radius
    f = s2fft.inverse_jax(F1_lm, L, spin=1, sampling=sampling)
    return f.imag, -f.real


def forward_current(u, v):
    """grid (u, v) -> spectral (vort, div)  [current dual-spin code]"""
    f_plus = -v + 1j * u
    f_minus = v + 1j * u
    F1 = s2fft.forward_jax(f_plus, L, spin=1, sampling=sampling)
    Fm1 = s2fft.forward_jax(f_minus, L, spin=-1, sampling=sampling)
    rp = l_factor[:, None] * F1 / radius
    rm = l_factor[:, None] * Fm1 / radius
    return (rp - rm) / (2j), (rp + rm) / 2


def inverse_dual(vort_lm, div_lm):
    """spectral (vort, div) -> grid (u, v)  [proposed dual-spin inverse]"""
    F1_lm = inv_l_factor[:, None] * (div_lm + 1j * vort_lm) * radius
    Fm1_lm = inv_l_factor[:, None] * (div_lm - 1j * vort_lm) * radius
    f1 = s2fft.inverse_jax(F1_lm, L, spin=1, sampling=sampling)
    fm1 = s2fft.inverse_jax(Fm1_lm, L, spin=-1, sampling=sampling)
    u = 0.5 * (f1.imag + fm1.imag)
    v = 0.5 * (fm1.real - f1.real)
    return u, v


def add_conjugate(flm, l0, m0, val):
    """Set flm[l0, m0] = val and flm[l0, -m0] = (-1)^m0 * conj(val)."""
    flm = flm.at[l0, L - 1 + m0].set(val)
    if m0 != 0:
        flm = flm.at[l0, L - 1 - m0].set((-1) ** m0 * val.conjugate())
    return flm


def compare(label, a, b, atol=1e-8):
    d = np.abs(np.asarray(a) - np.asarray(b)).max()
    mx = max(np.abs(np.asarray(a)).max(), np.abs(np.asarray(b)).max(), 1e-30)
    ok = d < atol
    tag = "OK" if ok else "FAIL"
    print(f"  {label:55s} max|err|={d:.4e}  rel={d / mx:.4e}  [{tag}]")
    return ok


# ── helpers ──────────────────────────────────────────────────────────────
def roundtrip(label, vort_lm, div_lm, inverse_fn):
    """inverse -> forward round-trip, prints results."""
    u, v = inverse_fn(vort_lm, div_lm)
    # For real fields u,v should already be real; take real part just in case
    u_real = np.real(np.asarray(u))
    v_real = np.real(np.asarray(v))
    is_real_u = np.abs(np.imag(np.asarray(u))).max()
    is_real_v = np.abs(np.imag(np.asarray(v))).max()
    vort_out, div_out = forward_current(u_real, v_real)
    ok1 = compare(f"{label}: vort", vort_lm, vort_out)
    ok2 = compare(f"{label}: div", div_lm, div_out)
    if is_real_u > 1e-10 or is_real_v > 1e-10:
        print(
            f"    ⚠  grid field is COMPLEX  max|Im(u)|={is_real_u:.2e}  max|Im(v)|={is_real_v:.2e}"
        )
    return ok1 and ok2


# ═══════════════════════════════════════════════════════════════════════
print("=" * 72)
print("PART A: Single mode WITHOUT conjugate partner (expect FAIL for m≠0)")
print("=" * 72)
pass_a = 0
total_a = 0
for l0 in [1, 2, 3]:
    for m0 in [0, 1, -1]:
        if abs(m0) > l0:
            continue
        total_a += 1
        vort = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
        div = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
        vort = vort.at[l0, L - 1 + m0].set(1.0 + 0j)  # NO conjugate
        ok = roundtrip(
            f"NO conj  l={l0} m={m0:+d} (current inv)", vort, div, inverse_current
        )
        if ok:
            pass_a += 1
print(f"\n  Part A passed {pass_a}/{total_a}")

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART B: Single mode WITH conjugate partner (expect PASS)")
print("=" * 72)
pass_b = 0
total_b = 0
for l0 in [1, 2, 3]:
    for m0 in [0, 1, 2]:
        if m0 > l0:
            continue
        total_b += 1
        vort = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
        div = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
        val = 0.6 + 0.4j if m0 != 0 else 1.0 + 0j
        vort = add_conjugate(vort, l0, m0, val)
        ok = roundtrip(
            f"WITH conj l={l0} m={m0} (current inv)", vort, div, inverse_current
        )
        if ok:
            pass_b += 1

        # also test divergence
        total_b += 1
        vort2 = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
        div2 = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
        div2 = add_conjugate(div2, l0, m0, val)
        ok2 = roundtrip(
            f"WITH conj l={l0} m={m0} div (current inv)", vort2, div2, inverse_current
        )
        if ok2:
            pass_b += 1
print(f"\n  Part B passed {pass_b}/{total_b}")

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART C: Multi-mode realistic field WITH conjugate symmetry")
print("=" * 72)
vort = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
div = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)

# vorticity: several modes
vort = add_conjugate(vort, 1, 0, 1.0 + 0j)
vort = add_conjugate(vort, 2, 0, 0.5 + 0j)
vort = add_conjugate(vort, 2, 1, 0.3 + 0.2j)
vort = add_conjugate(vort, 3, 1, 0.1 - 0.4j)
vort = add_conjugate(vort, 3, 2, 0.2 + 0.1j)
vort = add_conjugate(vort, 4, 3, 0.05 - 0.15j)

# divergence: a few modes
div = add_conjugate(div, 2, 0, 0.5 + 0j)
div = add_conjugate(div, 2, 1, 0.4 - 0.2j)
div = add_conjugate(div, 3, 2, 0.15 + 0.25j)

ok_c1 = roundtrip("multi-mode (current inv)", vort, div, inverse_current)
ok_c2 = roundtrip("multi-mode (dual inv)", vort, div, inverse_dual)

print(f"\n  Part C: current inverse {'PASS' if ok_c1 else 'FAIL'}")
print(f"  Part C: dual inverse    {'PASS' if ok_c2 else 'FAIL'}")

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART D: Does the *forward* transform produce conjugate-symmetric output?")
print("=" * 72)
print("  (i.e. does grid_to_spectral always give proper flm for real u,v?)")

# Make a real (u, v) field from conjugate-symmetric spectral input
u_grid, v_grid = inverse_current(vort, div)
u_real = np.real(np.asarray(u_grid))
v_real = np.real(np.asarray(v_grid))

vort_out, div_out = forward_current(u_real, v_real)


# Check conjugate symmetry of output
def check_conj_sym(name, flm):
    max_err = 0.0
    for ll in range(L):
        for mm in range(1, min(ll + 1, L)):
            pos = np.asarray(flm[ll, L - 1 + mm])
            neg = np.asarray(flm[ll, L - 1 - mm])
            expected_neg = (-1) ** mm * pos.conjugate()
            err = abs(neg - expected_neg)
            if err > max_err:
                max_err = err
    ok = max_err < 1e-10
    print(
        f"  {name:40s}  max conj-sym error = {max_err:.4e}  [{'OK' if ok else 'FAIL'}]"
    )
    return ok


check_conj_sym("vort_out from forward", vort_out)
check_conj_sym("div_out from forward", div_out)

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART E: Does the *real model* produce conjugate-symmetric spectral data?")
print("=" * 72)
print("  If the initial conditions or timestepping breaks conjugate symmetry,")
print("  the inverse transform will corrupt the fields even though the math is")
print("  correct for real fields.")
print()
print("  → This needs to be checked against the actual baroclinic wave init.")
print("    If init produces non-conjugate spectral data, THAT is the real bug.")
print()

# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
print(f"  Part A (no conjugate, expect fail for m≠0): {pass_a}/{total_a} passed")
print(f"  Part B (with conjugate, expect all pass):   {pass_b}/{total_b} passed")
print(f"  Part C multi-mode current inverse:          {'PASS' if ok_c1 else 'FAIL'}")
print(f"  Part C multi-mode dual inverse:             {'PASS' if ok_c2 else 'FAIL'}")
print()
if pass_b == total_b and ok_c1:
    print("  ✅ CONFIRMED: Current single-spin inverse is CORRECT for real fields.")
    print("     The original test failures were due to non-conjugate test inputs.")
    print("     The blow-up cause is NOT the inverse vector transform.")
    print()
    print("  🔍 NEXT: Check whether the model's spectral data maintains conjugate")
    print("     symmetry throughout the simulation.  If symmetry breaks, the")
    print("     inverse transform will produce complex grid values whose imaginary")
    print("     parts get silently discarded, corrupting the dynamics.")
elif pass_b < total_b:
    print("  ❌ CONJUGATE HYPOTHESIS REJECTED: Even with proper conjugate symmetry,")
    print("     the current inverse fails.  The dual-spin inverse IS needed.")
    if ok_c2:
        print("     The dual-spin inverse fixes the problem.")
    else:
        print("     Even the dual-spin inverse fails — deeper issue.")
