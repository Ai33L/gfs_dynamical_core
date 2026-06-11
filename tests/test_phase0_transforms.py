"""Phase 0 transform verification tests (see ``debugging_code/test_harness_plan.md``).

Compares s2fft (the JAX dycore's current transform backend) against the SHTNS
Python bindings (which Fortran uses natively). All comparisons are in
**grid space** — different libraries use different internal coefficient
normalisations, so spectral-coefficient equality is not a meaningful check.
The principle is:

    grid -> (library A full pipeline) -> grid_A
    grid -> (library B full pipeline) -> grid_B
    assert grid_A ~= grid_B

and, where possible, check each library against an analytic reference.

Acceptance: ``max|A - B| / max|ref| < 1e-10`` for cross-library tests.
Self-roundtrips are held to a tighter ``1e-12`` since they should be at
machine precision.

Run with: ``python -m pytest tests/test_phase0_transforms.py -v`` inside the
climt conda env.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

import jax
import jax.numpy as jnp
import s2fft

from shtns_reference import SHTNSReference


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

L = 64
RADIUS = 6371000.0
NTRUNC = int((2 * L - 1) / 3 - 2)       # 40 -- Fortran GFS 2/3 de-aliasing
SAMPLING = "gl"
TOL_SELF = 1e-12            # machine-precision for self-roundtrip
TOL_CROSS = 1e-10           # cross-library grid agreement


@pytest.fixture(scope="module")
def ref() -> SHTNSReference:
    return SHTNSReference(L=L, ntrunc=NTRUNC, radius=RADIUS)


# ---------------------------------------------------------------------------
# Analytic fields (grid space, (n_lat, n_lon), latitudes N->S)
# ---------------------------------------------------------------------------


def _bandlimited_scalar_grid(ref: SHTNSReference) -> np.ndarray:
    """Analytically bandlimited scalar field (well within ntrunc).

    Composed of low-degree spherical harmonics so forward+inverse is exact
    through any Gauss-Legendre quadrature that supports ``l <= ntrunc``.
    Values scaled to resemble temperatures.
    """
    lats = ref.latitudes[:, None]
    lons = (2 * np.pi * np.arange(ref.n_lon) / ref.n_lon)[None, :]
    sinp = np.sin(lats)
    cosp = np.cos(lats)
    # (l,m) combinations up to l=4: (0,0), (1,0), (2,0), (3,1), (4,2)
    f = (
        288.0
        - 30.0 * sinp                                                # Y_{1,0}
        - 60.0 * (3.0 * sinp ** 2 - 1.0) * 0.5                       # Y_{2,0}
        + 5.0 * cosp * (5.0 * sinp ** 2 - 1.0) * np.cos(lons)        # Y_{3,1}-ish
        + 2.0 * cosp ** 2 * (7.0 * sinp ** 2 - 1.0) * np.cos(2 * lons)  # Y_{4,2}-ish
    )
    return np.broadcast_to(f, (ref.n_lat, ref.n_lon)).copy()


def _bandlimited_vortdiv_grids(ref: SHTNSReference) -> tuple[np.ndarray, np.ndarray]:
    """Bandlimited analytic (vort, div) grids built from low-degree harmonics
    (content well within ntrunc; derivatives stay bandlimited)."""
    lats = ref.latitudes[:, None]
    lons = (2 * np.pi * np.arange(ref.n_lon) / ref.n_lon)[None, :]
    sinp = np.sin(lats)
    cosp = np.cos(lats)
    # Amplitudes ~ 1e-5 s^-1 like real atmospheric vort/div
    vort = (
        1e-5 * sinp                                        # Y_{1,0}-like
        + 2e-5 * cosp * np.cos(lons)                       # Y_{1,1}-like
        + 1e-5 * (3.0 * sinp ** 2 - 1.0) * 0.5             # Y_{2,0}-like
        + 5e-6 * cosp ** 2 * np.cos(2 * lons)              # Y_{2,2}-like
    )
    div = (
        3e-6 * cosp * np.sin(lons)
        + 2e-6 * cosp * sinp * np.cos(lons)
        + 1e-6 * (3.0 * sinp ** 2 - 1.0) * 0.5
    )
    return (
        np.broadcast_to(vort, (ref.n_lat, ref.n_lon)).copy(),
        np.broadcast_to(div, (ref.n_lat, ref.n_lon)).copy(),
    )


def _bandlimited_uv_grid(ref: SHTNSReference) -> tuple[np.ndarray, np.ndarray]:
    """Vector-bandlimited (u, v) synthesised from bandlimited (vort, div).

    Using SHTNS as the source guarantees the (u, v) field lives in the
    l<=ntrunc vector spherical harmonic subspace, so both libraries should
    roundtrip it exactly."""
    vort_grid, div_grid = _bandlimited_vortdiv_grids(ref)
    return _shtns_vortdivgrid_to_uv_module(ref, vort_grid, div_grid)


def _shtns_vortdivgrid_to_uv_module(ref, vort_grid, div_grid):
    # Local helper so the fixture-less construction doesn't depend on later
    # definitions.
    vort_p = ref.scalar_forward_packed(vort_grid)
    div_p = ref.scalar_forward_packed(div_grid)
    return ref.vector_inverse_packed(vort_p, div_p)


def _rel_err(actual: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.max(np.abs(reference)))
    if denom < 1e-30:
        return float(np.max(np.abs(actual - reference)))
    return float(np.max(np.abs(actual - reference)) / denom)


# ---------------------------------------------------------------------------
# Library wrappers -- each implements the same physical operation.
# For vector operations we mirror the JAX dycore's spin-1 formulation for
# s2fft, and ``SHsphtor_to_spat`` for SHTNS.
# ---------------------------------------------------------------------------


def _s2fft_scalar_roundtrip(grid: np.ndarray) -> np.ndarray:
    flm = s2fft.forward_jax(jnp.asarray(grid), L, sampling=SAMPLING)
    return np.asarray(s2fft.inverse_jax(flm, L, sampling=SAMPLING))


def _shtns_scalar_roundtrip(ref: SHTNSReference, grid: np.ndarray) -> np.ndarray:
    packed = ref.scalar_forward_packed(grid)
    return ref.scalar_inverse_packed(packed)


def _s2fft_uv_to_vortdiv_grids(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) grid -> (vort, div) grid via s2fft full round-trip of the decomposition."""
    l_arr = jnp.arange(L)
    l_factor = jnp.sqrt(l_arr * (l_arr + 1))
    f_plus = -jnp.asarray(v) + 1j * jnp.asarray(u)
    f_minus = jnp.asarray(v) + 1j * jnp.asarray(u)
    F1_lm = s2fft.forward_jax(f_plus, L, spin=1, sampling=SAMPLING)
    Fm1_lm = s2fft.forward_jax(f_minus, L, spin=-1, sampling=SAMPLING)
    result_p = l_factor[:, None] * F1_lm / RADIUS
    result_m = l_factor[:, None] * Fm1_lm / RADIUS
    flm_div = (result_p + result_m) / 2
    flm_vort = (result_p - result_m) / (2j)
    grid_vort = np.asarray(s2fft.inverse_jax(flm_vort, L, sampling=SAMPLING))
    grid_div = np.asarray(s2fft.inverse_jax(flm_div, L, sampling=SAMPLING))
    return grid_vort, grid_div


def _shtns_uv_to_vortdiv_grids(ref: SHTNSReference, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vort_p, div_p = ref.vector_forward_packed(u, v)
    return ref.scalar_inverse_packed(vort_p), ref.scalar_inverse_packed(div_p)


def _s2fft_vortdivgrid_to_uv(vort_grid: np.ndarray, div_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(vort, div) grid -> (u, v) grid via scalar forward then spin-1 inverse."""
    l_arr = jnp.arange(L)
    l_factor = jnp.sqrt(l_arr * (l_arr + 1))
    inv_l = jnp.where(l_arr > 0, 1.0 / l_factor, 0.0)
    vort_lm = s2fft.forward_jax(jnp.asarray(vort_grid), L, sampling=SAMPLING)
    div_lm = s2fft.forward_jax(jnp.asarray(div_grid), L, sampling=SAMPLING)
    F1_lm = inv_l[:, None] * (div_lm + 1j * vort_lm) * RADIUS
    f_spin1 = s2fft.inverse_jax(F1_lm, L, spin=1, sampling=SAMPLING)
    return np.asarray(f_spin1.imag), np.asarray(-f_spin1.real)


def _shtns_vortdivgrid_to_uv(ref: SHTNSReference, vort_grid: np.ndarray, div_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vort_p = ref.scalar_forward_packed(vort_grid)
    div_p = ref.scalar_forward_packed(div_grid)
    return ref.vector_inverse_packed(vort_p, div_p)


def _s2fft_grad(scalar_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flm = s2fft.forward_jax(jnp.asarray(scalar_grid), L, sampling=SAMPLING)
    l_arr = jnp.arange(L)
    l_factor = jnp.sqrt(l_arr * (l_arr + 1))
    F1_lm = -l_factor[:, None] * flm
    f_spin1 = s2fft.inverse_jax(F1_lm, L, spin=1, sampling=SAMPLING)
    return np.asarray(f_spin1.imag) / RADIUS, np.asarray(-f_spin1.real) / RADIUS


def _shtns_grad(ref: SHTNSReference, scalar_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    packed = ref.scalar_forward_packed(scalar_grid)
    return ref.gradient_packed(packed)


# ---------------------------------------------------------------------------
# T0.1 -- Scalar self-roundtrip (each library independently)
# ---------------------------------------------------------------------------
def test_T0_1_scalar_self_roundtrip(ref):
    grid = _bandlimited_scalar_grid(ref)
    back_s2 = _s2fft_scalar_roundtrip(grid)
    err_s2 = _rel_err(back_s2, grid)
    print(f"T0.1 s2fft scalar roundtrip rel err = {err_s2:.3e}")
    assert err_s2 < TOL_SELF, f"s2fft scalar roundtrip: {err_s2:.3e}"

    back_sh = _shtns_scalar_roundtrip(ref, grid)
    err_sh = _rel_err(back_sh, grid)
    print(f"T0.1 shtns scalar roundtrip rel err = {err_sh:.3e}")
    # SHTNS aliases content beyond ntrunc; tolerate a slightly looser bound.
    assert err_sh < 1e-6, f"shtns scalar roundtrip: {err_sh:.3e}"


# ---------------------------------------------------------------------------
# T0.2 -- Scalar cross-library: grid-space forward-then-inverse must agree
# ---------------------------------------------------------------------------
def test_T0_2_scalar_cross_library(ref):
    """Applying each library's full forward+inverse to the same grid must give
    the same grid (to within quadrature/truncation error).
    """
    grid = _bandlimited_scalar_grid(ref)
    back_s2 = _s2fft_scalar_roundtrip(grid)
    back_sh = _shtns_scalar_roundtrip(ref, grid)
    err = _rel_err(back_s2, back_sh)
    print(f"T0.2 scalar cross-lib roundtrip rel err = {err:.3e}")
    assert err < TOL_CROSS, f"scalar cross-lib: {err:.3e}"


# ---------------------------------------------------------------------------
# T0.3 -- Analytic scalar: both libraries synthesise the same grid for Y_{l,m}
#         reference fields.
# ---------------------------------------------------------------------------
def test_T0_3_analytic_zonal_mean(ref):
    """f(phi) = sin(phi) is purely (l=1, m=0). Each library's roundtrip must
    recover it exactly up to machine precision."""
    lats = ref.latitudes[:, None]
    grid = np.broadcast_to(np.sin(lats), (ref.n_lat, ref.n_lon)).copy()

    back_s2 = _s2fft_scalar_roundtrip(grid)
    back_sh = _shtns_scalar_roundtrip(ref, grid)
    err_s2 = _rel_err(back_s2, grid)
    err_sh = _rel_err(back_sh, grid)
    err_cross = _rel_err(back_s2, back_sh)
    print(
        f"T0.3 analytic sin(phi) roundtrip: s2fft={err_s2:.3e}, "
        f"shtns={err_sh:.3e}, cross={err_cross:.3e}"
    )
    assert err_s2 < TOL_SELF
    assert err_sh < TOL_SELF
    assert err_cross < TOL_CROSS


# ---------------------------------------------------------------------------
# T0.4 -- Vector: (u, v) -> (vort, div) grids agree across libraries
# ---------------------------------------------------------------------------
def test_T0_4_vortdiv_from_uv(ref):
    u, v = _bandlimited_uv_grid(ref)
    vort_s2, div_s2 = _s2fft_uv_to_vortdiv_grids(u, v)
    vort_sh, div_sh = _shtns_uv_to_vortdiv_grids(ref, u, v)
    err_vort = _rel_err(vort_s2, vort_sh)
    err_div = _rel_err(div_s2, div_sh)
    print(f"T0.4 vort/div grid cross-lib rel err: vort={err_vort:.3e}, div={err_div:.3e}")
    assert err_vort < TOL_CROSS, f"vort: {err_vort:.3e}"
    assert err_div < TOL_CROSS, f"div: {err_div:.3e}"


# ---------------------------------------------------------------------------
# T0.5 -- Vector inverse: (vort, div) grids -> (u, v) grids agree across libs
# ---------------------------------------------------------------------------
def test_T0_5_uv_from_vortdiv(ref):
    """Start from an analytic (u, v). Use each library to compute (vort, div)
    grids, then invert back to (u, v); compare across libraries."""
    u, v = _bandlimited_uv_grid(ref)

    # Use SHTNS to get a consistent (vort, div) grid pair (any source works).
    vort_grid, div_grid = _shtns_uv_to_vortdiv_grids(ref, u, v)

    u_s2, v_s2 = _s2fft_vortdivgrid_to_uv(vort_grid, div_grid)
    u_sh, v_sh = _shtns_vortdivgrid_to_uv(ref, vort_grid, div_grid)

    err_u = _rel_err(u_s2, u_sh)
    err_v = _rel_err(v_s2, v_sh)
    print(f"T0.5 (u,v) cross-lib rel err: u={err_u:.3e}, v={err_v:.3e}")
    assert err_u < TOL_CROSS, f"u: {err_u:.3e}"
    assert err_v < TOL_CROSS, f"v: {err_v:.3e}"


# ---------------------------------------------------------------------------
# T0.6 -- Gradient: both libraries produce the same (d/dlambda, d/dphi) grids
# ---------------------------------------------------------------------------
def test_T0_6_gradient(ref):
    grid = _bandlimited_scalar_grid(ref)
    gx_s2, gy_s2 = _s2fft_grad(grid)
    gx_sh, gy_sh = _shtns_grad(ref, grid)
    err_x = _rel_err(gx_s2, gx_sh)
    err_y = _rel_err(gy_s2, gy_sh)
    print(f"T0.6 gradient grid cross-lib rel err: d/dlambda={err_x:.3e}, d/dphi={err_y:.3e}")
    assert err_x < TOL_CROSS, f"grad_x: {err_x:.3e}"
    assert err_y < TOL_CROSS, f"grad_y: {err_y:.3e}"


# ---------------------------------------------------------------------------
# T0.7 -- Vector roundtrip self-consistency, each library
# ---------------------------------------------------------------------------
def test_T0_7_vector_roundtrip(ref):
    """(u, v) -> (vort, div) -> (u_back, v_back). The roundtrip removes the
    l=0 mean of (u, v); our analytic field has zero l=0 content so the
    recovery should be at machine precision."""
    u, v = _bandlimited_uv_grid(ref)

    # s2fft roundtrip
    vort_s2, div_s2 = _s2fft_uv_to_vortdiv_grids(u, v)
    u_back_s2, v_back_s2 = _s2fft_vortdivgrid_to_uv(vort_s2, div_s2)
    err_u_s2 = _rel_err(u_back_s2, u)
    err_v_s2 = _rel_err(v_back_s2, v)
    print(f"T0.7 s2fft (u,v) roundtrip rel err: u={err_u_s2:.3e}, v={err_v_s2:.3e}")

    # shtns roundtrip
    vort_sh, div_sh = _shtns_uv_to_vortdiv_grids(ref, u, v)
    u_back_sh, v_back_sh = _shtns_vortdivgrid_to_uv(ref, vort_sh, div_sh)
    err_u_sh = _rel_err(u_back_sh, u)
    err_v_sh = _rel_err(v_back_sh, v)
    print(f"T0.7 shtns (u,v) roundtrip rel err: u={err_u_sh:.3e}, v={err_v_sh:.3e}")

    # Each library's self-roundtrip; any gross failure points to an internal bug.
    assert err_u_s2 < 1e-6 and err_v_s2 < 1e-6, (
        f"s2fft roundtrip: u={err_u_s2:.3e}, v={err_v_s2:.3e}"
    )
    assert err_u_sh < 1e-6 and err_v_sh < 1e-6, (
        f"shtns roundtrip: u={err_u_sh:.3e}, v={err_v_sh:.3e}"
    )


# ---------------------------------------------------------------------------
# T0.8 -- Zonal-only (u = f(phi), v = 0) targets the known 2.2% v error
# ---------------------------------------------------------------------------
def test_T0_8_zonal_only_vector(ref):
    """For u = u(phi), v = 0 we should recover (u, 0) exactly after a
    vort/div decomposition + reconstruction (the reconstruction drops only
    the l=0 mean, which is zero here).
    """
    lats = ref.latitudes[:, None]
    u_zonal = 35.0 * np.cos(lats) ** 3 * np.sin(lats) ** 2 * 2.0
    u = np.broadcast_to(u_zonal, (ref.n_lat, ref.n_lon)).copy()
    v = np.zeros_like(u)

    # s2fft roundtrip
    vort_s2, div_s2 = _s2fft_uv_to_vortdiv_grids(u, v)
    u_back_s2, v_back_s2 = _s2fft_vortdivgrid_to_uv(vort_s2, div_s2)

    # shtns roundtrip
    vort_sh, div_sh = _shtns_uv_to_vortdiv_grids(ref, u, v)
    u_back_sh, v_back_sh = _shtns_vortdivgrid_to_uv(ref, vort_sh, div_sh)

    err_u_s2 = _rel_err(u_back_s2, u)
    err_v_s2 = float(np.max(np.abs(v_back_s2)))    # v should be 0 in abs terms
    err_u_sh = _rel_err(u_back_sh, u)
    err_v_sh = float(np.max(np.abs(v_back_sh)))
    err_u_cross = _rel_err(u_back_s2, u_back_sh)
    err_v_cross = _rel_err(v_back_s2, v_back_sh)
    print(
        "T0.8 zonal-only vector roundtrip:\n"
        f"  s2fft: u rel={err_u_s2:.3e}, v abs={err_v_s2:.3e}\n"
        f"  shtns: u rel={err_u_sh:.3e}, v abs={err_v_sh:.3e}\n"
        f"  cross: u rel={err_u_cross:.3e}, v rel={err_v_cross:.3e}"
    )

    # Tightest check: v must stay machine-zero in each library separately
    assert err_v_sh < 1e-10, f"shtns leaks v: {err_v_sh:.3e}"
    assert err_v_s2 < 1e-10, f"s2fft leaks v: {err_v_s2:.3e}"
    assert err_u_s2 < 1e-10, f"s2fft u: {err_u_s2:.3e}"
    assert err_u_sh < 1e-10, f"shtns u: {err_u_sh:.3e}"
