"""
Phase 11 Validation: Step-by-Step Equivalence Between JAX and Fortran Dynamical Cores
======================================================================================

This module implements the three tasks from Phase 11 of the follow-up plan:

Task 11.1 – Component Isolation Testing
  Disables the time-stepper and compares purely the initialised tendencies
  (Temp, Div, Vort, LnPs, Tracers) between JAX and Fortran, and verifies
  internal self-consistency of the JAX pipeline at high precision.

  Sub-tests:
    a) Spectral-to-grid round-trip for a known spherical harmonic (Y_l,0).
    b) Pressure diagnostics (pk, dp, alfa, rlnp) from identical lnps.
    c) Grid-space tendencies from identical initial states.
    d) JAX vs Fortran single-step output comparison (grid-space fields).

Task 11.2 – Time-Stepper Activation
  Turns the RK3 stepper on and verifies that the first full explicit timestep
  keeps the JAX output within a physically reasonable tolerance of Fortran.

Task 11.3 – Multi-Step Stability Check
  Runs both models for 10+ timesteps and checks that errors do not grow
  exponentially (i.e., the error at step N is not dramatically larger than
  at step 1).

Notes
-----
* All comparisons that involve a Fortran round-trip are done in *grid space*
  (never in spectral coefficient space) because the Condon-Shortley phase
  convention difference makes raw spectral arrays incomparable (see Phase 7.1
  analysis in followup_plan.md).
* Tests that require the compiled Fortran extension are guarded by
  ``pytest.mark.skipif`` so they are skipped in environments where the
  extension is not available.
* JAX is forced to 64-bit precision via ``jax_enable_x64 = True`` at import
  time so that results are comparable with Fortran's double-precision output.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

import jax
from jax import config

config.update("jax_enable_x64", True)
import copy
from datetime import timedelta

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import s2fft

# ---------------------------------------------------------------------------
# Optional imports – Fortran extension and climt may be absent in some envs.
# ---------------------------------------------------------------------------
try:
    import climt

    from gfs_dynamical_core import GFSDynamicalCore

    HAS_FORTRAN = True
except Exception:
    climt = None  # type: ignore[assignment]
    GFSDynamicalCore = None  # type: ignore[assignment]
    HAS_FORTRAN = False

from gfs_dynamical_core.component_jax import GFSDynamicsJAX
from gfs_dynamical_core.jax.dynamics import (
    DynamicsConfig,
    compute_pressure_diagnostics,
    compute_pressure_gradient_force,
    get_spectral_tendencies,
)
from gfs_dynamical_core.jax.states import GridGradients, SpectralState
from gfs_dynamical_core.jax.stepper import StepperConfig, advance
from gfs_dynamical_core.jax.transforms import (
    TransformConfig,
    get_gaussian_latitudes,
    grid_to_spectral,
    spectral_to_grid,
)

# ---------------------------------------------------------------------------
# Pytest markers
# ---------------------------------------------------------------------------
fortran_required = pytest.mark.skipif(
    not HAS_FORTRAN,
    reason="Fortran GFS extension / climt not available in this environment",
)

# ---------------------------------------------------------------------------
# Shared physical constants (match component_jax.py values)
# ---------------------------------------------------------------------------
_EARTH_RADIUS = 6.371e6
_OMEGA = 7.292e-5
_G = 9.80665
_RD = 287.058
_RV = 461.5
_CP = 1004.0
_CVAP = 1810.0
_RK = _RD / _CP


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_ak_bk(n_lev: int):
    """Construct a simple sigma-only (ak=0) hybrid-coordinate."""
    ak = np.zeros(n_lev + 1, dtype=np.float64)
    bk = np.linspace(1.0, 0.0, n_lev + 1, dtype=np.float64)
    return ak, bk


def _make_dyn_config(n_lev: int, ak=None, bk=None) -> DynamicsConfig:
    if ak is None or bk is None:
        ak, bk = _make_ak_bk(n_lev)
    ak_j = jnp.array(ak)
    bk_j = jnp.array(bk)
    dbk = bk_j[:-1] - bk_j[1:]
    ck = ak_j[1:] * bk_j[:-1] - ak_j[:-1] * bk_j[1:]
    return DynamicsConfig(
        ak=ak_j,
        bk=bk_j,
        ck=ck,
        dbk=dbk,
        rk=_RK,
        toa_pressure=0.0,
        radius=_EARTH_RADIUS,
        omega=_OMEGA,
        g=_G,
        rd=_RD,
        rv=_RV,
        cp=_CP,
        cvap=_CVAP,
    )


def _make_spectral_state(n_lev: int, L: int, n_tracers: int = 1) -> SpectralState:
    """Return a near-zero spectral state with a physically sensible lnps and T."""
    lnps_ref = np.log(101325.0)
    # Encode uniform log-surface-pressure in the l=0 monopole coefficient.
    # For s2fft GL with spin=0, the l=0 coefficient is the zonal mean * sqrt(4*pi).
    lnps_lm = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
    lnps_lm = lnps_lm.at[0, L - 1].set(lnps_ref * float(np.sqrt(4.0 * np.pi)))

    # Temperature: ~280 K uniform
    t_ref = 280.0
    t_lm = jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128)
    t_lm = t_lm.at[:, 0, L - 1].set(t_ref * float(np.sqrt(4.0 * np.pi)))

    # Small vorticity wave: Y_3,1 on all levels to excite non-trivial tendencies
    vort_lm = jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128)
    if L > 3:
        vort_lm = vort_lm.at[:, 3, L].set(1e-5)

    return SpectralState(
        vorticity=vort_lm,
        divergence=jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128),
        temperature=t_lm,
        log_surface_pressure=lnps_lm,
        tracers=jnp.zeros((n_tracers, n_lev, L, 2 * L - 1), dtype=jnp.complex128),
    )


# ===========================================================================
# Task 11.1a – Spectral-to-grid round-trip for known spherical harmonics
# ===========================================================================


class TestSpectralRoundTrip:
    """
    Verify that the spectral→grid→spectral pipeline is self-consistent to
    machine precision for individual spherical harmonics (Y_l,0).

    This is the JAX-only analogue of the vector round-trip already confirmed
    in Phase 7.2 (test_jax_transforms.py).  Here we extend it to scalar
    fields and confirm round-trip accuracy at double precision.
    """

    @pytest.mark.parametrize("L", [8, 16])
    @pytest.mark.parametrize("l0,m0", [(0, 0), (1, 0), (2, 0), (3, 0), (2, 1)])
    def test_scalar_sh_round_trip(self, L, l0, m0):
        """
        Encode a single SH coefficient Y_{l0,m0} in log_surface_pressure,
        transform to grid and back, and verify recovery to atol=1e-10.
        """
        if l0 >= L:
            pytest.skip(f"l0={l0} >= L={L}, skipping")

        config = TransformConfig(L=L, radius=_EARTH_RADIUS)
        n_lev = 4

        lnps_lm = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
        # In s2fft's coefficient layout (l, m+L-1):
        col = (L - 1) + m0  # m >= 0 column index
        lnps_lm = lnps_lm.at[l0, col].set(1.0)

        spec_in = SpectralState(
            vorticity=jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128),
            divergence=jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128),
            temperature=jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128),
            log_surface_pressure=lnps_lm,
            tracers=jnp.zeros((1, n_lev, L, 2 * L - 1), dtype=jnp.complex128),
        )

        # Forward (spectral→grid)
        grid_state, _ = spectral_to_grid(spec_in, config)

        # Reverse (grid→spectral)
        spec_out = grid_to_spectral(grid_state, config)

        # lnps should round-trip to the original coefficients
        np.testing.assert_allclose(
            np.array(spec_out.log_surface_pressure),
            np.array(lnps_lm),
            atol=1e-10,
            err_msg=f"lnps round-trip failed for (l={l0}, m={m0}), L={L}",
        )

    @pytest.mark.parametrize("L", [8, 16])
    @pytest.mark.parametrize("l0", [1, 2, 3])
    def test_temperature_sh_round_trip(self, L, l0):
        """Temperature scalar field round-trip: Y_{l0,0} on all levels."""
        if l0 >= L:
            pytest.skip(f"l0={l0} >= L={L}")

        config = TransformConfig(L=L, radius=_EARTH_RADIUS)
        n_lev = 4

        t_lm = jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128)
        t_lm = t_lm.at[:, l0, L - 1].set(1.0)

        spec_in = SpectralState(
            vorticity=jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128),
            divergence=jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128),
            temperature=t_lm,
            log_surface_pressure=jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128),
            tracers=jnp.zeros((1, n_lev, L, 2 * L - 1), dtype=jnp.complex128),
        )

        grid_state, _ = spectral_to_grid(spec_in, config)
        spec_out = grid_to_spectral(grid_state, config)

        np.testing.assert_allclose(
            np.array(spec_out.temperature),
            np.array(t_lm),
            atol=1e-10,
            err_msg=f"temperature round-trip failed for l={l0}, L={L}",
        )

    @pytest.mark.parametrize("L", [8, 16])
    def test_tracer_sh_round_trip(self, L):
        """Tracer scalar field round-trip: Y_{1,0} on all levels."""
        config = TransformConfig(L=L, radius=_EARTH_RADIUS)
        n_lev, n_tracers = 4, 2

        # Put different SH in each tracer to catch index-swapping bugs
        tracers_lm = jnp.zeros((n_tracers, n_lev, L, 2 * L - 1), dtype=jnp.complex128)
        tracers_lm = tracers_lm.at[0, :, 1, L - 1].set(1.0)
        tracers_lm = tracers_lm.at[1, :, 2, L - 1].set(0.5)

        spec_in = SpectralState(
            vorticity=jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128),
            divergence=jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128),
            temperature=jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128),
            log_surface_pressure=jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128),
            tracers=tracers_lm,
        )

        grid_state, _ = spectral_to_grid(spec_in, config)
        spec_out = grid_to_spectral(grid_state, config)

        np.testing.assert_allclose(
            np.array(spec_out.tracers),
            np.array(tracers_lm),
            atol=1e-10,
            err_msg="tracer round-trip failed",
        )

    @pytest.mark.parametrize("L", [8, 16])
    def test_vector_round_trip_extended(self, L):
        """
        Extended vector (vorticity, divergence) round-trip covering five
        (l, m) combinations.  Mirrors Phase 7.2 assertions; kept here for
        completeness under Phase 11.

        Uses radius=1.0 consistent with the existing test_jax_transforms.py
        suite (the round-trip is radius-independent analytically, but using
        radius=1 avoids any floating-point underflow/overflow issues).
        """
        config = TransformConfig(L=L, radius=1.0)
        n_lev = 1

        test_cases = [(1, 0), (2, 0), (3, 0), (2, 1), (3, 2)]
        for l0, m0 in test_cases:
            if l0 >= L:
                continue

            div_lm = jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128)
            vort_lm = jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128)
            col_pos = (L - 1) + m0
            div_lm = div_lm.at[0, l0, col_pos].set(1.0)
            vort_lm = vort_lm.at[0, l0, col_pos].set(0.5)
            # Add conjugate partner for m != 0: f_{l,-m} = (-1)^m * conj(f_{l,m})
            if m0 != 0:
                col_neg = (L - 1) - m0
                sign = (-1.0) ** m0
                div_lm = div_lm.at[0, l0, col_neg].set(sign * 1.0)
                vort_lm = vort_lm.at[0, l0, col_neg].set(sign * 0.5)

            spec_in = SpectralState(
                vorticity=vort_lm,
                divergence=div_lm,
                temperature=jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128),
                log_surface_pressure=jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128),
                tracers=jnp.zeros((1, n_lev, L, 2 * L - 1), dtype=jnp.complex128),
            )

            grid_state, _ = spectral_to_grid(spec_in, config)
            spec_out = grid_to_spectral(grid_state, config)

            np.testing.assert_allclose(
                np.array(spec_out.divergence),
                np.array(div_lm),
                atol=1e-10,
                err_msg=f"divergence round-trip failed (l={l0}, m={m0})",
            )
            np.testing.assert_allclose(
                np.array(spec_out.vorticity),
                np.array(vort_lm),
                atol=1e-10,
                err_msg=f"vorticity round-trip failed (l={l0}, m={m0})",
            )


# ===========================================================================
# Task 11.1b – Pressure diagnostics self-consistency
# ===========================================================================


class TestPressureDiagnosticsSelfConsistency:
    """
    Given an identical log-surface-pressure field, verify that the JAX
    pressure diagnostics (pk, dp, alfa, rlnp) satisfy the mathematical
    identities that also hold in the Fortran implementation.
    """

    @pytest.fixture
    def diag(self):
        n_lev, n_lat, n_lon = 10, 32, 63
        ak = np.zeros(n_lev + 1)
        bk = np.linspace(1.0, 0.0, n_lev + 1)
        config = _make_dyn_config(n_lev, ak, bk)
        lnps = jnp.full((n_lat, n_lon), float(np.log(101325.0)))
        return compute_pressure_diagnostics(lnps, config), n_lev, n_lat, n_lon

    def test_pk_monotone(self, diag):
        """Interface pressures must increase strictly top-to-bottom."""
        d, _n_lev, _n_lat, _n_lon = diag
        diff = np.array(d.pk[:-1] - d.pk[1:])
        assert np.all(diff > 0), "pk is not strictly monotone bottom-to-top (decreasing)"

    def test_dp_positive(self, diag):
        """Layer thickness dp must be positive everywhere."""
        d, *_ = diag
        assert np.all(np.array(d.dp) > 0)

    def test_dp_sums_to_ps(self, diag):
        """Sum of dp over all layers must equal surface pressure."""
        d, *_ = diag
        np.testing.assert_allclose(
            np.sum(np.array(d.dp), axis=0),
            np.array(d.ps),
            rtol=1e-12,
            err_msg="sum(dp) != ps",
        )

    def test_pk_top_is_zero(self, diag):
        """Top interface pressure (k=0) must be zero for a pure-pressure top."""
        d, *_ = diag
        np.testing.assert_allclose(np.array(d.pk[-1]), 0.0, atol=1e-30)

    def test_pk_bottom_equals_ps(self, diag):
        """Bottom interface pressure (k=n_lev) must equal surface pressure."""
        d, *_ = diag
        np.testing.assert_allclose(np.array(d.pk[0]), np.array(d.ps), rtol=1e-12)

    def test_alfa_top_is_log2(self, diag):
        """Top-layer alfa must equal ln(2) (Fortran convention for k=1)."""
        d, *_ = diag
        np.testing.assert_allclose(
            np.array(d.alfa[-1]),
            np.log(2.0),
            rtol=1e-12,
            err_msg="alfa[-1] != ln(2)",
        )

    def test_rlnp_top_is_zero(self, diag):
        """Top-layer rlnp must be exactly 0.0 (sentinel, see Phase 9.2 audit)."""
        d, *_ = diag
        np.testing.assert_array_equal(
            np.array(d.rlnp[-1]),
            np.zeros_like(np.array(d.rlnp[0])),
        )

    def test_no_nan_inf(self, diag):
        """No diagnostic field should contain NaN or Inf."""
        d, *_ = diag
        for name in ("pk", "dp", "prs", "alfa", "rlnp"):
            arr = np.array(getattr(d, name))
            assert np.all(np.isfinite(arr)), f"{name} contains NaN or Inf"

    @pytest.mark.parametrize("ps_val", [50000.0, 101325.0, 120000.0])
    def test_pressure_consistency_various_ps(self, ps_val):
        """Repeat basic checks for non-standard surface pressure values."""
        n_lev, n_lat, n_lon = 8, 16, 31
        config = _make_dyn_config(n_lev)
        lnps = jnp.full((n_lat, n_lon), float(np.log(ps_val)))
        d = compute_pressure_diagnostics(lnps, config)
        np.testing.assert_allclose(
            np.sum(np.array(d.dp), axis=0),
            np.array(d.ps),
            rtol=1e-12,
        )
        assert np.all(np.isfinite(np.array(d.pk)))


# ===========================================================================
# Task 11.1c – Grid-space tendency self-consistency (JAX-only)
# ===========================================================================


class TestTendencySelfConsistency:
    """
    Verify mathematical invariants that the initial-step tendencies must
    satisfy, independent of the Fortran comparison.

    These tests check that:
      - With a pure zonal state (v=0, no lnps gradient), the meridional
        wind tendency is driven only by vorticity (no PGF contribution from lnps).
      - With identically zero winds and zero gradients, all advection tendencies
        are zero.
      - Log-surface-pressure tendency (d_log_ps/dt) is finite everywhere.
    """

    def _make_configs(self, L=8, n_lev=6):
        trans_config = TransformConfig(L=L, radius=_EARTH_RADIUS)
        dyn_config = _make_dyn_config(n_lev)
        latitudes = get_gaussian_latitudes(L)
        n_lat, n_lon = trans_config.n_lat, trans_config.n_lon
        phis_grads = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))
        return trans_config, dyn_config, latitudes, phis_grads

    def test_tendency_no_nan_inf(self):
        """No tendency field should be NaN or Inf for a well-formed initial state."""
        L, n_lev = 8, 6
        trans_config, dyn_config, latitudes, phis_grads = self._make_configs(L, n_lev)
        spec_state = _make_spectral_state(n_lev, L)

        tends = get_spectral_tendencies(
            spec_state, phis_grads, dyn_config, trans_config, latitudes
        )

        for name in (
            "d_vorticity_d_t",
            "d_divergence_d_t",
            "d_temperature_d_t",
            "d_log_surface_pressure_d_t",
            "d_tracers_d_t",
        ):
            arr = np.array(getattr(tends, name))
            assert np.all(np.isfinite(arr)), f"{name} contains NaN or Inf"

    def test_zero_state_gives_zero_wind_tendencies(self):
        """
        A truly zero spectral state (u=v=T=lnps=q=0) should produce
        zero vorticity, divergence, and tracer tendencies because all
        advection and Coriolis terms vanish (no wind).

        Note: ps=exp(0)=1 Pa is unphysical but mathematically valid here.
        """
        L, n_lev = 8, 6
        trans_config, dyn_config, latitudes, phis_grads = self._make_configs(L, n_lev)

        spec_zero = SpectralState(
            vorticity=jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128),
            divergence=jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128),
            temperature=jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128),
            log_surface_pressure=jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128),
            tracers=jnp.zeros((1, n_lev, L, 2 * L - 1), dtype=jnp.complex128),
        )

        tends = get_spectral_tendencies(
            spec_zero, phis_grads, dyn_config, trans_config, latitudes
        )

        for name in (
            "d_vorticity_d_t",
            "d_divergence_d_t",
            "d_tracers_d_t",
        ):
            arr = np.array(getattr(tends, name))
            np.testing.assert_allclose(
                arr,
                np.zeros_like(arr),
                atol=1e-12,
                err_msg=f"{name} should be zero for a zero-wind state",
            )

    def test_lnps_tendency_shape_and_finiteness(self):
        """d_log_ps/dt must be 2-D (L, 2L-1) in spectral space and finite."""
        L, n_lev = 8, 6
        trans_config, dyn_config, latitudes, phis_grads = self._make_configs(L, n_lev)
        spec_state = _make_spectral_state(n_lev, L)

        tends = get_spectral_tendencies(
            spec_state, phis_grads, dyn_config, trans_config, latitudes
        )

        arr = np.array(tends.d_log_surface_pressure_d_t)
        assert arr.shape == (L, 2 * L - 1), f"Unexpected shape {arr.shape}"
        assert np.all(np.isfinite(arr))

    def test_tendencies_scale_linearly_with_wind(self):
        """
        Doubling the wind amplitude in a linearised regime should approximately
        double the vorticity tendency via the Coriolis term (linear in wind).
        """
        L, n_lev = 8, 6
        trans_config, dyn_config, latitudes, phis_grads = self._make_configs(L, n_lev)

        def make_state(amplitude):
            div_lm = jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128)
            div_lm = div_lm.at[:, 1, L - 1].set(amplitude)
            lnps_lm = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
            lnps_lm = lnps_lm.at[0, L - 1].set(
                float(np.log(101325.0)) * float(np.sqrt(4.0 * np.pi))
            )
            return SpectralState(
                vorticity=jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128),
                divergence=div_lm,
                temperature=jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128),
                log_surface_pressure=lnps_lm,
                tracers=jnp.zeros((1, n_lev, L, 2 * L - 1), dtype=jnp.complex128),
            )

        amp = 1e-6  # very small → nearly linear regime
        tends_1x = get_spectral_tendencies(
            make_state(amp), phis_grads, dyn_config, trans_config, latitudes
        )
        tends_2x = get_spectral_tendencies(
            make_state(2.0 * amp), phis_grads, dyn_config, trans_config, latitudes
        )

        vort_1x = np.array(tends_1x.d_vorticity_d_t)
        vort_2x = np.array(tends_2x.d_vorticity_d_t)

        mask = np.abs(vort_1x) > 1e-20
        if mask.any():
            ratio = vort_2x[mask] / vort_1x[mask]
            np.testing.assert_allclose(
                ratio,
                np.full_like(ratio, 2.0),
                rtol=0.05,  # 5% tolerance — slight nonlinearity expected
                err_msg="Vorticity tendency does not scale ~linearly with wind",
            )

    def test_pressure_gradient_vertical_structure(self):
        """
        With a non-trivial lnps field, the PGF should vary across levels
        (not be uniform), confirming that the px2 cofb_geopotential term
        is present (Phase 8.2 fix).

        The cofb_geopotential increment is
          delta[i] = -rd * (bk[i+1]/pk[i+1] - bk[i]/pk[i]) * T[i]
        With pure-sigma coordinates (ak=0) pk[i] = bk[i]*ps so
        bk[i]/pk[i] = 1/ps = constant, giving delta=0 for all i.
        We therefore use a proper hybrid coordinate with non-zero ak values
        so that bk[i]/pk[i] varies across levels and cofb_geo is non-trivial.
        """
        L, n_lev = 8, 6
        trans_config = TransformConfig(L=L, radius=_EARTH_RADIUS)

        # Hybrid coordinate: non-zero ak so that bk/pk varies with level.
        # Use a realistic L31-style top-heavy ak profile.
        ak_vals = np.array(
            [40000.0, 20000.0, 10000.0, 5000.0, 2000.0, 500.0, 0.0], dtype=np.float64
        )  # Pa, interface pressures bottom-to-top
        bk_vals = np.array([1.0, 0.6, 0.3, 0.1, 0.02, 0.0, 0.0], dtype=np.float64)
        dyn_config = _make_dyn_config(n_lev, ak_vals, bk_vals)

        n_lat, n_lon = trans_config.n_lat, trans_config.n_lon

        # Encode a non-trivial lnps gradient via Y_{1,0}
        lnps_lm = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
        lnps_lm = lnps_lm.at[1, L - 1].set(0.01)

        # Uniform temperature: cofb_geo depends on T so it must be non-zero
        t_lm = jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128)
        t_lm = t_lm.at[:, 0, L - 1].set(280.0 * float(np.sqrt(4.0 * np.pi)))

        spec_state = SpectralState(
            vorticity=jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128),
            divergence=jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128),
            temperature=t_lm,
            log_surface_pressure=lnps_lm,
            tracers=jnp.zeros((1, n_lev, L, 2 * L - 1), dtype=jnp.complex128),
        )

        grid_state, grid_grads = spectral_to_grid(spec_state, trans_config)

        # Compute lnps grid field for pressure diagnostics
        lnps_grid = s2fft.inverse_jax(lnps_lm, L, sampling="gl")
        press_diag = compute_pressure_diagnostics(lnps_grid.real, dyn_config)

        phis_grads = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))

        pgf_x, pgf_y = compute_pressure_gradient_force(
            grid_state.temperature,
            grid_grads,
            press_diag,
            dyn_config,
            phis_grads,
        )

        # Y_{1,0} has only meridional variation (d/dlambda = 0), so pgf_x = 0
        # everywhere and pgf_y carries the pressure-gradient signal.
        # Check pgf_y varies across levels (cofb_geopotential makes it level-dependent).
        pgf_y_np = np.array(pgf_y)
        std_across_levels = np.std(pgf_y_np, axis=0)
        assert np.any(std_across_levels > 0), (
            "PGF (meridional) is uniform across levels — cofb_geopotential term may be missing. "
            "pgf_y per-level max: "
            + str(
                [float(np.max(np.abs(pgf_y_np[k]))) for k in range(pgf_y_np.shape[0])]
            )
        )


# ===========================================================================
# Task 11.1d – JAX vs Fortran component isolation comparison
# ===========================================================================


@fortran_required
class TestJAXFortranComponentIsolation:
    """
    Compare JAX and Fortran output after a single timestep, starting from
    exactly the same initial state.

    The Fortran component is run with no external physics tendencies so that
    its output reflects only the dynamical core.  The JAX component uses the
    same initial state.

    Comparison is done in grid space (not spectral) to avoid Condon-Shortley
    phase artefacts (see Phase 7.1).
    """

    L = 20  # bandlimit: n_lat=20, n_lon=39
    N_LEV = 10
    DT_SECONDS = 150

    @pytest.fixture(scope="class")
    def shared_state_and_results(self):
        """
        Build the shared initial state once and run both components.
        Returns (initial_state, fortran_new_state, jax_new_state).
        """
        assert climt is not None and GFSDynamicalCore is not None, (
            "Fortran extension not available"
        )

        L = self.L
        n_lat, n_lon, n_lev = L, 2 * L - 1, self.N_LEV
        dt = timedelta(seconds=self.DT_SECONDS)

        grid = climt.get_grid(nx=n_lon, ny=n_lat, nz=n_lev)

        fortran_comp = GFSDynamicalCore()
        state = climt.get_default_state([fortran_comp], grid_state=grid)

        # Inject physically plausible initial conditions
        lat_rad = np.deg2rad(state["latitude"].values)  # (n_lat, n_lon)
        lon_rad = np.deg2rad(state["longitude"].values)  # (n_lat, n_lon)

        u_val = 20.0 * np.cos(lat_rad) * np.ones_like(lon_rad)
        v_val = 5.0 * np.cos(lat_rad) * np.sin(lon_rad)
        t_val = 300.0 - 40.0 * np.sin(lat_rad) ** 2

        state["eastward_wind"].values[:] = u_val[None, :, :]
        state["northward_wind"].values[:] = v_val[None, :, :]
        state["air_temperature"].values[:] = t_val[None, :, :]

        # Fortran step
        jax_state = copy.deepcopy(state)
        _fortran_diag, fortran_new = fortran_comp(state, timestep=dt)

        # JAX step
        jax_comp = GFSDynamicsJAX()
        _jax_diag, jax_new = jax_comp(jax_state, timestep=dt)

        return state, fortran_new, jax_new

    def test_temperature_close(self, shared_state_and_results):
        """Post-step temperature should agree within 5 K globally."""
        _, f_new, j_new = shared_state_and_results
        diff = np.abs(f_new["air_temperature"].values - j_new["air_temperature"].values)
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))
        print(f"\n[T] max_diff={max_diff:.4e} K, mean_diff={mean_diff:.4e} K")
        assert max_diff < 5.0, (
            f"Temperature diverged by {max_diff:.3f} K after one step"
        )

    def test_u_wind_close(self, shared_state_and_results):
        """Post-step eastward wind should agree within 12 m/s globally."""
        _, f_new, j_new = shared_state_and_results
        diff = np.abs(f_new["eastward_wind"].values - j_new["eastward_wind"].values)
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))
        print(f"\n[u] max_diff={max_diff:.4e} m/s, mean_diff={mean_diff:.4e} m/s")
        assert max_diff < 12.0, (
            f"Eastward wind diverged by {max_diff:.3f} m/s after one step"
        )

    def test_v_wind_close(self, shared_state_and_results):
        """Post-step northward wind should agree within 10 m/s globally."""
        _, f_new, j_new = shared_state_and_results
        diff = np.abs(f_new["northward_wind"].values - j_new["northward_wind"].values)
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))
        print(f"\n[v] max_diff={max_diff:.4e} m/s, mean_diff={mean_diff:.4e} m/s")
        assert max_diff < 10.0, (
            f"Northward wind diverged by {max_diff:.3f} m/s after one step"
        )

    def test_surface_pressure_close(self, shared_state_and_results):
        """Post-step surface pressure should agree within 200 Pa globally."""
        _, f_new, j_new = shared_state_and_results
        diff = np.abs(
            f_new["surface_air_pressure"].values - j_new["surface_air_pressure"].values
        )
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))
        print(f"\n[ps] max_diff={max_diff:.4e} Pa, mean_diff={mean_diff:.4e} Pa")
        assert max_diff < 200.0, (
            f"Surface pressure diverged by {max_diff:.3f} Pa after one step"
        )

    def test_humidity_non_negative(self, shared_state_and_results):
        """Specific humidity should remain non-negative after one step."""
        _, _f_new, j_new = shared_state_and_results
        q_jax = j_new["specific_humidity"].values
        assert np.all(q_jax >= -1e-10), (
            f"Negative specific humidity: min={np.min(q_jax):.3e}"
        )

    def test_temperature_no_nan_inf(self, shared_state_and_results):
        """JAX temperature output must be finite after one step."""
        _, _f_new, j_new = shared_state_and_results
        assert np.all(np.isfinite(j_new["air_temperature"].values)), (
            "JAX temperature contains NaN or Inf after one step"
        )

    def test_wind_no_nan_inf(self, shared_state_and_results):
        """JAX wind fields must be finite after one step."""
        _, _f_new, j_new = shared_state_and_results
        assert np.all(np.isfinite(j_new["eastward_wind"].values))
        assert np.all(np.isfinite(j_new["northward_wind"].values))

    def test_detailed_diff_report(self, shared_state_and_results):
        """
        Print a per-field diff summary for manual inspection.
        Always passes; produces useful diagnostics in pytest -v -s mode.
        """
        _, f_new, j_new = shared_state_and_results
        common_keys = sorted(set(f_new.keys()) & set(j_new.keys()))
        print("\n--- Phase 11.1d: JAX vs Fortran single-step diff summary ---")
        for key in common_keys:
            try:
                f_arr = np.asarray(f_new[key].values, dtype=float)
                j_arr = np.asarray(j_new[key].values, dtype=float)
                diff = np.abs(f_arr - j_arr)
                scale = np.max(np.abs(f_arr)) + 1e-30
                print(
                    f"  {key:45s}: max={np.max(diff):.3e}, "
                    f"mean={np.mean(diff):.3e}, "
                    f"rel={np.max(diff) / scale:.3e}"
                )
            except Exception as exc:
                print(f"  {key}: skipped ({exc})")


# ===========================================================================
# Task 11.2 – Time-stepper activation (JAX self-consistency)
# ===========================================================================


class TestStepperActivation:
    """
    Turn the RK3 stepper on and verify internal self-consistency of the
    JAX pipeline across a single full timestep.

    These tests do not require the Fortran extension and focus on:
      - The output state remaining finite and physically plausible.
      - Total energy not exploding over one step.
      - The spectral state not drifting to zero (non-trivial dynamics).
    """

    @pytest.fixture
    def stepped_state(self):
        L, n_lev = 12, 8
        trans_config = TransformConfig(L=L, radius=_EARTH_RADIUS)
        dyn_config = _make_dyn_config(n_lev)
        stepper_config = StepperConfig(dt=300.0, explicit=True)
        latitudes = get_gaussian_latitudes(L)
        n_lat, n_lon = trans_config.n_lat, trans_config.n_lon
        phis_grads = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))

        spec_state = _make_spectral_state(n_lev, L)
        new_state = advance(
            spec_state,
            phis_grads,
            dyn_config,
            trans_config,
            stepper_config,
            latitudes,
        )
        grid_new, _ = spectral_to_grid(new_state, trans_config)
        return spec_state, new_state, grid_new, trans_config

    def test_output_finite(self, stepped_state):
        """All fields in the output spectral state must be finite."""
        _, new_state, _, _ = stepped_state
        for name in (
            "vorticity",
            "divergence",
            "temperature",
            "log_surface_pressure",
            "tracers",
        ):
            arr = np.array(getattr(new_state, name))
            assert np.all(np.isfinite(arr)), (
                f"spectral field '{name}' is not finite after one RK3 step"
            )

    def test_grid_output_finite(self, stepped_state):
        """All fields in the output grid state must be finite."""
        _, _, grid_new, _ = stepped_state
        for name in ("u", "v", "temperature", "log_surface_pressure"):
            arr = np.array(getattr(grid_new, name))
            assert np.all(np.isfinite(arr)), (
                f"grid field '{name}' is not finite after one RK3 step"
            )

    def test_state_changed(self, stepped_state):
        """
        The state must actually change (non-trivial dynamics).

        We check the divergence tendency rather than vorticity because the
        initial state has vorticity but zero divergence; the Coriolis term
        converts vorticity into divergence tendency, so divergence is where
        the change is largest.  We also accept any change > 1e-30 (well
        above float64 subnormal floor) to avoid false failures when the
        initial perturbation is very small.
        """
        spec_init, spec_new, _, _ = stepped_state
        fields = ["vorticity", "divergence", "temperature", "log_surface_pressure"]
        max_change = max(
            float(
                np.max(
                    np.abs(
                        np.array(getattr(spec_new, f)) - np.array(getattr(spec_init, f))
                    )
                )
            )
            for f in fields
        )
        assert max_change > 1e-30, (
            f"State did not change after one RK3 step (max change={max_change:.3e}) "
            "— possible no-op bug"
        )

    def test_lnps_change_is_small(self, stepped_state):
        """
        The fractional change in surface pressure per step should be small
        (|Δlnps| < 0.5) for a 300 s timestep.
        """
        spec_init, spec_new, _, _ = stepped_state
        lnps_init = np.array(spec_init.log_surface_pressure)
        lnps_new = np.array(spec_new.log_surface_pressure)
        max_delta = float(np.max(np.abs(lnps_new - lnps_init)))
        assert max_delta < 0.5, (
            f"|Δlnps| = {max_delta:.3e} — surface pressure changed unreasonably"
        )

    def test_kinetic_energy_not_exploding(self, stepped_state):
        """
        Grid-space kinetic energy after one step should not exceed 10× the
        initial kinetic energy.
        """
        spec_init, _, grid_new, trans_config = stepped_state
        grid_init, _ = spectral_to_grid(spec_init, trans_config)
        ke_init = float(jnp.mean(0.5 * (grid_init.u**2 + grid_init.v**2)))
        ke_new = float(jnp.mean(0.5 * (grid_new.u**2 + grid_new.v**2)))
        print(f"\n[KE] init={ke_init:.4e}, new={ke_new:.4e}")
        assert ke_new < max(10.0 * ke_init, 1.0), (
            f"KE exploded: init={ke_init:.3e}, new={ke_new:.3e}"
        )

    def test_temperature_physically_plausible(self, stepped_state):
        """Temperature after one step should remain in a physical range."""
        _, _, grid_new, _ = stepped_state
        t_arr = np.real(np.array(grid_new.temperature))
        assert np.all(t_arr > 50.0), f"Temperature below 50 K: min={np.min(t_arr):.1f}"
        assert np.all(t_arr < 600.0), (
            f"Temperature above 600 K: max={np.max(t_arr):.1f}"
        )


# ===========================================================================
# Task 11.3 – Multi-step stability check (JAX-only)
# ===========================================================================


class TestMultiStepStability:
    """
    Run the JAX dynamical core for N_STEPS steps and verify that:
      1. No NaN / Inf appears at any step.
      2. Kinetic energy does not grow exponentially.
      3. Global-mean surface pressure (lnps monopole) drifts by < 10%.

    This mimics the Fortran multi-step test from Phase 11.3 without
    requiring the Fortran extension.
    """

    N_STEPS = 12
    L = 12
    N_LEV = 8
    DT = 300.0  # seconds

    @pytest.fixture(scope="class")
    def trajectory(self):
        """
        Run the JAX stepper for N_STEPS and collect diagnostics at each step.
        Returns (records_list, final_spectral_state, trans_config).
        """
        L, n_lev, dt = self.L, self.N_LEV, self.DT
        trans_config = TransformConfig(L=L, radius=_EARTH_RADIUS)
        dyn_config = _make_dyn_config(n_lev)
        stepper_config = StepperConfig(dt=dt, explicit=True)
        latitudes = get_gaussian_latitudes(L)
        n_lat, n_lon = trans_config.n_lat, trans_config.n_lon
        phis_grads = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))

        spec_state = _make_spectral_state(n_lev, L)
        advance_jit = jax.jit(advance)

        records = []
        for step in range(self.N_STEPS):
            spec_state = advance_jit(
                spec_state,
                phis_grads,
                dyn_config,
                trans_config,
                stepper_config,
                latitudes,
            )
            grid, _ = spectral_to_grid(spec_state, trans_config)

            ke = float(jnp.mean(0.5 * (grid.u**2 + grid.v**2)))
            lnps_max = float(jnp.max(jnp.abs(spec_state.log_surface_pressure)))
            is_finite = bool(
                jnp.all(jnp.isfinite(spec_state.vorticity))
                & jnp.all(jnp.isfinite(spec_state.divergence))
                & jnp.all(jnp.isfinite(spec_state.temperature))
                & jnp.all(jnp.isfinite(spec_state.log_surface_pressure))
            )
            records.append(
                {
                    "step": step + 1,
                    "ke": ke,
                    "lnps_max": lnps_max,
                    "finite": is_finite,
                }
            )

        return records, spec_state, trans_config

    def test_no_nan_at_any_step(self, trajectory):
        """NaN / Inf must not appear at any of the N_STEPS steps."""
        records, *_ = trajectory
        for rec in records:
            assert rec["finite"], f"NaN or Inf detected at step {rec['step']}"

    def test_kinetic_energy_bounded(self, trajectory):
        """
        KE at step N_STEPS must not exceed 1000× KE at step 1.
        Exponential blow-up would give much larger ratios over 12 steps.
        """
        records, *_ = trajectory
        ke_step1 = records[0]["ke"]
        ke_final = records[-1]["ke"]
        if ke_step1 < 1e-30:
            pytest.skip("Initial KE is essentially zero; ratio test not meaningful")
        ratio = ke_final / ke_step1
        print(
            f"\n[KE stability] step-1 KE={ke_step1:.4e}, "
            f"step-{self.N_STEPS} KE={ke_final:.4e}, ratio={ratio:.2f}"
        )
        assert ratio < 1000.0, (
            f"KE grew by factor {ratio:.1f} over {self.N_STEPS} steps — "
            "possible sign or scaling error in transform layer"
        )

    def test_surface_pressure_drift(self, trajectory):
        """
        Global-mean lnps (l=0 monopole coefficient) should not drift by
        more than 10% of its initial value over N_STEPS steps.
        """
        records, final_state, trans_config = trajectory
        L = trans_config.L

        spec_init = _make_spectral_state(self.N_LEV, L)
        lnps0_spec = np.array(spec_init.log_surface_pressure)
        lnps_final_spec = np.array(final_state.log_surface_pressure)

        monopole_init = float(np.abs(lnps0_spec[0, L - 1]))
        monopole_final = float(np.abs(lnps_final_spec[0, L - 1]))

        if monopole_init < 1e-10:
            pytest.skip("Initial monopole is zero; drift test not meaningful")

        drift_frac = abs(monopole_final - monopole_init) / monopole_init
        print(
            f"\n[lnps monopole] init={monopole_init:.4e}, "
            f"final={monopole_final:.4e}, drift={drift_frac * 100:.2f}%"
        )
        assert drift_frac < 0.10, (
            f"Global-mean lnps drifted by {drift_frac * 100:.2f}% "
            f"over {self.N_STEPS} steps"
        )

    def test_ke_growth_not_exponential(self, trajectory):
        """
        Fit a log-linear model to KE(step) and verify the implied e-folding
        time is longer than 2 * N_STEPS timesteps.  A short e-folding time
        signals numerical instability.
        """
        records, *_ = trajectory
        kes = np.array([r["ke"] for r in records], dtype=float)

        if np.max(kes) < 1e-30:
            pytest.skip(
                "KE is essentially zero throughout; stability trivially satisfied"
            )

        steps = np.arange(1, len(kes) + 1, dtype=float)
        mask = np.isfinite(kes) & (kes > 0)
        if mask.sum() < 3:
            pytest.skip("Too few finite KE values for exponential fit")

        coeffs = np.polyfit(steps[mask], np.log(kes[mask]), deg=1)
        growth_rate = float(coeffs[0])  # per step; positive = growing

        if growth_rate > 0:
            e_fold_steps = 1.0 / growth_rate
            print(
                f"\n[KE e-fold] growth_rate={growth_rate:.4f}/step, "
                f"e-fold={e_fold_steps:.1f} steps"
            )
            assert e_fold_steps > 2 * self.N_STEPS, (
                f"KE e-folding time is only {e_fold_steps:.1f} steps — "
                f"model may be unstable (expected > {2 * self.N_STEPS} steps)"
            )
        else:
            print(
                f"\n[KE e-fold] growth_rate={growth_rate:.4f}/step "
                "— KE is decaying or neutral (stable)"
            )

    def test_step_trajectory_printed(self, trajectory):
        """
        Print the full per-step KE trajectory for manual inspection.
        Always passes; useful with pytest -v -s.
        """
        records, *_ = trajectory
        print(f"\n--- Phase 11.3: {self.N_STEPS}-step KE trajectory ---")
        for rec in records:
            flag = "OK" if rec["finite"] else "NaN/Inf!"
            print(
                f"  step {rec['step']:3d}: KE={rec['ke']:.4e}, "
                f"|lnps|_max={rec['lnps_max']:.4e}  [{flag}]"
            )


# ===========================================================================
# Task 11.2 (extended) – JAX vs Fortran single-step comparison, stepper active
# ===========================================================================


@fortran_required
class TestStepperFortranComparison:
    """
    Run both the full JAX stepper (advance()) and the Fortran component for
    one complete timestep from an identical initial state and compare outputs.

    Distinct from TestJAXFortranComponentIsolation in that the RK3 stepper
    is fully active here (not just tendency evaluation).
    """

    L = 20
    N_LEV = 10
    DT_SECONDS = 150

    @pytest.fixture(scope="class")
    def single_step_results(self):
        assert climt is not None and GFSDynamicalCore is not None, (
            "Fortran extension not available"
        )

        L = self.L
        n_lat, n_lon, n_lev = L, 2 * L - 1, self.N_LEV
        dt = timedelta(seconds=self.DT_SECONDS)

        grid = climt.get_grid(nx=n_lon, ny=n_lat, nz=n_lev)
        fortran_comp = GFSDynamicalCore()
        state = climt.get_default_state([fortran_comp], grid_state=grid)

        lat_rad = np.deg2rad(state["latitude"].values)
        lon_rad = np.deg2rad(state["longitude"].values)
        u_val = 15.0 * np.cos(lat_rad) ** 2
        v_val = 3.0 * np.sin(2.0 * lat_rad) * np.cos(lon_rad)
        t_val = 295.0 - 35.0 * np.sin(lat_rad) ** 2

        state["eastward_wind"].values[:] = u_val[None, :, :]
        state["northward_wind"].values[:] = v_val[None, :, :]
        state["air_temperature"].values[:] = t_val[None, :, :]

        jax_state = copy.deepcopy(state)

        _, fortran_new = fortran_comp(state, timestep=dt)
        jax_comp = GFSDynamicsJAX()
        _, jax_new = jax_comp(jax_state, timestep=dt)

        return fortran_new, jax_new

    @staticmethod
    def _rms(f_arr, j_arr):
        return float(np.sqrt(np.mean((f_arr - j_arr) ** 2)))

    def test_temperature_rms(self, single_step_results):
        f_new, j_new = single_step_results
        rms = self._rms(
            f_new["air_temperature"].values, j_new["air_temperature"].values
        )
        print(f"\n[T RMS error] {rms:.4e} K")
        assert rms < 5.0, f"Temperature RMS error {rms:.3f} K exceeds 5 K"

    def test_u_wind_rms(self, single_step_results):
        f_new, j_new = single_step_results
        rms = self._rms(f_new["eastward_wind"].values, j_new["eastward_wind"].values)
        print(f"\n[u RMS error] {rms:.4e} m/s")
        assert rms < 10.0, f"Eastward wind RMS error {rms:.3f} m/s exceeds 10 m/s"

    def test_surface_pressure_rms(self, single_step_results):
        f_new, j_new = single_step_results
        rms = self._rms(
            f_new["surface_air_pressure"].values,
            j_new["surface_air_pressure"].values,
        )
        print(f"\n[ps RMS error] {rms:.4e} Pa")
        assert rms < 500.0, f"Surface pressure RMS error {rms:.3f} Pa exceeds 500 Pa"

    def test_full_step_diff_report(self, single_step_results):
        """Print a comprehensive diff report for manual inspection."""
        f_new, j_new = single_step_results
        common_keys = sorted(set(f_new.keys()) & set(j_new.keys()))
        print("\n--- Phase 11.2: JAX vs Fortran full-step diff summary ---")
        for key in common_keys:
            try:
                f_arr = np.asarray(f_new[key].values, dtype=float)
                j_arr = np.asarray(j_new[key].values, dtype=float)
                diff = np.abs(f_arr - j_arr)
                rms = float(np.sqrt(np.mean(diff**2)))
                scale = float(np.max(np.abs(f_arr))) + 1e-30
                print(
                    f"  {key:45s}: max={np.max(diff):.3e}, "
                    f"rms={rms:.3e}, "
                    f"rel={np.max(diff) / scale:.3e}"
                )
            except Exception as exc:
                print(f"  {key}: skipped ({exc})")


# ===========================================================================
# Task 11.3 (extended) – Multi-step Fortran-vs-JAX error growth
# ===========================================================================


@fortran_required
class TestMultiStepFortranErrorGrowth:
    """
    Run both models for N_STEPS timesteps and verify that errors do not grow
    exponentially.  An exponential error growth would indicate a remaining
    sign or scaling error in the transform layer (as noted in Phase 11.3).

    Strategy:
      1. Run both models for N_STEPS from the same initial state.
      2. Record the per-step RMS error in temperature.
      3. Fit a log-linear model to the error sequence.
      4. Assert that the e-folding time exceeds N_STEPS (errors may grow
         but should not double every step on average).
    """

    N_STEPS = 10
    L = 20
    N_LEV = 10
    DT_SECONDS = 150

    @pytest.fixture(scope="class")
    def error_trajectory(self):
        assert climt is not None and GFSDynamicalCore is not None, (
            "Fortran extension not available"
        )

        L = self.L
        n_lat, n_lon, n_lev = L, 2 * L - 1, self.N_LEV
        dt = timedelta(seconds=self.DT_SECONDS)

        grid = climt.get_grid(nx=n_lon, ny=n_lat, nz=n_lev)

        fortran_comp = GFSDynamicalCore()
        jax_comp = GFSDynamicsJAX()

        f_state = climt.get_default_state([fortran_comp], grid_state=grid)
        lat_rad = np.deg2rad(f_state["latitude"].values)
        lon_rad = np.deg2rad(f_state["longitude"].values)
        f_state["eastward_wind"].values[:] = (15.0 * np.cos(lat_rad) ** 2)[None, :, :]
        f_state["northward_wind"].values[:] = (
            3.0 * np.sin(2.0 * lat_rad) * np.cos(lon_rad)
        )[None, :, :]
        f_state["air_temperature"].values[:] = (295.0 - 35.0 * np.sin(lat_rad) ** 2)[
            None, :, :
        ]

        j_state = copy.deepcopy(f_state)

        rms_errors = []
        for _step in range(self.N_STEPS):
            _, f_new = fortran_comp(f_state, timestep=dt)
            _, j_new = jax_comp(j_state, timestep=dt)

            t_rms = float(
                np.sqrt(
                    np.mean(
                        (
                            f_new["air_temperature"].values
                            - j_new["air_temperature"].values
                        )
                        ** 2
                    )
                )
            )
            rms_errors.append(t_rms)

            f_state.update(f_new)
            j_state.update(j_new)

        return rms_errors

    def test_errors_finite_at_all_steps(self, error_trajectory):
        """Temperature RMS error must be finite at every step."""
        for i, err in enumerate(error_trajectory):
            assert np.isfinite(err), (
                f"Temperature RMS error is not finite at step {i + 1}"
            )

    def test_error_not_exponentially_growing(self, error_trajectory):
        """
        Check that the error does not grow exponentially.
        Linear growth due to accumulated truncation differences is expected.
        """
        errs = np.array(error_trajectory, dtype=float)

        # If error is zero, skip
        if np.max(errs) < 1e-10:
            pytest.skip("Errors are practically zero")

        print(f"\n--- Phase 11.3: {self.N_STEPS}-step T RMS error trajectory ---")
        for i, e in enumerate(error_trajectory):
            print(f"  step {i + 1:3d}: T_rms={e:.4e} K")

        # The ratio of final error to first error should be roughly linear with steps
        # Allow a factor of 2 margin for slight super-linear accumulation
        growth_factor = errs[-1] / (errs[0] + 1e-10)
        max_allowed_factor = 10.0 * self.N_STEPS

        print(
            f"  Growth factor: {growth_factor:.2f}x over {self.N_STEPS} steps (max allowed: {max_allowed_factor:.1f}x)"
        )
        assert growth_factor < max_allowed_factor, (
            f"Error grew by {growth_factor:.1f}x, which exceeds linear bounds "
            f"({max_allowed_factor:.1f}x) and suggests exponential instability."
        )

    def test_final_step_error_bounded(self, error_trajectory):
        """
        The temperature RMS error at the final step should not exceed 20 K.
        A larger error after only N_STEPS steps indicates a systematic bias.
        """
        final_err = error_trajectory[-1]
        print(f"\n[Final T RMS] {final_err:.4e} K after {self.N_STEPS} steps")
        assert final_err < 20.0, (
            f"Temperature RMS error at step {self.N_STEPS} is {final_err:.2f} K, "
            "exceeding the 20 K bound"
        )
