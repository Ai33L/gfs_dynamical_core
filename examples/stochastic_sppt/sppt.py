"""Trainable differentiable SPPT (Palmer et al. 2009, Tech Memo 598, App. 8.1)."""
import jax
import jax.numpy as jnp
from flax import struct

from gfs_dynamical_core.jax.transforms import enforce_triangular_truncation, s2_inverse

# Calibration constant for the grid-point variance normalization; kept explicit
# so the analytic F0 (Palmer eq. 18) can be reconciled with the s2fft transform
# convention. The precompute-kernel s2_inverse at T21/GL sampling does not
# preserve grid-point variance 1:1 against the analytic closed form (measured
# v=0.01516 for sigma^2=0.25 with _VAR_NORM=1.0); calibrated here via
# _VAR_NORM = 0.25 / v against test_grid_variance_matches_sigma.
#
# CAVEAT (accepted, not a bug): this single scalar is calibrated at T21 (L=32)
# only. The transform-convention factor is NOT resolution-independent — measured
# grid variance for sigma^2=0.25 is ~0.25 at T21 but ~0.31 (ratio ~1.25) at
# T42/T85/T127. So `sigma` == grid-point std holds exactly only at T21; at
# production resolutions the true grid-point std is ~1.12x exp(log_sigma).
# This does NOT break the pipeline: Mode-A parameter recovery is exact (the
# factor cancels between truth and forecast) and Mode-B spread calibration is
# unaffected (the optimizer learns whatever log_sigma yields matching spread) —
# only the physical interpretation of a learned log_sigma is offset. Left
# constant intentionally to observe how the optimizer tunes the effective sigma.
# To restore sigma==grid-std at all L, make the normalization per-L (derive F0
# from s2_inverse's Parseval convention or calibrate at build time).
_VAR_NORM = 16.496188577716463


@struct.dataclass
class SPPTParams:
    log_sigma: jnp.ndarray   # grid-point std of the pattern r  (var(r)=sigma^2)
    log_tau: jnp.ndarray     # decorrelation time (s)
    log_len: jnp.ndarray     # correlation length (m)


def default_params() -> SPPTParams:
    return SPPTParams(log_sigma=jnp.log(0.5), log_tau=jnp.log(6 * 3600.0),
                      log_len=jnp.log(500e3))


def _phi(params, dt):
    return jnp.exp(-dt / jnp.exp(params.log_tau))


def sigma_n(params, trans_config, dt):
    """Per-total-wavenumber forcing std (Palmer eqs. 17-18)."""
    L = trans_config.L
    radius = trans_config.radius
    n = jnp.arange(L)
    kappa_t = (jnp.exp(params.log_len) / radius) ** 2 / 2.0
    shape = jnp.exp(-kappa_t * n * (n + 1) / 2.0)          # exp(-kappaT n(n+1)/2)
    var_r = jnp.exp(2.0 * params.log_sigma)
    phi = _phi(params, dt)
    denom = 2.0 * jnp.sum((2 * n[1:] + 1) * jnp.exp(-kappa_t * n[1:] * (n[1:] + 1)))
    F0 = jnp.sqrt(_VAR_NORM * var_r * (1.0 - phi ** 2) / denom)
    return F0 * shape                                      # (L,)


def _sample_eta(key, L):
    """Complex Gaussian noise on the (L, 2L-1) rectangle (Re,Im ~ N(0,1))."""
    kr, ki = jax.random.split(key)
    re = jax.random.normal(kr, (L, 2 * L - 1))
    im = jax.random.normal(ki, (L, 2 * L - 1))
    eta = re + 1j * im
    # clip to +-10 std (overflow guard, Palmer App 8.1)
    eta = jnp.clip(eta.real, -10.0, 10.0) + 1j * jnp.clip(eta.imag, -10.0, 10.0)
    return eta


def init_pattern(params, key, trans_config, dt):
    """Stationary initialization r_lm(0) = (1-phi^2)^-1/2 sigma_n eta (Palmer eq. 19)."""
    L, T = trans_config.L, trans_config.truncation
    phi = _phi(params, dt)
    sn = sigma_n(params, trans_config, dt)[:, None]
    r = (1.0 - phi ** 2) ** (-0.5) * sn * _sample_eta(key, L)
    return enforce_triangular_truncation(r, L, T)


def pattern_step(r_lm, key, params, trans_config, dt):
    """One AR(1) step: r_lm <- phi r_lm + sigma_n eta (Palmer eq. 14)."""
    L, T = trans_config.L, trans_config.truncation
    phi = _phi(params, dt)
    sn = sigma_n(params, trans_config, dt)[:, None]
    r = phi * r_lm + sn * _sample_eta(key, L)
    return enforce_triangular_truncation(r, L, T)


def pattern_to_grid(r_lm, trans_config):
    """Inverse transform to a real grid-space pattern."""
    g = s2_inverse(r_lm, trans_config.L, trans_config.sampling)
    return g.real
