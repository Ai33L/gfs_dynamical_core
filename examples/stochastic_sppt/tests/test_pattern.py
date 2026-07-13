import jax
import jax.numpy as jnp
from examples.stochastic_sppt.config import ModelConfig
from examples.stochastic_sppt.model import build_model
from examples.stochastic_sppt import sppt


def _tc():
    return build_model(ModelConfig(resolution="T21")).trans_config


def test_pattern_is_reality_symmetric():
    tc = _tc()
    p = sppt.default_params()
    r = sppt.init_pattern(p, jax.random.PRNGKey(0), tc, dt=1800.0)
    # grid transform of a reality-symmetric field must be (numerically) real
    g = sppt.pattern_to_grid(r, tc)
    assert g.dtype.kind == "f"  # pattern_to_grid returns real part
    # reality condition: r[l,-m] == (-1)^m conj(r[l,m])
    L = tc.L
    m = jnp.arange(-L + 1, L)
    sign = jnp.where(m % 2 == 0, 1.0, -1.0)
    mirror = sign * jnp.conj(r[..., ::-1])
    neg = m < 0
    assert float(jnp.max(jnp.abs((r - mirror)[:, neg]))) < 1e-9


def test_grid_variance_matches_sigma():
    tc = _tc()
    p = sppt.SPPTParams(log_sigma=jnp.log(0.5), log_tau=jnp.log(6 * 3600.0),
                        log_len=jnp.log(500e3))
    keys = jax.random.split(jax.random.PRNGKey(1), 200)
    grids = jax.vmap(lambda k: sppt.pattern_to_grid(
        sppt.init_pattern(p, k, tc, dt=1800.0), tc))(keys)
    # grid-point variance should be ~ sigma**2 = 0.25 (within 20%)
    v = float(jnp.var(grids))
    assert 0.20 < v < 0.30


def test_ar1_stationarity():
    tc = _tc()
    p = sppt.default_params()
    key = jax.random.PRNGKey(2)
    r = sppt.init_pattern(p, key, tc, dt=1800.0)
    v0 = float(jnp.var(sppt.pattern_to_grid(r, tc)))
    for i in range(30):
        r = sppt.pattern_step(r, jax.random.fold_in(key, i), p, tc, dt=1800.0)
    v1 = float(jnp.var(sppt.pattern_to_grid(r, tc)))
    # variance stays within a factor ~1.6 (single-sample, stationary process)
    assert 0.5 < (v1 / v0) < 2.0


def test_pattern_is_differentiable_in_params():
    tc = _tc()
    key = jax.random.PRNGKey(3)

    def scalar(log_sigma):
        p = sppt.SPPTParams(log_sigma=log_sigma, log_tau=jnp.log(6 * 3600.0),
                            log_len=jnp.log(500e3))
        return jnp.sum(sppt.pattern_to_grid(sppt.init_pattern(p, key, tc, 1800.0), tc) ** 2)

    g = jax.grad(scalar)(jnp.log(0.5))
    assert jnp.isfinite(g) and float(g) != 0.0
