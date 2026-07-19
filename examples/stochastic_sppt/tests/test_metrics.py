import jax
import jax.numpy as jnp
from examples.stochastic_sppt import metrics


def test_afcrps_alpha1_equals_fair():
    ens = jax.random.normal(jax.random.PRNGKey(0), (8, 5, 5))
    y = jax.random.normal(jax.random.PRNGKey(1), (5, 5))
    a = metrics.afcrps(ens, y, alpha=1.0)
    f = metrics.fair_crps(ens, y)
    assert abs(float(a) - float(f)) < 1e-10


def test_afcrps_nonnegative_and_differentiable():
    ens = jax.random.normal(jax.random.PRNGKey(2), (6, 4))
    y = jax.random.normal(jax.random.PRNGKey(3), (4,))
    assert float(metrics.afcrps(ens, y, 0.95)) >= 0.0

    def loss(scale):
        return metrics.afcrps(ens * scale, y, 0.95)

    g = jax.grad(loss)(1.0)
    # finite-difference check
    eps = 1e-4
    fd = (loss(1.0 + eps) - loss(1.0 - eps)) / (2 * eps)
    assert abs(float(g) - float(fd)) < 1e-3


def test_reliable_ensemble_spread_error_ratio_near_one():
    key = jax.random.PRNGKey(4)
    M, N = 40, 4000
    # Reliable/exchangeable: a common latent center per location; truth and each
    # member are that center plus independent unit noise (M+1 exchangeable draws).
    kc, kt, ke = jax.random.split(key, 3)
    center = jax.random.normal(kc, (N,))
    truth = center + jax.random.normal(kt, (N,))
    ens = center[None] + jax.random.normal(ke, (M, N))
    r = float(metrics.spread_error_ratio(ens, truth))
    assert 0.9 < r < 1.1


def test_afcrps_multidraw_k1_equals_afcrps():
    ens = jax.random.normal(jax.random.PRNGKey(0), (8, 5, 5))
    y = jax.random.normal(jax.random.PRNGKey(1), (5, 5))
    single = metrics.afcrps(ens, y, 0.95)
    multi = metrics.afcrps_multidraw(ens, y[None], 0.95)   # K=1
    assert abs(float(single) - float(multi)) < 1e-12


def test_afcrps_multidraw_is_mean_over_draws():
    ens = jax.random.normal(jax.random.PRNGKey(2), (6, 4))
    truths = jax.random.normal(jax.random.PRNGKey(3), (5, 4))   # K=5 draws
    multi = float(metrics.afcrps_multidraw(ens, truths, 0.95))
    manual = float(jnp.mean(jnp.stack(
        [metrics.afcrps(ens, truths[k], 0.95) for k in range(truths.shape[0])])))
    assert abs(multi - manual) < 1e-12


def test_afcrps_multidraw_differentiable():
    ens = jax.random.normal(jax.random.PRNGKey(4), (6, 4))
    truths = jax.random.normal(jax.random.PRNGKey(5), (5, 4))

    def loss(scale):
        return metrics.afcrps_multidraw(ens * scale, truths, 0.95)

    g = jax.grad(loss)(1.0)
    eps = 1e-4
    fd = (loss(1.0 + eps) - loss(1.0 - eps)) / (2 * eps)
    assert abs(float(g) - float(fd)) < 1e-3
