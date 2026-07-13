import jax
import jax.numpy as jnp
from examples.stochastic_sppt.config import ModelConfig
from examples.stochastic_sppt.model import build_model, rest_state
from examples.stochastic_sppt import sppt, rollout


def test_zero_amplitude_members_identical():
    bundle = build_model(ModelConfig(resolution="T21"))
    spec0 = rest_state(bundle, jax.random.PRNGKey(0), t0=280.0)
    p = sppt.SPPTParams(log_sigma=jnp.log(1e-30), log_tau=jnp.log(6 * 3600.0),
                        log_len=jnp.log(500e3))
    keys = jax.random.split(jax.random.PRNGKey(1), 3)
    out = rollout.ensemble_rollout(p, spec0, keys, bundle, lead_steps=(2, 4))
    assert out["t500"].shape == (3, 2, 32, 63)
    # with ~zero noise, all members coincide
    assert float(jnp.std(out["t500"], axis=0).max()) < 1e-6


def test_rollout_is_differentiable_and_finite():
    bundle = build_model(ModelConfig(resolution="T21"))
    spec0 = rest_state(bundle, jax.random.PRNGKey(0), t0=280.0)
    keys = jax.random.split(jax.random.PRNGKey(2), 4)

    def loss(log_sigma):
        p = sppt.SPPTParams(log_sigma=log_sigma, log_tau=jnp.log(6 * 3600.0),
                            log_len=jnp.log(500e3))
        out = rollout.ensemble_rollout(p, spec0, keys, bundle, lead_steps=(3,))
        return jnp.mean(jnp.var(out["u850"], axis=0))

    g = jax.grad(loss)(jnp.log(0.5))
    assert jnp.isfinite(g)
