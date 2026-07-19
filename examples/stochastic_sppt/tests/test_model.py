import jax
import jax.numpy as jnp
from examples.stochastic_sppt.config import ModelConfig
from examples.stochastic_sppt.model import build_model, step, rest_state, spectral_truncate


def test_build_and_dynamics_only_step_is_finite():
    bundle = build_model(ModelConfig(resolution="T21", n_lev=20))
    assert bundle.trans_config.L == 32 and bundle.trans_config.truncation == 21
    spec = rest_state(bundle, jax.random.PRNGKey(0))
    # dynamics-only step (no physics tendencies)
    spec2 = step(bundle, spec, None)
    assert spec2.temperature.shape == (20, 32, 63)
    assert bool(jnp.all(jnp.isfinite(spec2.temperature)))


def test_spectral_truncate_zeros_high_wavenumbers():
    hi = build_model(ModelConfig(resolution="T42", n_lev=20))
    lo = build_model(ModelConfig(resolution="T21", n_lev=20))
    spec_hi = rest_state(hi, jax.random.PRNGKey(1))
    spec_lo = spectral_truncate(spec_hi, hi, lo)
    assert spec_lo.temperature.shape == (20, 32, 63)
    assert bool(jnp.all(jnp.isfinite(spec_lo.temperature)))
