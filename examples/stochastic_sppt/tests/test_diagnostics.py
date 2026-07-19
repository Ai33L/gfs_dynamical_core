import jax
import jax.numpy as jnp
from examples.stochastic_sppt.config import ModelConfig
from examples.stochastic_sppt.model import build_model, rest_state
from examples.stochastic_sppt import diagnostics


def test_extract_fields_shapes_and_ps():
    bundle = build_model(ModelConfig(resolution="T21"))
    spec = rest_state(bundle, jax.random.PRNGKey(0), ps0=1.0e5)
    f = diagnostics.extract_fields(spec, bundle)
    assert set(f) == {"u850", "v850", "t500", "vort500", "ps"}
    assert f["u850"].shape == (32, 63)
    assert abs(float(jnp.mean(f["ps"])) - 1.0e5) < 1.0  # ps = exp(lnps)


def test_vertical_interp_recovers_level_value():
    # linear-in-log-p field: interpolating at an exact layer pressure recovers it
    nlev, nlat, nlon = 5, 2, 2
    prs = jnp.array([9e4, 7e4, 5e4, 3e4, 1e4])[:, None, None] * jnp.ones((1, nlat, nlon))
    field = jnp.arange(nlev)[:, None, None] * jnp.ones((1, nlat, nlon)) * 1.0
    out = diagnostics.vertical_interp(field, prs, 5e4)
    assert float(jnp.max(jnp.abs(out - 2.0))) < 1e-6
