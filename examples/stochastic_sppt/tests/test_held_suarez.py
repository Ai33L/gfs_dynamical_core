import jax
import jax.numpy as jnp
from examples.stochastic_sppt.config import ModelConfig
from examples.stochastic_sppt.model import build_model, rest_state
from examples.stochastic_sppt.held_suarez import hs_tendencies
from gfs_dynamical_core.jax.transforms import spectral_to_grid


def _grid(bundle, seed=0):
    spec = rest_state(bundle, jax.random.PRNGKey(seed), t0=300.0)
    grid, _ = spectral_to_grid(spec, bundle.trans_config)
    return grid


def test_temperature_relaxes_toward_equilibrium():
    bundle = build_model(ModelConfig(resolution="T21"))
    grid = _grid(bundle)
    tends = hs_tendencies(grid, bundle)
    # At 300 K uniform, T is above equilibrium almost everywhere -> cooling.
    assert float(jnp.mean(tends.virtual_temperature)) < 0.0
    assert tends.virtual_temperature.shape == grid.temperature.shape


def test_friction_only_in_boundary_layer_and_opposes_wind():
    bundle = build_model(ModelConfig(resolution="T21"))
    grid = _grid(bundle)
    # inject a wind so friction is nonzero
    grid = grid.replace(u=grid.u + 10.0)
    tends = hs_tendencies(grid, bundle)
    # top level (k=-1, near TOA, sigma << sigma_b) has ~zero friction
    assert float(jnp.max(jnp.abs(tends.u[-1]))) < 1e-6
    # bottom level (k=0, surface) friction opposes the +10 m/s wind
    assert float(jnp.mean(tends.u[0])) < 0.0
    # lnps and tracer tendencies are zero (dry core)
    assert float(jnp.max(jnp.abs(tends.log_surface_pressure))) == 0.0
