import jax
import jax.numpy as jnp
from examples.stochastic_sppt.config import ModelConfig
from examples.stochastic_sppt.model import build_model, rest_state
from examples.stochastic_sppt.held_suarez import hs_tendencies
from examples.stochastic_sppt import sppt
from gfs_dynamical_core.jax.transforms import spectral_to_grid


def _setup(seed=0):
    bundle = build_model(ModelConfig(resolution="T21"))
    grid, _ = spectral_to_grid(rest_state(bundle, jax.random.PRNGKey(seed), t0=300.0),
                               bundle.trans_config)
    grid = grid.replace(u=grid.u + 10.0)
    tends = hs_tendencies(grid, bundle)
    return bundle, tends


def test_taper_zero_at_surface_one_in_troposphere():
    bundle, _ = _setup()
    mu = sppt.vertical_taper(bundle)
    assert mu.shape == (bundle.n_lev,)
    assert float(mu[0]) < 0.05           # surface level suppressed
    assert float(jnp.max(mu)) > 0.95     # full amplitude somewhere in troposphere


def test_zero_amplitude_leaves_tendencies_unchanged():
    bundle, tends = _setup()
    p = sppt.SPPTParams(log_sigma=jnp.log(1e-30), log_tau=jnp.log(6 * 3600.0),
                        log_len=jnp.log(500e3))
    r = sppt.pattern_to_grid(sppt.init_pattern(p, jax.random.PRNGKey(1),
                                               bundle.trans_config, bundle.dt), bundle.trans_config)
    mu = sppt.vertical_taper(bundle)
    out = sppt.apply_sppt(tends, r, mu, p)
    assert float(jnp.max(jnp.abs(out.virtual_temperature - tends.virtual_temperature))) < 1e-6


def test_apply_scales_uvt_only_and_clips():
    bundle, tends = _setup()
    p = sppt.SPPTParams(log_sigma=jnp.log(0.5), log_tau=jnp.log(6 * 3600.0),
                        log_len=jnp.log(500e3))
    r = sppt.pattern_to_grid(sppt.init_pattern(p, jax.random.PRNGKey(2),
                                               bundle.trans_config, bundle.dt), bundle.trans_config)
    mu = sppt.vertical_taper(bundle)
    out = sppt.apply_sppt(tends, r, mu, p)
    # factor bounded to [0.1, 1.9] -> perturbed tendency within 1.9x original
    ratio = jnp.abs(out.u) / (jnp.abs(tends.u) + 1e-30)
    assert float(jnp.max(ratio)) <= 1.9 + 1e-6
    # lnps/tracers untouched
    assert float(jnp.max(jnp.abs(out.log_surface_pressure))) == 0.0
