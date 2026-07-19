"""Checkpointed, vmap-able ensemble rollout with SPPT-perturbed physics."""
import jax
import jax.numpy as jnp

from gfs_dynamical_core.jax.transforms import spectral_to_grid

from . import sppt
from .diagnostics import extract_fields
from .held_suarez import hs_tendencies
from .model import step

_FIELD_NAMES = ("u850", "v850", "t500", "vort500", "ps")


def member_rollout(params, spec0, key, bundle, lead_steps, mu):
    """Advance one ensemble member's spectral state and AR(1) SPPT pattern
    together, capturing diagnostics at each lead time in `lead_steps`.

    `params` is captured as a closure variable (not a module-global and not
    threaded through the scan carry) so it stays a normal differentiable
    argument under jax.grad/vmap/jit.
    """

    def body(carry, key_t):
        spec, r_lm = carry
        grid, _ = spectral_to_grid(spec, bundle.trans_config)
        phys = hs_tendencies(grid, bundle)
        r_grid = sppt.pattern_to_grid(r_lm, bundle.trans_config)
        phys = sppt.apply_sppt(phys, r_grid, mu, params)
        spec = step(bundle, spec, phys)
        r_lm = sppt.pattern_step(r_lm, key_t, params, bundle.trans_config, bundle.dt)
        return (spec, r_lm), None

    body = jax.checkpoint(body)

    r0 = sppt.init_pattern(params, key, bundle.trans_config, bundle.dt)
    n_total = int(lead_steps[-1])
    step_keys = jax.random.split(key, n_total)

    fields = {k: [] for k in _FIELD_NAMES}
    carry = (spec0, r0)
    prev = 0
    for lead in lead_steps:
        lead = int(lead)
        seg_keys = step_keys[prev:lead]
        carry, _ = jax.lax.scan(body, carry, seg_keys)
        f = extract_fields(carry[0], bundle)
        for k in fields:
            fields[k].append(f[k])
        prev = lead
    return {k: jnp.stack(v, axis=0) for k, v in fields.items()}  # (n_leads, lat, lon)


def ensemble_rollout(params, spec0, keys, bundle, lead_steps):
    """vmap member_rollout over `keys`; params and spec0 are shared/broadcast."""
    mu = sppt.vertical_taper(bundle)

    def fn(key):
        return member_rollout(params, spec0, key, bundle, lead_steps, mu)

    return jax.vmap(fn)(keys)  # (M, n_leads, lat, lon) per field
