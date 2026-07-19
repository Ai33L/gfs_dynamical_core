"""Example 2 -- an ensemble for free with jax.vmap.

A Fortran dycore runs one state per process; an ensemble means launching many
executables. Here the model is a pure function of its state, so jax.vmap
vectorises the *same compiled* integration over a batch of perturbed initial
conditions -- the ensemble is one array axis, run in a single call.

Run:  python -m docs.gfs_jax_manual.examples.ex02_ensemble_vmap
"""
import argparse
from functools import partial

import os

os.environ["JAX_ENABLE_X64"] = "True"        # float64 (before importing jax)
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # CPU: Metal can't do float64

import jax
import jax.numpy as jnp

from ._model import build_model, global_mean, rest_state, step
from gfs_dynamical_core.jax.transforms import spectral_to_grid


@partial(jax.jit, static_argnums=(2,))
def rollout(bundle, spec0, n_steps):
    spec_final = jax.lax.scan(lambda s, _: (step(bundle, s), None),
                              spec0, None, length=n_steps)[0]
    grid, _ = spectral_to_grid(spec_final, bundle.trans_config)
    return global_mean(jnp.mean(0.5 * (grid.u ** 2 + grid.v ** 2), axis=0),
                       bundle.gauss_weights)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--days", type=float, default=3.0)
    a = ap.parse_args(argv)

    bundle = build_model("T21")
    n_steps = int(a.days * 86400.0 / bundle.dt)

    # Build an ensemble of ICs: identical base state + tiny independent
    # temperature perturbations (a stack along a new leading axis).
    base = rest_state(bundle, jax.random.PRNGKey(0))
    keys = jax.random.split(jax.random.PRNGKey(1), a.members)

    def perturb(key):
        dT = 1e-2 * jax.random.normal(key, base.temperature.shape)
        # perturb in spectral space directly (a valid state perturbation)
        return base.replace(temperature=base.temperature + dT)

    ens0 = jax.vmap(perturb)(keys)                      # batched SpectralState

    # vmap the *compiled* rollout over the ensemble axis. bundle is broadcast.
    ke = jax.vmap(lambda s: rollout(bundle, s, n_steps))(ens0)
    ke = jax.device_get(ke)
    print(f"T21 Held-Suarez ensemble: {a.members} members, {a.days} days")
    print(f"  final global-mean KE per member: "
          f"{', '.join(f'{v:.3f}' for v in ke)}")
    print(f"  ensemble mean {ke.mean():.4f} J/kg,  spread (std) {ke.std():.2e} J/kg")
    print("  -> a tiny IC perturbation has begun to diverge (chaotic growth).")


if __name__ == "__main__":
    main()
