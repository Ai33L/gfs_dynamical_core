"""Example 6 -- optimal perturbations (singular vectors) with jvp + vjp.

The tangent-linear model (how a small perturbation grows) and its adjoint are the
building blocks of predictability studies. Autodiff gives both directly:

  * jax.jvp(g, (x,), (dx,))  applies the tangent-linear operator M = dg/dx  (M dx);
  * jax.vjp(g, x)            returns a function applying the adjoint M^T   (M^T dy).

The leading *singular vector* -- the initial perturbation that grows fastest over
the window -- is the top eigenvector of M^T M, found by power iteration using only
jvp and vjp. Its singular value is the optimal amplification factor. No adjoint
model is written by hand; it is generated from the forward code.

Here g maps an initial temperature perturbation to the final (u, v) wind field.

Run:  python -m docs.gfs_jax_manual.examples.ex06_singular_vector
"""
import argparse
from functools import partial

import os

os.environ["JAX_ENABLE_X64"] = "True"        # float64 (before importing jax)
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # CPU: Metal can't do float64

import jax
import jax.numpy as jnp

from ._model import build_model, rest_state, step
from gfs_dynamical_core.jax.transforms import grid_to_spectral, spectral_to_grid


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--iters", type=int, default=6)
    a = ap.parse_args(argv)

    bundle = build_model("T21")
    spec0 = rest_state(bundle, jax.random.PRNGKey(0))
    spec0 = jax.lax.scan(lambda s, _: (step(bundle, s), None),
                         spec0, None, length=int(3 * 86400 / bundle.dt))[0]
    base_grid, _ = spectral_to_grid(spec0, bundle.trans_config)
    T_base = base_grid.temperature.real
    n_steps = int(a.hours * 3600.0 / bundle.dt)

    @jax.jit
    def g(T0):
        """initial temperature field -> final (u, v), a real vector map."""
        grid0 = base_grid.replace(temperature=T0)
        spec = grid_to_spectral(grid0, bundle.trans_config)
        spec = jax.lax.scan(lambda s, _: (step(bundle, s), None),
                            spec, None, length=n_steps)[0]
        grid, _ = spectral_to_grid(spec, bundle.trans_config)
        return jnp.stack([grid.u.real, grid.v.real])

    # Tangent-linear (M) via jvp, adjoint (M^T) via vjp, both at the base state.
    _, vjp_fn = jax.vjp(g, T_base)
    apply_M = jax.jit(lambda dx: jax.jvp(g, (T_base,), (dx,))[1])
    apply_MT = jax.jit(lambda dy: vjp_fn(dy)[0])

    def norm(z):
        return jnp.sqrt(jnp.vdot(z, z).real)

    # Power iteration on M^T M: dx <- M^T (M dx), renormalise; growth = ||M dx||.
    dx = jax.random.normal(jax.random.PRNGKey(1), T_base.shape)
    dx = dx / norm(dx)
    print(f"T21 optimal-perturbation growth over {a.hours} h (leading singular value):")
    for it in range(a.iters):
        Mdx = apply_M(dx)
        growth = float(norm(Mdx))                 # ||M dx|| with ||dx|| = 1
        dx = apply_MT(Mdx)
        dx = dx / norm(dx)
        print(f"  iter {it}:  amplification ||M dx|| / ||dx||  = {growth:.3f}")
    print("  (converges upward to the optimal growth factor; the final dx is the")
    print("   fastest-growing initial temperature perturbation for this window.)")
    print("  NB: growth depends on the norm; a physical study uses a total-energy")
    print("  norm rather than the plain L2 used here for illustration.")


if __name__ == "__main__":
    main()
