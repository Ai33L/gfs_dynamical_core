"""Example 4 -- Jacobians with jax.jacfwd / jax.jacrev.

Autodiff gives full Jacobians, not just gradients of scalars. Here we build a
small vector-to-vector map f: R^k -> R^m and compute its Jacobian two ways:

  * jax.jacrev  (reverse mode)  -- efficient when outputs m < inputs k;
  * jax.jacfwd  (forward mode)  -- efficient when inputs k < outputs m.

Both give the identical matrix (to rounding). The map: k scalar amplitudes drive
k fixed temperature-perturbation patterns on the initial condition; the m outputs
are physical diagnostics after a short forecast. df_j/dtheta_i is the linearised
response of diagnostic j to perturbation i -- a tangent-linear model, restricted
to a chosen input/output subspace.

Run:  python -m docs.gfs_jax_manual.examples.ex04_jacobian
"""
import argparse
from functools import partial

import os

os.environ["JAX_ENABLE_X64"] = "True"        # float64 (before importing jax)
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # CPU: Metal can't do float64

import jax
import jax.numpy as jnp

from ._model import build_model, global_mean, rest_state, step
from gfs_dynamical_core.jax.dynamics import compute_pressure_diagnostics
from gfs_dynamical_core.jax.transforms import spectral_to_grid


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4, help="number of input knobs")
    ap.add_argument("--hours", type=float, default=6.0)
    a = ap.parse_args(argv)

    bundle = build_model("T21")
    spec0 = rest_state(bundle, jax.random.PRNGKey(0))
    spec0 = jax.lax.scan(lambda s, _: (step(bundle, s), None),
                         spec0, None, length=int(2 * 86400 / bundle.dt))[0]

    # k fixed perturbation patterns on the spectral temperature field.
    kk = a.k
    pat_keys = jax.random.split(jax.random.PRNGKey(7), kk)
    patterns = jax.vmap(lambda key: 1e-3 * jax.random.normal(
        key, spec0.temperature.shape))(pat_keys)          # (k, n_lev, L, 2L-1)
    n_steps = int(a.hours * 3600.0 / bundle.dt)

    @jax.jit
    def f(theta):                                          # R^k -> R^m
        dT = jnp.tensordot(theta, patterns, axes=(0, 0))
        spec = spec0.replace(temperature=spec0.temperature + dT)
        spec = jax.lax.scan(lambda s, _: (step(bundle, s), None),
                            spec, None, length=n_steps)[0]
        grid, _ = spectral_to_grid(spec, bundle.trans_config)
        ke = global_mean(jnp.mean(0.5 * (grid.u ** 2 + grid.v ** 2), axis=0),
                         bundle.gauss_weights)
        press = compute_pressure_diagnostics(grid.log_surface_pressure.real,
                                             bundle.dyn_config)
        ps = global_mean(press.ps, bundle.gauss_weights)
        tmid = global_mean(grid.temperature.real[bundle.n_lev // 2],
                           bundle.gauss_weights)
        return jnp.array([ke, ps, tmid])                  # m = 3 diagnostics

    theta0 = jnp.zeros(kk)
    Jr = jax.jacrev(f)(theta0)
    Jf = jax.jacfwd(f)(theta0)
    Jr, Jf = jax.device_get(Jr), jax.device_get(Jf)

    print(f"Jacobian df/dtheta  (m=3 diagnostics x k={kk} knobs), {a.hours} h forecast")
    labels = ["d(KE)  ", "d(ps)  ", "d(Tmid)"]
    for name, row in zip(labels, Jr):
        print(f"  {name} : " + "  ".join(f"{v:+.3e}" for v in row))
    print(f"  max|jacrev - jacfwd| = {float(jnp.max(jnp.abs(Jr - Jf))):.2e}"
          "  (reverse and forward mode agree)")
    print("  Rows are diagnostics, columns are the k perturbation patterns;")
    print("  reverse mode is cheaper here (m<k), forward mode when k<m.")


if __name__ == "__main__":
    main()
