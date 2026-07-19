"""Example 5 -- optimising an initial condition (a 4D-Var flavour).

Because the forecast is differentiable end-to-end, we can *fit* the initial state
so that a forecast matches a target -- the essence of variational data
assimilation (4D-Var). Here a toy: adjust a few amplitudes of initial-temperature
perturbation patterns so that the global-mean kinetic energy after a short
forecast hits a prescribed target, by gradient descent on the squared miss.

Run:  python -m docs.gfs_jax_manual.examples.ex05_optimize_ic
"""
import argparse
from functools import partial
from pathlib import Path

import os

os.environ["JAX_ENABLE_X64"] = "True"        # float64 (before importing jax)
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # CPU: Metal can't do float64

import jax
import jax.numpy as jnp

from ._model import build_model, global_mean, rest_state, step
from gfs_dynamical_core.jax.transforms import spectral_to_grid

FIGDIR = Path(__file__).resolve().parents[1] / "figures"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)

    bundle = build_model("T21")
    spec0 = rest_state(bundle, jax.random.PRNGKey(0))
    spec0 = jax.lax.scan(lambda s, _: (step(bundle, s), None),
                         spec0, None, length=int(2 * 86400 / bundle.dt))[0]
    n_steps = int(a.hours * 3600.0 / bundle.dt)

    pats = jax.vmap(lambda key: 1e-3 * jax.random.normal(key, spec0.temperature.shape))(
        jax.random.split(jax.random.PRNGKey(7), a.k))

    @jax.jit
    def ke_of(theta):
        dT = jnp.tensordot(theta, pats, axes=(0, 0))
        spec = spec0.replace(temperature=spec0.temperature + dT)
        spec = jax.lax.scan(lambda s, _: (step(bundle, s), None),
                            spec, None, length=n_steps)[0]
        grid, _ = spectral_to_grid(spec, bundle.trans_config)
        return global_mean(jnp.mean(0.5 * (grid.u ** 2 + grid.v ** 2), axis=0),
                           bundle.gauss_weights)

    ke0 = float(ke_of(jnp.zeros(a.k)))
    target = 1.10 * ke0                                  # ask for 10% more KE
    print(f"unperturbed forecast KE = {ke0:.4f};  target = {target:.4f} J/kg")

    loss_and_grad = jax.jit(jax.value_and_grad(
        lambda theta: (ke_of(theta) - target) ** 2))

    theta = jnp.zeros(a.k)
    losses = []
    for it in range(a.steps):
        loss, g = loss_and_grad(theta)
        theta = theta - a.lr * g / (jnp.linalg.norm(g) + 1e-12)   # normalised GD
        losses.append(float(loss))
        if it % 5 == 0 or it == a.steps - 1:
            print(f"  step {it:3d}  loss {float(loss):.3e}  KE {float(ke_of(theta)):.4f}")
    print(f"final forecast KE = {float(ke_of(theta)):.4f}  (target {target:.4f})")

    if not a.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.semilogy(losses, "-o", ms=3, color="C3")
        ax.set_xlabel("gradient-descent step")
        ax.set_ylabel("(forecast KE - target)$^2$")
        ax.set_title("Fitting an initial condition to a forecast target (T21)")
        ax.grid(alpha=0.3)
        FIGDIR.mkdir(exist_ok=True)
        fig.tight_layout()
        fig.savefig(FIGDIR / "fig_optimize.png", dpi=130)
        print(f"  wrote {FIGDIR/'fig_optimize.png'}")


if __name__ == "__main__":
    main()
