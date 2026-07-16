"""Example 3 -- adjoint sensitivity for free with jax.grad.

The single capability a Fortran dycore cannot give you without hand-writing (and
maintaining) a separate adjoint model: the gradient of a forecast metric with
respect to the initial state. Here it is one call to jax.grad.

We define a scalar forecast metric J = global-mean kinetic energy after a few
hours, as a function of the *initial temperature field* T0(level, lat, lon), and
compute dJ/dT0 -- the map of where a perturbation to the initial temperature most
increases (or decreases) later kinetic energy. This is exactly what an adjoint
model computes; reverse-mode autodiff produces it automatically and at the cost
of ~one forward integration.

Run:  python -m docs.gfs_jax_manual.examples.ex03_gradient_sensitivity
"""
import argparse
from functools import partial
from pathlib import Path

import os

os.environ["JAX_ENABLE_X64"] = "True"        # float64 (before importing jax)
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # CPU: Metal can't do float64

import jax
import jax.numpy as jnp

from ._model import build_model, global_mean, rest_state
from gfs_dynamical_core.jax.states import GridState
from gfs_dynamical_core.jax.transforms import grid_to_spectral, spectral_to_grid
from ._model import step

FIGDIR = Path(__file__).resolve().parents[1] / "figures"


def make_metric(bundle, base_grid, n_steps):
    """Return J(T0) = global-mean KE after n_steps, as a function of the initial
    grid temperature field only (all other fields held at the base state)."""
    @partial(jax.jit, static_argnums=())
    def metric(T0):
        grid0 = base_grid.replace(temperature=T0)
        spec = grid_to_spectral(grid0, bundle.trans_config)
        spec = jax.lax.scan(lambda s, _: (step(bundle, s), None),
                            spec, None, length=n_steps)[0]
        grid, _ = spectral_to_grid(spec, bundle.trans_config)
        return global_mean(jnp.mean(0.5 * (grid.u ** 2 + grid.v ** 2), axis=0),
                           bundle.gauss_weights)
    return metric


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)

    bundle = build_model("T21")
    # A developed base flow (spin up a little so KE is sensitive to T0).
    spec0 = rest_state(bundle, jax.random.PRNGKey(0))
    spec0 = jax.lax.scan(lambda s, _: (step(bundle, s), None),
                         spec0, None, length=int(3 * 86400 / bundle.dt))[0]
    base_grid, _ = spectral_to_grid(spec0, bundle.trans_config)
    base_grid = base_grid.replace(temperature=base_grid.temperature.real,
                                  log_surface_pressure=base_grid.log_surface_pressure.real,
                                  vorticity=base_grid.vorticity.real,
                                  divergence=base_grid.divergence.real)

    n_steps = int(a.hours * 3600.0 / bundle.dt)
    metric = make_metric(bundle, base_grid, n_steps)

    T0 = base_grid.temperature
    J, dJ_dT0 = jax.value_and_grad(metric)(T0)      # <-- the adjoint, for free
    dJ_dT0 = jax.device_get(dJ_dT0)
    print(f"T21 Held-Suarez, {a.hours} h forecast")
    print(f"  metric J (global-mean KE)         = {float(J):.4f} J/kg")
    print(f"  ||dJ/dT0||  (sensitivity norm)    = {float(jnp.linalg.norm(dJ_dT0)):.3e}")
    kmax = int(jnp.argmax(jnp.abs(dJ_dT0)) // (dJ_dT0.shape[1] * dJ_dT0.shape[2]))
    print(f"  most sensitive model level (0=sfc) = {kmax}")
    print("  Interpretation: dJ/dT0[k,j,i] = how much a 1 K bump to the initial")
    print("  temperature at that point changes the later kinetic energy.")

    if not a.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        smap = dJ_dT0[kmax]
        vmax = float(jnp.max(jnp.abs(smap)))
        fig, ax = plt.subplots(figsize=(7, 3.6))
        im = ax.imshow(smap, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
                       extent=[0, 360, -90, 90])
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        ax.set_title(f"dJ/dT0 at level {kmax} (adjoint sensitivity, {a.hours:.0f} h)")
        fig.colorbar(im, ax=ax, label="J/kg per K")
        FIGDIR.mkdir(exist_ok=True)
        fig.tight_layout()
        fig.savefig(FIGDIR / "fig_sensitivity.png", dpi=130)
        print(f"  wrote {FIGDIR/'fig_sensitivity.png'}")


if __name__ == "__main__":
    main()
