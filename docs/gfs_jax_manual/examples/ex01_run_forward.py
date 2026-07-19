"""Example 1 -- forward integration with jit + lax.scan.

The most basic capability: compile the whole time loop once with jax.jit and run
it as a single fused lax.scan (no Python-level loop). We integrate the
Held-Suarez model from an isothermal rest state and watch kinetic energy grow as
baroclinic instability spins up the midlatitude jet.

Run:  python -m docs.gfs_jax_manual.examples.ex01_run_forward
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


@partial(jax.jit, static_argnums=(2,))
def run(bundle, spec0, n_steps):
    """Integrate n_steps, returning the final state and a per-step KE trace."""
    def body(spec, _):
        spec = step(bundle, spec)
        grid, _g = spectral_to_grid(spec, bundle.trans_config)
        ke = global_mean(jnp.mean(0.5 * (grid.u ** 2 + grid.v ** 2), axis=0),
                         bundle.gauss_weights)
        return spec, ke
    spec_final, ke_trace = jax.lax.scan(body, spec0, None, length=n_steps)
    return spec_final, ke_trace


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=6.0)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)

    bundle = build_model("T21")
    spec0 = rest_state(bundle, jax.random.PRNGKey(0))
    n_steps = int(a.days * 86400.0 / bundle.dt)

    spec_final, ke = run(bundle, spec0, n_steps)
    ke = jax.device_get(ke)
    print(f"T21 Held-Suarez, {a.days} days ({n_steps} steps)")
    print(f"  global-mean KE: start {ke[0]:.4f} -> end {ke[-1]:.2f} J/kg")

    if not a.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = (jnp.arange(n_steps) + 1) * bundle.dt / 86400.0
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(t, ke, color="C0")
        ax.set_xlabel("time (days)")
        ax.set_ylabel("global-mean kinetic energy (J/kg)")
        ax.set_title("Held-Suarez spin-up from rest (T21, jit+lax.scan)")
        ax.grid(alpha=0.3)
        FIGDIR.mkdir(exist_ok=True)
        fig.tight_layout()
        fig.savefig(FIGDIR / "fig_spinup.png", dpi=130)
        print(f"  wrote {FIGDIR/'fig_spinup.png'}")


if __name__ == "__main__":
    main()
