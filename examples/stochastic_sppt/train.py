"""Train SPPT params by minimizing almost-fair CRPS over identical-twin cases."""
import os

os.environ["JAX_ENABLE_X64"] = "True"
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # setdefault so a cluster's JAX_PLATFORMS=cuda still wins

import argparse
import pickle

import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)

from gfs_dynamical_core.jax.states import SpectralState

from . import sppt
from .config import ModelConfig, TrainConfig
from .generate_data import load_dataset
from .metrics import afcrps
from .model import build_model
from .rollout import ensemble_rollout

_FIELDS = ("u850", "v850", "t500", "vort500", "ps")


def member_keys(base_key, case_idx, n_members):
    return jax.random.split(jax.random.fold_in(base_key, case_idx), n_members)


def _case_ic(ic_specs, i):
    return SpectralState(
        vorticity=ic_specs.vorticity[i], divergence=ic_specs.divergence[i],
        temperature=ic_specs.temperature[i], log_surface_pressure=ic_specs.log_surface_pressure[i],
        tracers=ic_specs.tracers[i],
    )


def loss_fn(params, ic_specs, truth, case_indices, bundle, lead_steps, base_key, alpha, scales,
            n_members):
    total = 0.0
    for i in case_indices:
        spec0 = _case_ic(ic_specs, int(i))
        keys = member_keys(base_key, int(i), n_members)
        ens = ensemble_rollout(params, spec0, keys, bundle, lead_steps)
        for k in _FIELDS:
            total = total + afcrps(ens[k], truth[k][int(i)], alpha) / scales[k]
    return total / (len(case_indices) * len(_FIELDS))


def train(dataset, train_config: TrainConfig, init_params=None, n_members=4):
    bundle = build_model(ModelConfig(resolution=dataset["forecast_resolution"],
                                     n_lev=int(dataset["n_lev"])))
    lead_steps = tuple(int(s) for s in dataset["lead_steps"])
    ic_specs, truth = dataset["ic_specs"], dataset["truth"]
    scales = {k: float(jnp.std(truth[k]) + 1e-12) for k in _FIELDS}
    n_cases = truth[_FIELDS[0]].shape[0]

    params = init_params if init_params is not None else sppt.default_params()
    opt = optax.adam(train_config.lr)
    opt_state = opt.init(params)
    base_key = jax.random.PRNGKey(train_config.seed)

    grad_fn = jax.value_and_grad(loss_fn)
    history = []
    for it in range(train_config.n_opt_steps):
        b = min(train_config.batch_cases, n_cases)
        idx = jax.random.choice(jax.random.fold_in(base_key, it), n_cases, (b,), replace=False)
        # common random numbers: derive from a per-iteration key (fixed within this grad eval)
        eval_key = jax.random.fold_in(base_key, 10_000 + it)
        loss, grads = grad_fn(params, ic_specs, truth, [int(x) for x in idx],
                              bundle, lead_steps, eval_key, train_config.alpha, scales,
                              n_members)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        history.append(float(loss))
        if it % train_config.log_every == 0:
            print(f"step {it:4d}  afCRPS {float(loss):.5f}  "
                  f"sigma {float(jnp.exp(params.log_sigma)):.3f} "
                  f"tau {float(jnp.exp(params.log_tau)) / 3600:.2f}h "
                  f"len {float(jnp.exp(params.log_len)) / 1e3:.0f}km")
    return params, history


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--out", default="sppt_params.pkl")
    a = ap.parse_args(argv)
    data = load_dataset(a.dataset)
    tc = TrainConfig(alpha=a.alpha, lr=a.lr, n_opt_steps=a.steps)
    params, history = train(data, tc, n_members=a.members)
    with open(a.out, "wb") as f:
        pickle.dump({"params": jax.device_get(params), "history": history}, f)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
