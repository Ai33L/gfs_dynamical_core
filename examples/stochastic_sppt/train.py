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

from . import sppt
from .config import ModelConfig, TrainConfig
from .generate_data import load_dataset
from .metrics import afcrps
from .model import build_model
from .rollout import ensemble_rollout

_FIELDS = ("u850", "v850", "t500", "vort500", "ps")


def member_keys(base_key, case_idx, n_members):
    return jax.random.split(jax.random.fold_in(base_key, case_idx), n_members)


def loss_fn(params, ic_specs, truth, case_indices, bundle, lead_steps, base_key, alpha, scales,
            n_members):
    """afCRPS over a batch of cases, vmapped so the whole thing is jittable.

    `case_indices` is a traced jnp array of case indices (not a Python list),
    so this function has no data-dependent Python control flow and can be
    wrapped in a single `jax.jit` that never needs to retrace across steps.
    """
    batched_ics = jax.tree_util.tree_map(lambda a: a[case_indices], ic_specs)
    batched_keys = jax.vmap(lambda ci: member_keys(base_key, ci, n_members))(case_indices)

    def rollout_one(spec0, mkeys):
        return ensemble_rollout(params, spec0, mkeys, bundle, lead_steps)

    ens = jax.vmap(rollout_one)(batched_ics, batched_keys)  # each field: (batch, M, n_leads, lat, lon)

    total = 0.0
    for fi, k in enumerate(_FIELDS):
        batched_truth = truth[k][case_indices]  # (batch, n_leads, lat, lon)
        per_case = jax.vmap(lambda e, t: afcrps(e, t, alpha))(ens[k], batched_truth)  # (batch,)
        total = total + jnp.mean(per_case) / scales[fi]
    return total / len(_FIELDS)


def train(dataset, train_config: TrainConfig, init_params=None, n_members=4):
    bundle = build_model(ModelConfig(resolution=dataset["forecast_resolution"],
                                     n_lev=int(dataset["n_lev"])))
    lead_steps = tuple(int(s) for s in dataset["lead_steps"])
    ic_specs, truth = dataset["ic_specs"], dataset["truth"]
    # jnp array aligned to _FIELDS order (a traced constant, not a static dict --
    # dicts of Python floats aren't hashable for static_argnums).
    scales = jnp.array([float(jnp.std(truth[k]) + 1e-12) for k in _FIELDS])
    n_cases = truth[_FIELDS[0]].shape[0]

    params = init_params if init_params is not None else sppt.default_params()
    opt = optax.adam(train_config.lr)
    opt_state = opt.init(params)
    base_key = jax.random.PRNGKey(train_config.seed)

    # static_argnums: lead_steps (5), alpha (7), n_members (9) -- hashable and
    # constant across steps. `bundle` stays a normal traced arg: it's a flax
    # struct whose jnp-array leaves are traced and whose non-array fields
    # (trans_config.L, n_lev, dt, ...) are carried as static pytree aux_data,
    # so jit reuses one compiled graph across steps without retracing.
    grad_fn = jax.jit(jax.value_and_grad(loss_fn), static_argnums=(5, 7, 9))
    history = []
    for it in range(train_config.n_opt_steps):
        b = min(train_config.batch_cases, n_cases)
        idx = jax.random.choice(jax.random.fold_in(base_key, it), n_cases, (b,), replace=False)
        # common random numbers: derive from a per-iteration key (fixed within this grad eval)
        eval_key = jax.random.fold_in(base_key, 10_000 + it)
        loss, grads = grad_fn(params, ic_specs, truth, idx,
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
