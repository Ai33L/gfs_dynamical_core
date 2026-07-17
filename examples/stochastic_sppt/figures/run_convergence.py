"""Standalone, resumable T21 sigma-only recovery with K truth draws per IC.

Answers "what does sigma settle to?" by reducing the single-draw finite-sample
bias: Mode-A truth uses `--draws` stochastic trajectories per initial condition
(shape (n_cases, K, n_lead, lat, lon)), and the loss averages afCRPS over the K
draws that share one forecast ensemble. Sigma-only (tau, len frozen at truth).

Built for this machine (EXPERIMENT_LOG): float64 + CPU forced at the entry
point; runs in the FOREGROUND; per-step checkpoint + wall-clock --budget-seconds
so a cut-short run resumes on re-invocation.

Run (foreground):
  python -m examples.stochastic_sppt.figures.run_convergence --data-only   # build dataset once
  python -m examples.stochastic_sppt.figures.run_convergence               # then train, resumably

Artifacts (under figures/_artifacts/):
  convergence_dataset_T21.pkl   the K-draw identical-twin truth dataset
  convergence_ckpt.pkl          {step, params, opt_state, history}  (per-step)
"""
import os

os.environ["JAX_ENABLE_X64"] = "True"
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # setdefault so a cluster's JAX_PLATFORMS=cuda still wins

import argparse
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)

from examples.stochastic_sppt import sppt
from examples.stochastic_sppt.config import ExperimentConfig, ModelConfig
from examples.stochastic_sppt.generate_data import (load_dataset, mode_a_dataset_multidraw,
                                                    save_dataset)
from examples.stochastic_sppt.metrics import afcrps_multidraw
from examples.stochastic_sppt.model import build_model
from examples.stochastic_sppt.rollout import ensemble_rollout

_FIELDS = ("u850", "v850", "t500", "vort500", "ps")

ART = Path(__file__).parent / "_artifacts"
DATASET = ART / "convergence_dataset_T21.pkl"
CKPT = ART / "convergence_ckpt.pkl"

TRUE_SIGMA = 0.5


def member_keys(base_key, case_idx, n_members):
    return jax.random.split(jax.random.fold_in(base_key, case_idx), n_members)


def multidraw_loss_fn(params, ic_specs, truth, case_indices, bundle, lead_steps,
                      base_key, alpha, scales, n_members):
    """afCRPS over a batch of cases, averaged over K truth draws per case.

    truth[field]: (n_cases, K, n_lead, lat, lon). One forecast ensemble per case
    is scored against all K draws via afcrps_multidraw. Vmapped over the case
    batch so the whole loss is jittable with no data-dependent Python control flow.
    """
    batched_ics = jax.tree_util.tree_map(lambda a: a[case_indices], ic_specs)
    batched_keys = jax.vmap(lambda ci: member_keys(base_key, ci, n_members))(case_indices)

    def rollout_one(spec0, mkeys):
        return ensemble_rollout(params, spec0, mkeys, bundle, lead_steps)

    ens = jax.vmap(rollout_one)(batched_ics, batched_keys)  # field: (batch, M, n_lead, lat, lon)

    total = 0.0
    for fi, k in enumerate(_FIELDS):
        batched_truth = truth[k][case_indices]  # (batch, K, n_lead, lat, lon)
        per_case = jax.vmap(lambda e, t: afcrps_multidraw(e, t, alpha))(ens[k], batched_truth)
        total = total + jnp.mean(per_case) / scales[fi]
    return total / len(_FIELDS)


def _sigma_only(grads):
    """Freeze tau and len: keep only the log_sigma gradient."""
    return grads.replace(log_tau=jnp.zeros_like(grads.log_tau),
                         log_len=jnp.zeros_like(grads.log_len))


def ensure_dataset(a):
    if DATASET.exists():
        return load_dataset(DATASET)
    print(f"generating T21 K-draw dataset ({a.cases} cases x {a.draws} draws) -> {DATASET.name}")
    exp = ExperimentConfig(mode="A", forecast_resolution="T21", truth_resolution="T21",
                           n_members=a.members, lead_days=(a.lead_days,), n_cases=a.cases,
                           spinup_days=a.spinup_days, case_stride_days=5.0,
                           n_draws=a.draws, seed=0)
    t0 = time.time()
    data = mode_a_dataset_multidraw(exp, sppt.default_params())
    save_dataset(DATASET, data)
    print(f"  dataset ready ({time.time() - t0:.0f}s)")
    return data


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=32)
    ap.add_argument("--draws", type=int, default=32)
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--target-steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--init-sigma", type=float, default=0.2)
    ap.add_argument("--lead-days", type=int, default=1)
    ap.add_argument("--spinup-days", type=float, default=120.0)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--budget-seconds", type=float, default=520.0)
    ap.add_argument("--data-only", action="store_true")
    a = ap.parse_args(argv)
    ART.mkdir(exist_ok=True)

    data = ensure_dataset(a)
    if a.data_only:
        print("data-only: dataset ready, exiting before training")
        return

    bundle = build_model(ModelConfig(resolution=data["forecast_resolution"],
                                     n_lev=int(data["n_lev"])))
    lead_steps = tuple(int(s) for s in data["lead_steps"])
    ic_specs, truth = data["ic_specs"], data["truth"]
    scales = jnp.array([float(jnp.std(truth[k]) + 1e-12) for k in _FIELDS])
    n_cases = truth[_FIELDS[0]].shape[0]

    opt = optax.adam(a.lr)
    base_key = jax.random.PRNGKey(0)
    grad_fn = jax.jit(jax.value_and_grad(multidraw_loss_fn), static_argnums=(5, 7, 9))

    if CKPT.exists():
        with open(CKPT, "rb") as f:
            ck = pickle.load(f)
        params, opt_state, history, start = (ck["params"], ck["opt_state"],
                                             ck["history"], ck["step"])
        print(f"resuming from step {start} (sigma={float(jnp.exp(params.log_sigma)):.4f})")
    else:
        params = sppt.SPPTParams(log_sigma=jnp.log(a.init_sigma),
                                 log_tau=jnp.log(6 * 3600.0), log_len=jnp.log(500e3))
        opt_state = opt.init(params)
        history, start = [], 0
        print(f"cold start at sigma={a.init_sigma} (true sigma={TRUE_SIGMA})")

    t_start = time.time()
    step = start
    while step < a.target_steps:
        b = min(a.batch, n_cases)
        idx = jax.random.choice(jax.random.fold_in(base_key, step), n_cases, (b,), replace=False)
        eval_key = jax.random.fold_in(base_key, 10_000 + step)
        loss, grads = grad_fn(params, ic_specs, truth, idx, bundle, lead_steps,
                              eval_key, a.alpha, scales, a.members)
        updates, opt_state = opt.update(_sigma_only(grads), opt_state)
        params = optax.apply_updates(params, updates)
        step += 1
        history.append({"step": step, "loss": float(loss),
                        "sigma": float(jnp.exp(params.log_sigma))})
        with open(CKPT, "wb") as f:
            pickle.dump({"step": step, "params": jax.device_get(params),
                         "opt_state": jax.device_get(opt_state), "history": history}, f)
        el = time.time() - t_start
        print(f"step {step:3d}  afCRPS {float(loss):.5f}  sigma {history[-1]['sigma']:.4f}"
              f"  [{el:.0f}s]", flush=True)
        if el > a.budget_seconds:
            print(f"budget reached ({el:.0f}s) -> checkpointed at step {step}; re-run to continue")
            return

    sig = [h["sigma"] for h in history]
    last10 = sig[-10:] if len(sig) >= 10 else sig
    print(f"DONE: {a.target_steps} steps.  final sigma {sig[-1]:.4f}  "
          f"last-{len(last10)} mean {sum(last10) / len(last10):.4f}  (true {TRUE_SIGMA})")


if __name__ == "__main__":
    main()
