"""Resumable T21 sigma-only recovery sweep (Milestone M1).

Reproduces the EXPERIMENT_LOG's refined M1 run: T21 identical-twin Mode-A truth
(default params), recover sigma alone (tau, len frozen at truth) from a wrong
start sigma=0.2 toward the true sigma=0.5, via afCRPS + Adam.

Designed for this machine's constraints (EXPERIMENT_LOG, 2026-07-15):
  * float64 + CPU backend forced at the entry point;
  * runs in the FOREGROUND (background jobs get killed here);
  * per-step checkpoint to disk with a wall-clock budget, so a run that is
    cut short by the ~10-min tool window is fully resumable -- just invoke
    again and it continues from the last checkpoint.

Artifacts (under figures/_artifacts/):
  sweep_dataset_T21.pkl   the identical-twin truth dataset (generated once)
  sweep_ckpt.pkl          {step, params, opt_state, history}  (per-step)
"""
import os

os.environ["JAX_ENABLE_X64"] = "True"
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)

from examples.stochastic_sppt import sppt
from examples.stochastic_sppt.config import ExperimentConfig, ModelConfig, TrainConfig
from examples.stochastic_sppt.generate_data import load_dataset, mode_a_dataset, save_dataset
from examples.stochastic_sppt.model import build_model
from examples.stochastic_sppt.train import loss_fn

ART = Path(__file__).parent / "_artifacts"
# 1-day lead (48 T21 steps) keeps the reverse-mode compile + per-step small enough
# that the whole recovery fits inside this machine's ~10-min per-process ceiling
# (a 5-day-lead graph compiles for ~8.5 min alone). sigma-recovery is lead-agnostic
# -- it only matches ensemble spread to the identical-twin truth's spread -- so the
# short lead is a hardware adaptation, not a change to the M1 result. (EXPERIMENT_LOG)
DATASET = ART / "sweep_dataset_T21_lead1.pkl"
CKPT = ART / "sweep_ckpt.pkl"

# The refined M1 configuration from the EXPERIMENT_LOG (2026-07-15).
TRUE_SIGMA = 0.5
INIT_SIGMA = 0.2
LR = 0.03
N_MEMBERS = 4
BATCH_CASES = 3
N_CASES = 12
LEAD_DAYS = (1,)
SPINUP_DAYS = 120.0
TARGET_STEPS = 45


def ensure_dataset():
    if DATASET.exists():
        return load_dataset(DATASET)
    print(f"generating T21 identical-twin dataset -> {DATASET.name}")
    exp = ExperimentConfig(mode="A", forecast_resolution="T21", truth_resolution="T21",
                           n_members=N_MEMBERS, lead_days=LEAD_DAYS, n_cases=N_CASES,
                           spinup_days=SPINUP_DAYS, case_stride_days=5.0, seed=0)
    t0 = time.time()
    data = mode_a_dataset(exp, sppt.default_params())
    save_dataset(DATASET, data)
    print(f"  dataset ready ({time.time() - t0:.0f}s)")
    return data


def _sigma_only(grads):
    """Freeze tau and len: keep only the log_sigma gradient."""
    return grads.replace(log_tau=jnp.zeros_like(grads.log_tau),
                         log_len=jnp.zeros_like(grads.log_len))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-seconds", type=float, default=520.0,
                    help="wall-clock budget for this invocation, then checkpoint and exit")
    ap.add_argument("--target-steps", type=int, default=TARGET_STEPS)
    ap.add_argument("--data-only", action="store_true",
                    help="just build the identical-twin dataset artifact and exit")
    a = ap.parse_args(argv)
    ART.mkdir(exist_ok=True)

    data = ensure_dataset()
    if a.data_only:
        print("data-only: dataset ready, exiting before the sweep")
        return
    bundle = build_model(ModelConfig(resolution=data["forecast_resolution"],
                                     n_lev=int(data["n_lev"])))
    lead_steps = tuple(int(s) for s in data["lead_steps"])
    ic_specs, truth = data["ic_specs"], data["truth"]
    fields = ("u850", "v850", "t500", "vort500", "ps")
    scales = jnp.array([float(jnp.std(truth[k]) + 1e-12) for k in fields])
    n_cases = truth[fields[0]].shape[0]

    tc = TrainConfig(alpha=0.95, lr=LR, batch_cases=BATCH_CASES, seed=0)
    opt = optax.adam(tc.lr)
    base_key = jax.random.PRNGKey(tc.seed)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn), static_argnums=(5, 7, 9))

    # resume or cold-start
    if CKPT.exists():
        with open(CKPT, "rb") as f:
            ck = pickle.load(f)
        params, opt_state, history, start = (ck["params"], ck["opt_state"],
                                             ck["history"], ck["step"])
        print(f"resuming from step {start} (sigma={float(jnp.exp(params.log_sigma)):.3f})")
    else:
        params = sppt.SPPTParams(log_sigma=jnp.log(INIT_SIGMA),
                                 log_tau=jnp.log(6 * 3600.0), log_len=jnp.log(500e3))
        opt_state = opt.init(params)
        history, start = [], 0
        print(f"cold start at sigma={INIT_SIGMA}")

    t_start = time.time()
    step = start
    while step < a.target_steps:
        b = min(tc.batch_cases, n_cases)
        idx = jax.random.choice(jax.random.fold_in(base_key, step), n_cases, (b,), replace=False)
        eval_key = jax.random.fold_in(base_key, 10_000 + step)
        loss, grads = grad_fn(params, ic_specs, truth, idx, bundle, lead_steps,
                              eval_key, tc.alpha, scales, N_MEMBERS)
        updates, opt_state = opt.update(_sigma_only(grads), opt_state)
        params = optax.apply_updates(params, updates)
        step += 1
        history.append({"step": step, "loss": float(loss),
                        "sigma": float(jnp.exp(params.log_sigma)),
                        "tau_h": float(jnp.exp(params.log_tau)) / 3600.0,
                        "len_km": float(jnp.exp(params.log_len)) / 1e3})
        with open(CKPT, "wb") as f:
            pickle.dump({"step": step, "params": jax.device_get(params),
                         "opt_state": jax.device_get(opt_state), "history": history}, f)
        el = time.time() - t_start
        print(f"step {step:3d}  afCRPS {float(loss):.5f}  sigma {history[-1]['sigma']:.4f}"
              f"  [{el:.0f}s]", flush=True)
        if el > a.budget_seconds:
            print(f"budget reached ({el:.0f}s) -> checkpointed at step {step}; re-run to continue")
            return
    print(f"DONE: reached target {a.target_steps} steps; final sigma "
          f"{float(jnp.exp(params.log_sigma)):.4f}")


if __name__ == "__main__":
    main()
