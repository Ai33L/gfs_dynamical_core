"""M1 supplementary check: does the recovered sigma give a CALIBRATED ensemble?

Forward-only ensembles (no gradient -> no reverse-mode memory blow-up) at the
recovered sigma on a FRESH T21 identical-twin Mode-A dataset (different seed
from the sweep). Reproduces the EXPERIMENT_LOG calibration diagnostics
(2026-07-15): spread-error ratio ~1, flat rank histogram, truth-looks-like-a-
member anomaly overlay, and afCRPS-vs-lead (normalized per field -- the fix to
the log's unnormalized-panel caveat).

Artifacts: figures/_artifacts/calib_results.pkl
"""
import os

os.environ["JAX_ENABLE_X64"] = "True"
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import pickle
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from examples.stochastic_sppt import sppt
from examples.stochastic_sppt.config import ExperimentConfig, ModelConfig
from examples.stochastic_sppt.generate_data import load_dataset, mode_a_dataset, save_dataset
from examples.stochastic_sppt.metrics import (
    afcrps, rank_histogram, rmse_of_mean, spread, spread_error_ratio,
)
from examples.stochastic_sppt.model import build_model
from examples.stochastic_sppt.rollout import ensemble_rollout
from examples.stochastic_sppt.train import member_keys

ART = Path(__file__).parent / "_artifacts"
SWEEP_CKPT = ART / "sweep_ckpt.pkl"
DATASET = ART / "calib_dataset_T21.pkl"
OUT = ART / "calib_results.pkl"

FIELDS = ("u850", "v850", "t500", "vort500", "ps")
LEAD_DAYS = (3, 5)
SPINUP_DAYS = 120.0
N_CASES = 3
N_MEMBERS = 8
FALLBACK_SIGMA = 0.517  # the recovered value, if no sweep ckpt is present


def recovered_sigma():
    if SWEEP_CKPT.exists():
        with open(SWEEP_CKPT, "rb") as f:
            p = pickle.load(f)["params"]
        return float(jnp.exp(p.log_sigma))
    return FALLBACK_SIGMA


def ensure_dataset():
    if DATASET.exists():
        return load_dataset(DATASET)
    # fresh identical-twin dataset (seed=1 -> different draws than the sweep)
    exp = ExperimentConfig(mode="A", forecast_resolution="T21", truth_resolution="T21",
                           n_members=N_MEMBERS, lead_days=LEAD_DAYS, n_cases=N_CASES,
                           spinup_days=SPINUP_DAYS, case_stride_days=5.0, seed=1)
    data = mode_a_dataset(exp, sppt.default_params())
    save_dataset(DATASET, data)
    return data


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=None,
                    help="override the sigma used for the forward ensembles")
    ap.add_argument("--data-only", action="store_true",
                    help="just build the fresh calibration dataset artifact and exit")
    a = ap.parse_args(argv)
    ART.mkdir(exist_ok=True)

    data = ensure_dataset()
    if a.data_only:
        print("data-only: calibration dataset ready, exiting")
        return

    sigma = a.sigma if a.sigma is not None else recovered_sigma()
    print(f"forward calibration ensembles at sigma={sigma:.4f}")
    bundle = build_model(ModelConfig(resolution="T21", n_lev=int(data["n_lev"])))
    lead_steps = tuple(int(s) for s in data["lead_steps"])
    ic_specs, truth = data["ic_specs"], data["truth"]

    params = sppt.SPPTParams(log_sigma=jnp.log(sigma),
                             log_tau=jnp.log(6 * 3600.0), log_len=jnp.log(500e3))
    roll = jax.jit(partial(ensemble_rollout, lead_steps=lead_steps))
    base_key = jax.random.PRNGKey(123)

    # ensembles per case -> stack to (M, n_cases, n_leads, lat, lon)
    ens = {k: [] for k in FIELDS}
    for ci in range(N_CASES):
        spec0 = jax.tree_util.tree_map(lambda x: x[ci], ic_specs)
        mkeys = member_keys(base_key, ci, N_MEMBERS)
        e = roll(params, spec0, mkeys, bundle)
        for k in FIELDS:
            ens[k].append(e[k])
        print(f"  case {ci} rolled out", flush=True)
    ens = {k: jnp.stack(v, axis=1) for k, v in ens.items()}   # (M, n_cases, n_leads, lat, lon)

    n_leads = len(lead_steps)
    results = {"sigma": sigma, "lead_days": LEAD_DAYS, "fields": FIELDS,
               "n_members": N_MEMBERS, "n_cases": N_CASES,
               "spread_error": {}, "afcrps": {}, "spread": {}, "rmse": {}}
    for k in FIELDS:
        se, af, sp, rm = [], [], [], []
        for li in range(n_leads):
            e = ens[k][:, :, li]              # (M, n_cases, lat, lon)
            t = truth[k][:, li]               # (n_cases, lat, lon)
            std = float(jnp.std(t) + 1e-12)
            se.append(float(spread_error_ratio(e, t)))
            af.append(float(afcrps(e, t, 0.95)) / std)   # normalized (log caveat fix)
            sp.append(float(spread(e)))
            rm.append(float(rmse_of_mean(e, t)))
        results["spread_error"][k] = se
        results["afcrps"][k] = af
        results["spread"][k] = sp
        results["rmse"][k] = rm

    # rank histogram + anomaly overlay for t500 at the 5-day lead (last lead)
    li = n_leads - 1
    e = ens["t500"][:, :, li]                 # (M, n_cases, lat, lon)
    t = truth["t500"][:, li]                  # (n_cases, lat, lon)
    results["rank_hist"] = {"field": "t500", "lead_day": LEAD_DAYS[li],
                            "counts": [int(c) for c in rank_histogram(e, t)]}
    emean = jnp.mean(e, axis=0)
    results["anom"] = {"field": "t500", "lead_day": LEAD_DAYS[li],
                       "member": (e - emean[None]).ravel().tolist(),
                       "truth": (t - emean).ravel().tolist()}

    with open(OUT, "wb") as f:
        pickle.dump(results, f)
    print(f"wrote {OUT.name}")
    for k in FIELDS:
        print(f"  {k:8s} spread-error {results['spread_error'][k]}")


if __name__ == "__main__":
    main()
