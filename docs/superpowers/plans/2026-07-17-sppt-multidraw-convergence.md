# K-draw Mode-A Dataset + Longer σ-Convergence Run — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a Mode-A truth dataset with K stochastic trajectories per initial condition and run a longer σ-only recovery, so σ's settled value can be observed with the finite-sample bias reduced.

**Architecture:** Additive changes to the trainable-SPPT pipeline. A new `mode_a_dataset_multidraw` reuses the existing `ensemble_rollout` to draw K truth trajectories per IC (truth shape `(n_cases, K, n_lead, lat, lon)`). A new `afcrps_multidraw` metric averages almost-fair CRPS over the K draws sharing one forecast ensemble. A new standalone, resumable `figures/run_convergence.py` drives the longer σ-only training and reports the settled σ. The existing `mode_a_dataset`, `train.py`, and `figures/run_sweep.py` are untouched.

**Tech Stack:** Python, JAX (float64, CPU backend forced at entry), optax (Adam), pytest. Spectral Held–Suarez dynamical core from `gfs_dynamical_core.jax`.

## Global Constraints

- **float64 mandatory.** Every runnable entry point sets `os.environ["JAX_ENABLE_X64"] = "True"` and `os.environ.setdefault("JAX_PLATFORMS", "cpu")` **before** importing jax, then `jax.config.update("jax_enable_x64", True)` after. float32 silently corrupts the spectral dynamics; the Metal backend cannot do float64.
- **Foreground, resumable, background-killed.** Long jobs run in the foreground on this machine (background jobs get killed even when memory is healthy). Every runnable long job checkpoints per step to disk and honours a wall-clock `--budget-seconds`, so an interrupted run resumes on re-invocation.
- **Verification fields (fixed order):** `("u850", "v850", "t500", "vort500", "ps")`.
- **Test runner (absolute paths, no conda activate):** `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest`; interpreter `/Users/joymonteiro/miniconda3/envs/climt/bin/python`.
- **Tests are a package:** import as `examples.stochastic_sppt.tests...`; run pytest from repo root `/Users/joymonteiro/github/gfs_dynamical_core`.
- **Additive only:** do not modify `mode_a_dataset`, `train.py`, or `figures/run_sweep.py`. Existing 21-test suite must stay green.
- **σ-only training:** τ and ℓ gradients are zeroed (frozen at truth) in the convergence run.
- **T21 grid dims:** at T21 each verification field is `(32, 63)` (`lat=L=32`, `lon=2L-1=63`).

---

### Task 1: Add `n_draws` to `ExperimentConfig`

**Files:**
- Modify: `examples/stochastic_sppt/config.py` (the `ExperimentConfig` dataclass, around line 32-43)
- Test: `examples/stochastic_sppt/tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ExperimentConfig(..., n_draws: int = 1)` — new field consumed by Task 2's `mode_a_dataset_multidraw` and Task 4's script.

- [ ] **Step 1: Write the failing test**

Add to `examples/stochastic_sppt/tests/test_config.py`:

```python
def test_experiment_config_has_n_draws_default_one():
    ec = ExperimentConfig(mode="A", forecast_resolution="T21", truth_resolution="T21")
    assert ec.n_draws == 1
    ec2 = ExperimentConfig(mode="A", forecast_resolution="T21", truth_resolution="T21",
                           n_draws=32)
    assert ec2.n_draws == 32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_config.py::test_experiment_config_has_n_draws_default_one -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'n_draws'`.

- [ ] **Step 3: Add the field**

In `examples/stochastic_sppt/config.py`, inside `ExperimentConfig`, add the field after `ic_perturb_amp` and before `seed`:

```python
    ic_perturb_amp: float = 0.0     # 0 => SPPT-only spread
    n_draws: int = 1                # K: truth trajectories per IC (Mode-A K-draw)
    seed: int = 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_config.py -v`
Expected: PASS (all config tests, including the new one).

- [ ] **Step 5: Commit**

```bash
git add examples/stochastic_sppt/config.py examples/stochastic_sppt/tests/test_config.py
git commit -m "feat(sppt): add n_draws (K truth draws per IC) to ExperimentConfig

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `mode_a_dataset_multidraw` — K truth trajectories per IC

**Files:**
- Modify: `examples/stochastic_sppt/generate_data.py` (add two functions; do not touch `mode_a_dataset` or `_pack`)
- Test: `examples/stochastic_sppt/tests/test_generate_data.py`

**Interfaces:**
- Consumes: `ExperimentConfig.n_draws` (Task 1); existing `_sample_case_ics`, `_lead_steps`, `_stack_specs`, `_FIELDS`, `build_model`, `ModelConfig`; `rollout.ensemble_rollout(params, spec0, keys, bundle, lead_steps)` returning `{field: (M, n_lead, lat, lon)}`.
- Produces: `mode_a_dataset_multidraw(exp: ExperimentConfig, true_params) -> dict` with keys `ic_specs` (batched over cases), `truth[field]` of shape `(n_cases, n_draws, n_lead, lat, lon)`, `lead_steps`, `forecast_resolution`, `n_lev`. Consumed by Task 3's loss and Task 4's script.

- [ ] **Step 1: Write the failing test**

Add to `examples/stochastic_sppt/tests/test_generate_data.py`:

```python
def test_mode_a_multidraw_shapes():
    from examples.stochastic_sppt import sppt
    exp = ExperimentConfig(
        mode="A", forecast_resolution="T21", truth_resolution="T21",
        n_members=2, lead_days=(1,), n_cases=2, spinup_days=1.0,
        case_stride_days=1.0, n_draws=3, seed=0,
    )
    data = generate_data.mode_a_dataset_multidraw(exp, sppt.default_params())
    # truth: (n_cases, n_draws, n_leads, lat, lon)
    assert data["truth"]["t500"].shape == (2, 3, 1, 32, 63)
    assert data["forecast_resolution"] == "T21"
    # ic_specs batched over the 2 cases
    assert data["ic_specs"].temperature.shape[0] == 2
    assert set(data["truth"]) == {"u850", "v850", "t500", "vort500", "ps"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_generate_data.py::test_mode_a_multidraw_shapes -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'mode_a_dataset_multidraw'`.

- [ ] **Step 3: Implement `mode_a_dataset_multidraw` and `_pack_multidraw`**

In `examples/stochastic_sppt/generate_data.py`, add the `ensemble_rollout` import next to the existing imports (after the `from .model import ...` line):

```python
from .rollout import ensemble_rollout
```

Then add these two functions after `mode_a_dataset` (leave `mode_a_dataset` and `_pack` unchanged):

```python
def mode_a_dataset_multidraw(exp: ExperimentConfig, true_params):
    """Mode-A truth with K stochastic trajectories per IC (truth K-draw).

    Reuses `ensemble_rollout` to draw `exp.n_draws` truth trajectories from each
    sampled IC, so the truth carries the AR(1) SPPT pattern continuously across
    lead boundaries (the more correct behaviour vs. `mode_a_dataset`, which
    re-initialises the pattern at each lead). truth[field] has an extra draw axis.
    """
    bundle = build_model(ModelConfig(resolution=exp.forecast_resolution))
    key = jax.random.PRNGKey(exp.seed)
    lead_steps = _lead_steps(exp, bundle)

    case_ics = _sample_case_ics(bundle, key, exp)
    ic_specs, truth = [], {k: [] for k in _FIELDS}
    for i, ic in enumerate(case_ics):
        ic_specs.append(ic)
        draw_keys = jax.random.split(jax.random.fold_in(key, 1000 + i), exp.n_draws)
        ens = ensemble_rollout(true_params, ic, draw_keys, bundle, lead_steps)
        for k in _FIELDS:
            truth[k].append(ens[k])   # (n_draws, n_lead, lat, lon)
    return _pack_multidraw(ic_specs, truth, exp, bundle)


def _pack_multidraw(ic_specs, truth, exp, bundle):
    return {
        "ic_specs": _stack_specs(ic_specs),
        # (n_case, n_draws, n_lead, lat, lon)
        "truth": {k: jnp.stack(v) for k, v in truth.items()},
        "lead_steps": _lead_steps(exp, bundle),
        "forecast_resolution": exp.forecast_resolution,
        "n_lev": bundle.n_lev,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_generate_data.py -v`
Expected: PASS (both the new multidraw test and the existing single-draw shape test).

- [ ] **Step 5: Commit**

```bash
git add examples/stochastic_sppt/generate_data.py examples/stochastic_sppt/tests/test_generate_data.py
git commit -m "feat(sppt): mode_a_dataset_multidraw — K truth draws per IC via ensemble_rollout

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `afcrps_multidraw` — average afCRPS over K truth draws

**Files:**
- Modify: `examples/stochastic_sppt/metrics.py` (add one function; `afcrps` unchanged)
- Test: `examples/stochastic_sppt/tests/test_metrics.py`

**Interfaces:**
- Consumes: existing `metrics.afcrps(ensemble, truth, alpha)`.
- Produces: `afcrps_multidraw(ensemble, truths, alpha=0.95)` — `ensemble` shape `(M, ...)`, `truths` shape `(K, ...)`, returns the scalar mean over the K per-draw afCRPS values. Consumed by Task 4's loss.

- [ ] **Step 1: Write the failing test**

Add to `examples/stochastic_sppt/tests/test_metrics.py`:

```python
def test_afcrps_multidraw_k1_equals_afcrps():
    ens = jax.random.normal(jax.random.PRNGKey(0), (8, 5, 5))
    y = jax.random.normal(jax.random.PRNGKey(1), (5, 5))
    single = metrics.afcrps(ens, y, 0.95)
    multi = metrics.afcrps_multidraw(ens, y[None], 0.95)   # K=1
    assert abs(float(single) - float(multi)) < 1e-12


def test_afcrps_multidraw_is_mean_over_draws():
    ens = jax.random.normal(jax.random.PRNGKey(2), (6, 4))
    truths = jax.random.normal(jax.random.PRNGKey(3), (5, 4))   # K=5 draws
    multi = float(metrics.afcrps_multidraw(ens, truths, 0.95))
    manual = float(jnp.mean(jnp.stack(
        [metrics.afcrps(ens, truths[k], 0.95) for k in range(truths.shape[0])])))
    assert abs(multi - manual) < 1e-12


def test_afcrps_multidraw_differentiable():
    ens = jax.random.normal(jax.random.PRNGKey(4), (6, 4))
    truths = jax.random.normal(jax.random.PRNGKey(5), (5, 4))

    def loss(scale):
        return metrics.afcrps_multidraw(ens * scale, truths, 0.95)

    g = jax.grad(loss)(1.0)
    eps = 1e-4
    fd = (loss(1.0 + eps) - loss(1.0 - eps)) / (2 * eps)
    assert abs(float(g) - float(fd)) < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_metrics.py::test_afcrps_multidraw_k1_equals_afcrps -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'afcrps_multidraw'`.

- [ ] **Step 3: Implement `afcrps_multidraw`**

Add to `examples/stochastic_sppt/metrics.py` — first add `import jax` at the top (the file currently imports only `jax.numpy as jnp`), then add the function after `afcrps`:

```python
def afcrps_multidraw(ensemble, truths, alpha=0.95):
    """Mean almost-fair CRPS of one ensemble against K independent truth draws.

    ensemble: (M, ...), truths: (K, ...). Averages afcrps over the K draws that
    share the single forecast ensemble — the expensive rollout is not repeated
    per draw, only the cheap |ensemble - truth| term is."""
    return jnp.mean(jax.vmap(lambda t: afcrps(ensemble, t, alpha))(truths))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_metrics.py -v`
Expected: PASS (all metric tests).

- [ ] **Step 5: Commit**

```bash
git add examples/stochastic_sppt/metrics.py examples/stochastic_sppt/tests/test_metrics.py
git commit -m "feat(sppt): afcrps_multidraw — mean afCRPS over K truth draws

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `figures/run_convergence.py` — standalone resumable σ-only run

**Files:**
- Create: `examples/stochastic_sppt/figures/run_convergence.py`
- Test: `examples/stochastic_sppt/tests/test_run_convergence.py`

**Interfaces:**
- Consumes: `mode_a_dataset_multidraw` (Task 2), `afcrps_multidraw` (Task 3), `generate_data.{save_dataset,load_dataset}`, `sppt.{SPPTParams,default_params}`, `build_model`, `ModelConfig`, `ExperimentConfig`, `rollout.ensemble_rollout`.
- Produces: a runnable module `python -m examples.stochastic_sppt.figures.run_convergence` and, for the test, an importable `main(argv)` plus a `multidraw_loss_fn(params, ic_specs, truth, case_indices, bundle, lead_steps, base_key, alpha, scales, n_members)` helper.

- [ ] **Step 1: Write the failing test**

Create `examples/stochastic_sppt/tests/test_run_convergence.py`:

```python
import os
from pathlib import Path

from examples.stochastic_sppt.figures import run_convergence


def test_convergence_smoke_and_resume(tmp_path, monkeypatch):
    # Point artifacts at a temp dir so the test is hermetic.
    monkeypatch.setattr(run_convergence, "ART", tmp_path)
    monkeypatch.setattr(run_convergence, "DATASET", tmp_path / "ds.pkl")
    monkeypatch.setattr(run_convergence, "CKPT", tmp_path / "ck.pkl")

    argv = ["--cases", "2", "--draws", "2", "--members", "2", "--batch", "2",
            "--target-steps", "1", "--lead-days", "1", "--spinup-days", "1",
            "--budget-seconds", "600"]
    run_convergence.main(argv)
    assert (tmp_path / "ck.pkl").exists()
    assert (tmp_path / "ds.pkl").exists()

    # Re-invoking with a higher target resumes from the checkpoint (step advances).
    import pickle
    with open(tmp_path / "ck.pkl", "rb") as f:
        step_after_first = pickle.load(f)["step"]
    assert step_after_first == 1
    run_convergence.main(argv[:-2] + ["--target-steps", "2", "--budget-seconds", "600"])
    with open(tmp_path / "ck.pkl", "rb") as f:
        assert pickle.load(f)["step"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_run_convergence.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'examples.stochastic_sppt.figures.run_convergence'`.

- [ ] **Step 3: Implement `run_convergence.py`**

Create `examples/stochastic_sppt/figures/run_convergence.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/test_run_convergence.py -v`
Expected: PASS. (The smoke run compiles the T21 rollout once; allow a couple of minutes. Runs in the foreground.)

- [ ] **Step 5: Commit**

```bash
git add examples/stochastic_sppt/figures/run_convergence.py examples/stochastic_sppt/tests/test_run_convergence.py
git commit -m "feat(sppt): standalone resumable K-draw sigma-convergence run

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Full suite green + build the production dataset

**Files:**
- Run only (no source changes). Produces `examples/stochastic_sppt/figures/_artifacts/convergence_dataset_T21.pkl`.

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: the cached 32×32 T21 K-draw dataset artifact, ready for the longer training run the user drives.

- [ ] **Step 1: Run the whole suite to confirm nothing regressed**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest examples/stochastic_sppt/tests/ -q`
Expected: all tests pass (the prior 21 plus the new config/generate_data/metrics/run_convergence tests). If any prior test fails, stop — an additive change broke something; fix before proceeding.

- [ ] **Step 2: Build the production K-draw dataset (foreground)**

Run: `/Users/joymonteiro/miniconda3/envs/climt/bin/python -m examples.stochastic_sppt.figures.run_convergence --data-only`
Expected: prints `generating T21 K-draw dataset (32 cases x 32 draws)` then `dataset ready`. Writes `figures/_artifacts/convergence_dataset_T21.pkl`.
Note: this is the heaviest single step (32 cases × 32-member truth rollouts at T21). If it exceeds the foreground window it must be re-run — but `mode_a_dataset_multidraw` is not itself checkpointed, so it restarts the dataset build from scratch. If it does not finish in one window, reduce `--cases`/`--draws` for the build or hand this step to the user to run uninterrupted (per EXPERIMENT_LOG, heavy builds run foreground and uninterrupted here).

- [ ] **Step 3: Commit any artifact bookkeeping (if applicable)**

The `_artifacts/` pickles are gitignored (per EXPERIMENT_LOG build hygiene). Nothing to commit here unless `.gitignore` needs a new entry; verify with:

Run: `git status --porcelain examples/stochastic_sppt/figures/_artifacts/`
Expected: no output (artifacts ignored). If the dataset shows as untracked, add its pattern to `.gitignore` and commit that one-line change.

---

### Task 6: Run the longer training and record the settled σ

**Files:**
- Modify: `examples/stochastic_sppt/EXPERIMENT_LOG.md` (append one dated entry — append-only, do not rewrite prior entries).

**Interfaces:**
- Consumes: the dataset from Task 5 and `run_convergence.py` from Task 4.
- Produces: the observed settled σ and an EXPERIMENT_LOG entry documenting it.

- [ ] **Step 1: Run the σ-only convergence, resumably, to the target step count**

Run (repeat until it prints `DONE`, each invocation resumes from the checkpoint):
`/Users/joymonteiro/miniconda3/envs/climt/bin/python -m examples.stochastic_sppt.figures.run_convergence`
Expected: per-step lines `step N afCRPS ... sigma ...`; on budget it checkpoints and exits (`re-run to continue`); the final invocation prints `DONE: 150 steps. final sigma X.XXXX last-10 mean Y.YYYY (true 0.5)`.

- [ ] **Step 2: Append the result to the EXPERIMENT_LOG**

Append a new dated section to `examples/stochastic_sppt/EXPERIMENT_LOG.md` (before the closing `*(entries continue…)*` line), following the existing changelog style. Fill the bracketed values from the actual run output:

```markdown
### 2026-07-17 — K-draw Mode-A truth: σ convergence with reduced finite-sample bias
- **Motivation.** The single-draw Mode-A truth left a ~2–5% upward finite-sample
  bias in recovered σ (σ settled a touch above 0.5). To test convergence, truth
  now uses **K stochastic trajectories per IC** (`mode_a_dataset_multidraw`, via
  `ensemble_rollout`), and the loss averages afCRPS over the K draws sharing one
  forecast ensemble (`afcrps_multidraw`) — cheap at train time (rollouts not
  repeated per draw). Bonus: `ensemble_rollout` carries the AR(1) pattern across
  lead boundaries, removing the "pattern re-init per lead" imperfection of the
  old single-draw path.
- **Config.** T21 identical-twin, **32 cases × 32 draws**, σ-only (τ, ℓ frozen at
  truth), 1-day lead, lr=0.03, 4 members × batch 6, start σ=0.2, [TARGET] steps.
- **Result.** σ settled to **final [X.XXXX], last-10 mean [Y.YYYY]** (true 0.5) —
  [compare to the earlier 12-case×1-draw last-10 mean ~0.508–0.523; state whether
  the residual bias shrank as predicted by the ≈1/(cases·K) scaling].
- **Reproduce.** `python -m examples.stochastic_sppt.figures.run_convergence
  --data-only` then re-invoke without `--data-only` (foreground, resumable).
```

- [ ] **Step 3: Commit**

```bash
git add examples/stochastic_sppt/EXPERIMENT_LOG.md
git commit -m "docs(sppt): log K-draw sigma-convergence result (settled sigma recorded)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- **Do not modify** `mode_a_dataset`, `train.py`, or `figures/run_sweep.py`. All new behaviour is additive so the existing M1 figures still reproduce.
- **Foreground only** for anything that runs JAX compute (Tasks 4 smoke test, 5 build, 6 training). Background jobs get killed on this machine even when memory is healthy.
- **Compile cost:** the T21 1-day rollout graph compiles once per process (~15 s); subsequent steps are fast. The smoke test in Task 4 pays this once.
- **If the Task 5 dataset build overruns the window:** it is not internally checkpointed. Either hand it to the user to run uninterrupted, or temporarily build with fewer cases/draws to validate the path, then do the full 32×32 build in an uninterrupted foreground session.
