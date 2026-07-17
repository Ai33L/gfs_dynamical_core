# Design: K-draw Mode-A dataset + longer σ-convergence run

**Date:** 2026-07-17
**Branch:** `feat/trainable-sppt`
**Companion artifacts:** implementation plan `docs/superpowers/plans/2026-07-17-sppt-multidraw-convergence.md`;
prior design `docs/specs/wp5_trainable_sppt_design.md`; experiment log
`examples/stochastic_sppt/EXPERIMENT_LOG.md`.

## Problem

In the current trainable-SPPT Mode-A (parameter-recovery) pipeline, the "truth"
for each initial condition is a **single** stochastic SPPT trajectory. The
EXPERIMENT_LOG identifies this single-draw sampling as the source of a small
**upward finite-sample bias** in the recovered σ (it settles ~2–5% above the
true 0.5). The user is not convinced σ has converged and wants:

1. A **larger training dataset** that includes **multiple truth trajectories
   started from the same initial condition** (K draws per IC).
2. A **longer training run** to observe the value σ finally settles to.

The tutorial (`tutorial.pdf`) already analyses this: with K truth draws per IC
the bias shrinks ≈ 1/K, and it is **cheap at train time** because the single
forecast ensemble rollout is scored against all K truth draws (only the cheap
elementwise `|ensemble − truth|` term is repeated, not the rollouts).

## Goals

- Generate a Mode-A dataset with **32 initial conditions × 32 truth draws each**
  at T21, with truth shape `(n_cases, K, n_lead, lat, lon)`.
- Train **σ only** (τ, ℓ frozen at truth) for a **longer** run (target 150
  steps) and report where σ settles (final and last-10-step-mean σ).
- Do it **additively**: the existing `mode_a_dataset`, `train.py`, and
  `figures/run_sweep.py` remain untouched, so the current M1 figures still
  reproduce exactly.

## Non-goals

- Mode B (model-error) changes.
- Training τ or ℓ (σ-only keeps the convergence read clean).
- GPU/cluster training; this stays T21/CPU/foreground on the dev machine.
- Changing `_VAR_NORM` or the resolution-constant σ interpretation (see log).

## Design

### Component 1 — `config.py`: add a draws knob

Add one field to `ExperimentConfig`:

```python
n_draws: int = 1     # K: truth trajectories per IC (Mode-A K-draw)
```

Default `1` preserves all existing behaviour.

### Component 2 — `generate_data.py`: `mode_a_dataset_multidraw`

New function, additive (does not touch `mode_a_dataset`):

```python
def mode_a_dataset_multidraw(exp: ExperimentConfig, true_params):
    bundle = build_model(ModelConfig(resolution=exp.forecast_resolution))
    key = jax.random.PRNGKey(exp.seed)
    lead_steps = _lead_steps(exp, bundle)
    case_ics = _sample_case_ics(bundle, key, exp)          # reuse existing sampler
    ic_specs, truth = [], {k: [] for k in _FIELDS}
    for i, ic in enumerate(case_ics):
        ic_specs.append(ic)
        tkeys = jax.random.split(jax.random.fold_in(key, 1000 + i), exp.n_draws)
        ens = ensemble_rollout(true_params, ic, tkeys, bundle, lead_steps)  # (K,n_lead,lat,lon)
        for k in _FIELDS:
            truth[k].append(ens[k])
    # truth[k] stacked over cases -> (n_cases, K, n_lead, lat, lon)
    return _pack_multidraw(ic_specs, truth, exp, bundle)
```

- **Reuses `rollout.ensemble_rollout`** (the same vmap-over-keys, pattern-carrying
  rollout the forecast uses) to produce the K truth draws. This is the key reuse
  decision.
- **Correctness bonus:** `ensemble_rollout` carries the AR(1) SPPT pattern
  **continuously across lead boundaries**, fixing the "pattern re-init at each
  lead" imperfection the log flagged for the old `mode_a_dataset` single-draw
  path. (At τ=6 h with multi-day leads this is numerically inert, but at the
  1-day lead used here it is the more correct behaviour and removes the caveat.)
- `_pack_multidraw` mirrors `_pack` but its `truth` carries the extra K axis:
  `(n_case, K, n_lead, lat, lon)`. Same `ic_specs`, `lead_steps`,
  `forecast_resolution`, `n_lev` keys as `_pack`, so the dataset dict is
  drop-in for a K-draw-aware loss.

`main()` (the CLI) is left as-is (still single-draw Mode A / Mode B); the
multidraw path is driven by the new convergence script, not this CLI. (Optional,
low priority: a `--draws` flag could be added later; not in scope now.)

### Component 3 — K-draw metric (`metrics.py`) and loss (in the script)

The forecast side is unchanged: one M-member `ensemble_rollout` per case. The
truth side now has K draws, so afCRPS is averaged over the K draws sharing that
one ensemble. Add one small function to `metrics.py` (kept there so it is
unit-testable):

```python
def afcrps_multidraw(ensemble, truths, alpha=0.95):   # ensemble (M,...), truths (K,...)
    return jnp.mean(jax.vmap(lambda t: afcrps(ensemble, t, alpha))(truths))
```

Reuses `metrics.afcrps` unchanged. The O(M²) ensemble-internal term is recomputed
per draw, but M is small (4) so this is negligible — the expensive forecast
rollouts are **not** repeated per draw (the tutorial's "cheap at train time"
property). The convergence script defines its own `loss_fn` that mirrors
`train.loss_fn` (vmap over the case batch, per-field std normalisation, common
random numbers) but calls `afcrps_multidraw`, with `truth[k]` now
`(n_cases, K, n_lead, lat, lon)` and the per-case truth slice `(K, n_lead, …)`.

### Component 4 — `figures/run_convergence.py` (standalone, user-driven)

Self-contained CLI modeled on `run_sweep.py`'s proven machinery:

- **Entry-point hardening:** `JAX_ENABLE_X64=True`, `setdefault(JAX_PLATFORMS,"cpu")`
  before importing jax; `jax.config.update("jax_enable_x64", True)`.
- **Foreground + resumable:** per-step checkpoint `{step, params, opt_state,
  history}` to `figures/_artifacts/convergence_ckpt.pkl`; `--budget-seconds`
  wall-clock cap → checkpoint and exit; re-invoke to continue.
- **Dataset caching:** builds the K-draw dataset once to
  `figures/_artifacts/convergence_dataset_T21.pkl` (via
  `mode_a_dataset_multidraw`), reused on resume; `--data-only` to build and exit.
- **σ-only:** freeze τ, ℓ gradients (reuse the `_sigma_only` gradient mask pattern).
- **CLI knobs (defaults):** `--cases 32 --draws 32 --members 4 --batch 6
  --target-steps 150 --lr 0.03 --init-sigma 0.2 --lead-days 1
  --spinup-days 120 --budget-seconds 520`.
- **Readout on completion:** print **final σ** and **last-10-step mean σ** (the
  plateau), plus the trajectory checkpoint for plotting.

### Data flow

```
_sample_case_ics (32 ICs, T21)
   └─> per IC: ensemble_rollout(true_params, ic, 32 keys) -> truth (32 draws)
        └─> truth[k]: (32 cases, 32 draws, n_lead, lat, lon)  ─┐
ic_specs (32) ────────────────────────────────────────────────┤
                                                               ▼
run_convergence.py loop:
   sample batch of 6 cases -> ensemble_rollout(params, ic, 4 keys) forecast
      -> afcrps_multidraw(ens, truth_case[K=32]) per field, std-normalised
      -> mean -> value_and_grad -> σ-only mask -> Adam -> checkpoint
```

## Testing

- **Unit (`tests/`):** `afcrps_multidraw` with K=1 equals `afcrps`
  (regression); with identical draws equals the single-draw value; averages
  correctly for K>1 (mean of per-draw afCRPS).
- **Shape test:** `mode_a_dataset_multidraw` with small `n_cases`, `n_draws`, at
  T21, 1-day lead, returns `truth[k]` of shape `(n_cases, n_draws, n_lead, lat,
  lon)` and `ic_specs` batched over cases; existing single-draw
  `mode_a_dataset` shape test still passes (additive check).
- **Backward-compat:** run the existing test suite (21/21) — must stay green.
- **Smoke run:** `run_convergence.py --cases 2 --draws 2 --members 2 --batch 2
  --target-steps 1 --budget-seconds 60` completes and writes a checkpoint.

## Expected outcome

With 32 cases × 32 draws the finite-sample bias (≈ 1/(cases·K), previously
~2–5% from 12 cases × 1 draw) should be markedly smaller, so σ should settle
**much closer to 0.500**. The run reports the settled value, directly answering
the convergence question. A new dated EXPERIMENT_LOG entry records the K-draw
design, the `ensemble_rollout` reuse (and the pattern-carry correctness bonus),
and the observed settled σ.

## Files

| File | Change |
|------|--------|
| `examples/stochastic_sppt/config.py` | add `n_draws` field to `ExperimentConfig` |
| `examples/stochastic_sppt/generate_data.py` | add `mode_a_dataset_multidraw`, `_pack_multidraw` |
| `examples/stochastic_sppt/metrics.py` | add `afcrps_multidraw` |
| `examples/stochastic_sppt/figures/run_convergence.py` | new standalone resumable convergence run |
| `examples/stochastic_sppt/tests/` | new tests for the above |
| `examples/stochastic_sppt/EXPERIMENT_LOG.md` | append dated entry after the run |
