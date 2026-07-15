# Trainable-SPPT — Experiment & Design Log

**Purpose.** A pedagogical, append-only record of the *experimental and design
choices* made while building and running the trainable-SPPT pipeline, with the
**reasoning** behind each. It is meant to be handed to a student alongside the
code and the tutorial `README.md`: the README explains *what the pipeline is*;
this log explains *why we made the calls we made*, including the dead-ends and
the trade-offs. New entries are appended in date order — history is not
rewritten (corrections are added as new dated notes).

Companion artifacts: implementation plan `docs/superpowers/plans/2026-07-12-trainable-sppt.md`,
design spec `docs/specs/wp5_trainable_sppt_design.md`.

---

## Background: what the experiment is

We build a **differentiable, trainable SPPT** (Stochastically Perturbed
Parametrization Tendencies) scheme on a JAX Held–Suarez dynamical core, and
**learn its parameters by gradient descent** on an ensemble forecast score
(almost-fair CRPS). Two experiment modes:

- **Mode A (parameter recovery):** generate "truth" with *known* SPPT parameters
  θ\*, then check the optimizer recovers θ\* from a wrong start. This validates
  the machinery. (Milestone **M1**.)
- **Mode B (model-error calibration):** truth is a *higher-resolution* run
  (T127); the forecast is coarse (T42); train SPPT to calibrate the coarse
  ensemble against resolution error. (Milestone **M2**.)

The three trainable scalars are `SPPTParams(log_sigma, log_tau, log_len)`:
perturbation strength σ (grid-point std), decorrelation time τ, and spatial
correlation length ℓ.

---

## Decision log

### 2026-07-13 — Resolutions and the AR(1) pattern (Palmer 2009)
- **Choice.** Spectral AR(1) pattern generator following Palmer et al. (2009,
  ECMWF Tech Memo 598, App. 8.1): `phi = exp(-dt/tau)`,
  `sigma_n = F0 · exp(-kappaT·n(n+1)/2)`, stationary init. Default θ =
  (σ=0.5, τ=6 h, ℓ=500 km). Resolutions `(L, ntrunc, dt)`: T21=(32,21,1800 s),
  T42=(64,42,1200 s), T85, T127.
- **Why.** Palmer's scheme is the operational reference for SPPT; the spectral
  AR(1) gives a smooth, temporally-correlated, reproducible pattern whose
  variance and length-scale are set analytically by the three parameters.

### 2026-07-13 — `_VAR_NORM` calibrated at T21 only, left resolution-constant *(open trade-off)*
- **Context.** The analytic Palmer normalization `F0` must be reconciled with
  the `s2fft` inverse-transform's variance convention via a constant `_VAR_NORM`
  so that grid-point variance equals σ². Calibrated at T21: `_VAR_NORM ≈ 16.5`.
- **Finding.** The transform-convention factor is **not** resolution-independent:
  measured grid variance for σ²=0.25 is ~0.25 at T21 but ~0.31 (ratio ~1.25,
  flat) at T42/T85/T127. So `σ == grid-point std` holds *exactly only at T21*;
  at production resolutions the true grid std is ~1.12× exp(log_sigma).
- **Decision (user).** Leave `_VAR_NORM` constant across resolutions **for now**,
  to observe how the optimizer tunes the *effective* σ.
- **Why it's safe.** Mode-A recovery is *exact* regardless (the factor cancels
  between truth and forecast, which use the same code at the same resolution);
  Mode-B spread calibration is unaffected (the optimizer learns whatever σ gives
  matching spread). Only the *physical interpretation* of a learned σ is offset.
- **When to revisit.** If σ must literally mean grid-point std at all L, derive
  the per-L Parseval factor analytically (or calibrate at build time) and add a
  non-T21 variance test.

### 2026-07-13 — Loss: almost-fair CRPS (α = 0.95), common random numbers
- **Choice.** Train by minimizing **almost-fair CRPS** (Lang et al. 2024) with
  α = 0.95 (α = 1 ⇒ fair CRPS), aggregated over cases × leads × 5 verification
  fields (u850, v850, t500, vort500, ps), each normalized by its truth std.
  Ensemble members use **common random numbers** (member keys fixed per case
  within a gradient evaluation).
- **Why.** Fair CRPS removes the ensemble-size bias of plain CRPS; the "almost"
  variant avoids a degeneracy Lang et al. identified. Common random numbers make
  the stochastic gradient low-variance (the same noise draws are differentiated
  through each step), which is essential for optimizing a stochastic scheme.

### 2026-07-14 — Corrected `spread_error_ratio` to the textbook diagnostic
- **Context.** The plan's `spread_error_ratio` formula *and* its unit test were
  both statistically wrong (verified numerically).
- **Decision (user).** Fix to the **textbook** spread–error consistency relation
  (Fortin et al. 2014; Leutbecher 2018): for a reliable ensemble
  `E[(y − x̄)²] = ((M+1)/M)·E[s²]`, so the diagnostic that equals 1 for a
  reliable ensemble is `sqrt((M+1)/M)·spread/rmse`. The test now builds a genuine
  *exchangeable* ensemble (truth and members are M+1 i.i.d. draws from a common
  center); measured ratio 1.013.
- **Why it matters.** `spread_error_ratio → 1` is the Mode-B calibration target
  (M2); a wrong diagnostic would mislabel calibration. Lesson: a passing test
  against a *mis-constructed* scenario proves nothing — validate the scenario.

### 2026-07-14 — Float64 mandatory; on Apple Silicon pin the CPU backend
- **Finding.** The whole dynamical core requires float64. On the M3 Mac, JAX
  defaults to the **Metal** backend, which **cannot do float64** and crashes.
- **Choice.** Every entry point forces `JAX_ENABLE_X64=True` and
  `setdefault(JAX_PLATFORMS, "cpu")` before importing jax (a cluster's
  `JAX_PLATFORMS=cuda` still wins). CPU is the only float64-capable backend here.
- **Why.** float32 silently corrupts spectral dynamics; a wrong backend silently
  crashes or degrades. Make the requirement un-forgettable at the entry point.

### 2026-07-14 — Mode A truth data at T42 (identical-twin, θ\* = default params)
- **Choice.** Generate Mode A truth at **T42** with `default_params`
  (σ=0.5, τ=6 h, ℓ=500 km): 16 cases, leads (3,5,7,10) days, 200-day spin-up,
  5-day case stride. Dataset ≈ 189 MB, ~1 h 43 m on the M3 (CPU).
- **Why.** T42 is the target forecast resolution. Identical-twin (truth from the
  *same* model with known θ\*) isolates the learning machinery from model error —
  the clean setting to validate recovery (M1) before tackling model error (M2).
- **Accepted imperfection.** Mode-A "truth" re-initializes the SPPT pattern at
  each lead boundary instead of carrying it. This is **numerically inert** at
  τ=6 h with multi-day leads (`phi^n ≈ 1e-4…1e-6`, i.e. the pattern fully
  decorrelates between leads, so a fresh stationary draw ≈ a carried one). It
  would matter for long τ or sub-daily leads — flagged as a follow-up.

### 2026-07-15 — Jit the training step (vmap over cases)
- **Context.** The training loop was not `jax.jit`-wrapped, so every optimizer
  step retraced+recompiled the whole checkpointed vmap+scan graph (~98 s/step).
- **Choice.** Restructure `loss_fn` from a Python case-loop into a
  **vmap-over-cases** form (batch-gather ICs/truth, vmap `ensemble_rollout` and
  `afcrps` over the case batch); jit `value_and_grad` with `lead_steps`, `alpha`,
  `n_members` as static args (`bundle` stays a traced flax-struct arg).
- **Why.** Compile-once, then pay only compute per step — the enabler for any
  multi-step sweep. Verified numerics unchanged (first-step loss identical to the
  pre-jit run, digit-for-digit).

### 2026-07-15 — T42 gradient training is infeasible on the 18 GB M3 → sweep at T21, diagnose forward at T42
- **Finding (measured).** Reverse-mode gradient through the rollout stores a
  ~10 MB T42 spectral state **per step**; a 720-step (10-day-lead) rollout needs
  ~7.5 GB **per trajectory**, × (batch × members) lanes. Full config (batch 4,
  6 members) **OOM-killed** the process; the minimal config (batch 1, 2 members,
  3-day lead) ran at ~224 s/step / 3.7 GB but with a gradient far too noisy to
  converge.
- **Decision (user).** Do the M1 **convergence sweep at T21** (fast,
  memory-light — the plan's dev resolution, where M1 is defined), and use **T42
  only for forward-only diagnostics** (an ensemble forecast with no gradient has
  no reverse-mode memory blow-up). Real T42/T127 *training* needs a GPU/cluster
  (which the SLURM scripts assume).
- **Why.** Honestly matches the hardware; still lets us show T42 predictive
  distributions at the recovered parameters. Lesson: reverse-mode memory scales
  with rollout length × ensemble lanes — the binding constraint for
  differentiable weather models on a laptop.

### 2026-07-15 — M1 convergence sweep: σ-only, and learning-rate tuning
- **Choice.** T21 sweep recovering **σ only** (τ, ℓ frozen at truth), from a
  deliberately-wrong σ=0.2 toward true σ=0.5. Single-parameter recovery gives the
  cleanest, most interpretable convergence story.
- **First attempt (lr = 0.1).** σ recovered directionally but **overshot**:
  0.22 → 0.55 (step 10) → 0.73 (step 20), still climbing. Two causes:
  (1) adam lr too high (momentum carries σ past the target); (2) likely
  **finite-sample upward bias** — Mode-A truth is a *single* stochastic draw per
  case, and with only 8 cases the afCRPS-minimizing σ sits a bit above 0.5 (the
  optimizer inflates spread to cover an outlier truth draw). Also measured: each
  T21 grad step ≈ 100 s at 8 members × 4 cases (32 vmap lanes, CPU-bound), so a
  70-step sweep was a ~2 h job (it was killed before finishing).
- **Decision (user, option 1).** Refined gentle run: **lr = 0.03**, ~55 steps,
  **4 members × batch 3** (~12 lanes, ~40 s/step), **12 cases** (less
  finite-sample bias), with peak-RSS logging, per-step checkpointing, and dataset
  caching. Goal: show σ **settle near 0.5** without overshoot.
- **Why.** Separates the two failure modes: lower lr fixes overshoot; more cases
  reduces bias. A clean plateau near 0.5 is the convincing M1 evidence.

*(entries continue as the experiment proceeds)*
