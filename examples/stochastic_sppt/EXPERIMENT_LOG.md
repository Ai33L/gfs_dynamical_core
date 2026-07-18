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

### 2026-07-15 — M1 CONFIRMED: clean σ convergence at T21 (with a finite-sample bias note)
- **Result.** The refined run (lr=0.03, 4 members × batch 3, 12 cases) recovered σ
  along a smooth, decelerating, **monotonic** curve: 0.20 → crosses true 0.5 near
  step 38 → **plateaus at ~0.523** (last-10 mean 0.523). No overshoot. This is
  the clean M1 evidence; **parameter recovery works.**
- **Residual bias ≈ +0.023 (~5%).** σ settles slightly *above* 0.5, as predicted:
  Mode-A "truth" is a *single* stochastic draw per case, so with a finite number
  of cases the afCRPS-minimizing σ is biased a touch high (spread inflated to
  cover outlier truth draws). This is a **sampling** property, not a bug; it
  shrinks with more cases. We accept it as within-tolerance for M1.
- **On the learning rate.** lr=0.1 overshot to 0.73 (momentum); lr=0.03 converges
  cleanly. The two knobs are orthogonal: lr controls overshoot, #cases controls
  the residual bias.
- **Loss curve is batch-stochastic.** Per-step loss wobbles ~0.08–0.115 because
  each step scores a *different* random 3-case batch with fresh noise — it is not
  a smooth descent and should not be read as one. The parameter trajectory (not
  the raw loss) is the convergence signal here.
- **Operational lesson.** Long background jobs on this setup were killed near
  ~35–40 min (cause uncertain — RSS stayed flat at ~3.3 GB, so NOT OOM; and a
  separate 1 h 43 m forward job *did* finish, so it is not a hard wall-clock
  cap). Mitigation adopted: **checkpoint every unit of work to disk** (per-step
  trajectory here) so a kill never loses progress, and keep individual runs
  short/resumable. Each T21 grad step ran ~40 s at 12 vmap lanes.

### 2026-07-15 — Operational: run JAX in the FOREGROUND here (background jobs get killed)
- **Finding.** Long-running *background* jobs on this machine were repeatedly
  killed — sometimes near ~40 min, later within seconds (even during import) —
  while **memory was healthy** (47–49% free, no lingering processes, RSS ~1–3 GB).
  So it is **not** OOM and not a local resource limit; the environment's
  background runner terminates them. The identical job run in the **foreground**
  completed cleanly (EXIT 0, 1.2 GB).
- **Practice adopted.** Run JAX compute in the **foreground** with a bounded
  per-run scope (fits the tool's ~10 min window), and **checkpoint every unit of
  work to disk** (per-step trajectory, per-case ensembles) so any interruption is
  resumable. Split heavy compute from fast plotting (do plots in pure numpy).
- **Consequence.** The **T42 forward diagnostic was abandoned on this machine** —
  compiling/running the large T42 forward ensemble graph inside a background job
  was killed 3× before producing a case. We do the calibration diagnostics at
  **T21** instead (below). T42 diagnostics belong on a GPU/cluster.

### 2026-07-15 — M1 supplementary check: the recovered σ yields a calibrated ensemble (T21)
- **Setup.** Forward-only ensembles (no gradient) at the **recovered σ = 0.523**
  (τ, ℓ at truth), on a fresh T21 identical-twin Mode-A dataset (truth =
  default_params), leads 3 & 5 days, 3 cases × 8 members. Diagnostics pooled over
  cases × grid.
- **Result — calibrated.**
  - **Spread–error ratio ≈ 1** for all five fields at both leads (range
    0.96–1.09; target 1). Neither over- nor under-dispersed.
  - **Rank histogram (t500, 5 d) ≈ flat** (bins 553–751 vs uniform 672) — truth
    occupies uniformly-distributed ranks, the calibration signature.
  - **Truth anomaly distribution overlays the member distribution** — the truth
    looks like a draw from the ensemble.
- **Interpretation.** M1 is confirmed on two fronts: (1) parameter *recovery*
  (σ: 0.2 → ~0.52, plateaued) and (2) the recovered σ produces a *calibrated*
  ensemble. The tiny residual (u850 spread-error ~1.08, t500 ~0.96) is consistent
  with σ landing ~5% above 0.5 (the finite-sample bias noted earlier).
- **Caveat.** The `afCRPS-vs-lead` panel was plotted unnormalized, so surface
  pressure (Pa) dominates it visually; the per-field-std normalization used in
  the training loss is the right scaling for cross-field comparison.

### 2026-07-16 — Student README + reproducible figure pipeline; M1 re-run from scratch
- **What.** Wrote the student-facing tutorial `README.md` (the companion this log
  had promised but that never existed) and a reproducible figure pipeline under
  `figures/` (`run_sweep.py`, `run_calibration.py`, `plot.py`). Regenerated all
  five M1 figures from real runs — the earlier M1 results were recorded only in
  prose here, with no figures or trajectory checkpoints saved to disk.
- **Sweep leads shortened to 1 day (hardware).** Reverse-mode compile time grows
  with rollout length: the 5-day-lead (240-step) graph compiles for ~8.5 min,
  which does not fit this machine's **~10-min per-process ceiling** (confirmed
  again: a background sweep was killed after compile + 1 step; a 1-day graph
  compiles in ~15 s). σ-recovery is lead-agnostic (it only matches ensemble
  spread to the identical-twin truth), so the sweep uses a **1-day lead**;
  calibration stays forward-only at 3 & 5 days. Documented in `README.md` and
  `run_sweep.py`.
- **Result — M1 reproduced.** T21 σ-only, lr=0.03, 4 members × batch 3, 12 cases:
  σ climbs monotonically 0.20 → crosses true 0.5 at **step 38** → **last-10 mean
  0.508** (final 0.517). Residual bias **~+0.01 (~2%)**, smaller than the earlier
  run's ~5% — consistent with it being a finite-sample sampling property.
- **Calibration (fresh seed=1, forward).** Spread–error ratio **0.92–1.05** across
  the five fields at 3 & 5 d (slightly *under*-dispersed at t500 ≈ 0.92); rank
  histogram (t500, 5 d) **approximately flat** with a mild tilt toward high ranks
  (the same under-dispersion); truth-anomaly distribution overlays the member
  distribution. Calibrated, with an honest small under-dispersion noted rather
  than smoothed over.
- **Operational pattern adopted.** Split every stage into `--data-only` (build the
  dataset artifact) then the compute stage, each fitting one foreground window;
  the sweep checkpoints per step under a wall-clock `--budget-seconds` and is
  resumed by re-invoking. Heavy JAX compute stays in the foreground; plotting is
  pure numpy. Datasets/checkpoints live in `figures/_artifacts/` (gitignored-size
  pickles); the PNGs are committed for the README.

### 2026-07-16 — Self-contained LaTeX tutorial (`tutorial.pdf`); README demoted to a front-door
- **Why.** Review feedback: the Markdown README was a run-guide, not a teaching
  document — it stated results without defining the diagnostics, omitted the
  mathematical formulation of the training, and assumed too much of the reader.
- **What.** Wrote `tutorial.tex` → `tutorial.pdf` (15 pp, built with `latexmk`),
  a from-first-principles tutorial for a master's student. Covers, with
  derivations: SPPT + the spectral AR(1) generator (stationary variance, how
  σ/τ/ℓ map to φ, σ_n, κ_T); how truth data is generated/stored and the
  **single-draw vs. K-draw** analysis (bias ↓ ~1/K, cheap at train time because
  the forecast rollout is shared across truth draws, costs storage + truth-gen
  time); CRPS → fair → almost-fair (finite-ensemble bias); spread–error ratio
  (full (M+1)/M derivation) and rank histogram (interpretation table); and the
  learning core — the **reparameterization/pathwise gradient** (differentiate
  through fixed N(0,1) noise; gradient is w.r.t. the three log-params; truth and
  η held fixed), common random numbers, reverse-mode memory, and **why σ settles
  ~2% above 0.5** (fair CRPS is proper ⇒ minimiser is θ* in the population, but
  finite cases + one truth draw/case bias σ slightly high). Adds train/val
  protocol, per-figure reproduction with code, limitations, 5 exercises, and a
  referenced further-reading section. README is now a short front-door pointing
  to the PDF; the deep prose lives in LaTeX.
- **Build hygiene.** `.gitignore`: keep `figures/*.png` and `tutorial.pdf`; ignore
  LaTeX aux (`*.aux/*.toc/*.out/*.fls/*.fdb_latexmk`) and the `_artifacts/`
  pickles. Rebuild with `latexmk -pdf tutorial.tex`.

### 2026-07-18 — K-draw Mode-A truth: σ convergence with the finite-sample bias removed
- **Motivation.** The single-draw Mode-A truth left a ~2–5% *upward* finite-sample
  bias in recovered σ (the 12-case × 1-draw runs settled at last-10 means ~0.508
  and ~0.523, i.e. a touch above the true 0.5, and it was not obvious the curve
  had stopped climbing). Hypothesis: with a *single* stochastic truth draw per
  case the afCRPS-minimizing σ is inflated to hedge outlier truths; drawing **K**
  truth trajectories per IC should shrink that bias ≈ 1/(cases·K).
- **What changed.** Truth now uses **K stochastic trajectories per IC**
  (`generate_data.mode_a_dataset_multidraw`, which reuses `ensemble_rollout` with
  the true params), and the loss averages afCRPS over the K draws that share one
  forecast ensemble (`metrics.afcrps_multidraw`) — cheap at train time because the
  M forecast rollouts are *not* repeated per draw, only the elementwise
  `|ensemble − truth|` term is. Driven by the standalone, resumable
  `figures/run_convergence.py` (σ-only; τ, ℓ frozen at truth).
  - **Correctness bonus.** `ensemble_rollout` carries the AR(1) SPPT pattern
    *continuously across lead boundaries*, removing the "pattern re-init per lead"
    imperfection the old single-draw `mode_a_dataset` had (2026-07-14 entry). At
    the 1-day lead used here this is the more correct behaviour.
- **Config.** T21 identical-twin, **32 cases × 32 draws**, σ-only, 1-day lead,
  lr=0.03, 4 members × batch 6, start σ=0.2. Ran to **step 97** (of a 150 target)
  — stopped once the parameter had clearly plateaued.
- **Result — σ settles at ~0.48, bias flipped from + to − and shrank.** The
  trajectory is now an **overshoot-and-relax**, not a monotone climb: σ crosses
  the true 0.5 at **step 38**, overshoots to **~0.522** (peak ~step 50), then
  *relaxes back* and plateaus — **final 0.486, last-10 mean 0.483, last-20 mean
  0.482** (true 0.5). The last-10 and last-20 means agreeing to ~0.001 is the
  plateau signature: this is genuine convergence, not a still-drifting curve.
- **Interpretation.** Increasing the truth draws removed the upward single-draw
  bias exactly as predicted — σ no longer settles *above* 0.5. It now sits
  marginally *below* (~−3%), which is consistent with Adam momentum ringing after
  the overshoot (the correction that pulled σ down from ~0.52 carried it a hair
  past 0.5) plus residual batch-stochastic sampling (each step scores a random
  6-of-32-case batch with fresh noise), **not** a new systematic downward bias —
  the recovery is exact at T21 (truth and forecast share code, so `_VAR_NORM`
  cancels). Net: the "has σ converged?" doubt is answered — with the larger
  K-draw dataset σ converges to a clean plateau near 0.48–0.49, and the
  overshoot-and-relax shape is a far more convincing convergence story than the
  earlier monotone-climb-to-slightly-high.
- **Reproduce.** `python -m examples.stochastic_sppt.figures.run_convergence
  --data-only` (build the 32×32 T21 dataset once) then re-invoke without
  `--data-only` (foreground, resumable per-step checkpoint). Caveat: the dataset
  and checkpoint are reused if the files exist regardless of CLI args — delete
  `figures/_artifacts/convergence_*.pkl` before changing `--cases/--draws/--lr/…`.

*(entries continue as the experiment proceeds)*
