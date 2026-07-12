# Design: WP5 — trainable differentiable SPPT (pipeline in `examples/stochastic_sppt/`)

**Date:** 2026-07-12
**Branch:** `feat/trainable-sppt`
**Status:** design approved in brainstorming; ready for `writing-plans`.
**Builds on:** C-WP1 (`advance_with_tendencies`, `SpectralTendencies`), see
`docs/specs/wp5_stochastic_sppt_handoff.md` and `research_program_four_tracks.md` §3.3.

---

## 1. Goal & framing

Build a **complete, trainable, differentiable SPPT pipeline** — data generation,
the SPPT model, a SLURM training script, and analysis scripts — living in a new
`examples/stochastic_sppt/` folder. It has two jobs:

1. **Product:** a calibrated **stochastic T42 Held-Suarez model** whose ensemble
   spread realistically represents model uncertainty, for use as the walker model
   in the rare-event-sampling (RES) workflow (Track 3, §3.4 of the research doc).
2. **Pedagogy:** a readable, top-to-bottom learning script for a master's student
   who will fork it into **SKEB** (WP6). The spectral pattern generator is written
   so SKEB reuses it verbatim — only the injection point changes (SPPT multiplies
   grid-space physics tendencies; SKEB adds to `spec_tends.d_vorticity_d_t`).

Everything is a **pure JAX function** — `jit`/`grad`/`vmap`-able — and deliberately
readable over clever.

### Authoritative references (in `docs/refs/`)

- **Palmer et al. 2009**, ECMWF Tech Memo 598 — revised SPPT + spectral AR(1)
  pattern generator (Appendix 8.1). *The SPPT scheme is reproduced exactly from
  this.* (`11577-stochastic-parametrization-and-model-uncertainty.pdf`)
- **Leutbecher et al. 2017**, *JCP* — SPPT + SKEB state of the art; the SKEB
  parallel the student graduates to. (`Stochastic_representations_of_model_unce.pdf`)
- **Lang et al. 2024**, *npj AI* (AIFS-CRPS) — almost-fair CRPS training loss.
  (`s44387-026-00073-7.pdf`)

The **trainable/differentiable framing is the novel contribution**; the SPPT
scheme and the loss are reproduced faithfully from the papers above (not from
memory).

---

## 2. Locked design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Physics perturbed | **Held-Suarez** forcing (Newtonian relaxation + Rayleigh friction) | Self-contained; no ERA5 dependency; classic idealized testbed |
| Truth source | **Identical-twin** with a resolution gap | Deterministic perfect-model would train amplitude→0; the gap *is* the model error SPPT represents |
| Pattern wiring | **Spectral AR(1) generator → transform to grid → multiply grid physics tendencies** | Faithful multiplicative SPPT *and* the spectral machinery is exactly what SKEB reuses |
| Trainables | **3 physical scalars** `(log σ, log τ, log κ_T)` | Faithful trainable-SPPT; crystal-clear learning target; clean pytree seam for later NN |
| Loss | **almost-fair CRPS** (α≈0.95); `fair_crps` kept as a switch | NeuralGCM/AIFS choice; avoids fair-CRPS degeneracy |
| Verification fields | **850 hPa u, 850 hPa v, ≈500 hPa T, ≈500 hPa relative vorticity, surface pressure** | Low-level winds + rotational flow (the SKEB-relevant signal) + mass field |
| Cluster | **SLURM, single-GPU**; members via `vmap`; `jax.checkpoint` on the step | Simplest correct target; multi-GPU is a later seam |
| Product resolution | **T42** (forecast/target) | The stochastic model to be fed to RES |
| Truth resolution | **T127** nature run, coarse-grained (spectral truncation) to T42 | ~3× gap → strongest model-error signal; showcases cheap re-truncation |
| Dev resolution | **T21 forecast / T42 truth** | Laptop iteration |

### Two experiment modes

- **Mode A — parameter recovery (first milestone, laptop-runnable at T21).**
  "Truth" = single-member runs from a T42 (or T21) model with **known** SPPT
  params θ\*. Because the CRPS minimizer is the truth distribution, training the
  forecast ensemble (params θ) must drive θ → θ\*. End-to-end test of the
  CRPS + autodiff loop *with a known answer*.
- **Mode B — model-error calibration (production, cluster T42/T127).**
  Truth = T127 HS nature run → spectral-truncate to T42 → train the stochastic
  T42 to be reliable at 3/5/7/10-day leads against the truncated truth.

**Honest caveat (state in README):** Mode B calibrates T42 spread against
*resolution error relative to the T127 HS run* — the best available "truth" proxy
in an idealized twin, not real atmospheric model error. For an RES demonstrator
this is the correct target.

---

## 3. The SPPT scheme (faithful to Palmer 2009 App. 8.1)

**Application to tendencies (Eq. 2):** for each perturbed field `X ∈ {u, v, T}`,
```
X_p = (1 + µ · r) · X_c
```
with the **same univariate** grid pattern `r`, and vertical taper `µ ∈ [0,1]`
(zero in the lowest ≈300 m and in the stratosphere; smooth ramp). The factor is
clipped: `factor = 1 + clip(µ·r, -0.9, 0.9)` (keeps the semi-implicit stepper
stable, differentiable a.e.). The grid pattern is additionally bounded to ±3σ.
`log_surface_pressure` and `tracers` tendencies are **not** perturbed (dry HS core).

**Spectral AR(1) pattern generator (Appendix 8.1):**
```
r(x)          = Σ_{m,n} r̂_{mn} Y_{mn}(x)                     (grid ← spectral, s2_inverse)
r̂_{mn}(t+Δt) = φ · r̂_{mn}(t) + σ_n · η_{mn}(t)              (14)  AR(1)
φ             = exp(-Δt / τ)                                  (15)
σ_n           = F0 · exp(-κ_T · n(n+1) / 2)                   (17)  variance spectrum
F0            = sqrt( var(r) · (1 - φ²) /
                      (2 · Σ_{n=1}^{N} (2n+1) · exp(-κ_T n(n+1))) )   (18)  normalization
r̂_{mn}(0)    = (1 - φ²)^{-1/2} · σ_n · η_{mn}(0)             (19)  stationary init
```
- `η_{mn} ∈ ℂ`: Re and Im are independent `N(0,1)`, white in time, independent
  across harmonics. **Must respect the reality condition** for the codebase's
  `(L, 2L-1)` layout (`r̂_{l,-m} = (-1)^m conj(r̂_{l,m})`, `m=0` real). Reuse the
  existing reality utilities (`assert_reality_symmetric`,
  `enforce_triangular_truncation`) — the same invariant that a prior bug
  (reality-symmetry ghost) taught us to guard. Sample for `m ≥ 0`, fill `m < 0`
  by conjugate symmetry.
- `F0` fixes the **grid-point variance to `var(r) = σ²`** (uniform on the sphere).
- Correlation length `L_corr ≈ sqrt(2 κ_T) · R_E` (Gaussian on the sphere, Weaver
  & Courtier 2001). Overflow guard: bound `η` and `r̂` to ±10σ.

**Trainable parameters** (`SPPTParams`, a flax struct / pytree, positivity via
`exp`):
- `log_sigma`  → `σ` = grid-point standard deviation of `r` (`var(r) = σ²`).
- `log_tau`    → `τ` = decorrelation time (→ `φ`).
- `log_len`    → `L_corr` = correlation length in metres; `κ_T = (L_corr/R_E)² / 2`.

This is the *only* trainable object. Extension seams (SP2 two-scale, per-`n` τ, NN
amplitude field) are documented, not built.

---

## 4. Almost-fair CRPS (Lang et al. 2024, Eq. 4)

Per scalar (grid point × field × lead), for `M` members `{x_j}` and deterministic
truth `y`, numerically-stable positive-term form:
```
afCRPS_α = 1 / (2 M (M-1)) · Σ_j Σ_{k≠j} ( |x_j - y| + |x_k - y| - (1-ε) |x_j - x_k| )
ε        = (1 - α) / M ,   α ∈ (0,1]   (α = 1 → fair CRPS; default α ≈ 0.95)
```
Aggregated over grid points, verification fields, and leads with **per-variable
loss scaling** and an **upper-air pressure weighting** `w_pl = p_lev/1000`
(optionally floored at 0.2), per AIFS. `fair_crps` (α=1) is provided as a switch
for the README's fair-vs-almost-fair demonstration.

**Diagnostics (monitored, never trained on):** spread–error ratio (with the
finite-`M` correction), rank histograms.

---

## 5. Architecture — `examples/stochastic_sppt/`

```
examples/stochastic_sppt/
  README.md          # tutorial: what SPPT is, the math above, how to run each
                     #   stage, fair-vs-almost-fair, and "adapting this to SKEB"
  config.py          # ModelConfig / ExperimentConfig / TrainConfig dataclasses
  model.py           # build_model(): pure-JAX harness + IC builder + step closure
  held_suarez.py     # pure-JAX Held-Suarez forcing -> PhysicsTendencies
  sppt.py            # SPPTParams, pattern init/step, vertical taper, apply_sppt
  rollout.py         # jax.checkpoint'd lax.scan; carry=(spec_state, r_lm); vmap members
  diagnostics.py     # spectral state -> {850 u/v, 500 T, 500 vort, ps} on T42 grid
  metrics.py         # afcrps, fair_crps, spread, rmse_of_mean, spread_error_ratio, rank_histogram
  generate_data.py   # nature run + case sampling + coarse-grain -> truth dataset (CLI; Modes A/B)
  train.py           # optax Adam, afCRPS, common random numbers per grad eval (CLI + importable main)
  analyze.py         # spread-error curves, rank hists, CRPS(lead), pattern spectrum check, maps
  slurm/
    generate_data.sbatch   # single-GPU T127 nature-run data generation
    train.sbatch           # single-GPU training
  tests/
    test_pattern.py        # reality symmetry, F0 variance, AR(1) stationarity
    test_sppt.py           # zero-amplitude identity; clip/taper behaviour
    test_training.py       # afCRPS FD-vs-autodiff grad; Mode A recovery smoke test
```

### Component contracts

- **`model.py :: build_model(config) -> Model`.** Builds the pure-JAX substrate
  once: `dyn_config, trans_config, stepper_config, latitudes, gauss_weights,
  pdryini, phis_grads`. Reuses `init_semi_implicit_matrices`,
  `init_diffusion_operators`, `prebuild_kernels`, `get_gaussian_latitudes` (the
  same setup `component_jax.py::array_call` performs, factored into a pure helper
  rather than driven through the sympl/climt object). Provides
  `initial_spectral_state(...)` and a `step(spec_state, phys_tends) ->
  spec_state` closure over `advance_with_tendencies`. Depends on: `gfs_dynamical_core.jax.*`.
- **`held_suarez.py :: hs_tendencies(grid_state, dyn_config) -> PhysicsTendencies`.**
  Pure, differentiable Held-Suarez (1994): Newtonian relaxation of `T` toward
  `T_eq(φ, p)`, Rayleigh friction on `(u, v)` below `σ_b`. Returns grid-space
  `PhysicsTendencies` (u, v, virtual_temperature, zero lnps, zero tracers).
- **`sppt.py`.** `SPPTParams`; `init_pattern(params, key, trans_config) -> r_lm`
  (stationary, reality-symmetric); `pattern_step(r_lm, key, params, trans_config,
  dt) -> r_lm` (AR(1), reality + triangular guards); `pattern_to_grid(r_lm,
  trans_config) -> r_grid` (`s2_inverse`); `vertical_taper(dyn_config) -> µ`
  (per-level, from reference mid-level pressures); `apply_sppt(phys_tends,
  r_grid, µ, params) -> PhysicsTendencies` (multiply u, v, virtual_temperature).
- **`rollout.py :: ensemble_rollout(params, spec0, keys, n_steps, model,
  lead_steps) -> diagnostics`.** Per member: `lax.scan` with `jax.checkpoint` on
  the body `carry=(spec_state, r_lm)`; each step = HS tendencies → apply SPPT →
  `model.step`; pattern advanced by `pattern_step`. Emits verification
  diagnostics at each lead. `vmap` over member `keys`. Reverse-mode safe (the
  checkpoint keeps a 5-day T42 rollout inside the memory budget).
- **`diagnostics.py`.** From a `SpectralState`: `spectral_to_grid`, pressure
  diagnostics (`compute_pressure_diagnostics`), log-linear vertical interpolation
  to 850/500 hPa (differentiable), relative vorticity at 500 hPa, `ps =
  exp(lnps)`. All on the common T42 grid.
- **`metrics.py`.** `afcrps(ensemble, truth, alpha)`, `fair_crps`, `spread`,
  `rmse_of_mean`, `spread_error_ratio` (finite-`M` corrected), `rank_histogram`.
- **`generate_data.py`.** Mode B: spin up T127 HS nature run to statistical
  equilibrium; sample `N_case` initial states; spectrally truncate each to T42
  (forecast IC) and truncate the nature-run verification states at each lead to
  T42, extracting the verification fields. Mode A: run the T42/T21 model with
  known θ\* to produce single-member truth per case. Saves an on-disk dataset
  (`.npz`/`.zarr`): per case, the T42 IC spectral state + verification fields per
  lead. CLI + `slurm/generate_data.sbatch`.
- **`train.py`.** optax Adam over `SPPTParams`. Each step: batch of cases → `vmap`
  M members (each: `init_pattern` from key, checkpointed rollout, diagnostics at
  leads) → `afcrps` averaged over cases/leads/fields with per-variable + pressure
  weighting → `grad` wrt params → update. **Common random numbers:** member keys
  derived deterministically from `(base_key, case, member)` so they are identical
  within a gradient evaluation (per research doc §3.3.4). Checkpoints params;
  logs afCRPS, spread–error ratio. CLI + `slurm/train.sbatch`.
- **`analyze.py`.** Loads trained θ + held-out cases → spread–error ratio vs lead
  per field, rank histograms, afCRPS/fair-CRPS vs lead, pattern power spectrum vs
  the target Gaussian spectrum, example spread maps. Saves PNGs + a summary JSON.

---

## 6. Data flow

```
generate_data.py ─► truth dataset (per case: T42 IC spectral state + verification
                    fields {850u,850v,500T,500vort,ps} at leads {3,5,7,10 d})
        │
        ▼
train.py: for each opt step:
    sample batch of cases
    for each case:  keys = split(base_key, case)          # common random numbers
        ensemble = vmap_member( rollout(params, IC, key) )  # checkpointed lax.scan
    loss = mean_cases,leads,fields( weighted afCRPS(ensemble_field, truth_field) )
    params ← optax_update(params, grad(loss))
        │
        ▼
analyze.py: trained params + held-out cases ► calibration metrics + plots
```

---

## 7. Hard invariants (do not skip)

- Every spectral noise field passes `assert_reality_symmetric` +
  `enforce_triangular_truncation`.
- AR(1) pattern state (`r_lm`) lives in the `scan` carry, never global.
- Reverse-mode rollouts wrap the step body in `jax.checkpoint`.
- Common random numbers fixed within each gradient evaluation.
- `σ → 0` reduces the pipeline to the deterministic HS+dynamics step (unit test).

---

## 8. Milestones & acceptance

- **M1 — Mode A, laptop T21.** Training recovers known θ\* within tolerance; all
  unit tests green (reality symmetry, `F0` variance ≈ σ², AR(1) stationarity,
  afCRPS FD-vs-autodiff grad, zero-amplitude identity).
- **M2 — Mode B, cluster T42/T127.** After afCRPS training, **spread–error ratio
  → ≈1 at 5-day lead** across the verification fields; rank histograms flatten;
  spread demonstrably grows relative to the deterministic ensemble.

---

## 9. Extension seams (documented, not built — YAGNI)

- **SKEB (WP6):** reuse `sppt.py`'s spectral generator; inject additively via
  `spec_tends.d_vorticity_d_t`; modulate by `sqrt(dissipation)`; trainable
  backscatter amplitude/spectrum.
- SP2 two-scale pattern; per-wavenumber τ; NN amplitude field σ(x) (toward
  latent-noise, WP6); IC-perturbation ensemble seeding; ERA5 truth (C-WP3);
  multi-GPU member sharding (`shard_map`).

---

## 10. Risks / open items

- **T127 nature-run cost/storage** — cluster-side data generation (`sbatch`);
  store only what verification needs (truncated fields per lead), not full T127
  trajectories.
- **HS model-error gap may be modest** — HS "physics" is only relaxation +
  friction; Mode A de-risks the machinery independently of the gap size.
- **Differentiability through `spectral_to_grid` + vertical interpolation** —
  covered by the afCRPS FD-vs-autodiff test.
- **Reality-symmetric complex sampling** must match this codebase's spectral
  layout/normalization exactly (reuse validated utilities; prior bug precedent).
- **`set_physics_tendencies` virtual-T caveat** — avoided: we use the grid
  `PhysicsTendencies` (virtual_temperature) path, consistent with the component path.
