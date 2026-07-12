# Differentiable Hybrid Modeling Research Program — Four Tracks
## Design + implementation reference (self-contained)

**Status:** Strategy / design / implementation reference. Untracked working document — not part of the
repo history.
**Date:** 2026-06-12 (v2: fully self-contained; incorporates the extremes handoff verbatim as §3 and
expands all other tracks to design+implementation level).
**Intended use:** hand this to any capable implementer (human or model) and get faithful adherence.
Every track section contains: motivation, mathematical formulation, implementation notes with code
sketches against this repo's API, work packages with acceptance criteria, and risks. Where a choice
was made, it is recorded in the decision log (§8) with rationale — implementers should not silently
revisit logged decisions.

**Repo context:** `gfs_dynamical_core` on branch `jax-port`. Key existing assets:

- `gfs_dynamical_core/jax/stepper.py` — `advance(...)`: jitted pure spectral step
  (SpectralState → SpectralState).
- `gfs_dynamical_core/component_jax.py` — `GFSDynamicsJAX` (sympl `Stepper`), with the imperative
  `set_physics_tendencies` side-channel.
- `gfs_dynamical_core/component.py` — `GFSDynamicalCore` (Fortran-backed `TendencyStepper`), the
  parity reference for the component API.
- `docs/specs/jax_component_api_parity/` — approved design + implementation plan for the
  component-driven JAX API and the pure `advance_with_tendencies` function (currently deferred;
  promoted to keystone here, see C-WP1).
- `examples/ml_physics_emulator_jax.py` — working demonstration of online training (MLP physics
  emulator vs Held–Suarez through `jax.lax.scan` rollouts); critically assessed in §3.1.
- Bit-level Fortran parity tests (`tests/test_jax_*.py`); the reality-symmetry guard
  (`assert_reality_symmetric`) for spectral fields.
- A differentiable convective parameterization (separate codebase, referenced throughout as "the
  convection scheme").

---

## 0. Strategic frame (vs NeuralGCM as baseline)

The AI-weather space is crowded at exactly one point: deterministic/probabilistic medium-range skill
scores on WeatherBench-style benchmarks. That contest is compute-dominated; NeuralGCM-class training is
of order 10 days–weeks on 10²-scale accelerators per run, and Google/ECMWF/Huawei/Nvidia will outspend
any academic group. It is nearly empty at:

1. **Differentiable operational-lineage cores.** NeuralGCM's Dinosaur dycore is bespoke; ours is
   numerics-faithful to the GFS spectral core with bit-level Fortran parity tests. For calibration and
   operational credibility, "differentiable version of *the* model" beats "differentiable model".
2. **Component-resolved hybrids.** NeuralGCM's learned physics is one monolithic column NN — you cannot
   swap one parameterization, keep another physical, or attribute behavior to a component. Our
   climt/sympl `TendencyComponent` contract makes physical / learned / stochastic components
   interchangeable, unit-checked, and individually attributable.
3. **Gradient-based calibration of real parameterization suites** (Track 1) and
4. **Adjoint / rare-event methods for extremes** (Track 2).

Costs we accept in exchange: a dual API (sympl host-side composition path + pure jnp `advance` path)
maintained forever with equivalence tests, and legacy numerics choices (sigma coordinates, GFS
semi-implicit step) that the ML does not get a vote on.

**Positioning sentence:** *the platform where physical and learned components are interchangeable,
conserving, and differentiable* — calibration and extremes are the flagship science; a T62–T126
stochastic hybrid ensemble is the credibility demonstration, not the end goal.

### The four tracks

| # | Track | Product | Section |
|---|---|---|---|
| 1 | Calibration & adjoint applications | Calibrated physics suite; exact-adjoint 4D-Var & sensitivity demos | §2 |
| 2 | Extremes (heatwaves, extreme rainfall) | RES pipeline, instantons, calibrated tails | §3 (handoff, verbatim) |
| 3 | Probabilistic weather ensemble | CRPS-trained stochastic hybrid ensemble with attributable spread | §4 |
| 4 | Climate emulation | Stable AMIP hybrid with conservation by construction | §5 |

Work-package numbering: **C-WP\*** = common infrastructure (§1); **H-WP0–13** = the extremes handoff
roadmap (§3.7); **1a-\*, 1b-\*, 3-\*, 4-\*** = per-track packages.

---

## 1. Common infrastructure (the keystone work packages)

Everything below routes through these. They are shared, so they are specified once and referenced by
all tracks.

### C-WP1 — Component-API parity + `advance_with_tendencies` (the keystone)

The approved spec in `docs/specs/jax_component_api_parity/` (design + implementation plan). Delivers:

- `GFSDynamicsJAX(tendency_component_list=[...])` as a drop-in `TendencyStepper` replacement for the
  Fortran core: virtual-temperature correction
  (`virtual_temp_tend = T_tend·(1+fvirt·q) + fvirt·t_virt·q_tend`, `fvirt = (1−Rd/Rv)/(Rd/Rv)`),
  log-surface-pressure tendency (`lnps_tend = ps_tend/ps`), sympl tracer pack/unpack
  (`register_tracer`/`uses_tracers`), multi-tracer vmap through dynamics and transforms,
  `air_pressure` / `air_pressure_on_interface_levels` outputs (with the BTT↔TTB interface-level flip),
  and `zero_negative_moisture` post-step clipping — all per the spec.
- `advance_with_tendencies(spec_state, grid_tends, dyn_config, trans_config, stepper_config,
  latitudes, gauss_weights, pdryini) -> SpectralState` — module-level pure jnp function,
  `jit`/`grad`/`vmap`-able, no sympl/numpy in the path. `grid_tends` is a flax-struct dataclass with
  optional u, v, virtual-T, lnps, and tracer tendency fields. The component path and the manual
  `set_physics_tendencies` path become thin wrappers over it.
- Backward-compatibility hard requirement: `GFSDynamicsJAX()` with no arguments is bit-for-bit
  identical to today; all existing `tests/test_jax_*.py` pass unchanged.

**Every track needs this.** Track 1 needs the pure function for gradients and the component path for
composing physical schemes. Track 2's stochastic perturbations and RES walkers are components on the
outside and pure-function calls on the inside. Tracks 3–4 are built entirely from components.
**Decision (D-0): promoted from "deferred" to milestone 1 of whichever track starts first.**

### C-WP2 — Parallelism strategy (ensemble-first, model-sharding later)

Two distinct needs, in order of urgency:

**Mode 1 — batch/ensemble parallelism (now).** Tracks 1–3 at T62–T126 are dominated by *many
independent rollouts*: vmapped initial conditions (calibration batches), ensemble members (CRPS
training, EKI), RES walkers. This is embarrassingly parallel and requires no model-parallel sharding:

```python
# Within one device: vmap over the batch/member axis of stacked SpectralState pytrees.
batched_rollout = jax.vmap(rollout, in_axes=(None, 0, None))     # (params, ic_batch, n_steps)

# Across devices: shard the member axis. Prefer shard_map over pmap for new code.
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P
mesh = Mesh(jax.devices(), axis_names=("member",))
sharded_rollout = shard_map(batched_rollout, mesh=mesh,
                            in_specs=(P(), P("member"), P()), out_specs=P("member"))
```

Gradient aggregation across shards (for ensemble losses) uses `jax.lax.pmean` inside the
shard-mapped loss. This mode covers the entire program up to ~T126.

**Mode 2 — model sharding (later, gating >T126 / memory-bound configs).** Spectral transforms sharded
across devices: latitude-band parallelism for the FFTs, zonal-wavenumber (m) parallelism for the
Legendre transforms, with an all-to-all between them (the classical spectral-model transpose). This is
the single hardest engineering item and the main justification for a research-engineer hire.
**Decision (D-1): do not start Mode 2 before a Mode-1 track is producing science.**

**Known hot loop** (from §3.1.3): `apply_physics` projects grid tendencies to spectral space every
step (three `s2_forward` calls + truncation). Before any scaling work: batch the three transforms into
one stacked call and benchmark `vmap` vs explicit batching in s2fft; profile at T62 and T126.

**Memory budget arithmetic (needed by every reverse-mode workload).** A T62 grid state
(192×94 lon×lat, 28 levels, 5 prognostic fields, float64) is ≈ 20 MB; the spectral state is smaller.
Reverse-mode through an N-step `lax.scan` stores O(N) intermediates: a 5-day rollout at Δt = 900 s is
480 steps ≈ 10 GB of saved residuals per trajectory — already marginal, and fatal with ensemble
members. Therefore **every scanned rollout that will be differentiated must wrap the step in
`jax.checkpoint` (rematerialization)**:

```python
@jax.checkpoint        # store only the carry at step boundaries; recompute internals on backward
def ckpt_step(state, _):
    return advance_with_tendencies(state, tends_fn(state), *configs), None

final, _ = jax.lax.scan(ckpt_step, state0, None, length=n_steps)
```

For long horizons use two-level (√N) checkpointing: an outer scan over √N segments, each segment an
inner checkpointed scan — memory O(√N · state) ≈ 0.5 GB for the 480-step case. Implement this once as
`rollout(params, state0, n_steps, remat="sqrt")` in a shared module; all tracks use it.

### C-WP3 — Data pipeline (ERA5 ↔ model state, verification stores)

**Ingest chain (ERA5 → `SpectralState`):**

1. Download ERA5 model-level or pressure-level fields (u, v, T, q, ps, plus z_sfc once) for the
   chosen years; 0.25° native.
2. Horizontal regrid to the target Gaussian grid (T62 → 192×94; T126 → 384×190): bilinear for state
   variables (conservative remapping is only needed for fluxes/precip verification fields). Tooling:
   `xesmf` offline, weights cached.
3. Vertical interpolation to the core's sigma levels: σ = p/ps with the *regridded* ps; interpolate
   in ln p; extrapolate below ground following the standard lapse-rate rule for T, constant for q
   (clip q ≥ 0).
4. Derive spectral prognostics: (u, v) → (vorticity, divergence) via the repo's own transforms
   (`grid_to_spectral`, `s2_forward`) — never a foreign transform library, to avoid normalization
   mismatches (cf. memory: cross-library comparisons only in grid space); T → virtual T as the core
   expects; ln ps. Apply `enforce_triangular_truncation` and the reality-symmetry guard.
5. Store stacked pytrees (a leading sample axis over init times) as Zarr, chunked by sample;
   metadata: ERA5 version, regrid weights hash, level definition, code commit.

**Verification stores:** ERA5 itself (state verification, WeatherBench2-regridded versions where
comparing to published scores); IMERG half-hourly precipitation (+ gauge-corrected products over land
for Track 2's tail work, with a per-region observation-error term); WeatherBench2 archived baseline
forecasts (IFS ENS, NeuralGCM-ENS, GenCast) for Track 3 §4.5.

**Splits — global rule:** train/val/test by **year blocks** (e.g., train ≤2017, val 2018–2019, test
2020–2021), never random days: synoptic autocorrelation (~5–10 d) leaks across random splits and
inflates skill. Test years untouched until a track's final evaluation.

**Storage:** of order 10–50 TB for a multi-year subset at training resolution; Zarr on object storage
or parallel FS; never store what can be recomputed from the spectral state.

### C-WP4 — Shared physical components

Built once as sympl `TendencyComponent`s (host path) + pure jnp kernels (training path), used by
multiple tracks:

- **Moisture + precipitation channels** in the rollout (= H-WP1, §3.7): `dq/dt` with a conserving
  projection (or flux form) and a surface rain-rate diagnostic carried in `traj`. Acceptance: closed
  column moisture budget to FP tolerance over a 5-day rollout.
- **Differentiable prognostic land bucket** (= H-WP7): single-layer soil moisture W with
  `dW/dt = P − E − R`, evaporative fraction `EF = β(W) = clip(W/W_crit, 0, 1)` modulating surface
  evaporation and the Bowen ratio; 1–3 trainable parameters (W_crit, runoff exponent, depth);
  optional small stochastic term (§3.3.3 item 4). ~10 lines of JAX; needed by Tracks 2, 3, 4.
- **Stochastic-physics component family** (= H-WP5/6, full design in §3.3): trainable SPPT
  (spectral AR(1) pattern; trainable σ, τ, L̃), latent-noise NN inputs, SKEB-like vorticity
  perturbation. All spectral noise passes the reality-symmetry guard.
- **Pluggable heat-stress index module** (= H-WP8, §3.4.1): T2m, heat index, UTCI polynomial,
  Stull-WBGT as pure differentiable functions; implicit-diff hook (`lax.custom_root` /
  `jaxopt.ImplicitDiff`) for iterative indices. Track 2 primary; Track 4 reuses for climate-mode
  heat-extreme statistics.

---

## 2. Track 1 — Calibration & adjoint applications

### 2.1 Why this is the strongest card

Parameterization tuning at operational centers is a person-decades, largely manual activity; a
differentiable core with *physical* parameterizations attached turns it into gradient descent.
CliMA does ensemble-based calibration for their own new model; **nobody has gradient calibration for an
operational-lineage core.** Separately, every center that runs 4D-Var maintains hand-coded
tangent-linear/adjoint models that perpetually lag the nonlinear model — exact autodiff adjoints
eliminate that class of problem. Both stories are low-compute (single node), high-credibility, and the
Fortran parity work already done is precisely what makes the results trustworthy.

Two sub-tracks sharing infrastructure, different audiences: **1a — parameter calibration**
(model-development audience) and **1b — adjoint applications** (data-assimilation audience).

### 2.2 Identifiability analysis: FIM, sloppiness, MBAM (= H-WP0, expanded)

This is the gate on all calibration. Naive gradient descent on all parameters of a convection scheme
fails predictably: the loss surface has long flat valleys (parameter compensation), and the optimizer
returns whichever point on the valley floor it wandered to — right answer, wrong reason, zero
generalization. The cure is to measure, before training, *which parameter combinations the data can
constrain at all*.

#### 2.2.1 Formulation

Let the column scheme be a pure function `g(θ, s) → y` mapping parameters θ ∈ R^p and a column state
s to outputs y ∈ R^d (here: dT/dt at L levels, dq/dt at L levels, surface precipitation rate;
d = 2L + 1). Work in **unconstrained transformed parameters** φ = T(θ): `log` for positive
parameters, `logit((θ−a)/(b−a))` for parameters bounded in [a, b]. This makes sensitivities
scale-free (∂/∂φ of a log-parameter is the *relative* sensitivity) and lets every downstream
optimizer run unconstrained.

Scale outputs by per-channel structural-error magnitudes D = diag(σ_dT, σ_dq, σ_P) — defensible
defaults: σ_dT = 1 K day⁻¹ per level, σ_dq = 0.5 g kg⁻¹ day⁻¹ per level, σ_P = 5 mm day⁻¹. (These
play the role of observation/structural error; conclusions should be checked for robustness to ±3×
changes in them.)

For a sample of M column states {s_m} the (Gauss–Newton approximation to the) **Fisher Information
Matrix** is

```
J_m = ∂[D⁻¹ g(φ, s_m)]/∂φ   ∈ R^{d×p}          (per-regime sensitivity)
F   = Σ_m J_mᵀ J_m           ∈ R^{p×p}          (pooled FIM)
F_r = Σ_{m∈regime r} J_mᵀ J_m                    (per-regime FIM)
```

Eigendecompose F = V Λ Vᵀ with λ_1 ≥ … ≥ λ_p. Diagnostics, in order of importance:

1. **Spectrum shape.** Sloppy models show eigenvalues spanning many decades, roughly log-uniform.
   Count the *effective number of constrainable directions*
   `p_eff = #{i : λ_i > λ_1 · ε}` with ε = 10⁻⁶ (report the curve p_eff(ε), not one number).
   Expect p_eff ≪ p; this is the honest dimensionality of the calibration problem.
2. **Stiff subspace.** Columns of V for the top-p_eff eigenvalues span the constrainable subspace.
   Per-parameter **participation** `π_i = Σ_{k ≤ p_eff} V_{ik}²` ∈ [0, 1] measures how much of
   parameter i lies in the stiff subspace.
3. **Regime dependence.** Compute F_r for r ∈ {tropical-ocean high-CAPE, tropical-land,
   midlatitude-summer continental, midlatitude-marine, trade-cumulus} columns (sampled from ERA5,
   stratified by CAPE/CWV/latitude/land-sea). Compare stiff subspaces across regimes via principal
   angles between span(V_r[:, :p_eff]); parameters stiff in one regime and sloppy in another are
   prime candidates for the *stochastic/regime-dependent* treatment of §3.3.2 (parameter noise with
   state-dependent μ(x), σ(x)).

**Selection rule (the deliverable):** parameter i enters the trainable set iff π_i ≥ 0.1 in the
pooled FIM *and* in at least one regime FIM. All others are frozen at documented defaults, or kept
with a prior precision dominating the data information (equivalent in effect, cleaner for posterior
reporting). Additionally report the top-3 stiff eigenvectors *as parameter combinations* — these,
not individual parameters, are what the data constrains, and the calibration paper must present
posterior uncertainty in this basis.

#### 2.2.2 MBAM (optional level-2 analysis)

The Model Boundary Approximation Method (Transtrum et al.) rides geodesics of the model manifold
(metric F(φ)) starting along the *sloppiest* eigendirection; the geodesic typically runs to a manifold
boundary in finite "time", and the boundary identifies a reduced model (some parameter combination →
0 or ∞, e.g. "entrainment-dominated limit"). Use when the level-1 analysis leaves doubt about which
limit a sloppy direction represents. Cost: the geodesic ODE needs directional second derivatives —
`jax.jvp` of the Jacobian function gives them — but each step is ~p× the FIM cost. Mark as optional;
do not block calibration on it.

#### 2.2.3 Implementation sketch

```python
# Pure column scheme: g(theta, column) -> dict(dT=..., dq=..., precip=...)
SCALES = dict(dT=1.0/86400, dq=0.5e-3/86400, precip=5.0/86400)   # per-channel structural errors

def transform(theta_raw):          # physical -> unconstrained
    return {k: tfm[k](v) for k, v in theta_raw.items()}           # log / logit per parameter

def scaled_outputs(phi, column):
    theta = untransform(phi)
    out = convection_scheme(theta, column)
    return jnp.concatenate([(out["dT"] / SCALES["dT"]).ravel(),
                            (out["dq"] / SCALES["dq"]).ravel(),
                            jnp.atleast_1d(out["precip"] / SCALES["precip"])])

def fim(phi, columns):                                            # columns: stacked pytree, M samples
    Js = jax.vmap(lambda c: jax.jacfwd(scaled_outputs)(phi, c))(columns)   # (M, d, p); jacfwd: p small
    J = Js.reshape(-1, Js.shape[-1])
    return J.T @ J

lam, V = jnp.linalg.eigh(fim(phi_default, regime_columns))        # per regime and pooled
```

Notes: `jacfwd` because p ≲ 30 ≪ d; columns sampled from the C-WP3 pipeline with the §2.2.1 regime
stratification, M ≈ 200–1000 per regime; if the scheme has hard triggers, run this analysis with the
*smoothed* triggers of H-WP4 at the training temperature, since that is the object being calibrated.

**Acceptance criteria (H-WP0, sharpened):** eigenvalue spectra (pooled + ≥4 regimes) with p_eff(ε)
curves; participation table π_i; principal-angle matrix between regime stiff subspaces; a decisions
table (trainable / frozen / stochastic-candidate per parameter) with one-line justifications; all
robust to ±3× changes in D.

### 2.3 Track 1a: online calibration of physical parameterizations

#### 2.3.1 Setup

The differentiable convection scheme (plus other physical components as they exist: simple radiation,
boundary layer, the C-WP4 land bucket) runs inside the rollout via the component path; trainables are
(i) the H-WP0-selected physical parameters φ and (ii) optionally a small NN residual r_ψ(x) added to
the physics tendencies. Forecast mode throughout: initialize from ERA5 (C-WP3 batches), verify at
1–5 day leads. §3.2 (rainfall-targeted losses, stratified ICs, smoothed triggers) is the
extreme-rainfall *instance* of this sub-track; here the general protocol.

#### 2.3.2 Loss

Lead-time-weighted composite over a batch of B rollouts:

```
L(φ, ψ) = Σ_leads w_ℓ · [ L_state(ℓ) + Σ_j w_j L_process_j(ℓ) ]  +  L_prior(φ)  +  λ_ψ ‖ψ‖²
```

- `L_state(ℓ)`: per-variable scaled MSE vs ERA5 at lead ℓ ∈ {24, 48, 72, 120} h, with the standard
  per-level/per-variable inverse-variance scaling (climatological variance of ERA5 anomalies);
  w_ℓ ∝ 1/ℓ so short leads (healthier gradients) dominate early training. This is the anchor.
- `L_process_j`: what is actually being calibrated — by application: precipitation tail terms
  (pinball + exceedance, exactly §3.2.3), diurnal-cycle phase of land precipitation (loss on the
  first diurnal harmonic's phase/amplitude vs IMERG), CAPE-consumption timescale, MSE-budget
  residuals.
- `L_prior(φ) = ½ (φ − φ_0)ᵀ Σ_prior⁻¹ (φ − φ_0)` in transformed space, centered on documented
  defaults, widths from literature plausible ranges (a factor-of-3 range for a log parameter ⇒
  σ = ln 3 ≈ 1.1). Frozen parameters: prior precision → ∞ (i.e., actually frozen).

#### 2.3.3 Gradient hygiene (mandatory, shared with §3.2.4)

Gradients through chaotic dynamics grow ~e^{λt} with error-doubling ≈ 1.5–2 d in the midlatitude
troposphere; beyond ~5–7 d single-trajectory gradients are direction-corrupted even when finite.
Protocol:

1. **Global-norm clipping** at 1.0 (in the transformed/normalized parameter space).
2. **Curriculum on horizon:** train to convergence at 6 h, then warm-start 24 h → 72 h → 120 h.
   Monitor per-horizon gradient norms; the horizon where median norm grows ≳10× between curriculum
   stages is the truncation point.
3. **Truncated BPTT with windows** beyond the truncation point:

```python
def windowed_loss(params, ic, obs_windows, K):
    """Gradient flows within each K-step window; truncated at window boundaries."""
    state, total = ic, 0.0
    for w, obs_w in enumerate(obs_windows):              # unrolled python loop inside jit
        state, traj = rollout(params, state, K)          # rollout = C-WP2 checkpointed scan
        total += window_loss(traj, obs_w)
        state = jax.lax.stop_gradient(state)             # the truncation
    return total / len(obs_windows)
```

   Overlap windows by K/2 (stride K/2, halve the weight of doubly-counted steps) so no lead time sits
   permanently next to a truncation boundary.
4. **Smoothed triggers** (= H-WP4): every hard threshold in the scheme replaced during training by a
   sigmoid with temperature annealed downward (and Gumbel-softmax for discrete plume choices);
   optionally re-hardened at inference, with a re-validation pass quantifying the hardening gap.

#### 2.3.4 Ensemble Kalman Inversion (the gradient-free co-method)

**Decision (D-4): EKI is a first-class citizen, not a fallback of last resort.** It is robust to
chaotic-gradient corruption and non-smooth triggers, embarrassingly parallel (C-WP2 Mode 1), and the
gradient-vs-EKI comparison on identical problems is itself a publishable methods result.

Iglesias–Law–Stuart iteration, J ensemble members θ_j (in transformed space φ_j), forward map
G(φ) = the full rollout → stacked verification vector y (same content as the loss's data terms):

```python
def eki_step(phis, y_obs, Gamma, key):                   # phis: (J, p)
    G = parallel_forward(phis)                           # (J, n_obs); vmap/shard_map over members
    pm, Gm = phis.mean(0), G.mean(0)
    C_pg = (phis - pm).T @ (G - Gm) / (J - 1)            # (p, n_obs)
    C_gg = (G - Gm).T @ (G - Gm) / (J - 1)               # (n_obs, n_obs)
    noise = jax.random.multivariate_normal(key, jnp.zeros_like(y_obs), Gamma, (J,))
    increments = jnp.linalg.solve(C_gg + Gamma, (y_obs + noise - G).T)    # (n_obs, J)
    return phis + (C_pg @ increments).T
```

Practicalities: J = 50–100 (p ≲ 30 ⇒ no localization needed); Γ from the same structural-error
scales as §2.2.1; stop by the discrepancy principle (‖y − Ḡ‖²_Γ ≈ n_obs) — EKI iterated past
discrepancy overfits and collapses; mild multiplicative inflation (1.01) if collapse appears early.
For n_obs large, replace the dense solve with the (J−1)-rank subspace form (work in ensemble space).

**Comparison protocol (1a-4):** identical loss content, identical train years; report (i) wall-clock
and rollout-count to discrepancy/convergence, (ii) calibrated values and posterior spread (EKI
ensemble spread vs Laplace approximation `Σ_post ≈ (F_loss + Σ_prior⁻¹)⁻¹` at the gradient optimum),
(iii) val-year skill. Hypothesis to test: gradients win at ≤72 h horizons, EKI at longer horizons
and with hardened triggers.

#### 2.3.5 Validation

Held-out year blocks; process-oriented diagnostics (not just RMSE): diurnal cycle of land
precipitation, intensity distributions (PDF overlap + tail quantiles), MSE budget closure; ablations:
default-parameters baseline, EKI vs gradient, residual-NN-off (to show how much the *physical*
recalibration alone does — this is the headline ablation).

#### 2.3.6 Work packages

| # | Package | Depends on | Acceptance criterion |
|---|---|---|---|
| 1a-1 | Physical-suite-in-the-loop: convection (+ land bucket) running via component path inside scanned rollouts | C-WP1, C-WP4 | One-step output matches offline scheme call to FP tolerance; rollout jits; checkpointed reverse-mode runs at 120 h |
| 1a-2 | Parameter infrastructure: transforms, priors, trainable-subset masking from H-WP0 | §2.2 | Gradients of 24 h loss w.r.t. every trainable parameter finite and FD-validated (relative error < 10⁻⁴ at ε = 10⁻⁶ central differences in float64) |
| 1a-3 | Forecast-mode calibration harness (vmapped ERA5 ICs, lead-weighted loss, curriculum, truncated BPTT) | C-WP3, 1a-2 | Stable training to 120 h; val-year L_state ≤ default-parameter baseline |
| 1a-4 | EKI baseline on identical problem + comparison protocol | 1a-3 | Comparison report per §2.3.4 (iii criteria) |
| 1a-5 | Headline experiment: calibrated suite vs defaults on held-out years, process diagnostics, posterior in stiff basis | 1a-3, 1a-4 | Improvement on ≥2 process diagnostics without mean-state degradation; posterior reported in stiff-eigenvector basis |

**Risks.** Chaotic gradient corruption beyond ~5–7 d (mitigated: curriculum + truncation + EKI);
parameter compensation producing right-answer-wrong-reason fits (mitigated: H-WP0 subspace reporting +
process diagnostics); structural biases of the sigma-coordinate legacy core that no parameter setting
can fix (mitigated: the residual NN term, and honesty about it in the paper).

### 2.4 Track 1b: exact-adjoint applications

**The pitch.** Hand-coded adjoints are the single largest software liability in variational DA: they
lag the forward model, drop physics ("simplified-physics" adjoints), and consume specialist
person-years. `jax.vjp` of the parity-validated core gives the *exact* adjoint of the *actual* model,
including through physics components, at zero maintenance cost.

#### 2.4.1 1b-i: Incremental 4D-Var OSSE

Fully self-contained Observing System Simulation Experiment:

1. **Nature run:** long trajectory of the (hybrid or default-physics) model at T62; this is "truth".
2. **Synthetic observations:** y_k = H_k(x_true(t_k)) + ε_k, ε_k ~ N(0, R). Observation operators in
   increasing order of realism: point values of u, v, T at random (lat, lon, level) sites
   (radiosonde-like, 00/12 UTC); column-integrated water (GPS-like); vertically weighted T integrals
   (radiance-like, fixed weighting functions). All are a few lines of differentiable JAX
   (interpolation + weighted sums).
3. **Cost function** over an assimilation window [0, T_w] (T_w = 6–12 h), preconditioned via the
   change of variable δx₀ = B^{1/2} χ (so the background term is ½‖χ‖² and conditioning is set by
   R and the dynamics):

```python
def fourdvar_cost(chi, x_b, obs_seq, B_sqrt_apply, configs):
    x0 = x_b + B_sqrt_apply(chi)                       # spectral-space preconditioning
    def step(x, obs_k):
        x = ckpt_advance(x, configs)                   # jax.checkpoint-wrapped model step(s)
        d = obs_k.y - apply_H(x, obs_k.meta)
        return x, 0.5 * d @ obs_k.Rinv @ d
    _, Jo = jax.lax.scan(step, x0, obs_seq)
    return 0.5 * jnp.vdot(chi, chi).real + Jo.sum()

# Minimize with L-BFGS on jax.value_and_grad(fourdvar_cost) — optax.lbfgs or jaxopt.LBFGS.
```

   Note this is *strong-constraint 4D-Var on the nonlinear trajectory*: because `jax.grad` of the
   full nonlinear rollout is available, the classical incremental (inner/outer loop, tangent-linear
   inner model) structure is optional. Implement the direct nonlinear form first; add an incremental
   variant only if minimization stalls (it then serves as a Gauss–Newton globalization).
4. **B matrix (NMC method):** collect differences of 48 h and 24 h forecasts valid at the same time
   over ≥1 season of the nature run; in spectral space, fit variance per total wavenumber n, per
   level, per variable, plus a vertical correlation matrix per n (or one fixed vertical correlation).
   B^{1/2} then acts diagonally in (n, variable) with a vertical Cholesky factor — cheap, standard,
   and adequate for an OSSE.
5. **Cycling:** 6 h windows over ≥2 weeks; analysis at window end propagates to the next background.
6. **Baselines:** 3D-Var (same B, observations lumped at analysis time) — the floor; and **4D-Var
   with a deliberately degraded adjoint** (physics tendencies detached from the backward pass via
   `stop_gradient` — a one-line "simplified-physics adjoint") — this ablation quantifies exactly what
   adjoint exactness buys, and is the paper's headline figure.

**Acceptance:** cycled analysis RMSE (vs nature run) of 4D-Var < 3D-Var by a margin exceeding cycle
variability; innovation statistics χ² consistent with prescribed R; degraded-adjoint gap quantified.

#### 2.4.2 1b-ii: Singular vectors / optimal perturbations

Leading singular vectors of the tangent-linear propagator **M** over an optimization interval
[0, t_opt] (t_opt = 48 h), under the total-energy norm

```
⟨x, x⟩_E = ½ ∫∫ [ u′² + v′² + (c_p/T_r) T′² ] dp/g dA  +  ½ ∫ R_d T_r p_r (ln p_s′)² /g dA ,
T_r = 300 K, p_r = 1000 hPa.
```

Maximizing ⟨**M**x, **M**x⟩_E / ⟨x, x⟩_E is the symmetric eigenproblem `A z = σ² z` with
`A = E^{-1/2,T} Mᵀ E M E^{-1/2}` and z = E^{1/2} x. Everything is matrix-free:

```python
def M_jvp(dx):   # tangent-linear: jvp through the (checkpointed) rollout at the linearization state
    _, dout = jax.jvp(lambda x: rollout_traj_end(x), (x_lin,), (dx,))
    return dout

def MT_vjp(dy):  # adjoint
    _, vjp_fn = jax.vjp(lambda x: rollout_traj_end(x), x_lin)
    return vjp_fn(dy)[0]

def A_matvec(z):
    x  = apply_E_invhalf(z)
    return apply_E_half_T(MT_vjp(apply_E(M_jvp(x))))   # symmetric PSD by construction
```

Top-k eigenpairs via Lanczos: either `jax.experimental.sparse.linalg.lobpcg_standard`, or (simplest,
**decision D-8**) SciPy's `eigsh` with a host-callback matvec — the outer Krylov loop is host-side
Python and only the matvec is device code; k ≈ 10, ~50–100 matvecs, each matvec = one TL + one AD
rollout. Validate: (i) σ² from the solver matches the Rayleigh quotient of the returned vectors;
(ii) nonlinear growth of ±ε·SV1 brackets the linear prediction for small ε; (iii) structures are
baroclinically tilted, lower-tropospheric, upstream of the verification region — the classical
signatures.

Feeds: Track 3 IC perturbations (§4.2a), Track 2 adjoint features and steering directions
(H-WP11/12).

#### 2.4.3 1b-iii: Forecast sensitivity / case-study attribution

`∂(scalar forecast metric)/∂(initial state)` for real cases by a single `jax.grad` through the
rollout — e.g., a monsoon-depression rainfall metric's sensitivity to upstream low-level moisture.
Cheap (one backward pass per case), fast papers, demonstrates the tool to the synoptic community.
Implementation is a trivial specialization of the §2.4.2 machinery (one vjp, no eigensolve).

#### 2.4.4 Work packages

| # | Package | Depends on | Acceptance criterion |
|---|---|---|---|
| 1b-1 | Observation operators + OSSE harness (nature run, synthetic obs, R; NMC B-matrix) | C-WP1, C-WP3 | Innovation χ² statistics consistent with prescribed R; B reproduces forecast-difference spectra |
| 1b-2 | 4D-Var minimization (preconditioned L-BFGS on jax.grad; checkpointed window) + cycling | 1b-1 | Cycled analysis RMSE < 3D-Var baseline; minimization to gradient-norm tolerance in < 100 iterations |
| 1b-3 | Matrix-free SV solver (Lanczos/eigsh on jvp/vjp pairs, energy norm) | C-WP1 | Validations (i)–(iii) of §2.4.2 pass; growth rates stable to solver tolerance |
| 1b-4 | Degraded-adjoint ablation + paper experiment | 1b-2 | Quantified analysis/forecast gain of exact vs simplified adjoint over ≥2-week cycling |

**Compute & memory:** reverse-mode through a 6–12 h window at T62 is single-GPU with C-WP2
checkpointing. The entire sub-track never needs more than one node.

---

## 3. Track 2 — Differentiable Hybrid Modeling for Weather Extremes
### Forecast-mode calibration, stochastic physics, and rare-event sampling for heatwaves and extreme rainfall

*(This section is the extremes handoff document, incorporated verbatim with renumbered headings;
its work packages are labeled H-WP0–13 and are referenced throughout this document.)*

**Status:** Design / handoff for implementation.
**Context:** We have (a) a differentiable convective parameterization, and (b) a differentiable JAX
port of the GFS spectral dynamical core (`gfs_dynamical_core.jax`), demonstrated in
`ml_physics_emulator_jax.py` (online training of an MLP physics emulator against Held–Suarez through
`jax.lax.scan` rollouts). Target applications: (1) extreme rainfall representation, (2) stochastic
parameterization for phase-space exploration, (3) rare event sampling (RES) for heatwaves defined by
an arbitrary heat-stress index (T2m, heat index, UTCI, WBGT), in the spirit of AI+RES (Lancelin et
al. 2025, arXiv:2510.27066).
**Mode:** Forecast mode (initialized from analyses, verified against observations at 1–15 day leads),
NOT free-running climate mode. This choice governs loss design, gradient health, and data strategy
throughout.

### 3.1 Critical assessment of `ml_physics_emulator_jax.py`

#### 3.1.1 Keep (the skeleton is right)

- **Online training architecture.** Trainable physics inside `jax.lax.scan`, `jax.value_and_grad` of
  a trajectory loss through N calls to `advance`. This is the NeuralGCM design pattern and the
  correct foundation.
- **Reality-symmetry guard** (`assert_reality_symmetric`). Hard-won invariant checking; keep it on
  every new spectral pathway, including stochastic perturbation fields generated in spectral space
  (§3.3).
- **Pluggable `physics_fn` closure.** Swapping Held–Suarez for the differentiable convection scheme +
  NN correction is structurally trivial. Preserve the signature, extend the outputs (§3.1.2 item 4).
- **Tendency output scaling** (`_TEND_SCALE`) so the network initializes near zero forcing. Extend
  with a per-channel scale for `dq/dt` and surface rain rate.
- **Warm-up-and-harvest pattern** for configs (semi-implicit matrices, transform kernels) via one
  throwaway climt step.

#### 3.1.2 Replace (each item is a concrete work package)

1. **The target.** Held–Suarez is dry, smooth, deterministic, quasi-linear, and pointwise — a
   best-case adversary. Every success of the demo exploits this. Replace the *emulation* objective
   with a *calibration/correction* objective: the differentiable convection scheme runs inside the
   loop as the physics backbone; trainables are (i) its physical parameters and (ii) an NN
   correction/residual term.
2. **The loss.** Trajectory MSE on (u, v, T) learns conditional means. For rainfall this is exactly
   the failure mode (double penalty, smoothed extremes). Replace with the composite
   distribution-aware loss of §3.2.3.
3. **The training distribution.** One DCMIP initial condition, one reference trajectory, no
   train/test split → regime overfitting. Replace with batched rollouts from many analysis states
   spanning seasons, regimes, and (eventually) perturbed climates. `vmap` over initial conditions.
4. **The state vector.** The MLP outputs (du, dv, dT) only; humidity is a passive tracer and there is
   no precipitation channel. Add `dq/dt` (with positivity / column-moisture-conservation constraints,
   e.g., output a *flux* form or project tendencies onto a conserving basis) and a surface
   precipitation diagnostic that participates in the loss.
5. **Rollout-length strategy.** Six steps (1 h) is benign; gradients through chaotic dynamics grow
   ~exponentially with horizon. Forecast mode keeps us at 1–15 day horizons where this is manageable,
   but implement: gradient clipping, truncated BPTT with overlapping windows, and curriculum on
   rollout length (start 6 h, grow to 5 d).
6. **Triggers.** Convective trigger functions create piecewise-constant maps with zero/undefined
   gradients. Replace hard thresholds in the convection scheme with smoothed versions during training
   (sigmoid CAPE trigger with temperature annealing; Gumbel-softmax for discrete plume choices),
   optionally re-hardening at inference.

#### 3.1.3 Known constraint to design around

`apply_physics` projects grid tendencies to spectral space every step (three `s2_forward` calls +
truncation). Fine at L=16; at production resolution this is the hot loop — batch the three transforms
and check `vmap` vs explicit batching in s2fft.

### 3.2 Forecast-mode calibration for extreme rainfall

#### 3.2.1 Why online (through the dycore), why forecast mode

Extreme rainfall is not column-local: it emerges from the feedback between convective heating and
large-scale moisture convergence. Offline (column) calibration cannot see this loop; online
calibration through the differentiable dycore includes the path
`∂(rain) / ∂(params)` ⊃ `∂rain/∂heating · ∂heating-induced-circulation/∂params`.

Forecast mode (initialize from ERA5/IFS analyses, verify at 1–5 day leads against IMERG/GPM and gauge
data) is preferred over free-running climate calibration because:
- gradients stay informative (horizons short of severe Lyapunov corruption);
- thousands of effective samples of heavy-rain situations exist in a few years of analyses;
- verification against *specific observed events* is possible, which climate-mode statistics can't
  give;
- it sidesteps the (separate, hard) problem of differentiating long-run statistics of chaotic
  systems.

#### 3.2.2 Data plan

- **Initial conditions:** ERA5 analyses, stratified sampling that over-represents moist unstable
  environments (e.g., sample dates by CAPE/CWV percentile so the batch contains exceedances at the
  target quantiles).
- **Verification:** IMERG half-hourly rain (note: satellite QPE biases at the extreme tail; consider
  gauge-corrected products over land, and a per-region observation-error term in the loss).
- **Splits:** train/val/test by *year blocks* (not random days — autocorrelation leaks).

#### 3.2.3 Loss design

Composite loss over a batch of rollouts:

```
L = w1 * L_pinball  +  w2 * L_exceed  +  w3 * L_state  +  w4 * L_phys
```

- `L_pinball`: quantile (pinball) loss at q ∈ {0.9, 0.99, 0.999} on space–time-pooled rain. Targets
  the tail directly; needs exceedances in every batch (hence stratified ICs).
- `L_exceed`: differentiable threshold-exceedance frequency match,
  `freq_model = mean(sigmoid((R - R_thr)/s))` vs observed frequency, for thresholds
  {10, 25, 50, 100} mm/day. Anneal smoothing `s` downward.
- `L_state`: standard scaled MSE/CRPS on (u, v, T, q) vs analyses at lead time — an anchor preventing
  the tail terms from degrading the mean state.
- `L_phys`: soft physics penalties — negative-moisture penalty, column moist-static-energy budget
  residual.

#### 3.2.4 Code sketch (forecast-mode calibration loop)

```python
# Assumes the script's API: rollout(physics_fn, spec0, n_steps) -> (final, traj)
# extended so traj includes precipitation and q.

import jax, jax.numpy as jnp, optax
from functools import partial

QUANTILES = jnp.array([0.90, 0.99, 0.999])
THRESHOLDS = jnp.array([10., 25., 50., 100.]) / 86400.0   # mm/day -> kg m-2 s-1 equiv

def pinball(pred, obs, q):
    e = obs - pred
    return jnp.mean(jnp.maximum(q * e, (q - 1.0) * e))

def quantile_loss(rain_model, rain_obs):
    # pool space-time, compare model quantiles to obs quantiles via pinball on pooled samples
    qs_obs = jnp.quantile(rain_obs.ravel(), QUANTILES)
    losses = jnp.array([pinball(jnp.quantile(rain_model.ravel(), q), qo, q)
                        for q, qo in zip(QUANTILES, qs_obs)])
    return losses.sum()

def exceedance_loss(rain_model, rain_obs, smooth=1e-5):
    fm = jax.vmap(lambda t: jnp.mean(jax.nn.sigmoid((rain_model - t) / smooth)))(THRESHOLDS)
    fo = jax.vmap(lambda t: jnp.mean((rain_obs > t).astype(jnp.float32)))(THRESHOLDS)
    return jnp.sum((jnp.log(fm + 1e-9) - jnp.log(fo + 1e-9)) ** 2)   # log-frequency match

def physics_penalty(traj):
    q = traj.q
    neg_moisture = jnp.mean(jnp.maximum(-q, 0.0) ** 2)
    return neg_moisture   # + MSE budget residual term

def forecast_loss(params, ic_batch, obs_batch, rollout, n_steps, weights):
    def single(ic, obs):
        _, traj = rollout(make_physics(params), ic, n_steps)       # params = (conv_params, nn_params)
        return (weights[0] * quantile_loss(traj.rain, obs.rain)
              + weights[1] * exceedance_loss(traj.rain, obs.rain)
              + weights[2] * state_anchor(traj, obs)
              + weights[3] * physics_penalty(traj))
    return jnp.mean(jax.vmap(single)(ic_batch, obs_batch))

@partial(jax.jit, static_argnames=("rollout", "n_steps"))
def train_step(params, opt_state, ic_batch, obs_batch, rollout, n_steps, weights, optimizer):
    loss, grads = jax.value_and_grad(forecast_loss)(params, ic_batch, obs_batch,
                                                    rollout, n_steps, weights)
    grads = clip_by_global_norm_tree(grads, 1.0)                    # gradient hygiene: mandatory
    updates, opt_state = optimizer.update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state, loss
```

**Curriculum:** train at 6 h rollouts first, then 24 h, 72 h, 120 h, reusing weights. Monitor
gradient norms per horizon; if they explode at a given horizon, that horizon is your BPTT truncation
point.

#### 3.2.5 Role of the regime-sensitivity analysis

Run the Jacobian/sensitivity analysis of the convection scheme across regimes *first* (= H-WP0, full
methodology in §2.2). It determines: (i) which physical parameters the extreme-rain gradients can
constrain (sensitive, non-degenerate) → trainable set; (ii) which are sloppy directions (cf.
MBAM/FIM analysis) → freeze or regularize heavily; (iii) where stochastic degrees of freedom belong
(§3.3).

### 3.3 Stochastic physics: design, training, and the heatwave question

#### 3.3.1 Two distinct roles for stochasticity — don't conflate them

1. **Physical role:** represent genuine subgrid variability and model error (a given large-scale
   state admits a distribution of convective outcomes). Calibrated against observed *variability*
   (CRPS, spread–error).
2. **Algorithmic role:** generate trajectory divergence so that RES cloning produces distinct
   children. Calibrated for *sampling efficiency* within the inter-resampling interval.

The same machinery serves both, but the tuning objective differs.

#### 3.3.2 The taxonomy (all differentiable, all trainable)

| Scheme | What is perturbed | Best for | Trainables |
|---|---|---|---|
| **Parameter noise** | Convection-scheme parameters θ → μ(x) + σ(x)·ε | Rainfall extremes; physically interpretable spread | μ, σ networks; correlation scales |
| **Differentiable SPPT** | Multiplicative AR(1) pattern on total physics tendencies | General-purpose spread; cheap | amplitude σ, decorrelation time τ, length scale L̃ |
| **SKEB-like** | Additive perturbation to *vorticity* tendency | Rotational / large-scale (Rossby wave, blocking) spread | backscatter amplitude field, spectrum shape |
| **Latent-noise NN** (NeuralGCM-ENS style) | Correlated random fields z fed as NN inputs; network shapes the response | Most expressive; subsumes the above | the network itself + field correlation params |

Key fact: **learned latent noise subsumes SPPT.** SPPT is the special case "multiply tendencies by
(1 + σ·r)" with hand-tuned σ, τ, L̃. In a differentiable model, implement SPPT exactly but make
(σ, τ, L̃) trainable — then graduate to latent-noise inputs if expressiveness is insufficient.

#### 3.3.3 Answer to: "divergence is mostly large-scale dynamics — how should I do SPPT?"

For midlatitude heatwaves, ensemble divergence is *powered* by baroclinic instability and Rossby-wave
dynamics — the atmosphere amplifies any sufficiently energetic small perturbation at Lyapunov rates
(error doubling ≈ 1.5–2 days in the midlatitude troposphere). So convective-parameter noise *will*
eventually diverge trajectories. The real questions are **efficiency** and **relevance**:

- **Efficiency:** does spread reach useful amplitude within the RES inter-resampling interval
  (typically a few days)? Noise that projects weakly onto growing baroclinic structures wastes part
  of the interval "finding" the unstable subspace.
- **Relevance:** does the spread explore the directions that matter for the event (ridge
  amplification, wave-breaking, blocking onset, soil-moisture–temperature coupling), not just generic
  synoptic scatter?

**Recommendation:**
1. **Primary:** a learned stochastic perturbation at the physics-NN level (latent-noise fields as NN
   inputs), trained with CRPS / spread–error at 3–10 day leads on T2m and Z500 — the lead range
   matching the resampling interval and the variables defining the event. This lets training
   *discover* the noise structure that produces calibrated large-scale spread, instead of
   hand-placing it.
2. **Secondary, targeted:** a SKEB-like additive perturbation to the vorticity tendency (generated in
   spectral space — reuse the reality-symmetry guard), which injects spread directly into the
   rotational flow that controls Rossby dynamics. Trainable amplitude/spectrum.
3. **Demote (for this application):** convective-parameter noise. Keep it for the rainfall-extremes
   track where the perturbed quantities are physically meaningful; for heatwave RES it is an
   indirect, inefficient route to large-scale spread.
4. **Land caveat:** heatwave persistence runs through soil moisture. If the hybrid model lacks a
   prognostic land state (§3.5), stochastic atmospheric perturbations alone cannot explore the
   soil-moisture-preconditioned pathways that dominate the most extreme events. A simple
   differentiable bucket land scheme with its own (small) stochastic term is likely worth more than
   any amount of atmospheric noise tuning.

#### 3.3.4 Code sketches

**Reparameterized parameter noise (for the rainfall track):**

```python
def stochastic_conv_params(nn_params, features, eps):
    """theta(x) = mu(x) + softplus(rho(x)) * eps   -- gradients flow via reparameterization."""
    mu, rho = param_net(nn_params, features)          # per-column or per-region
    sigma = jax.nn.softplus(rho)
    return mu + sigma * eps                            # eps ~ N(0, I), sampled OUTSIDE the graph
```

**Differentiable SPPT (spectral AR(1) pattern, trainable scales):**

```python
def sppt_pattern_step(r_lm, key, params, L):
    """One AR(1) step of the SPPT pattern in spectral space.
    params = dict(log_sigma, log_tau, log_len)  -- all trainable scalars (or per-l vectors)."""
    tau   = jnp.exp(params["log_tau"])
    phi   = jnp.exp(-DT / tau)
    l     = jnp.arange(L)[:, None]
    # Gaussian spatial spectrum; normalization chosen so grid-point variance = sigma^2
    spec  = jnp.exp(-l * (l + 1) * jnp.exp(2 * params["log_len"]) / 4.0)
    eta   = sample_complex_normal(key, r_lm.shape)     # respect reality condition!
    r_new = phi * r_lm + jnp.sqrt(1 - phi**2) * spec * eta
    return enforce_triangular_truncation(r_new, L, T_trunc)

def apply_sppt(du, dv, dT, dq, r_grid, params):
    sigma = jnp.exp(params["log_sigma"])
    factor = 1.0 + jnp.clip(sigma * r_grid, -0.9, 0.9)  # clip for stability, still differentiable a.e.
    return du * factor, dv * factor, dT * factor, dq * factor
```

**Fair-CRPS ensemble loss (drives spread calibration):**

```python
def fair_crps(ensemble, obs):
    """ensemble: (M, ...) members; obs: (...) verification. Fair (unbiased) estimator."""
    M = ensemble.shape[0]
    term1 = jnp.mean(jnp.abs(ensemble - obs[None]), axis=0)
    pair  = jnp.abs(ensemble[:, None] - ensemble[None, :])      # (M, M, ...)
    term2 = jnp.sum(pair, axis=(0, 1)) / (2 * M * (M - 1))
    return jnp.mean(term1 - term2)

def ensemble_loss(params, ic, obs, keys, rollout, n_steps):
    def member(key):
        _, traj = rollout(make_stochastic_physics(params, key), ic, n_steps)
        return traj.t2m[-1]                                     # verify at lead time
    ens = jax.vmap(member)(keys)                                # (M, lat, lon)
    return fair_crps(ens, obs.t2m)
```

**Gradient hygiene for stochastic training:**
- **Common random numbers:** fix the `keys` across the params-gradient evaluation (they already are,
  by construction above) and across line-search/comparison evaluations, or gradient variance swamps
  signal.
- **Discrete events:** any on/off stochastic choice (does this column trigger?) must be relaxed —
  Gumbel-softmax / concrete distribution with annealed temperature — to stay in the autodiff graph.
- **Clipping the multiplicative factor** (as above) prevents rare sign-flips of tendencies from
  destabilizing the semi-implicit stepper.

### 3.4 Rare event sampling for heatwaves with a differentiable model

#### 3.4.1 Event definition (differentiable heat-stress indices)

Event index over region Ω and window [t1, t2]:

```
A = (1/|Ω| (t2 - t1)) ∫∫ HSI(T, q, u10, SW, ...) dx dt
```

where HSI is any differentiable heat-stress index:
- **T2m:** trivially differentiable (the AI+RES choice).
- **Heat index / humidex:** closed-form, differentiable.
- **UTCI:** the 6th-order polynomial approximation is differentiable as-is.
- **WBGT:** simplified forms (Stull 2011 wet-bulb; ACSM outdoor approximation) are closed-form
  differentiable. The Liljegren model is iterative — wrap with implicit differentiation
  (`jaxopt.ImplicitDiff` / `lax.custom_root`) if needed.

Because A is differentiable in the model state, all gradient machinery below applies to *any* of
these indices interchangeably — implement HSI as a pluggable pure function.

#### 3.4.2 The DMC / RES backbone (what AI+RES does)

Diffusion Monte Carlo with N walkers (parallel stochastic model runs). At resampling times t_k, each
walker i gets a weight

```
V_k^i = exp( C_k * S(X_{t_k}^i) ) / Z_k
```

built from a **score function** S; walkers are cloned/killed proportionally to V; unbiased event
probabilities are recovered from the product of normalizers Z_k. Two facts to engrave:

1. **Unbiasedness holds for ANY score function** (with correct weight bookkeeping). The score affects
   only the *variance* of the estimator and the yield of event samples.
2. **The optimal score is the committor** (more precisely, the Doob h-transform / conditional
   expectation of the future index). Everything about score design is about approximating this
   cheaply.

AI+RES's contribution: use an AI weather emulator's ensemble forecast of the future index as S —
replacing the persistence score that fails for blocking-driven heatwaves. They report unbiased return
periods to 50,000 years with 400 walkers, ~100× speedup (PlaSim testbed).

#### 3.4.3 What differentiability changes — and what it does not

**It does NOT remove the need for a score/committor approximation.** A raw adjoint gradient
∂A(t_f)/∂x(t_k) along one trajectory is a local, single-realization sensitivity: it is not an
expectation over future noise, and it is corrupted by Lyapunov growth beyond ~ a week. The committor
is an expectation; the gradient is a derivative. You still need S.

**It changes how you build S, and adds tools around the sampler:**

**(a) The model as its own score (likely the first thing to try).**
AI+RES needed a separate NN emulator because PlaSim isn't differentiable or especially fast
per-walker on GPU. A jitted JAX hybrid model on GPU may be fast enough that a *small ensemble
forecast of the model itself* (their PFS+RES "perfect emulator" baseline — the upper bound on
performance) is affordable as the score:

```python
def model_score(state, key, n_members=8, horizon_steps=H):
    """S(x) = mean over a mini-ensemble of the future heatwave index."""
    keys = jax.random.split(key, n_members)
    def member(k):
        _, traj = rollout(make_stochastic_physics(params, k), state, horizon_steps)
        return heatwave_index(traj)                  # pluggable HSI
    return jnp.mean(jax.vmap(member)(keys))
```

Cost control: run the score forecasts at *reduced resolution / fewer levels* (re-truncate the
spectral state — cheap in this codebase) and/or shorter horizon than t_f, then hand off to a learned
S as it matures.

**(b) A learned committor with gradient features.**
Train S_φ(x) (small NN) to predict the future index / event indicator from walker states harvested
across RES iterations (adaptive: rerun RES with the improved score). Differentiability contributes
**adjoint features**: short-horizon sensitivities like ∂(3-day mean Z500 over the ridge region)/∂x
compress dynamical information that a state-only network learns slowly.

```python
def committor_features(state):
    g = jax.grad(lambda s: short_horizon_index(s, steps=K_SHORT))(state)  # K_SHORT ~ 2-3 days
    return jnp.concatenate([coarse_state_features(state), coarse(g)])

def committor_loss(phi, states, final_indices):
    pred = jax.vmap(lambda s: committor_net(phi, committor_features(s)))(states)
    return jnp.mean((pred - final_indices) ** 2)     # or pinball at the event quantile
```

**(c) Gradient-steered cloning — with the Girsanov caveat.**
When a walker is cloned, instead of relying on noise alone to separate children, perturb a clone
along the adjoint direction of the index:

```python
def steered_clone(state, key, alpha):
    g = jax.grad(lambda s: model_score(s, key))(state)         # or short-horizon index grad
    g = g / (tree_norm(g) + 1e-12)
    return tree_add(state, tree_scale(g, alpha))
```

**Two modes, choose explicitly:**
- *Catalog mode:* steering allowed freely → maximizes yield of physically realistic extreme-event
  samples (ensemble-boosting-style). Return periods from these runs are **biased**; don't report
  them.
- *Statistics mode:* steering must be accounted for. If the perturbation is implemented as a **mean
  shift of the stochastic forcing** (shift the latent noise: ε → ε + μ with μ ∝ projected gradient),
  the importance weight is the exact Gaussian likelihood ratio (Girsanov factor)
  `exp(-μ·ε - |μ|²/2)`, multiplied into the walker weight. This keeps the estimator unbiased while
  still steering. Implement steering this way (through the noise, never as a raw state perturbation)
  whenever return periods are the product.

**(d) Instantons / most-probable paths (direct bridge to the MPP-IG framework).**
With stochastic forcing entering the rollout, optimize the noise sequence E = (ε_1, …, ε_K) to find
the most probable path to the event — the trajectory-space generalization of the most-probable-point
construction already developed for UTCI/WBGT attribution:

```python
def action_objective(E, ic, target):
    _, traj = rollout_with_noise(params, ic, E)       # noise injected per step from E
    A = heatwave_index(traj)
    quadratic_action = 0.5 * sum_tree(lambda e: jnp.sum(e**2), E)
    return quadratic_action + LAMBDA * jax.nn.softplus(target - A)   # hit constraint softly

opt = optax.adam(1e-2)
E = init_noise_like(rollout_noise_shapes)
for _ in range(N_OPT):
    loss, gE = jax.value_and_grad(action_objective)(E, ic, target)
    updates, opt_state = opt.update(gE, opt_state)
    E = optax.apply_updates(E, updates)
```

Uses: (i) physical diagnosis — *what synoptic/noise sequence most probably produces a 1000-year
event from this initial state?*; (ii) an importance-sampling center: tilt the walker noise toward E*
with Girsanov reweighting; (iii) initialization/validation of the learned committor (the instanton
path's progress variable should correlate with S).

#### 3.4.4 Putting it together: recommended RES architecture

```
Stochastic hybrid model (§3.3: latent-noise NN + SKEB, CRPS-trained)
        │  provides walkers AND divergence at cloning
        ▼
DMC with quantile-style adaptive levels, resampling every Δt ≈ 2–4 days
        │
Score S:  stage 1 — model-as-own-forecast mini-ensembles (8 members, coarse res)
          stage 2 — learned committor S_φ with adjoint features, trained on stage-1 output
        │
Steering: noise-mean-shift along adjoint direction + Girsanov weights (statistics mode)
          unconstrained gradient steering (catalog mode)
        │
Instanton solver: independent diagnostic + IS center + committor sanity check
```

Validation plan mirrors AI+RES: an intermediate-complexity / coarse-resolution configuration cheap
enough for a long DNS ground truth; compare return-period curves of DMC vs DNS vs EVT(GPD/POT) fits;
check composite synoptic precursors of sampled events against DNS composites.

### 3.5 NeuralGCM: what was and wasn't learned (land/ocean question)

**Question:** NeuralGCM used only a dycore + learned physics — does this hold with land/ocean
contrast; did the NN learn it?

**Answer:** The framing holds, with a precise asterisk about what "learned" means:

- **Prescribed forcings, not learned ocean:** sea surface temperature and sea-ice concentration were
  *inputs* (prescribed from data), not predicted. The NN learned the atmospheric response to a given
  ocean state; it learned no ocean.
- **Static surface descriptors:** orography and the land–sea mask were static input features.
  Land/ocean *contrast* in fluxes, diurnal cycle, and boundary-layer behavior was learned
  **implicitly** — the network infers "over land, behave like this" from the mask — not from any
  explicit surface energy balance.
- **No prognostic land state:** no soil moisture, no snowpack, no vegetation memory. The land surface
  has no carried state between timesteps beyond what the atmosphere remembers.

**Implication for this project (first-order, not cosmetic):** soil-moisture preconditioning is a
dominant amplifier and persistence mechanism for midlatitude heatwaves (and AI+RES explicitly lists
soil-moisture information as a needed improvement to their score). A hybrid model without land memory
will under-represent exactly the slow pathways RES must explore for the rarest events.
**Recommendation:** include a minimal *prognostic, differentiable* land scheme (bucket / single-layer
soil moisture with evaporative-fraction coupling, ~10 lines of JAX) rather than relying on the NN to
absorb land memory it structurally cannot carry. Make its 1–3 parameters trainable; give it a small
stochastic term (§3.3.3, item 4).

### 3.6 Decision summary (the three clarifications, condensed)

1. **NeuralGCM & surfaces:** dycore + learned column physics, with prescribed SST/sea-ice and static
   land–sea mask/orography; land/ocean contrast learned implicitly; **no land memory** → add a
   differentiable prognostic land bucket for heatwave work.
2. **Stochasticity for heatwave RES:** train a **stochastic NN perturbation** (latent-noise inputs,
   CRPS at 3–10 day leads), optionally + SKEB-like vorticity perturbations for rotational spread;
   differentiable SPPT with *trainable* (σ, τ, L̃) is the simplest starting point and a special case
   of the latent-noise scheme. Reserve convection-parameter noise for the rainfall-extremes track.
3. **Committor:** still needed — DMC is unbiased for any score but variance/yield depend on score
   quality, and gradients are local quantities, not conditional expectations. Differentiability
   changes *how* you build it (model-as-own-forecast score; adjoint features for a learned committor;
   Girsanov-reweighted gradient steering; instantons) — it does not eliminate it.

### 3.7 Build-out roadmap (ordered work packages, H-WP0–13)

| # | Work package | Depends on | Acceptance criterion |
|---|---|---|---|
| H-WP0 | Regime sensitivity analysis of convection scheme (Jacobians across tropical/midlat/warm profiles; FIM/sloppiness — full methodology §2.2) | — | Ranked trainable parameter set; degenerate directions identified |
| H-WP1 | Moisture + precipitation channels in rollout (`dq/dt`, conserving projection, rain diagnostic in `traj`) | — | Closed column moisture budget to FP tolerance in a 5-day rollout |
| H-WP2 | Batched forecast-mode harness: `vmap` over ERA5 ICs, year-block splits, stratified heavy-rain sampling | H-WP1 | ≥30 exceedances of the 99.9th percentile per batch |
| H-WP3 | Composite rainfall loss (§3.2.3–3.2.4) + curriculum on rollout length + gradient clipping/truncated BPTT | H-WP2 | Stable training to 120 h rollouts; tail quantiles improve on val years without mean-state degradation |
| H-WP4 | Smoothed triggers in convection scheme (annealed sigmoid / Gumbel-softmax) | H-WP0 | Nonzero, finite gradients of rain w.r.t. trigger parameters across regimes |
| H-WP5 | Differentiable SPPT (trainable σ, τ, L̃; spectral AR(1) pattern passing reality guard) | — | Spread–error ratio ≈ 1 at 5-day lead on T2m after CRPS training |
| H-WP6 | Latent-noise NN physics + fair-CRPS ensemble training; SKEB vorticity perturbation | H-WP5 | Beats trained SPPT on CRPS for Z500 & T2m at 3–10 d |
| H-WP7 | Differentiable prognostic land bucket (soil moisture, evaporative fraction), trainable + stochastic | — | Reproduces observed lagged soil-moisture/T2m correlation sign & decay scale |
| H-WP8 | Pluggable heat-stress index module (T2m, HI, UTCI poly, Stull-WBGT; implicit-diff hook for iterative indices) | — | `jax.grad(index)` finite & validated against FD for all indices |
| H-WP9 | DMC sampler (walkers via `vmap`/`pmap`, weight bookkeeping, quantile-adaptive levels) | H-WP6 | Unbiased return periods vs long DNS in a coarse test config |
| H-WP10 | Score stage 1: model-as-own-forecast mini-ensemble score (coarse-res re-truncated forecasts) | H-WP9 | ≥10× sampling speedup vs persistence score at 100-year return period |
| H-WP11 | Score stage 2: learned committor with adjoint features; adaptive retraining loop | H-WP10 | Variance speedup over stage 1 on held-out RES runs |
| H-WP12 | Girsanov-reweighted noise-shift steering (statistics mode) + unconstrained steering (catalog mode) | H-WP9 | Statistics mode: unbiased vs DNS; catalog mode: ≥2× event yield |
| H-WP13 | Instanton solver over noise sequences; use as IS center & committor validation | H-WP6, H-WP8 | Instanton index value reaches target; tilted sampling reproduces DNS probabilities within error bars |

**Suggested first milestone (≈ H-WP0–3, H-WP5):** a calibrated deterministic + SPPT-stochastic
hybrid in forecast mode with demonstrably improved 99.9th-percentile rainfall at 3-day lead — every
later package builds on this artifact. (This is the same object as Track 1a's first artifact; see
§6.)

---

## 4. Track 3 — Probabilistic weather ensemble (the credibility demonstration)

### 4.1 Goal and honest framing

A CRPS-trained stochastic hybrid ensemble at T62 (development) → T126 (headline), evaluated on
WeatherBench2 against IFS ENS, NeuralGCM-ENS, and GenCast-class emulators. The goal is **not** to win
those scoreboards — at our compute that is not realistic — but to (i) be *credible* on them (clearly
better than climatology + persistence + lagged ensembles, within striking distance of NeuralGCM-ENS
at matched resolution), and (ii) deliver the thing the others structurally cannot: **attributable
spread** (§4.4).

### 4.2 System design — three uncertainty sources

An ensemble system samples initial-condition uncertainty, model uncertainty, and (for some systems)
boundary uncertainty. §3.3 specifies the model-uncertainty machinery; this section adds the rest.

#### (a) Initial-condition uncertainty (not covered in §3 — new design)

Options, in recommended order (**decision D-5: SV-based perturbations are the target mechanism**):

1. **Singular-vector perturbations from §2.4.2.** Compute the leading k ≈ 20 SVs {v_i, σ_i} of the
   48 h tangent-linear propagator at each initialization (energy norm, extratropical target domains
   N and S). Sample perturbations in the SV subspace:

   ```
   x₀^{(j)} = x₀ + Σ_{i=1}^{k} c_i^{(j)} v_i ,   c^{(j)} ~ N(0, γ² I) ,  members in ± pairs
   ```

   (± pairs center the ensemble on the control). The single tunable γ is set so that day-1 ensemble
   spread matches the day-1 RMSE of the control forecast (the classical calibration); thereafter γ
   can be made *trainable* alongside the stochastic-physics parameters in the CRPS stage — a small
   but novel twist (joint IC-and-model-uncertainty training) that the differentiable SV machinery
   makes possible. SVs are flow-dependent and target growing structures, which is precisely what
   emulator ensembles (no tangent-linear model) cannot generate — part of the differentiation story.
2. **ERA5 EDA-derived perturbations:** ERA5 provides a 10-member ensemble of data assimilations at
   reduced resolution; differences from the EDA mean, rescaled, give observation-grounded
   perturbations. Use as cross-check of SV amplitude/structure and as fallback.
3. **Lagged ensembles** (forecasts from staggered earlier inits): zero implementation cost; the
   development-phase baseline every configuration must beat.

#### (b) Model uncertainty

Exactly the §3.3 taxonomy promoted from "RES enabler" to product: v1 = trainable SPPT (H-WP5);
v2 = latent-noise NN + SKEB (H-WP6). All implemented as swappable `TendencyComponent`s on the host
path with pure-jnp kernels on the training path; all spectral noise passes the reality-symmetry
guard; the SPPT pattern's AR(1) state is carried in the rollout scan's carry.

#### (c) The deterministic backbone

**Decision (D-2): physical suite + NN residual** (the Track 1a artifact), not a fully learned column
physics. Rationale: preserves the attribution story, shares all infrastructure with Track 1, degrades
gracefully out of distribution (needed by Tracks 2 and 4). Accept that a fully learned physics would
score higher; revisit only if v1 scores are not credible (revisiting this is a logged-decision
change, not a silent one).

### 4.3 Training protocol

1. **Deterministic pretrain:** backbone calibration per §2.3 (curriculum to 120 h, truncated BPTT,
   state anchor + process terms).
2. **Stochastic fine-tune:** fair CRPS (estimator in §3.3.4) with M = 8–16 members via `vmap` over
   noise keys, **common random numbers across each gradient evaluation**; verification variables
   T2m, Z500, T850, u/v850, 24 h precip at leads {1, 3, 5, 7, 10} d with weights w_ℓ ∝ 1/ℓ; spread–
   error ratio and rank histograms monitored on val years but *not* trained on (they are the
   overfitting alarm for CRPS). Optionally add the IC amplitude γ to the trainables (§4.2a).
3. **Memory:** M members × N steps of reverse-mode = M × the §C-WP2 budget ⇒ the checkpointed
   rollout is mandatory, plus gradient accumulation over member subsets when M × √N states exceed
   device memory; at T126, members shard across devices (C-WP2 Mode 1).
4. **Evaluation:** M = 50 at inference (members are forward-only — cheap), WeatherBench2 protocol,
   test years untouched until the end.

### 4.4 The attribution deliverable: the spread budget

Because every spread source is a named, switchable component with its own PRNG stream, the ensemble
variance can be decomposed by source — the flagship analysis no monolithic system (NeuralGCM-ENS,
GenCast) can produce.

**Design.** Sources z = (z_IC, z_SPPT, z_latent, z_SKEB, z_land). For verification variable X at lead
ℓ (per grid point or region-averaged):

- **Only-k ensembles:** all sources off except k → variance V_k^only(X, ℓ).
- **All-on ensemble:** V_tot(X, ℓ).
- **Without-k ensembles:** all on except k → V_{-k}(X, ℓ).

First-order share `S_k = V_k^only / V_tot`; total-effect share `T_k = 1 − V_{-k}/V_tot`; interaction
magnitude `I = V_tot − Σ_k V_k^only` (zero for additive-linear dynamics; its growth with lead time
*measures* the nonlinearity of spread generation — itself a result). This is exactly the
Sobol first-order/total-effect decomposition with the variance estimated by ensembles. **Common
random numbers:** retained sources keep identical keys across all ablations, so differences are
attributable to the toggled source, not sampling noise; M = 50 per ablation, K + 2 ablation
ensembles per case, ~30 cases stratified by season — forward-only runs, embarrassingly parallel.

**Headline outputs:** stacked-area charts of S_k(ℓ) per variable/region (e.g., "at day 5, X% of T2m
spread over land originates in the land-surface stochastic term"); maps of the dominant source at
fixed lead; the I(ℓ) nonlinearity curve. Acceptance: Σ_k S_k + I/V_tot ≈ 1 within ensemble sampling
error (closure check).

### 4.5 Evaluation harness

WeatherBench2 conventions: 00/12 UTC inits over the test years; ERA5 verification at WB2's standard
regridding; deterministic scores (RMSE, ACC, bias) for the ensemble mean and control; probabilistic
scores: CRPS (fair), spread–error ratio, rank histograms / reliability, Brier scores for threshold
events (T2m > climatological 95th percentile — connecting to Track 2's event definitions); SEEPS or
CRPS for 24 h precip, never plain MSE (double penalty). **Protocol rule:** reproduce the published
baseline scores (IFS ENS, NeuralGCM-ENS from WB2's archived forecasts) within tolerance *before*
scoring our own system — this validates the harness and disciplines comparisons to matched
resolution/variables.

### 4.6 Work packages

| # | Package | Depends on | Acceptance criterion |
|---|---|---|---|
| 3-1 | IC perturbation module (lagged → EDA → SV-based, ± pairs, γ calibration) | 1b-3 for SVs | Day-1 spread within 10% of control day-1 RMSE; SV ensembles beat lagged on day-3–7 CRPS |
| 3-2 | v1 ensemble: backbone + trained SPPT + IC perturbations @ T62 | C-WP1/3/4, H-WP5, 1a-3 | Spread–error ∈ [0.9, 1.1] at day 5 on Z500/T2m (val years); beats lagged-ensemble CRPS at all leads |
| 3-3 | v2 stochastic physics: latent-noise NN + SKEB, CRPS fine-tune (+ trainable γ) | 3-2, H-WP6 | Beats v1 CRPS on Z500 & T2m at 3–10 d (val years) |
| 3-4 | WeatherBench2 evaluation harness + external baselines | C-WP3 | Reproduces published baseline scores from archived forecasts before scoring ours |
| 3-5 | Spread-budget attribution study (§4.4) | 3-3 | Closure within sampling error; S_k(ℓ) curves for ≥3 variables × 3 regions |
| 3-6 | T126 scale-up (member sharding; transform throughput from C-WP2) | 3-3, C-WP2 | Same training protocol stable at T126; step time within 3× of linear scaling from T62 |

**Compute:** v1/v2 at T62: a single 8-GPU node for weeks. T126 headline training: 16–32 GPUs for
2–6 weeks per serious run — the program's compute peak; schedule against a national-HPC or
cloud-credit allocation. Spread-budget runs are forward-only and fill queue gaps.

**Risks.** Skill gap vs pure emulators (mitigation: framing — attribution + hybrid robustness, and
matched-resolution comparisons only); transform throughput at T126 (mitigation: §3.1.3 benchmarks
before commitment); CRPS-trained spread collapsing onto a few effective members (monitor rank
histograms; fair-CRPS estimator specifically mitigates M-dependence).

---

## 5. Track 4 — Climate emulation (stable AMIP hybrids with conservation by construction)

### 5.1 Goal and honest framing

Multi-decade AMIP-style runs (prescribed observed SST/sea-ice, 1979–2014) of the hybrid model that
are (i) stable, (ii) close energy and water budgets *by construction*, and (iii) carry arbitrary
tracers — the three places NeuralGCM's climate configuration is weakest (conservation handled post
hoc; adding prognostic variables/tracers is invasive; physics unattributable). We do **not** chase
climate-mean skill contests; the product is *trustworthy long-run behavior with auditable budgets*,
as the foundation for climate-mode extremes (Track 2's statistics at scale) and perturbed-climate
experiments.

### 5.2 Conservation by construction (the signature move)

**Decision (D-6): per-component conservation fixers; global fixers are backstops only.** A monolithic
NN can only be fixed globally, and you never learn which process leaks; per-component fixing turns
conservation into an *audit*.

#### 5.2.1 Formulation

Every component declares the column budgets its tendencies must satisfy, in terms of its own
diagnostic outputs. With pressure-mass element dm = dp/g and column integral ⟨·⟩ = ∫·dm:

- **Moisture:** a component producing tendency `(dq/dt)` and diagnostics precip P, surface
  evaporation E must satisfy `⟨dq/dt⟩ = E − P`. Residual `r_q = ⟨dq/dt⟩ − (E − P)`.
- **Energy (moist static):** with declared boundary fluxes F_net (net SW + LW at TOA and surface +
  SH + LH that the component claims to represent):
  `⟨c_p dT/dt⟩ + L_v ⟨dq/dt⟩ = F_net`, residual `r_E` analogous. (For components that do no phase
  change or radiation, F_net = 0 and the constraint is "do no net column heating".)

**Fixer:** distribute the residual over the column with a weight profile w(p):

```
(dX/dt)′ = dX/dt − r · w(p) / ⟨w⟩
```

Weight choice matters: `w = |dX/dt|` (proportional-to-activity) confines the correction to layers
the component is already modifying and cannot create activity where there was none; uniform-in-mass
`w = 1` is the fallback when the tendency is identically zero. Moisture fixing must additionally not
create negatives: apply the fixer, then resolve any q < 0 by borrowing from the layer's neighbors
(the standard downward-borrowing filler), and charge what cannot be borrowed back to P (so the
budget stays closed rather than the profile staying pretty).

#### 5.2.2 Implementation

A generic wrapper, enabled by the sympl property declarations (named quantities + units are what
make this *generic*):

```python
class ConservationFixer(ImplicitTendencyComponent):
    """Wraps any TendencyComponent; enforces declared column budgets; logs leaks."""
    def __init__(self, component, budgets=("moisture", "energy"), weight="activity"):
        self.component = component
        # inherit & merge input/output/diagnostic properties from the wrapped component
        ...
    def array_call(self, state, timestep):
        tends, diags = self.component.array_call(state, timestep)
        leaks = {}
        if "moisture" in self.budgets:
            tends, leaks["moisture_Wm2_equiv"] = fix_column_moisture(tends, diags, state)
        if "energy" in self.budgets:
            tends, leaks["energy_Wm2"] = fix_column_energy(tends, diags, state)
        diags.update({f"{self.component.name}_leak_{k}": v for k, v in leaks.items()})
        return tends, diags
```

The leak diagnostics accumulate into the **audit table**: time-mean W m⁻² (energy) and
mm day⁻¹-equivalent (water) leak per component — published alongside any climate result. The pure
training path gets the same fixers as jnp functions composed into `tends_fn` (one implementation,
two thin wrappers — same dual-path pattern as everything else). A small **global backstop fixer**
(uniform correction of the residual global-mean imbalance after all components) catches dynamics-side
truncation effects; if the backstop is doing significant work, that is a bug report, not a feature.

### 5.3 Long-run stability (the second hard problem)

Forecast-mode training does not guarantee climate stability, and differentiating long-run statistics
of a chaotic system directly is a known dead end (§3.2.1 sidesteps it for the same reason).
**Decision (D-3): stability is achieved by screening + curriculum + nudged fine-tuning + parameter
anchoring — never by differentiating free-running chaos.** In order:

1. **Train in forecast mode, test in climate mode (screening).** During Track 1/3 development, every
   candidate configuration gets 1-year free runs from 4 seasonal start dates as a *validation
   metric* (never a training signal). Automated metrics: global-mean TOA imbalance drift (linear fit
   over the year), stratospheric T drift, total-mass and total-water drift, kinetic-energy spectrum
   shape vs ERA5 (detecting noise accumulation at the truncation scale), blow-up watchdog. Gate
   model selection on these.
2. **Rollout-length curriculum as implicit regularizer.** Longer training horizons empirically
   correlate with long-run stability (NeuralGCM's reported experience); push the §2.3.3 curriculum as
   far as gradient health allows, with truncated BPTT beyond.
3. **Nudged long-rollout fine-tuning.** The controlled way to inject long-horizon information without
   differentiating free chaos: run multi-month rollouts with linear relaxation toward 6-hourly ERA5,

   ```
   x_{n+1} = Advance(x_n; φ, ψ) + Δt · (x_ref(t_n) − x_n)/τ ,    τ ≈ 12–24 h on (u, v, T), none on q
   ```

   and train on the **nudging increment itself**:
   `L_nudge = mean_n ‖(x_ref(t_n) − x_n)/τ‖²_scaled` — a smaller increment means the model needs less
   external support, i.e. is more self-consistent at climate timescales. The nudging keeps the
   trajectory bounded, so gradients through months stay finite and informative (this is the
   long-horizon analogue of teacher forcing). Use truncated BPTT over ~10-day windows within the
   long rollout; fine-tune only a low-dimensional subset (the physical parameters φ and the residual
   NN's output scale) to avoid washing out the forecast-mode skill.
4. **Physical-parameter anchoring.** The §2.3.2 priors keep the suite in physically defensible
   territory — the structural reason a hybrid out-extrapolates a pure emulator; monitor the residual
   NN's share of the total tendency (a *residual-dominance diagnostic*: if the NN grows to dominate
   the physical scheme in some regime, extrapolation trust is gone there — flag it).

### 5.4 Configuration & evaluation

- **Boundary components (swappable — the modular pattern again):** prescribed SST/sea-ice as a
  `DiagnosticComponent` reading monthly AMIP forcing with mid-month interpolation; later a
  slab/mixed-layer ocean (`c_w h dT_s/dt = F_net + Q_flux`, with the Q-flux computed to reproduce the
  observed SST climatology in a calibration run) as a drop-in replacement — enabling +4K and
  pattern-perturbation experiments. Land bucket (C-WP4) prognostic throughout; its parameters get a
  climate-mode recalibration pass (same machinery as §2.3, nudged-mode loss).
- **Tracers:** the multi-tracer path (C-WP1) exercised with at least one passive tracer from day one —
  an idealized age-of-air-like tracer (source at surface, clock tracer aloft) — both a correctness
  probe of transport (no spurious extrema, mass conserved) and the demonstrator of the extensibility
  claim vs NeuralGCM.
- **Evaluation tiers:** (i) *stability/budgets*: the §5.3.1 metrics over 30-yr runs + the §5.2 audit
  table; (ii) *climatology*: seasonal-mean maps and zonal means vs ERA5, precipitation distribution
  including the tail (reuse §3.2.3 terms as *diagnostics*), monsoon onset/withdrawal dates
  (regionally motivated and a known emulator weak spot); (iii) *variability*: ENSO-teleconnection
  *response* (SST prescribed, so ENSO itself is given), tropical intraseasonal variability within
  T62–T126 honesty limits; (iv) *OOD generalization*: AMIP+4K-style runs (uniform SST warming, and a
  patterned variant) — does the hybrid's response stay physical where pure emulators drift? Run the
  same protocol through a pure-emulator baseline (e.g., an open NeuralGCM checkpoint at matched
  resolution) — this comparison is a flagship paper for the hybrid thesis, and an honest null result
  is still a result.

### 5.5 Work packages

| # | Package | Depends on | Acceptance criterion |
|---|---|---|---|
| 4-1 | ConservationFixer wrapper (host + pure paths) + audit-table diagnostics | C-WP1 | Per-component leak table produced; global energy drift < 0.1 W m⁻² in a 1-yr run with backstop doing < 10% of total fixing |
| 4-2 | AMIP boundary components (SST/sea-ice reader, calendar/forcing plumbing) | C-WP1, C-WP3 | Bit-reproducible 1-yr AMIP run from config alone |
| 4-3 | Stability screening harness (1-yr runs, automated §5.3.1 metrics, gating) | 4-2 | Screening report auto-generated per candidate; ≥1 unstable configuration correctly caught |
| 4-4 | Nudged long-rollout fine-tuning | 4-3 | Nudging-increment norm decreases through fine-tune; post-fine-tune free-run drift improves vs pre |
| 4-5 | 30-yr AMIP production runs + tier (ii)/(iii) evaluation | 4-4 | Stable 30-yr run; precip climatology ≥ default-physics baseline at same resolution; audit table published |
| 4-6 | Passive-tracer extensibility demo | C-WP1 | Tracer transported 10 yr: mass conserved to FP tolerance, no spurious extrema |
| 4-7 | AMIP+4K OOD comparison vs pure-emulator baseline | 4-5 | Documented hybrid-vs-emulator response comparison (or honest null result) |

**Compute:** a 30-yr T62 run is single-GPU, days; the cost center is *model selection* (many
candidates × 1-yr screening) and T126 production — same node as Track 3, naturally time-shared
(training bursts vs long low-intensity runs).

**Risks.** Drift resisting all four §5.3 strategies (fallback: ship the forecast-mode artifact and
publish the stability-methodology study honestly); NN-residual extrapolation under +4K (mitigation:
physics-heavy backbone + residual-dominance diagnostic); conservation fixers degrading forecast
skill (measure the trade-off explicitly — it is itself a result).

---

## 6. Cross-track dependency map and sequencing

```
C-WP1 (component parity + pure advance)  ──────────────┬──────────────┬───────────────┐
C-WP3 (ERA5 pipeline)                                  │              │               │
C-WP4 (moisture/precip H-WP1, land H-WP7, HSI H-WP8)   │              │               │
        │                                              │              │               │
        ▼                                              ▼              ▼               ▼
  H-WP0 / §2.2 FIM ───► Track 1a calibration      Track 1b       Track 3 ensemble  Track 4 climate
  (identifiability)        (§2.3)                adjoints/SV         (§4)              (§5)
                              │                     (§2.4)            ▲                ▲
                              │                       │   SVs ────────┘ (§4.2a)        │
                              │                       │                                │
                              ▼                       ▼                                │
                      Track 2 rainfall         Track 2 committor/steering             │
                      calibration (§3.2)       adjoint features (H-WP11/12)           │
                              │                                                        │
                      stochastic physics H-WP5/6 (§3.3) ────► shared with Track 3 ────┘
                      RES pipeline H-WP9–13 (§3.4)
```

**Recommended sequencing for a 2–4 FTE group:**

1. **Quarter 1–2:** C-WP1 (keystone), C-WP3, C-WP4 basics; H-WP0/§2.2; Track 1a packages 1–3.
   First artifact: calibrated physical suite in forecast mode — the same object as the §3.7 first
   milestone (H-WP0–3 + H-WP5), so starting either track starts both.
2. **Quarter 2–4:** fork by person — one line continues Track 1b (4D-Var OSSE, SVs: papers with
   single-GPU compute), one line does H-WP5/6 stochastic physics → Track 3 v1 ensemble at T62.
   Track 2 RES work proceeds per the §3.7 roadmap on the same artifacts.
3. **Year 2:** Track 3 v2 + spread-budget paper; Track 4 stability/AMIP program; T126 scale-up
   (C-WP2 Mode 2 engineering) only once T62 science is publishing.

**Compute roll-up:** development = one 8-GPU node continuously; peaks = Track 3 T126 training
(16–32 GPUs × weeks, scheduled allocation) and Track 4 model-selection sweeps (time-shared with the
same node). Storage 10–50 TB. Tracks 1 and 2 (the most distinctive science) never need more than the
single node.

**Staffing:** PI + 1 research engineer (data pipeline, training infra, eventually transform sharding —
the highest-leverage hire) + 1–2 students (Track 1b and Track 2 make excellent self-contained PhD
chapters). AI coding agents give ~3–5× on the engineering packages (the C-WPs, harnesses, evaluation
plumbing) and ~1× on experiment iteration, numerical debugging, and scientific judgment — plan
person-time accordingly: the bottleneck is GPU-queue iteration and analysis, not code.

---

## 7. Paper / deliverable map

| Track | Deliverable | Venue-shape | Compute class |
|---|---|---|---|
| 1a | Gradient (vs EKI) calibration of an operational-lineage physics suite | JAMES / GMD methods + results | 1 node |
| 1b | 4D-Var with exact autodiff adjoints (OSSE, degraded-adjoint ablation); SVs via matrix-free AD | MWR / QJRMS | 1 GPU–1 node |
| 2 | RES for heatwaves w/ differentiable scores; instantons; rainfall-tail calibration (§3.7 roadmap) | per handoff | 1–2 nodes |
| 3 | Stochastic hybrid ensemble + **spread-budget attribution** | GRL/JAMES + WeatherBench entry | node → 32 GPUs |
| 4 | Conservation-by-construction AMIP hybrid w/ audit table; +4K OOD hybrid-vs-emulator | JAMES / J. Climate | 1 node sustained |

The platform itself (component-resolved differentiable hybrid framework, dual API, parity-validated)
is a software paper (JOSS / GMD-development) once C-WP1 + two tracks' usage exist — the
community-extensibility claim needs demonstrated usage, which the tracks themselves provide.

---

## 8. Decision log

Positions taken deliberately in this document. Implementers: do not silently revisit; a change here
is a change to the program.

- **D-0:** C-WP1 (component parity + `advance_with_tendencies`) is promoted from "deferred" to
  milestone 1 of whichever track starts first.
- **D-1:** Ensemble-parallelism (vmap/shard_map over members/ICs/walkers) before model-sharding;
  transform sharding deferred until a Mode-1 track is producing science, and tied to the engineer
  hire.
- **D-2:** Track 3/4 backbone = physical suite + NN residual, not fully learned column physics —
  preserves attribution, OOD robustness, Track-1 reuse; accepts a skill gap.
- **D-3:** Forecast-mode training everywhere; climate stability via screening + curriculum + nudged
  fine-tuning + parameter anchoring — never by differentiating free-running long-run statistics.
- **D-4:** EKI is a first-class co-method beside gradients in Track 1a, with a defined comparison
  protocol (§2.3.4).
- **D-5:** Initial-condition uncertainty via singular-vector perturbations (Track 1b output), with
  EDA and lagged ensembles as cross-check and baseline.
- **D-6:** Per-component conservation fixers with a published audit table; global fixers are
  backstops whose workload is itself a diagnostic.
- **D-7 (from the handoff, §3.6):** stochasticity for heatwave RES = latent-noise NN (+ SKEB);
  trainable SPPT as the starting special case; convection-parameter noise reserved for the rainfall
  track; a committor approximation remains necessary — differentiability changes how it is built,
  not whether.
- **D-8:** SV eigensolver = host-side Lanczos (`scipy eigsh`) over device matvecs first;
  JAX-native lobpcg only if the host loop becomes a bottleneck.
- **D-9:** Year-block data splits everywhere; test years untouched until final evaluation; baseline
  scores reproduced before own scores are reported (§4.5).

---

## 9. References

**Track 2 / handoff core:**

- Lancelin, Wikner, Dubus, Le Priol, Abbot, Bouchet, Hassanzadeh, Weare (2025). *AI-boosted rare
  event sampling to characterize extreme weather.* arXiv:2510.27066. (AI+RES: DMC + AI-emulator
  score; PlaSim heatwaves; unbiased return periods to 5×10⁴ yr; ~100× speedup; lists probabilistic
  emulators and soil moisture as needed improvements.)
- Kochkov et al. (2024). *Neural general circulation models for weather and climate.* Nature.
  (Differentiable spectral dycore + learned column physics; prescribed SST/sea-ice; CRPS-trained
  stochastic ensemble variant; baseline system for all four tracks.)
- Held & Suarez (1994). BAMS. (Reference forcing in the demo script.)
- Ragone, Wouters & Bouchet (2018). PNAS. (RES for heatwaves with persistence-type scores; the
  baseline AI+RES improves on.)
- Buizza, Miller & Palmer (1999); Palmer et al. (2009). (SPPT / stochastic physics lineage; SKEB.)
- Stull (2011). J. Appl. Meteor. Climatol. (Closed-form wet-bulb — differentiable WBGT ingredient.)
- Bröde et al. (2012). Int. J. Biometeorol. (UTCI polynomial approximation.)
- Gelbrecht et al. (2023). *Differentiable programming for Earth system modeling.* GMD. (Survey of
  the gradients-through-dycore paradigm.)
- E, Ren & Vanden-Eijnden (2004); Grafke & Vanden-Eijnden (2019). (Minimum action methods /
  instantons — trajectory-space analogue of the MPP construction.)

**Track 1 (identifiability, calibration, DA):**

- Transtrum, Machta & Sethna (2010, 2011); Transtrum & Qiu (2014). (Sloppy models, FIM geometry,
  MBAM — §2.2 methodology.)
- Iglesias, Law & Stuart (2013). *Ensemble Kalman methods for inverse problems.* Inverse Problems.
  (EKI — §2.3.4.)
- Schneider, Lan, Stuart & Teixeira (2017). *Earth system modeling 2.0.* GRL; and subsequent CliMA
  calibrate-emulate-sample papers. (The ensemble-calibration program Track 1a benchmarks against.)
- Talagrand & Courtier (1987). QJRMS. (Variational assimilation foundations.)
- Parrish & Derber (1992). MWR. (The NMC method for B.)
- Buizza & Palmer (1995). JAS. (Singular vectors, total-energy norm — §2.4.2.)

**Track 3 (ensembles, evaluation):**

- Leutbecher & Palmer (2008). J. Comp. Phys. (Ensemble prediction theory; spread–error, SV-based IC
  perturbations.)
- Ferro, Richardson & Weigel (2008). Met. Apps. (Fair CRPS / ensemble-size adjustment.)
- Rasp et al. (2024). *WeatherBench 2.* JAMES. (Evaluation protocol and baseline scores.)
- Price et al. (2024). *GenCast.* Nature. (Pure-emulator probabilistic baseline.)
- Saltelli et al. (2008). *Global Sensitivity Analysis: The Primer.* (Sobol first-order/total-effect
  decomposition — §4.4.)

**Track 4 (climate protocol):**

- Eyring et al. (2016). GMD. (CMIP6/AMIP experimental design.)
- Gates et al. (1999). BAMS. (AMIP I retrospective — protocol lineage.)



