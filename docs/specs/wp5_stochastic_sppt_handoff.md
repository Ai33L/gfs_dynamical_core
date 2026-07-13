# Handoff: WP5 — trainable differentiable SPPT (→ WP6 SKEB + latent-noise NN)

**Date:** 2026-07-12
**Branch:** work merged to `jax-port` (C-WP1 keystone, merge `d3edadd`).
**Goal of this track:** a stochastic version of climt for the midlatitudes using
kinetic-energy backscatter (SKEB) with ML components. In the four-track program
(`docs/specs/research_program_four_tracks.md`) this is Track 3 §4, model-
uncertainty source (b): v2 = latent-noise NN + SKEB (H-WP6). Ordered path:
**C-WP1 (done) → H-WP5 (trainable SPPT) → H-WP6 (SKEB + latent-noise NN, fair-CRPS).**

---

## What C-WP1 delivered (the substrate WP5 builds on)

All merged and tested (31 JAX tests + 5 Fortran parity tests green).

- **`gfs_dynamical_core/jax/stepper.py :: advance_with_tendencies(...)`** — pure,
  `jit`/`grad`/`vmap`-able one-step function. Signature:
  ```
  advance_with_tendencies(spec_state, phys_tends, phis_grads, dyn_config,
      trans_config, stepper_config, latitudes, gauss_weights=None,
      pdryini=None, spec_tends=None) -> SpectralState
  ```
  It advances the dynamics one step, then applies a single time-split increment
  assembled from **two** containers (either may be `None`):
  - `phys_tends` : grid-space `PhysicsTendencies` (transformed to spectral internally).
  - **`spec_tends` : spectral-space `SpectralTendencies` — added directly, no
    transform. THIS is the SPPT / SKEB injection point.**
- **`gfs_dynamical_core/jax/states.py :: SpectralTendencies`** — flax struct with
  `d_vorticity_d_t, d_divergence_d_t, d_temperature_d_t,
  d_log_surface_pressure_d_t, d_tracers_d_t`. Build one with zero fields except
  the perturbed field.
- **`GFSDynamicsJAX(tendency_component_list=[...])`** — component-driven
  `TendencyStepper`, drop-in for the Fortran core; full virtual-T/lnps/tracer
  tendency construction, multi-tracer pack/unpack, `air_pressure` outputs,
  post-step moisture clipping.

Gradients through `spec_tends` are verified differentiable
(`tests/test_jax_component_api.py::test_advance_with_spectral_tendencies_is_differentiable`).

---

## WP5 — trainable differentiable SPPT

Design detail: `research_program_four_tracks.md` §3.3.2–3.3.4 (code sketches),
acceptance in the H-WP5 row (§3.7): **spread–error ratio ≈ 1 at 5-day lead on
T2m after CRPS training.**

**What SPPT is here:** a multiplicative AR(1) pattern `r` on the *total physics
tendencies*. `tend' = tend * (1 + clip(σ·r, -0.9, 0.9))`. Trainables: amplitude
σ, decorrelation time τ, length scale L̃ (all `jnp.exp(log_*)` for positivity).
The AR(1) pattern lives in **spectral space** (a Gaussian spatial spectrum over
total wavenumber), carried in the rollout `scan` carry.

**How it plugs in:** SPPT modulates the physics tendencies BEFORE they enter the
increment. Two viable wirings — pick during brainstorming:
1. Scale the grid-space physics tendencies inside a component, then let the
   existing `phys_tends` path carry them (simplest; SPPT as a tendency wrapper).
2. Or generate the perturbation spectrally and pass via `spec_tends` (keeps the
   pattern native-spectral; better shared code path with SKEB).

**Hard invariants (do not skip):**
- Every spectral noise field MUST pass the **reality-symmetry guard**
  (`assert_reality_symmetric`) — same invariant the whole spectral pipeline
  relies on. Sample complex noise respecting the reality condition; see the
  §3.3.4 `sppt_pattern_step` sketch and `enforce_triangular_truncation`.
- The AR(1) pattern state is part of the `scan` carry, not global.
- Reverse-mode rollouts must wrap the step in `jax.checkpoint` (C-WP2 memory
  budget, §1 of the research doc) — a 5-day T62 rollout is ~10 GB of residuals
  otherwise.

**Training:** fair-CRPS estimator (§3.3.4 `fair_crps`) over M=8–16 members via
`vmap` over noise keys, **common random numbers across each gradient eval**.
Verify T2m/Z500 at 3–10 d leads. Spread–error ratio and rank histograms are the
overfitting alarms — monitor, don't train on them.

**Suggested first milestone:** trainable SPPT wired through a checkpointed
`lax.scan` rollout of `GFSDynamicsJAX` at low resolution (T21/T42), CRPS loss on
a small batch, showing spread grows and the spread–error ratio moves toward 1
after a few optimizer steps. Then scale up.

---

## WP6 (after WP5) — SKEB + latent-noise NN

- **SKEB:** additive perturbation to the *vorticity* tendency, generated natively
  in spectral space → inject via `spec_tends.d_vorticity_d_t` (this is exactly
  why the dual path exists). Trainable backscatter amplitude/spectrum. Targets
  the rotational flow that controls Rossby-wave / blocking dynamics — the
  midlatitude spread that matters. Reuses the WP5 spectral-noise machinery.
- **Latent-noise NN:** correlated random fields fed as NN inputs; the network
  shapes the response. Subsumes SPPT. fair-CRPS trained. Acceptance (H-WP6):
  beats trained SPPT on CRPS for Z500 & T2m at 3–10 d.

### Note: is classic SKEB flow-dependent? (amplitude yes, structure no)

Verified against Palmer et al. 2009 (Tech Memo 598, §2.2.2–2.2.3, App. 8.2).
Classic SPBS/SKEB is `F_ψ = √(b_R · D_tot / B_tot) · F_ψ*` — **two factors, and the
flow-dependence lives in only one:**

- `F_ψ*` (the random pattern) is **flow-independent** — a spectral AR(1) field with
  a *fixed* power-law spectrum `g(n)`; same machinery as the WP5 SPPT generator,
  just for streamfunction. Which structures it excites does not depend on state.
- `√D_tot` (the amplitude modulator) is **flow-dependent** — `D_tot(x,y,z,t)` is the
  total dissipation rate (§2.2.3: **numerical** = explicit biharmonic + implicit
  diffusion energy loss, **gravity-wave-drag**, and **convective** dissipation). So
  backscatter is strong where the flow actively dissipates energy (sharp gradients,
  active jets, convection, mountains). `b_R` is the scalar backscatter ratio.

So classic SKEB gives flow-dependent **amplitude/location**, *not* flow-dependent
**structure**. It energizes dynamically active regions (correlated with baroclinic
activity) but does not preferentially excite the ridge-amplification / blocking-onset
directions that define a heatwave event. **Three rungs of flow-dependence:**

1. **SPPT** — perturbation location follows the physics-tendency field (weak,
   incidental); pattern fixed.
2. **Classic SKEB** — amplitude follows `√D_tot` (genuine physical flow-dependence
   in *where*); pattern still fixed-spectrum.
3. **Learned SKEB / latent-noise NN** — the network makes the *structure*
   state-dependent (excite the event-relevant growing modes). This is the
   flow-dependence RES event-targeting actually needs (§3.3.3 "relevance").

**Dry-Held-Suarez caveat:** `D_tot`'s convective and GWD components do not exist in
the dry core. Only **`D_num`** survives — and we already have that operator (the
hyperdiffusion `disspec` in `StepperConfig`). A classic SKEB here is
`pattern × √D_num`: genuinely flow-dependent (backscatter where hyperdiffusion
removes energy — small scales, sharp gradients, active jets), but thinner than the
IFS `D_tot`. The differentiable version keeps the `√D_tot` amplitude prior "for
free" and makes `b_R`, the spectrum `g(n)`, and ultimately a state-dependent
amplitude field trainable — that trainable state-dependent amplitude is rung 3.

#### Restoring `D_conv` with the JAX Emanuel convection scheme

The dry-HS `D_num`-only limitation lifts if the pipeline runs with the JAX Emanuel
convection scheme (`~/github/climt`, `climt/_components/emanuel/pure_python_v3.py`).
Its `array_call` exposes everything needed to infer a convective energy-input rate:
tendencies `FT` (convective heating, K/s), `FQ, FU, FV`; and diagnostics
`cloud_base_mass_flux` (CBMF, kg m⁻² s⁻¹), `atmosphere_convective_available_potential_energy`
(CAPE, J/kg), `convective_precipitation_rate` (P), `convective_downdraft_velocity_scale`
(`wd`, m/s). Three estimators, most direct first:

1. **KE generation from CAPE consumption (recommended):**
   `D_conv ≈ CBMF · CAPE`  — units `kg m⁻² s⁻¹ · J kg⁻¹ = W m⁻²`, the
   column-integrated rate convection converts APE→draft KE, i.e. exactly the
   energy-input rate `B` that SKEB re-injects a fraction `b_R` of. Both factors are
   direct outputs; it is zero where convection is inactive (`IFLAG`/CBMF=0), which
   is the desired flow-dependence.
2. **Heat-engine cross-check:** `D_conv ≈ ε · L_v · P_conv` with `ε` the convective
   thermodynamic efficiency (few %); agrees with (1) in quasi-equilibrium.
3. **Downdraft dissipation (local, near-surface):** `D_down ≈ ρ · wd³ / ℓ` from the
   downdraft velocity scale — complementary cold-pool/downdraught term.

**Vertical distribution:** SKEB needs `D_conv(x,y,z)` on model levels; use the
per-level convective heating profile `FT` as the shape function to spread the
column-integrated rate from (1) vertically (backscatter where convection is
thermodynamically active).

**Two payoffs:** (a) restores a real `D_tot = D_num + D_conv`, so classic SKEB is
genuinely flow-dependent in amplitude over convective regions, not just numerically;
(b) since the Emanuel scheme is differentiable and CBMF/CAPE/P/`wd`/`FT` are its
outputs, `D_conv` — and hence `√D_conv` — is a differentiable function of state, so
the flow-dependent-amplitude SKEB (and a learned residual on top of the `√D_conv`
prior) is end-to-end trainable. It also gives SPPT real leverage: a moist model has
large physics tendencies (`FT, FQ`) to multiply, unlike dry HS.

**Caveat:** the exact operational SPBS `D_conv` formula is in Berner et al. 2009
(MWR), not yet in `docs/refs/`. The estimators above are physically derived and
standard; add the Berner PDF to pin the operational form verbatim.

---

## Open items / decisions carried forward

- **`zero_negative_moisture`** is now implemented (clip tracer-0 in grid space +
  re-transform to spectral, updating the cached state). It adds a per-step
  tracer-0 forward transform when enabled (default). Inert for dry runs. If it
  becomes a hot spot at scale, consider jitting the clip+re-transform.
- **`set_physics_tendencies` (manual path)** applies plain `t_t` without the
  `fvirt·q` virtual-T correction the component path uses — the two paths diverge
  for nonzero-humidity manual users. Fine at q=0. Reconcile if the manual path
  is used with humidity.
- **Data pipeline (C-WP3)** — WP5 CRPS training against ERA5 needs the ERA5 →
  `SpectralState` ingest (research doc §1 C-WP3). Not started; may be stubbed
  with idealized ICs for the first milestone.

## Where to start next session

Brainstorm the WP5 spec (superpowers:brainstorming): decide the SPPT wiring
(tendency-wrapper vs spectral `spec_tends`), the AR(1) pattern parameterization,
and the first-milestone resolution/loss. Then writing-plans → execute.
See memory `project_stochastic_skeb_track`.
