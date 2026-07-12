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
