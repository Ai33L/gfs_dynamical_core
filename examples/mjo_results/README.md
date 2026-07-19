# MJO upper-level structure as a nonlinear steady-state solve on the GFS JAX core

Experiment inspired by Monteiro, Adames, Wallace & Sukhatme (2014, *GRL*,
"Interpreting the upper level structure of the Madden-Julian oscillation"):
impose an equator-straddling tropical heating at 90E and sweep a zonally-
symmetric subtropical westerly jet from rest (0 m/s) to realistic strength
(30 m/s), looking for the transition from the Matsuno-Gill response to the
upper-level **quadrupole** of flanking Rossby gyres.

## What is novel here

1. **The dycore is never time-stepped.** The steady state is found *directly* by
   minimising the total-tendency residual `||F(X) + G(X)||^2` with the L-BFGS
   optimizer, taking gradients straight through the differentiable GFS JAX core
   (`get_spectral_tendencies`) with `jax.grad`. This exercises the JAX port as a
   differentiable model.
2. **The model is the full 3-D primitive-equation core**, not the single-layer
   shallow-water model of the 2014 paper.

Forcing `G` = Newtonian relaxation of temperature toward `T_bg + dT` (the imposed
deep, equator-straddling heating, zonal mean removed) + Rayleigh friction toward
the gradient-wind-balanced background jet. Optimisation variables are the real
grid fields `(u, v, T, lnps)`; `grid_to_spectral` (differentiable) maps them to
the spectral state and enforces the reality/truncation conditions.

## How to reproduce

```
JAX_PLATFORMS=cpu JAX_ENABLE_X64=True python examples/mjo_quadrupole_steady_state.py \
    --L 32 --maxiter 1500 --tau-m 3 --tau-t 3 \
    --u-sweep 0 8 16 24 30 --out examples/mjo_results/mjo_quadrupole_sweep.png
```

`run.log` is the optimiser trace; the rest case converges to `J/J0 ~ 1e-3` in
~110 iterations, the jet cases warm-start from it. Residuals reach ~1e-10.

## Result and interpretation

![sweep](mjo_quadrupole_sweep.png)

**The compact subtropical quadrupole does NOT emerge in the deep 3-D core.** As
the jet strengthens:

- the **rotational (streamfunction) response** stays a broad *planetary-scale*
  pair of cells (negative west of the heating, positive east, equatorially
  symmetric) that intensifies modestly but does not coalesce into four compact
  gyres tightly flanking the jet at ~28N/S;
- the **upper-level geopotential** is dominated by a localized low over the
  heating plus a weak global far-field;
- the geopotential/wind ratio *decreases* (3405 -> 2707) with jet strength — the
  opposite of the paper's increase (8 -> 36) that signals quadrupole formation.

**Why:** the 2014 quadrupole is a *single-baroclinic-mode* phenomenon. The
paper's shallow-water model has one vertical mode with an equatorial deformation
radius of ~10-15 deg, which localizes the Gill/quadrupole pattern. A *deep*
heating in the full 3-D primitive-equation core projects strongly onto the
gravest (barotropic/external) vertical mode, whose deformation radius is
near-global, so the response delocalizes onto planetary scales and the compact
quadrupole mechanism (meridional advection of planetary vorticity + advection of
relative vorticity by the mean jet, balanced over the deformation radius of one
baroclinic mode) does not get a chance to organise.

This is a faithful, honest contrast between the two models, not a tuning failure:
the steady states are well converged (residuals ~1e-10) and the conclusion is
robust to damping strength (tested tau = 3 and 10 days).

To *recover* the quadrupole one would confine the dynamics to a single baroclinic
mode — e.g. a half-wave heating that projects onto the first internal mode with
the barotropic component strongly damped, or an equivalent single active layer.
That reduction was deliberately not taken here (see the design doc, section 9).

See `docs/superpowers/specs/2026-06-12-mjo-quadrupole-steady-state-design.md`.
