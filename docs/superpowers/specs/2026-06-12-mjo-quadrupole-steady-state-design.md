# Reproducing the MJO upper-level quadrupole as a steady-state solve on the GFS JAX core

**Date:** 2026-06-12
**Status:** Design — awaiting review
**Author:** Joy Monteiro (with Claude)

## 1. Goal

Reproduce the central result of Monteiro, Adames, Wallace & Sukhatme (2014,
*GRL*, "Interpreting the upper level structure of the Madden-Julian
oscillation"): impose an equator-straddling tropical heating and show that, as a
zonally-symmetric subtropical westerly jet is strengthened from rest toward
realistic speed, the upper-level eddy response transitions from the
equatorially-trapped **Matsuno-Gill** pattern, through a **tilted Rossby wave
train**, into a compact **quadrupole** of flanking Rossby gyres at ~28°N/S.

Two differences from the 2014 paper, both deliberate:

1. **Model.** The paper used a single-layer spherical *shallow-water* model. Here
   we use the project's own **3-D primitive-equation GFS dynamical core**
   (`gfs_dynamical_core.jax`), exercising the JAX port. The quadrupole mechanism
   (Eq. 2 of the paper: meridional advection of planetary vorticity + advection
   of relative vorticity by the mean jet) is generic, so the qualitative
   transition should survive the change of model — this is a reproduction on a
   *different* model, not a pixel match.

2. **Method.** Instead of time-integrating to a steady state (paper: day 99), we
   find the steady state **directly** by minimizing the full nonlinear
   total-tendency residual `||F(X) + G(X)||^2` with a gradient-based optimizer
   (L-BFGS), taking the gradient through the differentiable core with `jax.grad`.
   No time-stepping, no linearization of the physics (see §3).

## 2. Physical formulation

The dycore is a map `dX/dt = F(X)` where `X = (vorticity, divergence,
temperature, log_surface_pressure)` in spectral space (tracers held fixed /
ignored; the experiment is dry). We add a forcing/damping term `G`:

```
dX/dt = F(X) + G(X)
G(X) = -(vort - vort_bg)/tau_M            # Rayleigh friction on rotational wind
       -(div  - 0)/tau_M                  # Rayleigh friction on divergent wind
       -(T    - T_eq)/tau_T               # Newtonian relaxation toward T_eq
T_eq = T_bg(sigma) + dT(lambda, phi, sigma)   # the imposed temperature anomaly
```

`tau_M = 20 days`, `tau_T = 10 days` (paper values).

### Background state `X_bg` (the jet — swept)

Zonally symmetric, equatorially symmetric.

- Zonal wind `U_bg(phi, sigma) = U_max * J(phi) * V(sigma)` where `J(phi)` peaks
  at ~30°N/S (e.g. `sin(2*phi)`-type lobe vanishing at equator and pole, or the
  paper's `1-(sin phi)^2` family) and `V(sigma)` increases upward (subtropical
  jet is upper-tropospheric). `V_bg = 0`.
- `U_max` is the **swept parameter**: `{0, 8, 16, 24, 30} m/s` (paper Fig 2a–f,
  via `H0 = 0, 500, 1000, 1500, 2000 m`).
- Background vorticity `vort_bg = -1/(a cos phi) d(U_bg cos phi)/d phi`, computed
  spectrally; `div_bg = 0`.
- Reference temperature `T_bg(sigma)` is a function of height only (e.g. a
  realistic tropical mean sounding or isothermal ~250 K). Because `X_bg` is
  zonally symmetric, `F(X_bg)` is zonally symmetric and contributes **no eddy
  forcing** — so `T_bg` need not be in exact thermal-wind balance with the jet
  for the *eddy* problem (see §3). This is standard linear-stationary-wave
  practice.

### Forcing (the heating — fixed)

Equator-straddling Gaussian centred at 90°E (paper values):

```
dT(lambda, phi, sigma) = A * exp(-((lambda - 90E)/Lx)^2 - (phi/Ly)^2) * W(sigma)
```

- `Lx = 30 deg`, `Ly = 10 deg`, centre `phi_p = 0` (straddles equator).
- `W(sigma)`: deep tropospheric vertical profile peaking mid-troposphere
  (canonical Gill/MJO heating), zero at top and surface.
- `A`: a few K — small enough to stay in the linear regime (paper shows the
  response is essentially linear in heating amplitude).
- **Zonal mean removed:** following the paper, subtract the zonal mean of `dT` so
  the forcing is purely eddy and no zonal-mean response develops. This makes the
  eddy response cleanly attributable to the heating.

## 3. Solver: nonlinear steady-state via gradient-based optimization

We solve the **full nonlinear** steady-state problem — no linearization. The
steady state is where the total tendency vanishes, `F(X) + G(X) = 0`. We find it
by minimizing the residual norm with a gradient-based optimizer, taking the
gradient through the differentiable core with `jax.grad`:

```
minimize   J(theta) = ½ * sum_i  w_i * || [F(X) + G(X)]_i ||^2
over        theta = (u, v, T, lnps)  real grid-space fields
where       X = grid_to_spectral(theta)         # reality + truncation enforced
            i ranges over prognostic vars (vorticity, divergence, temperature, lnps)
```

Design choices that make this well-posed and differentiable:

- **Optimize in real grid space** `theta = (u, v, T, lnps)`, not the complex
  spectral state. `grid_to_spectral(theta)` (differentiable) maps to the spectral
  state and automatically enforces the reality condition and triangular
  truncation, so we never optimize redundant/complex DOF or leave the physical
  manifold. Mirrors how `component_jax` builds initial conditions (u, v with
  vorticity/divergence derived).
- **Forcing `G`** is added to the spectral dynamics tendency: Rayleigh relaxation
  `-(vort - vort_bg)/tau_M`, `-(div)/tau_M`, and Newtonian
  `-(temp - Teq)/tau_T`, with `Teq = grid_to_spectral(T_bg + dT)`.
- **Per-variable weights `w_i`** normalize the very different tendency
  magnitudes/units across (vorticity ~1e-10, divergence, dT/dt, dlnps/dt) so the
  optimizer balances them rather than chasing the stiff gravity-wave components.
  Characteristic scales from the background + forcing; a tunable knob.
- **Initial guess** `theta_0 = (U_bg, 0, T_bg, lnps_ref)` — the background jet
  with zero eddy. The optimizer grows the stationary-wave response that balances
  the heating against advection + damping.
- **Optimizer:** L-BFGS (`optax.lbfgs`, fallback `jax.scipy.optimize.minimize`
  BFGS / Adam) on the scalar objective `J`. The problem is a nonlinear
  least-squares, so L-BFGS converges quickly near the minimum; run to a residual
  tolerance or a fixed large iteration budget. Damping (`tau_M`, `tau_T`)
  guarantees the minimizer exists and is finite.

The eddy response plotted in §4 is `X - zonal_mean(X)` (equivalently the
deviation from the zonally-symmetric background). Because the heating's zonal
mean is removed (§2), the zonal-mean state stays close to `X_bg` and the eddies
are cleanly attributable to the heating.

**Why nonlinear (vs a linear solve):** the paper reports the response is *nearly*
linear, but solving the true nonlinear residual (a) needs no
zonally-symmetric-background assumption, (b) captures any amplitude-dependent
distortion of the quadrupole, and (c) is the literal "optimizer through the
differentiable dycore" demonstration. Convergence is the main risk (§7).

## 4. Sweep & outputs

Loop `U_max in {0, 8, 16, 24, 30} m/s`; for each, build `X_bg`, the operator `L`,
solve for `x'`, and transform to grid.

**Figure (mirrors paper Fig 2):** a 5- or 6-panel grid. Each panel shows, at an
upper-tropospheric level (sigma ~ 0.2, ~200 hPa):
- eddy streamfunction `psi' = ∇⁻² zeta'` (inverse Laplacian of eddy vorticity) or
  eddy geopotential height, as filled contours;
- eddy horizontal wind as vectors;
- the heating footprint marked (ellipse at 90°E / equator);
- jet-maximum latitude lines.

Expected: panel 0 (rest) → Matsuno-Gill, tropically trapped; panel ~1 (8 m/s) →
tilted wave train; panels at 24–30 m/s → quadrupole at ~28°N/S, cyclonic gyres
east of the heating, anticyclonic west, Kelvin response on the equator largely
unchanged.

**Scalar diagnostic:** per panel, the paper's `psi_MJO = max|geopotential| /
max|wind|`, expected to climb (paper: 8 → 36) as the jet strengthens — a compact
numerical signature that the quadrupole has formed.

## 5. Numerical configuration

- Resolution: `L = 64` (`n_lon = 127`, `n_lat = 64`), `n_lev = 20` — matches the
  example and the JAX core defaults. GL sampling.
- Constants / vertical coordinate (`ak`, `bk`): obtained from `climt` exactly as
  in `examples/baroclinic_wave_jax.py` (`set_constant`, `climt.get_grid`,
  `climt.get_default_state`), so the dycore config is built identically to
  production use.
- Adiabatic, dry: call `get_spectral_tendencies` with the dry-mass fixer
  disabled (`gauss_weights=None, pdryini=None, dt=None`); flat topography
  (`phis = 0`, so `phis_grads = (0, 0)`).
- `float64` (`JAX_ENABLE_X64=True`), CPU.

## 6. Deliverable

A single self-contained script `examples/mjo_quadrupole_steady_state.py` that:
1. builds the dycore config from climt,
2. defines background-jet, heating, and damping builders,
3. assembles `total_tendency` and the linearized solve,
4. sweeps `U_max`, solves, and
5. writes the multi-panel figure + prints the `psi_MJO` table.

Runtime target: minutes on CPU (5 linear solves at L=64).

## 7. Risks & mitigations

- **Quadrupole may not emerge cleanly on a PE core.** Mechanism is generic, but
  vertical structure / jet baroclinicity differ from shallow water. Mitigation:
  if the upper-level transition is muddy, concentrate the jet and heating in the
  upper troposphere (closer to the paper's single upper layer) before falling
  back to a single-active-layer reduction.
- **Optimizer convergence / stiffness.** Fast gravity/Kelvin modes give large,
  stiff residual components. Mitigations: per-variable weighting `w_i` (§3),
  good initial guess (background jet), L-BFGS with line search, and increasing
  damping slightly if the residual plateaus. Monitor `J` and the max grid-space
  tendency to confirm a true steady state, not just a flat optimizer.
- **Local minima / non-convergence.** If L-BFGS stalls far from zero residual,
  warm-start from a weaker-jet solution (continuation in `U_max`) so each solve
  starts near its neighbour. The rest case (`U_max=0`) is the easiest and seeds
  the sweep.

## 8. Out of scope

- Time-dependent / transient development (paper Fig 4).
- Tuning to match observed ERA-Interim amplitudes; we aim for the qualitative
  transition and the `psi_MJO` trend.

## 9. Implementation log

- **2026-06-12 (numerical test):** First nonlinear runs (L=24) converge but
  stall: `J/J0` plateaus at ~0.62 after 800 L-BFGS iters and the eddy response
  is negligible (psi_MJO ~0.7, no quadrupole). **Diagnosis:** the objective is
  dominated (~1e5x) by the **divergence** tendency. That residual (~1e-9 s^-2) is
  the *background jet's gradient-wind imbalance* — `T_bg(sigma)` has no meridional
  gradient, so `F(X_bg) != 0` and the optimizer spends its budget slowly
  balancing the jet rather than growing the heating response. **Fix:** build a
  gradient-wind/thermal-wind **balanced** background temperature from `U_bg` so
  `F(X_bg) ~ 0`; then the heating (temperature residual) dominates and the
  optimizer grows the stationary-wave response. (code-reading + numerical-test)
- **2026-06-12 (numerical test, cont.):** Floored r0-normalised weighting fixed
  the conditioning — rest case converges to `J/J0~1e-5`, strong jet descends
  steadily. BUT the eddy response is a **global zonal-wavenumber-1** dipole, not
  a heating-localized Gill/quadrupole, in BOTH rest and U=30. Confirmed NOT a
  weak-damping free-mode artifact: persists with good convergence at tau=3 d.
  **Interpretation:** a *deep* heating in the 3-D PE core projects strongly onto
  the gravest (barotropic/external) vertical mode, whose deformation radius is
  ~global, so the response delocalizes — unlike the paper's single-layer
  shallow-water model (one baroclinic mode, deformation radius ~10-15 deg, hence
  localized). This is the PE-vs-shallow-water mismatch (risk in S7), not a solver
  bug. **Next decision (user):** reduce to an equivalent single active layer /
  project heating onto one baroclinic mode / confine + damp the external mode.
  (numerical-test)
