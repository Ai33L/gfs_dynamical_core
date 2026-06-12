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
   find the steady eddy response **directly** by solving the linearized
   steady-state equation with a matrix-free linear solver. The differentiability
   of the JAX core makes the tangent-linear operator available for free via
   `jax.linearize`. No time-stepping.

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

## 3. Solver: linearized steady-state ("optimizer instead of the dycore")

Write `X = X_bg + x'`. To first order in the eddy `x'`:

```
0 = F(X_bg + x') + G(X_bg + x')  ≈  [F(X_bg) + G(X_bg)]  +  L x'
```

where `L = d(F+G)/dX |_{X_bg}` is the tangent-linear operator. Because `X_bg` is
zonally symmetric, `F(X_bg) + G(X_bg)` has only the zonal-mean heating residual,
whose eddy part is the (zonal-mean-removed) heating `b`. So the **eddy stationary
response** solves the linear system

```
L x' = -b ,   b = forcing projected onto the temperature tendency (eddy only)
```

- `L` is obtained matrix-free from `jax.linearize(total_tendency, X_bg)` — the
  JVP closure. No explicit matrix is formed.
- `L` is **block-diagonal in zonal wavenumber m** (linearization about a
  zonally-symmetric state), so each `m` is independent and the `m=0` null space
  is avoided by the eddy-only forcing.
- The Rayleigh + Newtonian damping makes `L` non-singular on the eddy subspace
  (this is why the paper needs drag).
- Solve with **GMRES** (`jax.scipy.sparse.linalg.gmres`, pytree-valued, complex)
  since `L` is non-symmetric. Fallback if convergence is poor: least-squares via
  CG on the normal equations `L* L x' = -L* b`, or `optax`/`jaxopt` minimization
  of `½‖L x' + b‖²` (this is the literal "optimizer" form; mathematically the
  same minimizer).

The matches the paper's own linear vorticity-balance interpretation (their
Eq. 2), so the diagnostics map directly.

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
- **GMRES conditioning.** Fast gravity/Kelvin modes make `L` stiff. Mitigation:
  the damping regularizes; if needed, increase drag slightly, precondition by the
  diagonal, or switch to the normal-equations / optimizer fallback (§3).
- **Background not exactly steady.** Handled by construction — eddy-only forcing
  means the background's zonal-mean imbalance does not force eddies at linear
  order (§2, §3).

## 8. Out of scope

- Nonlinear steady state (offered as a later cross-check toggle, not built now).
- Time-dependent / transient development (paper Fig 4).
- Tuning to match observed ERA-Interim amplitudes; we aim for the qualitative
  transition and the `psi_MJO` trend.
