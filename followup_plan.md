# Follow-Up Implementation Plan: Attaining Byte-for-Byte Equivalence

Now that the core mathematical machinery (PGF, advection, IMEX stepper, and vector transforms) has been ported, the next critical step is to reconcile the structural, scaling, and discretization differences between the JAX ecosystem (`s2fft`) and the Fortran ecosystem (`shtns` / `climt` configuration), and fix several algorithmic gaps discovered during code review.

## Phase 6: Spectral Grid & Truncation Reconciliation
- [x] **Task 6.1: Decouple Bandlimit from Grid Size**
  - Fortran uses an oversampled Gaussian grid (e.g., $N_{lat}=32, N_{lon}=64$) with an independent triangular truncation limit $T=19$.
  - JAX's `s2fft` strictly couples the grid size to the bandlimit $L$ (e.g., $N_{lon} = 2L - 1$).
  - **Action:** Create a spatial interpolation or padding wrapper in JAX that safely truncates or pads the $N_{lon}=64$ input arrays to $N_{lon}=63$ before calling `s2fft.forward_jax`, and interpolates back to $N_{lon}=64$ after `s2fft.inverse_jax`. Alternatively, adjust `L` to 33 ($2L-1 = 65$) and zero-pad/slice the grid, explicitly zeroing out harmonics where $\ell > T$.
- [x] **Task 6.2: Enforce Triangular Truncation ($T$)**
  - Even with a matched grid, JAX calculates all harmonics up to $L-1$. Fortran explicitly truncates at $T$.
  - **Action:** Apply a masking array to the JAX `SpectralState` to strictly zero out all spherical harmonic coefficients where the degree $\ell > T$ or order $m > T$, exactly mimicking Fortran's truncation bounds.

## Phase 7: Normalization & Scaling Alignment [✅ RESOLVED]
- [x] **Task 7.1: Reconcile Spherical Harmonic Normalization** [Priority 1 — Resolved by analysis, no code change needed]
  - **Finding:** The only difference between `s2fft` and SHTns `SHT_ORTHONORMAL+SHT_NO_CS_PHASE` is the Condon-Shortley phase `(-1)^m` applied to positive-order columns (`m > 0`). The amplitude normalization is **identical** between the two libraries. The CS phase is purely a coefficient storage convention.
  - **Why no fix is needed:** Every path through the JAX pipeline that touches spectral coefficients uses `s2fft.forward_jax` and `s2fft.inverse_jax` symmetrically (Laplacian operators depend only on degree $\ell$, not the sign of $m$; the truncation mask depends only on $|\ell|$ and $|m|$). The phase cancels end-to-end. Grid-space fields are correct without any correction.
  - **Cost:** Spectral coefficient arrays in `SpectralState` cannot be directly byte-compared against Fortran's SHTns arrays. Phase 11 validation must be done in grid space, which the sub-tasks already specify.
  - **Empirical verification:** For $Y_{3,1}$ (a positive-$m$ case where CS phase flips sign), `s2fft` and the CS-corrected field produce grid values that differ by exactly a factor of $-1$, confirming the analysis. The amplitude is the same.
- [x] **Task 7.2: Vector Transform Amplitude Scaling** [Priority 4 — Confirmed correct, no code change needed]
  - **Finding:** Five independent analytical checks passed at machine precision ($< 10^{-15}$):
    1. Internal round-trip `(vort, div) → grid → (vort, div)` for five different `(l, m)` combinations.
    2. Divergence amplitude: `D_{1,0}=1` produces `v(θ) = -(R/2)·√(3/4π)·sin θ`, matching Fortran `getuv` exactly.
    3. Vorticity amplitude: `ζ_{2,0}=1` produces `u(θ) = (R/2)·√(5/4π)·cos θ sin θ`, matching Fortran `getuv` exactly.
    4. Scalar gradient amplitude: `lnps = Y_{1,0}` produces `∂lnps/∂φ = (1/R)·√(3/4π)·sin θ`.
    5. Radius scaling: wind amplitude scales exactly as `1/R` between `R=1` and `R=6371000`.
  - **Action taken:** The commented-out assertions in `test_jax_transforms.py` (vector round-trip test) were **re-enabled** and confirmed passing at `atol=1e-10`.

## Phase 8: Algorithmic Corrections [✅ COMPLETE]
- [x] **Task 8.1: Add Tracer Horizontal Advection** [Priority 2]
  - Fortran computes tracer tendencies as `-u * d_tracer/d_lambda - v * d_tracer/d_phi - vadv_tracer` (horizontal + vertical advection). The JAX `assemble_grid_tendencies` previously only computed `-vadv_tracer` (vertical advection only), **missing the dominant horizontal advection terms entirely**.
  - **Implementation:**
    - Added `d_tracers_d_phi: jnp.ndarray` and `d_tracers_d_lambda: jnp.ndarray` fields (shape `(n_tracers, levels, n_lat, n_lon)`) to `GridGradients` in `states.py`.
    - Extended `spectral_to_grid` in `transforms.py` to compute spectral gradients of each tracer field using the same spin-1 approach already used for temperature gradients (`F1_lm = -l_factor * tracer_lm`, then spin-1 inverse transform). Gradients are computed inside the per-level `vmap` then transposed to `(n_tracers, levels, …)` convention.
    - Updated `compute_tracer_tend` in `assemble_grid_tendencies` (`dynamics.py`) to `return -u * dq_dlambda - v * dq_dphi - vadv_q`, with the new gradient fields passed through `jax.vmap`.
  - **Reference:** Fortran `getdyntend`, lines ~392–402 in `dyn_run.f90`.
  - **Regression test:** `test_tracer_horizontal_advection` in `test_jax_dynamics.py` verifies that a uniform `u=10 m/s` wind with `dq/dλ=1e-5` produces exactly `-u·dq/dλ = -1e-4` as an additional tendency increment, to `rtol=1e-10`.
- [x] **Task 8.2: Fix Pressure Gradient Force Dual-`cofb` Issue** [Priority 3]
  - In Fortran's `getpresgrad`, the variable `cofb` is computed **twice** with different meanings:
    1. First as a pressure-coordinate coefficient (`cofb_pressure`): `cofb(k) = -(1/dpk(k)) * (bk(k)*rlnp(k) + alfa(k)*dbk(k))`, used for the `px0` term: `cofb_pressure * rd * T * ps * grad(lnps)`.
    2. Then **overwritten** as a cumulative geopotential integral (`cofb_geopotential`): `cofb(nlevs)=0`, accumulated bottom-to-top as `cofb(k) = cofb(k+1) - rd*(bk(k+2)/pk(k+2) - bk(k+1)/pk(k+1))*T(k)`, and added as the `px2` term: `cofb_geopotential * ps * grad(lnps)`.
  - The previous JAX implementation only had the first `cofb` and used it in a single combined expression, **missing the `px2` geopotential integral term entirely**.
  - **Implementation:** In `compute_pressure_gradient_force` (`dynamics.py`):
    - Renamed original `cofb` → `cofb_pressure`.
    - Added `cofb_geopotential` (`px2_factor`): computed as `delta[i] = -rd*(bk[i+1]/pk[i+1] - bk[i]/pk[i])*T[i]`, assembled via `jnp.cumsum` on the reversed level sequence (with top-layer delta zeroed), then flipped back to top-down order.
    - Final PGF assembly now explicitly labels all five terms `px0`–`px5` with comments matching the Fortran notation.
  - **Reference:** Fortran `getpresgrad`, lines ~527–564 in `dyn_run.f90`.
  - **Regression test:** `test_pgf_cofb_geopotential_term` in `test_jax_dynamics.py` verifies that the PGF vertical profile is **not** uniform across levels when `grad(lnps) ≠ 0`, which would be the case if `px2` were still missing.

## Phase 9: Grid & Coordinate Fixes [🟡 IN PROGRESS]
- [x] **Task 9.1: Use Gaussian Quadrature Latitudes** [Priority 5 — ✅ Complete]
  - `component_jax.py` previously used `jnp.linspace(-pi/2, pi/2, n_lat)` for latitudes. Fortran uses **Gauss-Legendre quadrature points** from SHTNS (`shtns_cos_array`). This caused systematic errors in the Coriolis force ($f = 2\Omega\sin\phi$) and any latitude-dependent computation.
  - **Implementation:** Added `get_gaussian_latitudes(L)` to `transforms.py`, which uses `np.polynomial.legendre.leggauss(L)` to compute the exact GL quadrature points matching `s2fft`'s internal GL grid definition. Ordering is north-to-south (colatitude ascending), matching `s2fft` row order.
  - **Key discovery:** `climt.get_grid()` already provides Gaussian quadrature latitudes by default (verified to machine precision). This resolved the issue cleanly.
  - **Related:** This fix was delivered as part of the broader grid architecture simplification below (see Phase 9.0).
- [ ] **Task 9.2: Audit `rlnp[0]` Sentinel Value**
  - Fortran sets `rlnp(:,:,1) = 99999.99` as a sentinel for the top layer (should never be used). JAX sets `rlnp[0] = 0.0`. While this *should* be harmless, verify that no code path (particularly in `cofb`, `cofa`, or `px3u`/`px3v`) accidentally references `rlnp[0]`. If any path does, the JAX code will silently produce wrong-but-plausible results rather than flagging an error.
  - **Note:** With the Task 8.2 fix now in place, `cofb_pressure` at the bottom layer (JAX index `nlevs-1`) is correctly set to the special case `-(1/dp)*alfa*dbk` (no `rlnp` term), mirroring Fortran's `cofb(k=1)` special case. However the full audit of all `rlnp[0]` references is still warranted.

## Phase 9.0: Grid Architecture Simplification [✅ COMPLETE]
- [x] **Eliminate unnecessary longitude resampling**
  - **Discovery:** Every timestep was resampling inputs from climt's grid (e.g. `n_lon=64`) to `s2fft`'s native grid (`2L-1=63`) and back, introducing FFT interpolation artifacts that accumulate over many timesteps.
  - **Root cause:** The code was treating climt's grid as a fixed external constraint, but climt has no fixed grid — it creates one based on the requirements of individual components. `GFSDynamicsJAX` should simply declare its native sizes and climt provides matching arrays.
  - **Implementation:**
    - Removed `n_lat` and `n_lon` as constructor fields from `TransformConfig`; they are now derived `@property` values from `L` and `sampling`.
    - Removed `resample_lon_to_s2fft()` and `resample_lon_from_s2fft()` entirely — both functions and all call sites deleted.
    - `spectral_to_grid`, `grid_to_spectral`, and `grid_to_spectral_tendencies` now operate directly on native s2fft grid sizes with no resampling.
    - `component_jax.py` simplified: arrays arrive already at native size, output returned directly without shape-patching.
    - Tests updated to request `nx=2*L-1, ny=L` from `climt.get_grid()` — both JAX and Fortran components accept this grid.
  - **Verification:** `climt.get_grid(nx=2*L-1, ny=L)` provides Gaussian quadrature latitudes matching `leggauss(L)` to machine precision, confirming no upstream latitude fix is needed in climt.
  - **Result:** ~50 lines removed from the transform pipeline; no per-timestep interpolation artifacts; Task 9.1 (Gaussian latitudes) resolved as a side effect.

## Phase 10: Physical Conservation Adjustments [🟢 NEEDED FOR MULTI-STEP]
- [ ] **Task 10.1: Implement Dry Mass Fixer** [Priority 6]
  - Fortran dynamically corrects the surface pressure to conserve initial dry mass after physics (see `dry_mass_fixer` in `dyn_run.f90`). This computes `pdry = sum(areawts * psg) - g * sum(areawts * pwat)`, determines a proportional correction `pcorr = (pdryini + g*pwat_global) / ps_global_mean`, and applies it as a multiplicative adjustment to `ps` before converting back to a spectral `lnps` tendency.
  - **Action:** Port the `dry_mass_fixer` subroutine into JAX. This requires Gaussian quadrature weights (`areawts`) for global integrals. Wire it into the time stepper after physics tendencies are applied (or at the end of `advance` for adiabatic runs if `massfix=True`).

## Phase 11: Final Step-by-Step Validation [🟢 VERIFICATION]
- [ ] **Task 11.1: Component Isolation Testing** [Priority 7]
  - Temporarily disable the time-stepper in both models and output purely the initialized tendencies (Temp, Div, Vort, LnPs, **Tracers**) to prove that the initial conditions and spatial derivatives match exactly $\sim 10^{-12}$.
  - **Sub-tasks:**
    - Compare spectral-to-grid round-trip for a known spherical harmonic.
    - Compare pressure diagnostics (`pk`, `dp`, `alfa`, `rlnp`) given identical `lnps`.
    - Compare grid-space tendencies (PGF, vertical advection, energy conversion, **tracer advection**) given identical grid states.
    - [x] ~~Un-comment and pass the vector round-trip assertions in `test_jax_transforms.py`.~~ *(Done in Phase 7.2 — assertions re-enabled and passing at `atol=1e-10`.)*
- [ ] **Task 11.2: Time-Stepper Activation**
  - Turn the RK3 stepper back on and verify that the first explicit full timestep maintains the expected machine-precision equivalence across the entire output state.
- [ ] **Task 11.3: Multi-Step Stability Check**
  - Run both models for 10+ timesteps and verify that errors do not grow exponentially. If they do, the source is likely a remaining scaling or sign error in the transform layer.

---

## Summary of Priorities

| Priority | Task | Phase | Status | Impact |
|----------|------|-------|--------|--------|
| ~~🔴 1~~ | ~~7.1 — SH Normalization~~ | 7 | ✅ Resolved by analysis — no code change needed | CS phase is a convention difference only; grid-space output is correct |
| ~~🔴 2~~ | ~~8.1 — Tracer Horizontal Advection~~ | 8 | ✅ Implemented & tested | `-u·dq/dλ - v·dq/dφ` terms now included |
| ~~🔴 3~~ | ~~8.2 — PGF Dual-cofb Fix~~ | 8 | ✅ Implemented & tested | All 5 PGF terms (px0–px5) now present |
| ~~🟡 4~~ | ~~7.2 — Vector Transform Scaling~~ | 7 | ✅ Verified correct — no code change needed | 5 analytical checks pass at `< 1e-15` |
| ~~🟡 5~~ | ~~9.1 — Gaussian Latitudes~~ | 9 | ✅ Implemented via `get_gaussian_latitudes(L)` + grid arch simplification | Coriolis now uses exact GL quadrature points |
| ~~🟡 0~~ | ~~9.0 — Grid Architecture Simplification~~ | 9 | ✅ All resampling removed; `TransformConfig` uses derived `n_lat`/`n_lon` | No more per-timestep FFT interpolation artifacts |
| 🟡 6 | 9.2 — `rlnp[0]` Sentinel Audit | 9 | ⬜ Not started | Silent correctness risk in PGF/omega |
| 🟢 7 | 10.1 — Dry Mass Fixer | 10 | ⬜ Not started | Needed for multi-step conservation |
| 🟢 8 | 11.1–11.3 — Validation | 11 | ⬜ Partially done (vector round-trip ✅) | Confirm everything works end-to-end |