# JAX Dynamical Core — Blow-up Debugging Log

**Problem**: The JAX implementation of the GFS dynamical core (`baroclinic_wave_jax.py`)
produces growing surface-pressure variations (~985–1004 hPa by step 30, still growing),
while the Fortran reference stays tightly bounded (~999–1001 hPa throughout 100 steps).
The JAX run does not crash outright but diverges steadily rather than settling.

**Reference simulation (Fortran, 100 steps × 5 min):**
```
Step   0:  999.6 – 1000.4 hPa
Step  10:  999.2 – 1000.9 hPa
Step  20:  999.5 – 1000.6 hPa
Step  30:  999.6 – 1000.5 hPa   ← stable
Step 100:  999.6 – 1000.4 hPa
```

**JAX simulation (30 steps × 5 min, current state):**
```
Step   0: 1000.0 – 1000.0 hPa
Step  10:  998.6 – 1001.0 hPa
Step  20:  994.2 – 1002.9 hPa
Step  30:  985.0 – 1004.1 hPa   ← diverging
```

---

## Confirmed Fixes (bugs that were real and are now corrected)

### 1. Vertical Advection Sign Error — FIXED ✅

**File**: `gfs_dynamical_core/jax/dynamics.py` — `compute_vertical_advection()`

**Bug**: The boundary stencils for vertical advection used the wrong sign. The
Fortran code stores levels bottom-to-top (index 0 = surface) and uses the
matching `etadot` sign convention. The original JAX port had the top and bottom
boundary terms negated relative to what Fortran computes.

**Original (wrong):**
```python
vadv_bot = -(0.5/dp[0])  * etadot[1]  * (data[0] - data[1])
vadv_top =  (0.5/dp[-1]) * etadot[-2] * (data[-2] - data[-1])
```

**Fixed:**
```python
vadv_bot = (0.5/dp[0])  * etadot[1]  * (data[0] - data[1])
vadv_top = (0.5/dp[-1]) * etadot[-2] * (data[-2] - data[-1])
```

**Verification**: `examples/test_vadv.py` confirms JAX matches Fortran exactly
(max|error| = 0.0 at all layers) after the fix.

**Impact**: Was producing wrong vertical transport at the surface and TOA layers.
Fixing it did not stop the divergence.

---

### 2. Dual-Spin Forward Vector Transform — FIXED ✅

**File**: `gfs_dynamical_core/jax/transforms.py` — `grid_to_spectral()` and
`grid_to_spectral_tendencies()`

**Bug**: The original forward (grid→spectral) vector transform used only a
single `spin=+1` transform to decompose `(u, v)` into `(vorticity, divergence)`
spectral coefficients. For `m = 0` modes this is fine, but for `m ≠ 0` modes
the divergence and vorticity spectral coefficients are complex-valued, and a
single spin transform cannot separate their real and imaginary parts.

**Fix**: Use *both* `spin=+1` and `spin=-1` forward transforms:
```python
f_plus  = -v + 1j*u   # spin +1 input
f_minus =  v + 1j*u   # spin -1 input

F1_lm  = s2fft.forward_jax(f_plus,  L, spin= 1, sampling=sampling)
Fm1_lm = s2fft.forward_jax(f_minus, L, spin=-1, sampling=sampling)

result_p = l_factor[:,None] * F1_lm  / radius   # D + i*zeta
result_m = l_factor[:,None] * Fm1_lm / radius   # D - i*zeta

flm_div  = (result_p + result_m) / 2
flm_vort = (result_p - result_m) / (2j)
```

The same dual-spin fix was applied to `grid_to_spectral_tendencies()`.

**Verification**: `examples/test_vector_roundtrip.py` TEST 1 confirms that with
conjugate-symmetric spectral inputs (the physically correct case for real fields),
the round-trip `spectral → grid → spectral` is now exact to machine precision
for all `(l, m)` modes.

**Impact**: This was a real bug affecting all `m ≠ 0` modes. Fixing it did not
stop the divergence.

---

### 3. JIT Tracer Error in Debug Dumps — FIXED ✅

**File**: `gfs_dynamical_core/jax/stepper.py` — `dump_jax_intermediate()`

**Bug**: `dump_jax_intermediate()` attempted to convert JAX traced arrays to
NumPy inside JIT-compiled code, raising a `ConcretizationTypeError`.

**Fix**: Added a guard to skip the dump when running under JIT:
```python
if isinstance(grid_state.u, jax.core.Tracer):
    return
```

**Impact**: Was causing 5 `TestMultiStepStability` tests to error. After the fix
all 5 pass. No effect on numerical stability.

---

### 4. Spectral State Caching (avoid repeated grid↔spectral round-trips) — ADDED ✅

**File**: `gfs_dynamical_core/component_jax.py`

**Bug**: Every call to `array_call()` was re-ingesting the grid-space outputs from
the previous step by running `grid_to_spectral()`. This meant the spectral
representation was only ever derived from the previous step's grid output, which
had itself been produced by `spectral_to_grid()`. The accumulated truncation error
from this repeated round-trip degrades accuracy.

The Fortran dycore never does this: it initialises from the grid state once, then
advances entirely in spectral space, only projecting to grid for diagnostic output.

**Fix**: Added `self._spec_state` to cache the spectral state between calls.
`grid_to_spectral` is now called only on the first invocation (to ingest initial
conditions). Every subsequent call advances from the cached spectral state:
```python
if self._spec_state is None:
    spec_orig = grid_to_spectral(grid_orig, self.trans_config)
else:
    spec_orig = self._spec_state
...
self._spec_state = spec_final
```

**Impact**: Eliminates the repeated round-trip truncation error that was
accumulating each timestep. **However, this alone did not stop the divergence.**
The PS range still grows (~985–1004 hPa by step 36) after this fix, confirming
that the root cause lies elsewhere.

---

## Hypotheses Investigated and Ruled Out

### H1. Inverse vector transform (spectral→grid) uses only spin+1 — RULED OUT ❌

**Hypothesis**: The inverse transform in `spectral_to_grid()` uses only `spin=+1`,
while the forward transform was fixed to use both `spin=+1` and `spin=-1`. The
asymmetry was thought to be the primary source of error.

**Investigation**: `examples/test_vector_roundtrip.py` showed apparent ~50% errors
for all `m ≠ 0` modes. This was initially interpreted as evidence that the inverse
transform was wrong.

**Resolution** (`examples/test_conjugate_hypothesis.py`): The ~50% errors were an
artefact of the *test inputs*, not the transform. The test was setting a single mode
`(l, m≠0)` without its conjugate partner `(l, -m)`. A physically real vector field
requires conjugate symmetry: `f_{l,-m} = (-1)^m conj(f_{l,m})`. When inputs satisfy
this condition:
- `examples/test_conjugate_hypothesis.py` Part B: **16/16 single-mode tests PASS**
- Part C multi-mode test: **PASS** for both the current single-spin and the proposed
  dual-spin inverse

The single-spin `spin=+1` inverse transform is **mathematically correct for all real
vector fields**. No fix to the inverse transform is needed.

**Key insight**: `₋₁f = conj(₁f)` for real fields, so the spin-1 and spin-(−1)
inverses are redundant. Averaging them gives nothing new; the single-spin code is
already exact.

---

### H2. Conjugate symmetry breaking during time-stepping — RULED OUT ❌

**Hypothesis**: Some operation in the IMEX RK scheme (semi-implicit solve,
diffusion, linear tendency computation) breaks the conjugate symmetry
`f_{l,-m} = (-1)^m conj(f_{l,m})` of the spectral coefficients. Once broken,
the inverse transform would produce complex-valued grid fields, and taking `.real`
would silently discard half the information, corrupting the dynamics.

**Investigation**: `examples/check_conj_symmetry_in_model.py` ran 30 timesteps of
the full baroclinic wave, checking conjugate symmetry of all spectral fields
(vorticity, divergence, temperature, log_surface_pressure, tracers) at every step.

**Results**:
```
Step   0:  max conj_sym_err = 8.6e-14  [machine precision — OK]
Step  10:  max conj_sym_err = 5.3e-13  [still machine precision — OK]
Step  20:  max conj_sym_err = 2.7e-12  [still machine precision — OK]
Step  29:  max conj_sym_err = 6.9e-12  [still machine precision — OK]
```
Grid imaginary parts: `|Im(u)| = |Im(v)| = 0.0` at every step.

Conjugate symmetry is maintained to machine precision throughout. The inverse
transform is never fed non-symmetric data, so the `.real` extraction never
discards meaningful information.

---

### H3. Non-conjugate test inputs cause false "transform failure" — RULED OUT AS A BUG ❌

**Hypothesis** (related to H1): The original `test_vector_roundtrip.py` TEST 1 was
believed to demonstrate a fundamental transform failure affecting the simulation.

**Resolution**: See H1. The 50% errors only appear for inputs lacking conjugate
symmetry, which never occurs in the actual simulation. `test_vector_roundtrip.py`
needs to be interpreted with this understanding; its TEST 1 failures are a test
design issue, not a simulation bug.

---

### H4. Repeated grid→spectral→grid round-trips cause divergence — PARTIALLY ADDRESSED, NOT ROOT CAUSE ❌

**Hypothesis**: The `component_jax.py` was re-running `grid_to_spectral()` every
step (see Fix #4). This was suspected to be the main driver of the growing PS range.

**Resolution**: Fix #4 was applied (spectral state caching). The PS range *still*
grows after this fix:
```
Step   0: 1000.0 – 1000.0 hPa
Step  12:  997.8 – 1001.5 hPa
Step  24:  989.6 – 1003.5 hPa
Step  36:  974.4 – 1004.9 hPa
```
The caching reduces redundant work and is still correct to keep, but the divergence
is driven by a different numerical error in the dynamics.

### H5. Coriolis term has wrong sign, magnitude, or latitude convention — RULED OUT ❌

**Hypothesis**: The Coriolis parameter `f = 2Ω sin(φ)` in `assemble_grid_tendencies()`
might use the wrong latitude convention (geographic vs colatitude), wrong sign, or be
applied incorrectly in the absolute-vorticity flux vectors. Because Coriolis directly
couples `u` and `v`, an error here would preferentially corrupt `v` while leaving `T`
relatively unaffected — matching the observed one-step error pattern.

**Investigation**: Full line-by-line trace of the Fortran flux assembly in
`dyn_run.f90` (lines 316–364) against the JAX `assemble_grid_tendencies()`.

**Latitude convention — confirmed identical:**
- Fortran (`shtns.f90` L178–182): `lats1 = cos(colatitude)` via `shtns_cos_array`,
  then `lats = asin(lats1)` = geographic latitude φ in radians.
- JAX (`transforms.py` → `get_gaussian_latitudes`): `latitudes = π/2 − arccos(cos_θ)`
  from `numpy.polynomial.legendre.leggauss` = geographic latitude φ in radians.
  Both evaluate `sin(φ)`, giving the same `f` at every grid point.

**Flux construction — confirmed identical:**

Fortran (with variable lifetimes resolved — `dlnpsdx` is repurposed as scratch for `f`):
```fortran
dlnpsdx  = 2.*omega*sin(lats)
prsgx(:,:,k) = u*(vorticity + f) + (vadv_v - pgf_y)   ! u_flux
prsgy(:,:,k) = v*(vorticity + f) - (vadv_u - pgf_x)   ! v_flux
```
JAX:
```python
f        = 2.0 * config.omega * jnp.sin(latitudes)
abs_vort = vort + f[None, :, None]
u_flux   = u * abs_vort + (vadv_v - pgf_y)
v_flux   = v * abs_vort - (vadv_u - pgf_x)
```
Sign, formula, and broadcasting all match exactly.

**Tendency assignment — confirmed identical:**

Fortran calls `getvrtdivspec(prsgx, prsgy, ddivspecdt, dvrtspecdt)` where, by SHTNS
convention, the 3rd argument receives the **curl** and the 4th receives the
**divergence** of the input vector. The 4th is then negated:
```
ddivspecdt = curl(u_flux, v_flux)    → divergence tendency
dvrtspecdt = −div(u_flux, v_flux)   → vorticity tendency
```
JAX `grid_to_spectral_tendencies()`:
```python
d_div  = curl_of_flux
d_vort = −div_of_flux
```
Identical.

**Resolution**: The Coriolis term is definitively correct. It is not the source
of the divergence.

**Note on the reported "3300% v-error"**: The DCMIP initial `v` (northward wind)
is nearly zero — it is a near-zonal flow. The large *relative* error in `v` after
one step is dominated by a near-zero denominator, not a large absolute error. The
absolute error in `v` after a single grid→spectral→grid round-trip is only
~1.5×10⁻³ m/s (measured in CHECK 0c of `check_conj_symmetry_in_model.py`). The
33× ratio is therefore misleading as a signal of a specific component failure.

---

## What is Still Unknown (Active Search Space)

The divergence is confirmed to be a numerical accuracy issue in the dynamics
itself, not in the transform infrastructure or the Coriolis term. The two
highest-priority remaining suspects are:

### Suspect 1 (HIGH): `compute_pressure_gradient_force` vs Fortran `getpresgrad` — PARTIALLY FIXED (Major driver, but not the final fix) ⚠️

**Status**: Fixed. This was *a* major driver of the blow-up, but the simulation still blows up eventually. The short-term trajectory is now dramatically stabilized.

**Findings**: The `compute_pressure_gradient_force` function suffered from four distinct errors due to confusion between Fortran's split top-to-bottom/bottom-to-top indexing and JAX's strict bottom-to-top indexing:
1. **Swapped Interfaces in `cofa`**: The `term1` and `cofa_coef` computations swapped the top and bottom interfaces of each layer.
2. **Reversed Geopotential Integration**: The geopotential integral `px2_factor` was accumulating from TOA downwards, instead of from the surface upwards.
3. **Wrong Sign**: The difference `bk_ratio_diff` used `top - bot` instead of `bot - top`.
4. **Reversed Temperature Gradient Integration**: The `px3u` and `px3v` terms were accumulating from TOA downwards instead of surface upwards.

**Fix**: Corrected all indexing and directions of integration (`jnp.cumsum`), swapping top/bottom bounds in coefficient formulas, and ensuring integrations correctly start at the surface and move upwards.

**Verification**: `examples/compare_one_step.py` demonstrates the JAX surface pressure trajectory is now strictly stable over the first 10 steps, bounding between ~999.47 and 1000.53 hPa, mirroring Fortran's stability in the short term. However, long-term simulations (e.g., 24 hours) still eventually go unstable, indicating another bug remains.

### Suspect 2 (MEDIUM): `compute_vertical_velocities` vs Fortran `getomega`

`compute_vertical_velocities()` computes `etadot` and `omega` (the vertical
velocity diagnostics that feed into vertical advection and the energy conversion
term). The Fortran `getomega` uses top-to-bottom indexing for `etadot` while the
JAX code uses bottom-to-top. While a previous fix corrected the *advection*
stencil signs, the `etadot` *magnitude* and intermediate `dlnpdtg` values have
not been cross-validated against Fortran numerically.

### Suspect 3 (LOWER): Semi-implicit matrix ordering in `init_semi_implicit_matrices`

The `d_hyb_m` inversion and the `amhyb`/`bmhyb` matrices involve level-indexed
linear algebra that maps Fortran column-major top-to-bottom arrays to JAX
row-major bottom-to-top arrays. A transposition or index-reversal error in these
matrices would affect the divergence and temperature tendencies in the implicit
solve, causing the simulation to diverge. This has been inspected but not
numerically validated at the matrix-element level.

---

## Important Reference: Fortran vs JAX Array Orientations

A major source of bugs in the JAX port has been the mismatched array orientations.
- **JAX**: All level-dependent arrays are strictly **Bottom-to-Top** (index 0 is surface, index -1 is TOA).
- **Fortran**: Uses a split convention where some arrays are Top-to-Bottom and others are Bottom-to-Top.

When reading the Fortran source (`getpresgrad`, `getomega`, `getvadv`, etc.), remember:
* **Top-to-Bottom (k=1 is TOA)**:
  - Hybrid coefficients: `ak`, `bk`, `ck`, `dbk`
  - Pressure diagnostics: `pk` (interfaces), `dpk`, `alfa`, `rlnp`
  - Vertical velocity: `etadot` (interfaces)
* **Bottom-to-Top (k=1 is Surface)**:
  - Grid state: `ug`, `vg`, `virtempg`, `divg`, `vortg`, tracers
  - Fluxes/Tendencies: `prsgx`, `prsgy`, `vadv` (all outputs of vertical advection)

Any translation of Fortran loops involving both state arrays and pressure diagnostics must carefully map indices.

---

## Diagnostic Tools Created

| Script | Purpose |
|--------|---------|
| `examples/test_vadv.py` | Verify vertical advection matches Fortran exactly |
| `examples/test_vector_roundtrip.py` | 6-part vector transform validation suite |
| `examples/test_conjugate_hypothesis.py` | Confirms conjugate-symmetric inputs round-trip correctly; rules out H1 |
| `examples/check_conj_symmetry_in_model.py` | Runs 30-step simulation checking conjugate symmetry at every step; rules out H2 |
| `examples/compare_one_step.py` | Single-step field-by-field comparison (JAX vs Fortran) |
| `examples/compare_debug_dumps.py` | Load Fortran binary stage dumps and compare to JAX at each RK stage |

---

## Key Fortran vs JAX Architectural Difference

The Fortran dycore (`psgm.f90` / `run.f90`) **never exposes the spectral state
to the outside world**. It initialises from grid-space once, converts to spectral,
then runs entirely internally. The `climt` wrapper only reads grid-space diagnostics.

The original JAX `component_jax.py` design tried to conform to the `sympl` Stepper
interface by accepting and returning grid-space arrays, which forced a
`grid→spectral` conversion every step. Fix #4 eliminates this for the state, but
the *output* fields (u, v, T, PS, q) are still grid-space values derived from
`spectral_to_grid()` applied to the final spectral state. This is correct: the
grid output is for diagnostics only, and the internal spectral state is what
advances the model.

---

## Suspects Resolved by Line-by-Line Verification

### Suspect 1 (PGF): `compute_pressure_gradient_force` — VERIFIED CORRECT ✅

**Status**: After the fix in `b019f40`, all terms in `compute_pressure_gradient_force`
match the Fortran `getpresgrad` (`dyn_run.f90` lines 487–563) exactly:
- `cofb_pressure` (interface pressure ratios)
- `cofa` (layer-mean geopotential coefficient)
- `px2_factor` (geopotential integral, surface→TOA cumulative sum)
- `px3u` / `px3v` (temperature-gradient correction, surface→TOA cumulative sum)
- Final assembly: `pgf_x = px2u * dlnpdx + px3u`, `pgf_y = px2v * dlnpdy + px3v`

**Conclusion**: PGF is no longer a suspect. The short-term trajectory matching
Fortran (~999.47–1000.53 hPa over 10 steps) confirms this.

---

### Suspect 2 (Vertical velocities): `compute_vertical_velocities` — VERIFIED CORRECT ✅

**Status**: All terms in `compute_vertical_velocities` match the Fortran `getomega`
(`dyn_run.f90` lines 419–485) exactly:
- `cg` (sigma-coordinate ratio `dbk / (dpk * 2)`)
- `db` / `cb` cumulative sums (column-integrated mass flux partitioning)
- `d_log_ps_d_t` (surface pressure tendency from divergence)
- `etadot` (vertical mass flux at interfaces, BTU→TTB mapping correct)
- `workb` / `workc` (omega intermediate terms)
- `omega` (pressure vertical velocity)

The BTU↔TTB index mapping was verified throughout: `etadot` is computed bottom-to-top
but used as top-to-bottom input to `compute_vertical_advection`, which is consistent
with the Fortran convention where `etadot` is top-to-bottom (`getvadv` comment:
"etadot top to bottom").

**Conclusion**: Vertical velocities are correct. Not a source of the blow-up.

---

### Suspect 3 (Semi-implicit matrices): `init_semi_implicit_matrices` — VERIFIED CORRECT ✅

**Status**: All matrices in `init_semi_implicit_matrices` match the Fortran
`semimp_data.f90` exactly:
- Reference profile: `pkref` / `dpkref` / `alfaref` (hydrostatic layer means)
- `yecm` / `tecm` operators (linearised divergence–temperature coupling)
- BTU↔TTB flip of `yecm` before matrix algebra
- `ym` = `I + coeff² * dt² * yecm @ tecm` (implicit operator)
- `d_hyb_m[l]` = `ym⁻¹` per zonal wavenumber (inversion for implicit solve)
- `amhyb` / `bmhyb` (back-substitution matrices)
- `tor_hyb` / `svhyb` (linearised tendency operators)
- Butcher tableau coefficients (`aa22`, `aa33`, `bb4`) match exactly

**Conclusion**: Semi-implicit matrices are correct. Not a source of the blow-up.

---

## Root Cause Found: Dry Mass Fixer Misapplication — NEW BUG 🐛

### Summary

The dry mass fixer is being applied **incorrectly in two ways**, causing
the long-term instability that persists after the PGF fix.

### Problem 1: Fixer is active for the adiabatic DCMIP test case (it shouldn't be)

In the Fortran (`run.f90` line 330):
```fortran
if (.not. adiabatic) then
    ! ... all physics + dry mass fixer code ...
    if (massfix) then
        call dry_mass_fixer(psg,pwat,dlnpsspecdt1,dt)   ! line 358
    endif
    lnpsspec = lnpsspec + dt*dlnpsspecdt1               ! line 360
endif
```

The fixer is **inside the `if (.not. adiabatic)` block** — it is never applied
for adiabatic (dry dynamics-only) runs like the DCMIP baroclinic wave test case.

In the JAX code, `component_jax.py` **always** initializes `_gauss_weights` (line 204)
and `_pdryini` (line 213), regardless of whether the run is adiabatic. These are
unconditionally passed to `advance()`, which builds `_dmf_kwargs` (stepper.py line 442)
and applies the fixer at every timestep.

### Problem 2: Fixer is applied inside the IMEX RK scheme (should be after)

In the Fortran, the fixer runs **after** the complete IMEX RK3 integration:
- Lines 349–353: RK scheme updates vrt, div, virtemp, tracer spectral fields
- Lines 357–360: Dry mass fixer modifies `dlnpsspecdt1`, then `lnpsspec += dt * dlnpsspecdt1`

This is a **post-RK correction** — it doesn't interfere with the implicit gravity
wave solver.

In the JAX code, the fixer is applied **at Stage 3 of the RK scheme**
(`stepper.py` line 616–617):
```python
# --- Stage 3 ---
tends2 = get_spectral_tendencies(state2, ..., **_dmf_kwargs)
```

Inside `get_spectral_tendencies` (`dynamics.py` lines 463–484), when the fixer
arguments are present, it **replaces** the physical `lnps` spectral tendency:
```python
if pdryini is not None and gauss_weights is not None and dt is not None:
    corrected_lnps_tend = compute_dry_mass_fixer(...)
    spec_tends = SpectralTendencies(
        ...,
        d_log_surface_pressure_d_t=corrected_lnps_tend,  # REPLACES physical tendency
        ...
    )
```

For the dry DCMIP test case (q ≈ 0), `compute_dry_mass_fixer` computes:
- `pcorr = (pdryini + g * pwat_global) / pmean ≈ pdryini / pmean ≈ 1.0`
- `lnps_target_grid = log(ps * pcorr) ≈ log(ps)`
- The returned tendency = `(lnps_target_spec - lnps_current) / dt` ≈ 0 for all
  non-global-mean spectral modes

Since Stage 3 carries weight `b3 = 2/3` in the RK scheme, this effectively zeroes
out **two-thirds** of the surface pressure tendency at every timestep, preventing
proper gravity wave propagation through the semi-implicit solver.

### Why the short-term trajectory looks OK

The fixer preserves the global mean of surface pressure correctly. The baroclinic
wave perturbation grows slowly from a small initial anomaly, so the first ~10 steps
appear stable. But the systematic suppression of non-zonal lnps tendencies at Stage 3
progressively inhibits the baroclinic development, and the resulting energy imbalance
eventually manifests as numerical instability.

### Evidence Chain

| File | Line(s) | Evidence |
|------|---------|----------|
| `component_jax.py` | 204–226 | `_gauss_weights` and `_pdryini` always initialized (no adiabatic check) |
| `stepper.py` | 442–446 | `_dmf_kwargs` always non-empty when weights/pdryini provided |
| `stepper.py` | 616–617 | `get_spectral_tendencies(state2, ..., **_dmf_kwargs)` applies fixer at Stage 3 |
| `dynamics.py` | 463–484 | Fixer **replaces** lnps tendency (not corrects it) |
| `dynamics.py` | 398–422 | `compute_dry_mass_fixer` returns ≈ 0 for all non-mean modes in dry case |
| `run.f90` | 330 | `if (.not. adiabatic) then` — Fortran skips fixer for adiabatic runs |
| `run.f90` | 357–360 | Fortran applies fixer **after** IMEX RK, not inside it |

### Proposed Fix

**Immediate (fixes the blow-up):** Disable the dry mass fixer for adiabatic runs.
Either add an `adiabatic` flag to `GFSDynamicsJAX` or simply don't pass
`gauss_weights`/`pdryini` to `advance()`.

**Structural (needed for future moist runs):** Move the fixer from inside the RK
scheme (Stage 3) to a post-RK correction step, matching Fortran's `run.f90`
lines 357–360. The fixer should modify the final `lnps` spectral state after the
RK update is complete, not replace a stage tendency.

### Secondary Issue (not the blow-up cause): Virtual Temperature

The Fortran wrapper (`component.py` line 448) converts `T → Tv` (virtual temperature)
before dynamics and `Tv → T` after:
```python
t_virt = state["air_temperature"] * (1 + self._fvirt * state["tracers"][0])
```

The JAX `component_jax.py` passes regular `T` directly. For the dry DCMIP test
(q ≈ 0, so Tv ≈ T) this is harmless, but it will need fixing for moist simulations.

---

## Current State of the Codebase

| Component | Status |
|-----------|--------|
| `jax/dynamics.py` | Vertical advection sign fixed; Coriolis confirmed correct; PGF confirmed correct; vertical velocities confirmed correct; `compute_dry_mass_fixer` **identified as misapplied** |
| `jax/transforms.py` | Forward vector transform fixed (dual-spin); inverse transform correct as-is |
| `jax/stepper.py` | JIT tracer guard added; IMEX RK structure matches Fortran; semi-implicit matrices confirmed correct; **dry mass fixer applied at wrong stage** |
| `component_jax.py` | Spectral state caching added; **dry mass fixer unconditionally enabled (should be off for adiabatic)** |
| `tests/test_jax_dynamics.py` | Updated for corrected vertical advection signs |
| `tests/test_jax_transforms.py` | Passes for conjugate-symmetric inputs (correct) |

---

*Last updated: All three original suspects (PGF, vertical velocities, semi-implicit matrices) verified correct by line-by-line comparison with Fortran. Root cause identified: dry mass fixer is (a) active for adiabatic runs when it shouldn't be, and (b) applied inside the IMEX RK scheme instead of after it.*
*Branch: `develop`.*

---

## Changelog

### 2026-03-22 — JW2006 Diagnostics & Refined Root Cause

#### New Diagnostics

Ran `examples/jw06_diagnostics.py`, implementing JW2006 Eqs 14–16 (l2 zonal
asymmetry, zonal-mean drift, surface pressure difference) for both JAX and
Fortran over 30 simulated days. Config: L=64, ntrunc=40, dt=600s (JAX),
dt=300s (Fortran), DCMIP test 4.1 (dry baroclinic wave, `adiabatic=False`
in both codes).

**Key observations from the plots:**

- `jw06_steady_errors.png`: JAX l2 zonal asymmetry grows **exponentially**
  from ~10^-11 to ~10^1 in 8 days (e-folding time ~7 hours). Fortran grows
  from ~10^-11 to ~10^-3 in 30 days (expected slow growth from the
  perturbation).
- `jw06_ps_min.png`: JAX PS minimum crashes from ~1000 hPa to near 0 between
  days 8–10. Fortran PS minimum evolves smoothly.
- `jw06_wave_snapshots_JAX.png`: Realistic baroclinic wave development at
  days 4, 6, 8; completely destroyed by day 9.
- `jw06_wave_snapshots_Fortran.png`: Stable realistic development through
  day 10.
- `jw06_vorticity.png`: Day 7 JAX vorticity matches Fortran well; day 9
  JAX vorticity field is blank/noise.

The exponential growth with constant e-folding time is the signature of an
**undamped linear instability**, not a nonlinear blow-up. This pointed the
investigation toward modes that receive no diffusion.

#### Refined Root Cause: Missing Triangular Truncation in Dry Mass Fixer

The previously identified dry mass fixer issues (Problems 1 and 2 above —
wrong adiabatic gating and wrong RK placement) are real but secondary. The
**primary mechanism** driving the day 7–8 blow-up is:

**File**: `gfs_dynamical_core/jax/dynamics.py` — `compute_dry_mass_fixer()`
(line ~421)

**Bug**: The fixer calls `s2fft.forward_jax(log(ps * pcorr), L, sampling="gl")`
which produces spectral coefficients for ALL modes l=0..L-1=63. The Fortran
equivalent `grdtospec()` uses packed triangular storage (`ndimspec` elements)
that inherently limits output to l=0..ntrunc=40. Modes l=41..63 simply cannot
exist in Fortran.

In JAX, these extra modes (l=41..63) in `lnps_target_spec` are injected into
the spectral state every timestep because the fixer tendency **replaces**
`lnps`:

```
new_lnps = old_lnps + dt * fixer_tend = lnps_target_spec
```

**Why this causes exponential blow-up:**

1. Each timestep, the `exp/log` round-trip in the fixer
   (`log(exp(lnps) * pcorr)`) introduces ~10^-15 noise per step at modes
   l=41..63.
2. These modes get **ZERO diffusion**: `disspec = 0` for l > ntrunc = 40
   (by design — `stepper.py` line 357–358 sets `disspec_2d = 0` outside the
   triangular truncation mask).
3. No truncation is applied to the spectral **state** — only to spectral
   **tendencies** (inside `grid_to_spectral_tendencies`, which calls
   `enforce_triangular_truncation`). The fixer bypasses this path.
4. The l > 40 lnps modes create small-scale grid features in surface pressure.
5. Grid-space dynamics respond, transferring energy to l <= 40 through
   nonlinear coupling (aliasing).
6. Exponential growth with ~7-hour e-folding time, catastrophic by day 7–8.

**Evidence chain:**

| Fact | Location |
|------|----------|
| `s2fft.forward_jax` produces L×(2L-1) = 64×127 spectral array | s2fft docs |
| Fortran `grdtospec` produces `ndimspec = (ntrunc+1)*(ntrunc+2)/2 = 861` packed coefficients | `shtns.f90` |
| `disspec = 0` for l > ntrunc | `stepper.py:357-358` |
| `enforce_triangular_truncation` applied to tendencies but NOT to state | `transforms.py:376-380` |
| Fixer replaces lnps tendency (not additive correction) | `dynamics.py:463-484` |
| Exponential growth with constant e-folding = undamped linear mode | `jw06_steady_errors.png` |

**Proposed fix**: Apply `enforce_triangular_truncation` to `lnps_target_spec`
in `compute_dry_mass_fixer` before computing the tendency:

```python
lnps_target_spec = s2fft.forward_jax(lnps_target_grid, L, sampling="gl")
lnps_target_spec = enforce_triangular_truncation(lnps_target_spec, ntrunc)
return (lnps_target_spec - log_surface_pressure) / dt
```

This matches the Fortran behavior where only l <= ntrunc modes can exist.

#### Additional Bugs Found (not root cause of blow-up, but incorrect)

**Bug A: Tracer vertical advection sign error**

**File**: `dynamics.py` — `compute_vertical_advection_tracers()` (lines 244–246)

The three flux terms are sign-flipped relative to Fortran `getvadv_tracers`.
Fortran computes `F_bottom - F_top`; JAX computes `F_top - F_bottom`.

Current (wrong):
```python
vadv = (1.0 / dp) * (
    (datag_half_full[1:] * etadot[1:] - datag_half_full[:-1] * etadot[:-1])
    + data * (etadot[:-1] - etadot[1:])
)
```

Fix — swap `[:-1]` and `[1:]`:
```python
vadv = (1.0 / dp) * (
    (datag_half_full[:-1] * etadot[:-1] - datag_half_full[1:] * etadot[1:])
    + data * (etadot[1:] - etadot[:-1])
)
```

Impact: Harmless for the dry DCMIP test (q ~ 0), but will matter for moist runs.

**Bug B: Boundary `datag_d` sign mismatch in tracer limiter**

**File**: `dynamics.py` — `compute_vertical_advection_tracers()` (lines 200–213)

The boundary formulas for `datag_d_bot` and `datag_d_top` use Fortran's sign
convention (below-minus-above), while the interior `datag_d = data[1:] - data[:-1]`
uses above-minus-below. The phi ratio at boundaries becomes negative, causing
the limiter to clamp to 0 (first-order upwind at boundaries only).

Fix: Negate boundary formulas to match interior convention.

Impact: Minor — only affects first and last levels of tracer advection limiter.

**Bug C: Minor TOA `workb` discrepancy in `getomega`**

Fortran k=1 `workb` has an extra factor of `dbk(1)` compared to JAX.
Impact: Negligible since `dbk` at TOA ~ 0.

#### Verified Correct (no discrepancies found)

These components were verified by line-by-line comparison with Fortran during
this session and confirmed correct:

- `compute_pressure_diagnostics` vs `calc_pressdata`
- `compute_pressure_gradient_force` vs `getpresgrad`
- `compute_vertical_advection` (non-tracer) vs `getvadv`
- `compute_vertical_velocities` vs `getomega` (minor TOA note above)
- `compute_energy_conversion` vs Fortran energy conversion term
- `assemble_grid_tendencies` vs `getdyntend` (Coriolis, flux assembly)
- IMEX RK3 Butcher tableau coefficients
- `init_semi_implicit_matrices` vs `init_semimpdata`
- `init_diffusion_operators` vs `setdampspec`
- Spectral transform conventions (s2fft spin-weighted vs SHTNS sphtor)
- KE Laplacian in divergence tendency

#### Next Steps

1. **Fix the truncation bug** in `compute_dry_mass_fixer` (apply
   `enforce_triangular_truncation` to the forward-transformed lnps).
2. **Fix tracer vertical advection sign** (Bug A above).
3. **Fix boundary `datag_d` signs** (Bug B above).
4. Re-run `jw06_diagnostics.py` to confirm the blow-up is eliminated.