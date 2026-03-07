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

### Suspect 1 (HIGH): `compute_pressure_gradient_force` vs Fortran `getpresgrad`

`compute_pressure_gradient_force()` in `dynamics.py` is the most complex function
in the JAX port and has **never been directly validated** against the Fortran
`getpresgrad` subroutine. It computes `pgf_x` and `pgf_y` — the horizontal
pressure gradient force that enters both momentum flux vectors. A sign or
coefficient error here would inject a systematic per-step error that grows over
time, consistent with the observed behaviour.

The function computes a hydrostatic integral involving `alfa`, `rlnp`, `bk`, `pk`,
`dpk` and temperature gradients. The level-indexing conventions (bottom-to-top in
JAX vs top-to-bottom in Fortran) must be carefully accounted for in every cumsum
and einsum. This is the most likely place for a subtle indexing or sign error.

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

## Current State of the Codebase

| Component | Status |
|-----------|--------|
| `jax/dynamics.py` | Vertical advection sign fixed; Coriolis confirmed correct (matches Fortran exactly); `compute_pressure_gradient_force` and `compute_vertical_velocities` **not yet numerically validated vs Fortran** |
| `jax/transforms.py` | Forward vector transform fixed (dual-spin); inverse transform correct as-is |
| `jax/stepper.py` | JIT tracer guard added; IMEX RK structure matches Fortran by inspection; semi-implicit matrices **not yet validated at element level** |
| `component_jax.py` | Spectral state caching added; no more repeated round-trips |
| `tests/test_jax_dynamics.py` | Updated for corrected vertical advection signs |
| `tests/test_jax_transforms.py` | Passes for conjugate-symmetric inputs (correct) |

---

*Last updated: Coriolis term verified correct and ruled out as blow-up cause (H5). Active suspects: `compute_pressure_gradient_force` (HIGH) and `compute_vertical_velocities` (MEDIUM).*
*Branch: `develop`.*