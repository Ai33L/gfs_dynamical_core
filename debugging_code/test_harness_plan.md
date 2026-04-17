# JAX Dynamical Core: Test Harness Design & Rewrite Strategy

## Problem Statement

The JAX dynamical core blows up at day 7-8 of a baroclinic wave simulation. Extensive debugging has found and fixed ~8 bugs, yet the model still blows up. A persistent 2.2% error in the meridional wind (v) exists from step 1, is 100% in the m=0 zonal-mean component, varies by level (ratio 0.92-1.01), and has never been traced to a specific line of code. The explicit=True mode also blows up, proving the bug is in the shared tendency path, not the IMEX solver.

Multiple AI models have declared the code "identical to Fortran" and been wrong every time. Code reading has reached its limit -- we need machine-precision numerical verification of every function.

## Strategy: Bottom-Up Verified Rebuild

**Core principle: No function is trusted until it passes a numerical test against Fortran output at machine precision (relative error < 1e-12 for 64-bit arithmetic).**

The approach has three layers:

1. **Layer 0 -- Transforms**: Verify s2fft against shtns Python bindings directly (no Fortran binary dumps needed)
2. **Layer 1 -- Grid-space dynamics**: Instrument Fortran to dump per-function intermediates, feed identical inputs to JAX, compare outputs
3. **Layer 2 -- Full pipeline**: End-to-end spectral tendency comparison, then multi-step integration

Within each layer, functions are tested in dependency order. A failing test blocks all downstream work until fixed.

---

## Part 1: Test Infrastructure

### 1.1 Fortran Intermediate Dump Extension

**Goal**: Extend `getdyntend()` in `dyn_run.f90` to dump every intermediate quantity after each sub-computation.

**New dump subroutine**: `dump_dyntend_intermediates(stage)`

**File**: `dyn_run.f90`, called at the end of `getdyntend()` (before return)

**Contents** (all written as raw binary, Fortran column-major order):

```
Header: nlons, nlats, nlevs, ndimspec (4 integers)

Block 1 -- Grid state (from spectral->grid at start of getdyntend):
  ug(nlons, nlats, nlevs)           -- zonal wind
  vg(nlons, nlats, nlevs)           -- meridional wind
  virtempg(nlons, nlats, nlevs)     -- virtual temperature
  divg(nlons, nlats, nlevs)         -- divergence
  vrtg(nlons, nlats, nlevs)         -- vorticity
  lnpsg(nlons, nlats)               -- log surface pressure
  tracerg(nlons, nlats, nlevs, 1)   -- specific humidity

Block 2 -- Gradients (from getgrad calls):
  dvirtempdx(nlons, nlats, nlevs)   -- dT/dlambda
  dvirtempdy(nlons, nlats, nlevs)   -- dT/dphi
  dlnpsdx(nlons, nlats)             -- d(ln ps)/dlambda
  dlnpsdy(nlons, nlats)             -- d(ln ps)/dphi
  dtracerdx(nlons, nlats, nlevs)    -- dq/dlambda  (if ntrac > 0)
  dtracerdy(nlons, nlats, nlevs)    -- dq/dphi  (if ntrac > 0)

Block 3 -- Pressure diagnostics (from calc_pressdata):
  pk(nlons, nlats, nlevs+1)         -- interface pressures
  dpk(nlons, nlats, nlevs)          -- layer thickness
  prs(nlons, nlats, nlevs)          -- mean layer pressure
  alfa(nlons, nlats, nlevs)         -- log-pressure coefficient
  rlnp(nlons, nlats, nlevs)         -- log(pk(k+1)/pk(k))
  psg(nlons, nlats)                 -- surface pressure

Block 4 -- Vertical velocities (from getomega):
  etadot(nlons, nlats, nlevs+1)     -- eta-dot on interfaces
  dlnpdtg(nlons, nlats, nlevs)      -- omega (d ln p / dt)
  dlnpsdt(nlons, nlats)             -- surface pressure tendency

Block 5 -- Pressure gradient force (from getpresgrad):
  prsgx(nlons, nlats, nlevs)        -- PGF x-component
  prsgy(nlons, nlats, nlevs)        -- PGF y-component

Block 6 -- Vertical advection (from getvadv):
  vadvu(nlons, nlats, nlevs)        -- vertical advection of u
  vadvv(nlons, nlats, nlevs)        -- vertical advection of v
  vadvt(nlons, nlats, nlevs)        -- vertical advection of T

Block 7 -- Energy conversion:
  energy_conv(nlons, nlats, nlevs)  -- kappa * omega * Tv / denom

Block 8 -- Assembled grid tendencies (before spectral transform):
  u_flux(nlons, nlats, nlevs)       -- u * (zeta+f) + (vadv_v - pgf_y)
  v_flux(nlons, nlats, nlevs)       -- v * (zeta+f) - (vadv_u - pgf_x)
  temp_tend(nlons, nlats, nlevs)    -- temperature tendency
  ke(nlons, nlats, nlevs)           -- kinetic energy = 0.5*(u^2+v^2)

Block 9 -- Spectral tendencies (output of getdyntend):
  dvrtspecdt(ndimspec, nlevs)        -- vorticity tendency (complex)
  ddivspecdt(ndimspec, nlevs)        -- divergence tendency (complex)
  dvirtempspecdt(ndimspec, nlevs)    -- temperature tendency (complex)
  dlnpsspecdt(ndimspec)              -- lnps tendency (complex)
  dtracerspecdt(ndimspec, nlevs, 1)  -- tracer tendency (complex)
```

**Filename convention**: `debug_data/fortran_dyntend_step_{step}_stage_{stage}.bin`

**Trigger**: First 5 timesteps only (to keep dump size manageable).

### 1.2 Python Loader Library

**File**: `tests/fortran_loader.py`

A single module that:
1. Reads the binary dump file
2. Reshapes from Fortran (nlons, nlats, nlevs) column-major to JAX (nlevs, nlat, nlon) row-major
3. Handles the latitude flip (Fortran south->north vs s2fft north->south)
4. Handles the level flip (Fortran TTB vs JAX BTU) where applicable
5. Returns named dictionaries matching JAX function signatures

```python
def load_dyntend_dump(path: str) -> dict:
    """Load Fortran getdyntend intermediates and convert to JAX conventions.

    Returns dict with keys:
        'grid_state': dict with u, v, temperature, vorticity, divergence,
                      log_surface_pressure, tracers
        'gradients': dict with d_t_d_lambda, d_t_d_phi, d_log_ps_d_lambda, etc.
        'pressure': dict with ps, pk, dp, prs, alfa, rlnp
        'vertical_velocities': dict with etadot, omega, d_log_ps_d_t
        'pgf': dict with pgf_x, pgf_y
        'vertical_advection': dict with vadv_u, vadv_v, vadv_t
        'energy_conv': ndarray
        'grid_tendencies': dict with u_flux, v_flux, temp_tend, ke
        'spectral_tendencies': dict with d_vort, d_div, d_temp, d_lnps, d_tracers

    All arrays are in JAX convention:
        - 3D fields: (n_lev, n_lat, n_lon), k=0 is surface, k=-1 is TOA
        - 2D fields: (n_lat, n_lon)
        - Interface fields: (n_lev+1, n_lat, n_lon), k=0 is surface
        - Latitude: north-to-south (s2fft GL order)
    """
```

**Critical conversion details documented in the loader**:
- Fortran grid state arrays (ug, vg, etc.) are BTU (k=1 surface) -- just reverse level axis
- Fortran pressure arrays (pk, dpk, alfa, rlnp) are TTB (k=1 TOA) -- reverse level axis
- Fortran etadot is TTB -- reverse level axis
- Fortran dlnpdtg is BTU -- just reverse level axis
- Latitude axis: Fortran south->north, s2fft north->south -- reverse lat axis
- Spectral: Fortran uses packed triangular (ndimspec), JAX uses rectangular (L, 2L-1) -- need unpacking function

### 1.3 Spectral Layout Converter

The Fortran spectral data uses SHTNS packed triangular storage: a 1D array of length `ndimspec = (ntrunc+1)*(ntrunc+2)/2`, indexed by `(l, m)` pairs with `0 <= m <= l <= ntrunc`.

The JAX/s2fft spectral data uses a 2D rectangular layout: `(L, 2L-1)` where L = band-limit, with m ranging from `-(L-1)` to `+(L-1)`.

We need bidirectional converters:
```python
def shtns_packed_to_s2fft_rect(packed, ntrunc, L):
    """Convert SHTNS packed triangular -> s2fft rectangular (L, 2L-1).

    SHTNS stores only m >= 0. For real fields, the m < 0 modes
    satisfy f_{l,-m} = (-1)^m * conj(f_{l,m}).

    SHTNS uses NO Condon-Shortley phase. s2fft includes CS phase.
    The converter must apply (-1)^m correction for the phase convention.
    """

def s2fft_rect_to_shtns_packed(rect, L, ntrunc):
    """Inverse of above."""
```

**Phase convention**: SHTNS is initialized with `SHT_NO_CS_PHASE`. s2fft includes CS phase by default. For real scalar fields, this difference cancels in the forward+inverse roundtrip, but it matters when comparing raw spectral coefficients between libraries. The converter must account for this.

### 1.4 SHTNS Python Reference Wrapper

**File**: `tests/shtns_reference.py`

Wraps the SHTNS Python bindings to match the JAX function signatures:

```python
class SHTNSReference:
    """Reference implementation using SHTNS Python bindings.

    Configured to match the Fortran dycore exactly:
    - SHT_NO_CS_PHASE flag
    - Gauss-Legendre grid
    - Same truncation
    """

    def __init__(self, L, ntrunc):
        self.sh = shtns.sht(ntrunc, ntrunc)
        self.sh.set_grid(L, 2*L-1,
                         flags=shtns.SHT_PHI_CONTIGUOUS | shtns.SHT_NO_CS_PHASE)

    def scalar_forward(self, grid_field):
        """grid (n_lat, n_lon) -> spectral packed (ndimspec,)"""

    def scalar_inverse(self, spec_field):
        """spectral packed (ndimspec,) -> grid (n_lat, n_lon)"""

    def vector_forward(self, u, v):
        """(u, v) grid -> (vorticity, divergence) spectral"""

    def vector_inverse(self, vort_spec, div_spec):
        """(vorticity, divergence) spectral -> (u, v) grid"""

    def gradient(self, scalar_spec):
        """scalar spectral -> (d/dlambda, d/dphi) grid"""
```

This wrapper handles all latitude/longitude ordering to match s2fft's conventions, so tests can compare outputs directly.

---

## Part 2: Testing Phases

### Phase 0: Transform Verification (SHTNS <-> s2fft)

**No Fortran binary dumps needed.** Uses SHTNS Python bindings directly.

**Acceptance criterion**: max|s2fft - shtns| / max|shtns| < 1e-12

| Test ID | Description | Input | Compare |
|---------|-------------|-------|---------|
| T0.1 | Scalar forward | Analytic field (JW06 T profile) | s2fft.forward vs shtns.analys |
| T0.2 | Scalar inverse | Random spectral coefficients | s2fft.inverse vs shtns.synth |
| T0.3 | Scalar roundtrip | JW06 temperature | forward->inverse error |
| T0.4 | Vector inverse (vort,div -> u,v) | JW06 spectral vort/div | s2fft spin-1 vs shtns.SHsphtor_to_spat |
| T0.5 | Vector forward (u,v -> vort,div) | JW06 u,v grid | s2fft dual-spin vs shtns.spat_to_SHsphtor |
| T0.6 | Gradient | JW06 temperature spectral | s2fft spin-1 gradient vs shtns.synth_grad |
| T0.7 | Vector roundtrip | JW06 u,v | forward->inverse error (both libraries) |
| T0.8 | m=0 isolated | Zonal-only fields (u=cos(lat), v=0) | Verify m=0 handling matches |

**Critical**: Test T0.8 directly targets the known 2.2% v error. If s2fft and shtns disagree on the m=0 component of v after a vector roundtrip, **the transform library is the root cause** and we switch to shtns Python bindings.

**If any test fails**: Characterize the discrepancy (which modes, what magnitude). If the error is systematic and cannot be resolved by convention corrections, **switch to SHTNS Python bindings as the transform backend** for the JAX dycore. s2fft can be revisited later once correctness is established.

### Phase 1: Per-Function Dynamics Verification

**Requires**: Fortran intermediate dumps (Section 1.1) and passing Phase 0 tests.

**Method**: For each function, load Fortran's inputs from the dump, call the JAX function, compare against Fortran's output from the dump.

**Acceptance criterion**: max|JAX - Fortran| / max|Fortran| < 1e-12

**Testing order** (matches dependency chain -- each function depends only on already-verified functions):

| Test ID | Function | Fortran Inputs (from dump) | Fortran Output (from dump) | JAX Function |
|---------|----------|---------------------------|---------------------------|--------------|
| D1.1 | Pressure diagnostics | log_surface_pressure | pk, dp, prs, alfa, rlnp | `compute_pressure_diagnostics` |
| D1.2 | Vertical velocities | grid_state + gradients + press_diag | etadot, omega, d_log_ps_d_t | `compute_vertical_velocities` |
| D1.3 | Vertical advection (u) | u + etadot + dp | vadv_u | `compute_vertical_advection` |
| D1.4 | Vertical advection (v) | v + etadot + dp | vadv_v | `compute_vertical_advection` |
| D1.5 | Vertical advection (T) | T + etadot + dp | vadv_t | `compute_vertical_advection` |
| D1.6 | PGF | virtual_temp + gradients + press_diag + phis_grads | pgf_x, pgf_y | `compute_pressure_gradient_force` |
| D1.7 | Energy conversion | omega + virtual_temp + q | energy_conv | `compute_energy_conversion` |
| D1.8 | Tendency assembly | all of the above | u_flux, v_flux, temp_tend, ke | `assemble_grid_tendencies` |
| D1.9 | Spectral tendency transform | u_flux, v_flux, temp_tend, lnps_tend, ke | spectral tendencies | `grid_to_spectral_tendencies` |

**For test D1.1 (pressure diagnostics)**: Pay special attention to the TOA layer.
- Fortran sets `rlnp(:,:,1) = 99999.99` (sentinel), JAX sets `rlnp[-1] = 0.0`
- Fortran sets `alfa(:,:,1) = log(2)`, JAX sets `alfa[-1] = log(2)` -- should match
- **Document** any intentional differences. If the sentinel value propagates into downstream computations (vertical velocities, PGF), that's a real discrepancy that needs resolution.

**For test D1.6 (PGF)**: This function has been the source of 4 prior bugs. Test each of the 5 additive terms separately if the overall comparison fails:
- cofb_pressure term
- cofa term
- px2_factor (geopotential integral)
- px3u/px3v (temperature gradient integral)
- alfa term (direct temperature gradient)

**For test D1.9 (spectral tendency transform)**: This test requires converting between SHTNS packed triangular and s2fft rectangular spectral layouts. Use the converter from Section 1.3. The comparison is against Fortran's spectral output, so phase convention matters.

### Phase 2: Full Pipeline Verification

**Requires**: All Phase 1 tests passing.

| Test ID | Description | Method |
|---------|-------------|--------|
| P2.1 | `get_spectral_tendencies` end-to-end | Feed identical spectral state to both Fortran and JAX, compare spectral tendencies |
| P2.2 | `advance` one step | Compare full spectral state after 1 timestep |
| P2.3 | `advance` 10 steps | Compare spectral state; must match to < 1e-10 |
| P2.4 | `advance` 100 steps | Compare spectral state; acceptable growth to ~1e-8 |
| P2.5 | `advance` 8-day simulation | Verify no blow-up, qualitative match with Fortran |

**For P2.1**: This is the critical end-to-end test. If all Phase 1 tests pass but P2.1 fails, the bug is in the calling convention (how `full_dynamics_step` assembles its results before passing to `grid_to_spectral_tendencies`), or in `spectral_to_grid` (the inverse transform that produces grid inputs from spectral state).

**For P2.2-P2.4**: Error growth is expected due to nonlinear dynamics. The criterion is:
- Step 1: < 1e-12 relative error
- Step 10: < 1e-10 (some amplification from nonlinear feedback)
- Step 100: < 1e-8 (more amplification, still controlled)
- If errors grow faster than this, there's a remaining systematic discrepancy

---

## Part 3: Decision Framework

### When to Fix vs. Rewrite a Function

**Fix** (patch the existing code) when:
- The test failure is a simple sign error, index off-by-one, or axis flip
- The fix is <= 5 lines of code
- The fix is obvious from comparing Fortran line-by-line

**Rewrite** (delete and rebuild from scratch) when:
- Multiple interacting errors in the same function
- The function's structure doesn't map cleanly to the Fortran original
- The indexing convention is confused throughout (mixing BTU and TTB)
- More than 2 fix attempts fail

### The Blocking Rule

```
For each test T[i] in dependency order:
    Run test T[i]
    If PASS:
        Mark T[i] as verified
        Proceed to T[i+1]
    If FAIL:
        Diagnose the discrepancy
        Fix or rewrite the function under test
        Re-run test T[i]
        Do NOT proceed to T[i+1] until T[i] passes
        Also re-run all previously passing tests (regression check)
```

This is strict but essential. The debugging history proves that "close enough" compounds into instability over thousands of timesteps.

### What "Machine Precision" Means

For float64 arithmetic:
- **Exact match** (< 1e-15 relative error): Expected for simple arithmetic (additions, multiplications)
- **Near-exact** (< 1e-12 relative error): Expected for transcendental functions (log, exp, pow) and accumulated sums. This is the standard acceptance threshold.
- **Suspicious** (1e-12 to 1e-8): Investigate. May be acceptable for long cumulative sums, but document the source.
- **Failure** (> 1e-8): Indicates a real code difference. Must be resolved.

Special cases:
- **Near-zero denominators**: Compare absolute error, not relative, when the reference value is < 1e-10
- **TOA boundary**: rlnp, alfa may differ intentionally (sentinel vs guard). Document and test downstream impact.
- **Spectral layout**: After conversion between packed/rectangular, compare only modes with |m| <= l <= ntrunc. Modes outside truncation are meaningless.

---

## Part 4: Transform Library Decision

### Current Situation
- s2fft: Used for all transforms in the JAX dycore. Research-grade, designed for GPU/differentiability. Uses spin-weighted spherical harmonics.
- SHTNS: Used by Fortran dycore. Production-grade, optimized for atmospheric modeling. Uses toroidal/poloidal (sphtor) decomposition. **Python bindings are available and working.**

### Decision Point: After Phase 0 Tests

**If Phase 0 passes** (all transforms match to 1e-12):
- Keep s2fft as the transform backend
- The 2.2% v error is NOT in the transforms -- focus on dynamics functions

**If Phase 0 fails on specific operations** (e.g., vector transforms, gradients, m=0):
- Option A: Identify and fix the convention mismatch in the s2fft wrapper code
- Option B: **Replace s2fft with SHTNS Python bindings** for the JAX dycore

**If we switch to SHTNS**: The SHTNS Python module provides all needed operations:
- `sh.analys()` / `sh.synth()` -- scalar forward/inverse
- `sh.spat_to_SHsphtor()` / `sh.SHsphtor_to_spat()` -- vector forward/inverse
- `sh.synth_grad()` -- gradient computation
- Grid is Gauss-Legendre, matching the Fortran dycore exactly

The drawback: SHTNS returns NumPy arrays, losing JAX differentiability and GPU acceleration. However, **correctness comes first**. Once the model runs correctly with SHTNS, we can later replace individual transforms with s2fft equivalents, verifying each swap with the test harness.

The SHTNS spectral layout (packed triangular, complex coefficients for m >= 0) is also the **native** format of the Fortran dycore, eliminating all spectral layout conversion issues.

---

## Part 5: Implementation Plan

### Step 1: Build the Fortran dump extension (1 session)
- Add `dump_dyntend_intermediates()` to `dyn_run.f90`
- Call it from `getdyntend()` (first 5 steps only)
- Recompile the Fortran library
- Run Fortran dycore for 1 step to generate dump files
- Verify dump files load correctly

### Step 2: Build the Python test infrastructure (1 session)
- `tests/fortran_loader.py` -- binary dump reader + layout converter
- `tests/shtns_reference.py` -- SHTNS Python wrapper
- `tests/spectral_converter.py` -- packed <-> rectangular converter
- Verify the infrastructure with known-good data (e.g., load Fortran u, transform to spectral, transform back, compare)

### Step 3: Phase 0 -- Transform tests (1 session)
- Write and run all T0.x tests
- **Decision point**: Keep s2fft or switch to SHTNS?
- If switching, write the SHTNS-based transform module

### Step 4: Phase 1 -- Dynamics function tests (2-3 sessions)
- Work through D1.1 -> D1.9 in order
- Fix or rewrite each function as needed
- Each function must pass before proceeding

### Step 5: Phase 2 -- Full pipeline tests (1 session)
- Run P2.1 -> P2.5
- Verify multi-step stability
- Confirm the 8-day baroclinic wave simulation completes without blow-up

### Step 6: Cleanup (1 session)
- Remove debug dump code (or gate behind a flag)
- Update documentation
- Commit the verified code

---

## Part 6: Key Risks and Mitigations

### Risk 1: Fortran recompilation issues
**Mitigation**: The dump code is write-only I/O -- minimal risk of breaking the Fortran dycore. Keep changes in a separate `#ifdef DEBUG` block if needed.

### Risk 2: Spectral layout conversion errors mask real bugs
**Mitigation**: Validate the converter itself first, using known analytic fields (e.g., a single spherical harmonic mode). The converter should be tested independently before being used in dynamics tests.

### Risk 3: The TOA sentinel creates intentional divergence
**Mitigation**: Document the sentinel, measure its downstream impact. If Fortran's `rlnp(:,:,1) = 99999.99` produces numerically different vertical velocities at the top layer, and these propagate into a stabilizing effect, we'll need to either replicate the sentinel in JAX or add an equivalent damping mechanism (sponge layer).

### Risk 4: s2fft latitude ordering differs from SHTNS
**Mitigation**: The latitude issue (south->north vs north->south) has been a recurring source of bugs. The loader and SHTNS wrapper must handle this consistently. Add an explicit assertion at the start of every test: `assert latitudes[0] > latitudes[-1]  # north-to-south`.

### Risk 5: Virtual temperature vs temperature
**Mitigation**: Fortran dynamics operate on virtual temperature (`Tv = T * (1 + fvirt*q)`). JAX dynamics operate on temperature `T` directly. For the dry DCMIP test (q ~ 0), this is harmless. But the Fortran dumps contain `virtempg` (virtual temperature), not `T`. The loader must either: (a) convert Tv -> T before feeding to JAX functions that expect T, or (b) the JAX functions must be updated to use Tv. For now, since q ~ 0, this is a documentation issue, not a blocking issue. But it MUST be resolved before moist runs.

---

## Appendix A: Array Convention Quick Reference

| Source | Level ordering | Level 0 | Lat ordering | Layout |
|--------|--------------|---------|--------------|--------|
| Fortran grid (ug, vg, etc.) | BTU | surface | S->N | (lon, lat, lev) F-order |
| Fortran pressure (pk, dpk, alfa, rlnp) | TTB | TOA | S->N | (lon, lat, lev) F-order |
| Fortran etadot | TTB | TOA | S->N | (lon, lat, lev+1) F-order |
| Fortran spectral | -- | -- | -- | packed (ndimspec,) complex |
| JAX grid | BTU | surface | N->S | (lev, lat, lon) C-order |
| JAX pressure | BTU | surface | N->S | (lev, lat, lon) C-order |
| JAX etadot | BTU | surface | N->S | (lev+1, lat, lon) C-order |
| JAX spectral | -- | -- | -- | rect (L, 2L-1) complex |

## Appendix B: Function Dependency Graph

```
spectral_to_grid()  <-  SpectralState
    |
    +-- GridState (u, v, T, vort, div, lnps, tracers)
    +-- GridGradients (dT/dlambda, dT/dphi, d(lnps)/dlambda, d(lnps)/dphi, dq/dlambda, dq/dphi)
         |
         v
compute_pressure_diagnostics(lnps)  ->  PressureDiagnostics (ps, pk, dp, prs, alfa, rlnp)
         |
         +------------------------------+
         v                              v
compute_vertical_velocities()    compute_pressure_gradient_force()
  -> VerticalVelocities               -> (pgf_x, pgf_y)
  (omega, etadot, d_lnps_dt)          |
         |                              |
         +--------------+               |
         v              v               |
compute_vertical    compute_energy      |
_advection(u,v,T)   _conversion()       |
  -> vadv_u,v,t      -> energy_conv     |
         |              |               |
         v              v               v
assemble_grid_tendencies()
  -> GridTendencies (u_flux, v_flux, temp_tend, lnps_tend, tracer_tends, ke)
         |
         v
grid_to_spectral_tendencies()
  -> SpectralTendencies (d_vort, d_div, d_temp, d_lnps, d_tracers)
```

Each node in this graph is a test target. Tests proceed top-to-bottom; a failure at any node blocks all nodes below it.
