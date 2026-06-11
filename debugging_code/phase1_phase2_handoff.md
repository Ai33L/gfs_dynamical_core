# Phase 1 / Phase 2 handoff

Status snapshot for the next session. Picks up after Phase 1 (per-function
grid-space + grid→spectral verification) completed on 2026-04-19.

## Short version

* **Phase 1 — DONE, all tests pass.** Nine `test_d1_*` tests in
  [tests/test_phase1_dynamics.py](../tests/test_phase1_dynamics.py)
  verify every stage of `getdyntend` from pressure diagnostics through
  spectral tendency transform, using inputs from
  `debug_data/fortran_dyntend_call_00001.bin` and comparing to dumped
  outputs bit-for-bit.
* **Two real bugs found and fixed during Phase 1:**
  1. **`toa_pressure` hardcoded to 0.0** in [component_jax.py:116](../gfs_dynamical_core/component_jax.py)
     — production bug, caused a 0.02% pressure bias at surface
     propagating through all pressure-dependent quantities. Fixed by
     reading `get_constant("top_of_model_pressure", "Pa")`. (Not the
     2.2% v-error source on its own; propagation amplitude unverified.)
  2. **Erroneous latitude flip** in `fortran_loader.py` helpers
     (`_btu_to_jax`, `_ttb_to_jax`, `_2d_to_jax`, `_tracers_to_jax`).
     Test-only bug: flipped loaded arrays from N→S to S→N, causing
     D1.8 to fail at rel err 29.9 because `get_gaussian_latitudes`
     (N→S) did not match the loaded data. **Does NOT affect production.**
     Fixed by removing the `[::-1]` on the lat axis.
* **The 2.2% v-error is NOT in the forward tendency pipeline.** Every
  grid-space and grid→spectral stage matches Fortran/SHTNS to transform
  noise floor (≤ 3e-11 worst case, machine precision for most). The
  bug must enter downstream.

## Phase 1 test results (all passing)

| test | outputs tested | max rel err |
|------|----------------|-------------|
| D1.1 pressure diagnostics | ps, pk, dp, prs, alfa, rlnp | < 1e-12 interior |
| D1.2 vertical velocities | d_log_ps_d_t, etadot, omega | < 1e-12 interior |
| D1.3–1.5 vertical advection | vadv_u, vadv_v, vadv_t | < 1e-12 |
| D1.6 pressure gradient force | pgf_x, pgf_y | < 1e-12 interior |
| D1.7 energy conversion | kappa·omega·Tv / denom | < 1e-12 |
| D1.8 tendency assembly | u_flux, v_flux, temp_tend, ke | 5e-15 |
| D1.9 spectral tendency transform | d_vort, d_div, d_T, d_lnps | 4e-15 to 3e-11 |

TOA-only residuals exist at ~1e-9 level in `alfa[-1]`, `rlnp[-1]`,
`omega[-1]`, `pgf_y[-1]` — all traced to the same float32-literal
`log(2.)` in `pressure_data.f90:125` plus the `99999.99` rlnp sentinel.
Both are benign historical artifacts, documented and asserted
explicitly in the test file (not hidden).

## What the next session should do

The 2.2% v-error enters somewhere after grid_to_spectral_tendencies.
Three candidates, in priority order:

### Candidate A — implicit solve + diffusion (most likely)
The spectral-space update (`d_hyb_m`, `amhyb`, `bmhyb`, `tor_hyb`,
`svhyb` matrices) and the diffusion operators (`disspec`,
`diff_prof`, `dmp_prof`) are applied to spectral tendencies in
[gfs_dynamical_core/jax/stepper.py](../gfs_dynamical_core/jax/stepper.py).
This path has had suspicious commits historically (see
`334ab88`, `124a5f1`) and the sigma-level indexing there is the
active open suspect in `MEMORY.md::project_jax_dycore_debugging`.

### Candidate B — spectral → grid inverse transform
`spectral_to_grid` in [transforms.py](../gfs_dynamical_core/jax/transforms.py).
Phase 0 verified the forward transforms against SHTNS at machine
precision, but the inverse for UV reconstruction (where m=0 lives)
wasn't the primary focus — and the original 2.2% v-error was
localized to m=0 zonal mean in section 5 of the investigation log.

### Candidate C — tiny per-step residual that accumulates
Current dumps cover a single call. If the bug is ~10⁻⁴ per step it
would take days to reach 2.2%. Less likely given the earlier finding
that the error was visible at stage 0 after a single grid→spectral→grid
roundtrip (section 10 of investigation log), but worth keeping in
mind if the other two candidates come up clean.

## Concrete next steps

### Step 1: extend the Fortran dump to include post-solve spectral state

The current dump emits `dvrtspecdt, ddivspecdt, dvirtempspecdt,
dlnpsspecdt, dtracerspecdt` (the output of `getdyntend`). We need the
NEXT step's output: the state AFTER the implicit solve and diffusion
have been applied. Look at `run.f90` to find where the spectral state
is updated and add a second dump block with:

- `vrtspec`, `divspec`, `virtempspec`, `lnpsspec`, `tracerspec`
  at the END of each RK stage (after implicit solve + diffusion)
- Ideally keyed to the same `rkstage` / `call_idx` as the existing
  dumps so Phase 2 tests can chain.

### Step 2: mirror the dump loader

Extend [tests/fortran_loader.py](../tests/fortran_loader.py) to read
the new block. Re-use `shtns_packed_to_s2fft_rect` from
`spectral_converter.py`. Do **not** reintroduce the `[::-1]` lat flip.

### Step 3: D2.x tests

For each sub-stage of the implicit solve + diffusion:
- D2.1 semi-implicit matrix application to `dvrtspecdt`, `ddivspecdt`
- D2.2 diffusion via `disspec`, `diff_prof`, `dmp_prof`
- D2.3 Rayleigh damping (if separate from diffusion)
- D2.4 full spectral update (input = spectral state + tendency, output
  = post-solve spectral state)

Each test feeds Fortran-dumped inputs into the JAX equivalent and
compares against the Fortran-dumped post-solve state.

## Files that matter for Phase 2

- [gfs_dynamical_core/jax/stepper.py](../gfs_dynamical_core/jax/stepper.py) —
  `advance`, `init_semi_implicit_matrices`, `init_diffusion_operators`
- [gfs_dynamical_core/_lib/GFS/run.f90](../gfs_dynamical_core/_lib/GFS/run.f90) —
  main time-stepping loop, site for new dump hooks
- [gfs_dynamical_core/_lib/GFS/semimp_data.f90](../gfs_dynamical_core/_lib/GFS/semimp_data.f90) —
  implicit solve matrix construction (Fortran reference)
- [tests/fortran_loader.py](../tests/fortran_loader.py) — extend, not replace
- [tests/spectral_converter.py](../tests/spectral_converter.py) — already
  handles SHTNS packed ↔ s2fft rectangular conversion

## Gotchas carried over from Phase 1

1. **`JAX_PLATFORMS=cpu`** is required — METAL backend errors on
   float64 `jnp.array` (see `get_gaussian_latitudes` failure). Tests
   set this via env default.
2. **No lat flip in loader.** The existing flip-free loader is correct.
   If a future dump block includes new lat-dependent fields, do NOT
   reintroduce `[::-1]` on the lat axis.
3. **Use absolute paths to the climt env binaries:**
   - `/Users/joymonteiro/miniconda3/envs/climt/bin/pytest`
   - `/Users/joymonteiro/miniconda3/envs/climt/bin/python`
   Avoids `conda activate` per call. See `MEMORY.md::reference_climt_env`.
4. **TOA artifacts are benign.** `alfa[-1]`, `rlnp[-1]`, and derived
   fields at k=-1 inherit float32-literal and sentinel artifacts ~1e-9.
   Interior-only assertions plus bounded-TOA assertions are the
   pattern; do not chase these as bugs.
5. **Fortran `dlnpsdx` is overwritten with planetary vorticity**
   (`2*omega*sin(lats)`) before the tendency-assembly block executes
   (`dyn_run.f90:362`). The variable name is misleading — it holds `f`,
   not grad(ln ps), by the time `dump_uflux`/`dump_vflux` are written.

## Files modified in the 2026-04-19 Phase 1 session

- `gfs_dynamical_core/component_jax.py` — toa_pressure fix (line 116)
- `gfs_dynamical_core/_lib/GFS/dyn_run.f90` — dump infrastructure
  (already committed on jax-port branch)
- `tests/test_phase1_dynamics.py` — NEW, 9 passing tests
- `tests/fortran_loader.py` — NEW, produces N→S arrays matching JAX
  convention (no lat flip)
- `tests/spectral_converter.py` — NEW, SHTNS packed ↔ s2fft rect
- `tests/shtns_reference.py` — NEW, reference SHTNS bindings for tests
- `tests/conftest.py` — NEW, path fixture plumbing
- `debugging_code/investigation_log_20260323.md` — appended D1.1
  through D1.9 entries
- `debugging_code/phase1_phase2_handoff.md` — this file
