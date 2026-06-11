# Phase 0 / Phase 1 handoff

Status snapshot for the next session. Picks up from the test harness
described in `debugging_code/test_harness_plan.md`.

## Short version

* **Phase 0 (transform layer) — done, clean.** s2fft and SHTNS agree at
  machine precision on every scalar, vector, and gradient operation in
  the dycore, **including the m=0-only zonal case** that was the prime
  suspect for the 2.2% v-error. The transforms are *not* the bug.
* **Phase 1 infra — done and verified.** Fortran writes
  `debug_data/fortran_dyntend_call_NNNNN.bin` on the first 15 calls to
  `getdyntend` (5 timesteps × 3 RK stages). The 65024-byte alignment bug
  is fixed (see 2026-04-18 entry in `investigation_log_20260323.md`);
  file size matches the expected 34213100 bytes and
  `debugging_code/verify_dump_alignment.py` confirms all 9 blocks read
  with plausible values (ps=1e5, alfa 0.004–0.693, dlnpsdt ≈ 1e-6, etc.).
* **Ready to start D1.x tests** — no blocking infrastructure work remains.

## What the next session should do first

1. Start writing D1.x tests per `debugging_code/test_harness_plan.md`.
   Recommended order: D1.1 pressure diagnostics, D1.2 gradients,
   D1.3 vertical velocities, D1.4 PGF, D1.5 vertical advection,
   D1.6 energy conversion, D1.7 tendency assembly, D1.8 spectral
   tendencies.
2. For each test: feed the dumped grid state from
   `tests/fortran_loader.py:load_dyntend_dump` into the JAX equivalent
   of the corresponding Fortran operation and compare grid-space outputs.
3. Expected outcome: the first D1.x that fails identifies the location
   of the 2.2% v-error. Given Phase 0 is clean, the likely suspects are
   (in order): pressure gradient force, vertical advection, energy
   conversion, KE Laplacian in the divergence tendency.

## Alignment bug — resolved

Previously the file was 65024 bytes short (one 2D slab missing).
**Root cause**: `pressure_data.f90:75` declares `pyInterfacePressure`
as `(nlons,nlats,nlevs)` even though the Python-side allocation and
`calc_pressdata` use `nlevs+1`. `write(iu) pk` in the dump subroutine
honoured the pointer's declared shape and emitted one 2D slab too few.

**Fix**: local rebind in the dump subroutine only (production code
untouched). In `dyn_run.f90:dump_dyntend_intermediates`:

```fortran
real(r_kind), pointer :: pk_full(:,:,:) => null()
call c_f_pointer(c_loc(pk(1,1,1)), pk_full, [nlons, nlats, nlevs+1])
...
write(iu) pk_full   ! was: write(iu) pk
```

Verification script: `debugging_code/verify_dump_alignment.py`.

## Phase 0 final status

All 8 tests in `tests/test_phase0_transforms.py` pass.

| test | what it checks | s2fft vs shtns |
|---|---|---|
| T0.1 | scalar self-roundtrip (each library) | s2fft 3.0e-13 / shtns 2.9e-14 |
| T0.2 | scalar cross-library (grid-space) | 3.3e-13 |
| T0.3 | analytic `sin(phi)` (pure Y_{1,0}) roundtrip | s2fft 3.8e-13 / shtns 3.7e-14 |
| T0.4 | (u,v) → (vort,div) grids cross-library | vort 5.2e-12 / div 4.0e-11 |
| T0.5 | (vort,div) → (u,v) grids cross-library | 6.5e-15 |
| T0.6 | scalar gradient grids cross-library | d/dλ 1.2e-12, d/dφ 3.6e-11 |
| T0.7 | (u,v) full roundtrip (each library) | s2fft 2.2e-13 / shtns 2.4e-14 |
| **T0.8** | **zonal-only (u=f(φ), v=0) — the 2.2% v-error suspect** | **s2fft u=3.8e-14, v=1.3e-14 (absolute); shtns u=2.3e-15, v=9.4e-16** |

**Key insight**: the tests originally compared spectral coefficients
via a packed↔rect converter and all 8 failed. Both libraries use
different internal coefficient normalisations, so spectral-coefficient
equality is not meaningful — only grid-space comparisons are. After
rewriting every test to compare grid-space outputs, the transforms
agree at machine precision. This is now saved as a feedback memory so
future cross-library comparisons don't repeat the mistake.

**Implication for the 2.2% error**: it is NOT in the transform code.
The dycore's spin-1 forward/inverse and dual-spin formulation match
SHTNS's `SHsphtor_to_spat` to machine precision for every physical
operation the model performs. The bug must live in the dynamics layer
(pressure gradient force, vertical advection, energy conversion,
semi-implicit operator, or tendency assembly).

## Phase 1 infrastructure inventory

### Fortran side (`gfs_dynamical_core/_lib/GFS/dyn_run.f90`)

Added at module scope (lines ~30–40):

```fortran
integer, parameter :: DYNTEND_DUMP_MAX = 15   ! 5 timesteps × 3 RK stages
integer, save      :: dyntend_dump_counter = 0
```

Inside `getdyntend` (after `early_return` check, lines ~307–340):
* `do_dump` flag toggled if counter < MAX
* Snapshot buffers allocated: `dump_prsgx/dump_prsgy`, `dump_vadvu/v/t`,
  `dump_energy`, `dump_dlnpsdx/dy`, `dump_uflux/vflux/temp_tend/ke`,
  `dump_prs_layer`, `dump_dvirtempdx/dy`.
* `dlnpsdx/dy` and `dvirtempdx/dy` are snapshotted **immediately** at
  allocation time because they are overwritten later in the routine
  (dlnpsdx becomes planetary-vorticity, dvirtempdx is reused for
  tracer gradients).
* `prsgx/prsgy` snapshotted right after `getpresgrad`.
* `vadvu/v/t` snapshotted right after `getvadv` calls.
* `vadvq` (energy conversion) snapshotted after its dedicated loop.
* `u_flux / v_flux / temp_tend / ke / prs_layer` reconstructed in a
  dedicated loop outside the OMP region using the saved intermediates.

At end of routine (lines ~500–512):
* Counter incremented.
* `call dump_dyntend_intermediates(...)` with 19 arguments.
* `dump_*` buffers deallocated.

New module subroutine `dump_dyntend_intermediates` (lines ~847–932)
writes all blocks as Fortran column-major stream (no record markers)
to `debug_data/fortran_dyntend_call_NNNNN.bin`.

**Important**: the stale one-shot PGF dump block was removed; the
`fortran_pgf.bin` string no longer appears in the compiled `.so`.

### Python side (all under `tests/`)

* **`conftest.py`** — puts `tests/` on `sys.path` so sibling helpers
  can be imported.
* **`spectral_converter.py`** — `ndimspec`, and bi-directional
  conversion between SHTNS packed-triangular and s2fft rectangular
  `(L, 2L-1)` layouts with (-1)^m Condon-Shortley phase correction.
  (s2fft includes CS phase by default; Fortran SHTNS is initialised
  with `SHT_NO_CS_PHASE`.) **Still useful for bridging spectral state
  from the Fortran dump into s2fft-based dycore code; NOT useful for
  coefficient-level validation.**
* **`shtns_reference.py`** — `SHTNSReference(L, ntrunc, radius)`
  wrapper around `shtns.sht` with scalar, vector (vort/div ↔ u,v),
  and gradient operations. Mirrors Fortran `getuv`/`getvrtdivspec`/
  `getgrad` exactly (uses `invlap*R*vrt` and `invlap*R*div` as the
  `(slm, tlm)` arguments to `SHsphtor_to_spat`, etc.).
* **`fortran_loader.py`** — reads a `fortran_dyntend_call_*.bin` file
  and returns a dict matching the JAX dycore's state/tendency
  conventions:
  - 3D BTU: `transpose(2, 1, 0)[:, ::-1, :]` (level axis first, lat N→S)
  - 3D TTB: `transpose(2, 1, 0)[::-1, ::-1, :]` (flip level + lat)
  - 2D: `.T[::-1, :]` (lat flip)
  - tracers: `(nlons, nlats, nlevs, ntrac) → (ntrac, nlevs, nlats, nlons)`
  - spec packed → s2fft rect via converter, then level axis moved front
* **`test_phase0_transforms.py`** — 8 passing Phase 0 tests.

### Grid conventions (verified in this session)

| library | latitude order | N→S? |
|---|---|---|
| SHTNS Python (`sht_gauss \| SHT_PHI_CONTIGUOUS \| SHT_NO_CS_PHASE`) | `cos_theta` descending from +1 → -1 | yes |
| s2fft GL (`sampling="gl"`) | `flip(arccos(leggauss(L)[0]))` → θ ascending | yes |
| Fortran dycore grid dumps | S→N (loader applies `[::-1]` flip) | inverted vs SHTNS/s2fft |

Numerically verified: SHTNS `cos_theta` and s2fft-derived `cos_theta`
match to machine precision (identity arrays at L=16, and presumably
any L).

### Compiled artefact

```
gfs_dynamical_core/_gfs_dynamics.cpython-310-darwin.so   (Apr 18 21:51)
```

Contains: `___dyn_run_MOD_dump_dyntend_intermediates.constprop.0`
and `___dyn_run_MOD_dyntend_dump_counter`, and the string
`debug_data/fortran_dyntend_call_.bin`. Stale `fortran_pgf.bin`
string is gone.

**Timestamp warning**: between two previous sessions the installed
`.so` was newer than `dyn_run.f90` even though the compile of that
file had failed silently. Always verify the expected symbols *and*
strings are present after a build before trusting the binary.

## Known bug in the current dump — HISTORICAL (fixed 2026-04-18)

> Kept for archival reference. See "Alignment bug — resolved" section above.

The emitted file was **65024 bytes (exactly 1 × 127 × 64 × 8) short**:

```
expected total: 34213100
actual file:    34148076
shortfall:      65024 (1.00 × 2D)
```

Diagnostic script to find the misaligned block — save as
`debugging_code/inspect_dump.py` if helpful:

```python
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate climt && python debugging_code/inspect_dump.py
import numpy as np
path = 'debug_data/fortran_dyntend_call_00001.bin'
nlons, nlats, nlevs, nd, ntrac = 127, 64, 20, 861, 1
data = np.fromfile(path, dtype=np.uint8)
offset = 28  # header
def rf(n):
    global offset
    out = data[offset:offset+n*8].view(np.float64).copy()
    offset += n*8
    return out
# Block 1
for name in ('ug', 'vg', 'tv', 'divg', 'vrtg'):
    a = rf(nlons*nlats*nlevs)
    print(f'{name:8s} offset_after={offset} range {a.min():.3e} .. {a.max():.3e}')
# ... continue with the block structure in the loader ...
```

In my first run (from this session):
* Block 1 values look correct (lnps ≈ 11.513, ug ≈ 35 m/s max).
* Block 2 values look correct.
* Block 3 `pk/dpk/prs` look correct, but `alfa/rlnp/psg` are
  clearly shifted (`psg` is 0.0 instead of ~1e5 Pa).
* Block 4 `dlnpsdt` shows values near 1e-15 — suspiciously
  zero-ish, which might indicate it is reading uninitialised
  memory.

**Two hypotheses** to test next session:
1. One `write(iu) <field>` in `dump_dyntend_intermediates` never
   actually emits data because the variable is unallocated/unassociated
   at dump time. Candidates: `dlnpsdt` (the `dlnpsdt` pointer might be
   unset on the first call), or one of the `rlnp`/`psg` writes.
2. A variable has a different shape than the loader assumes. E.g.,
   `alfa` or `rlnp` might actually be (nlons, nlats, nlevs+1) on this
   branch despite the `pressure_data` declaration.

To narrow it down: move the diagnostic inspection forward one read at
a time and find the first block whose offset makes the subsequent
known-valued field (e.g. `psg` ≈ 1e5 Pa) line up.

## Files changed in this session

* `gfs_dynamical_core/_lib/GFS/dyn_run.f90` — dump scaffolding,
  snapshot buffers, dump subroutine, removed stale PGF dump, added
  `dvirtempdx_in/dy_in` plumbing after an initial compile failure.
* `tests/conftest.py` — new (sys.path shim).
* `tests/spectral_converter.py` — new.
* `tests/shtns_reference.py` — new.
* `tests/fortran_loader.py` — new.
* `tests/test_phase0_transforms.py` — new (after a rewrite: the
  original version compared spectral coefficients and all 8 tests
  failed).

## Feedback / principles saved to memory this session

* **Compare cross-library transforms in grid space, not spectral.**
  Each library uses its own internal coefficient normalisation. This
  is now in `~/.claude/.../memory/feedback_transform_comparison.md`
  and indexed in `MEMORY.md`.

## Outstanding todos (next session)

1. ~~Fix the 65024-byte dump-alignment bug~~ — done 2026-04-18
   (`dyn_run.f90`, `c_f_pointer` rebind of `pk` in the dump subroutine).
2. ~~Regenerate dumps and confirm loader reads without error~~ — done
   2026-04-18 (`debugging_code/verify_dump_alignment.py`).
3. **Implement Phase 1 tests D1.1–D1.9** per
   `debugging_code/test_harness_plan.md`. For each test: run the JAX
   equivalent of the Fortran operation on the dumped input state and
   compare grid-space against the dumped Fortran output.
4. Expected outcome: the first D1.x that fails identifies the location
   of the 2.2% v-error. Given Phase 0 is clean, the likely suspects are
   (in order): pressure gradient force, vertical advection, energy
   conversion, KE Laplacian in the divergence tendency.
5. Once Phase 1 narrows down the function-level bug, write a Phase 2
   full-pipeline test (P2.x) and then clean up the dump scaffolding
   behind a build-time flag before merging.

## Watch out: spectral vs grid comparisons in D1.x

Most D1.x tests compare **grid-space** outputs, and this is correct because
the operations in question are pure grid→grid arithmetic:

* D1.1 pressure diagnostics — `calc_pressdata` is exp/log/multiply on `lnpsg`
* D1.3 vertical velocities — vertical integration
* D1.4 PGF — arithmetic on already-gridded gradients
* D1.5 vadv, D1.6 energy conversion, D1.7 tendency assembly — grid arithmetic

Two D1.x tests cross the spectral boundary and need special care:

* **D1.2 gradients** — Fortran computes `dvirtempdx/dy` by taking the
  spectral derivative of `virtempspec` and inverse-transforming.

  **Important**: do NOT try to feed Fortran's spectral arrays directly
  into JAX. SHTNS and s2fft have different internal coefficient
  normalisations; `spectral_converter.shtns_packed_to_s2fft_rect`
  handles layout and CS phase but not the magnitude normalisation, so
  converted SHTNS coefficients fed into s2fft will give wrong-scale
  grid values. See the feedback memory
  `feedback_transform_comparison.md` and Phase 0's insight that
  "spectral-coefficient equality is not meaningful — only grid-space
  comparisons are".

  **Correct recipe**: feed the dumped `virtempg` (grid) into JAX's
  production gradient routine and compare the output grid against the
  dumped Fortran `dvirtempdx/dy`. Under the hood, JAX does a
  forward transform → spectral derivative → inverse transform, but
  Phase 0 T0.6 already proved this whole pipeline matches SHTNS's
  equivalent pipeline to machine precision in grid space (d/dλ at
  1.2e-12, d/dφ at 3.6e-11). So the noise floor for D1.2 is still
  ~1e-10 — no need to dump spectral state.

* **D1.8 spectral tendencies** — output is in spectral space. Follow
  Phase 0's rule: **do not compare packed-vs-rect coefficients**.
  Inverse-transform both the Fortran output (via `shtns_reference.py`)
  and the JAX output, and compare in grid space.

Why this matters: Phase 0 proved s2fft and SHTNS agree to machine
precision in grid space. So in D1.x, any discrepancy above ~1e-10 is a
real bug, not a transform artefact — *provided* every comparison is
made in grid space and no comparison tries to relate raw SHTNS
coefficients to raw s2fft coefficients.

## Quick-start for the next session

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate climt

# (Re)build the .so if Fortran has changed since last run:
python setup.py build_ext --inplace

# Regenerate dumps by running any driver that calls getdyntend (e.g.
# debugging_code/compare_full_run_cs.py). The first 15 calls write
# debug_data/fortran_dyntend_call_00001.bin ... _00015.bin.

# Sanity-check that the dump loads cleanly:
python debugging_code/verify_dump_alignment.py

# Then start on D1.1 per debugging_code/test_harness_plan.md.
```

Key files to open first next session:

* `debugging_code/test_harness_plan.md` — D1.x test specs.
* `tests/fortran_loader.py` — `load_dyntend_dump` returns the dumped
  state in JAX conventions.
* `tests/shtns_reference.py` — SHTNS wrapper matching Fortran's
  `getuv`/`getvrtdivspec`/`getgrad` exactly (useful if a D1.x test
  needs to operate in spectral space).
* `gfs_dynamical_core/jax/` — the JAX dycore routines that the D1.x
  tests will exercise; D1.1 needs whatever computes `pk/dp/prs/alfa/rlnp`.
