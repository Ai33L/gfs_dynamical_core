"""Compare Fortran vs JAX debug binary dumps to find where they diverge.

Usage:
    conda activate climt
    python examples/compare_debug.py

The script reads the binary files produced by dump_intermediate / dump_jax_intermediate
and compares fields at every step/stage that both codes have written.

Key facts about the two codes:
- Fortran uses ntrunc = int(nlons/3 - 2) = 40 for nlons=127, so ndimspec = 41*42/2 = 861
- JAX uses L = nlats = 64, so ntrunc = 63, spectral shape (L, 2L-1) = (64, 127)
- Fortran stores VIRTUAL temperature; JAX stores ACTUAL temperature
- Fortran grid arrays are (nlons, nlats, nlevs) column-major; JAX is (nlevs, nlats, nlons)
- Both use bottom-to-top vertical ordering for grid fields
- Fortran tracers: ntrac=1 (specific humidity only, from the climt wrapper)
- JAX tracers: ntrac=1
"""

import os
import sys
from pathlib import Path

import numpy as np

# ---------- configuration ---------------------------------------------------
DEBUG_DIR = Path(__file__).resolve().parent.parent / "debug_data"

# Grid dimensions used by the debug runs
L = 64
N_LAT = L  # 64
N_LON = 2 * L - 1  # 127
N_LEV = 20

# Fortran spectral truncation: int(127/3 - 2) = 40
NTRUNC_FORTRAN = int(N_LON / 3 - 2)  # 40
NDIMSPEC_FORTRAN = (NTRUNC_FORTRAN + 1) * (NTRUNC_FORTRAN + 2) // 2  # 861

# JAX spectral dimensions
L_JAX = L  # 64
M_JAX = 2 * L_JAX - 1  # 127

N_TRAC = 1  # both codes use 1 tracer in this setup

# Physical constants (for virtual temp conversion)
Rd = 287.05
Rv = 461.50
FVIRT = Rv / Rd - 1.0

# ---------- readers ---------------------------------------------------------


def read_fortran_bin(path):
    """Read a Fortran debug binary (stream / unformatted).

    Layout written by dump_intermediate in run.f90:
        write(unit_num) ug           ! (nlons, nlats, nlevs) real64
        write(unit_num) vg           ! (nlons, nlats, nlevs) real64
        write(unit_num) virtempg     ! (nlons, nlats, nlevs) real64
        write(unit_num) lnpsg        ! (nlons, nlats)        real64
        write(unit_num) tracerg      ! (nlons, nlats, nlevs, ntrac) real64
        write(unit_num) vrtspec      ! (ndimspec, nlevs) complex128
        write(unit_num) divspec      ! (ndimspec, nlevs) complex128
        write(unit_num) virtempspec  ! (ndimspec, nlevs) complex128
        write(unit_num) lnpsspec     ! (ndimspec,)       complex128
    All written with access='stream', form='unformatted' = raw bytes, no record markers.
    """
    data = np.fromfile(path, dtype=np.float64)
    offset = 0

    def take(n):
        nonlocal offset
        out = data[offset : offset + n]
        offset += n
        return out

    nlons, nlats, nlevs = N_LON, N_LAT, N_LEV
    ntrac = N_TRAC
    ndimspec = NDIMSPEC_FORTRAN

    # Grid fields are Fortran-order (column-major): shape (nlons, nlats, ...)
    ug = take(nlons * nlats * nlevs).reshape((nlons, nlats, nlevs), order="F")
    vg = take(nlons * nlats * nlevs).reshape((nlons, nlats, nlevs), order="F")
    tg = take(nlons * nlats * nlevs).reshape((nlons, nlats, nlevs), order="F")
    lnpsg = take(nlons * nlats).reshape((nlons, nlats), order="F")
    tracerg = take(nlons * nlats * nlevs * ntrac).reshape(
        (nlons, nlats, nlevs, ntrac), order="F"
    )

    # Convert Fortran (lon, lat, lev) -> (lev, lat, lon)
    ug = np.ascontiguousarray(ug.transpose(2, 1, 0))
    vg = np.ascontiguousarray(vg.transpose(2, 1, 0))
    tg = np.ascontiguousarray(tg.transpose(2, 1, 0))
    lnpsg = np.ascontiguousarray(lnpsg.transpose(1, 0))
    tracerg = np.ascontiguousarray(tracerg.transpose(3, 2, 1, 0))

    # Spectral fields are complex128 (16 bytes = 2 float64 each)
    remaining = data[offset:]
    cdata = remaining.view(np.complex128)
    coffset = 0

    def ctake(n):
        nonlocal coffset
        out = cdata[coffset : coffset + n]
        coffset += n
        return out

    vrtspec = ctake(ndimspec * nlevs).reshape((ndimspec, nlevs), order="F")
    divspec = ctake(ndimspec * nlevs).reshape((ndimspec, nlevs), order="F")
    virtempspec = ctake(ndimspec * nlevs).reshape((ndimspec, nlevs), order="F")
    lnpsspec = ctake(ndimspec)

    return dict(
        u=ug,
        v=vg,
        virtemp=tg,  # Fortran stores virtual temperature
        lnps=lnpsg,
        tracers=tracerg,
        vrtspec=vrtspec,
        divspec=divspec,
        virtempspec=virtempspec,
        lnpsspec=lnpsspec,
    )


def read_jax_bin(path):
    """Read a JAX debug binary.

    Layout written by dump_jax_intermediate in stepper.py:
        # Grid fields: transposed to Fortran layout then written with order='F'
        u_f   = np.asfortranarray(u.transpose(2,1,0))    -> (nlon, nlat, nlev)
        v_f   = ...
        t_f   = ...
        ps_f  = np.asfortranarray(lnps.transpose(1,0))   -> (nlon, nlat)
        q_f   = np.asfortranarray(tracers.transpose(3,2,1,0))

        # Spectral fields written as C-order complex128
        vorticity       (nlevs, L, 2L-1) complex128
        divergence      (nlevs, L, 2L-1) complex128
        temperature     (nlevs, L, 2L-1) complex128
        log_surface_pressure (L, 2L-1) complex128
    """
    data = np.fromfile(path, dtype=np.float64)
    offset = 0

    def take(n):
        nonlocal offset
        out = data[offset : offset + n]
        offset += n
        return out

    nlons, nlats, nlevs = N_LON, N_LAT, N_LEV
    ntrac = N_TRAC
    Lj = L_JAX
    M = M_JAX

    # Grid fields written as Fortran-order bytes (tobytes order='F')
    ug = take(nlons * nlats * nlevs).reshape((nlons, nlats, nlevs), order="F")
    vg = take(nlons * nlats * nlevs).reshape((nlons, nlats, nlevs), order="F")
    tg = take(nlons * nlats * nlevs).reshape((nlons, nlats, nlevs), order="F")
    lnpsg = take(nlons * nlats).reshape((nlons, nlats), order="F")
    qg = take(nlons * nlats * nlevs * ntrac).reshape(
        (nlons, nlats, nlevs, ntrac), order="F"
    )

    # Convert to (lev, lat, lon)
    ug = np.ascontiguousarray(ug.transpose(2, 1, 0))
    vg = np.ascontiguousarray(vg.transpose(2, 1, 0))
    tg = np.ascontiguousarray(tg.transpose(2, 1, 0))
    lnpsg = np.ascontiguousarray(lnpsg.transpose(1, 0))
    qg = np.ascontiguousarray(qg.transpose(3, 2, 1, 0))

    # Spectral fields: complex128, C-order rectangular (nlevs, L, M)
    remaining = data[offset:]
    cdata = remaining.view(np.complex128)
    coffset = 0

    def ctake(n):
        nonlocal coffset
        out = cdata[coffset : coffset + n]
        coffset += n
        return out

    vrtspec = ctake(nlevs * Lj * M).reshape((nlevs, Lj, M))
    divspec = ctake(nlevs * Lj * M).reshape((nlevs, Lj, M))
    tempspec = ctake(nlevs * Lj * M).reshape((nlevs, Lj, M))
    lnpsspec = ctake(Lj * M).reshape((Lj, M))

    return dict(
        u=ug,
        v=vg,
        temp=tg,  # JAX stores actual temperature (not virtual)
        lnps=lnpsg,
        tracers=qg,
        vrtspec=vrtspec,
        divspec=divspec,
        tempspec=tempspec,
        lnpsspec=lnpsspec,
    )


# ---------- comparison helpers -----------------------------------------------


def field_stats(name, arr):
    """Print basic stats for a field."""
    if np.iscomplexobj(arr):
        print(
            f"    {name:25s}  shape={str(arr.shape):20s}  "
            f"max|val|={np.max(np.abs(arr)):.6e}  "
            f"mean|val|={np.mean(np.abs(arr)):.6e}"
        )
    else:
        print(
            f"    {name:25s}  shape={str(arr.shape):20s}  "
            f"range=[{np.min(arr):.6e}, {np.max(arr):.6e}]  "
            f"mean={np.mean(arr):.6e}"
        )


def compare_field(name, f_arr, j_arr, rtol=1e-6, atol=1e-10):
    """Print comparison stats for a single field. Returns True if they differ."""
    if f_arr.shape != j_arr.shape:
        print(f"  {name:25s}  SHAPE MISMATCH  fortran={f_arr.shape}  jax={j_arr.shape}")
        return True

    diff = np.abs(f_arr - j_arr)
    f_range = np.abs(f_arr)
    j_range = np.abs(j_arr)

    max_abs_diff = np.max(diff)
    mean_abs_diff = np.mean(diff)
    max_val = max(np.max(f_range), np.max(j_range))
    rel_diff = diff / np.where(f_range > 1e-30, f_range, 1.0)
    max_rel_diff = np.max(rel_diff)

    idx = np.unravel_index(np.argmax(diff), diff.shape)

    match = np.allclose(f_arr, j_arr, rtol=rtol, atol=atol)
    status = "OK" if match else "DIFFERS"

    print(
        f"  {name:25s}  {status:8s}  "
        f"max|diff|={max_abs_diff:.4e}  mean|diff|={mean_abs_diff:.4e}  "
        f"max|rel|={max_rel_diff:.4e}  max|val|={max_val:.4e}  "
        f"@idx={idx}"
    )

    if not match:
        print(
            f"  {' ':25s}  F range: [{np.min(f_arr.real):.6e}, {np.max(f_arr.real):.6e}]  "
            f"J range: [{np.min(j_arr.real):.6e}, {np.max(j_arr.real):.6e}]"
        )
        print(f"  {' ':25s}  F@max_diff={f_arr[idx]}  J@max_diff={j_arr[idx]}")

        # Per-level breakdown for 3D fields
        if f_arr.ndim == 3 and f_arr.shape[0] == N_LEV:
            worst_levels = []
            for k in range(f_arr.shape[0]):
                ld = np.max(np.abs(f_arr[k] - j_arr[k]))
                if ld > atol * 10:
                    worst_levels.append((k, ld))
            worst_levels.sort(key=lambda x: -x[1])
            if worst_levels:
                print(f"  {' ':25s}  worst levels (of {len(worst_levels)} differing):")
                for k, ld in worst_levels[:5]:
                    fr = f_arr[k]
                    print(
                        f"  {' ':25s}    lev {k:2d}: max|diff|={ld:.4e}  "
                        f"F=[{np.min(fr):.4e},{np.max(fr):.4e}]"
                    )

    return not match


def compare_step_stage(step, stage, verbose=True):
    """Compare Fortran and JAX data for a given step and stage."""
    f_path = DEBUG_DIR / f"fortran_step_{step}_stage_{stage}.bin"
    j_path = DEBUG_DIR / f"jax_step_{step}_stage_{stage}.bin"

    if not f_path.exists() or not j_path.exists():
        return None

    if verbose:
        print(f"\n{'=' * 90}")
        print(f"Step {step}, Stage {stage}")
        print(f"{'=' * 90}")

    try:
        f_data = read_fortran_bin(f_path)
    except Exception as e:
        print(f"  ERROR reading Fortran file: {e}")
        import traceback

        traceback.print_exc()
        return None

    try:
        j_data = read_jax_bin(j_path)
    except Exception as e:
        print(f"  ERROR reading JAX file: {e}")
        import traceback

        traceback.print_exc()
        return None

    any_diff = False

    # ---- Grid fields ----
    print("  --- Grid fields ---")

    # u, v, lnps are directly comparable
    for name in ["u", "v", "lnps"]:
        d = compare_field(name, f_data[name], j_data[name])
        if d:
            any_diff = True

    # Tracers (specific humidity)
    f_q = f_data["tracers"][0]
    j_q = j_data["tracers"][0]
    d = compare_field("q (spec humidity)", f_q, j_q)
    if d:
        any_diff = True

    # Temperature: Fortran stores VIRTUAL temp, JAX stores ACTUAL temp
    # T_v = T * (1 + fvirt * q), so T = T_v / (1 + fvirt * q)
    print("  --- Temperature (Fortran=virtual, JAX=actual) ---")

    f_virtemp = f_data["virtemp"]
    j_temp = j_data["temp"]

    # Direct comparison (should differ because of virtual temp)
    d = compare_field("T_direct(virt vs act)", f_virtemp, j_temp)

    # Convert Fortran virtual temp -> actual temp
    t_actual_from_f = f_virtemp / (1.0 + FVIRT * f_q)
    d = compare_field("T_act(from F virt)", t_actual_from_f, j_temp)
    if d:
        any_diff = True

    # Convert JAX actual temp -> virtual temp
    t_virt_from_j = j_temp * (1.0 + FVIRT * j_q)
    d = compare_field("T_virt(from J act)", t_virt_from_j, f_virtemp)
    if d:
        any_diff = True

    # ---- Derived checks ----
    print("  --- Derived checks ---")

    # Surface pressure
    f_ps = np.exp(f_data["lnps"])
    j_ps = np.exp(j_data["lnps"])
    d = compare_field("ps = exp(lnps)", f_ps, j_ps)
    if d:
        any_diff = True

    return any_diff


def quick_compare(step):
    """Quick single-line comparison for a step (stage 0 only)."""
    f_path = DEBUG_DIR / f"fortran_step_{step}_stage_0.bin"
    j_path = DEBUG_DIR / f"jax_step_{step}_stage_0.bin"
    if not f_path.exists() or not j_path.exists():
        return None

    try:
        f_data = read_fortran_bin(f_path)
        j_data = read_jax_bin(j_path)
    except Exception as e:
        return f"ERROR: {e}"

    f_q = f_data["tracers"][0]
    j_q = j_data["tracers"][0]

    u_diff = np.max(np.abs(f_data["u"] - j_data["u"]))
    v_diff = np.max(np.abs(f_data["v"] - j_data["v"]))
    lnps_diff = np.max(np.abs(f_data["lnps"] - j_data["lnps"]))
    q_diff = np.max(np.abs(f_q - j_q))

    # Compare actual temperature (convert Fortran virtual -> actual)
    t_actual_f = f_data["virtemp"] / (1.0 + FVIRT * f_q)
    t_diff = np.max(np.abs(t_actual_f - j_data["temp"]))

    # Also the direct (virtual) temperature diff to see if JAX is using virtual or actual
    tvirt_diff = np.max(np.abs(f_data["virtemp"] - j_data["temp"]))

    return (
        f"|du|={u_diff:.4e}  |dv|={v_diff:.4e}  "
        f"|dT_act|={t_diff:.4e}  |dT_virt|={tvirt_diff:.4e}  "
        f"|dlnps|={lnps_diff:.4e}  |dq|={q_diff:.4e}"
    )


# ---------- main -------------------------------------------------------------


def main():
    if not DEBUG_DIR.exists():
        print(f"Debug directory not found: {DEBUG_DIR}")
        sys.exit(1)

    print(f"Fortran: ntrunc={NTRUNC_FORTRAN}, ndimspec={NDIMSPEC_FORTRAN}")
    print(f"JAX:     L={L_JAX}, ntrunc={L_JAX - 1}, spectral shape=({L_JAX},{M_JAX})")
    print(f"Grid:    {N_LON}x{N_LAT}x{N_LEV}, ntrac={N_TRAC}")
    print(f"FVIRT = Rv/Rd - 1 = {FVIRT:.6f}")
    print()

    # Verify file sizes
    f1 = DEBUG_DIR / "fortran_step_1_stage_0.bin"
    j1 = DEBUG_DIR / "jax_step_1_stage_0.bin"
    if f1.exists():
        fsize = f1.stat().st_size
        expected_f = (
            3 * N_LON * N_LAT * N_LEV + N_LON * N_LAT + N_TRAC * N_LON * N_LAT * N_LEV
        ) * 8 + (3 * NDIMSPEC_FORTRAN * N_LEV + NDIMSPEC_FORTRAN) * 16
        print(
            f"Fortran file size: {fsize} bytes (expected {expected_f}, {'OK' if fsize == expected_f else 'MISMATCH'})"
        )
    if j1.exists():
        jsize = j1.stat().st_size
        expected_j = (
            3 * N_LON * N_LAT * N_LEV + N_LON * N_LAT + N_TRAC * N_LON * N_LAT * N_LEV
        ) * 8 + (3 * N_LEV * L_JAX * M_JAX + L_JAX * M_JAX) * 16
        print(
            f"JAX file size:     {jsize} bytes (expected {expected_j}, {'OK' if jsize == expected_j else 'MISMATCH'})"
        )
    print()

    # Find all available steps
    fortran_files = sorted(DEBUG_DIR.glob("fortran_step_*_stage_0.bin"))
    jax_files = sorted(DEBUG_DIR.glob("jax_step_*_stage_0.bin"))

    f_steps = set()
    for f in fortran_files:
        parts = f.stem.split("_")
        f_steps.add(int(parts[2]))

    j_steps = set()
    for f in jax_files:
        parts = f.stem.split("_")
        j_steps.add(int(parts[2]))

    common_steps = sorted(f_steps & j_steps)
    print(f"Fortran steps: {min(f_steps)}-{max(f_steps)} ({len(f_steps)} total)")
    print(f"JAX steps:     {min(j_steps)}-{max(j_steps)} ({len(j_steps)} total)")
    print(f"Common steps:  {len(common_steps)} total")
    print()

    if not common_steps:
        print("No common steps to compare!")
        sys.exit(1)

    # Detailed comparison of first few steps
    max_detail = min(3, len(common_steps))
    for step in common_steps[:max_detail]:
        for stage in range(4):
            compare_step_stage(step, stage)

    # Summary for all common steps
    print(f"\n{'=' * 90}")
    print("Summary: stage 0 (input to advance) differences per step")
    print(f"{'=' * 90}")
    for step in common_steps:
        result = quick_compare(step)
        if result is not None:
            print(f"  step {step:3d}:  {result}")


if __name__ == "__main__":
    main()
