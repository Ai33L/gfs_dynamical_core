"""
Absolute vs relative error comparison: Fortran vs JAX spectral state.

Tests whether the ~200% relative error in div is simply because div is a
small-amplitude field (so even a small absolute error dominates the relative metric).

Reads Fortran dumps for steps 1-10, runs JAX for 10 steps, converts JAX
spectral state to packed format with (-1)^m CS correction, and reports both
absolute and relative errors for vrt and div.
"""
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np
import jax.numpy as jnp
from datetime import timedelta
import climt
from sympl import set_constant
from gfs_dynamical_core.component_jax import GFSDynamicsJAX

# ── Config ──
L = 64
N_LON = 2 * L - 1
N_LAT = L
N_LEV = 20
N_STEPS = 10
DT_MIN = 5
NTRUNC = 40

set_constant("reference_air_pressure", value=1e5, units="Pa")
grid = climt.get_grid(nx=N_LON, ny=N_LAT, nz=N_LEV)
dcmip = climt.DcmipInitialConditions(add_perturbation=True)
dt = timedelta(minutes=DT_MIN)

NDIMSPEC = (NTRUNC + 1) * (NTRUNC + 2) // 2


# ── Conversion functions ──

def s2fft_to_packed(arr_2d, ntrunc):
    """Convert s2fft (L, 2L-1) to SHTNS packed (ndimspec,).

    Applies (-1)^m CS correction: SHTNS uses Condon-Shortley phase,
    s2fft does not (or vice versa), so we multiply by (-1)^m.
    """
    L_loc = arr_2d.shape[0]
    ndimspec = (ntrunc + 1) * (ntrunc + 2) // 2
    out = np.zeros(ndimspec, dtype=arr_2d.dtype)
    idx = 0
    for m in range(ntrunc + 1):
        cs = (-1) ** m
        for l in range(m, ntrunc + 1):
            out[idx] = arr_2d[l, L_loc - 1 + m] * cs
            idx += 1
    return out


def s2fft_to_packed_3d(arr_3d, ntrunc):
    """Convert (nlevs, L, 2L-1) to (ndimspec, nlevs) with (-1)^m CS correction."""
    nlevs, L_loc, _ = arr_3d.shape
    ndimspec = (ntrunc + 1) * (ntrunc + 2) // 2
    out = np.zeros((ndimspec, nlevs), dtype=arr_3d.dtype)
    idx = 0
    for m in range(ntrunc + 1):
        cs = (-1) ** m
        for l in range(m, ntrunc + 1):
            out[idx, :] = arr_3d[:, l, L_loc - 1 + m] * cs
            idx += 1
    return out


def s2fft_to_packed_no_cs(arr_2d, ntrunc):
    """Same as s2fft_to_packed but WITHOUT CS correction."""
    L_loc = arr_2d.shape[0]
    ndimspec = (ntrunc + 1) * (ntrunc + 2) // 2
    out = np.zeros(ndimspec, dtype=arr_2d.dtype)
    idx = 0
    for m in range(ntrunc + 1):
        for l in range(m, ntrunc + 1):
            out[idx] = arr_2d[l, L_loc - 1 + m]
            idx += 1
    return out


def s2fft_to_packed_3d_no_cs(arr_3d, ntrunc):
    """Same as s2fft_to_packed_3d but WITHOUT CS correction."""
    nlevs, L_loc, _ = arr_3d.shape
    ndimspec = (ntrunc + 1) * (ntrunc + 2) // 2
    out = np.zeros((ndimspec, nlevs), dtype=arr_3d.dtype)
    idx = 0
    for m in range(ntrunc + 1):
        for l in range(m, ntrunc + 1):
            out[idx, :] = arr_3d[:, l, L_loc - 1 + m]
            idx += 1
    return out


def read_fortran_dump(step):
    """Read Fortran binary dump for a given step."""
    path = f"debug_data/fortran_step_{step:05d}.bin"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        ndimspec, nlevs, step_num = np.fromfile(f, np.int32, 3)

        def read_c2d():
            return np.fromfile(f, np.complex128, ndimspec * nlevs).reshape(
                (ndimspec, nlevs), order="F"
            )

        def read_c1d():
            return np.fromfile(f, np.complex128, ndimspec)

        data = {}
        data["vrtspec"] = read_c2d()
        data["divspec"] = read_c2d()
        data["virtempspec"] = read_c2d()
        data["lnpsspec"] = read_c1d()
        data["dvrtdt"] = read_c2d()
        data["ddivdt"] = read_c2d()
        data["dtempdt"] = read_c2d()
        data["dlnpsdt"] = read_c1d()
        data["ndimspec"] = ndimspec
        data["nlevs"] = nlevs
    return data


def main():
    print(f"Grid: {N_LON}x{N_LAT}x{N_LEV}, L={L}, ntrunc={NTRUNC}")
    print(f"Running {N_STEPS} steps at dt={DT_MIN} min")
    print(f"NDIMSPEC = {NDIMSPEC}")
    print()

    # ── Init JAX ──
    dycore_j = GFSDynamicsJAX()
    state_j = climt.get_default_state([dycore_j], grid_state=grid)
    state_j.update(dcmip(state_j))

    # First, determine which CS convention is correct by checking step 1
    # with and without the correction
    print("=" * 100)
    print("STEP 0: Checking CS correction convention (step 1 preview)")
    print("=" * 100)

    # Step JAX once
    diag_j, out_j = dycore_j(state_j, timestep=dt)
    state_j.update(out_j)
    state_j["time"] += dt

    fdata = read_fortran_dump(1)
    jspec = dycore_j._spec_state

    j_vrt_cs = s2fft_to_packed_3d(np.array(jspec.vorticity), NTRUNC)
    j_vrt_no = s2fft_to_packed_3d_no_cs(np.array(jspec.vorticity), NTRUNC)

    err_cs = np.abs(fdata["vrtspec"] - j_vrt_cs).max()
    err_no = np.abs(fdata["vrtspec"] - j_vrt_no).max()
    amp_f = np.abs(fdata["vrtspec"]).max()

    print(f"  vrt max|F|       = {amp_f:.6e}")
    print(f"  vrt err WITH CS  = {err_cs:.6e}  (rel={err_cs/amp_f:.6e})")
    print(f"  vrt err NO CS    = {err_no:.6e}  (rel={err_no/amp_f:.6e})")
    use_cs = err_cs < err_no
    print(f"  -> Using {'WITH' if use_cs else 'WITHOUT'} CS correction")
    print()

    # Pick the conversion functions
    if use_cs:
        pack_3d = s2fft_to_packed_3d
        pack_1d = s2fft_to_packed
    else:
        pack_3d = s2fft_to_packed_3d_no_cs
        pack_1d = s2fft_to_packed_no_cs

    # Now reset and redo from scratch
    dycore_j = GFSDynamicsJAX()
    state_j = climt.get_default_state([dycore_j], grid_state=grid)
    state_j.update(dcmip(state_j))

    # Header
    print("=" * 130)
    print(f"{'step':>4} | {'':^52s} | {'':^52s} |")
    print(f"{'':>4} | {'VORTICITY':^52s} | {'DIVERGENCE':^52s} |")
    print(f"{'':>4} | {'max|F|':>12} {'max|J|':>12} {'abs_err':>12} {'rel_err':>12} | "
          f"{'max|F|':>12} {'max|J|':>12} {'abs_err':>12} {'rel_err':>12} |")
    print("-" * 130)

    for step in range(1, N_STEPS + 1):
        # Step JAX
        diag_j, out_j = dycore_j(state_j, timestep=dt)
        state_j.update(out_j)
        state_j["time"] += dt

        # Read Fortran dump
        fdata = read_fortran_dump(step)
        if fdata is None:
            print(f"{step:4d} | Fortran dump not found")
            continue

        jspec = dycore_j._spec_state

        # Convert JAX to packed
        j_vrt = pack_3d(np.array(jspec.vorticity), NTRUNC)
        j_div = pack_3d(np.array(jspec.divergence), NTRUNC)
        j_temp = pack_3d(np.array(jspec.temperature), NTRUNC)
        j_lnps = pack_1d(np.array(jspec.log_surface_pressure), NTRUNC)

        # Vorticity
        f_vrt_amp = np.abs(fdata["vrtspec"]).max()
        j_vrt_amp = np.abs(j_vrt).max()
        vrt_abs = np.abs(fdata["vrtspec"] - j_vrt).max()
        vrt_rel = vrt_abs / max(f_vrt_amp, 1e-30)

        # Divergence
        f_div_amp = np.abs(fdata["divspec"]).max()
        j_div_amp = np.abs(j_div).max()
        div_abs = np.abs(fdata["divspec"] - j_div).max()
        div_rel = div_abs / max(f_div_amp, 1e-30)

        print(
            f"{step:4d} | "
            f"{f_vrt_amp:12.4e} {j_vrt_amp:12.4e} {vrt_abs:12.4e} {vrt_rel:12.4e} | "
            f"{f_div_amp:12.4e} {j_div_amp:12.4e} {div_abs:12.4e} {div_rel:12.4e} |"
        )

    # Summary
    print()
    print("=" * 130)
    print("SUMMARY")
    print("=" * 130)

    # Re-read final step for summary
    fdata = read_fortran_dump(N_STEPS)
    jspec = dycore_j._spec_state
    j_vrt = pack_3d(np.array(jspec.vorticity), NTRUNC)
    j_div = pack_3d(np.array(jspec.divergence), NTRUNC)
    j_temp = pack_3d(np.array(jspec.temperature), NTRUNC)
    j_lnps = pack_1d(np.array(jspec.log_surface_pressure), NTRUNC)

    for name, f_arr, j_arr in [
        ("vrt", fdata["vrtspec"], j_vrt),
        ("div", fdata["divspec"], j_div),
        ("T", fdata["virtempspec"], j_temp),
        ("lnps", fdata["lnpsspec"], j_lnps),
    ]:
        f_amp = np.abs(f_arr).max()
        j_amp = np.abs(j_arr).max()
        abs_err = np.abs(f_arr - j_arr).max()
        rel_err = abs_err / max(f_amp, 1e-30)
        print(f"  {name:6s}: max|F|={f_amp:.4e}  max|J|={j_amp:.4e}  "
              f"abs_err={abs_err:.4e}  rel_err={rel_err:.4e}")

    print()
    vrt_abs_final = np.abs(fdata["vrtspec"] - j_vrt).max()
    div_abs_final = np.abs(fdata["divspec"] - j_div).max()
    print(f"  abs_err(div) / abs_err(vrt) = {div_abs_final / max(vrt_abs_final, 1e-30):.4f}")
    print()
    if div_abs_final < vrt_abs_final * 10:
        print("  CONCLUSION: div absolute error is comparable to vrt absolute error.")
        print("  The ~200% relative error in div is because div is a SMALL field,")
        print("  not because the JAX dycore computes div incorrectly.")
    else:
        print("  CONCLUSION: div absolute error is MUCH LARGER than vrt absolute error.")
        print("  There is likely a genuine bug in the div computation.")


if __name__ == "__main__":
    main()
