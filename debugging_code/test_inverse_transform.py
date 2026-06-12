"""
Test inverse vector transform: SHTNS getuv vs s2fft spin-1 inverse.

Takes the Fortran stage-0 dump which contains BOTH spectral (vrt, div) and
grid (u, v) fields. The grid fields were produced by SHTNS getuv. We feed
the same spectral fields to s2fft inverse and compare.

This directly tests whether the inverse vector transform is the source of
the 2.2% v error.
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np
import jax.numpy as jnp
import s2fft

L = 64
N_LON = 2 * L - 1  # 127
N_LAT = L           # 64
N_LEV = 20
NTRUNC = 40
NDIMSPEC = (NTRUNC + 1) * (NTRUNC + 2) // 2  # 861


def read_stage_dump(path):
    """Read Fortran stage dump: grid fields then spectral fields."""
    data = {}
    with open(path, "rb") as f:
        gsz = N_LON * N_LAT
        # Grid fields are (nlons, nlats, nlevs) in Fortran column-major
        data["ug"] = np.frombuffer(f.read(gsz * N_LEV * 8),
                                    dtype=np.float64).reshape((N_LON, N_LAT, N_LEV), order="F").copy()
        data["vg"] = np.frombuffer(f.read(gsz * N_LEV * 8),
                                    dtype=np.float64).reshape((N_LON, N_LAT, N_LEV), order="F").copy()
        data["virtempg"] = np.frombuffer(f.read(gsz * N_LEV * 8),
                                          dtype=np.float64).reshape((N_LON, N_LAT, N_LEV), order="F").copy()
        data["lnpsg"] = np.frombuffer(f.read(gsz * 8),
                                       dtype=np.float64).reshape((N_LON, N_LAT), order="F").copy()
        data["tracerg"] = np.frombuffer(f.read(gsz * N_LEV * 1 * 8),
                                         dtype=np.float64).reshape((N_LON, N_LAT, N_LEV, 1), order="F").copy()
        # Spectral fields: (ndimspec, nlevs) complex128 column-major
        data["vrtspec"] = np.frombuffer(f.read(NDIMSPEC * N_LEV * 16),
                                         dtype=np.complex128).reshape((NDIMSPEC, N_LEV), order="F").copy()
        data["divspec"] = np.frombuffer(f.read(NDIMSPEC * N_LEV * 16),
                                         dtype=np.complex128).reshape((NDIMSPEC, N_LEV), order="F").copy()
        data["virtempspec"] = np.frombuffer(f.read(NDIMSPEC * N_LEV * 16),
                                             dtype=np.complex128).reshape((NDIMSPEC, N_LEV), order="F").copy()
        data["lnpsspec"] = np.frombuffer(f.read(NDIMSPEC * 16),
                                          dtype=np.complex128).copy()
    return data


def packed_to_s2fft(packed, ntrunc, L):
    """Convert SHTNS m-first packed (ndimspec,) or (ndimspec, nlevs) to s2fft 2D (L, 2L-1).

    SHTNS m-first: for m=0: l=0..ntrunc, for m=1: l=1..ntrunc, etc.
    s2fft 2D: axis 0 = l (0..L-1), axis 1 = m (-L+1..L-1), m=0 at index L-1.
    """
    if packed.ndim == 1:
        out = np.zeros((L, 2 * L - 1), dtype=packed.dtype)
        idx = 0
        for m in range(ntrunc + 1):
            for l in range(m, ntrunc + 1):
                out[l, (L - 1) + m] = packed[idx]
                if m > 0:
                    # For real fields, negative m = conj(positive m) * (-1)^m
                    # But SHTNS stores complex coefficients where this is already handled
                    out[l, (L - 1) - m] = np.conj(packed[idx]) * (-1)**m
                idx += 1
        return out
    elif packed.ndim == 2:
        nlevs = packed.shape[1]
        out = np.zeros((nlevs, L, 2 * L - 1), dtype=packed.dtype)
        idx = 0
        for m in range(ntrunc + 1):
            for l in range(m, ntrunc + 1):
                out[:, l, (L - 1) + m] = packed[idx, :]
                if m > 0:
                    out[:, l, (L - 1) - m] = np.conj(packed[idx, :]) * (-1)**m
                idx += 1
        return out


def shtns_inverse_vector(vrtspec_2d, divspec_2d, L, radius):
    """Replicate SHTNS getuv using s2fft spin-1 inverse.

    getuv computes: u = invlap * R * curl(vrt) component
                    v = invlap * R * (-grad(div)) component

    In spin-1 formulation:
    F1_lm = invlap * R * (-vrt_lm + i*div_lm)  [??]

    Actually, the standard relationship is:
    u + iv = sum_lm [ invlap * R * (vrt_lm * Y^1_lm) ] for the vorticity part
    u + iv = sum_lm [ invlap * R * (-div_lm * ???) ] for the divergence part

    Let's use the direct approach from transforms.py.
    """
    # From transforms.py spectral_to_grid (lines ~80-100):
    # F1_lm = invlap_factor * R * (vrt + 1j * div)
    # f_spin1 = s2fft.inverse(F1_lm, spin=1)
    # u = f_spin1.real
    # v = -f_spin1.imag

    l_arr = jnp.arange(L, dtype=jnp.float64)
    lap = l_arr * (l_arr + 1.0)
    invlap = jnp.where(lap > 0, 1.0 / lap, 0.0)

    vrt_2d = jnp.array(vrtspec_2d)
    div_2d = jnp.array(divspec_2d)

    # Combine: F1 = invlap * R * (vrt + 1j * div)
    F1 = invlap[None, :, None] * radius * (vrt_2d + 1j * div_2d)

    # Inverse spin-1 transform level by level
    n_lev = F1.shape[0]
    u_all = []
    v_all = []
    for k in range(n_lev):
        f_spin1 = s2fft.inverse_jax(F1[k], L, spin=1, sampling="gl")
        u_all.append(np.array(f_spin1.real))
        v_all.append(np.array(-f_spin1.imag))

    return np.stack(u_all, axis=0), np.stack(v_all, axis=0)


def main():
    # Read the Fortran stage-0 dump (initial state, after getuv reconstruction)
    dump_path = "debug_data/fortran_step_1_stage_0.bin"
    if not os.path.exists(dump_path):
        print(f"ERROR: {dump_path} not found")
        return

    print(f"Reading {dump_path}...")
    data = read_stage_dump(dump_path)

    # Fortran grid fields: (nlons, nlats, nlevs) → transpose to (nlevs, nlats, nlons)
    # s2fft GL sampling: (nlats, nlons) = (L, 2L-1) = (64, 127)
    # Fortran stores (nlons, nlats, nlevs) column-major
    u_fortran = np.transpose(data["ug"], (2, 1, 0))  # (nlevs, nlats, nlons)
    v_fortran = np.transpose(data["vg"], (2, 1, 0))

    print(f"Fortran grid u shape: {u_fortran.shape}, max|u| = {np.abs(u_fortran).max():.6e}")
    print(f"Fortran grid v shape: {v_fortran.shape}, max|v| = {np.abs(v_fortran).max():.6e}")

    # Convert spectral fields from packed to s2fft 2D format
    print("\nConverting spectral to s2fft format...")
    vrt_2d = packed_to_s2fft(data["vrtspec"], NTRUNC, L)  # (nlevs, L, 2L-1)
    div_2d = packed_to_s2fft(data["divspec"], NTRUNC, L)

    # Reconstruct (u, v) using s2fft inverse
    from sympl import get_constant
    radius = get_constant("planetary_radius", "m")
    print(f"Running s2fft inverse vector transform (R = {radius:.1f} m)...")
    u_s2fft, v_s2fft = shtns_inverse_vector(vrt_2d, div_2d, L, radius)

    print(f"s2fft grid u shape: {u_s2fft.shape}, max|u| = {np.abs(u_s2fft).max():.6e}")
    print(f"s2fft grid v shape: {v_s2fft.shape}, max|v| = {np.abs(v_s2fft).max():.6e}")

    # Compare
    print("\n" + "=" * 80)
    print("INVERSE VECTOR TRANSFORM COMPARISON")
    print("SHTNS getuv (Fortran dump) vs s2fft spin-1 inverse")
    print("=" * 80)

    for name, f_arr, j_arr in [("u", u_fortran, u_s2fft), ("v", v_fortran, v_s2fft)]:
        diff = np.abs(f_arr - j_arr)
        max_f = np.abs(f_arr).max()
        max_diff = diff.max()
        rel = max_diff / max_f if max_f > 1e-30 else 0.0
        rms = np.sqrt((diff**2).mean())
        print(f"\n  {name}:")
        print(f"    max|F|     = {max_f:12.6e}")
        print(f"    max|s2fft| = {np.abs(j_arr).max():12.6e}")
        print(f"    max|diff|  = {max_diff:12.6e}")
        print(f"    rel_err    = {rel:12.6e}")
        print(f"    rms_diff   = {rms:12.6e}")

        if rel > 1e-6:
            print(f"\n    Per-level breakdown:")
            for k in range(N_LEV):
                dk = np.abs(f_arr[k] - j_arr[k]).max()
                mk = np.abs(f_arr[k]).max()
                rk = dk / mk if mk > 1e-30 else 0.0
                flag = " <<<" if rk > 1e-4 else ""
                print(f"      lev {k:2d}: max|F|={mk:12.4e}  max|diff|={dk:12.4e}  rel={rk:12.4e}{flag}")

    # Decompose into zonal mean vs eddy
    print("\n" + "=" * 80)
    print("ZONAL MEAN vs EDDY DECOMPOSITION")
    print("=" * 80)

    for name, f_arr, j_arr in [("u", u_fortran, u_s2fft), ("v", v_fortran, v_s2fft)]:
        zm_f = f_arr.mean(axis=-1)  # (nlevs, nlats)
        zm_j = j_arr.mean(axis=-1)
        eddy_f = f_arr - zm_f[..., None]
        eddy_j = j_arr - zm_j[..., None]

        zm_diff = np.abs(zm_f - zm_j).max()
        eddy_diff = np.abs(eddy_f - eddy_j).max()
        print(f"\n  {name}:")
        print(f"    Zonal mean diff: max = {zm_diff:12.6e}")
        print(f"    Eddy diff:       max = {eddy_diff:12.6e}")

    # Also test scalar inverse (temperature) as a control
    print("\n" + "=" * 80)
    print("SCALAR INVERSE TRANSFORM (control)")
    print("=" * 80)

    temp_2d = packed_to_s2fft(data["virtempspec"], NTRUNC, L)
    t_s2fft = np.zeros((N_LEV, N_LAT, N_LON))
    for k in range(N_LEV):
        t_s2fft[k] = np.array(s2fft.inverse_jax(jnp.array(temp_2d[k]), L, spin=0, sampling="gl")).real

    t_fortran = np.transpose(data["virtempg"], (2, 1, 0))
    diff_t = np.abs(t_fortran - t_s2fft)
    print(f"  Temperature: max|diff| = {diff_t.max():.6e}, rel = {diff_t.max()/np.abs(t_fortran).max():.6e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
