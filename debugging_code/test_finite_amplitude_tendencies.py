"""Compare JAX vs Fortran TOTAL spectral tendencies at finite amplitude.

The side-by-side run (/tmp/gfs_compare_scratch) makes Fortran dump its
spectral state + stage-1 total tendencies every step
(debug_data/fortran_step_NNNNN.bin). The state dumped at step N is the
state at the start of step N+1 (up to the tiny mass-fixer lnps
adjustment applied after the dump point). So:

    JAX_tend( state_from_file(N) )  vs  fortran_tend_from_file(N+1)

isolates the shared tendency path (transforms + dynamics + forward
transform) at ANY amplitude — including during wave breaking — with
no IMEX/diffusion involvement. Phase 1 (D1.x) only tested the
near-balanced initial state where vadv/omega/energy-conversion terms
are ~zero; this closes that gap.

Usage:
    python test_finite_amplitude_tendencies.py [dump_dir] [step step ...]
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
from spectral_converter import (  # noqa: E402
    ndimspec,
    s2fft_rect_to_shtns_packed,
    shtns_packed_to_s2fft_rect,
)

NTRUNC = 40
L = 64
N_LEV = 20
ND = ndimspec(NTRUNC)  # 861


def load_step(path):
    with open(path, "rb") as f:
        hdr = np.fromfile(f, dtype=np.int32, count=3)
        nd, nl, step = int(hdr[0]), int(hdr[1]), int(hdr[2])
        assert nd == ND and nl == N_LEV, (nd, nl)

        def rd3():
            # Fortran (ndimspec, nlevs) column-major -> (nlevs, ndimspec)
            return np.fromfile(f, dtype=np.complex128, count=nd * nl).reshape(nl, nd)

        def rd2():
            return np.fromfile(f, dtype=np.complex128, count=nd)

        out = {
            "step": step,
            "vrt": rd3(),
            "div": rd3(),
            "temp": rd3(),
            "lnps": rd2(),
            "dvrtdt": rd3(),
            "ddivdt": rd3(),
            "dtempdt": rd3(),
            "dlnpsdt": rd2(),
        }
        rest = np.fromfile(f, dtype=np.uint8)
        assert rest.size == 0, f"trailing bytes: {rest.size}"
        return out


def packed_levels_to_rect(arr_ln):
    """(nlevs, ndimspec) packed -> (nlevs, L, 2L-1) s2fft rect (CS applied)."""
    rect = shtns_packed_to_s2fft_rect(arr_ln.T, NTRUNC, L)  # (L, 2L-1, nlevs)
    return np.moveaxis(rect, -1, 0)


def rect_levels_to_packed(arr_rect):
    """(nlevs, L, 2L-1) -> (nlevs, ndimspec) packed (CS removed)."""
    packed = s2fft_rect_to_shtns_packed(np.moveaxis(arr_rect, 0, -1), L, NTRUNC)
    return packed.T


def rel_err(j, f):
    denom = max(np.abs(f).max(), 1e-300)
    return np.abs(j - f).max() / denom


def main():
    dump_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gfs_compare_scratch/debug_data"
    steps = [int(s) for s in sys.argv[2:]] or [1, 200, 600, 1000, 1400, 1800, 2000, 2100, 2200]

    import jax.numpy as jnp

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from tests.test_phase1_dynamics import _build_config_from_climt
    from gfs_dynamical_core.jax.dynamics import get_spectral_tendencies
    from gfs_dynamical_core.jax.states import SpectralState
    from gfs_dynamical_core.jax.transforms import TransformConfig, get_gaussian_latitudes

    dyn_config = _build_config_from_climt(n_lev=N_LEV, n_lat=L, n_lon=2 * L - 1)
    trans_config = TransformConfig(L=L, sampling="gl", radius=dyn_config.radius)
    latitudes = get_gaussian_latitudes(L)

    # Surface geopotential gradients from the actual DCMIP state, computed
    # exactly the way component_jax.array_call does (spin-1 spectral gradient).
    import climt
    import s2fft
    from gfs_dynamical_core.component_jax import GFSDynamicsJAX

    grid = climt.get_grid(nx=2 * L - 1, ny=L, nz=N_LEV)
    _dyc = GFSDynamicsJAX()
    _state = climt.get_default_state([_dyc], grid_state=grid)
    _state.update(climt.DcmipInitialConditions(add_perturbation=True)(_state))
    phis = jnp.asarray(np.asarray(_state["surface_geopotential"]))
    phis_lm = s2fft.forward_jax(phis, L, sampling="gl")
    l_arr = jnp.arange(L)
    l_factor = jnp.sqrt(l_arr * (l_arr + 1))
    f_phis_spin1 = s2fft.inverse_jax(
        -l_factor[:, None] * phis_lm, L, spin=1, sampling="gl"
    )
    phis_grads = (
        f_phis_spin1.imag / dyn_config.radius,
        -f_phis_spin1.real / dyn_config.radius,
    )
    print(f"max|phis| = {float(jnp.abs(phis).max()):.3e}")

    print(f"{'step':>6} {'amp(vrt)':>10} | {'vrt_rel':>9} {'div_rel':>9} {'T_rel':>9} {'lnps_rel':>9}"
          f" | {'vrt_abs':>9} {'div_abs':>9}")

    for N in steps:
        f_state = os.path.join(dump_dir, f"fortran_step_{N:05d}.bin")
        f_tend = os.path.join(dump_dir, f"fortran_step_{N + 1:05d}.bin")
        if not (os.path.exists(f_state) and os.path.exists(f_tend)):
            print(f"{N:>6}  -- dumps not available yet --")
            continue
        st = load_step(f_state)
        td = load_step(f_tend)

        spec = SpectralState(
            vorticity=jnp.asarray(packed_levels_to_rect(st["vrt"])),
            divergence=jnp.asarray(packed_levels_to_rect(st["div"])),
            temperature=jnp.asarray(packed_levels_to_rect(st["temp"])),
            log_surface_pressure=jnp.asarray(
                shtns_packed_to_s2fft_rect(st["lnps"], NTRUNC, L)
            ),
            tracers=jnp.zeros((1, N_LEV, L, 2 * L - 1), dtype=jnp.complex128),
        )
        tends = get_spectral_tendencies(spec, phis_grads, dyn_config, trans_config, latitudes)

        j_vrt = rect_levels_to_packed(np.asarray(tends.d_vorticity_d_t))
        j_div = rect_levels_to_packed(np.asarray(tends.d_divergence_d_t))
        j_tmp = rect_levels_to_packed(np.asarray(tends.d_temperature_d_t))
        j_lnp = s2fft_rect_to_shtns_packed(
            np.asarray(tends.d_log_surface_pressure_d_t), L, NTRUNC
        )

        print(
            f"{N:>6} {np.abs(st['vrt']).max():>10.3e} |"
            f" {rel_err(j_vrt, td['dvrtdt']):>9.2e}"
            f" {rel_err(j_div, td['ddivdt']):>9.2e}"
            f" {rel_err(j_tmp, td['dtempdt']):>9.2e}"
            f" {rel_err(j_lnp, td['dlnpsdt']):>9.2e} |"
            f" {np.abs(j_vrt - td['dvrtdt']).max():>9.2e}"
            f" {np.abs(j_div - td['ddivdt']).max():>9.2e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
