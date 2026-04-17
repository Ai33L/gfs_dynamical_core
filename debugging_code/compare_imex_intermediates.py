"""
Compare IMEX stage-1 intermediates between Fortran and JAX.

Runs both dycores for 1 step from identical JW06 ICs (no perturbation),
reads the Fortran binary dump of IMEX intermediates, replicates the
same computation on the JAX side, and diffs term by term.
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

from datetime import timedelta
import climt
import jax.numpy as jnp
import numpy as np
from sympl import set_constant

from gfs_dynamical_core import GFSDynamicalCore
from gfs_dynamical_core.component_jax import GFSDynamicsJAX

# ── Grid parameters ──────────────────────────────────────────────────────
L = 64
N_LON = 2 * L - 1
N_LAT = L
N_LEV = 20
NTRUNC = 40
NDIMSPEC = (NTRUNC + 1) * (NTRUNC + 2) // 2  # 861

set_constant("reference_air_pressure", value=1e5, units="Pa")


def read_fortran_imex_dump(path, ndimspec, nlevs):
    """Read the binary dump produced by dump_imex_stage1."""
    data = {}
    with open(path, "rb") as f:
        def read_c2d():
            return np.frombuffer(f.read(ndimspec * nlevs * 16),
                                 dtype=np.complex128).reshape((ndimspec, nlevs), order="F").copy()
        def read_c1d():
            return np.frombuffer(f.read(ndimspec * 16), dtype=np.complex128).copy()
        def read_r1d(n):
            return np.frombuffer(f.read(n * 8), dtype=np.float64).copy()
        def read_r2d(n, m):
            return np.frombuffer(f.read(n * m * 8),
                                 dtype=np.float64).reshape((n, m), order="F").copy()
        def read_i1d(n):
            return np.frombuffer(f.read(n * 4), dtype=np.int32).copy()

        data["ddivdt_total"] = read_c2d()
        data["dtempdt_total"] = read_c2d()
        data["dlnpsdt_total"] = read_c1d()
        data["ddivdt_lin"] = read_c2d()
        data["dtempdt_lin"] = read_c2d()
        data["dlnpsdt_lin"] = read_c1d()
        data["ddivdt_nl"] = read_c2d()
        data["dtempdt_nl"] = read_c2d()
        data["dlnpsdt_nl"] = read_c1d()
        data["disspec"] = read_r1d(ndimspec)
        data["diff_prof"] = read_r1d(nlevs)
        data["dmp_prof"] = read_r1d(nlevs)
        data["amhyb"] = read_r2d(nlevs, nlevs)
        data["bmhyb"] = read_r2d(nlevs, nlevs)
        data["tor_hyb"] = read_r1d(nlevs)
        data["svhyb"] = read_r1d(nlevs)
        data["lap"] = read_r1d(ndimspec)
        data["degree"] = read_i1d(ndimspec)
        remaining = f.read(1)
        if remaining:
            print(f"  WARNING: {len(remaining)} extra bytes in dump!")
    return data


def jax_to_packed(arr_2d, degree_map, ntrunc):
    """Convert JAX 2D (L, 2L-1) or (nlevs, L, 2L-1) to Fortran packed triangular.

    SHTNS packed order is M-FIRST:
    (l=0,m=0), (l=1,m=0), ..., (l=ntrunc,m=0),  [all m=0]
    (l=1,m=1), (l=2,m=1), ..., (l=ntrunc,m=1),  [all m=1]
    ...
    (l=ntrunc,m=ntrunc)                           [m=ntrunc]

    In JAX 2D layout: axis 0 = l (0..L-1), axis 1 = m (-L+1..L-1).
    m=0 is at index L-1, m=k is at index L-1+k, m=-k is at index L-1-k.
    """
    arr = np.array(arr_2d)
    ndimspec = len(degree_map)

    if arr.ndim == 2:
        # (L, 2L-1) -> (ndimspec,)
        L_loc = arr.shape[0]
        out = np.zeros(ndimspec, dtype=arr.dtype)
        idx = 0
        for m in range(ntrunc + 1):
            for l in range(m, ntrunc + 1):
                out[idx] = arr[l, (L_loc - 1) + m]
                idx += 1
        return out
    elif arr.ndim == 3:
        # (nlevs, L, 2L-1) -> (ndimspec, nlevs)
        nlevs = arr.shape[0]
        L_loc = arr.shape[1]
        out = np.zeros((ndimspec, nlevs), dtype=arr.dtype)
        idx = 0
        for m in range(ntrunc + 1):
            for l in range(m, ntrunc + 1):
                out[idx, :] = arr[:, l, (L_loc - 1) + m]
                idx += 1
        return out


def compare(name, f_packed, j_packed, show_levels=True, show_modes=False):
    """Compare two arrays in packed format. Returns (max_diff, rel_err)."""
    diff = np.abs(f_packed - j_packed)
    max_f = np.abs(f_packed).max()
    max_diff = diff.max()
    rel = max_diff / max_f if max_f > 1e-30 else 0.0

    flag = " <<<" if rel > 1e-4 else ""
    print(f"  {name:30s}  max|F|={max_f:12.4e}  max|diff|={max_diff:12.4e}  rel={rel:12.4e}{flag}")

    if rel > 1e-6 and show_levels and f_packed.ndim == 2:
        nlevs = f_packed.shape[1]
        for k in range(nlevs):
            dk = np.abs(f_packed[:, k] - j_packed[:, k]).max()
            mk = np.abs(f_packed[:, k]).max()
            rk = dk / mk if mk > 1e-30 else 0.0
            if rk > 1e-8:
                print(f"    lev {k:2d}: max|F|={mk:12.4e}  max|diff|={dk:12.4e}  rel={rk:12.4e}")

    if rel > 1e-6 and show_modes and f_packed.ndim == 2:
        # Show which packed modes have the largest errors
        mode_err = np.abs(f_packed - j_packed).max(axis=1)
        mode_mag = np.abs(f_packed).max(axis=1)
        worst = np.argsort(mode_err)[-10:][::-1]
        for idx in worst:
            if mode_err[idx] > 1e-30:
                print(f"    mode {idx:4d}: |F|={mode_mag[idx]:12.4e}  |diff|={mode_err[idx]:12.4e}  rel={mode_err[idx]/max(mode_mag[idx],1e-30):12.4e}")

    return max_diff, rel


def compare_1d(name, f_arr, j_arr):
    f = np.asarray(f_arr).ravel()
    j = np.asarray(j_arr).ravel()
    n = min(len(f), len(j))
    diff = np.abs(f[:n] - j[:n])
    max_f = np.abs(f[:n]).max()
    max_diff = diff.max()
    rel = max_diff / max_f if max_f > 1e-30 else 0.0
    flag = " <<<" if rel > 1e-4 else ""
    print(f"  {name:30s}  max|F|={max_f:12.4e}  max|diff|={max_diff:12.4e}  rel={rel:12.4e}{flag}")
    if rel > 1e-6:
        worst = np.argsort(diff)[-5:][::-1]
        for i in worst:
            if diff[i] > 1e-30:
                print(f"    [{i:3d}]: F={f[i]:16.8e}  J={j[i]:16.8e}  diff={diff[i]:12.4e}")
    return max_diff, rel


def main():
    print(f"Grid: {N_LON}x{N_LAT}x{N_LEV}, L={L}, ntrunc={NTRUNC}, ndimspec={NDIMSPEC}")

    grid = climt.get_grid(nx=N_LON, ny=N_LAT, nz=N_LEV)
    timestep = timedelta(minutes=5)
    dt = timestep.total_seconds()

    dcmip = climt.DcmipInitialConditions(add_perturbation=False)

    # ── Run Fortran for 1 step ────────────────────────────────────────────
    print("\n[1] Running Fortran dycore for 1 step...")
    os.makedirs("debug_data", exist_ok=True)
    dycore_f = GFSDynamicalCore()
    state_f = climt.get_default_state([dycore_f], grid_state=grid)
    out = dcmip(state_f)
    state_f.update(out)
    diag_f, out_f = dycore_f(state_f, timestep=timestep)

    # ── Initialize JAX (but save spectral state BEFORE stepping) ──────────
    print("[2] Initializing JAX dycore...")
    dycore_j = GFSDynamicsJAX()
    state_j = climt.get_default_state([dycore_j], grid_state=grid)
    out = dcmip(state_j)
    state_j.update(out)

    # Do a dummy call to initialize configs, then grab initial spectral state
    # Actually, we need to trigger initialization without stepping.
    # Let's just call it and save the initial state beforehand.
    from gfs_dynamical_core.jax.transforms import grid_to_spectral, get_gaussian_latitudes
    from gfs_dynamical_core.jax.dynamics import GridState, get_spectral_tendencies

    u = jnp.array(state_j["eastward_wind"])
    v = jnp.array(state_j["northward_wind"])
    temp = jnp.array(state_j["air_temperature"])
    ps = jnp.array(state_j["surface_air_pressure"])
    q = jnp.array(state_j["specific_humidity"])
    phis = jnp.array(state_j["surface_geopotential"])
    ak = jnp.array(state_j["atmosphere_hybrid_sigma_pressure_a_coordinate_on_interface_levels"])
    bk = jnp.array(state_j["atmosphere_hybrid_sigma_pressure_b_coordinate_on_interface_levels"])

    # Build configs manually (same as component_jax.py)
    from gfs_dynamical_core.jax.dynamics import DynamicsConfig
    from gfs_dynamical_core.jax.transforms import TransformConfig
    from gfs_dynamical_core.jax.stepper import StepperConfig, init_semi_implicit_matrices, init_diffusion_operators
    from sympl import get_constant
    import s2fft

    dbk = bk[:-1] - bk[1:]
    ck = ak[:-1] * bk[1:] - ak[1:] * bk[:-1]

    dc = DynamicsConfig(
        ak=ak, bk=bk, ck=ck, dbk=dbk,
        rk=get_constant("gas_constant_of_dry_air", "J kg^-1 K^-1")
           / get_constant("heat_capacity_of_dry_air_at_constant_pressure", "J kg^-1 K^-1"),
        toa_pressure=0.0,
        radius=get_constant("planetary_radius", "m"),
        omega=get_constant("planetary_rotation_rate", "s^-1"),
        g=get_constant("gravitational_acceleration", "m s^-2"),
        rd=get_constant("gas_constant_of_dry_air", "J kg^-1 K^-1"),
        rv=get_constant("gas_constant_of_vapor_phase", "J kg^-1 K^-1"),
        cp=get_constant("heat_capacity_of_dry_air_at_constant_pressure", "J kg^-1 K^-1"),
        cvap=get_constant("heat_capacity_of_vapor_phase", "J kg^-1 K^-1"),
    )
    tc = TransformConfig(L=L, sampling="gl", radius=dc.radius)

    _default_sc = StepperConfig(dt=dt)
    si_matrices = init_semi_implicit_matrices(dc, tc, dt,
        aa22=_default_sc.aa22, aa33=_default_sc.aa33, bb4=_default_sc.bb4)
    diff_ops = init_diffusion_operators(dc, tc, dt)
    sc = StepperConfig(
        dt=dt, explicit=False,
        amhyb=si_matrices["amhyb"], bmhyb=si_matrices["bmhyb"],
        tor_hyb=si_matrices["tor_hyb"], svhyb=si_matrices["svhyb"],
        d_hyb_m=si_matrices["d_hyb_m"],
        disspec=diff_ops["disspec"], diff_prof=diff_ops["diff_prof"],
        dmp_prof=diff_ops["dmp_prof"],
    )

    # Build initial spectral state
    grid0 = GridState(
        u=u, v=v, temperature=temp,
        vorticity=jnp.zeros_like(u), divergence=jnp.zeros_like(u),
        log_surface_pressure=jnp.log(ps),
        tracers=q[None, ...],
    )
    spec0 = grid_to_spectral(grid0, tc)

    # Compute phis gradients
    sampling = tc.sampling
    radius = dc.radius
    phis_spec = s2fft.forward_jax(phis, L, spin=0, sampling=sampling)
    l_factor = jnp.arange(L) * 1.0
    l_factor = jnp.sqrt(l_factor * (l_factor + 1.0))
    F1_phis = -l_factor[:, None] * phis_spec
    f_phis_spin1 = s2fft.inverse_jax(F1_phis, L, spin=1, sampling=sampling)
    dphisdx = f_phis_spin1.imag / radius
    dphisdy = -f_phis_spin1.real / radius
    phis_grads = (dphisdx, dphisdy)

    latitudes = get_gaussian_latitudes(L)

    # ── Compute JAX tendencies ────────────────────────────────────────────
    print("[3] Computing JAX IMEX stage-1 intermediates...")
    tends = get_spectral_tendencies(spec0, phis_grads, dc, tc, latitudes)

    l_arr = jnp.arange(L)
    lap_jax = -l_arr * (l_arr + 1.0)

    # Linear tendencies
    temp_term = jnp.einsum("ij,j...->i...", sc.amhyb, spec0.temperature)
    lnps_term = sc.tor_hyb[:, None, None] * spec0.log_surface_pressure[None, :, :]
    ddivdt_lin_j = -lap_jax[None, :, None] * (temp_term + lnps_term)
    dtempdt_lin_j = -jnp.einsum("ij,j...->i...", sc.bmhyb, spec0.divergence)
    dlnpsdt_lin_j = -jnp.einsum("i,i...->...", sc.svhyb, spec0.divergence)

    # Nonlinear = total - linear
    ddivdt_nl_j = tends.d_divergence_d_t - ddivdt_lin_j
    dtempdt_nl_j = tends.d_temperature_d_t - dtempdt_lin_j
    dlnpsdt_nl_j = tends.d_log_surface_pressure_d_t - dlnpsdt_lin_j

    # ── Load Fortran IMEX dump ────────────────────────────────────────────
    dump_path = "debug_data/fortran_imex_step_1.bin"
    if not os.path.exists(dump_path):
        print(f"\nERROR: {dump_path} not found!")
        for f in sorted(os.listdir("debug_data")):
            sz = os.path.getsize(os.path.join("debug_data", f))
            print(f"  {f}  ({sz} bytes)")
        return

    expected_size = (
        3 * NDIMSPEC * N_LEV * 16  # total tends (2d)
        + 3 * NDIMSPEC * 16        # total tends (1d)
        + 3 * NDIMSPEC * N_LEV * 16  # lin tends (2d)
        + 3 * NDIMSPEC * 16          # lin tends (1d)
        + 3 * NDIMSPEC * N_LEV * 16  # nl tends (2d)
        + 3 * NDIMSPEC * 16          # nl tends (1d)
        + NDIMSPEC * 8               # disspec
        + N_LEV * 8                  # diff_prof
        + N_LEV * 8                  # dmp_prof
        + N_LEV * N_LEV * 8 * 2      # amhyb, bmhyb
        + N_LEV * 8 * 2              # tor_hyb, svhyb
        + NDIMSPEC * 8               # lap
        + NDIMSPEC * 4               # degree
    )
    actual_size = os.path.getsize(dump_path)
    print(f"\n[4] Loading Fortran IMEX dump: expected {expected_size} bytes, got {actual_size} bytes")
    if actual_size != expected_size:
        print("  SIZE MISMATCH! Adjusting read...")

    f_data = read_fortran_imex_dump(dump_path, NDIMSPEC, N_LEV)
    degree_map = f_data["degree"]

    # Convert JAX arrays to packed format for direct comparison
    print("\n[5] Converting JAX arrays to packed format...")
    j_ddivdt_total_pk = jax_to_packed(tends.d_divergence_d_t, degree_map, NTRUNC)
    j_dtempdt_total_pk = jax_to_packed(tends.d_temperature_d_t, degree_map, NTRUNC)
    j_dlnpsdt_total_pk = jax_to_packed(tends.d_log_surface_pressure_d_t, degree_map, NTRUNC)

    j_ddivdt_lin_pk = jax_to_packed(ddivdt_lin_j, degree_map, NTRUNC)
    j_dtempdt_lin_pk = jax_to_packed(dtempdt_lin_j, degree_map, NTRUNC)
    j_dlnpsdt_lin_pk = jax_to_packed(dlnpsdt_lin_j, degree_map, NTRUNC)

    j_ddivdt_nl_pk = jax_to_packed(ddivdt_nl_j, degree_map, NTRUNC)
    j_dtempdt_nl_pk = jax_to_packed(dtempdt_nl_j, degree_map, NTRUNC)
    j_dlnpsdt_nl_pk = jax_to_packed(dlnpsdt_nl_j, degree_map, NTRUNC)

    # ══════════════════════════════════════════════════════════════════════
    # COMPARISONS
    # ══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("OPERATOR COMPARISON")
    print("=" * 80)

    compare_1d("diff_prof", f_data["diff_prof"], np.array(sc.diff_prof))
    compare_1d("dmp_prof", f_data["dmp_prof"], np.array(sc.dmp_prof))

    # Fortran lap is packed, JAX lap is per-degree. Expand JAX to packed.
    j_lap_packed = np.array([-l * (l + 1.0) for l in range(NTRUNC + 1)])
    f_lap_by_degree = np.array([f_data["lap"][i] for i in range(NDIMSPEC)])
    j_lap_expanded = np.array([j_lap_packed[degree_map[i]] for i in range(NDIMSPEC)])
    compare_1d("lap (packed)", f_lap_by_degree, j_lap_expanded)

    # Semi-implicit matrices: Fortran is top-to-bottom, JAX is bottom-to-top
    # Compare both orderings to determine which matches
    print("\n  --- amhyb: testing level ordering ---")
    j_amhyb = np.array(sc.amhyb)
    f_amhyb = f_data["amhyb"]
    d_raw = np.abs(f_amhyb - j_amhyb).max()
    d_flip = np.abs(f_amhyb[::-1, ::-1] - j_amhyb).max()
    print(f"    raw (no flip):  max|diff| = {d_raw:.4e}")
    print(f"    flipped:        max|diff| = {d_flip:.4e}")
    if d_flip < d_raw:
        print("    → Fortran is TOP-TO-BOTTOM, JAX is BOTTOM-TO-TOP (flip needed)")
        level_flip = True
    elif d_raw < d_flip:
        print("    → Same ordering (no flip needed)")
        level_flip = False
    else:
        print("    → Both match (symmetric or identical)")
        level_flip = False

    compare_1d("tor_hyb (raw)", f_data["tor_hyb"], np.array(sc.tor_hyb))

    j_svhyb = np.array(sc.svhyb)
    d_raw = np.abs(f_data["svhyb"] - j_svhyb).max()
    d_flip = np.abs(f_data["svhyb"][::-1] - j_svhyb).max()
    print(f"\n  --- svhyb: raw diff={d_raw:.4e}, flipped diff={d_flip:.4e}")

    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("TENDENCY COMPARISON")
    print("=" * 80)

    # Note: Fortran tendencies are in top-to-bottom level order.
    # If level_flip is True, we need to flip the level axis of Fortran arrays.
    if level_flip:
        print("  (Flipping Fortran level axis for comparison)")
        flip = lambda x: x[:, ::-1] if x.ndim == 2 else x
    else:
        flip = lambda x: x

    print("\n--- Total tendencies (before split) ---")
    compare("ddivdt_total", flip(f_data["ddivdt_total"]), j_ddivdt_total_pk, show_modes=True)
    compare("dtempdt_total", flip(f_data["dtempdt_total"]), j_dtempdt_total_pk)
    compare("dlnpsdt_total", f_data["dlnpsdt_total"], j_dlnpsdt_total_pk)

    print("\n--- Linear tendencies ---")
    compare("ddivdt_lin", flip(f_data["ddivdt_lin"]), j_ddivdt_lin_pk, show_modes=True)
    compare("dtempdt_lin", flip(f_data["dtempdt_lin"]), j_dtempdt_lin_pk)
    compare("dlnpsdt_lin", f_data["dlnpsdt_lin"], j_dlnpsdt_lin_pk)

    print("\n--- Nonlinear tendencies (total - linear) ---")
    compare("ddivdt_nl", flip(f_data["ddivdt_nl"]), j_ddivdt_nl_pk, show_modes=True)
    compare("dtempdt_nl", flip(f_data["dtempdt_nl"]), j_dtempdt_nl_pk)
    compare("dlnpsdt_nl", f_data["dlnpsdt_nl"], j_dlnpsdt_nl_pk)

    # ── Check initial spectral state directly ─────────────────────────────
    print("\n" + "=" * 80)
    print("INITIAL SPECTRAL STATE")
    print("=" * 80)

    stage0_path = "debug_data/fortran_step_1_stage_0.bin"
    if os.path.exists(stage0_path):
        with open(stage0_path, "rb") as fh:
            gsz = N_LON * N_LAT
            fh.read(gsz * N_LEV * 8 * 3 + gsz * 8 + gsz * N_LEV * 1 * 8)
            f_vrt0 = np.frombuffer(fh.read(NDIMSPEC * N_LEV * 16),
                                   dtype=np.complex128).reshape((NDIMSPEC, N_LEV), order="F").copy()
            f_div0 = np.frombuffer(fh.read(NDIMSPEC * N_LEV * 16),
                                   dtype=np.complex128).reshape((NDIMSPEC, N_LEV), order="F").copy()
            f_temp0 = np.frombuffer(fh.read(NDIMSPEC * N_LEV * 16),
                                    dtype=np.complex128).reshape((NDIMSPEC, N_LEV), order="F").copy()
            f_lnps0 = np.frombuffer(fh.read(NDIMSPEC * 16), dtype=np.complex128).copy()

        j_vrt0_pk = jax_to_packed(spec0.vorticity, degree_map, NTRUNC)
        j_div0_pk = jax_to_packed(spec0.divergence, degree_map, NTRUNC)
        j_temp0_pk = jax_to_packed(spec0.temperature, degree_map, NTRUNC)
        j_lnps0_pk = jax_to_packed(spec0.log_surface_pressure, degree_map, NTRUNC)

        compare("init vorticity", flip(f_vrt0), j_vrt0_pk)
        compare("init divergence", flip(f_div0), j_div0_pk)
        compare("init temperature", flip(f_temp0), j_temp0_pk)
        compare("init lnps", f_lnps0, j_lnps0_pk)

        # Level orientation check
        print("\n  Level orientation (T at l=0, m=0):")
        print(f"    Fortran lev  0: {f_temp0[0, 0].real:.2f} K")
        print(f"    Fortran lev 19: {f_temp0[0, 19].real:.2f} K")
        j_t0 = np.array(spec0.temperature)
        print(f"    JAX     lev  0: {j_t0[0, 0, L-1].real:.2f} K  (l=0,m=0)")
        print(f"    JAX     lev 19: {j_t0[19, 0, L-1].real:.2f} K  (l=0,m=0)")
        print("    (Top of atmosphere ~200K, Surface ~300K)")
    else:
        print(f"  {stage0_path} not found, skipping initial state check")

    # ── Final v comparison ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("OUTPUT STATE (after 1 step)")
    print("=" * 80)

    state_f.update(out_f)
    # Run JAX too
    diag_j, out_j = dycore_j(state_j, timestep=timestep)
    state_j.update(out_j)

    for key, short in [("eastward_wind", "u"), ("northward_wind", "v"),
                        ("air_temperature", "T"), ("surface_air_pressure", "ps")]:
        fa = np.asarray(state_f[key])
        ja = np.asarray(state_j[key])
        diff = np.abs(fa - ja)
        rel = diff.max() / max(np.abs(fa).max(), 1e-30)
        print(f"  {short:4s}  max|diff|={diff.max():12.4e}  rel={rel:12.4e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
