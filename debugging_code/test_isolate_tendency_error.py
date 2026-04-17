"""
Isolate the 2.2% divergence tendency error.

Strategy: Use the Fortran stage-0 dump which contains BOTH spectral and grid fields.
1. Compare grid-space fields (u, v, T, lnps) from Fortran dump vs JAX inverse transform
2. Feed identical grid-space fields to JAX tendency computation
3. Compare grid-space tendency intermediates
4. Identify which sub-computation introduces the 2.2% error

This uses the actual JAX dyncore transforms, NOT standalone reimplementations.
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np
import jax.numpy as jnp

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
    """Convert SHTNS m-first packed to s2fft 2D format."""
    if packed.ndim == 1:
        out = np.zeros((L, 2 * L - 1), dtype=packed.dtype)
        idx = 0
        for m in range(ntrunc + 1):
            for l in range(m, ntrunc + 1):
                out[l, (L - 1) + m] = packed[idx]
                if m > 0:
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


def compare(name, f_arr, j_arr, threshold=1e-6):
    """Compare two arrays and print diagnostics."""
    max_f = np.abs(f_arr).max()
    max_j = np.abs(j_arr).max()
    diff = np.abs(f_arr - j_arr)
    max_diff = diff.max()
    rel = max_diff / max_f if max_f > 1e-30 else 0.0
    flag = " <<<" if rel > threshold else ""
    print(f"  {name:30s} max|F|={max_f:12.4e}  max|J|={max_j:12.4e}  max|diff|={max_diff:12.4e}  rel={rel:12.4e}{flag}")
    return rel


def main():
    dump_path = "debug_data/fortran_step_1_stage_0.bin"
    if not os.path.exists(dump_path):
        print(f"ERROR: {dump_path} not found")
        return

    print("Reading Fortran stage-0 dump...")
    data = read_stage_dump(dump_path)

    # Fortran grid fields: (nlons, nlats, nlevs) -> (nlevs, nlats, nlons)
    u_f = np.transpose(data["ug"], (2, 1, 0))
    v_f = np.transpose(data["vg"], (2, 1, 0))
    t_f = np.transpose(data["virtempg"], (2, 1, 0))
    lnps_f = data["lnpsg"].T  # (nlats, nlons)
    q_f = np.transpose(data["tracerg"][:, :, :, 0], (2, 1, 0))

    # Convert spectral to s2fft format
    vrt_2d = jnp.array(packed_to_s2fft(data["vrtspec"], NTRUNC, L))
    div_2d = jnp.array(packed_to_s2fft(data["divspec"], NTRUNC, L))
    temp_2d = jnp.array(packed_to_s2fft(data["virtempspec"], NTRUNC, L))
    lnps_2d = jnp.array(packed_to_s2fft(data["lnpsspec"], NTRUNC, L))

    # ========================================================================
    # STEP 1: Compare inverse transforms (spectral -> grid)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: INVERSE TRANSFORM COMPARISON")
    print("Fortran (SHTNS) grid fields vs JAX (s2fft) inverse transform")
    print("=" * 80)

    import s2fft
    from gfs_dynamical_core.jax.transforms import TransformConfig, spectral_to_grid
    from gfs_dynamical_core.jax.states import SpectralState

    trans_config = TransformConfig(L=L, sampling="gl", ntrunc=NTRUNC, radius=6371000.0)

    # Build spectral state with tracers
    tracers_2d = jnp.zeros((1, N_LEV, L, 2 * L - 1))  # q ≈ 0 for dry test

    spec_state = SpectralState(
        vorticity=vrt_2d,
        divergence=div_2d,
        temperature=temp_2d,
        log_surface_pressure=lnps_2d,
        tracers=tracers_2d,
    )

    grid_state, grid_grads = spectral_to_grid(spec_state, trans_config)

    u_j = np.array(grid_state.u)
    v_j = np.array(grid_state.v)
    t_j = np.array(grid_state.temperature)
    lnps_j = np.array(grid_state.log_surface_pressure)
    vrt_j = np.array(grid_state.vorticity)
    div_j = np.array(grid_state.divergence)

    print("\n  Grid-space fields:")
    compare("u (zonal wind)", u_f, u_j)
    compare("v (meridional wind)", v_f, v_j)
    compare("T (temperature)", t_f, t_j)
    compare("lnps", lnps_f, lnps_j)

    # Zonal mean vs eddy for u
    u_zm_f = u_f.mean(axis=-1)
    u_zm_j = u_j.mean(axis=-1)
    u_eddy_f = u_f - u_zm_f[..., None]
    u_eddy_j = u_j - u_zm_j[..., None]
    print("\n  u decomposition:")
    compare("u zonal mean", u_zm_f, u_zm_j)
    compare("u eddy", u_eddy_f, u_eddy_j)

    print("\n  Gradients (JAX only, no Fortran reference):")
    print(f"    d_lnps/d_lambda: max = {np.abs(np.array(grid_grads.d_log_ps_d_lambda)).max():.4e}")
    print(f"    d_lnps/d_phi:    max = {np.abs(np.array(grid_grads.d_log_ps_d_phi)).max():.4e}")
    print(f"    d_T/d_lambda:    max = {np.abs(np.array(grid_grads.d_t_d_lambda)).max():.4e}")
    print(f"    d_T/d_phi:       max = {np.abs(np.array(grid_grads.d_t_d_phi)).max():.4e}")

    # ========================================================================
    # STEP 2: Compare forward vector transform (grid tendencies -> spectral)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: FORWARD VECTOR TRANSFORM ROUNDTRIP")
    print("Check: forward(inverse(vrt_spec, div_spec)) ≈ (vrt_spec, div_spec)?")
    print("=" * 80)

    from gfs_dynamical_core.jax.transforms import grid_to_spectral

    # Reconstruct spectral from JAX grid state
    spec_roundtrip = grid_to_spectral(grid_state, trans_config)

    compare("vrt roundtrip", np.array(vrt_2d), np.array(spec_roundtrip.vorticity))
    compare("div roundtrip", np.array(div_2d), np.array(spec_roundtrip.divergence))
    compare("T roundtrip", np.array(temp_2d), np.array(spec_roundtrip.temperature))
    compare("lnps roundtrip", np.array(lnps_2d), np.array(spec_roundtrip.log_surface_pressure))

    # ========================================================================
    # STEP 3: Compute JAX grid-space tendencies and compare pieces
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: JAX GRID-SPACE TENDENCY COMPUTATION")
    print("Computing pressure diagnostics, vertical velocities, PGF, etc.")
    print("=" * 80)

    from gfs_dynamical_core.jax.dynamics import (
        DynamicsConfig,
        compute_pressure_diagnostics,
        compute_vertical_velocities,
        compute_pressure_gradient_force,
        compute_energy_conversion,
        assemble_grid_tendencies,
        full_dynamics_step,
    )
    from gfs_dynamical_core.jax.transforms import get_gaussian_latitudes
    from sympl import get_constant
    import climt
    from gfs_dynamical_core.component_jax import GFSDynamicsJAX

    # Get ak, bk via climt default state (same method as compare_imex_intermediates.py)
    grid = climt.get_grid(nx=N_LON, ny=N_LAT, nz=N_LEV)
    dycore_j = GFSDynamicsJAX()
    state_j = climt.get_default_state([dycore_j], grid_state=grid)
    ak_key = "atmosphere_hybrid_sigma_pressure_a_coordinate_on_interface_levels"
    bk_key = "atmosphere_hybrid_sigma_pressure_b_coordinate_on_interface_levels"
    ak = jnp.array(state_j[ak_key])
    bk = jnp.array(state_j[bk_key])

    dbk = bk[:-1] - bk[1:]
    ck = ak[:-1] * bk[1:] - ak[1:] * bk[:-1]

    dyn_config = DynamicsConfig(
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

    print(f"  ak range: [{float(ak[0]):.2f}, {float(ak[-1]):.2f}]")
    print(f"  bk range: [{float(bk[0]):.6f}, {float(bk[-1]):.6f}]")

    latitudes = get_gaussian_latitudes(L)

    # Surface geopotential gradients (should be zero for JW06 on sphere)
    phis_grads = (jnp.zeros((N_LAT, N_LON)), jnp.zeros((N_LAT, N_LON)))

    # Compute full dynamics step
    grid_tends = full_dynamics_step(grid_state, grid_grads, phis_grads, dyn_config, latitudes)

    print(f"\n  Grid-space tendency magnitudes:")
    print(f"    u_flux:   max = {np.abs(np.array(grid_tends.u_flux)).max():.4e}")
    print(f"    v_flux:   max = {np.abs(np.array(grid_tends.v_flux)).max():.4e}")
    print(f"    T_tend:   max = {np.abs(np.array(grid_tends.temp_tend)).max():.4e}")
    print(f"    lnps_tend:max = {np.abs(np.array(grid_tends.log_ps_tend)).max():.4e}")
    print(f"    KE:       max = {np.abs(np.array(grid_tends.kinetic_energy)).max():.4e}")

    # Zonal mean vs eddy decomposition of flux terms
    u_flux = np.array(grid_tends.u_flux)
    v_flux = np.array(grid_tends.v_flux)
    ke = np.array(grid_tends.kinetic_energy)

    print(f"\n  Zonal mean of grid tendencies:")
    print(f"    u_flux zonal mean max = {np.abs(u_flux.mean(axis=-1)).max():.4e}")
    print(f"    u_flux eddy max       = {np.abs(u_flux - u_flux.mean(axis=-1)[..., None]).max():.4e}")
    print(f"    v_flux zonal mean max = {np.abs(v_flux.mean(axis=-1)).max():.4e}")
    print(f"    v_flux eddy max       = {np.abs(v_flux - v_flux.mean(axis=-1)[..., None]).max():.4e}")

    # ========================================================================
    # STEP 4: Forward transform of grid tendencies -> spectral tendencies
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: FORWARD TRANSFORM OF TENDENCIES")
    print("=" * 80)

    from gfs_dynamical_core.jax.transforms import grid_to_spectral_tendencies
    spec_tends = grid_to_spectral_tendencies(grid_tends, trans_config)

    ddivdt_j = np.array(spec_tends.d_divergence_d_t)
    dvrtdt_j = np.array(spec_tends.d_vorticity_d_t)
    dtempdt_j = np.array(spec_tends.d_temperature_d_t)

    print(f"\n  JAX spectral tendencies:")
    print(f"    d_div/dt:  max = {np.abs(ddivdt_j).max():.4e}")
    print(f"    d_vrt/dt:  max = {np.abs(dvrtdt_j).max():.4e}")
    print(f"    d_T/dt:    max = {np.abs(dtempdt_j).max():.4e}")

    # Check m=0 even-l modes of div tendency
    print(f"\n  d_div/dt m=0 even-l modes (JAX):")
    m0_idx = L - 1  # m=0 column in s2fft layout
    for l_val in range(0, min(NTRUNC + 1, 22), 2):
        val = ddivdt_j[:, l_val, m0_idx]
        print(f"    l={l_val:2d}: max|val| = {np.abs(val).max():.4e}")

    # ========================================================================
    # STEP 5: Now test with Fortran grid fields fed into JAX tendency
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: FEED FORTRAN GRID FIELDS INTO JAX TENDENCY COMPUTATION")
    print("If this matches Fortran tendency -> error is in forward transform")
    print("If this still differs -> error is in inverse transform (grid fields differ)")
    print("=" * 80)

    from gfs_dynamical_core.jax.states import GridState, GridGradients

    # We need gradients from Fortran too, but we don't have them in the dump.
    # Instead, compute gradients from Fortran spectral coefficients
    # (these should be identical to JAX gradients since spectral state matches)
    # So we use JAX gradients but Fortran grid fields

    # Build grid state from Fortran fields
    grid_state_f = GridState(
        u=jnp.array(u_f),
        v=jnp.array(v_f),
        temperature=jnp.array(t_f),
        vorticity=jnp.array(vrt_j),  # Use JAX vrt/div since Fortran dump doesn't have them
        divergence=jnp.array(div_j),  # separately (they'd need inverse scalar transform)
        log_surface_pressure=jnp.array(lnps_f),
        tracers=jnp.array(q_f)[None, :, :, :],  # (1, nlev, nlat, nlon)
    )

    # Compute tendencies with Fortran grid fields but JAX gradients
    grid_tends_f = full_dynamics_step(grid_state_f, grid_grads, phis_grads, dyn_config, latitudes)

    # Forward transform to spectral
    spec_tends_f = grid_to_spectral_tendencies(grid_tends_f, trans_config)
    ddivdt_fj = np.array(spec_tends_f.d_divergence_d_t)

    print(f"\n  Tendency from Fortran grid fields:")
    print(f"    d_div/dt (F→J): max = {np.abs(ddivdt_fj).max():.4e}")
    print(f"    d_div/dt (J→J): max = {np.abs(ddivdt_j).max():.4e}")
    compare("d_div/dt: (F grid→J tend) vs (J grid→J tend)", ddivdt_j, ddivdt_fj, threshold=1e-10)

    # Mode breakdown
    print(f"\n  d_div/dt m=0 even-l mode comparison:")
    for l_val in range(0, min(NTRUNC + 1, 22), 2):
        val_j = ddivdt_j[:, l_val, m0_idx]
        val_fj = ddivdt_fj[:, l_val, m0_idx]
        diff = np.abs(val_j - val_fj).max()
        maxv = np.abs(val_j).max()
        rel = diff / maxv if maxv > 1e-30 else 0.0
        print(f"    l={l_val:2d}: |J|={maxv:.4e}  |diff|={diff:.4e}  rel={rel:.4e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
