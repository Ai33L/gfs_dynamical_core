"""
Test grid_to_spectral_tendencies for m=0 modes.

Creates a synthetic purely-zonal u_flux (m=0 only) and checks
what vorticity/divergence tendencies are produced. Then does the
inverse to see what v comes out.

The hypothesis: the dual-spin forward transform in
grid_to_spectral_tendencies has a systematic error for m=0 modes
that causes the ~5% underestimate of |v|.
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

import jax
import jax.numpy as jnp
import numpy as np
import s2fft

L = 64
N_LON = 2 * L - 1
N_LAT = L
SAMPLING = "gl"

# Get Gaussian latitudes for the GL grid
from numpy.polynomial.legendre import leggauss
_, _ = leggauss(L)


def test_vector_tendency_roundtrip():
    """
    Test: given a known u_flux = cos(lat), v_flux = 0 (purely zonal, m=0),
    compute curl and div in spectral space, then convert back to grid.

    For u_flux = cos(lat), v_flux = 0:
    - curl(u_flux, 0) should give a specific pattern
    - div(u_flux, 0) should give a specific pattern

    We compare the dual-spin approach vs a single-spin approach.
    """
    print("=" * 80)
    print("TEST 1: Vector tendency transform roundtrip for m=0")
    print("=" * 80)

    # Create Gaussian latitudes
    from gfs_dynamical_core.jax.transforms import get_gaussian_latitudes
    lats = get_gaussian_latitudes(L)  # (L,) in radians, N->S

    # u_flux = cos(lat), v_flux = 0  (purely zonal)
    u_flux = jnp.cos(lats)[:, None] * jnp.ones((1, N_LON))
    v_flux = jnp.zeros_like(u_flux)

    radius = 6.3712e6
    l_arr = jnp.arange(L)
    l_factor = jnp.sqrt(l_arr * (l_arr + 1))
    inv_l_factor = jnp.where(l_arr > 0, 1.0 / l_factor, 0.0)

    # Method 1: Dual-spin (current JAX code)
    f_plus = -v_flux + 1j * u_flux   # spin +1
    f_minus = v_flux + 1j * u_flux   # spin -1

    F1_lm = s2fft.forward_jax(f_plus, L, spin=1, sampling=SAMPLING)
    Fm1_lm = s2fft.forward_jax(f_minus, L, spin=-1, sampling=SAMPLING)

    result_p = l_factor[:, None] * F1_lm / radius
    result_m = l_factor[:, None] * Fm1_lm / radius

    div_of_flux_dual = (result_p + result_m) / 2
    curl_of_flux_dual = (result_p - result_m) / (2j)

    d_vort_dual = -div_of_flux_dual
    d_div_dual = curl_of_flux_dual

    # Method 2: Single spin+1 only (older approach, using .real/.imag split)
    F1_only = s2fft.forward_jax(f_plus, L, spin=1, sampling=SAMPLING)
    result_single = l_factor[:, None] * F1_only / radius
    d_div_single = result_single.real    # Re(D + i*zeta) = D
    d_vort_single_raw = result_single.imag  # Im(D + i*zeta) = zeta
    # d_vort = -div_of_flux = ... well, for single spin:
    # result = (div + i*curl) of the flux
    # So div_of_flux = result.real, curl_of_flux = result.imag
    # d_vort = -div_of_flux = -result.real
    # d_div  = curl_of_flux = result.imag
    d_vort_single = -result_single.real
    d_div_single = result_single.imag

    print("\n  Spectral tendency comparison (dual vs single spin):")
    print(f"    d_vort dual  m=0 max: {float(jnp.abs(d_vort_dual[:, L-1]).max()):.6e}")
    print(f"    d_vort single m=0 max: {float(jnp.abs(d_vort_single[:, L-1]).max()):.6e}")
    diff_vort = jnp.abs(d_vort_dual - d_vort_single)
    print(f"    d_vort diff max (all m): {float(diff_vort.max()):.6e}")
    print(f"    d_vort diff max (m=0):   {float(jnp.abs(d_vort_dual[:, L-1] - d_vort_single[:, L-1]).max()):.6e}")

    print(f"\n    d_div dual  m=0 max: {float(jnp.abs(d_div_dual[:, L-1]).max()):.6e}")
    print(f"    d_div single m=0 max: {float(jnp.abs(d_div_single[:, L-1]).max()):.6e}")
    diff_div = jnp.abs(d_div_dual - d_div_single)
    print(f"    d_div diff max (all m): {float(diff_div.max()):.6e}")

    # Now convert back to grid: do spectral -> grid for the resulting vort/div
    # to get the implied u,v change
    F1_vort_div = inv_l_factor[:, None] * (d_div_dual + 1j * d_vort_dual) * radius
    f_uv = s2fft.inverse_jax(F1_vort_div, L, spin=1, sampling=SAMPLING)
    u_from_tends = f_uv.imag
    v_from_tends = -f_uv.real

    print(f"\n  Implied u,v from dual-spin tendency -> inverse transform:")
    print(f"    u max: {float(jnp.abs(u_from_tends).max()):.6e}")
    print(f"    v max: {float(jnp.abs(v_from_tends).max()):.6e}")

    # Same with single-spin tendencies
    F1_vort_div_s = inv_l_factor[:, None] * (d_div_single + 1j * d_vort_single) * radius
    f_uv_s = s2fft.inverse_jax(F1_vort_div_s, L, spin=1, sampling=SAMPLING)
    u_from_tends_s = f_uv_s.imag
    v_from_tends_s = -f_uv_s.real

    print(f"\n  Implied u,v from single-spin tendency -> inverse transform:")
    print(f"    u max: {float(jnp.abs(u_from_tends_s).max()):.6e}")
    print(f"    v max: {float(jnp.abs(v_from_tends_s).max()):.6e}")

    print(f"\n  Difference dual vs single in reconstructed v:")
    print(f"    max: {float(jnp.abs(v_from_tends - v_from_tends_s).max()):.6e}")


def test_m0_vortdiv_to_uv():
    """
    Test: for purely m=0 spectral vorticity, what u,v does the spin-1
    inverse give? Compare with a direct Legendre approach.
    """
    print(f"\n{'='*80}")
    print("TEST 2: m=0 spectral vorticity -> u,v via spin-1 inverse")
    print("=" * 80)

    radius = 6.3712e6
    l_arr = jnp.arange(L)
    l_factor = jnp.sqrt(l_arr * (l_arr + 1))
    inv_l_factor = jnp.where(l_arr > 0, 1.0 / l_factor, 0.0)

    # Create a simple m=0 vorticity spectrum: vort_lm = delta(l, 1)
    vort = jnp.zeros((L, 2*L-1), dtype=complex)
    vort = vort.at[1, L-1].set(1.0 + 0j)  # l=1, m=0
    div = jnp.zeros_like(vort)

    # Inverse vector transform
    F1_lm = inv_l_factor[:, None] * (div + 1j * vort) * radius
    f_spin1 = s2fft.inverse_jax(F1_lm, L, spin=1, sampling=SAMPLING)
    u = f_spin1.imag
    v = -f_spin1.real

    print(f"  Input: vort(l=1, m=0) = 1.0")
    print(f"  Output u max: {float(jnp.abs(u).max()):.6e} (should be nonzero)")
    print(f"  Output v max: {float(jnp.abs(v).max()):.6e} (should be zero for m=0 vort)")
    print(f"  Output u zonal mean max: {float(jnp.abs(u.mean(axis=-1)).max()):.6e}")
    print(f"  Output v zonal mean max: {float(jnp.abs(v.mean(axis=-1)).max()):.6e}")

    # For l=1, m=0 divergence instead
    vort2 = jnp.zeros_like(vort)
    div2 = jnp.zeros_like(vort)
    div2 = div2.at[1, L-1].set(1.0 + 0j)

    F1_lm2 = inv_l_factor[:, None] * (div2 + 1j * vort2) * radius
    f_spin1_2 = s2fft.inverse_jax(F1_lm2, L, spin=1, sampling=SAMPLING)
    u2 = f_spin1_2.imag
    v2 = -f_spin1_2.real

    print(f"\n  Input: div(l=1, m=0) = 1.0")
    print(f"  Output u max: {float(jnp.abs(u2).max()):.6e} (should be zero for m=0 div)")
    print(f"  Output v max: {float(jnp.abs(v2).max()):.6e} (should be nonzero)")
    print(f"  Output u zonal mean max: {float(jnp.abs(u2.mean(axis=-1)).max()):.6e}")
    print(f"  Output v zonal mean max: {float(jnp.abs(v2.mean(axis=-1)).max()):.6e}")


def test_full_tendency_pathway():
    """
    Test the full pathway:
    u_flux (grid) -> spectral vort/div tendency -> grid u,v

    For a purely zonal u_flux with v_flux=0, the resulting v should
    have a specific profile. We compare with SHTNS via the Fortran dycore.
    """
    print(f"\n{'='*80}")
    print("TEST 3: Full tendency pathway comparison with Fortran")
    print("=" * 80)

    from datetime import timedelta
    import climt
    from sympl import set_constant
    from gfs_dynamical_core import GFSDynamicalCore
    from gfs_dynamical_core.component_jax import GFSDynamicsJAX

    set_constant("reference_air_pressure", value=1e5, units="Pa")
    grid = climt.get_grid(nx=N_LON, ny=N_LAT, nz=20)
    timestep = timedelta(minutes=5)
    dcmip = climt.DcmipInitialConditions(add_perturbation=False)

    # Initialize JAX dycore and get configs
    dycore_j = GFSDynamicsJAX()
    state_j = climt.get_default_state([dycore_j], grid_state=grid)
    state_j.update(dcmip(state_j))
    _, _ = dycore_j(state_j, timestep=timestep)  # initialize configs

    trans_config = dycore_j.trans_config

    # Build initial spectral state
    from gfs_dynamical_core.jax.states import GridState
    from gfs_dynamical_core.jax.transforms import grid_to_spectral, spectral_to_grid

    u0 = jnp.array(state_j["eastward_wind"])
    v0 = jnp.array(state_j["northward_wind"])
    temp0 = jnp.array(state_j["air_temperature"])
    ps0 = jnp.array(state_j["surface_air_pressure"])
    q0 = jnp.array(state_j["specific_humidity"])

    grid_orig = GridState(
        u=u0, v=v0, temperature=temp0,
        vorticity=jnp.zeros_like(u0),
        divergence=jnp.zeros_like(u0),
        log_surface_pressure=jnp.log(ps0),
        tracers=jnp.stack([q0], axis=0),
    )
    spec0 = grid_to_spectral(grid_orig, trans_config)

    # Check: the initial spectral vorticity should be purely m=0 (zonal flow)
    vort_spec = np.array(spec0.vorticity)
    m0_energy = np.sum(np.abs(vort_spec[:, :, L-1])**2)
    total_energy = np.sum(np.abs(vort_spec)**2)
    print(f"  Initial vorticity: m=0 energy fraction = {m0_energy/total_energy:.10f}")
    print(f"  Initial divergence max: {np.abs(np.array(spec0.divergence)).max():.6e}")

    # Now reconstruct grid from spectral
    grid0, grads0 = spectral_to_grid(spec0, trans_config)
    print(f"\n  After roundtrip: u max = {float(jnp.abs(grid0.u).max()):.6e}")
    print(f"  After roundtrip: v max = {float(jnp.abs(grid0.v).max()):.6e}")
    print(f"  (v should be ~0 for zonally symmetric flow)")

    # Compute the vorticity tendency from u_flux
    # For the initial state, u_flux = u*(vort+f) - pgf_y + vadv_v
    # This is dominated by u*(vort+f) - pgf_y
    # Let's just look at what spectral tendency gets produced
    from gfs_dynamical_core.jax.dynamics import get_spectral_tendencies

    spec_tends = get_spectral_tendencies(
        spec0, dycore_j._phis_grads, dycore_j.dyn_config,
        trans_config, dycore_j._latitudes,
    )

    # Check tendency spectra
    vort_tend = np.array(spec_tends.d_vorticity_d_t)
    div_tend = np.array(spec_tends.d_divergence_d_t)

    print(f"\n  Vorticity tendency:")
    print(f"    m=0 energy: {np.sum(np.abs(vort_tend[:, :, L-1])**2):.6e}")
    print(f"    m!=0 energy: {np.sum(np.abs(vort_tend)**2) - np.sum(np.abs(vort_tend[:, :, L-1])**2):.6e}")

    print(f"\n  Divergence tendency:")
    print(f"    m=0 energy: {np.sum(np.abs(div_tend[:, :, L-1])**2):.6e}")
    print(f"    m!=0 energy: {np.sum(np.abs(div_tend)**2) - np.sum(np.abs(div_tend[:, :, L-1])**2):.6e}")

    # The key question: for m=0, are the vort/div tendencies purely real?
    # (They should be for a zonally symmetric state)
    print(f"\n  m=0 tendencies (should be purely real for zonal flow):")
    for name, arr in [("vort_tend", vort_tend), ("div_tend", div_tend)]:
        m0 = arr[:, :, L-1]  # m=0 column
        print(f"    {name} m=0: max|real|={np.abs(m0.real).max():.6e}  max|imag|={np.abs(m0.imag).max():.6e}")


if __name__ == "__main__":
    test_vector_tendency_roundtrip()
    test_m0_vortdiv_to_uv()
    test_full_tendency_pathway()
