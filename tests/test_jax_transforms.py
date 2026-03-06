from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
import s2fft
import s2fft.utils.signal_generator as sg
from gfs_dynamical_core.jax.states import SpectralState, GridState
from gfs_dynamical_core.jax.transforms import TransformConfig, spectral_to_grid, grid_to_spectral_tendencies

def test_spectral_identity():
    # Use small L for testing
    L = 16
    n_lat = L
    n_lon = 2 * L - 1
    n_lev = 5
    config = TransformConfig(L=L, n_lat=n_lat, n_lon=n_lon, sampling="gl")
    
    # Generate random spectral temperature for multiple levels
    rng = np.random.default_rng(0)
    # sg.generate_flm doesn't support batching directly, so we vmap or loop
    temp_spec = jnp.stack([jnp.array(sg.generate_flm(rng, L, reality=False)) for _ in range(n_lev)])
    
    # Empty others
    vort = jnp.zeros_like(temp_spec)
    div = jnp.zeros_like(temp_spec)
    lnps = jnp.zeros((L, 2*L-1), dtype=jnp.complex128)
    tracers = jnp.zeros((1, n_lev, L, 2*L-1), dtype=jnp.complex128)
    
    spec_state = SpectralState(
        vorticity=vort,
        divergence=div,
        temperature=temp_spec,
        log_surface_pressure=lnps,
        tracers=tracers
    )
    
    # Forward: Spectral -> Grid
    grid_state, _ = spectral_to_grid(spec_state, config)
    
    # Verify grid shape
    assert grid_state.temperature.shape == (n_lev, n_lat, n_lon)
    
    # Backward: Grid -> Spectral
    # We need a mock GridTendencies object or just use the field directly if we had a pure scalar transform
    # Since grid_to_spectral_tendencies expects a GridTendencies object, we'll mock it.
    from gfs_dynamical_core.jax.dynamics import GridTendencies
    
    grid_tends = GridTendencies(
        u_flux=grid_state.u,
        v_flux=grid_state.v,
        temp_tend=grid_state.temperature,
        log_ps_tend=grid_state.log_surface_pressure,
        tracer_tends=grid_state.tracers,
        kinetic_energy=jnp.zeros_like(grid_state.u)
    )
    
    recovered_spec = grid_to_spectral_tendencies(grid_tends, config)
    
    # Verify temperature round-trip
    np.testing.assert_allclose(
        recovered_spec.d_temperature_d_t, 
        spec_state.temperature, 
        atol=1e-12, rtol=1e-12
    )
