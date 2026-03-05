import jax
import jax.numpy as jnp
import numpy as np
import s2fft
import s2fft.utils.signal_generator as sg
from gfs_dynamical_core.jax.states import SpectralState
from gfs_dynamical_core.jax.transforms import TransformConfig, spectral_to_grid, grid_to_spectral

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
    temp_spec = jnp.stack([jnp.array(sg.generate_flm(rng, L)) for _ in range(n_lev)])
    
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
    recovered_spec = grid_to_spectral(grid_state, config)
    
    # Verify temperature round-trip
    np.testing.assert_allclose(
        recovered_spec.temperature, 
        spec_state.temperature, 
        atol=1e-4, rtol=1e-4
    )
