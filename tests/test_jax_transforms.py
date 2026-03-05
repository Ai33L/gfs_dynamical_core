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
    n_lat = 2 * L # Approximate for GL
    n_lon = 2 * L
    config = TransformConfig(L=L, n_lat=n_lat, n_lon=n_lon, sampling="gl")
    
    # Generate random spectral temperature
    rng = np.random.default_rng(0)
    temp_spec = sg.generate_flm(rng, L)
    # Ensure it's for 1 level
    temp_spec = jnp.expand_dims(jnp.array(temp_spec), 0) # (1, L, 2L-1)
    
    # Empty others
    vort = jnp.zeros_like(temp_spec)
    div = jnp.zeros_like(temp_spec)
    lnps = jnp.zeros((L, 2*L-1), dtype=jnp.complex128)
    tracers = jnp.zeros((1, 1, L, 2*L-1), dtype=jnp.complex128)
    
    spec_state = SpectralState(
        vorticity=vort,
        divergence=div,
        temperature=temp_spec,
        log_surface_pressure=lnps,
        tracers=tracers
    )
    
    # Forward: Spectral -> Grid
    grid_state, _ = spectral_to_grid(spec_state, config)
    
    # Backward: Grid -> Spectral
    recovered_spec = grid_to_spectral(grid_state, config)
    
    # Verify temperature round-trip
    np.testing.assert_allclose(
        recovered_spec.temperature[0], 
        spec_state.temperature[0], 
        atol=1e-5, rtol=1e-5
    )
