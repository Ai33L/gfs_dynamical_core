import jax.numpy as jnp
from gfs_dynamical_core.jax.states import SpectralState, GridState

def test_spectral_state():
    state = SpectralState(
        vorticity=jnp.zeros((10, 100)),
        divergence=jnp.zeros((10, 100)),
        temperature=jnp.zeros((10, 100)),
        log_surface_pressure=jnp.zeros((100,)),
        tracers=jnp.zeros((2, 10, 100))
    )
    assert state.vorticity.shape == (10, 100)

def test_grid_state():
    state = GridState(
        u=jnp.zeros((10, 32, 64)),
        v=jnp.zeros((10, 32, 64)),
        temperature=jnp.zeros((10, 32, 64)),
        log_surface_pressure=jnp.zeros((32, 64)),
        tracers=jnp.zeros((2, 10, 32, 64))
    )
    assert state.u.shape == (10, 32, 64)
