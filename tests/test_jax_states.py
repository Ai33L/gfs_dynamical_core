import jax.numpy as jnp
from gfs_dynamical_core.jax.states import SpectralState, GridState

def test_spectral_state():
    state = SpectralState(
        vorticity=jnp.zeros((10, 16, 31), dtype=jnp.complex128),
        divergence=jnp.zeros((10, 16, 31), dtype=jnp.complex128),
        temperature=jnp.zeros((10, 16, 31), dtype=jnp.complex128),
        log_surface_pressure=jnp.zeros((16, 31), dtype=jnp.complex128),
        tracers=jnp.zeros((2, 10, 16, 31), dtype=jnp.complex128)
    )
    assert state.vorticity.shape == (10, 16, 31)

def test_grid_state():
    state = GridState(
        u=jnp.zeros((10, 32, 64)),
        v=jnp.zeros((10, 32, 64)),
        temperature=jnp.zeros((10, 32, 64)),
        vorticity=jnp.zeros((10, 32, 64)),
        divergence=jnp.zeros((10, 32, 64)),
        log_surface_pressure=jnp.zeros((32, 64)),
        tracers=jnp.zeros((2, 10, 32, 64))
    )
    assert state.u.shape == (10, 32, 64)
