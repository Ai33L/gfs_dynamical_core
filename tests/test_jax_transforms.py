import jax
from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from gfs_dynamical_core.jax.dynamics import GridTendencies
from gfs_dynamical_core.jax.states import SpectralState
from gfs_dynamical_core.jax.transforms import (
    TransformConfig,
    grid_to_spectral_tendencies,
    spectral_to_grid,
)


def test_spectral_gradients():
    L = 8
    config = TransformConfig(L=L, radius=1.0)

    # Create a state where log_surface_pressure = Y_1,0 \propto \cos\theta
    spec_state = SpectralState(
        vorticity=jnp.zeros((1, L, 2 * L - 1), dtype=jnp.complex128),
        divergence=jnp.zeros((1, L, 2 * L - 1), dtype=jnp.complex128),
        temperature=jnp.zeros((1, L, 2 * L - 1), dtype=jnp.complex128),
        log_surface_pressure=jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
        .at[1, L - 1]
        .set(1.0),
        tracers=jnp.zeros((1, 1, L, 2 * L - 1), dtype=jnp.complex128),
    )

    grid_state, grid_grads = spectral_to_grid(spec_state, config)

    # Analytical gradient of Y_10 is purely in the theta direction.
    # v_theta = -sin(theta), v_phi = 0.
    # grad_x (lambda) = v_phi = 0
    # grad_y (phi_lat) = -v_theta = sin(theta)
    # So grad_x should be 0.
    np.testing.assert_allclose(grid_grads.d_log_ps_d_lambda, 0.0, atol=1e-10)

    # At equator (theta = pi/2), grad_y should be max.
    # We don't have theta array here easily, but we know it's not zero.
    assert np.max(np.abs(grid_grads.d_log_ps_d_phi)) > 0.1


def test_vector_transforms():
    L = 8
    config = TransformConfig(L=L, radius=1.0)
    n_lat, n_lon = config.n_lat, config.n_lon

    # Divergence = Y_1,0
    # Vorticity = Y_2,1
    div_lm = jnp.zeros((1, L, 2 * L - 1), dtype=jnp.complex128).at[0, 1, L - 1].set(1.0)
    vort_lm = jnp.zeros((1, L, 2 * L - 1), dtype=jnp.complex128).at[0, 2, L].set(1.0)

    spec_state = SpectralState(
        vorticity=vort_lm,
        divergence=div_lm,
        temperature=jnp.zeros((1, L, 2 * L - 1), dtype=jnp.complex128),
        log_surface_pressure=jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128),
        tracers=jnp.zeros((1, 1, L, 2 * L - 1), dtype=jnp.complex128),
    )

    grid_state, _ = spectral_to_grid(spec_state, config)

    # Round-trip: grid -> spectral should recover the original coefficients
    from gfs_dynamical_core.jax.dynamics import GridTendencies

    grid_tends = GridTendencies(
        u_flux=grid_state.u,
        v_flux=grid_state.v,
        temp_tend=grid_state.temperature,
        log_ps_tend=grid_state.log_surface_pressure,
        tracer_tends=grid_state.tracers,
        kinetic_energy=jnp.zeros_like(grid_state.temperature),
    )

    spec_tends = grid_to_spectral_tendencies(grid_tends, config)

    print("max div err:", np.max(np.abs(spec_tends.d_divergence_d_t - div_lm)))
    print("max vort err:", np.max(np.abs(spec_tends.d_vorticity_d_t - vort_lm)))
    np.testing.assert_allclose(spec_tends.d_divergence_d_t, div_lm, atol=1e-10)
    np.testing.assert_allclose(spec_tends.d_vorticity_d_t, vort_lm, atol=1e-10)
