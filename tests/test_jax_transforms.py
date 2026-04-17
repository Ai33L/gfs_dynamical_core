import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from gfs_dynamical_core.jax.states import GridState, SpectralState
from gfs_dynamical_core.jax.transforms import (
    TransformConfig,
    grid_to_spectral,
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
    """Round-trip spectral -> grid -> spectral for vector fields.

    Uses conjugate-symmetric inputs (physically valid real fields) and
    grid_to_spectral (not grid_to_spectral_tendencies, which applies
    curl/div decomposition with sign changes).
    """
    L = 8
    radius = 6371000.0
    config = TransformConfig(L=L, radius=radius)
    T = config.truncation  # modes l <= T survive truncation

    # Divergence = Y_1,0 (m=0, self-conjugate)
    # Vorticity = Y_2,1 + conjugate partner Y_2,-1
    # Conjugate symmetry: f_{l,-m} = (-1)^m * conj(f_{l,m})
    div_lm = jnp.zeros((1, L, 2 * L - 1), dtype=jnp.complex128).at[0, 1, L - 1].set(1.0)
    vort_lm = (
        jnp.zeros((1, L, 2 * L - 1), dtype=jnp.complex128)
        .at[0, 2, L].set(1.0)        # (l=2, m=+1)
        .at[0, 2, L - 2].set(-1.0)   # (l=2, m=-1) = (-1)^1 * conj(1.0)
    )

    spec_state = SpectralState(
        vorticity=vort_lm,
        divergence=div_lm,
        temperature=jnp.zeros((1, L, 2 * L - 1), dtype=jnp.complex128),
        log_surface_pressure=jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128),
        tracers=jnp.zeros((1, 1, L, 2 * L - 1), dtype=jnp.complex128),
    )

    grid_state, _ = spectral_to_grid(spec_state, config)

    # Round-trip via grid_to_spectral (state transform, not tendency transform)
    spec_rt = grid_to_spectral(grid_state, config)

    # grid_to_spectral applies enforce_triangular_truncation with T = config.truncation.
    # Both test modes (l=1 and l=2) are within T=3, so they should survive.
    print("max div err:", np.max(np.abs(spec_rt.divergence - div_lm)))
    print("max vort err:", np.max(np.abs(spec_rt.vorticity - vort_lm)))
    np.testing.assert_allclose(spec_rt.divergence, div_lm, atol=1e-10)
    np.testing.assert_allclose(spec_rt.vorticity, vort_lm, atol=1e-10)
