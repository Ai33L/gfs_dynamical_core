from flax import struct
import jax.numpy as jnp

@struct.dataclass
class SpectralState:
    """State of the dynamical core in spectral space."""
    vorticity: jnp.ndarray  # (levels, L, 2*L-1)
    divergence: jnp.ndarray # (levels, L, 2*L-1)
    temperature: jnp.ndarray # (levels, L, 2*L-1)
    log_surface_pressure: jnp.ndarray # (L, 2*L-1)
    tracers: jnp.ndarray # (n_tracers, levels, L, 2*L-1)

@struct.dataclass
class GridState:
    """State of the dynamical core in grid space."""
    u: jnp.ndarray # (levels, n_lat, n_lon)
    v: jnp.ndarray # (levels, n_lat, n_lon)
    temperature: jnp.ndarray # (levels, n_lat, n_lon)
    vorticity: jnp.ndarray # (levels, n_lat, n_lon)
    divergence: jnp.ndarray # (levels, n_lat, n_lon)
    log_surface_pressure: jnp.ndarray # (n_lat, n_lon)
    tracers: jnp.ndarray # (n_tracers, levels, n_lat, n_lon)

@struct.dataclass
class GridGradients:
    """Spatial gradients in grid space."""
    d_log_ps_d_phi: jnp.ndarray # (n_lat, n_lon)
    d_log_ps_d_lambda: jnp.ndarray # (n_lat, n_lon)
    d_t_d_phi: jnp.ndarray # (levels, n_lat, n_lon)
    d_t_d_lambda: jnp.ndarray # (levels, n_lat, n_lon)
