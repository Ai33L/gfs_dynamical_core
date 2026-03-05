from flax import struct
import jax.numpy as jnp

@struct.dataclass
class SpectralState:
    """State of the dynamical core in spectral space."""
    vorticity: jnp.ndarray  # (levels, n_spec)
    divergence: jnp.ndarray # (levels, n_spec)
    temperature: jnp.ndarray # (levels, n_spec)
    log_surface_pressure: jnp.ndarray # (n_spec,)
    tracers: jnp.ndarray # (n_tracers, levels, n_spec)

@struct.dataclass
class GridState:
    """State of the dynamical core in grid space."""
    u: jnp.ndarray # (levels, n_lat, n_lon)
    v: jnp.ndarray # (levels, n_lat, n_lon)
    temperature: jnp.ndarray # (levels, n_lat, n_lon)
    log_surface_pressure: jnp.ndarray # (n_lat, n_lon)
    tracers: jnp.ndarray # (n_tracers, levels, n_lat, n_lon)

@struct.dataclass
class GridGradients:
    """Spatial gradients in grid space."""
    d_log_ps_d_phi: jnp.ndarray # (n_lat, n_lon)
    d_log_ps_d_lambda: jnp.ndarray # (n_lat, n_lon)
    d_t_d_phi: jnp.ndarray # (levels, n_lat, n_lon)
    d_t_d_lambda: jnp.ndarray # (levels, n_lat, n_lon)
