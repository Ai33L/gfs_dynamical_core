import jax
import jax.numpy as jnp
from flax import struct
import s2fft
from .states import SpectralState, GridState, GridGradients

@struct.dataclass
class TransformConfig:
    """Configuration for spectral transforms."""
    L: int  # Bandlimit (truncation + 1)
    n_lat: int
    n_lon: int
    sampling: str = "gl"  # Gauss-Legendre for GFS
    
    @property
    def truncation(self):
        return self.L - 1

def spectral_to_grid(spec_state: SpectralState, config: TransformConfig) -> tuple[GridState, GridGradients]:
    """
    Transforms spectral state to grid space, including gradients.
    """
    L = config.L
    sampling = config.sampling
    
    # spec_state coefficients are assumed to be in s2fft format (L, 2L-1)
    # If they are in GFS packed format, a conversion is needed.
    # For now, we assume the inputs to this function are already in s2fft compatible format.
    
    def transform_level(vort, div, temp, tracers):
        # 1. Scalar transforms
        grid_t = s2fft.inverse(temp, L, sampling=sampling)
        
        # 2. Vector transforms (vort, div -> u, v)
        # Note: s2fft handles this via spin-1 transforms
        # For now, use placeholder logic
        grid_u = jnp.zeros_like(grid_t)
        grid_v = jnp.zeros_like(grid_t)
        
        # 3. Tracers
        grid_tracers = jax.vmap(lambda flm: s2fft.inverse(flm, L, sampling=sampling))(tracers)
        
        return grid_u, grid_v, grid_t, grid_tracers

    # Vmap over levels
    grid_u, grid_v, grid_t, grid_tracers = jax.vmap(transform_level)(
        spec_state.vorticity, 
        spec_state.divergence, 
        spec_state.temperature,
        spec_state.tracers.transpose(1, 0, 2, 3) # (levels, tracers, L, 2L-1)
    )
    
    # 4. Surface pressure
    grid_lnps = s2fft.inverse(spec_state.log_surface_pressure, L, sampling=sampling)
    
    grid_state = GridState(
        u=grid_u,
        v=grid_v,
        temperature=grid_t,
        log_surface_pressure=grid_lnps,
        tracers=grid_tracers.transpose(1, 0, 2, 3) # (tracers, levels, n_lat, n_lon)
    )
    
    # 5. Gradients (can be computed in spectral space then transformed)
    # Placeholder for gradients
    grid_grads = GridGradients(
        d_log_ps_d_phi=jnp.zeros_like(grid_lnps),
        d_log_ps_d_lambda=jnp.zeros_like(grid_lnps),
        d_t_d_phi=jnp.zeros_like(grid_t),
        d_t_d_lambda=jnp.zeros_like(grid_t)
    )
    
    return grid_state, grid_grads

def grid_to_spectral(grid_tendencies: GridState, config: TransformConfig) -> SpectralState:
    """
    Transforms grid-space tendencies back to spectral space.
    """
    L = config.L
    sampling = config.sampling
    
    def forward_level(u, v, t, tracers):
        # Forward scalar
        flm_t = s2fft.forward(t, L, sampling=sampling)
        
        # Forward vector (u, v -> vort, div)
        flm_vort = jnp.zeros_like(flm_t)
        flm_div = jnp.zeros_like(flm_t)
        
        # Tracers
        flm_tracers = jax.vmap(lambda f: s2fft.forward(f, L, sampling=sampling))(tracers)
        
        return flm_vort, flm_div, flm_t, flm_tracers

    flm_vort, flm_div, flm_t, flm_tracers = jax.vmap(forward_level)(
        grid_tendencies.u,
        grid_tendencies.v,
        grid_tendencies.temperature,
        grid_tendencies.tracers.transpose(1, 0, 2, 3)
    )
    
    flm_lnps = s2fft.forward(grid_tendencies.log_surface_pressure, L, sampling=sampling)
    
    spec_tend = SpectralState(
        vorticity=flm_vort,
        divergence=flm_div,
        temperature=flm_t,
        log_surface_pressure=flm_lnps,
        tracers=flm_tracers.transpose(1, 0, 2, 3)
    )
    
    return spec_tend
