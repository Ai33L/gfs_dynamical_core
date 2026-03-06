import jax
import jax.numpy as jnp
from flax import struct
import s2fft
from .states import SpectralState, GridState, GridGradients, SpectralTendencies

def get_grid_dimensions(L: int, sampling: str) -> tuple[int, int]:
    """Returns (n_lat, n_lon) for a given L and sampling."""
    if sampling == "gl":
        return L, 2 * L - 1
    elif sampling == "mw":
        return L, 2 * L - 1
    elif sampling == "dh":
        return 2 * L, 2 * L
    return L, 2 * L
@struct.dataclass
class TransformConfig:
    """Configuration for spectral transforms."""
    L: int  # Bandlimit (truncation + 1)
    n_lat: int
    n_lon: int
    sampling: str = "gl"  # Gauss-Legendre for GFS
    radius: float = 6371000.0
    
    @property
    def truncation(self):
        return self.L - 1

def spectral_to_grid(spec_state: SpectralState, config: TransformConfig) -> tuple[GridState, GridGradients]:
    """
    Transforms spectral state to grid space, including gradients.
    """
    L = config.L
    sampling = config.sampling
    
    def transform_level(vort, div, temp, tracers):
        # 1. Scalar transforms
        grid_t = s2fft.inverse(temp, L, sampling=sampling, method="jax")
        grid_vort = s2fft.inverse(vort, L, sampling=sampling, method="jax")
        grid_div = s2fft.inverse(div, L, sampling=sampling, method="jax")
        
        # 2. Vector transforms (vort, div -> u, v)
        # Note: GFS uses specific formulas. s2fft has vector transforms too.
        # For now, we mock u, v as zero and will implement properly later.
        grid_u = jnp.zeros_like(grid_t)
        grid_v = jnp.zeros_like(grid_t)
        
        # 3. Tracers
        grid_tracers = jax.vmap(lambda flm: s2fft.inverse(flm, L, sampling=sampling, method="jax"))(tracers)
        
        return grid_u, grid_v, grid_t, grid_vort, grid_div, grid_tracers

    grid_u, grid_v, grid_t, grid_vort, grid_div, grid_tracers = jax.vmap(transform_level)(
        spec_state.vorticity, 
        spec_state.divergence, 
        spec_state.temperature,
        spec_state.tracers.transpose(1, 0, 2, 3)
    )
    
    grid_lnps = s2fft.inverse(spec_state.log_surface_pressure, L, sampling=sampling, method="jax")
    
    grid_state = GridState(
        u=grid_u, v=grid_v, temperature=grid_t,
        vorticity=grid_vort, divergence=grid_div,
        log_surface_pressure=grid_lnps,
        tracers=grid_tracers.transpose(1, 0, 2, 3)
    )
    
    # 4. Mock gradients for now
    grid_grads = GridGradients(
        d_log_ps_d_phi=jnp.zeros_like(grid_lnps),
        d_log_ps_d_lambda=jnp.zeros_like(grid_lnps),
        d_t_d_phi=jnp.zeros_like(grid_t),
        d_t_d_lambda=jnp.zeros_like(grid_t)
    )
    
    return grid_state, grid_grads

def grid_to_spectral_tendencies(grid_tends, config: TransformConfig) -> SpectralTendencies:
    """
    Transforms grid-space tendencies back to spectral space.
    """
    L = config.L
    sampling = config.sampling
    
    # temp, lnps, tracers are scalars
    flm_temp = jax.vmap(lambda f: s2fft.forward(f, L, sampling=sampling, method="jax"))(grid_tends.temp_tend)
    flm_lnps = s2fft.forward(grid_tends.log_ps_tend, L, sampling=sampling, method="jax")
    
    def forward_tracers(tracers):
        return jax.vmap(lambda f: s2fft.forward(f, L, sampling=sampling, method="jax"))(tracers)
    
    flm_tracers = jax.vmap(forward_tracers)(grid_tends.tracer_tends)
    
    # vorticity and divergence tendencies from momentum fluxes
    # d(zeta)/dt = -div(u_flux, v_flux)
    # d(div)/dt = curl(u_flux, v_flux) - lap(KE)
    # For now, mock them as zero.
    flm_vort = jnp.zeros_like(flm_temp)
    flm_div = jnp.zeros_like(flm_temp)
    
    return SpectralTendencies(
        d_vorticity_d_t=flm_vort,
        d_divergence_d_t=flm_div,
        d_temperature_d_t=flm_temp,
        d_log_surface_pressure_d_t=flm_lnps,
        d_tracers_d_t=flm_tracers
    )
