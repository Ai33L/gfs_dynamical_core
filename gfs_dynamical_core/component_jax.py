from sympl import Stepper, get_constant
import jax
import jax.numpy as jnp
from .jax.dynamics import DynamicsConfig, full_dynamics_step
from .jax.transforms import TransformConfig, spectral_to_grid, grid_to_spectral
from .jax.states import GridState, GridGradients, SpectralState

class GFSDynamicsJAX(Stepper):
    """
    JAX-based implementation of the GFS dynamical core.
    """
    
    input_properties = {
        'air_temperature': {'units': 'K', 'dims': ['mid_levels', 'lat', 'lon']},
        'eastward_wind': {'units': 'm s^-1', 'dims': ['mid_levels', 'lat', 'lon']},
        'northward_wind': {'units': 'm s^-1', 'dims': ['mid_levels', 'lat', 'lon']},
        'surface_air_pressure': {'units': 'Pa', 'dims': ['lat', 'lon']},
        'specific_humidity': {'units': 'kg kg^-1', 'dims': ['mid_levels', 'lat', 'lon']},
        'surface_geopotential': {'units': 'm^2 s^-2', 'dims': ['lat', 'lon']},
        'atmosphere_hybrid_sigma_pressure_a_coordinate_on_interface_levels': {
            'units': 'dimensionless', 'dims': ['interface_levels'], 'alias': 'a_coord'
        },
        'atmosphere_hybrid_sigma_pressure_b_coordinate_on_interface_levels': {
            'units': 'dimensionless', 'dims': ['interface_levels'], 'alias': 'b_coord'
        },
    }
    
    output_properties = {
        'air_temperature': {'units': 'K', 'dims': ['mid_levels', 'lat', 'lon']},
        'eastward_wind': {'units': 'm s^-1', 'dims': ['mid_levels', 'lat', 'lon']},
        'northward_wind': {'units': 'm s^-1', 'dims': ['mid_levels', 'lat', 'lon']},
        'surface_air_pressure': {'units': 'Pa', 'dims': ['lat', 'lon']},
        'specific_humidity': {'units': 'kg kg^-1', 'dims': ['mid_levels', 'lat', 'lon']},
    }
    
    diagnostic_properties = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dyn_config = None
        self.trans_config = None
        self._jit_full_step = jax.jit(full_dynamics_step)

    def array_call(self, state, timestep):
        u = jnp.array(state['eastward_wind'])
        v = jnp.array(state['northward_wind'])
        temp = jnp.array(state['air_temperature'])
        ps = jnp.array(state['surface_air_pressure'])
        q = jnp.array(state['specific_humidity'])
        phis = jnp.array(state['surface_geopotential'])
        
        ak_jnp = jnp.array(state['a_coord'])
        bk_jnp = jnp.array(state['b_coord'])
        
        n_lev, n_lat, n_lon = temp.shape
        
        if self.dyn_config is None:
            dbk = bk_jnp[1:] - bk_jnp[:-1]
            ck = ak_jnp[1:] * bk_jnp[:-1] - ak_jnp[:-1] * bk_jnp[1:]
            
            self.dyn_config = DynamicsConfig(
                ak=ak_jnp,
                bk=bk_jnp,
                ck=ck,
                dbk=dbk,
                rk=get_constant('gas_constant_of_dry_air', 'J kg^-1 K^-1') / get_constant('heat_capacity_of_dry_air_at_constant_pressure', 'J kg^-1 K^-1'),
                toa_pressure=0.0,
                radius=get_constant('planetary_radius', 'm'),
                omega=get_constant('planetary_rotation_rate', 's^-1'),
                g=get_constant('gravitational_acceleration', 'm s^-2'),
                rd=get_constant('gas_constant_of_dry_air', 'J kg^-1 K^-1'),
                rv=get_constant('gas_constant_of_vapor_phase', 'J kg^-1 K^-1'),
                cp=get_constant('heat_capacity_of_dry_air_at_constant_pressure', 'J kg^-1 K^-1'),
                cvap=get_constant('heat_capacity_of_vapor_phase', 'J kg^-1 K^-1'),
            )
        
        if self.trans_config is None:
            L = n_lat // 2 
            self.trans_config = TransformConfig(L=L, n_lat=n_lat, n_lon=n_lon, sampling="gl")

        grid_state = GridState(
            u=u, v=v, temperature=temp,
            vorticity=jnp.zeros_like(u),
            divergence=jnp.zeros_like(u),
            log_surface_pressure=jnp.log(ps),
            tracers=jnp.expand_dims(q, 0)
        )
        
        grid_grads = GridGradients(
            d_log_ps_d_phi=jnp.zeros((n_lat, n_lon)),
            d_log_ps_d_lambda=jnp.zeros((n_lat, n_lon)),
            d_t_d_phi=jnp.zeros((n_lev, n_lat, n_lon)),
            d_t_d_lambda=jnp.zeros((n_lev, n_lat, n_lon))
        )
        phis_grads = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))
        latitudes = jnp.linspace(-jnp.pi/2, jnp.pi/2, n_lat)
        
        grid_tends = self._jit_full_step(
            grid_state, grid_grads, phis_grads, self.dyn_config, latitudes
        )
        
        # Apply timestep (Forward Euler) to get new state
        dt = timestep.total_seconds()
        
        new_temp = temp + grid_tends.temp_tend * dt
        # Simplified momentum update for now
        new_u = u + grid_tends.u_flux * dt 
        new_v = v + grid_tends.v_flux * dt
        
        lnps_new = jnp.log(ps) + grid_tends.log_ps_tend * dt
        new_ps = jnp.exp(lnps_new)
        
        new_q = q + grid_tends.tracer_tends[0] * dt

        def from_jax(arr):
            return jax.device_get(arr)

        return {}, {
            'air_temperature': from_jax(new_temp),
            'eastward_wind': from_jax(new_u),
            'northward_wind': from_jax(new_v),
            'surface_air_pressure': from_jax(new_ps),
            'specific_humidity': from_jax(new_q),
        }
