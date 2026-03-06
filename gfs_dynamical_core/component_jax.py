import jax
import jax.numpy as jnp
import s2fft
from sympl import Stepper, get_constant

from .jax.dynamics import DynamicsConfig
from .jax.states import GridState
from .jax.stepper import StepperConfig, advance
from .jax.transforms import (
    TransformConfig,
    get_gaussian_latitudes,
    grid_to_spectral,
    spectral_to_grid,
)


class GFSDynamicsJAX(Stepper):
    """
    JAX-based implementation of the GFS dynamical core.
    """

    input_properties = {
        "air_temperature": {"units": "K", "dims": ["mid_levels", "lat", "lon"]},
        "eastward_wind": {"units": "m s^-1", "dims": ["mid_levels", "lat", "lon"]},
        "northward_wind": {"units": "m s^-1", "dims": ["mid_levels", "lat", "lon"]},
        "surface_air_pressure": {"units": "Pa", "dims": ["lat", "lon"]},
        "specific_humidity": {
            "units": "kg kg^-1",
            "dims": ["mid_levels", "lat", "lon"],
        },
        "surface_geopotential": {"units": "m^2 s^-2", "dims": ["lat", "lon"]},
        "atmosphere_hybrid_sigma_pressure_a_coordinate_on_interface_levels": {
            "units": "dimensionless",
            "dims": ["interface_levels"],
            "alias": "a_coord",
        },
        "atmosphere_hybrid_sigma_pressure_b_coordinate_on_interface_levels": {
            "units": "dimensionless",
            "dims": ["interface_levels"],
            "alias": "b_coord",
        },
    }

    output_properties = {
        "air_temperature": {"units": "K", "dims": ["mid_levels", "lat", "lon"]},
        "eastward_wind": {"units": "m s^-1", "dims": ["mid_levels", "lat", "lon"]},
        "northward_wind": {"units": "m s^-1", "dims": ["mid_levels", "lat", "lon"]},
        "surface_air_pressure": {"units": "Pa", "dims": ["lat", "lon"]},
        "specific_humidity": {
            "units": "kg kg^-1",
            "dims": ["mid_levels", "lat", "lon"],
        },
    }

    diagnostic_properties = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dyn_config = None
        self.trans_config = None
        self.stepper_config = None
        self._phis_grads = None
        self._latitudes = None

        self._jit_advance = jax.jit(advance)

    def array_call(self, state, timestep):
        u = jnp.array(state["eastward_wind"])
        v = jnp.array(state["northward_wind"])
        temp = jnp.array(state["air_temperature"])
        ps = jnp.array(state["surface_air_pressure"])
        q = jnp.array(state["specific_humidity"])
        phis = jnp.array(state["surface_geopotential"])

        ak_jnp = jnp.array(state["a_coord"])
        bk_jnp = jnp.array(state["b_coord"])

        n_lev, n_lat, n_lon = temp.shape
        dt = timestep.total_seconds()

        if self.dyn_config is None:
            dbk = bk_jnp[1:] - bk_jnp[:-1]
            ck = ak_jnp[1:] * bk_jnp[:-1] - ak_jnp[:-1] * bk_jnp[1:]

            self.dyn_config = DynamicsConfig(
                ak=ak_jnp,
                bk=bk_jnp,
                ck=ck,
                dbk=dbk,
                rk=get_constant("gas_constant_of_dry_air", "J kg^-1 K^-1")
                / get_constant(
                    "heat_capacity_of_dry_air_at_constant_pressure", "J kg^-1 K^-1"
                ),
                toa_pressure=0.0,
                radius=get_constant("planetary_radius", "m"),
                omega=get_constant("planetary_rotation_rate", "s^-1"),
                g=get_constant("gravitational_acceleration", "m s^-2"),
                rd=get_constant("gas_constant_of_dry_air", "J kg^-1 K^-1"),
                rv=get_constant("gas_constant_of_vapor_phase", "J kg^-1 K^-1"),
                cp=get_constant(
                    "heat_capacity_of_dry_air_at_constant_pressure", "J kg^-1 K^-1"
                ),
                cvap=get_constant("heat_capacity_of_vapor_phase", "J kg^-1 K^-1"),
            )

        if self.trans_config is None:
            # For GL sampling: n_lat = L, n_lon = 2*L - 1.
            # climt provides the grid we asked for, so arrays already
            # arrive at the native s2fft size — no resampling needed.
            L = n_lat
            self.trans_config = TransformConfig(
                L=L, sampling="gl", radius=self.dyn_config.radius
            )

        if self.stepper_config is None:
            self.stepper_config = StepperConfig(dt=dt, explicit=True)

        L = self.trans_config.L

        # Build grid state directly — arrays are already at native s2fft size
        grid_orig = GridState(
            u=u,
            v=v,
            temperature=temp,
            vorticity=jnp.zeros_like(u),
            divergence=jnp.zeros_like(u),
            log_surface_pressure=jnp.log(ps),
            tracers=jnp.stack([q], axis=0),
        )

        spec_orig = grid_to_spectral(grid_orig, self.trans_config)

        # Compute phis gradients on the native grid (once, since topography is static)
        if self._phis_grads is None:
            sampling = self.trans_config.sampling
            radius = self.dyn_config.radius
            phis_lm = s2fft.forward_jax(phis, L, sampling=sampling)
            l_arr = jnp.arange(L)
            l_factor = jnp.sqrt(l_arr * (l_arr + 1))
            F1_phis_lm = -l_factor[:, None] * phis_lm
            f_phis_spin1 = s2fft.inverse_jax(F1_phis_lm, L, spin=1, sampling=sampling)
            dphisdx = f_phis_spin1.imag / radius
            dphisdy = -f_phis_spin1.real / radius
            self._phis_grads = (dphisdx, dphisdy)

        # Gaussian quadrature latitudes matching s2fft GL sampling
        if self._latitudes is None:
            self._latitudes = get_gaussian_latitudes(L)

        # Advance one timestep
        spec_final = self._jit_advance(
            spec_orig,
            self._phis_grads,
            self.dyn_config,
            self.trans_config,
            self.stepper_config,
            self._latitudes,
        )

        grid_final, _ = spectral_to_grid(spec_final, self.trans_config)

        def to_numpy(arr):
            if jnp.iscomplexobj(arr):
                arr = arr.real
            return jax.device_get(arr)

        return {}, {
            "air_temperature": to_numpy(grid_final.temperature),
            "eastward_wind": to_numpy(grid_final.u),
            "northward_wind": to_numpy(grid_final.v),
            "surface_air_pressure": to_numpy(jnp.exp(grid_final.log_surface_pressure)),
            "specific_humidity": to_numpy(grid_final.tracers[0]),
        }
