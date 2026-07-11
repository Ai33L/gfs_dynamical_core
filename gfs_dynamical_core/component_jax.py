import jax
import jax.numpy as jnp
import numpy as np
from sympl import (
    TendencyStepper,
    get_constant,
    get_tracer_names,
    ImplicitTendencyComponentComposite,
    get_numpy_arrays_with_properties,
    restore_data_arrays_with_properties,
)

from ._property_utils import GFSError, get_valid_properties
from .jax.dynamics import DynamicsConfig
from .jax.states import GridState
from .jax.stepper import (
    StepperConfig,
    advance,
    advance_with_tendencies,
    PhysicsTendencies,
    init_diffusion_operators,
    init_semi_implicit_matrices,
)
from .jax.transforms import (
    TransformConfig,
    s2_forward,
    s2_inverse,
    get_gaussian_latitudes,
    grid_to_spectral,
    spectral_to_grid,
)


class GFSDynamicsJAX(TendencyStepper):
    """
    JAX-based implementation of the GFS dynamical core (component-driven).
    """

    uses_tracers = True
    tracer_dims = ("tracer", "mid_levels", "lat", "lon")
    prepend_tracers = (("specific_humidity", "kg/kg"),)

    # base (dynamics-only) property dicts; merged with component props in __init__
    _gfs_input_properties = {
        "air_temperature": {"units": "K", "dims": ["mid_levels", "lat", "lon"]},
        "eastward_wind": {"units": "m s^-1", "dims": ["mid_levels", "lat", "lon"]},
        "northward_wind": {"units": "m s^-1", "dims": ["mid_levels", "lat", "lon"]},
        "surface_air_pressure": {"units": "Pa", "dims": ["lat", "lon"]},
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
        "air_pressure": {"units": "Pa", "dims": ["mid_levels", "lat", "lon"]},
    }

    _gfs_output_properties = {
        "air_temperature": {"units": "K", "dims": ["mid_levels", "lat", "lon"]},
        "eastward_wind": {"units": "m s^-1", "dims": ["mid_levels", "lat", "lon"]},
        "northward_wind": {"units": "m s^-1", "dims": ["mid_levels", "lat", "lon"]},
        "surface_air_pressure": {"units": "Pa", "dims": ["lat", "lon"]},
        "air_pressure": {"units": "Pa", "dims": ["mid_levels", "lat", "lon"]},
        "air_pressure_on_interface_levels": {
            "units": "Pa",
            "dims": ["interface_levels", "lat", "lon"],
        },
    }

    _gfs_diagnostic_properties = {}

    # Override TendencyStepper's abstract/computed properties so these can be
    # set as plain instance attributes in __init__ (mirrors GFSDynamicalCore
    # in component.py). Class-level values default to the base (no extra
    # components) property dicts so that climt.get_default_state([cls], ...)
    # works even when called with the class itself (not yet instantiated) —
    # matches the convention used by climt TendencyComponents such as
    # HeldSuarez, which expose input_properties as a class attribute.
    # __init__ overwrites these with per-instance dicts merged with whatever
    # tendency_component_list was passed in.
    input_properties = _gfs_input_properties
    output_properties = _gfs_output_properties
    diagnostic_properties = _gfs_diagnostic_properties

    @property
    def spectral_names(self):
        return (
            "eastward_wind",
            "northward_wind",
            "air_temperature",
            "surface_air_pressure",
        ) + get_tracer_names()

    def __init__(
        self,
        tendency_component_list=None,
        adiabatic=False,
        zero_negative_moisture=True,
        **kwargs,
    ):
        tendency_component_list = tendency_component_list or []
        self._tendency_component = ImplicitTendencyComponentComposite(
            *tendency_component_list
        )
        bad = set(
            self._tendency_component.diagnostic_properties.keys()
        ).intersection(self.spectral_names)
        if bad:
            raise GFSError(
                "Components may not emit {} as diagnostics; these are stepped "
                "spectrally.".format(bad)
            )

        self.input_properties = dict(self._gfs_input_properties)
        self.output_properties = dict(self._gfs_output_properties)
        self.diagnostic_properties = dict(self._gfs_diagnostic_properties)

        super().__init__(**kwargs)

        self.input_properties.update(
            get_valid_properties(
                self._gfs_input_properties,
                self._tendency_component.input_properties,
                "input",
            )
        )
        self.output_properties.update(
            get_valid_properties(
                self._gfs_output_properties,
                self._tendency_component.tendency_properties,
                "output",
            )
        )
        self.diagnostic_properties.update(
            get_valid_properties(
                self._gfs_diagnostic_properties,
                self._tendency_component.diagnostic_properties,
                "diagnostic",
            )
        )

        self.adiabatic = adiabatic
        self._zero_negative_moisture = zero_negative_moisture
        # ---- existing jnp-pipeline state (unchanged) ----
        self.dyn_config = None
        self.trans_config = None
        self.stepper_config = None
        self._phis_grads = None
        self._latitudes = None
        # Dry-mass fixer state (computed once from the initial state).
        # Not used when adiabatic=True (matches Fortran run.f90 line 330).
        self._gauss_weights = None
        self._pdryini = None

        # JIT-compile the stepper. All config dataclasses are flax structs
        # (static fields marked pytree_node=False), so they trace cleanly.
        # First call compiles (~30 s); subsequent steps run fully fused.
        self._jit_advance = jax.jit(advance)
        # Pure-function variant that also applies a time-split physics
        # increment (grid-space PhysicsTendencies -> spectral) inside the
        # same jit trace. Used whenever self._phys_tendencies is set.
        self._jit_advance_t = jax.jit(advance_with_tendencies)
        # Cached spectral state: only do grid_to_spectral once (for the
        # initial condition).  After that, advance directly in spectral
        # space to avoid repeated grid→spectral→grid round-trip errors.
        self._spec_state = None
        # Optional physics tendencies (numpy/jnp grid arrays), applied
        # time-split after the dynamics step — mirrors Fortran run.f90's
        # getphytend adjustment and the wrapper's assign_tendencies path.
        self._phys_tendencies = None

    def _fvirt(self):
        """(1 - Rd/Rv) / (Rd/Rv) — virtual-temperature moisture coefficient."""
        rd = get_constant("gas_constant_of_dry_air", "J kg^-1 K^-1")
        rv = get_constant("gas_constant_of_vapor_phase", "J kg^-1 K^-1")
        return (1 - rd / rv) / (rd / rv)

    def set_physics_tendencies(self, u_tend=None, v_tend=None, t_tend=None):
        """Set grid-space physics tendencies (e.g. Held-Suarez forcing) to be
        applied as a time-split adjustment after the next dynamics step.

        Arrays must be (n_lev, n_lat, n_lon) bottom-to-top, SI units
        (m s^-2 for winds, K s^-1 for temperature). Call with no arguments
        to clear. Mirrors Fortran: tendencies are computed from the state
        at the START of the step and applied AFTER dynamics as
        ``x += dt * tendency`` (run.f90 physics block).
        """
        if u_tend is None and v_tend is None and t_tend is None:
            self._phys_tendencies = None
        else:
            u_t = jnp.asarray(u_tend)
            v_t = jnp.asarray(v_tend)
            t_t = jnp.asarray(t_tend)
            n_lev, n_lat, n_lon = t_t.shape
            # Number of tracers must match the dynamical core's tracer stack
            # (always >= 1: specific_humidity is prepended). Zero tendency
            # for all of them since this legacy path only forces u/v/T.
            n_tracers = len(self._tracer_packer.tracer_names)
            self._phys_tendencies = PhysicsTendencies(
                u=u_t,
                v=v_t,
                virtual_temperature=t_t,
                log_surface_pressure=jnp.zeros((n_lat, n_lon)),
                tracers=jnp.zeros((n_tracers, n_lev, n_lat, n_lon)),
            )

    def __call__(self, state, timestep):
        """Fortran-style, component-driven step: gather physics tendencies
        from ``self._tendency_component`` and route them through the pure
        ``advance_with_tendencies`` alongside the dynamics step.

        Mirrors ``GFSDynamicalCore.__call__`` in component.py, simplified
        since JAX handles its own internal sub-stepping (no air_pressure
        consistency assertions needed here).
        """
        raw_state = get_numpy_arrays_with_properties(state, self.input_properties)
        raw_state["tracers"] = self._tracer_packer.pack(state)
        raw_state["time"] = state["time"]

        tendencies, diagnostics = self._tendency_component(state, timestep)
        for name, value in tendencies.items():
            if name in self.input_properties:
                tendencies[name] = value.to_units(
                    self.input_properties[name]["units"] + " s^-1"
                )

        raw_diag, raw_new = self.array_call(
            raw_state, timestep, prognostic_tendencies=tendencies
        )

        new_state = self._tracer_packer.unpack(raw_new.pop("tracers"), state)
        new_state.update(
            restore_data_arrays_with_properties(
                raw_new, self.output_properties, state, self.input_properties,
                ignore_missing=True,
            )
        )
        diagnostics.update(
            restore_data_arrays_with_properties(
                raw_diag, self.diagnostic_properties, state, self.input_properties,
                ignore_missing=True,
            )
        )
        for key in state.keys():
            if key not in new_state:
                new_state[key] = state[key]
        return diagnostics, new_state

    def array_call(self, state, timestep, prognostic_tendencies=None):
        prognostic_tendencies = prognostic_tendencies or {}
        u = jnp.array(state["eastward_wind"])
        v = jnp.array(state["northward_wind"])
        temp = jnp.array(state["air_temperature"])
        ps = jnp.array(state["surface_air_pressure"])
        # specific_humidity is always tracer index 0 (prepend_tracers puts it
        # first; see TracerPacker.tracer_names).
        q = jnp.array(state["tracers"][0])
        phis = jnp.array(state["surface_geopotential"])

        ak_jnp = jnp.array(state["a_coord"])
        bk_jnp = jnp.array(state["b_coord"])

        n_lev, n_lat, n_lon = temp.shape
        dt = timestep.total_seconds()

        if self.dyn_config is None:
            # dbk = bk_below - bk_above (to be positive)
            # bk is bottom-up: [1.0 (surf), ..., 0.0 (toa)]
            # dbk[k] = bk_below - bk_above = bk[k] - bk[k+1]
            dbk = bk_jnp[:-1] - bk_jnp[1:]
            # ck[k] = ak_below*bk_above - ak_above*bk_below = ak[k]*bk[k+1] - ak[k+1]*bk[k]
            # This matches the Fortran convention: ck(k) = ak(k+1)*bk(k) - ak(k)*bk(k+1)
            # where Fortran ak/bk are top-to-bottom.  When mapped to bottom-to-top the
            # sign flips, giving ak_btop[k]*bk_btop[k+1] - ak_btop[k+1]*bk_btop[k].
            ck = ak_jnp[:-1] * bk_jnp[1:] - ak_jnp[1:] * bk_jnp[:-1]

            self.dyn_config = DynamicsConfig(
                ak=ak_jnp,
                bk=bk_jnp,
                ck=ck,
                dbk=dbk,
                rk=get_constant("gas_constant_of_dry_air", "J kg^-1 K^-1")
                / get_constant(
                    "heat_capacity_of_dry_air_at_constant_pressure", "J kg^-1 K^-1"
                ),
                toa_pressure=get_constant("top_of_model_pressure", "Pa"),
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
            # Build precompute transform kernels eagerly — kernel
            # construction is not jit-traceable.
            from .jax.transforms import prebuild_kernels

            prebuild_kernels(L, "gl")

        if self.stepper_config is None:
            # Build a temporary default config to read the canonical IMEX
            # coefficients (aa22, aa33, bb4) so the precomputed d_hyb_m
            # matrices stay consistent with whatever StepperConfig uses.
            _default_sc = StepperConfig(dt=dt)
            # Compute semi-implicit matrices (mirrors Fortran init_semimpdata),
            # passing coefficients explicitly to avoid hardcoding them.
            si_matrices = init_semi_implicit_matrices(
                self.dyn_config,
                self.trans_config,
                dt,
                aa22=_default_sc.aa22,
                aa33=_default_sc.aa33,
                bb4=_default_sc.bb4,
            )
            # Compute hyper-diffusion and Rayleigh damping operators
            diff_ops = init_diffusion_operators(self.dyn_config, self.trans_config, dt)
            self.stepper_config = StepperConfig(
                dt=dt,
                explicit=False,
                amhyb=si_matrices["amhyb"],
                bmhyb=si_matrices["bmhyb"],
                tor_hyb=si_matrices["tor_hyb"],
                svhyb=si_matrices["svhyb"],
                d_hyb_m=si_matrices["d_hyb_m"],
                disspec=diff_ops["disspec"],
                diff_prof=diff_ops["diff_prof"],
                dmp_prof=diff_ops["dmp_prof"],
            )

        L = self.trans_config.L

        # On the very first call, convert grid initial conditions to spectral
        # space.  On subsequent calls, reuse the cached spectral state so we
        # never do the lossy grid→spectral round-trip again.
        if self._spec_state is None:
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
        else:
            spec_orig = self._spec_state

        # Compute phis gradients on the native grid (once, since topography is static)
        if self._phis_grads is None:
            sampling = self.trans_config.sampling
            radius = self.dyn_config.radius
            phis_lm = s2_forward(phis, L, sampling)
            l_arr = jnp.arange(L)
            l_factor = jnp.sqrt(l_arr * (l_arr + 1))
            F1_phis_lm = -l_factor[:, None] * phis_lm
            f_phis_spin1 = s2_inverse(F1_phis_lm, L, sampling, spin=1)
            dphisdx = f_phis_spin1.imag / radius
            dphisdy = -f_phis_spin1.real / radius
            self._phis_grads = (dphisdx, dphisdy)

        # Gaussian quadrature latitudes matching s2fft GL sampling
        if self._latitudes is None:
            self._latitudes = get_gaussian_latitudes(L)

        # Gaussian quadrature weights and initial dry surface pressure — only
        # needed by the dry-mass fixer. Skip for adiabatic runs (matches
        # Fortran run.f90 line 330: `if (.not. adiabatic) then`).
        if not self.adiabatic:
            if self._gauss_weights is None:
                _, raw_weights = np.polynomial.legendre.leggauss(L)
                # leggauss returns weights summing to 2; normalise to sum to 1.
                self._gauss_weights = jnp.array(raw_weights / 2.0)

            if self._pdryini is None:
                from .jax.dynamics import compute_pressure_diagnostics

                lnps_grid = jnp.log(ps)
                press_diag_init = compute_pressure_diagnostics(lnps_grid, self.dyn_config)
                q_init = jnp.array(q)  # (n_lev, n_lat, n_lon)
                g = self.dyn_config.g
                pwat_init = (
                    jnp.sum(q_init * press_diag_init.dp, axis=0) / g
                )  # (n_lat, n_lon)
                w = self._gauss_weights[:, None]  # (n_lat, 1)
                pmean_init = float(jnp.sum(w * ps) / n_lon)
                pwat_global_init = float(jnp.sum(w * pwat_init) / n_lon)
                self._pdryini = pmean_init - g * pwat_global_init

        # Build grid-space physics tendencies from component output (bottom-
        # to-top), applying the virtual-temperature correction so temperature
        # and moisture tendencies combine consistently — mirrors the Fortran
        # wrapper's assign_tendencies path.
        if prognostic_tendencies:
            def tend_or_zero(name, shape):
                if name in prognostic_tendencies:
                    arr = (
                        prognostic_tendencies[name]
                        .to_units(self.input_properties[name]["units"] + " s^-1")
                        .transpose(*self.input_properties[name]["dims"])
                        .values
                    )
                    return jnp.asarray(np.ascontiguousarray(arr))
                return jnp.zeros(shape)

            u_t = tend_or_zero("eastward_wind", (n_lev, n_lat, n_lon))
            v_t = tend_or_zero("northward_wind", (n_lev, n_lat, n_lon))
            t_t = tend_or_zero("air_temperature", (n_lev, n_lat, n_lon))
            ps_t = tend_or_zero("surface_air_pressure", (n_lat, n_lon))
            fvirt = self._fvirt()
            t_virt = temp * (1 + fvirt * q)

            n_tracers = state["tracers"].shape[0]
            tracer_t = (
                jnp.stack(
                    [
                        tend_or_zero(name, (n_lev, n_lat, n_lon))
                        for name in self._tracer_packer.tracer_names
                    ],
                    axis=0,
                )
                if n_tracers
                else jnp.zeros((0, n_lev, n_lat, n_lon))
            )
            q_t = tracer_t[0] if n_tracers else jnp.zeros((n_lev, n_lat, n_lon))

            virtual_temp_tend = t_t * (1 + fvirt * q) + fvirt * t_virt * q_t
            lnps_tend = ps_t / ps

            self._phys_tendencies = PhysicsTendencies(
                u=u_t,
                v=v_t,
                virtual_temperature=virtual_temp_tend,
                log_surface_pressure=lnps_tend,
                tracers=tracer_t,
            )

        # Advance one timestep (dry-mass fixer activated via gauss_weights +
        # pdryini when adiabatic=False, matching Fortran run.f90 lines 349-356).
        # Time-split physics adjustment (Held-Suarez etc.), matching the
        # Fortran wrapper's assign_tendencies + run.f90 physics block, is
        # folded into the same pure-function call when tendencies are set.
        if self._phys_tendencies is not None:
            spec_final = self._jit_advance_t(
                spec_orig,
                self._phys_tendencies,
                self._phis_grads,
                self.dyn_config,
                self.trans_config,
                self.stepper_config,
                self._latitudes,
                self._gauss_weights,
                self._pdryini,
            )
        else:
            spec_final = self._jit_advance(
                spec_orig,
                self._phis_grads,
                self.dyn_config,
                self.trans_config,
                self.stepper_config,
                self._latitudes,
                self._gauss_weights,
                self._pdryini,
            )

        # Cache the spectral state for the next call so we never re-do
        # grid→spectral (which would accumulate truncation error).
        self._spec_state = spec_final

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
            "tracers": to_numpy(grid_final.tracers),
        }
