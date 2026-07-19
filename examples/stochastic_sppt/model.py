"""Pure-JAX Held-Suarez model harness built on gfs_dynamical_core.jax."""
import climt
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from sympl import get_constant, set_constant

from gfs_dynamical_core.jax.dynamics import DynamicsConfig
from gfs_dynamical_core.jax.states import GridState, SpectralState
from gfs_dynamical_core.jax.stepper import (
    StepperConfig,
    advance_with_tendencies,
    init_diffusion_operators,
    init_semi_implicit_matrices,
)
from gfs_dynamical_core.jax.transforms import (
    TransformConfig,
    enforce_triangular_truncation,
    get_gaussian_latitudes,
    grid_to_spectral,
    prebuild_kernels,
)


@struct.dataclass
class ModelBundle:
    dyn_config: DynamicsConfig
    trans_config: TransformConfig
    stepper_config: StepperConfig
    latitudes: jnp.ndarray
    gauss_weights: jnp.ndarray
    pdryini: float = struct.field(pytree_node=False)
    phis_grads: tuple
    dt: float = struct.field(pytree_node=False)
    n_lev: int = struct.field(pytree_node=False)


def _ak_bk_constants(L, n_lev):
    """Fetch ak/bk and Earth constants from climt (setup-time only).

    NOTE (deviation from brief): climt.get_grid(...) returns a plain dict
    keyed by the full CF standard names (not the sympl component aliases
    "a_coord"/"b_coord" -- those aliases are only resolved by the
    TendencyStepper/array_call property machinery, e.g. in
    component_jax.py::array_call, which receives an already-alias-resolved
    state dict). Calling code against the raw climt.get_grid() dict must use
    the full standard names.
    """
    set_constant("reference_air_pressure", value=1e5, units="Pa")
    grid = climt.get_grid(nx=2 * L - 1, ny=L, nz=n_lev)
    ak = jnp.array(
        np.asarray(
            grid[
                "atmosphere_hybrid_sigma_pressure_a_coordinate_on_interface_levels"
            ].values,
            dtype=np.float64,
        )
    )
    bk = jnp.array(
        np.asarray(
            grid[
                "atmosphere_hybrid_sigma_pressure_b_coordinate_on_interface_levels"
            ].values,
            dtype=np.float64,
        )
    )
    c = dict(
        rd=get_constant("gas_constant_of_dry_air", "J kg^-1 K^-1"),
        cp=get_constant("heat_capacity_of_dry_air_at_constant_pressure", "J kg^-1 K^-1"),
        rv=get_constant("gas_constant_of_vapor_phase", "J kg^-1 K^-1"),
        cvap=get_constant("heat_capacity_of_vapor_phase", "J kg^-1 K^-1"),
        toa_pressure=get_constant("top_of_model_pressure", "Pa"),
        radius=get_constant("planetary_radius", "m"),
        omega=get_constant("planetary_rotation_rate", "s^-1"),
        g=get_constant("gravitational_acceleration", "m s^-2"),
    )
    return ak, bk, c


def build_model(model_config) -> ModelBundle:
    L, ntrunc, dt = model_config.L, model_config.ntrunc, model_config.dt
    n_lev = model_config.n_lev
    ak, bk, c = _ak_bk_constants(L, n_lev)

    dbk = bk[:-1] - bk[1:]
    ck = ak[:-1] * bk[1:] - ak[1:] * bk[:-1]
    dyn_config = DynamicsConfig(
        ak=ak, bk=bk, ck=ck, dbk=dbk, rk=c["rd"] / c["cp"],
        toa_pressure=c["toa_pressure"], radius=c["radius"], omega=c["omega"],
        g=c["g"], rd=c["rd"], rv=c["rv"], cp=c["cp"], cvap=c["cvap"],
    )
    trans_config = TransformConfig(L=L, sampling="gl", ntrunc=ntrunc, radius=c["radius"])
    prebuild_kernels(L, "gl")

    _sc = StepperConfig(dt=dt)
    si = init_semi_implicit_matrices(dyn_config, trans_config, dt,
                                     aa22=_sc.aa22, aa33=_sc.aa33, bb4=_sc.bb4)
    diff = init_diffusion_operators(dyn_config, trans_config, dt)
    stepper_config = StepperConfig(
        dt=dt, explicit=False,
        amhyb=si["amhyb"], bmhyb=si["bmhyb"], tor_hyb=si["tor_hyb"],
        svhyb=si["svhyb"], d_hyb_m=si["d_hyb_m"],
        disspec=diff["disspec"], diff_prof=diff["diff_prof"], dmp_prof=diff["dmp_prof"],
    )

    latitudes = get_gaussian_latitudes(L)
    _, raw_w = np.polynomial.legendre.leggauss(L)
    gauss_weights = jnp.array(raw_w / 2.0)
    n_lat, n_lon = L, 2 * L - 1
    phis_grads = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))

    # Dry initial mean surface pressure (q=0): pdryini = area-mean ps.
    pdryini = float(jnp.sum(gauss_weights[:, None] * jnp.full((n_lat, n_lon), 1e5)) / n_lon)

    return ModelBundle(
        dyn_config=dyn_config, trans_config=trans_config, stepper_config=stepper_config,
        latitudes=latitudes, gauss_weights=gauss_weights, pdryini=pdryini,
        phis_grads=phis_grads, dt=dt, n_lev=n_lev,
    )


def step(bundle, spec_state, phys_tends):
    """One differentiable model step (dynamics + optional physics increment)."""
    return advance_with_tendencies(
        spec_state, phys_tends, bundle.phis_grads, bundle.dyn_config,
        bundle.trans_config, bundle.stepper_config, bundle.latitudes,
        bundle.gauss_weights, bundle.pdryini,
    )


def rest_state(bundle, key, t0=250.0, ps0=1.0e5, noise=1.0e-3) -> SpectralState:
    """Isothermal rest state + tiny temperature noise, in spectral space."""
    n_lev = bundle.n_lev
    n_lat, n_lon = bundle.trans_config.L, 2 * bundle.trans_config.L - 1
    temp = t0 + noise * jax.random.normal(key, (n_lev, n_lat, n_lon))
    zeros3 = jnp.zeros((n_lev, n_lat, n_lon))
    grid = GridState(
        u=zeros3, v=zeros3, temperature=temp,
        vorticity=zeros3, divergence=zeros3,
        log_surface_pressure=jnp.full((n_lat, n_lon), jnp.log(ps0)),
        tracers=jnp.zeros((1, n_lev, n_lat, n_lon)),
    )
    return grid_to_spectral(grid, bundle.trans_config)


def spectral_truncate(spec_state, bundle_from, bundle_to) -> SpectralState:
    """Coarse-grain a spectral state from a higher L to a lower L by extracting
    the central (l, m) block and re-enforcing the target triangular truncation."""
    Lf = bundle_from.trans_config.L
    Lt = bundle_to.trans_config.L
    Tt = bundle_to.trans_config.truncation
    m0 = Lf - 1  # index of m=0 in the (2Lf-1) axis
    lo, hi = m0 - (Lt - 1), m0 + (Lt - 1) + 1  # central 2Lt-1 columns

    def cut(flm):
        return flm[..., :Lt, lo:hi]

    def cut_enf(flm):
        return enforce_triangular_truncation(cut(flm), Lt, Tt)

    return SpectralState(
        vorticity=cut_enf(spec_state.vorticity),
        divergence=cut_enf(spec_state.divergence),
        temperature=cut_enf(spec_state.temperature),
        log_surface_pressure=cut_enf(spec_state.log_surface_pressure),
        tracers=cut_enf(spec_state.tracers),
    )
