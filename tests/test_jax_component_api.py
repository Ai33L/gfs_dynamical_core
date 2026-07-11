def test_get_valid_properties_filters_known_quantities():
    from gfs_dynamical_core._property_utils import get_valid_properties
    gfs_props = {"air_temperature": {"dims": ["mid_levels", "lat", "lon"]}}
    prognostic = {
        "air_temperature": {"dims": ["mid_levels", "lat", "lon"], "units": "K"},
        "my_forcing": {"dims": ["mid_levels", "lat", "lon"], "units": "K s^-1"},
    }
    result = get_valid_properties(gfs_props, prognostic, "input")
    assert "air_temperature" not in result   # already known to the core
    assert "my_forcing" in result            # extra quantity surfaced


import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"
import jax
from jax import config as _jaxcfg
_jaxcfg.update("jax_enable_x64", True)
import jax.numpy as jnp
from gfs_dynamical_core.jax.stepper import (
    StepperConfig, advance, advance_with_tendencies, PhysicsTendencies,
)
from gfs_dynamical_core.jax.dynamics import DynamicsConfig
from gfs_dynamical_core.jax.states import SpectralState
from gfs_dynamical_core.jax.transforms import (
    TransformConfig, get_gaussian_latitudes,
)


def _mock_dyn_config(n_lev):
    ak = jnp.zeros(n_lev + 1)
    bk = jnp.linspace(1, 0, n_lev + 1)  # BTT: surface bk=1, TOA bk=0
    dbk = bk[:-1] - bk[1:]
    ck = ak[:-1] * bk[1:] - ak[1:] * bk[:-1]
    return DynamicsConfig(
        ak=ak, bk=bk, ck=ck, dbk=dbk, rk=0.286, toa_pressure=0.0,
        radius=6.371e6, omega=7.292e-5, g=9.81, rd=287.0, rv=461.0,
        cp=1004.0, cvap=1810.0,
    )


def _mock_state(n_lev, L, n_tracers=1):
    z = lambda: jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128)
    lnps = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128).at[0, L - 1].set(1.0)
    return SpectralState(
        vorticity=z(), divergence=z(), temperature=z(),
        log_surface_pressure=lnps,
        tracers=jnp.zeros((n_tracers, n_lev, L, 2 * L - 1), dtype=jnp.complex128),
    )


def test_advance_handles_two_tracers():
    L, n_lev = 8, 10
    dyn = _mock_dyn_config(n_lev)
    trans = TransformConfig(L=L, radius=1.0)
    sc = StepperConfig(dt=10.0, explicit=True)
    lat = get_gaussian_latitudes(L)
    n_lat, n_lon = trans.n_lat, trans.n_lon
    phis = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))
    state = _mock_state(n_lev, L, n_tracers=2)

    out = advance(state, phis, dyn, trans, sc, lat)
    assert out.tracers.shape == (2, n_lev, L, 2 * L - 1)
    assert not jnp.isnan(out.tracers).any()


def test_advance_with_zero_tendencies_matches_advance():
    L, n_lev = 8, 10
    dyn = _mock_dyn_config(n_lev)
    trans = TransformConfig(L=L, radius=1.0)
    sc = StepperConfig(dt=10.0, explicit=True)
    lat = get_gaussian_latitudes(L)
    n_lat, n_lon = trans.n_lat, trans.n_lon
    phis = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))
    state = _mock_state(n_lev, L)

    base = advance(state, phis, dyn, trans, sc, lat)
    zero_tends = PhysicsTendencies(
        u=jnp.zeros((n_lev, n_lat, n_lon)),
        v=jnp.zeros((n_lev, n_lat, n_lon)),
        virtual_temperature=jnp.zeros((n_lev, n_lat, n_lon)),
        log_surface_pressure=jnp.zeros((n_lat, n_lon)),
        tracers=jnp.zeros((1, n_lev, n_lat, n_lon)),
    )
    with_t = advance_with_tendencies(state, zero_tends, phis, dyn, trans, sc, lat)
    assert jnp.allclose(base.vorticity, with_t.vorticity)
    assert jnp.allclose(base.divergence, with_t.divergence)
    assert jnp.allclose(base.temperature, with_t.temperature)
    assert jnp.allclose(base.log_surface_pressure, with_t.log_surface_pressure)
    assert jnp.allclose(base.tracers, with_t.tracers)


from gfs_dynamical_core.jax.states import SpectralTendencies


def _zero_spec_tends(n_lev, L, n_tracers=1):
    z = lambda: jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128)
    return SpectralTendencies(
        d_vorticity_d_t=z(), d_divergence_d_t=z(), d_temperature_d_t=z(),
        d_log_surface_pressure_d_t=jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128),
        d_tracers_d_t=jnp.zeros((n_tracers, n_lev, L, 2 * L - 1), dtype=jnp.complex128),
    )


def test_spectral_vorticity_tendency_applied_directly():
    """A spectral vorticity tendency is added as spec_new.vorticity + dt*tend,
    with no grid round-trip (the SKEB injection path)."""
    L, n_lev = 8, 10
    dyn = _mock_dyn_config(n_lev)
    trans = TransformConfig(L=L, radius=1.0)
    sc = StepperConfig(dt=10.0, explicit=True)
    lat = get_gaussian_latitudes(L)
    n_lat, n_lon = trans.n_lat, trans.n_lon
    phis = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))
    state = _mock_state(n_lev, L)

    base = advance(state, phis, dyn, trans, sc, lat)
    dvort = _zero_spec_tends(n_lev, L)
    bump = jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128).at[:, 2, L].set(0.3)
    dvort = dvort.replace(d_vorticity_d_t=bump)

    out = advance_with_tendencies(
        state, None, phis, dyn, trans, sc, lat, spec_tends=dvort
    )
    assert jnp.allclose(out.vorticity, base.vorticity + sc.dt * bump)
    assert jnp.allclose(out.divergence, base.divergence)


def test_grid_and_spectral_paths_compose_additively():
    """phys_tends + spec_tends == sum of each applied alone (minus one base)."""
    L, n_lev = 8, 10
    dyn = _mock_dyn_config(n_lev)
    trans = TransformConfig(L=L, radius=1.0)
    sc = StepperConfig(dt=10.0, explicit=True)
    lat = get_gaussian_latitudes(L)
    n_lat, n_lon = trans.n_lat, trans.n_lon
    phis = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))
    state = _mock_state(n_lev, L)

    grid_t = PhysicsTendencies(
        u=0.01 * jnp.ones((n_lev, n_lat, n_lon)),
        v=jnp.zeros((n_lev, n_lat, n_lon)),
        virtual_temperature=jnp.zeros((n_lev, n_lat, n_lon)),
        log_surface_pressure=jnp.zeros((n_lat, n_lon)),
        tracers=jnp.zeros((1, n_lev, n_lat, n_lon)),
    )
    spec_t = _zero_spec_tends(n_lev, L).replace(
        d_temperature_d_t=jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128)
        .at[:, 1, L].set(0.2)
    )

    base = advance(state, phis, dyn, trans, sc, lat)
    only_grid = advance_with_tendencies(state, grid_t, phis, dyn, trans, sc, lat)
    only_spec = advance_with_tendencies(
        state, None, phis, dyn, trans, sc, lat, spec_tends=spec_t
    )
    both = advance_with_tendencies(
        state, grid_t, phis, dyn, trans, sc, lat, spec_tends=spec_t
    )
    # increments are additive about the common dynamics step
    for f in ("vorticity", "divergence", "temperature",
              "log_surface_pressure", "tracers"):
        expected = (getattr(only_grid, f) + getattr(only_spec, f)
                    - getattr(base, f))
        assert jnp.allclose(getattr(both, f), expected)


def test_advance_with_tendencies_is_differentiable():
    L, n_lev = 8, 10
    dyn = _mock_dyn_config(n_lev)
    trans = TransformConfig(L=L, radius=1.0)
    sc = StepperConfig(dt=10.0, explicit=True)
    lat = get_gaussian_latitudes(L)
    n_lat, n_lon = trans.n_lat, trans.n_lon
    phis = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))
    state = _mock_state(n_lev, L)

    def loss(scale):
        tends = PhysicsTendencies(
            u=scale * jnp.ones((n_lev, n_lat, n_lon)),
            v=jnp.zeros((n_lev, n_lat, n_lon)),
            virtual_temperature=jnp.zeros((n_lev, n_lat, n_lon)),
            log_surface_pressure=jnp.zeros((n_lat, n_lon)),
            tracers=jnp.zeros((1, n_lev, n_lat, n_lon)),
        )
        out = advance_with_tendencies(state, tends, phis, dyn, trans, sc, lat)
        return jnp.sum(jnp.abs(out.divergence) ** 2)

    g = jax.grad(loss)(1.0)
    assert jnp.isfinite(g)


def test_advance_with_spectral_tendencies_is_differentiable():
    """Gradients must flow through the spectral (spec_tends) injection path —
    this is the path SKEB/SPPT train through."""
    L, n_lev = 8, 10
    dyn = _mock_dyn_config(n_lev)
    trans = TransformConfig(L=L, radius=1.0)
    sc = StepperConfig(dt=10.0, explicit=True)
    lat = get_gaussian_latitudes(L)
    n_lat, n_lon = trans.n_lat, trans.n_lon
    phis = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))
    state = _mock_state(n_lev, L)

    def loss(amp):
        bump = jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128).at[:, 2, L].set(1.0)
        spec_t = _zero_spec_tends(n_lev, L).replace(d_vorticity_d_t=amp * bump)
        out = advance_with_tendencies(
            state, None, phis, dyn, trans, sc, lat, spec_tends=spec_t
        )
        return jnp.sum(jnp.abs(out.vorticity) ** 2)

    g = jax.grad(loss)(1.0)
    assert jnp.isfinite(g)
