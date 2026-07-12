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


import climt


def test_jax_core_accepts_component_list_and_merges_properties():
    from gfs_dynamical_core.component_jax import GFSDynamicsJAX
    hs = climt.HeldSuarez()
    dycore = GFSDynamicsJAX(tendency_component_list=[hs])
    # air_pressure now an output (needed by internal components)
    assert "air_pressure" in dycore.output_properties
    assert "air_pressure_on_interface_levels" in dycore.output_properties
    # constructed without error and is a TendencyStepper
    from sympl import TendencyStepper
    assert isinstance(dycore, TendencyStepper)


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


from datetime import timedelta
import numpy as np


def test_component_path_matches_manual_tendencies():
    L, n_lev = 16, 10
    grid = climt.get_grid(nx=2 * L - 1, ny=L, nz=n_lev)
    hs = climt.HeldSuarez()
    from gfs_dynamical_core.component_jax import GFSDynamicsJAX
    from gfs_dynamical_core.jax.dynamics import compute_pressure_diagnostics

    ts = timedelta(minutes=10)

    # --- component path ---
    dyn_a = GFSDynamicsJAX(tendency_component_list=[hs])
    state_a = climt.get_default_state([dyn_a], grid_state=grid)
    _, out_a = dyn_a(state_a, timestep=ts)

    # --- manual path: same forcing pushed via set_physics_tendencies ---
    dyn_b = GFSDynamicsJAX()  # no components
    state_b = climt.get_default_state([dyn_b], grid_state=grid)
    tend, _ = hs(state_b)
    u_t = tend["eastward_wind"].transpose("mid_levels", "lat", "lon").values
    v_t = tend["northward_wind"].transpose("mid_levels", "lat", "lon").values
    t_t = tend["air_temperature"].transpose("mid_levels", "lat", "lon").values
    dyn_b.set_physics_tendencies(u_t, v_t, t_t)
    _, out_b = dyn_b(state_b, timestep=ts)

    # HS produces only u/v/T tendencies (no moisture), so virtual-T parity
    # reduces to the plain-T path and the two must agree closely.
    np.testing.assert_allclose(
        out_a["eastward_wind"].values, out_b["eastward_wind"].values,
        rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        out_a["air_temperature"].values, out_b["air_temperature"].values,
        rtol=1e-6, atol=1e-6)


def _grid_state(L=16, n_lev=10):
    grid = climt.get_grid(nx=2 * L - 1, ny=L, nz=n_lev)
    from gfs_dynamical_core.component_jax import GFSDynamicsJAX
    dycore = GFSDynamicsJAX()
    state = climt.get_default_state([dycore], grid_state=grid)
    return dycore, state


def test_array_call_outputs_air_pressure():
    dycore, state = _grid_state()
    _, out = dycore(state, timestep=timedelta(minutes=10))
    assert "air_pressure" in out
    assert "air_pressure_on_interface_levels" in out
    ap = out["air_pressure"]
    assert ap.shape[0] == state["air_temperature"].shape[0]  # mid_levels
    assert (ap.values > 0).all()

    # array_call must actually COMPUTE these from the stepped surface
    # pressure, not merely pass the input state's stale values through
    # __call__'s "carry over missing keys" fallback. Verify by calling
    # array_call directly and checking the keys are present in its own
    # return dict (independent of any state passthrough).
    raw_state = {
        "eastward_wind": state["eastward_wind"].transpose(
            "mid_levels", "lat", "lon").values,
        "northward_wind": state["northward_wind"].transpose(
            "mid_levels", "lat", "lon").values,
        "air_temperature": state["air_temperature"].transpose(
            "mid_levels", "lat", "lon").values,
        "surface_air_pressure": state["surface_air_pressure"].transpose(
            "lat", "lon").values,
        "surface_geopotential": state["surface_geopotential"].transpose(
            "lat", "lon").values,
        "a_coord": state[
            "atmosphere_hybrid_sigma_pressure_a_coordinate_on_interface_levels"
        ].values,
        "b_coord": state[
            "atmosphere_hybrid_sigma_pressure_b_coordinate_on_interface_levels"
        ].values,
        "tracers": state["specific_humidity"].transpose(
            "mid_levels", "lat", "lon").values[None, ...],
    }
    _, raw_out = dycore.array_call(raw_state, timedelta(minutes=10))
    assert "air_pressure" in raw_out
    assert "air_pressure_on_interface_levels" in raw_out
    assert raw_out["air_pressure_on_interface_levels"].shape[0] == (
        state["air_temperature"].shape[0] + 1
    )


def _register_test_tracer(name, default_value, units="kg/kg"):
    """Register a tracer AND give climt a default value for it, so
    ``climt.get_default_state`` can initialize it (registration alone is not
    enough — climt needs an entry in its ``default_values`` registry)."""
    from sympl import register_tracer
    import climt._core.initialization as cinit
    register_tracer(name, units)
    cinit.default_values[name] = {
        "value": default_value, "units": units, "domain": "atmosphere"}


def _unregister_test_tracer(name):
    """Undo _register_test_tracer so nothing leaks into other tests. sympl
    tracks tracers in TWO registries — _tracer_unit_dict (units) and
    _tracer_names (what get_tracer_names() returns) — so both must be cleared,
    else a later test sees a phantom tracer with no units (KeyError)."""
    try:
        from sympl._core.tracers import _tracer_unit_dict, _tracer_names
        _tracer_unit_dict.pop(name, None)
        while name in _tracer_names:
            _tracer_names.remove(name)
    except Exception:
        pass
    try:
        import climt._core.initialization as cinit
        cinit.default_values.pop(name, None)
    except Exception:
        pass


def test_multi_tracer_roundtrip_through_step():
    _register_test_tracer("test_tracer", 0.0)
    try:
        L, n_lev = 16, 10
        grid = climt.get_grid(nx=2 * L - 1, ny=L, nz=n_lev)
        from gfs_dynamical_core.component_jax import GFSDynamicsJAX
        dycore = GFSDynamicsJAX()
        state = climt.get_default_state([dycore], grid_state=grid)
        assert "test_tracer" in state
        _, out = dycore(state, timestep=timedelta(minutes=10))
        assert "test_tracer" in out
        assert out["test_tracer"].shape == state["test_tracer"].shape
        assert not np.isnan(out["test_tracer"].values).any()
    finally:
        _unregister_test_tracer("test_tracer")


def test_multi_tracer_values_not_scrambled():
    """Two tracers set to distinct, spatially-uniform constants must each come
    back near their OWN constant after a step. A uniform tracer field has zero
    horizontal/vertical gradient, so advection and hyperdiffusion leave it
    unchanged (the l=0 spectral mode is undamped) — so both tracers are
    preserved essentially exactly. A bug that swaps/aliases the tracer axis
    would return test_tracer at ~0.01 (specific_humidity's value) or vice
    versa; this test catches that (the all-zero tracers of the Task-4 test
    could not).
    """
    _register_test_tracer("test_tracer", 5.0)
    try:
        L, n_lev = 16, 10
        grid = climt.get_grid(nx=2 * L - 1, ny=L, nz=n_lev)
        from gfs_dynamical_core.component_jax import GFSDynamicsJAX
        dycore = GFSDynamicsJAX()
        state = climt.get_default_state([dycore], grid_state=grid)

        # Distinct, well-separated, spatially-uniform tracer fields.
        Q0, T0 = 0.01, 5.0                       # tracer 0 (humidity), tracer 1
        state["specific_humidity"].values[:] = Q0
        state["test_tracer"].values[:] = T0

        _, out = dycore(state, timestep=timedelta(minutes=10))

        q_out = out["specific_humidity"].values
        t_out = out["test_tracer"].values

        # Each tracer stays essentially at its own uniform constant.
        assert np.allclose(q_out, Q0, rtol=1e-6, atol=1e-8), (
            "specific_humidity drifted from its uniform constant: "
            f"mean={q_out.mean()}, expected {Q0}")
        assert np.allclose(t_out, T0, rtol=1e-6, atol=1e-8), (
            "test_tracer drifted from its uniform constant: "
            f"mean={t_out.mean()}, expected {T0}")

        # Explicit anti-scramble: each output is far closer to its own
        # constant than to the other tracer's constant.
        assert abs(t_out.mean() - T0) < 0.1 * abs(t_out.mean() - Q0)
        assert abs(q_out.mean() - Q0) < 0.1 * abs(q_out.mean() - T0)
    finally:
        _unregister_test_tracer("test_tracer")


def test_zero_negative_moisture_clips_tracer_zero():
    """With zero_negative_moisture=True (default, Fortran parity), negative
    specific humidity is clipped to zero after the step; with it disabled the
    negatives survive. A uniform field is advection/diffusion-invariant, so the
    only thing that can change a uniform -1e-3 humidity is the clip."""
    from gfs_dynamical_core.component_jax import GFSDynamicsJAX
    L, n_lev = 16, 10
    grid = climt.get_grid(nx=2 * L - 1, ny=L, nz=n_lev)
    ts = timedelta(minutes=10)

    # Clipping ON (default): negatives removed.
    dyc = GFSDynamicsJAX()
    state = climt.get_default_state([dyc], grid_state=grid)
    state["specific_humidity"].values[:] = -1e-3
    _, out = dyc(state, timestep=ts)
    assert (out["specific_humidity"].values >= 0.0).all()

    # Clipping OFF: the negative humidity survives the step.
    dyc2 = GFSDynamicsJAX(zero_negative_moisture=False)
    state2 = climt.get_default_state([dyc2], grid_state=grid)
    state2["specific_humidity"].values[:] = -1e-3
    _, out2 = dyc2(state2, timestep=ts)
    assert (out2["specific_humidity"].values < 0.0).any()
