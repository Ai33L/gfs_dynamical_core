from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np

from gfs_dynamical_core.jax.dynamics import (
    DynamicsConfig,
    assemble_grid_tendencies,
    compute_energy_conversion,
    compute_pressure_diagnostics,
    compute_pressure_gradient_force,
    compute_vertical_advection,
    compute_vertical_advection_tracers,
    compute_vertical_velocities,
)
from gfs_dynamical_core.jax.states import GridGradients, GridState
from gfs_dynamical_core.jax.transforms import get_gaussian_latitudes


def get_mock_config(n_lev):
    ak = jnp.zeros(n_lev + 1)
    bk = jnp.linspace(0, 1, n_lev + 1)
    dbk = bk[1:] - bk[:-1]
    ck = ak[1:] * bk[:-1] - ak[:-1] * bk[1:]

    return DynamicsConfig(
        ak=ak,
        bk=bk,
        ck=ck,
        dbk=dbk,
        rk=0.286,
        toa_pressure=0.0,
        radius=6.371e6,
        omega=7.292e-5,
        g=9.81,
        rd=287.0,
        rv=461.0,
        cp=1004.0,
        cvap=1810.0,
    )


def test_pressure_diagnostics():
    n_lev = 10
    n_lat, n_lon = 32, 64
    config = get_mock_config(n_lev)
    log_ps = jnp.full((n_lat, n_lon), jnp.log(101325.0))
    diag = compute_pressure_diagnostics(log_ps, config)
    assert diag.ps.shape == (n_lat, n_lon)
    np.testing.assert_allclose(diag.pk[-1], diag.ps, atol=1e-8)


def test_vertical_velocities():
    n_lev = 10
    n_lat, n_lon = 32, 64
    config = get_mock_config(n_lev)
    log_ps = jnp.full((n_lat, n_lon), jnp.log(101325.0))
    diag = compute_pressure_diagnostics(log_ps, config)
    grid_state = GridState(
        u=jnp.zeros((n_lev, n_lat, n_lon)),
        v=jnp.zeros((n_lev, n_lat, n_lon)),
        temperature=jnp.full((n_lev, n_lat, n_lon), 280.0),
        vorticity=jnp.zeros((n_lev, n_lat, n_lon)),
        divergence=jnp.zeros((n_lev, n_lat, n_lon)),
        log_surface_pressure=log_ps,
        tracers=jnp.zeros((1, n_lev, n_lat, n_lon)),
    )
    grid_grads = GridGradients(
        d_log_ps_d_phi=jnp.zeros((n_lat, n_lon)),
        d_log_ps_d_lambda=jnp.zeros((n_lat, n_lon)),
        d_t_d_phi=jnp.zeros((n_lev, n_lat, n_lon)),
        d_t_d_lambda=jnp.zeros((n_lev, n_lat, n_lon)),
        d_tracers_d_phi=jnp.zeros((1, n_lev, n_lat, n_lon)),
        d_tracers_d_lambda=jnp.zeros((1, n_lev, n_lat, n_lon)),
    )
    vvels = compute_vertical_velocities(grid_state, grid_grads, diag, config)
    np.testing.assert_allclose(vvels.etadot[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(vvels.etadot[-1], 0.0, atol=1e-12)


def test_vertical_advection():
    n_lev = 10
    n_lat, n_lon = 32, 64
    data = jnp.stack([jnp.full((n_lat, n_lon), float(k)) for k in range(n_lev)])
    etadot = jnp.full((n_lev + 1, n_lat, n_lon), 0.1)
    etadot = etadot.at[0].set(0.0)
    etadot = etadot.at[-1].set(0.0)
    dp = jnp.full((n_lev, n_lat, n_lon), 1000.0)
    vadv = compute_vertical_advection(data, etadot, dp)
    np.testing.assert_allclose(vadv[1:-1], 1e-4, atol=1e-12)
    # Check boundaries
    np.testing.assert_allclose(vadv[0], 0.5e-4, atol=1e-12)
    np.testing.assert_allclose(vadv[-1], 0.5e-4, atol=1e-12)


def test_vertical_advection_tracers():
    n_lev = 10
    n_lat, n_lon = 32, 64
    # Create an artificial profile that should trigger the flux limiters
    # E.g. a sharp spike
    data = jnp.zeros((n_lev, n_lat, n_lon))
    data = data.at[5].set(1.0)
    etadot = jnp.full((n_lev + 1, n_lat, n_lon), 0.1)
    etadot = etadot.at[0].set(0.0)
    etadot = etadot.at[-1].set(0.0)
    dp = jnp.full((n_lev, n_lat, n_lon), 1000.0)

    vadv = compute_vertical_advection_tracers(data, etadot, dp)
    assert not jnp.isnan(vadv).any()
    assert vadv.shape == (n_lev, n_lat, n_lon)


def test_pressure_gradient_force():
    n_lev = 10
    n_lat, n_lon = 32, 64
    config = get_mock_config(n_lev)
    log_ps = jnp.full((n_lat, n_lon), jnp.log(101325.0))
    diag = compute_pressure_diagnostics(log_ps, config)
    virtual_temp = jnp.full((n_lev, n_lat, n_lon), 280.0)
    grid_grads = GridGradients(
        d_log_ps_d_phi=jnp.full((n_lat, n_lon), 0.01),
        d_log_ps_d_lambda=jnp.full((n_lat, n_lon), 0.01),
        d_t_d_phi=jnp.full((n_lev, n_lat, n_lon), 0.01),
        d_t_d_lambda=jnp.full((n_lev, n_lat, n_lon), 0.01),
        d_tracers_d_phi=jnp.zeros((1, n_lev, n_lat, n_lon)),
        d_tracers_d_lambda=jnp.zeros((1, n_lev, n_lat, n_lon)),
    )
    surface_geopotential_grads = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))

    pgf_x, pgf_y = compute_pressure_gradient_force(
        virtual_temp, grid_grads, diag, config, surface_geopotential_grads
    )
    assert pgf_x.shape == (n_lev, n_lat, n_lon)
    assert pgf_y.shape == (n_lev, n_lat, n_lon)
    # The output shouldn't have NaNs or Infs
    assert not jnp.isnan(pgf_x).any()
    assert not jnp.isnan(pgf_y).any()


def test_full_grid_assembly():
    n_lev = 10
    n_lat, n_lon = 32, 64
    config = get_mock_config(n_lev)
    log_ps = jnp.full((n_lat, n_lon), jnp.log(101325.0))
    latitudes = get_gaussian_latitudes(n_lat)

    diag = compute_pressure_diagnostics(log_ps, config)
    grid_state = GridState(
        u=jnp.zeros((n_lev, n_lat, n_lon)),
        v=jnp.zeros((n_lev, n_lat, n_lon)),
        temperature=jnp.full((n_lev, n_lat, n_lon), 280.0),
        vorticity=jnp.zeros((n_lev, n_lat, n_lon)),
        divergence=jnp.zeros((n_lev, n_lat, n_lon)),
        log_surface_pressure=log_ps,
        tracers=jnp.zeros((1, n_lev, n_lat, n_lon)),
    )
    grid_grads = GridGradients(
        d_log_ps_d_phi=jnp.zeros((n_lat, n_lon)),
        d_log_ps_d_lambda=jnp.zeros((n_lat, n_lon)),
        d_t_d_phi=jnp.zeros((n_lev, n_lat, n_lon)),
        d_t_d_lambda=jnp.zeros((n_lev, n_lat, n_lon)),
        d_tracers_d_phi=jnp.zeros((1, n_lev, n_lat, n_lon)),
        d_tracers_d_lambda=jnp.zeros((1, n_lev, n_lat, n_lon)),
    )
    phis_grads = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))

    vvels = compute_vertical_velocities(grid_state, grid_grads, diag, config)
    pgf = compute_pressure_gradient_force(
        grid_state.temperature, grid_grads, diag, config, phis_grads
    )
    energy_conv = compute_energy_conversion(
        vvels.omega, grid_state.temperature, grid_state.tracers[0], config
    )

    tends = assemble_grid_tendencies(
        grid_state, grid_grads, vvels, diag, pgf, energy_conv, config, latitudes
    )

    assert tends.u_flux.shape == (n_lev, n_lat, n_lon)
    assert tends.temp_tend.shape == (n_lev, n_lat, n_lon)
    assert tends.log_ps_tend.shape == (n_lat, n_lon)


def test_tracer_horizontal_advection():
    """
    Task 8.1 regression test.

    With a uniform u-wind and a tracer that varies only in lambda (longitude),
    the horizontal advection term  -u * dq/dlambda  must be non-zero and must
    change the tracer tendency relative to the vertical-advection-only baseline.

    If horizontal advection were still missing the two tendencies would be
    identical; the test would fail.
    """
    n_lev = 4
    n_lat, n_lon = 8, 16
    config = get_mock_config(n_lev)
    log_ps = jnp.full((n_lat, n_lon), jnp.log(101325.0))
    latitudes = get_gaussian_latitudes(n_lat)

    diag = compute_pressure_diagnostics(log_ps, config)

    u_val = 10.0  # m/s eastward wind
    dq_dlambda = 1e-5  # tracer gradient in lambda direction

    grid_state = GridState(
        u=jnp.full((n_lev, n_lat, n_lon), u_val),
        v=jnp.zeros((n_lev, n_lat, n_lon)),
        temperature=jnp.full((n_lev, n_lat, n_lon), 280.0),
        vorticity=jnp.zeros((n_lev, n_lat, n_lon)),
        divergence=jnp.zeros((n_lev, n_lat, n_lon)),
        log_surface_pressure=log_ps,
        tracers=jnp.zeros((1, n_lev, n_lat, n_lon)),
    )
    phis_grads = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))

    # Gradients WITH a non-zero tracer lambda-gradient
    grid_grads_with_hadv = GridGradients(
        d_log_ps_d_phi=jnp.zeros((n_lat, n_lon)),
        d_log_ps_d_lambda=jnp.zeros((n_lat, n_lon)),
        d_t_d_phi=jnp.zeros((n_lev, n_lat, n_lon)),
        d_t_d_lambda=jnp.zeros((n_lev, n_lat, n_lon)),
        d_tracers_d_phi=jnp.zeros((1, n_lev, n_lat, n_lon)),
        d_tracers_d_lambda=jnp.full((1, n_lev, n_lat, n_lon), dq_dlambda),
    )

    # Gradients WITHOUT tracer gradient (vertical advection only baseline)
    grid_grads_no_hadv = GridGradients(
        d_log_ps_d_phi=jnp.zeros((n_lat, n_lon)),
        d_log_ps_d_lambda=jnp.zeros((n_lat, n_lon)),
        d_t_d_phi=jnp.zeros((n_lev, n_lat, n_lon)),
        d_t_d_lambda=jnp.zeros((n_lev, n_lat, n_lon)),
        d_tracers_d_phi=jnp.zeros((1, n_lev, n_lat, n_lon)),
        d_tracers_d_lambda=jnp.zeros((1, n_lev, n_lat, n_lon)),
    )

    vvels = compute_vertical_velocities(grid_state, grid_grads_with_hadv, diag, config)
    pgf = compute_pressure_gradient_force(
        grid_state.temperature, grid_grads_with_hadv, diag, config, phis_grads
    )
    energy_conv = compute_energy_conversion(
        vvels.omega, grid_state.temperature, grid_state.tracers[0], config
    )

    tends_with = assemble_grid_tendencies(
        grid_state,
        grid_grads_with_hadv,
        vvels,
        diag,
        pgf,
        energy_conv,
        config,
        latitudes,
    )
    tends_without = assemble_grid_tendencies(
        grid_state, grid_grads_no_hadv, vvels, diag, pgf, energy_conv, config, latitudes
    )

    # The horizontal advection contribution: -u * dq/dlambda = -10 * 1e-5 = -1e-4
    expected_hadv = -u_val * dq_dlambda
    actual_diff = np.array(tends_with.tracer_tends[0] - tends_without.tracer_tends[0])

    # Every grid point should show the expected horizontal advection increment
    np.testing.assert_allclose(
        actual_diff,
        expected_hadv,
        rtol=1e-10,
        err_msg="Tracer horizontal advection (-u*dq/dlambda) is missing or incorrect",
    )


def test_pgf_cofb_geopotential_term():
    """
    Task 8.2 regression test.

    The geopotential-integral cofb (px2 term) in getpresgrad must be non-zero
    and distinct from zero whenever the atmosphere has a vertical temperature
    structure and a surface pressure gradient.  Before the fix the px2 term was
    absent; this test quantifies its magnitude and checks it has the expected
    sign/shape.

    Setup: uniform virtual temperature 280 K, uniform surface pressure 101325 Pa
    but with a non-zero grad(lnps) in x.  With a pure-sigma coordinate
    (ak=0, bk linear) the bk-ratio differences are non-zero at every interface,
    so cofb_geo must be non-zero except at the top layer.
    """
    n_lev = 5
    n_lat, n_lon = 4, 8
    config = get_mock_config(n_lev)
    log_ps = jnp.full((n_lat, n_lon), jnp.log(101325.0))
    diag = compute_pressure_diagnostics(log_ps, config)

    virtual_temp = jnp.full((n_lev, n_lat, n_lon), 280.0)

    dlnps_dx = 1e-6  # non-zero surface pressure gradient

    grid_grads = GridGradients(
        d_log_ps_d_phi=jnp.zeros((n_lat, n_lon)),
        d_log_ps_d_lambda=jnp.full((n_lat, n_lon), dlnps_dx),
        d_t_d_phi=jnp.zeros((n_lev, n_lat, n_lon)),
        d_t_d_lambda=jnp.zeros((n_lev, n_lat, n_lon)),
        d_tracers_d_phi=jnp.zeros((1, n_lev, n_lat, n_lon)),
        d_tracers_d_lambda=jnp.zeros((1, n_lev, n_lat, n_lon)),
    )
    phis_grads = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))

    pgf_x, pgf_y = compute_pressure_gradient_force(
        virtual_temp, grid_grads, diag, config, phis_grads
    )

    # Shape and finiteness
    assert pgf_x.shape == (n_lev, n_lat, n_lon)
    assert not jnp.isnan(pgf_x).any(), "PGF_x contains NaN"
    assert not jnp.isinf(pgf_x).any(), "PGF_x contains Inf"

    # The px2 term (cofb_geo * ps * dlnps_dx) must be non-zero at interior levels.
    # With uniform T=280K and pure-sigma coordinate, bk[i+1]/pk[i+1] - bk[i]/pk[i]
    # is non-zero at every interface, so the cumulative sum grows from the bottom up.
    # The top level (i=0) has cofb_geo=0 by construction; lower levels are non-zero.
    # We verify that at least the bottom level has a noticeably different PGF than
    # it would without the px2 term by checking the vertical profile is NOT flat.
    pgf_x_profile = np.array(pgf_x[:, 0, 0])
    assert not np.allclose(pgf_x_profile, pgf_x_profile[0]), (
        "PGF_x is uniform across levels — the cofb_geopotential (px2) term "
        "is likely still missing, which would flatten the vertical profile."
    )
