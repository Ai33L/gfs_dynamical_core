from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
from gfs_dynamical_core.jax.dynamics import (
    compute_pressure_diagnostics, compute_vertical_velocities, 
    compute_vertical_advection, compute_pressure_gradient_force, DynamicsConfig
)
from gfs_dynamical_core.jax.states import GridState, GridGradients

def get_mock_config(n_lev):
    ak = jnp.zeros(n_lev + 1)
    bk = jnp.linspace(0, 1, n_lev + 1)
    dbk = bk[1:] - bk[:-1]
    ck = ak[1:] * bk[:-1] - ak[:-1] * bk[1:]
    
    return DynamicsConfig(
        ak=ak, bk=bk, ck=ck, dbk=dbk,
        rk=0.286, toa_pressure=0.0,
        radius=6.371e6, omega=7.292e-5, g=9.81,
        rd=287.0, rv=461.0, cp=1004.0, cvap=1810.0
    )

def test_pressure_diagnostics():
    n_lev = 10
    n_lat, n_lon = 32, 64
    config = get_mock_config(n_lev)
    log_ps = jnp.full((n_lat, n_lon), jnp.log(101325.0))
    
    diag = compute_pressure_diagnostics(log_ps, config)
    
    assert diag.ps.shape == (n_lat, n_lon)
    assert diag.pk.shape == (n_lev + 1, n_lat, n_lon)
    assert diag.dp.shape == (n_lev, n_lat, n_lon)
    assert diag.prs.shape == (n_lev, n_lat, n_lon)
    
    np.testing.assert_allclose(diag.pk[-1], diag.ps, atol=1e-8)
    np.testing.assert_allclose(diag.pk[0], config.ak[0], atol=1e-8)
    
    jit_func = jax.jit(compute_pressure_diagnostics)
    diag_jit = jit_func(log_ps, config)
    np.testing.assert_allclose(diag_jit.prs, diag.prs)

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
        tracers=jnp.zeros((1, n_lev, n_lat, n_lon))
    )
    
    grid_grads = GridGradients(
        d_log_ps_d_phi=jnp.zeros((n_lat, n_lon)),
        d_log_ps_d_lambda=jnp.zeros((n_lat, n_lon)),
        d_t_d_phi=jnp.zeros((n_lev, n_lat, n_lon)),
        d_t_d_lambda=jnp.zeros((n_lev, n_lat, n_lon))
    )
    
    vvels = compute_vertical_velocities(grid_state, grid_grads, diag, config)
    
    np.testing.assert_allclose(vvels.d_log_ps_d_t, 0.0, atol=1e-12)
    np.testing.assert_allclose(vvels.etadot, 0.0, atol=1e-12)
    np.testing.assert_allclose(vvels.omega, 0.0, atol=1e-12)
    
    grid_state = grid_state.replace(divergence=jnp.full_like(grid_state.divergence, 1e-6))
    vvels = compute_vertical_velocities(grid_state, grid_grads, diag, config)
    
    assert jnp.any(vvels.d_log_ps_d_t != 0.0)
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
    assert vadv.shape == (n_lev, n_lat, n_lon)
    np.testing.assert_allclose(vadv[1:-1], 1e-4, atol=1e-12)

def test_pressure_gradient_force():
    n_lev = 10
    n_lat, n_lon = 32, 64
    config = get_mock_config(n_lev)
    log_ps = jnp.full((n_lat, n_lon), jnp.log(101325.0))
    diag = compute_pressure_diagnostics(log_ps, config)
    
    virtual_temp = jnp.full((n_lev, n_lat, n_lon), 280.0)
    grid_grads = GridGradients(
        d_log_ps_d_phi=jnp.zeros((n_lat, n_lon)),
        d_log_ps_d_lambda=jnp.zeros((n_lat, n_lon)),
        d_t_d_phi=jnp.zeros((n_lev, n_lat, n_lon)),
        d_t_d_lambda=jnp.zeros((n_lev, n_lat, n_lon))
    )
    phis_grads = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))
    
    pgf_x, pgf_y = compute_pressure_gradient_force(virtual_temp, grid_grads, diag, config, phis_grads)
    
    np.testing.assert_allclose(pgf_x, 0.0, atol=1e-12)
    np.testing.assert_allclose(pgf_y, 0.0, atol=1e-12)
