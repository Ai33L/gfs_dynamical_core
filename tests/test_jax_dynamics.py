from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
from gfs_dynamical_core.jax.dynamics import (
    compute_pressure_diagnostics, DynamicsConfig
)

def test_pressure_diagnostics():
    n_lev = 10
    n_lat, n_lon = 32, 64
    
    # Mock ak, bk (linear for simplicity)
    # GFS hybrid coordinates: ak[top]=0, ak[bot]=0, bk[top]=0, bk[bot]=1
    ak = jnp.zeros(n_lev + 1)
    bk = jnp.linspace(0, 1, n_lev + 1)
    
    config = DynamicsConfig(
        ak=ak,
        bk=bk,
        rk=0.286,
        toa_pressure=0.0,
        radius=6.371e6,
        omega=7.292e-5,
        g=9.81,
        rd=287.0,
        rv=461.0,
        cp=1004.0,
        cvap=1810.0
    )
    
    log_ps = jnp.full((n_lat, n_lon), jnp.log(101325.0))
    
    diag = compute_pressure_diagnostics(log_ps, config)
    
    # Verify shapes
    assert diag.ps.shape == (n_lat, n_lon)
    assert diag.pk.shape == (n_lev + 1, n_lat, n_lon)
    assert diag.dp.shape == (n_lev, n_lat, n_lon)
    assert diag.prs.shape == (n_lev, n_lat, n_lon)
    
    # Verify surface pressure matches (last interface)
    np.testing.assert_allclose(diag.pk[-1], diag.ps, atol=1e-8)
    
    # Verify pk[0] is ak[0] (if toa_pressure=0)
    np.testing.assert_allclose(diag.pk[0], config.ak[0], atol=1e-8)
    
    # JIT check
    jit_func = jax.jit(compute_pressure_diagnostics)
    diag_jit = jit_func(log_ps, config)
    np.testing.assert_allclose(diag_jit.prs, diag.prs)
