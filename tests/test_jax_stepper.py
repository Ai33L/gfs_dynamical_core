import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from gfs_dynamical_core.jax.dynamics import DynamicsConfig
from gfs_dynamical_core.jax.states import SpectralState, SpectralTendencies
from gfs_dynamical_core.jax.stepper import StepperConfig, advance
from gfs_dynamical_core.jax.transforms import TransformConfig, get_gaussian_latitudes


def get_mock_config(n_lev):
    ak = jnp.zeros(n_lev + 1)
    bk = jnp.linspace(1, 0, n_lev + 1)  # BTT: surface (k=0) bk=1, TOA bk=0
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


def _get_mock_state(n_lev, L):
    return SpectralState(
        vorticity=jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128),
        divergence=jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128),
        temperature=jnp.zeros((n_lev, L, 2 * L - 1), dtype=jnp.complex128),
        log_surface_pressure=jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
        .at[0, L - 1]
        .set(1.0),
        tracers=jnp.zeros((1, n_lev, L, 2 * L - 1), dtype=jnp.complex128),
    )


def test_advance_explicit():
    L = 4
    n_lev = 10

    dyn_config = get_mock_config(n_lev)
    trans_config = TransformConfig(L=L, radius=1.0)
    stepper_config = StepperConfig(dt=10.0, explicit=True)

    n_lat, n_lon = trans_config.n_lat, trans_config.n_lon

    state = _get_mock_state(n_lev, L)

    phis_grads = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))
    latitudes = get_gaussian_latitudes(L)

    new_state = advance(
        state, phis_grads, dyn_config, trans_config, stepper_config, latitudes
    )

    assert new_state.vorticity.shape == (n_lev, L, 2 * L - 1)
    assert not jnp.isnan(new_state.vorticity).any()
    assert not jnp.isnan(new_state.divergence).any()


def test_advance_implicit():
    L = 4
    n_lev = 10

    dyn_config = get_mock_config(n_lev)
    trans_config = TransformConfig(L=L, radius=1.0)

    n_lat, n_lon = trans_config.n_lat, trans_config.n_lon

    # Mock semi-implicit arrays
    amhyb = jnp.eye(n_lev)
    bmhyb = jnp.eye(n_lev)
    tor_hyb = jnp.ones((n_lev,))
    svhyb = jnp.ones((n_lev,))
    # d_hyb_m shape should be (3, L, n_lev, n_lev)
    # let's just make it identity over the last two dims
    d_hyb_m = jnp.tile(jnp.eye(n_lev)[None, None, :, :], (3, L, 1, 1))

    stepper_config = StepperConfig(
        dt=10.0,
        explicit=False,
        amhyb=amhyb,
        bmhyb=bmhyb,
        tor_hyb=tor_hyb,
        svhyb=svhyb,
        d_hyb_m=d_hyb_m,
    )

    state = _get_mock_state(n_lev, L)
    phis_grads = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))
    latitudes = get_gaussian_latitudes(L)

    new_state = advance(
        state, phis_grads, dyn_config, trans_config, stepper_config, latitudes
    )

    assert new_state.vorticity.shape == (n_lev, L, 2 * L - 1)
    assert not jnp.isnan(new_state.vorticity).any()
    assert not jnp.isnan(new_state.divergence).any()
