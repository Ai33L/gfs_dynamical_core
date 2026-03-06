from jax import config

config.update("jax_enable_x64", True)

import os

os.environ["JAX_PLATFORMS"] = "cpu"

from datetime import timedelta

import climt
import numpy as np
import pytest

from gfs_dynamical_core.component_jax import GFSDynamicsJAX


def test_jax_component_init():
    component = GFSDynamicsJAX()
    assert component is not None


def test_jax_component_call():
    # 1. Setup grid — use native s2fft sizes for GL sampling:
    #    n_lat = L, n_lon = 2*L - 1.  No resampling needed.
    L = 32
    n_lat, n_lon, n_lev = L, 2 * L - 1, 10
    grid = climt.get_grid(nx=n_lon, ny=n_lat, nz=n_lev)

    # 2. Get default state
    state = climt.get_default_state([GFSDynamicsJAX], grid_state=grid)

    # 3. Initialize component
    component = GFSDynamicsJAX()

    # 4. Call component
    timestep = timedelta(seconds=1200)
    diagnostics, new_state = component(state, timestep=timestep)

    # 5. Verify output
    assert isinstance(diagnostics, dict)
    assert "air_temperature" in new_state
    assert "eastward_wind" in new_state
    assert "northward_wind" in new_state
    assert "surface_air_pressure" in new_state

    # Check shapes
    assert new_state["air_temperature"].shape == state["air_temperature"].shape
