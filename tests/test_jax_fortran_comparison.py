from jax import config

config.update("jax_enable_x64", True)

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import copy
from datetime import timedelta

import climt
import numpy as np
import pytest

from gfs_dynamical_core import GFSDynamicalCore
from gfs_dynamical_core.component_jax import GFSDynamicsJAX


def test_jax_fortran_comparison():
    # 1. Setup common grid — use native s2fft sizes for GL sampling:
    #    n_lat = L, n_lon = 2*L - 1.
    L = 32
    n_lat, n_lon, n_lev = L, 2 * L - 1, 10
    grid = climt.get_grid(nx=n_lon, ny=n_lat, nz=n_lev)

    # 2. Initialize Fortran component to get default state and coordinate arrays
    fortran_comp = GFSDynamicalCore()
    state = climt.get_default_state([fortran_comp], grid_state=grid)

    # Inject non-zero winds and temperature gradients
    lat_rad = np.deg2rad(state["latitude"].values)
    lon_rad = np.deg2rad(state["longitude"].values)

    # A simple Rossby-Haurwitz-like wave or just some zonal jet
    u_wind = 20.0 * np.cos(lat_rad)
    v_wind = 10.0 * np.cos(lat_rad) * np.sin(lon_rad)
    temp_grad = 300.0 - 30.0 * np.sin(lat_rad) ** 2

    state["eastward_wind"].values[:] = u_wind[None, :, :]
    state["northward_wind"].values[:] = v_wind[None, :, :]
    state["air_temperature"].values[:] = temp_grad[None, :, :]

    # 3. Initialize JAX component
    jax_comp = GFSDynamicsJAX()

    # 4. Perform a single step with both
    timestep = timedelta(seconds=1200)

    # Copy state for JAX to ensure no shared memory side effects
    jax_state = copy.deepcopy(state)

    # Fortran step
    fortran_diag, fortran_new_state = fortran_comp(state, timestep=timestep)

    # JAX step
    jax_diag, jax_new_state = jax_comp(jax_state, timestep=timestep)

    # 5. Compare results
    common_keys = set(fortran_new_state.keys()) & set(jax_new_state.keys())

    print("\nComparing keys:", sorted(list(common_keys)))
    for key in sorted(common_keys):
        f_val = fortran_new_state[key].values
        j_val = jax_new_state[key].values

        print(f"JAX {key} sample:", j_val.flatten()[:5])
        print(f"FORTRAN {key} sample:", f_val.flatten()[:5])

        diff = np.abs(f_val - j_val)
        print(f"{key}: max_diff={np.max(diff):.2e}, mean_diff={np.mean(diff):.2e}")
