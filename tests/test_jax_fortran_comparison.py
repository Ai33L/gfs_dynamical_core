from jax import config
config.update("jax_enable_x64", True)

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pytest
import climt
import copy
from gfs_dynamical_core import GFSDynamicalCore
from gfs_dynamical_core.component_jax import GFSDynamicsJAX
from datetime import timedelta

def test_jax_fortran_comparison():
    # 1. Setup common grid
    n_lat, n_lon, n_lev = 32, 64, 10
    grid = climt.get_grid(nx=n_lon, ny=n_lat, nz=n_lev)
    
    # 2. Initialize Fortran component to get default state and coordinate arrays
    fortran_comp = GFSDynamicalCore()
    state = climt.get_default_state([fortran_comp], grid_state=grid)
    
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
        
        diff = np.abs(f_val - j_val)
        print(f"{key}: max_diff={np.max(diff):.2e}, mean_diff={np.mean(diff):.2e}")
