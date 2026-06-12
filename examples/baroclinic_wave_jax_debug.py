import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

import climt
from sympl import set_constant
import numpy as np
from datetime import timedelta
from gfs_dynamical_core.component_jax import GFSDynamicsJAX

# 1. Setup Constants and Grid
set_constant('reference_air_pressure', value=1e5, units='Pa')

L = 64
n_lon = 2 * L - 1 # 127
n_lat = L # 64
n_lev = 20

print(f"Initializing Grid: {n_lon}x{n_lat}x{n_lev} (L={L})")
grid = climt.get_grid(nx=n_lon, ny=n_lat, nz=n_lev)

# 2. Initialize Components
dycore = GFSDynamicsJAX()
dcmip = climt.DcmipInitialConditions(add_perturbation=True)

# 3. Setup Initial State
print("Setting up initial state...")
my_state = climt.get_default_state([dycore], grid_state=grid)

out = dcmip(my_state)
my_state.update(out)

# 5. Simulation Loop
timestep = timedelta(minutes=5)
n_steps = 100
print(f"Starting JAX simulation with dt = {timestep} for {n_steps} steps")

try:
    for i in range(n_steps):
        diag, output = dycore(my_state, timestep=timestep)
        my_state.update(output)
        my_state["time"] += timestep
        
        if i % 10 == 0:
            ps_min = np.min(my_state["surface_air_pressure"].values)
            ps_max = np.max(my_state["surface_air_pressure"].values)
            print(f"Step {i:3d}, Time: {my_state['time']}, PS range: {ps_min/100.0:.1f} - {ps_max/100.0:.1f} hPa")
            if np.isnan(ps_max) or ps_max > 2000.0 * 100.0:
                print("Exploded!")
                break

    print("JAX simulation complete.")

except Exception as e:
    print(f"Error during simulation: {e}")
