import climt
from sympl import set_constant
import numpy as np
from datetime import timedelta
from gfs_dynamical_core import GFSDynamicalCore
import os

# 1. Setup Constants and Grid
set_constant('reference_air_pressure', value=1e5, units='Pa')

L = 64
n_lon = 2 * L - 1 # 127
n_lat = L # 64
n_lev = 20

print(f"Initializing Grid: {n_lon}x{n_lat}x{n_lev} (L={L})")
grid = climt.get_grid(nx=n_lon, ny=n_lat, nz=n_lev)

# 2. Initialize Components
# Use the FORTRAN version of the dycore
dycore = GFSDynamicalCore()
# DCMIP Test Case 4.1: Baroclinic Wave
dcmip = climt.DcmipInitialConditions(add_perturbation=True)

# 3. Setup Initial State
print("Setting up initial state...")
my_state = climt.get_default_state([dycore], grid_state=grid)

# Apply DCMIP initial conditions
out = dcmip(my_state)
my_state.update(out)

# 5. Simulation Loop
timestep = timedelta(minutes=5)
n_steps = 100
print(f"Starting FORTRAN simulation with dt = {timestep} for {n_steps} steps")

if not os.path.exists('debug_data'):
    os.makedirs('debug_data')

try:
    for i in range(n_steps):
        diag, output = dycore(my_state, timestep=timestep)
        my_state.update(output)
        my_state["time"] += timestep
        
        if i % 10 == 0:
            ps_min = np.min(my_state["surface_air_pressure"].values)
            ps_max = np.max(my_state["surface_air_pressure"].values)
            print(f"Step {i:3d}, Time: {my_state['time']}, PS range: {ps_min/100.0:.1f} - {ps_max/100.0:.1f} hPa")

    print("FORTRAN simulation complete.")

except Exception as e:
    print(f"Error during simulation: {e}")
