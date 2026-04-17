import climt
import numpy as np
from gfs_dynamical_core import GFSDynamicalCore

n_lat, n_lon, n_lev = 32, 64, 10
grid = climt.get_grid(nx=n_lon, ny=n_lat, nz=n_lev)
fortran_comp = GFSDynamicalCore()
state = climt.get_default_state([fortran_comp], grid_state=grid)

print("Initial temp sample:", state['air_temperature'].values.flatten()[:5])
print("Initial ps sample:", state['surface_air_pressure'].values.flatten()[:5])
