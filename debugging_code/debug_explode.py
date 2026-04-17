import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)
import numpy as np
from datetime import timedelta
import climt
from gfs_dynamical_core.component_jax import GFSDynamicsJAX

L = 20
n_lat, n_lon, n_lev = L, 2 * L - 1, 10
dt = timedelta(seconds=600)

grid = climt.get_grid(nx=n_lon, ny=n_lat, nz=n_lev)
jax_comp = GFSDynamicsJAX()
state = climt.get_default_state([jax_comp], grid_state=grid)

lat_rad = np.deg2rad(state["latitude"].values)
lon_rad = np.deg2rad(state["longitude"].values)
u_val = 20.0 * np.cos(lat_rad) * np.ones_like(lon_rad)
v_val = 5.0 * np.cos(lat_rad) * np.sin(lon_rad)
t_val = 300.0 - 40.0 * np.sin(lat_rad) ** 2

state["eastward_wind"].values[:] = u_val[None, :, :]
state["northward_wind"].values[:] = v_val[None, :, :]
state["air_temperature"].values[:] = t_val[None, :, :]

_, j_new = jax_comp(state, timestep=dt)
for k, v in j_new.items():
    print(f"{k}: max={np.max(np.abs(v.values)):.3e}, mean={np.mean(np.abs(v.values)):.3e}")
