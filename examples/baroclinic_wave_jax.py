import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

from datetime import timedelta

import climt
import matplotlib.pyplot as plt
import numpy as np
from sympl import PlotFunctionMonitor, set_constant

from gfs_dynamical_core.component_jax import GFSDynamicsJAX

# 1. Setup Constants and Grid
# We use L=64, which implies ny=64 and nx=127 for s2fft's native GL grid.
set_constant("reference_air_pressure", value=1e5, units="Pa")

L = 64
n_lon = 2 * L - 1  # 127
n_lat = L  # 64
n_lev = 20

print(f"Initializing Grid: {n_lon}x{n_lat}x{n_lev} (L={L})")
grid = climt.get_grid(nx=n_lon, ny=n_lat, nz=n_lev)

# 2. Initialize Components
# Use the JAX version of the dycore
dycore = GFSDynamicsJAX()
# DCMIP Test Case 4.1: Baroclinic Wave
dcmip = climt.DcmipInitialConditions(add_perturbation=True)

# 3. Setup Initial State
print("Setting up initial state...")
my_state = climt.get_default_state([dycore], grid_state=grid)

# Apply DCMIP initial conditions
out = dcmip(my_state)
my_state.update(out)


# 4. Define Plotting Function
def plot_baroclinic_wave(fig, state):
    fig.set_size_inches(12, 10)

    # Surface Pressure
    ax1 = fig.add_subplot(2, 1, 1)
    ps = state["surface_air_pressure"].values / 100.0  # hPa
    lon = state["longitude"].values
    lat = state["latitude"].values
    c1 = ax1.contourf(
        lon, lat, ps, levels=np.linspace(940, 1020, 21), cmap="viridis", extend="both"
    )
    fig.colorbar(c1, ax=ax1, label="Surface Pressure (hPa)")
    ax1.set_title(f"Surface Pressure at {state['time']}")
    ax1.set_ylabel("Latitude")

    # Temperature at level 18 (near surface)
    ax2 = fig.add_subplot(2, 1, 2)
    temp = state["air_temperature"].values[18, :, :]
    c2 = ax2.contourf(
        lon, lat, temp, levels=np.linspace(240, 310, 21), cmap="RdBu_r", extend="both"
    )
    fig.colorbar(c2, ax=ax2, label="Temperature (K)")
    ax2.set_title("Temperature near surface (~950 hPa)")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")

    plt.tight_layout()


# 5. Simulation Loop
timestep = timedelta(minutes=5)
print(f"Starting simulation with dt = {timestep}")

# Use PlotFunctionMonitor to save frames (or just save final)
# monitor = PlotFunctionMonitor(plot_baroclinic_wave)

try:
    for i in range(288):  # Run for 24 hours (288 steps at 5 min)
        # Check for stability
        ps_max = np.max(my_state["surface_air_pressure"].values)
        if ps_max > 2e5 or np.isnan(ps_max):
            print(f"Simulation unstable at step {i}! PS max: {ps_max}")
            break

        diag, output = dycore(my_state, timestep=timestep)
        my_state.update(output)
        my_state["time"] += timestep

        if i % 12 == 0:
            ps_min = np.min(my_state["surface_air_pressure"].values)
            print(
                f"Step {i:3d}, Time: {my_state['time']}, PS range: {ps_min / 100.0:.1f} - {ps_max / 100.0:.1f} hPa"
            )

    print("Simulation complete.")

except Exception as e:
    print(f"Error during simulation: {e}")

# 6. Final Visualization
print("Generating final plot...")
fig = plt.figure()
plot_baroclinic_wave(fig, my_state)
plt.savefig("baroclinic_wave_jax_final.png")
print("Plot saved to baroclinic_wave_jax_final.png")
