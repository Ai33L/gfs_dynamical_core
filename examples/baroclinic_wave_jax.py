import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

from datetime import timedelta

import climt
import matplotlib

matplotlib.use("Agg")
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
    # Remove the zonal mean of surface pressure
    ps_anomaly = ps - np.mean(ps, axis=1, keepdims=True)

    lon = state["longitude"].values
    lat = state["latitude"].values
    c1 = ax1.contourf(
        lon,
        lat,
        ps_anomaly,
        levels=np.linspace(-30, 30, 21),
        cmap="RdBu_r",
        extend="both",
    )
    fig.colorbar(c1, ax=ax1, label="Surface Pressure Anomaly (hPa)")
    ax1.set_title(f"Surface Pressure Anomaly at {state['time']}")
    ax1.set_ylabel("Latitude")

    # Temperature at level 18 (near surface)
    ax2 = fig.add_subplot(2, 1, 2)
    temp = state["air_temperature"].values[18, :, :]
    # Remove the zonal mean of temperature
    temp_anomaly = temp - np.mean(temp, axis=1, keepdims=True)

    c2 = ax2.contourf(
        lon,
        lat,
        temp_anomaly,
        levels=np.linspace(-20, 20, 21),
        cmap="RdBu_r",
        extend="both",
    )
    fig.colorbar(c2, ax=ax2, label="Temperature Anomaly (K)")
    ax2.set_title("Temperature Anomaly near surface (~950 hPa)")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")

    plt.tight_layout()


# 5. Simulation Loop
timestep = timedelta(minutes=5)
print(f"Starting simulation with dt = {timestep}")

# Use PlotFunctionMonitor to save frames (or just save final)
# monitor = PlotFunctionMonitor(plot_baroclinic_wave)

try:
    for i in range(144 * 20):  # Run for 20 days (144 steps/day at 10 min)
        diag, output = dycore(my_state, timestep=timestep)
        my_state.update(output)
        my_state["time"] += timestep

        ps_vals = my_state["surface_air_pressure"].values
        ps_min = np.min(ps_vals)
        ps_max = np.max(ps_vals)

        if (i + 1) % 6 == 0:  # Every hour (6 steps of 10 mins)
            hour = (i + 1) // 6
            print(
                f"Hour {hour:2d}, Step {i + 1:3d}, Time: {my_state['time']}, PS range: {ps_min / 100.0:.1f} - {ps_max / 100.0:.1f} hPa"
            )
            fig = plt.figure()
            plot_baroclinic_wave(fig, my_state)
            plt.savefig(f"baroclinic_wave_jax_hour_{hour:02d}.png")
            plt.close(fig)

        if ps_max > 2e5 or np.isnan(ps_max):
            print(
                f"Simulation unstable at step {i + 1}! PS range: {ps_min / 100.0:.1f} - {ps_max / 100.0:.1f} hPa"
            )
            break

    print("Simulation complete.")

except Exception as e:
    import traceback

    traceback.print_exc()
    print(f"Error during simulation: {e}")

# 6. Final Visualization
print("Generating final plot...")
fig = plt.figure()
plot_baroclinic_wave(fig, my_state)
plt.savefig("baroclinic_wave_jax_final.png")
print("Plot saved to baroclinic_wave_jax_final.png")
