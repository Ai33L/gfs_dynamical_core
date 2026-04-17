"""Diagnose spectral energy growth mode by mode.

Runs the JAX dycore for a short integration and prints the spectral power
spectrum of divergence at selected timesteps, to identify which modes are
growing fastest.
"""
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

from datetime import timedelta

import climt
import jax.numpy as jnp
import numpy as np
from sympl import set_constant

from gfs_dynamical_core.component_jax import GFSDynamicsJAX

set_constant("reference_air_pressure", value=1e5, units="Pa")

L = 64
n_lon = 2 * L - 1
n_lat = L
n_lev = 20
ntrunc = int(n_lon / 3 - 2)  # 40

grid = climt.get_grid(nx=n_lon, ny=n_lat, nz=n_lev)

dycore = GFSDynamicsJAX(adiabatic=True)  # No fixer
dcmip = climt.DcmipInitialConditions(add_perturbation=True)

my_state = climt.get_default_state([dycore], grid_state=grid)
out = dcmip(my_state)
my_state.update(out)

timestep = timedelta(minutes=10)
dt = timestep.total_seconds()

print(f"L={L}, ntrunc={ntrunc}, dt={dt}s")
print(f"Grid: {n_lon}x{n_lat}x{n_lev}")
print()

# Header
print(f"{'Step':>5s}  {'Hour':>5s}  {'PS range (hPa)':>16s}  "
      f"{'div E(l<=T)':>12s}  {'div E(l>T)':>12s}  "
      f"{'vrt E(l<=T)':>12s}  {'vrt E(l>T)':>12s}  "
      f"{'tmp E(l<=T)':>12s}  {'tmp E(l>T)':>12s}  "
      f"{'lnps E(l<=T)':>13s}  {'lnps E(l>T)':>12s}")

for i in range(144 * 10):  # 10 days
    diag, output = dycore(my_state, timestep=timestep)
    my_state.update(output)
    my_state["time"] += timestep

    if (i + 1) % 6 == 0:  # Every hour
        hour = (i + 1) // 6
        ps = my_state["surface_air_pressure"].values
        ps_min, ps_max = np.min(ps) / 100, np.max(ps) / 100

        # Access the cached spectral state
        spec = dycore._spec_state
        if spec is None:
            continue

        # Compute spectral power per mode l
        def spectral_power(flm, within_trunc=True):
            """Sum |flm|^2 for modes l <= ntrunc or l > ntrunc."""
            arr = np.array(flm)
            # flm shape: (..., L, 2L-1)
            # Sum over all dims except the last two (l, m)
            power = np.sum(np.abs(arr) ** 2, axis=tuple(range(arr.ndim - 2)))
            # power shape: (L, 2L-1)
            # Sum over m for each l
            power_l = np.sum(power, axis=1)  # shape (L,)
            if within_trunc:
                return np.sum(power_l[:ntrunc + 1])
            else:
                return np.sum(power_l[ntrunc + 1:])

        div_in = spectral_power(spec.divergence, True)
        div_out = spectral_power(spec.divergence, False)
        vrt_in = spectral_power(spec.vorticity, True)
        vrt_out = spectral_power(spec.vorticity, False)
        tmp_in = spectral_power(spec.temperature, True)
        tmp_out = spectral_power(spec.temperature, False)
        lnps_in = spectral_power(spec.log_surface_pressure, True)
        lnps_out = spectral_power(spec.log_surface_pressure, False)

        print(f"{i+1:5d}  {hour:5d}  {ps_min:7.1f}-{ps_max:7.1f}  "
              f"{div_in:12.4e}  {div_out:12.4e}  "
              f"{vrt_in:12.4e}  {vrt_out:12.4e}  "
              f"{tmp_in:12.4e}  {tmp_out:12.4e}  "
              f"{lnps_in:13.4e}  {lnps_out:12.4e}")

        if ps_max > 2e5 / 100 or np.isnan(ps_max):
            print("UNSTABLE — stopping")
            break

        # Also print per-l spectrum every 24 hours
        if hour % 24 == 0:
            div_arr = np.array(spec.divergence)
            power_per_l = np.sum(np.abs(div_arr) ** 2,
                                 axis=tuple(range(div_arr.ndim - 2)))
            power_l = np.sum(power_per_l, axis=1)
            print(f"  Div power spectrum at hour {hour}:")
            for ll in range(0, L, 5):
                end = min(ll + 5, L)
                vals = "  ".join(f"l={l}: {power_l[l]:.2e}" for l in range(ll, end))
                print(f"    {vals}")
            print()
