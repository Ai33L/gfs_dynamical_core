import numpy as np
from compare_debug_data import read_debug_file

L = 64
n_lon, n_lat, n_lev = 127, 64, 20
n_trac = 1

f_data = read_debug_file('debug_data/fortran_step_1_stage_0.bin', n_lon, n_lat, n_lev, n_trac)
j_data = read_debug_file('debug_data/jax_step_1_stage_0.bin', n_lon, n_lat, n_lev, n_trac)

print("--- U Wind Profile at (lon=0, lat=32) ---")
print("Level | Fortran | JAX")
for k in range(n_lev):
    print(f"{k:5d} | {f_data['u'][0, 32, k]:7.2f} | {j_data['u'][0, 32, k]:7.2f}")

print("\n--- Temperature Profile at (lon=0, lat=32) ---")
print("Level | Fortran | JAX")
for k in range(n_lev):
    print(f"{k:5d} | {f_data['t'][0, 32, k]:7.2f} | {j_data['t'][0, 32, k]:7.2f}")
