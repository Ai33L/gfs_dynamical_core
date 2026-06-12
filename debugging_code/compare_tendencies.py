import numpy as np
from compare_debug_data import read_debug_file

L = 64
n_lon, n_lat, n_lev = 127, 64, 20
n_trac = 1
dt = 300.0

def get_tend(step, stage_start, stage_end):
    f0 = read_debug_file(f'debug_data/fortran_step_{step}_stage_{stage_start}.bin', n_lon, n_lat, n_lev, n_trac)
    f1 = read_debug_file(f'debug_data/fortran_step_{step}_stage_{stage_end}.bin', n_lon, n_lat, n_lev, n_trac)
    j0 = read_debug_file(f'debug_data/jax_step_{step}_stage_{stage_start}.bin', n_lon, n_lat, n_lev, n_trac)
    j1 = read_debug_file(f'debug_data/jax_step_{step}_stage_{stage_end}.bin', n_lon, n_lat, n_lev, n_trac)
    
    # RK Stage 1 tendency is (stage1 - stage0) / (a21 * dt)
    # For a21 = 1.0
    
    f_tend = {var: (f1[var] - f0[var]) / dt for var in f0}
    j_tend = {var: (j1[var] - j0[var]) / dt for var in j0}
    
    return f_tend, j_tend

step = 1
f_tend, j_tend = get_tend(step, 0, 1)

print(f"--- Comparing Tendencies (Step {step}, Stage 0->1) ---")
for var in f_tend:
    # Check if they have same sign on average
    dot_prod = np.sum(f_tend[var] * j_tend[var])
    f_norm = np.sum(f_tend[var]**2)
    j_norm = np.sum(j_tend[var]**2)
    cos_sim = dot_prod / (np.sqrt(f_norm * j_norm) + 1e-20)
    
    print(f"  {var:2s}: cos_sim={cos_sim:7.4f}, max_f={np.max(np.abs(f_tend[var])):.2e}, max_j={np.max(np.abs(j_tend[var])):.2e}")

