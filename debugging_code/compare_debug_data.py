import numpy as np
import os

def read_debug_file(filename, n_lon, n_lat, n_lev, n_trac):
    with open(filename, 'rb') as f:
        data = f.read()
    
    offset = 0
    
    def get_next(shape):
        nonlocal offset
        size = np.prod(shape)
        arr = np.frombuffer(data, dtype=np.float64, count=size, offset=offset)
        offset += size * 8
        return arr.reshape(shape, order='F')

    u = get_next((n_lon, n_lat, n_lev))
    v = get_next((n_lon, n_lat, n_lev))
    t = get_next((n_lon, n_lat, n_lev))
    ps = get_next((n_lon, n_lat))
    q = get_next((n_lon, n_lat, n_lev, n_trac))
    
    return {'u': u, 'v': v, 't': t, 'ps': ps, 'q': q}

L = 64
n_lon, n_lat, n_lev = 127, 64, 20
n_trac = 1 # GFS default

step = 10
for stage in range(4):
    f_file = f'debug_data/fortran_step_{step}_stage_{stage}.bin'
    j_file = f'debug_data/jax_step_{step}_stage_{stage}.bin'
    
    if not os.path.exists(f_file) or not os.path.exists(j_file):
        print(f"Skipping Step {step} Stage {stage} - missing files")
        continue
        
    print(f"\n--- Comparing Step {step} Stage {stage} ---")
    f_data = read_debug_file(f_file, n_lon, n_lat, n_lev, n_trac)
    j_data = read_debug_file(j_file, n_lon, n_lat, n_lev, n_trac)
    
    for var in f_data:
        diff = np.abs(f_data[var] - j_data[var])
        max_diff = np.max(diff)
        mean_val = np.mean(np.abs(f_data[var]))
        rel_diff = max_diff / (mean_val + 1e-10)
        print(f"  {var:2s}: max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}")

