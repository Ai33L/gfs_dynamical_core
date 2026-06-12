import numpy as np
import os

L = 64
n_lon, n_lat, n_lev = 127, 64, 20
n_trac = 1
ndimspec = (63+1)*(63+2)//2 # 2080

def read_debug_file_spec(filename, is_jax):
    with open(filename, 'rb') as f:
        data = f.read()
    
    # Skip grid fields
    grid_size = (n_lon*n_lat*n_lev*3 + n_lon*n_lat + n_lon*n_lat*n_lev*n_trac) * 8
    offset = grid_size
    
    if is_jax:
        # JAX saves as (levels, L, 2L-1) or (L, 2L-1)
        # vorticity, divergence, temperature, lnps
        def get_next(shape):
            nonlocal offset
            size = np.prod(shape)
            arr = np.frombuffer(data, dtype=np.complex128, count=size, offset=offset)
            offset += size * 16
            return arr.reshape(shape)
        
        vort = get_next((n_lev, L, 2*L-1))
        div = get_next((n_lev, L, 2*L-1))
        temp = get_next((n_lev, L, 2*L-1))
        lnps = get_next((L, 2*L-1))
        return {'vort': vort, 'div': div, 'temp': temp, 'lnps': lnps}
    else:
        # Fortran saves as (ndimspec, nlevs)
        def get_next(shape):
            nonlocal offset
            size = np.prod(shape)
            arr = np.frombuffer(data, dtype=np.complex128, count=size, offset=offset)
            offset += size * 16
            return arr.reshape(shape, order='F')
        
        vort = get_next((ndimspec, n_lev))
        div = get_next((ndimspec, n_lev))
        temp = get_next((ndimspec, n_lev))
        lnps = get_next((ndimspec,))
        return {'vort': vort, 'div': div, 'temp': temp, 'lnps': lnps}

step = 1
stage = 0
f_data = read_debug_file_spec(f'debug_data/fortran_step_{step}_stage_{stage}.bin', False)
j_data = read_debug_file_spec(f'debug_data/jax_step_{step}_stage_{stage}.bin', True)

print(f"--- Comparing Spectral Magnitudes (Step {step}, Stage {stage}) ---")
for var in f_data:
    f_mag = np.sqrt(np.sum(np.abs(f_data[var])**2))
    j_mag = np.sqrt(np.sum(np.abs(j_data[var])**2))
    print(f"  {var:4s}: mag_f={f_mag:.2e}, mag_j={j_mag:.2e}, ratio={j_mag/(f_mag+1e-20):.4f}")

