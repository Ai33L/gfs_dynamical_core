with open('gfs_dynamical_core/jax/stepper.py', 'r') as f:
    text = f.read()

import_code = '''
import os
import numpy as np

def dump_jax_intermediate(grid_state, step, stage):
    if step > 100:
        return
    filename = f'debug_data/jax_step_{step}_stage_{stage}.bin'
    
    # Transpose JAX (levels, lat, lon) to Fortran (lon, lat, levels)
    # Note: JAX tracers is (ntrac, levels, lat, lon) -> (lon, lat, levels, ntrac)
    u_f = np.array(grid_state.u).transpose(2, 1, 0)
    v_f = np.array(grid_state.v).transpose(2, 1, 0)
    t_f = np.array(grid_state.temperature).transpose(2, 1, 0)
    ps_f = np.array(jnp.exp(grid_state.log_surface_pressure)).transpose(1, 0)
    q_f = np.array(grid_state.tracers).transpose(3, 2, 1, 0)
    
    with open(filename, 'wb') as f:
        f.write(u_f.tobytes())
        f.write(v_f.tobytes())
        f.write(t_f.tobytes())
        f.write(ps_f.tobytes())
        f.write(q_f.tobytes())

'''

if 'def dump_jax_intermediate' not in text:
    text = text.replace('from .states import SpectralState, SpectralTendencies', 
                        'from .states import SpectralState, SpectralTendencies\\n' + import_code)

# Add step counter to advance
text = text.replace('def advance(', 'jax_debug_step = 0\\n\\ndef advance(')
# Wait, advance is a function, jax_debug_step should be global or passed.
# I'll use a global.

text = text.replace('    dt = stepper_config.dt', '    global jax_debug_step\\n    jax_debug_step += 1\\n    dt = stepper_config.dt')

# Add stage dumps
text = text.replace('    # --- Stage 1 ---', 
                    '    grid_init, _ = spectral_to_grid(state, trans_config)\\n    dump_jax_intermediate(grid_init, jax_debug_step, 0)\\n    # --- Stage 1 ---')

text = text.replace('    state1 = SpectralState(', 
                    '    grid1, _ = spectral_to_grid(state1_pre, trans_config)\\n    dump_jax_intermediate(grid1, jax_debug_step, 1)\\n    state1 = SpectralState(')
# Need to define state1_pre... 
# Actually I'll just dump after state1 is created.

text = text.replace('    state1 = SpectralState(', '    state1 = SpectralState(') # no change

# Better approach: find where state1, state2 are created and dump them.

with open('gfs_dynamical_core/jax/stepper.py', 'w') as f:
    f.write(text)
