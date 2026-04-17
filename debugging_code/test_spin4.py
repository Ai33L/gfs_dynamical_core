import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import s2fft
import numpy as np

L = 8
flm = jnp.zeros((L, 2*L-1), dtype=jnp.complex128)
flm = flm.at[1, L-1].set(1.0) # Y_1,0 -> f \propto cos(theta)

# gradient of Y_1,0: v_theta = -sin(theta), v_phi = 0
l_arr = np.arange(L)
l_factor = np.sqrt(l_arr * (l_arr + 1))
F1_lm = -l_factor[:, None] * flm

f_spin1 = s2fft.inverse_jax(F1_lm, L, spin=1)

v_theta = f_spin1.real
v_phi = f_spin1.imag

print("Max v_phi:", jnp.max(jnp.abs(v_phi)))
print("v_theta at equator:", v_theta[L//2, :])
