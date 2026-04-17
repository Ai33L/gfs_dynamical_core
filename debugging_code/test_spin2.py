import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import s2fft
import numpy as np

L = 4
flm = jnp.zeros((L, 2*L-1), dtype=jnp.complex128)
flm = flm.at[1, L].set(1.0) # l=1, m=1

# compute v_theta + i v_phi using s2fft
# c_lm = -sqrt(l(l+1)) * (chi_lm - i psi_lm)
# chi = Y_11, psi = 0
# chi_lm = 1 for (1,1), else 0.
clm = jnp.zeros((L, 2*L-1), dtype=jnp.complex128)
l = 1
clm = clm.at[1, L].set( -np.sqrt(l*(l+1)) * 1.0 )

f_spin1 = s2fft.inverse_jax(clm, L, spin=1)

# we can compare it with analytical gradient of f
f = s2fft.inverse_jax(flm, L)

print("spin1 real (v_theta?) at equator:", f_spin1.real[2, :])
print("spin1 imag (v_phi?) at equator:", f_spin1.imag[2, :])

