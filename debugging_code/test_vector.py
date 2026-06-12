import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import s2fft
import numpy as np

L = 8
# Let's create a streamfunction psi = Y_1,0
# => v_theta = 0, v_phi = -d_theta psi = sin(theta)
# Divergence = 0, Vorticity = laplacian psi = -2 psi
flm_psi = jnp.zeros((L, 2*L-1), dtype=jnp.complex128)
flm_psi = flm_psi.at[1, L-1].set(1.0) # Y_1,0

flm_zeta = -1.0 * 2.0 * flm_psi
flm_div = jnp.zeros_like(flm_psi)

# Form the spin-1 coefficients
# v_theta + i v_phi = - sum (chi_lm + i psi_lm) * sqrt(l(l+1)) * _1 Y_lm
# Here chi = 0, so it's - i psi_lm * sqrt(l(l+1))
clm_spin1 = jnp.zeros((L, 2*L-1), dtype=jnp.complex128)
l = 1
clm_spin1 = clm_spin1.at[1, L-1].set( -1j * 1.0 * np.sqrt(l*(l+1)) )

f_spin1 = s2fft.inverse_jax(clm_spin1, L, spin=1)

print("Max v_theta:", jnp.max(jnp.abs(f_spin1.real)))
print("Max v_phi:", jnp.max(jnp.abs(f_spin1.imag)))
print("v_phi values around equator:")
print(f_spin1.imag[L//2, :])
