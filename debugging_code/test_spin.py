import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import s2fft

L = 4
flm = jnp.zeros((L, 2*L-1), dtype=jnp.complex128)
flm = flm.at[1, L].set(1.0) # l=1, m=1

f = s2fft.inverse_jax(flm, L)

flm_spin1 = jnp.zeros((L, 2*L-1), dtype=jnp.complex128)
flm_spin1 = flm_spin1.at[1, L].set(1.0)
f_spin1 = s2fft.inverse_jax(flm_spin1, L, spin=1)

print("Scalar f (Y_1,1):")
print("real", f.real[1, 0:3])
print("imag", f.imag[1, 0:3])
print("Spin 1 f (real = v_theta, imag = v_phi??):")
print("real", f_spin1.real[1, 0:3])
print("imag", f_spin1.imag[1, 0:3])
