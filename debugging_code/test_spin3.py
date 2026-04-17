import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import s2fft
import numpy as np

L = 8

# Create known divergence and vorticity
D_lm = jnp.zeros((L, 2*L-1), dtype=jnp.complex128)
zeta_lm = jnp.zeros((L, 2*L-1), dtype=jnp.complex128)

D_lm = D_lm.at[1, L-1].set(1.0) # D = Y_1,0
zeta_lm = zeta_lm.at[2, L].set(1.0) # zeta = Y_2,1

# Then chi = -D / l(l+1), psi = -zeta / l(l+1)
chi_lm = jnp.zeros_like(D_lm)
psi_lm = jnp.zeros_like(zeta_lm)
chi_lm = chi_lm.at[1, L-1].set(-1.0 / 2.0)
psi_lm = psi_lm.at[2, L].set(-1.0 / 6.0)

# 1F_lm = -sqrt(l(l+1)) * (chi_lm + i psi_lm)
F1_lm = jnp.zeros_like(D_lm)
F1_lm = F1_lm.at[1, L-1].set(-np.sqrt(2.0) * chi_lm[1, L-1])
F1_lm = F1_lm.at[2, L].set(-np.sqrt(6.0) * 1j * psi_lm[2, L])

# Convert to grid
# f_spin1 = v_theta + i v_phi
f_spin1 = s2fft.inverse_jax(F1_lm, L, spin=1)

# Now imagine we have v_theta and v_phi, and we want to recover D and zeta
# 1F_lm_rec = forward(v_theta + i v_phi, spin=1)
F1_lm_rec = s2fft.forward_jax(f_spin1, L, spin=1)

# D_rec + i zeta_rec = sqrt(l(l+1)) * 1F_lm_rec
l_arr = np.arange(L)
l_factor = np.sqrt(l_arr * (l_arr + 1))
D_plus_i_zeta = F1_lm_rec * l_factor[:, None]

D_rec = D_plus_i_zeta.real
zeta_rec = D_plus_i_zeta.imag

print("Max error in D:", jnp.max(jnp.abs(D_rec - D_lm.real)))
print("Max error in zeta:", jnp.max(jnp.abs(zeta_rec - zeta_lm.real)))

