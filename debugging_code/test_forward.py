import jax
import jax.numpy as jnp
import s2fft

jax.config.update("jax_enable_x64", True)
L = 8
n_lat, n_lon = L, 2*L-1 # 'gl'
temp = jnp.full((n_lat, n_lon), 300.0)
flm = s2fft.forward_jax(temp, L)
print("flm max:", jnp.max(jnp.abs(flm)))
temp_rec = s2fft.inverse_jax(flm, L)
print("temp_rec:", temp_rec[0,0])
