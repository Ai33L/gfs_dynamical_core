import jax
import jax.numpy as jnp
import s2fft

jax.config.update("jax_enable_x64", True)
temp = jnp.full((32, 64), 290.0)

# s2fft only sees the first 63 longitudes because L=32 and sampling='gl' -> 32, 63
temp_sliced = temp[:, :63]
flm = s2fft.forward_jax(temp_sliced, L=32, sampling='gl')
temp_rec = s2fft.inverse_jax(flm, L=32, sampling='gl')

print("flm[0, 31]:", flm[0, 31]) # l=0, m=0
print("temp_rec[0,0]:", temp_rec[0,0])
