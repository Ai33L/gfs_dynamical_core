import jax
import jax.numpy as jnp
import s2fft

jax.config.update("jax_enable_x64", True)
temp = jnp.full((32, 64), 290.0)

# passing (32, 64)
try:
    flm = s2fft.forward_jax(temp, L=32, sampling='gl')
    print("flm shape:", flm.shape)
    grid_rec = s2fft.inverse_jax(flm, L=32, sampling='gl')
    print("grid_rec shape:", grid_rec.shape)
    print("grid_rec value:", grid_rec[0,0])
except Exception as e:
    print("Error:", e)
