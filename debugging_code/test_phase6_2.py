import jax
import jax.numpy as jnp
import s2fft

jax.config.update("jax_enable_x64", True)

temp = jnp.full((32, 64), 290.0)

# Try L=32, nside=... 'mw'
try:
    flm = s2fft.forward_jax(temp, L=32, sampling='mw')
    print("mw shape:", flm.shape)
except Exception as e:
    print("mw error:", e)

# What if we just pass a 32x64 array to gl?
try:
    flm = s2fft.forward_jax(temp, L=32, sampling='gl')
    print("gl 32x64 shape:", flm.shape)
except Exception as e:
    print("gl error:", e)
