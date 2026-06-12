import jax
import jax.numpy as jnp
import s2fft

jax.config.update("jax_enable_x64", True)
import numpy as np

temp = jnp.full((32, 64), 290.0)

x_k = jnp.fft.rfft(temp, axis=-1)
x_k_63 = x_k[..., :32]
x_63 = jnp.fft.irfft(x_k_63, n=63, axis=-1)
x_63 *= 63.0 / 64.0

print(x_63[0,0])

# Back to 64
x_k_rec = jnp.fft.rfft(x_63, axis=-1)
# 63 points gives 32 frequencies
x_k_rec_64 = jnp.pad(x_k_rec, ((0,0), (0, 1))) # pad one zero
x_64_rec = jnp.fft.irfft(x_k_rec_64, n=64, axis=-1)
x_64_rec *= 64.0 / 63.0

print(x_64_rec[0,0])

