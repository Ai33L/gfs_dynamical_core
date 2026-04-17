import jax
import jax.numpy as jnp
import s2fft

jax.config.update("jax_enable_x64", True)
import numpy as np

# We can manually resample 64 longitudes to 63 before transforming
def to_s2fft_grid(x_64):
    # x_64 has shape (..., n_lat, 64)
    # We want (..., n_lat, 63)
    # Just FFT in longitude, truncate high frequencies, and IFFT
    x_k = jnp.fft.rfft(x_64, axis=-1)
    
    # 64 points: k = 0, ..., 32
    # 63 points: k = 0, ..., 31
    x_k_63 = x_k[..., :32]
    
    x_63 = jnp.fft.irfft(x_k_63, n=63, axis=-1)
    
    # Actually wait. Does s2fft scale the longitudes evenly?
    # Yes, phis = 2*pi * p / (2L-1)
    # So if we scale by 63 / 64
    x_63 *= 63 / 64
    return x_63

