import jax
import jax.numpy as jnp
import s2fft

jax.config.update("jax_enable_x64", True)

# Fortran setup
n_lat = 32
n_lon = 64
T = 19 # Triangular truncation

# For s2fft, we need an L that can support a 32x64 grid.
# Actually, s2fft 'gl' sampling: n_lat = L, n_lon = 2L - 1.
# So if L=32, n_lat=32, n_lon=63.
# Wait, s2fft allows n_lon to be arbitrary if we don't use the standard inverse, or maybe it doesn't?
# Let's check s2fft documentation.
