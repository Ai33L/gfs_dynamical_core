import jax.numpy as jnp
import numpy as np

# We can perform the FFT in the longitude direction manually, 
# and then use a Legendre transform for the latitude direction, 
# but s2fft already does both. The problem is s2fft strictly 
# takes N_lon = 2*L-1.
# GFS has N_lon = 64, N_lat = 32.
# So N_lon doesn't match 2*L-1 = 63.
# Let's verify if we can just pad the spatial grid and use L=33?
# L=33 -> N_lat = 33, N_lon = 65.
# But then N_lat doesn't match!
# GFS uses N_lat = 32. 
