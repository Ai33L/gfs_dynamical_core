import jax
import jax.numpy as jnp
import s2fft

jax.config.update("jax_enable_x64", True)
import numpy as np

temp = np.full((32, 64), 290.0)

def fwd_bwd(L, pad=False):
    if pad:
        # Pad longitudes by copying the first element to the end?
        # No, the grid is periodic over 2pi.
        pass

    # For now, let's just see how to get s2fft to return 32x64.
    # L=32 means 32 latitudes. For gl, it returns 32 x 63.
    flm = s2fft.forward_jax(jnp.array(temp[:, :63]), L=32, sampling='gl')
    
    # We want to keep only T=19.
    T = 19
    l_arr = np.arange(32)
    m_arr = np.arange(-31, 32)
    l_grid, m_grid = np.meshgrid(l_arr, m_arr, indexing='ij')
    
    # mask where l > T or |m| > T
    mask = (l_grid <= T) & (np.abs(m_grid) <= T)
    
    # s2fft returns (L, 2L-1), but the ordering is L, 2L-1.
    # m index is -L+1 to L-1
    flm_trunc = jnp.where(mask, flm, 0.0)
    
    rec = s2fft.inverse_jax(flm_trunc, L=32, sampling='gl')
    print("Rec shape:", rec.shape)
    print("Rec value:", rec[0,0])

fwd_bwd(32)
