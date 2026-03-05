import jax
import jax.numpy as jnp
from flax import struct

@struct.dataclass
class DynamicsConfig:
    """Configuration and constants for GFS dynamics."""
    ak: jnp.ndarray # (n_lev + 1,)
    bk: jnp.ndarray # (n_lev + 1,)
    rk: float # R / Cp
    toa_pressure: float # Top of atmosphere pressure
    radius: float
    omega: float
    g: float
    rd: float
    rv: float
    cp: float
    cvap: float

@struct.dataclass
class PressureDiagnostics:
    """Pressure-related diagnostic fields."""
    ps: jnp.ndarray # (n_lat, n_lon)
    pk: jnp.ndarray # (n_lev + 1, n_lat, n_lon) - interface pressure
    dp: jnp.ndarray # (n_lev, n_lat, n_lon) - layer pressure thickness
    prs: jnp.ndarray # (n_lev, n_lat, n_lon) - layer mean pressure
    alfa: jnp.ndarray # (n_lev, n_lat, n_lon)
    rlnp: jnp.ndarray # (n_lev, n_lat, n_lon)

def compute_pressure_diagnostics(log_ps: jnp.ndarray, config: DynamicsConfig) -> PressureDiagnostics:
    """
    Computes pressure-related diagnostics from log surface pressure.
    
    Equivalent to Fortran's calc_pressdata.
    """
    n_lev = config.ak.shape[0] - 1
    
    # 1. Surface pressure
    ps = jnp.exp(log_ps)
    
    # 2. Interface pressures (pk)
    # ak and bk go top to bottom (index 0 is TOA, index n_lev is surface)
    # pk = ak + bk * (ps - toa_pressure)
    # We use vmap or broadcasting to compute for all levels
    pk = config.ak[:, None, None] + config.bk[:, None, None] * (ps[None, :, :] - config.toa_pressure)
    
    # 3. Pressure thickness (dp)
    # dp(k) = pk(k+1) - pk(k)
    dp = pk[1:] - pk[:-1]
    
    # 4. Layer mean pressure (prs)
    # GFS uses a specific formula for layer mean pressure
    rk = config.rk
    prs_raw = ((pk[1:]**(rk+1) - pk[:-1]**(rk+1)) / ((rk+1) * dp))**(1/rk)
    
    # Note: GFS snippet had: prs(:,:,nlevs-k+1) = ... (reversed index)
    # In JAX, we'll keep it consistent with pk indexing (top-to-bottom) for now
    # and reverse if needed during the final port.
    prs = prs_raw
    
    # 5. rlnp and alfa
    # rlnp = log(pk(k+1) / pk(k))
    # alfa = 1 - (pk(k) / dp(k)) * rlnp
    
    rlnp = jnp.log(pk[1:] / pk[:-1])
    alfa = 1.0 - (pk[:-1] / dp) * rlnp
    
    # Handle top level (k=0 in 0-indexing, which is k=1 in Fortran)
    # alfa[0] = log(2.0)
    alfa = alfa.at[0].set(jnp.log(2.0))
    # rlnp[0] is set to a dummy value in Fortran, we'll use 0.0 or nan
    rlnp = rlnp.at[0].set(0.0)
    
    return PressureDiagnostics(
        ps=ps,
        pk=pk,
        dp=dp,
        prs=prs,
        alfa=alfa,
        rlnp=rlnp
    )
