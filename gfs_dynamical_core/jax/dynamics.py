import jax
import jax.numpy as jnp
from flax import struct
from .states import GridState, GridGradients

@struct.dataclass
class DynamicsConfig:
    """Configuration and constants for GFS dynamics."""
    ak: jnp.ndarray # (n_lev + 1,)
    bk: jnp.ndarray # (n_lev + 1,)
    ck: jnp.ndarray # (n_lev,) - ak(k+1)*bk(k)-ak(k)*bk(k+1)
    dbk: jnp.ndarray # (n_lev,) - bk(k+1)-bk(k)
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

@struct.dataclass
class VerticalVelocities:
    """Vertical velocity diagnostic fields."""
    omega: jnp.ndarray # (n_lev, n_lat, n_lon) - d(ln p)/dt on layers
    etadot: jnp.ndarray # (n_lev + 1, n_lat, n_lon) - eta dot on interfaces
    d_log_ps_d_t: jnp.ndarray # (n_lat, n_lon) - surface pressure tendency

def compute_pressure_diagnostics(log_ps: jnp.ndarray, config: DynamicsConfig) -> PressureDiagnostics:
    """Computes pressure-related diagnostics."""
    ps = jnp.exp(log_ps)
    pk = config.ak[:, None, None] + config.bk[:, None, None] * (ps[None, :, :] - config.toa_pressure)
    dp = pk[1:] - pk[:-1]
    rk = config.rk
    prs = ((pk[1:]**(rk+1) - pk[:-1]**(rk+1)) / ((rk+1) * dp))**(1/rk)
    rlnp = jnp.log(pk[1:] / pk[:-1])
    alfa = 1.0 - (pk[:-1] / dp) * rlnp
    alfa = alfa.at[0].set(jnp.log(2.0))
    rlnp = rlnp.at[0].set(0.0)
    return PressureDiagnostics(ps=ps, pk=pk, dp=dp, prs=prs, alfa=alfa, rlnp=rlnp)

def compute_vertical_velocities(
    grid_state: GridState, 
    grid_grads: GridGradients, 
    press_diag: PressureDiagnostics,
    config: DynamicsConfig
) -> VerticalVelocities:
    """Computes omega, etadot, and log surface pressure tendency."""
    cg = grid_state.u * grid_grads.d_log_ps_d_lambda + grid_state.v * grid_grads.d_log_ps_d_phi
    div_dp = grid_state.divergence * press_diag.dp
    cg_dbk = cg * config.dbk[:, None, None]
    db = jnp.cumsum(div_dp, axis=0)
    cb = jnp.cumsum(cg_dbk, axis=0)
    d_log_ps_d_t = -db[-1] / press_diag.ps - cb[-1]
    cb_with_zero = jnp.concatenate([jnp.zeros_like(cb[:1]), cb], axis=0)
    db_with_zero = jnp.concatenate([jnp.zeros_like(db[:1]), db], axis=0)
    bk = config.bk[:, None, None]
    etadot = -press_diag.ps[None, :, :] * (bk * d_log_ps_d_t[None, :, :] + cb_with_zero) - db_with_zero
    ps = press_diag.ps[None, :, :]
    db_km1 = db_with_zero[:-1]
    cb_km1 = cb_with_zero[:-1]
    workb = press_diag.rlnp * (db_km1 + ps * cb_km1) + \
            press_diag.alfa * (div_dp + ps * cg * config.dbk[:, None, None])
    workc = ps * cg * (config.dbk[:, None, None] + config.ck[:, None, None] * press_diag.rlnp / press_diag.dp)
    omega = (workc - workb) / press_diag.dp
    return VerticalVelocities(omega=omega, etadot=etadot, d_log_ps_d_t=d_log_ps_d_t)

def compute_vertical_advection(data: jnp.ndarray, etadot: jnp.ndarray, dp: jnp.ndarray) -> jnp.ndarray:
    """
    Computes vertical advection using second-order centered differences.
    Equivalent to Fortran's getvadv.
    """
    n_lev = data.shape[0]
    
    # Fortran logic (bottom-to-top):
    # vadv(nlevs) = (0.5/dp(1)) * etadot(2) * (data(nlevs-1) - data(nlevs))
    # vadv(1) = (0.5/dp(nlevs)) * etadot(nlevs) * (data(1) - data(2))
    # vadv(k) = (0.5/dp(k)) * (etadot(k+1)*(data(k-1)-data(k)) + etadot(k)*(data(k)-data(k+1)))
    
    # Converting to top-to-bottom JAX indexing (k=0 is top):
    # data[0] is top layer, etadot[0] is TOA (0), etadot[1] is first interface.
    # etadot[n_lev] is bottom interface (0).
    
    # Boundary: Top layer (k=0)
    vadv_top = (0.5 / dp[0]) * etadot[1] * (data[1] - data[0])
    
    # Boundary: Bottom layer (k=n_lev-1)
    vadv_bot = (0.5 / dp[-1]) * etadot[-2] * (data[-2] - data[-1])
    
    # Interior
    # vadv[k] = (0.5/dp[k]) * (etadot[k+1]*(data[k+1]-data[k]) + etadot[k]*(data[k]-data[k-1]))
    # Note: data[k+1] is "below" in top-to-bottom.
    # Fortran: etadot(k+1)*(data(k-1)-data(k)) -- data(k-1) is "below" in bottom-to-top.
    # So (data[k+1]-data[k]) is correct for top-to-bottom.
    
    k = jnp.arange(1, n_lev - 1)
    vadv_mid = (0.5 / dp[1:-1]) * (
        etadot[2:-1] * (data[2:] - data[1:-1]) + 
        etadot[1:-2] * (data[1:-1] - data[:-2])
    )
    
    return jnp.concatenate([vadv_top[None], vadv_mid, vadv_bot[None]], axis=0)

def compute_pressure_gradient_force(
    virtual_temp: jnp.ndarray,
    grid_grads: GridGradients,
    press_diag: PressureDiagnostics,
    config: DynamicsConfig,
    surface_geopotential_grads: tuple[jnp.ndarray, jnp.ndarray]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes horizontal pressure gradient force components.
    Equivalent to Fortran's getpresgrad.
    """
    rd = config.rd
    ps = press_diag.ps
    dlnpsdx = grid_grads.d_log_ps_d_lambda
    dlnpsdy = grid_grads.d_log_ps_d_phi
    dphisdx, dphisdy = surface_geopotential_grads
    
    # cofa(k) = ak(k+1)*bk(k) - ak(k)*bk(k+1) / (ps * dbk(k)) -- wait, cofa is more complex
    # Let's re-read getpresgrad cofa/cofb logic.
    # cofb(:,:,1)=-(1./dpk(:,:,1))*(alfa(:,:,1)*dbk(1))
    # cofa(:,:,1)=-(1./(psg(:,:)*dpk(:,:,1)))*(alfa(:,:,1)*ck(1))
    
    # 표준화된 TOP-TO-BOTTOM indexing
    cofb = -(1.0 / press_diag.dp) * (press_diag.alfa * config.dbk[:, None, None])
    cofa = -(1.0 / (ps[None, :, :] * press_diag.dp)) * (press_diag.alfa * config.ck[:, None, None])
    
    # px3 terms (summation)
    # Fortran: px3u(:,:,nlevs-k)=px3u(:,:,nlevs+1-k)-rd*rlnp(:,:,nlevs+1-k)*dvirtempdx(:,:,k)
    # This is a cumulative sum from bottom to top.
    # In top-to-bottom, it's a cumulative sum from bottom upwards.
    
    term_x = rd * press_diag.rlnp * grid_grads.d_t_d_lambda # Note: d_t_d_lambda should be virtual temp grad
    term_y = rd * press_diag.rlnp * grid_grads.d_t_d_phi
    
    # Flip, cumsum, flip back to get bottom-up sum
    px3u = jnp.flip(jnp.cumsum(jnp.flip(term_x, axis=0), axis=0), axis=0)
    px3v = jnp.flip(jnp.cumsum(jnp.flip(term_y, axis=0), axis=0), axis=0)
    
    # PGF = -grad(phi) - rd * T_v * grad(ln p)
    # The Fortran code assembles it into prsgx/y
    
    pgf_x = -dphisdx[None, :, :] + cofb * ps[None, :, :] * dlnpsdx[None, :, :] + px3u - \
            rd * press_diag.alfa * grid_grads.d_t_d_lambda - \
            cofa * rd * virtual_temp * ps[None, :, :] * dlnpsdx[None, :, :]
            
    pgf_y = -dphisdy[None, :, :] + cofb * ps[None, :, :] * dlnpsdy[None, :, :] + px3v - \
            rd * press_diag.alfa * grid_grads.d_t_d_phi - \
            cofa * rd * virtual_temp * ps[None, :, :] * dlnpsdy[None, :, :]
            
    return pgf_x, pgf_y
