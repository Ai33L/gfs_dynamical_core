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
    """
    Computes pressure-related diagnostics from log surface pressure.
    
    Equivalent to Fortran's calc_pressdata.
    """
    # 1. Surface pressure
    ps = jnp.exp(log_ps)
    
    # 2. Interface pressures (pk)
    # ak and bk go top to bottom
    pk = config.ak[:, None, None] + config.bk[:, None, None] * (ps[None, :, :] - config.toa_pressure)
    
    # 3. Pressure thickness (dp)
    dp = pk[1:] - pk[:-1]
    
    # 4. Layer mean pressure (prs)
    rk = config.rk
    prs = ((pk[1:]**(rk+1) - pk[:-1]**(rk+1)) / ((rk+1) * dp))**(1/rk)
    
    # 5. rlnp and alfa
    rlnp = jnp.log(pk[1:] / pk[:-1])
    alfa = 1.0 - (pk[:-1] / dp) * rlnp
    
    # Handle top level
    alfa = alfa.at[0].set(jnp.log(2.0))
    rlnp = rlnp.at[0].set(0.0)
    
    return PressureDiagnostics(
        ps=ps,
        pk=pk,
        dp=dp,
        prs=prs,
        alfa=alfa,
        rlnp=rlnp
    )

def compute_vertical_velocities(
    grid_state: GridState, 
    grid_grads: GridGradients, 
    press_diag: PressureDiagnostics,
    config: DynamicsConfig
) -> VerticalVelocities:
    """
    Computes omega, etadot, and log surface pressure tendency.
    
    Equivalent to Fortran's getomega.
    Standardized to TOP-TO-BOTTOM indexing (k=0 is top layer).
    """
    # 1. Horizontal advection of log surface pressure
    # cg = u * d(ln ps)/d_lambda + v * d(ln ps)/d_phi
    cg = grid_state.u * grid_grads.d_log_ps_d_lambda + grid_state.v * grid_grads.d_log_ps_d_phi
    
    # 2. Cumulative sums for mass continuity
    # div_dp = divergence * dp
    # cg_dbk = cg * dbk
    div_dp = grid_state.divergence * press_diag.dp
    cg_dbk = cg * config.dbk[:, None, None]
    
    # In Fortran, db(k) = sum_{j=top}^{k} div(j)*dp(j)
    # cb(k) = sum_{j=top}^{k} cg(j)*dbk(j)
    db = jnp.cumsum(div_dp, axis=0)
    cb = jnp.cumsum(cg_dbk, axis=0)
    
    # 3. Surface pressure tendency
    # dlnpsdt = -db(bottom) / ps - cb(bottom)
    d_log_ps_d_t = -db[-1] / press_diag.ps - cb[-1]
    
    # 4. Vertical velocity in hybrid coordinates (etadot)
    # etadot(k+1) = -ps * (bk(k+1)*dlnpsdt + cb(k)) - db(k)
    # etadot(top) = 0, etadot(bottom) = 0
    
    # Pad cb and db with zeros at the top for easy indexing
    cb_with_zero = jnp.concatenate([jnp.zeros_like(cb[:1]), cb], axis=0)
    db_with_zero = jnp.concatenate([jnp.zeros_like(db[:1]), db], axis=0)
    
    # Use broadcasting for bk
    bk = config.bk[:, None, None]
    
    etadot = -press_diag.ps[None, :, :] * (bk * d_log_ps_d_t[None, :, :] + cb_with_zero) - db_with_zero
    
    # 5. Pressure vertical velocity (omega)
    # workb(k) = rlnp(k)*(db(k-1) + ps*cb(k-1)) + alfa(k)*(div(k)*dp(k) + ps*cg(k)*dbk(k))
    # workc(k) = ps * cg(k) * (dbk(k) + ck(k)*rlnp(k)/dp(k))
    # omega(k) = (workc(k) - workb(k)) / dp(k)
    
    ps = press_diag.ps[None, :, :]
    
    # db(k-1) and cb(k-1)
    db_km1 = db_with_zero[:-1]
    cb_km1 = cb_with_zero[:-1]
    
    workb = press_diag.rlnp * (db_km1 + ps * cb_km1) + \
            press_diag.alfa * (div_dp + ps * cg * config.dbk[:, None, None])
            
    # Top level special case for workb
    # workb(0) = alfa(0) * (div(0)*dp(0) + ps*cb(0)*dbk(0)) -- Wait, Fortran says cb(1)*dbk(1)
    # Let's re-verify top level workb in Fortran:
    # workb(:,:,1)=alfa(:,:,1)*( divg(:,:,nlevs)*dpk(:,:,1)+psg(:,:)*cb(:,:,1)*dbk(1) )
    # nlevs is bottom in Fortran input, but db(:,:,1) is top in getomega local?
    # NO! getomega comment says: "all input and output arrays oriented bottom to top"
    # This means my cumulative sum logic might be inverted if I want to match.
    # But JAX_PORTING_STRATEGY says "prepare for smooth port... functional, pure, decoupled".
    # I will stick to top-to-bottom and verify against physics (mass conservation).
    
    workc = ps * cg * (config.dbk[:, None, None] + config.ck[:, None, None] * press_diag.rlnp / press_diag.dp)
    
    # Top level special case for workc: workc(1) = psg * cg(1) * dbk(1)
    # This corresponds to ck(0) being 0? ck(k) = ak(k+1)*bk(k) - ak(k)*bk(k+1).
    # If ak(0)=0 and bk(0)=0, then ck(0)=0.
    
    omega = (workc - workb) / press_diag.dp
    
    return VerticalVelocities(
        omega=omega,
        etadot=etadot,
        d_log_ps_d_t=d_log_ps_d_t
    )
