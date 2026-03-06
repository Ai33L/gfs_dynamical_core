import jax
import jax.numpy as jnp
from flax import struct
from .states import GridState, GridGradients, SpectralState, SpectralTendencies
from .transforms import spectral_to_grid, grid_to_spectral_tendencies, TransformConfig

@struct.dataclass
class DynamicsConfig:
    """Configuration and constants for GFS dynamics."""
    ak: jnp.ndarray # (n_lev + 1,)
    bk: jnp.ndarray # (n_lev + 1,)
    ck: jnp.ndarray # (n_lev,) - ak(k+1)*bk(k)-ak(k)*bk(k+1)
    dbk: jnp.ndarray # (n_lev,) - bk(k+1)-bk(k)
    rk: float # R / Cp (kappa)
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

@struct.dataclass
class GridTendencies:
    """Tendencies in grid space before transformation to spectral."""
    u_flux: jnp.ndarray # ug * (zeta + f) + v_flux_term
    v_flux: jnp.ndarray # vg * (zeta + f) - u_flux_term
    temp_tend: jnp.ndarray 
    log_ps_tend: jnp.ndarray
    tracer_tends: jnp.ndarray
    kinetic_energy: jnp.ndarray # 0.5 * (u^2 + v^2)

def compute_pressure_diagnostics(log_ps: jnp.ndarray, config: DynamicsConfig) -> PressureDiagnostics:
    """
    Computes pressure-related diagnostics from log surface pressure.
    
    Equivalent to Fortran's calc_pressdata.

    Args:
        log_ps: Log of surface pressure (n_lat, n_lon).
        config: Dynamics configuration and constants.

    Returns:
        PressureDiagnostics: Containing interface pressures, thickness, etc.
    """
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
    """
    Computes omega, etadot, and log surface pressure tendency.
    
    Equivalent to Fortran's getomega.
    Standardized to TOP-TO-BOTTOM indexing (k=0 is top layer).

    Args:
        grid_state: State in grid space.
        grid_grads: Gradients in grid space.
        press_diag: Pressure diagnostics.
        config: Dynamics configuration.

    Returns:
        VerticalVelocities: omega, etadot, and dlnps/dt.
    """
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

    Args:
        data: Field to advect (levels, n_lat, n_lon).
        etadot: Vertical velocity on interfaces (levels+1, n_lat, n_lon).
        dp: Layer pressure thickness (levels, n_lat, n_lon).

    Returns:
        jnp.ndarray: Vertical advection tendency.
    """
    n_lev = data.shape[0]
    vadv_top = (0.5 / dp[0]) * etadot[1] * (data[1] - data[0])
    vadv_bot = (0.5 / dp[-1]) * etadot[-2] * (data[-2] - data[-1])
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

    Args:
        virtual_temp: Virtual temperature.
        grid_grads: Gradients in grid space.
        press_diag: Pressure diagnostics.
        config: Dynamics configuration.
        surface_geopotential_grads: Gradients of surface geopotential (grad_x, grad_y).

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: (pgf_x, pgf_y) tendencies.
    """
    rd = config.rd
    ps = press_diag.ps
    dlnpsdx = grid_grads.d_log_ps_d_lambda
    dlnpsdy = grid_grads.d_log_ps_d_phi
    dphisdx, dphisdy = surface_geopotential_grads
    cofb = -(1.0 / press_diag.dp) * (press_diag.alfa * config.dbk[:, None, None])
    cofa = -(1.0 / (ps[None, :, :] * press_diag.dp)) * (press_diag.alfa * config.ck[:, None, None])
    term_x = rd * press_diag.rlnp * grid_grads.d_t_d_lambda
    term_y = rd * press_diag.rlnp * grid_grads.d_t_d_phi
    px3u = jnp.flip(jnp.cumsum(jnp.flip(term_x, axis=0), axis=0), axis=0)
    px3v = jnp.flip(jnp.cumsum(jnp.flip(term_y, axis=0), axis=0), axis=0)
    pgf_x = -dphisdx[None, :, :] + cofb * ps[None, :, :] * dlnpsdx[None, :, :] + px3u - \
            rd * press_diag.alfa * grid_grads.d_t_d_lambda - \
            cofa * rd * virtual_temp * ps[None, :, :] * dlnpsdx[None, :, :]
    pgf_y = -dphisdy[None, :, :] + cofb * ps[None, :, :] * dlnpsdy[None, :, :] + px3v - \
            rd * press_diag.alfa * grid_grads.d_t_d_phi - \
            cofa * rd * virtual_temp * ps[None, :, :] * dlnpsdy[None, :, :]
    return pgf_x, pgf_y

def compute_energy_conversion(
    omega: jnp.ndarray, 
    virtual_temp: jnp.ndarray, 
    specific_humidity: jnp.ndarray, 
    config: DynamicsConfig
) -> jnp.ndarray:
    """
    Computes thermodynamic energy conversion term.

    Args:
        omega: Pressure vertical velocity (divided by pressure).
        virtual_temp: Virtual temperature.
        specific_humidity: Specific humidity.
        config: Dynamics configuration.

    Returns:
        jnp.ndarray: Energy conversion tendency.
    """
    term = 1.0 + (config.cvap / config.cp - 1.0) * specific_humidity
    return config.rk * omega * virtual_temp / term

def assemble_grid_tendencies(
    grid_state: GridState,
    grid_grads: GridGradients,
    vvels: VerticalVelocities,
    press_diag: PressureDiagnostics,
    pgf: tuple[jnp.ndarray, jnp.ndarray],
    energy_conv: jnp.ndarray,
    config: DynamicsConfig,
    latitudes: jnp.ndarray
) -> GridTendencies:
    """
    Assembles all grid-space tendencies.

    Args:
        grid_state: State in grid space.
        grid_grads: Gradients in grid space.
        vvels: Vertical velocities.
        press_diag: Pressure diagnostics.
        pgf: Pressure gradient force components.
        energy_conv: Energy conversion term.
        config: Dynamics configuration.
        latitudes: Latitudes in radians.

    Returns:
        GridTendencies: Assembled grid-space tendencies.
    """
    u, v = grid_state.u, grid_state.v
    vort = grid_state.vorticity
    pgf_x, pgf_y = pgf
    vadv_u = compute_vertical_advection(u, vvels.etadot, press_diag.dp)
    vadv_v = compute_vertical_advection(v, vvels.etadot, press_diag.dp)
    vadv_t = compute_vertical_advection(grid_state.temperature, vvels.etadot, press_diag.dp)
    f = 2.0 * config.omega * jnp.sin(latitudes)
    abs_vort = vort + f[None, :, None]
    u_flux = u * abs_vort + (vadv_v - pgf_y)
    v_flux = v * abs_vort - (vadv_u - pgf_x)
    temp_tend = -u * grid_grads.d_t_d_lambda - v * grid_grads.d_t_d_phi - vadv_t + energy_conv
    def compute_tracer_tend(q):
        vadv_q = compute_vertical_advection(q, vvels.etadot, press_diag.dp)
        return -vadv_q
    tracer_tends = jax.vmap(compute_tracer_tend)(grid_state.tracers)
    ke = 0.5 * (u**2 + v**2)
    return GridTendencies(
        u_flux=u_flux, v_flux=v_flux, temp_tend=temp_tend,
        log_ps_tend=vvels.d_log_ps_d_t, tracer_tends=tracer_tends,
        kinetic_energy=ke
    )

def full_dynamics_step(
    grid_state: GridState,
    grid_grads: GridGradients,
    phis_grads: tuple[jnp.ndarray, jnp.ndarray],
    config: DynamicsConfig,
    latitudes: jnp.ndarray
) -> GridTendencies:
    """
    Performs a full dynamical core step in grid space.

    Args:
        grid_state: State in grid space.
        grid_grads: Gradients in grid space.
        phis_grads: Surface geopotential gradients.
        config: Dynamics configuration.
        latitudes: Latitudes in radians.

    Returns:
        GridTendencies: Resulting tendencies in grid space.
    """
    press_diag = compute_pressure_diagnostics(grid_state.log_surface_pressure, config)
    vvels = compute_vertical_velocities(grid_state, grid_grads, press_diag, config)
    pgf = compute_pressure_gradient_force(grid_state.temperature, grid_grads, press_diag, config, phis_grads)
    energy_conv = compute_energy_conversion(vvels.omega, grid_state.temperature, grid_state.tracers[0], config)
    tends = assemble_grid_tendencies(grid_state, grid_grads, vvels, press_diag, pgf, energy_conv, config, latitudes)
    return tends

def get_spectral_tendencies(
    spec_state: SpectralState,
    phis_grads: tuple[jnp.ndarray, jnp.ndarray],
    dyn_config: DynamicsConfig,
    trans_config: TransformConfig,
    latitudes: jnp.ndarray
) -> SpectralTendencies:
    """
    Computes spectral tendencies from spectral state (equivalent to getdyntend).

    Args:
        spec_state: State in spectral space.
        phis_grads: Surface geopotential gradients.
        dyn_config: Dynamics configuration.
        trans_config: Transform configuration.
        latitudes: Latitudes in radians.

    Returns:
        SpectralTendencies: Tendencies in spectral space.
    """
    grid_state, grid_grads = spectral_to_grid(spec_state, trans_config)
    grid_tends = full_dynamics_step(grid_state, grid_grads, phis_grads, dyn_config, latitudes)
    spec_tends = grid_to_spectral_tendencies(grid_tends, trans_config)
    return spec_tends
