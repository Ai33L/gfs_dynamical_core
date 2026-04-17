import jax
import jax.numpy as jnp
from flax import struct

from .states import GridGradients, GridState, SpectralState, SpectralTendencies
from .transforms import (
    TransformConfig,
    enforce_triangular_truncation,
    grid_to_spectral_tendencies,
    spectral_to_grid,
)


@struct.dataclass
class DynamicsConfig:
    """Configuration and constants for GFS dynamics."""

    ak: jnp.ndarray  # (n_lev + 1,) - Bottom to Top
    bk: jnp.ndarray  # (n_lev + 1,) - Bottom to Top
    ck: jnp.ndarray  # (n_lev,) - ak(k+1)*bk(k)-ak(k)*bk(k+1)
    dbk: jnp.ndarray  # (n_lev,) - bk(k)-bk(k+1)
    rk: float  # R / Cp (kappa)
    toa_pressure: float  # Top of atmosphere pressure
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

    ps: jnp.ndarray  # (n_lat, n_lon)
    pk: jnp.ndarray  # (n_lev + 1, n_lat, n_lon) - interface pressure (Bottom to Top)
    dp: jnp.ndarray  # (n_lev, n_lat, n_lon) - layer pressure thickness
    prs: jnp.ndarray  # (n_lev, n_lat, n_lon) - layer mean pressure
    alfa: jnp.ndarray  # (n_lev, n_lat, n_lon)
    rlnp: jnp.ndarray  # (n_lev, n_lat, n_lon)


@struct.dataclass
class VerticalVelocities:
    """Vertical velocity diagnostic fields."""

    omega: jnp.ndarray  # (n_lev, n_lat, n_lon) - d(ln p)/dt on layers
    etadot: jnp.ndarray  # (n_lev + 1, n_lat, n_lon) - eta dot on interfaces
    d_log_ps_d_t: jnp.ndarray  # (n_lat, n_lon) - surface pressure tendency


@struct.dataclass
class GridTendencies:
    """Tendencies in grid space before transformation to spectral."""

    u_flux: jnp.ndarray  # ug * (zeta + f) + v_flux_term
    v_flux: jnp.ndarray  # vg * (zeta + f) - u_flux_term
    temp_tend: jnp.ndarray
    log_ps_tend: jnp.ndarray
    tracer_tends: jnp.ndarray
    kinetic_energy: jnp.ndarray  # 0.5 * (u^2 + v^2)


def compute_pressure_diagnostics(
    log_ps: jnp.ndarray, config: DynamicsConfig
) -> PressureDiagnostics:
    """
    Computes pressure-related diagnostics from log surface pressure.
    Standardized to BOTTOM-TO-TOP indexing (k=0 is surface).
    """
    ps = jnp.exp(log_ps)
    pk = config.ak[:, None, None] + config.bk[:, None, None] * (
        ps[None, :, :] - config.toa_pressure
    )
    # Interface pressures decrease from surface (0) to TOA (N).
    dp = pk[:-1] - pk[1:]
    rk = config.rk
    # prs[k] = mean pressure of layer k
    pk_below = pk[:-1]
    pk_above = pk[1:]
    # Avoid 0**rk issues at model top
    prs = (
        (
            jnp.maximum(pk_below, 1e-10) ** (rk + 1)
            - jnp.maximum(pk_above, 1e-10) ** (rk + 1)
        )
        / ((rk + 1) * dp)
    ) ** (1 / rk)
    rlnp = jnp.log(pk[:-1] / pk[1:])
    alfa = 1.0 - (pk[1:] / dp) * rlnp

    # Boundaries for top layer (index -1)
    alfa = alfa.at[-1].set(jnp.log(2.0))
    rlnp = rlnp.at[-1].set(0.0)

    return PressureDiagnostics(ps=ps, pk=pk, dp=dp, prs=prs, alfa=alfa, rlnp=rlnp)


def compute_vertical_velocities(
    grid_state: GridState,
    grid_grads: GridGradients,
    press_diag: PressureDiagnostics,
    config: DynamicsConfig,
) -> VerticalVelocities:
    """
    Computes omega, etadot, and log surface pressure tendency.
    Standardized to BOTTOM-TO-TOP indexing (k=0 is surface).
    """
    cg = (
        grid_state.u * grid_grads.d_log_ps_d_lambda
        + grid_state.v * grid_grads.d_log_ps_d_phi
    )
    div_dp = grid_state.divergence * press_diag.dp
    cg_dbk = cg * config.dbk[:, None, None]

    # Integrals from TOA down to surface
    db_rev = jnp.cumsum(jnp.flip(div_dp, axis=0), axis=0)
    cb_rev = jnp.cumsum(jnp.flip(cg_dbk, axis=0), axis=0)
    db = jnp.flip(db_rev, axis=0)
    cb = jnp.flip(cb_rev, axis=0)

    cb_with_zero = jnp.concatenate([cb, jnp.zeros_like(cb[:1])], axis=0)
    db_with_zero = jnp.concatenate([db, jnp.zeros_like(db[:1])], axis=0)

    # Tendency at surface (interface 0)
    d_log_ps_d_t = -db_with_zero[0] / press_diag.ps - cb_with_zero[0]

    bk = config.bk[:, None, None]
    etadot = (
        -press_diag.ps[None, :, :] * (bk * d_log_ps_d_t[None, :, :] + cb_with_zero)
        - db_with_zero
    )

    ps = press_diag.ps[None, :, :]
    db_km1 = db_with_zero[1:]
    cb_km1 = cb_with_zero[1:]

    workb = press_diag.rlnp * (db_km1 + ps * cb_km1) + press_diag.alfa * (
        div_dp + ps * cg * config.dbk[:, None, None]
    )
    workc = (
        ps
        * cg
        * (
            config.dbk[:, None, None]
            + config.ck[:, None, None] * press_diag.rlnp / press_diag.dp
        )
    )
    omega = (workc - workb) / press_diag.dp

    return VerticalVelocities(omega=omega, etadot=etadot, d_log_ps_d_t=d_log_ps_d_t)


def compute_vertical_advection(
    data: jnp.ndarray, etadot: jnp.ndarray, dp: jnp.ndarray
) -> jnp.ndarray:
    """
    Computes vertical advection using second-order centered differences.
    Standardized to BOTTOM-TO-TOP indexing.
    """
    n_lev = data.shape[0]
    # Fortran getvadv has datag bottom-to-top but etadot TOP-to-bottom.
    # The Fortran middle-layer formula (TTB index k) is:
    #   vadv(nlevs+1-k) = (0.5/dpk(k)) * (
    #       etadot(k+1) * (datag(nlevs-k)   - datag(nlevs+1-k))
    #     + etadot(k)   * (datag(nlevs+1-k) - datag(nlevs+2-k)))
    #
    # Converting to all-BTU (layer j, 0-based):
    #   etadot_TTB(k+1) -> etadot_BTU[j]   (interface BELOW layer j)
    #   etadot_TTB(k)   -> etadot_BTU[j+1] (interface ABOVE layer j)
    #   datag(nlevs-k)    = data[j-1]       (layer BELOW in BTU)
    #   datag(nlevs+1-k)  = data[j]         (current layer)
    #   datag(nlevs+2-k)  = data[j+1]       (layer ABOVE in BTU)
    #
    # So the correct all-BTU formula is:
    #   vadv[j] = (0.5/dp[j]) * (
    #       etadot[j]   * (data[j-1] - data[j])
    #     + etadot[j+1] * (data[j] - data[j+1]))
    vadv_bot = (0.5 / dp[0]) * etadot[1] * (data[0] - data[1])
    vadv_top = (0.5 / dp[-1]) * etadot[-2] * (data[-2] - data[-1])

    vadv_mid = (0.5 / dp[1:-1]) * (
        etadot[1:-2] * (data[:-2] - data[1:-1]) + etadot[2:-1] * (data[1:-1] - data[2:])
    )
    return jnp.concatenate([vadv_bot[None], vadv_mid, vadv_top[None]], axis=0)


def compute_vertical_advection_tracers(
    data: jnp.ndarray, etadot: jnp.ndarray, dp: jnp.ndarray
) -> jnp.ndarray:
    """
    Computes vertical advection of tracers using Thuburn's (1993)
    positive-definite flux-limited scheme.
    Standardized to BOTTOM-TO-TOP indexing.
    """
    n_lev = data.shape[0]
    datag_half = 0.5 * (data[:-1] + data[1:])
    datag_d = data[1:] - data[:-1]

    # Bottom boundary (i=0, surface side in BTU).
    # Ghost value extrapolated below surface; sign must match interior
    # convention (above minus below).
    data_bot = data[0]
    data_above_bot = data[1]
    datag_d_bot = jnp.where(
        data_bot >= 0,
        data_bot - jnp.maximum(0.0, 2.0 * data_bot - data_above_bot),
        data_bot - jnp.minimum(0.0, 2.0 * data_bot - data_above_bot),
    )

    # Top boundary (i=n_lev, TOA side in BTU).
    # Ghost value extrapolated above TOA; sign must match interior
    # convention (above minus below).
    data_top = data[-1]
    data_below_top = data[-2]
    datag_d_top = jnp.where(
        data_top >= 0,
        jnp.maximum(0.0, 2.0 * data_top - data_below_top) - data_top,
        jnp.minimum(0.0, 2.0 * data_top - data_below_top) - data_top,
    )

    datag_d_full = jnp.concatenate(
        [datag_d_bot[None], datag_d, datag_d_top[None]], axis=0
    )

    epstiny = jnp.finfo(data.dtype).tiny
    datag_d_full = jnp.where(
        jnp.abs(datag_d_full) < epstiny,
        jnp.sign(datag_d_full) * epstiny + jnp.where(datag_d_full == 0, epstiny, 0),
        datag_d_full,
    )

    etadot_int = etadot[1:-1]
    # Downward flux means etadot > 0 (from layer i into layer i-1)
    phi_down = datag_d_full[2:] / datag_d_full[1:-1]
    phi_up = datag_d_full[:-2] / datag_d_full[1:-1]

    limiter_down = (phi_down + jnp.abs(phi_down)) / (1.0 + jnp.abs(phi_down))
    limiter_up = (phi_up + jnp.abs(phi_up)) / (1.0 + jnp.abs(phi_up))

    datag_half_limited = jnp.where(
        etadot_int > 0.0,
        data[1:] + limiter_down * (datag_half - data[1:]),
        data[:-1] + limiter_up * (datag_half - data[:-1]),
    )

    datag_half_full = jnp.concatenate([data[:1], datag_half_limited, data[-1:]], axis=0)

    # Tendency: (1/dp) * (F_bottom - F_top), matching Fortran getvadv_tracers
    vadv = (1.0 / dp) * (
        (datag_half_full[:-1] * etadot[:-1] - datag_half_full[1:] * etadot[1:])
        + data * (etadot[1:] - etadot[:-1])
    )

    return vadv


def compute_pressure_gradient_force(
    virtual_temp: jnp.ndarray,
    grid_grads: GridGradients,
    press_diag: PressureDiagnostics,
    config: DynamicsConfig,
    surface_geopotential_grads: tuple[jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes horizontal pressure gradient force components.
    Standardized to BOTTOM-TO-TOP indexing.
    """
    rd = config.rd
    ps = press_diag.ps
    dlnpsdx = grid_grads.d_log_ps_d_lambda
    dlnpsdy = grid_grads.d_log_ps_d_phi
    dphisdx, dphisdy = surface_geopotential_grads

    bk_bot = config.bk[:-1, None, None]
    bk_top = config.bk[1:, None, None]

    # Guard against 0 * log(Inf) = NaN at TOA
    bk_rlnp = jnp.where(bk_top > 1e-10, bk_top * press_diag.rlnp, 0.0)
    cofb_pressure = -(1.0 / press_diag.dp) * (
        bk_rlnp + press_diag.alfa * config.dbk[:, None, None]
    )
    # Top layer (index -1) special case: no rlnp term
    cofb_pressure = cofb_pressure.at[-1].set(
        -(1.0 / press_diag.dp[-1]) * (press_diag.alfa[-1] * config.dbk[-1])
    )

    pk_bot = press_diag.pk[:-1]
    pk_top = press_diag.pk[1:]

    term1 = bk_bot * pk_top / jnp.where(pk_bot > 1e-10, pk_bot, 1.0) - bk_top
    # Guard against 0 * log(Inf) = NaN at TOA
    cofa_coef = bk_top - pk_top * config.dbk[:, None, None] / press_diag.dp
    term2 = jnp.where(jnp.abs(cofa_coef) > 1e-15, press_diag.rlnp * cofa_coef, 0.0)
    cofa = -(1.0 / press_diag.dp) * (term1 + term2)

    # px2_factor: geopotential integral of layers BELOW (Fortran accumulates from surface up)
    safe_pk = jnp.where(press_diag.pk > 1e-10, press_diag.pk, 1.0)
    bk_ratio = jnp.where(press_diag.pk > 1e-10, config.bk[:, None, None] / safe_pk, 0.0)

    # Correct sign for bk_ratio difference (bot - top)
    bk_ratio_diff = bk_ratio[:-1] - bk_ratio[1:]
    delta_geo = -rd * bk_ratio_diff * virtual_temp

    # Integrate from surface upwards (layers below)
    shifted_delta_geo = jnp.concatenate(
        [jnp.zeros_like(delta_geo[:1]), delta_geo[:-1]], axis=0
    )
    px2_factor = jnp.cumsum(shifted_delta_geo, axis=0)

    # px3: cumulative temperature gradient of layers BELOW
    integrand_x = -rd * press_diag.rlnp * grid_grads.d_t_d_lambda
    integrand_y = -rd * press_diag.rlnp * grid_grads.d_t_d_phi

    # Integrate from surface upwards
    shifted_x = jnp.concatenate(
        [jnp.zeros_like(integrand_x[:1]), integrand_x[:-1]], axis=0
    )
    shifted_y = jnp.concatenate(
        [jnp.zeros_like(integrand_y[:1]), integrand_y[:-1]], axis=0
    )

    px3u = jnp.cumsum(shifted_x, axis=0)
    px3v = jnp.cumsum(shifted_y, axis=0)

    pgf_x = (
        cofb_pressure * rd * virtual_temp * ps[None, :, :] * dlnpsdx[None, :, :]
        - dphisdx[None, :, :]
        + px2_factor * ps[None, :, :] * dlnpsdx[None, :, :]
        + px3u
        - rd * press_diag.alfa * grid_grads.d_t_d_lambda
        - cofa * rd * virtual_temp * ps[None, :, :] * dlnpsdx[None, :, :]
    )
    pgf_y = (
        cofb_pressure * rd * virtual_temp * ps[None, :, :] * dlnpsdy[None, :, :]
        - dphisdy[None, :, :]
        + px2_factor * ps[None, :, :] * dlnpsdy[None, :, :]
        + px3v
        - rd * press_diag.alfa * grid_grads.d_t_d_phi
        - cofa * rd * virtual_temp * ps[None, :, :] * dlnpsdy[None, :, :]
    )
    return pgf_x, pgf_y


def compute_energy_conversion(
    omega: jnp.ndarray,
    virtual_temp: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    config: DynamicsConfig,
) -> jnp.ndarray:
    """Computes thermodynamic energy conversion term."""
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
    latitudes: jnp.ndarray,
) -> GridTendencies:
    """Assembles all grid-space tendencies."""
    u, v = grid_state.u, grid_state.v
    vort = grid_state.vorticity
    pgf_x, pgf_y = pgf
    vadv_u = compute_vertical_advection(u, vvels.etadot, press_diag.dp)
    vadv_v = compute_vertical_advection(v, vvels.etadot, press_diag.dp)
    vadv_t = compute_vertical_advection(
        grid_state.temperature, vvels.etadot, press_diag.dp
    )

    f = 2.0 * config.omega * jnp.sin(latitudes)
    abs_vort = vort + f[None, :, None]
    u_flux = u * abs_vort + (vadv_v - pgf_y)
    v_flux = v * abs_vort - (vadv_u - pgf_x)
    temp_tend = (
        -u * grid_grads.d_t_d_lambda - v * grid_grads.d_t_d_phi - vadv_t + energy_conv
    )

    def compute_tracer_tend(q, dq_dlambda, dq_dphi):
        vadv_q = compute_vertical_advection_tracers(q, vvels.etadot, press_diag.dp)
        return -u * dq_dlambda - v * dq_dphi - vadv_q

    tracer_tends = jax.vmap(compute_tracer_tend)(
        grid_state.tracers,
        grid_grads.d_tracers_d_lambda,
        grid_grads.d_tracers_d_phi,
    )
    ke = 0.5 * (u**2 + v**2)
    return GridTendencies(
        u_flux=u_flux,
        v_flux=v_flux,
        temp_tend=temp_tend,
        log_ps_tend=vvels.d_log_ps_d_t,
        tracer_tends=tracer_tends,
        kinetic_energy=ke,
    )


def compute_dry_mass_fixer(
    ps: jnp.ndarray,
    tracers: jnp.ndarray,
    dp: jnp.ndarray,
    log_surface_pressure: jnp.ndarray,
    lnps_spectral_tend: jnp.ndarray,
    gauss_weights: jnp.ndarray,
    pdryini: float,
    g: float,
    dt: float,
    ntrunc: int = None,
) -> jnp.ndarray:
    """Adjusts spectral lnps tendency to conserve dry surface pressure."""
    import s2fft

    q = tracers[0]
    pwat = jnp.sum(q * dp, axis=0) / g
    n_lon = ps.shape[-1]
    w = gauss_weights[:, None]
    pmean = jnp.sum(w * ps) / n_lon
    pwat_global = jnp.sum(w * pwat) / n_lon
    pcorr = (pdryini + g * pwat_global) / pmean
    lnps_target_grid = jnp.log(ps * pcorr)
    L = log_surface_pressure.shape[0]
    lnps_target_spec = s2fft.forward_jax(lnps_target_grid, L, sampling="gl")
    # Enforce triangular truncation to match Fortran's packed spectral storage.
    # Without this, modes l > ntrunc receive no diffusion and accumulate noise.
    if ntrunc is not None:
        lnps_target_spec = enforce_triangular_truncation(lnps_target_spec, L, ntrunc)
    return (lnps_target_spec - log_surface_pressure) / dt


def full_dynamics_step(
    grid_state: GridState,
    grid_grads: GridGradients,
    phis_grads: tuple[jnp.ndarray, jnp.ndarray],
    config: DynamicsConfig,
    latitudes: jnp.ndarray,
) -> GridTendencies:
    """Performs a full dynamical core step in grid space."""
    press_diag = compute_pressure_diagnostics(grid_state.log_surface_pressure, config)
    vvels = compute_vertical_velocities(grid_state, grid_grads, press_diag, config)
    pgf = compute_pressure_gradient_force(
        grid_state.temperature, grid_grads, press_diag, config, phis_grads
    )
    energy_conv = compute_energy_conversion(
        vvels.omega, grid_state.temperature, grid_state.tracers[0], config
    )
    return assemble_grid_tendencies(
        grid_state, grid_grads, vvels, press_diag, pgf, energy_conv, config, latitudes
    )


def get_spectral_tendencies(
    spec_state: SpectralState,
    phis_grads: tuple[jnp.ndarray, jnp.ndarray],
    dyn_config: DynamicsConfig,
    trans_config: TransformConfig,
    latitudes: jnp.ndarray,
    gauss_weights: jnp.ndarray | None = None,
    pdryini: float | None = None,
    dt: float | None = None,
) -> SpectralTendencies:
    """Computes spectral tendencies from spectral state."""
    grid_state, grid_grads = spectral_to_grid(spec_state, trans_config)
    grid_tends = full_dynamics_step(
        grid_state, grid_grads, phis_grads, dyn_config, latitudes
    )
    spec_tends = grid_to_spectral_tendencies(grid_tends, trans_config)

    if pdryini is not None and gauss_weights is not None and dt is not None:
        press_diag = compute_pressure_diagnostics(
            grid_state.log_surface_pressure, dyn_config
        )
        corrected_lnps_tend = compute_dry_mass_fixer(
            ps=press_diag.ps,
            tracers=grid_state.tracers,
            dp=press_diag.dp,
            log_surface_pressure=spec_state.log_surface_pressure,
            lnps_spectral_tend=spec_tends.d_log_surface_pressure_d_t,
            gauss_weights=gauss_weights,
            pdryini=pdryini,
            g=dyn_config.g,
            dt=dt,
            ntrunc=trans_config.truncation,
        )
        spec_tends = SpectralTendencies(
            d_vorticity_d_t=spec_tends.d_vorticity_d_t,
            d_divergence_d_t=spec_tends.d_divergence_d_t,
            d_temperature_d_t=spec_tends.d_temperature_d_t,
            d_log_surface_pressure_d_t=corrected_lnps_tend,
            d_tracers_d_t=spec_tends.d_tracers_d_t,
        )
    return spec_tends
