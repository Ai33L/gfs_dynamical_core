import jax
import jax.numpy as jnp
from flax import struct

from .states import GridGradients, GridState, SpectralState, SpectralTendencies
from .transforms import TransformConfig, grid_to_spectral_tendencies, spectral_to_grid


@struct.dataclass
class DynamicsConfig:
    """Configuration and constants for GFS dynamics."""

    ak: jnp.ndarray  # (n_lev + 1,)
    bk: jnp.ndarray  # (n_lev + 1,)
    ck: jnp.ndarray  # (n_lev,) - ak(k+1)*bk(k)-ak(k)*bk(k+1)
    dbk: jnp.ndarray  # (n_lev,) - bk(k+1)-bk(k)
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
    pk: jnp.ndarray  # (n_lev + 1, n_lat, n_lon) - interface pressure
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

    Equivalent to Fortran's calc_pressdata.

    Args:
        log_ps: Log of surface pressure (n_lat, n_lon).
        config: Dynamics configuration and constants.

    Returns:
        PressureDiagnostics: Containing interface pressures, thickness, etc.

    Notes on rlnp[0] sentinel (Task 9.2 audit):
        Fortran sets rlnp(:,:,1) = 99999.99 as a sentinel for the top model
        layer, because the top interface pressure pk[0] = ak[0] = 0 makes
        log(pk[1]/pk[0]) = log(pk[1]/0) = +inf.  Fortran's sentinel avoids
        using that infinity in arithmetic.

        JAX sets rlnp[0] = 0.0 instead.  The audit below confirms this is
        numerically correct for all code paths that reference rlnp[0]:

        1. cofb_pressure (PGF px0):  bk_top[0] * rlnp[0].
           For any valid hybrid coordinate bk[0] = 0 (pure-pressure top),
           so bk_top[0] = 0.  Using raw rlnp[0] = inf gives 0 * inf = NaN;
           using 0.0 gives 0 * 0.0 = 0.0, which is correct.

        2. cofa term2 (PGF px5):  rlnp[0] * (bk_top[0] - pk_top[0]*dbk[0]/dp[0]).
           bk_top[0] = 0 and dbk[0] = bk[1] - bk[0] = 0 for standard hybrid
           coords (bk = 0 for all pure-pressure top layers).  The coefficient
           is therefore 0; using raw inf would give NaN, 0.0 gives 0.0.

        3. omega workb:  rlnp[0] * (db_km1[0] + ps * cb_km1[0]).
           db_km1[0] = db_with_zero[0] = 0 and cb_km1[0] = 0 (prepended
           zeros); result is 0 regardless of rlnp[0].

        4. omega workc:  ck[0] * rlnp[0] / dp[0].
           ck[0] = ak[1]*bk[0] - ak[0]*bk[1] = 0 for standard hybrid
           coords; result is 0 regardless of rlnp[0].

        5. px3 integrand:  -rd * rlnp[0] * dT/dlambda[0].
           The shifted integrand construction in compute_pressure_gradient_force
           excludes layer-0 from every px3 output level (shifted_integrand[0]
           maps to integrand[1], so rlnp[0] never propagates to any output).

        Conclusion: rlnp[0] = 0.0 is correct and necessary to prevent NaN.
        The Fortran sentinel 99999.99 achieves the same end because it is
        always multiplied by a zero coefficient (bk[0] = 0 or ck[0] = 0).
    """
    ps = jnp.exp(log_ps)
    pk = config.ak[:, None, None] + config.bk[:, None, None] * (
        ps[None, :, :] - config.toa_pressure
    )
    dp = pk[1:] - pk[:-1]
    rk = config.rk
    prs = ((pk[1:] ** (rk + 1) - pk[:-1] ** (rk + 1)) / ((rk + 1) * dp)) ** (1 / rk)
    rlnp = jnp.log(pk[1:] / pk[:-1])
    alfa = 1.0 - (pk[:-1] / dp) * rlnp
    alfa = alfa.at[0].set(jnp.log(2.0))
    # Set rlnp[0] = 0.0 to avoid NaN from 0 * inf.  See docstring for full
    # audit confirming this is numerically correct for all uses of rlnp.
    rlnp = rlnp.at[0].set(0.0)
    return PressureDiagnostics(ps=ps, pk=pk, dp=dp, prs=prs, alfa=alfa, rlnp=rlnp)


def compute_vertical_velocities(
    grid_state: GridState,
    grid_grads: GridGradients,
    press_diag: PressureDiagnostics,
    config: DynamicsConfig,
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
    cg = (
        grid_state.u * grid_grads.d_log_ps_d_lambda
        + grid_state.v * grid_grads.d_log_ps_d_phi
    )
    div_dp = grid_state.divergence * press_diag.dp
    cg_dbk = cg * config.dbk[:, None, None]
    db = jnp.cumsum(div_dp, axis=0)
    cb = jnp.cumsum(cg_dbk, axis=0)
    d_log_ps_d_t = -db[-1] / press_diag.ps - cb[-1]
    cb_with_zero = jnp.concatenate([jnp.zeros_like(cb[:1]), cb], axis=0)
    db_with_zero = jnp.concatenate([jnp.zeros_like(db[:1]), db], axis=0)
    bk = config.bk[:, None, None]
    etadot = (
        -press_diag.ps[None, :, :] * (bk * d_log_ps_d_t[None, :, :] + cb_with_zero)
        - db_with_zero
    )
    ps = press_diag.ps[None, :, :]
    db_km1 = db_with_zero[:-1]
    cb_km1 = cb_with_zero[:-1]
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

    Args:
        data: Field to advect (levels, n_lat, n_lon).
        etadot: Vertical velocity on interfaces (levels+1, n_lat, n_lon).
        dp: Layer pressure thickness (levels, n_lat, n_lon).

    Returns:
        jnp.ndarray: Vertical advection tendency.
    """
    n_lev = data.shape[0]
    vadv_top = (0.5 / dp[0]) * etadot[1] * (data[1] - data[0])
    vadv_bot = (0.5 / dp[-1]) * etadot[-2] * (data[-1] - data[-2])
    k = jnp.arange(1, n_lev - 1)
    vadv_mid = (0.5 / dp[1:-1]) * (
        etadot[2:-1] * (data[2:] - data[1:-1]) + etadot[1:-2] * (data[1:-1] - data[:-2])
    )
    return jnp.concatenate([vadv_top[None], vadv_mid, vadv_bot[None]], axis=0)


def compute_vertical_advection_tracers(
    data: jnp.ndarray, etadot: jnp.ndarray, dp: jnp.ndarray
) -> jnp.ndarray:
    """
    Computes vertical advection of tracers using Thuburn's (1993)
    positive-definite flux-limited scheme, matching Fortran `getvadv_tracers`.

    Args:
        data: Field to advect (levels, n_lat, n_lon). Top to bottom.
        etadot: Vertical velocity on interfaces (levels+1, n_lat, n_lon). Top to bottom.
        dp: Layer pressure thickness (levels, n_lat, n_lon).

    Returns:
        jnp.ndarray: Vertical advection tendency.
    """
    # Note: Fortran indexes bottom to top (1 is surface, nlevs is top).
    # In JAX, we index top to bottom (0 is top, nlevs-1 is surface).
    # So `data[0]` is top layer, `data[-1]` is bottom layer.
    # Fortran:
    # datag_half(k) = 0.5*(datag(nlevs-k)+datag(nlevs+1-k))
    # datag_d(k) = datag(nlevs-k) - datag(nlevs+1-k)
    # This means datag_half at interface k is average of layer above and below.
    # In our top-down JAX array, layer `i` has interfaces `i` (top) and `i+1` (bottom).

    n_lev = data.shape[0]

    # datag_half corresponds to interfaces. There are n_lev + 1 interfaces.
    # We will compute datag_half and datag_d for internal interfaces 1 to n_lev-1.
    datag_half = 0.5 * (data[:-1] + data[1:])
    datag_d = data[:-1] - data[1:]

    # Boundary conditions for datag_d
    # Top interface (i=0):
    # Fortran:
    # where (datag(nlevs) >= 0)
    #    datag_d(0) = datag(nlevs) - max(0, 2*datag(nlevs) - datag(nlevs-1))
    # elsewhere
    #    datag_d(0) = datag(nlevs) - min(0, 2*datag(nlevs) - datag(nlevs-1))

    data_top = data[0]
    data_below_top = data[1]

    datag_d_top = jnp.where(
        data_top >= 0,
        data_top - jnp.maximum(0.0, 2.0 * data_top - data_below_top),
        data_top - jnp.minimum(0.0, 2.0 * data_top - data_below_top),
    )

    # Bottom interface (i=n_lev):
    # Fortran:
    # where (datag(1) >= 0)
    #    datag_d(nlevs) = max(0, 2*datag(1) - datag(2)) - datag(1)
    # elsewhere
    #    datag_d(nlevs) = min(0, 2*datag(1) - datag(2)) - datag(1)

    data_bot = data[-1]
    data_above_bot = data[-2]

    datag_d_bot = jnp.where(
        data_bot >= 0,
        jnp.maximum(0.0, 2.0 * data_bot - data_above_bot) - data_bot,
        jnp.minimum(0.0, 2.0 * data_bot - data_above_bot) - data_bot,
    )

    # Assemble full datag_d (shape: n_lev + 1)
    datag_d_full = jnp.concatenate(
        [datag_d_top[None], datag_d, datag_d_bot[None]], axis=0
    )

    # Prevent NaNs
    epstiny = jnp.finfo(data.dtype).tiny
    datag_d_full = jnp.where(
        jnp.abs(datag_d_full) < epstiny,
        jnp.sign(datag_d_full) * epstiny + jnp.where(datag_d_full == 0, epstiny, 0),
        datag_d_full,
    )

    # Apply flux limiter for internal interfaces (1 to n_lev-1)
    # Fortran etadot is top-to-bottom. etadot[k] is at interface k.
    # JAX etadot[i] is at interface i.
    # Fortran loop over k=1..nlevs-1 (interfaces from bottom-up, k=1 is one level above surface)
    # Fortran etadot(k+1) is the velocity at that interface.
    # phi = datag_d(k-1)/datag_d(k) or datag_d(k+1)/datag_d(k)
    # In JAX (top-down), for interface `i` (1 to n_lev-1):
    # If etadot[i] > 0 (downward flux):
    #   phi = datag_d_full[i-1] / datag_d_full[i]
    #   datag_half[i-1] = data[i-1] + (phi + abs(phi)) / (1 + abs(phi)) * (datag_half[i-1] - data[i-1])
    # If etadot[i] <= 0 (upward flux):
    #   phi = datag_d_full[i+1] / datag_d_full[i]
    #   datag_half[i-1] = data[i] + (phi + abs(phi)) / (1 + abs(phi)) * (datag_half[i-1] - data[i])

    etadot_int = etadot[1:-1]

    phi_down = datag_d_full[:-2] / datag_d_full[1:-1]
    phi_up = datag_d_full[2:] / datag_d_full[1:-1]

    limiter_down = (phi_down + jnp.abs(phi_down)) / (1.0 + jnp.abs(phi_down))
    limiter_up = (phi_up + jnp.abs(phi_up)) / (1.0 + jnp.abs(phi_up))

    datag_half_limited = jnp.where(
        etadot_int > 0.0,
        data[:-1] + limiter_down * (datag_half - data[:-1]),
        data[1:] + limiter_up * (datag_half - data[1:]),
    )

    # Re-insert boundaries for datag_half. Top interface is data[0], bottom is data[-1]
    datag_half_full = jnp.concatenate([data[:1], datag_half_limited, data[-1:]], axis=0)

    # Compute advection tendency for each layer
    # Fortran:
    # vadv(layer) = (1/dp) * (datag_half(k)*etadot(k+1) - datag_half(k-1)*etadot(k) +
    #                         data(layer) * (etadot(k) - etadot(k+1)))
    # In JAX, for layer i:
    # top interface is i, bottom interface is i+1
    # Note: etadot[i] is positive downward (towards higher pressure).
    # Flux into layer i from top is datag_half_full[i] * etadot[i]
    # Flux out of layer i from bottom is datag_half_full[i+1] * etadot[i+1]
    # The Fortran formulation uses `etadot(k)` as top and `etadot(k+1)` as bottom in its bottom-up loop.
    vadv = (1.0 / dp) * (
        (datag_half_full[1:] * etadot[1:] - datag_half_full[:-1] * etadot[:-1])
        + data * (etadot[:-1] - etadot[1:])
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

    Implements all five PGF terms from Fortran's getpresgrad (ON 462):
      px0 = cofb_pressure * rd * T * ps * grad(lnps)
      px1 = -grad(phis)
      px2 = cofb_geopotential * ps * grad(lnps)   [Task 8.2 — was missing]
      px3 = -rd * cumulative_rlnp * grad(T)
      px4 = -rd * alfa * grad(T)
      px5 = -cofa * rd * T * ps * grad(lnps)

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

    # ------------------------------------------------------------------ #
    # cofb_pressure  (Fortran's FIRST cofb — pressure-coordinate coeff)  #
    # Used for px0: cofb_pressure * rd * T * ps * grad(lnps)             #
    # ------------------------------------------------------------------ #
    # JAX top-down indexing: layer i=0 is top, i=nlevs-1 is bottom.
    # Fortran bottom-to-top: cofb(k=1) is the BOTTOM layer special case.
    # Mapping: Fortran cofb(k) → JAX cofb_pressure[nlevs-k] (flipped).
    bk_top = config.bk[:-1]  # bk at top interface of each layer, shape (nlevs,)
    cofb_pressure = -(1.0 / press_diag.dp) * (
        bk_top[:, None, None] * press_diag.rlnp
        + press_diag.alfa * config.dbk[:, None, None]
    )
    # Bottom layer (JAX index nlevs-1) special case: no rlnp term.
    cofb_pressure = cofb_pressure.at[-1].set(
        -(1.0 / press_diag.dp[-1]) * (press_diag.alfa[-1] * config.dbk[-1])
    )

    # ------------------------------------------------------------------ #
    # cofa  (pressure-coordinate coeff for px5)                          #
    # ------------------------------------------------------------------ #
    pk_top = press_diag.pk[:-1]  # interface pressure at top of each layer
    pk_bot = press_diag.pk[1:]  # interface pressure at bottom of each layer
    bk_bot = config.bk[1:]  # bk at bottom interface of each layer

    term1 = bk_bot[:, None, None] * pk_top / pk_bot - bk_top[:, None, None]
    term2 = press_diag.rlnp * (
        bk_top[:, None, None] - pk_top * config.dbk[:, None, None] / press_diag.dp
    )
    cofa = -(1.0 / press_diag.dp) * (term1 + term2)

    # ------------------------------------------------------------------ #
    # cofb_geopotential  (Fortran's SECOND cofb — geopotential integral) #
    # Task 8.2: previously missing; corresponds to px2 term.             #
    #                                                                     #
    # Fortran recurrence (bottom-to-top, k_F=nlevs is top):              #
    #   cofb_geo(nlevs) = 0                                              #
    #   cofb_geo(nlevs-j) = cofb_geo(nlevs-j+1)                         #
    #                       - rd*(bk(nlevs-j+2)/pk(nlevs-j+2)           #
    #                            - bk(nlevs-j+1)/pk(nlevs-j+1))         #
    #                         * T(j)   for j=1..nlevs-1                  #
    #                                                                     #
    # In JAX (i=0 top, i=nlevs-1 bottom), the per-layer increment is:   #
    #   delta[i] = -rd * (bk[i+1]/pk[i+1] - bk[i]/pk[i]) * T[i]        #
    # cofb_geo is built as a cumulative sum from the bottom up, with the #
    # top-layer increment (delta[0]) excluded.                           #
    # Then applied flipped: px2[i_out] uses cofb_geo[nlevs-1-i_out].    #
    # ------------------------------------------------------------------ #
    bk_ratio_diff = (
        config.bk[1:, None, None] / press_diag.pk[1:]
        - config.bk[:-1, None, None] / press_diag.pk[:-1]
    )  # shape (nlevs, n_lat, n_lon)
    delta_geo = -rd * bk_ratio_diff * virtual_temp  # shape (nlevs, n_lat, n_lon)

    # Build the sequence in JAX top-to-bottom j-order:
    #   j=0 → 0 (top layer contribution is zero by definition)
    #   j=1 → delta_geo[nlevs-1]   (bottom layer's delta)
    #   j=2 → delta_geo[nlevs-2]
    #   ...
    #   j=nlevs-1 → delta_geo[1]   (second-from-top layer's delta)
    # Note: delta_geo[0] (top layer) is intentionally excluded.
    delta_j_order = jnp.concatenate(
        [
            jnp.zeros_like(delta_geo[:1]),  # j=0: zero
            jnp.flip(delta_geo[1:], axis=0),
        ],  # j=1..nlevs-1: reversed
        axis=0,
    )
    cofb_geo = jnp.cumsum(delta_j_order, axis=0)  # shape (nlevs, n_lat, n_lon)

    # Assembly: at JAX output level i_out, use cofb_geo[nlevs-1-i_out].
    px2_factor = jnp.flip(cofb_geo, axis=0)  # shape (nlevs, n_lat, n_lon)

    # ------------------------------------------------------------------ #
    # px3  (cumulative geopotential gradient from temperature gradients) #
    # ------------------------------------------------------------------ #
    integrand_x = -rd * press_diag.rlnp * grid_grads.d_t_d_lambda
    integrand_y = -rd * press_diag.rlnp * grid_grads.d_t_d_phi
    # Shift by one level: for layer i the integral sums levels i+1..nlevs-1.
    shifted_integrand_x = jnp.concatenate(
        [integrand_x[1:], jnp.zeros_like(integrand_x[:1])], axis=0
    )
    shifted_integrand_y = jnp.concatenate(
        [integrand_y[1:], jnp.zeros_like(integrand_y[:1])], axis=0
    )
    px3u = jnp.flip(jnp.cumsum(jnp.flip(shifted_integrand_x, axis=0), axis=0), axis=0)
    px3v = jnp.flip(jnp.cumsum(jnp.flip(shifted_integrand_y, axis=0), axis=0), axis=0)

    # ------------------------------------------------------------------ #
    # Final assembly                                                      #
    # px0: cofb_pressure * rd * T * ps * grad(lnps)                      #
    # px1: -grad(phis)                                                   #
    # px2: cofb_geopotential * ps * grad(lnps)   ← Task 8.2 addition    #
    # px3: cumulative rlnp temperature gradient term                     #
    # px4: -rd * alfa * grad(T)                                          #
    # px5: -cofa * rd * T * ps * grad(lnps)                              #
    # ------------------------------------------------------------------ #
    pgf_x = (
        cofb_pressure * rd * virtual_temp * ps[None, :, :] * dlnpsdx[None, :, :]  # px0
        - dphisdx[None, :, :]  # px1
        + px2_factor * ps[None, :, :] * dlnpsdx[None, :, :]  # px2
        + px3u  # px3
        - rd * press_diag.alfa * grid_grads.d_t_d_lambda  # px4
        - cofa * rd * virtual_temp * ps[None, :, :] * dlnpsdx[None, :, :]  # px5
    )
    pgf_y = (
        cofb_pressure * rd * virtual_temp * ps[None, :, :] * dlnpsdy[None, :, :]  # px0
        - dphisdy[None, :, :]  # px1
        + px2_factor * ps[None, :, :] * dlnpsdy[None, :, :]  # px2
        + px3v  # px3
        - rd * press_diag.alfa * grid_grads.d_t_d_phi  # px4
        - cofa * rd * virtual_temp * ps[None, :, :] * dlnpsdy[None, :, :]  # px5
    )
    return pgf_x, pgf_y


def compute_energy_conversion(
    omega: jnp.ndarray,
    virtual_temp: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    config: DynamicsConfig,
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
    latitudes: jnp.ndarray,
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

    # Task 8.1: include horizontal advection  -u*dq/dlambda - v*dq/dphi
    # in addition to the existing vertical advection term -vadv_q.
    # Tracer gradients (n_tracers, levels, n_lat, n_lon) are computed
    # spectrally in spectral_to_grid and stored in GridGradients.
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
) -> jnp.ndarray:
    """
    Adjusts the spectral lnps tendency so the global integral of dry surface
    pressure returns to its initial value ``pdryini`` after one timestep.

    Mirrors Fortran ``dyn_run.f90: dry_mass_fixer``.

    Algorithm (Fortran lines 715–748):
      1. Compute global-mean precipitable water:
            pwat_global = sum(areawts * pwat)
         where pwat[j] = (1/g) * sum_k(q[k,j] * dp[k,j])
      2. Compute global-mean surface pressure:
            pmean = sum(areawts * ps)
      3. The global-mean dry surface pressure is:
            pdry = pmean - g * pwat_global
      4. The multiplicative correction that restores dry mass to pdryini:
            pcorr = (pdryini + g * pwat_global) / pmean
      5. Apply the correction as a new target lnps field:
            lnps_target = log(ps * pcorr)
      6. Convert lnps_target to spectral space, then compute the corrected
         tendency as:
            dlnpsspecdt_corrected = (lnps_target_spec - lnps_spec) / dt

    The correction is applied as a tendency rather than directly to lnps so
    that it is applied consistently within the IMEX stepper, matching the
    Fortran approach of modifying ``dlnpsspecdt1`` in ``run.f90``.

    Args:
        ps: Surface pressure in Pa, shape ``(n_lat, n_lon)``.
        tracers: Tracer fields, shape ``(n_tracers, n_lev, n_lat, n_lon)``.
            The first tracer (index 0) is specific humidity.
        dp: Layer pressure thickness, shape ``(n_lev, n_lat, n_lon)``.
        log_surface_pressure: Current spectral lnps coefficients,
            shape ``(L, 2*L-1)``.
        lnps_spectral_tend: Current spectral lnps tendency (before fixer),
            shape ``(L, 2*L-1)``.
        gauss_weights: Normalised Gaussian quadrature weights summing to 1,
            shape ``(n_lat,)``.  Each weight applies to the full longitude ring.
        pdryini: Initial global mean dry surface pressure (Pa).
        g: Gravitational acceleration (m s⁻²).
        dt: Timestep in seconds.

    Returns:
        jnp.ndarray: Corrected spectral lnps tendency, same shape as
            ``lnps_spectral_tend``.
    """
    import s2fft

    # ------------------------------------------------------------------
    # Step 1: precipitable water column per grid point (Pa/g * g = Pa)
    # pwat[j] = (1/g) * sum_k(q[k,j] * dp[k,j])   (Fortran units: kg/m²)
    # ------------------------------------------------------------------
    q = tracers[0]  # specific humidity, shape (n_lev, n_lat, n_lon)
    pwat = jnp.sum(q * dp, axis=0) / g  # (n_lat, n_lon)

    # ------------------------------------------------------------------
    # Step 2-3: global means via Gaussian quadrature weights
    # areawts is 2D in Fortran but constant in longitude; we use 1D weights
    # broadcast over longitude.  weights are already normalised (sum=1).
    # ------------------------------------------------------------------
    n_lon = ps.shape[-1]
    w = gauss_weights[:, None]  # (n_lat, 1)  — broadcast over lon
    pmean = jnp.sum(w * ps) / n_lon  # scalar global mean ps
    pwat_global = jnp.sum(w * pwat) / n_lon  # scalar global mean pwat

    # ------------------------------------------------------------------
    # Step 4: multiplicative correction factor
    # pcorr = (pdryini + g * pwat_global) / pmean
    # ------------------------------------------------------------------
    pcorr = (pdryini + g * pwat_global) / pmean

    # ------------------------------------------------------------------
    # Step 5-6: build corrected lnps in grid space, transform to spectral,
    # then form the tendency  dlnps_corrected = (lnps_target - lnps) / dt
    # ------------------------------------------------------------------
    lnps_target_grid = jnp.log(ps * pcorr)

    # Determine L and sampling from the spectral coefficient shape.
    # log_surface_pressure has shape (L, 2*L-1).
    L = log_surface_pressure.shape[0]
    # We need to infer the sampling used; for GL it is always "gl" in this port.
    sampling = "gl"
    lnps_target_spec = s2fft.forward_jax(lnps_target_grid, L, sampling=sampling)

    corrected_tend = (lnps_target_spec - log_surface_pressure) / dt
    return corrected_tend


def full_dynamics_step(
    grid_state: GridState,
    grid_grads: GridGradients,
    phis_grads: tuple[jnp.ndarray, jnp.ndarray],
    config: DynamicsConfig,
    latitudes: jnp.ndarray,
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
    pgf = compute_pressure_gradient_force(
        grid_state.temperature, grid_grads, press_diag, config, phis_grads
    )
    energy_conv = compute_energy_conversion(
        vvels.omega, grid_state.temperature, grid_state.tracers[0], config
    )
    tends = assemble_grid_tendencies(
        grid_state, grid_grads, vvels, press_diag, pgf, energy_conv, config, latitudes
    )
    return tends


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
    """
    Computes spectral tendencies from spectral state (equivalent to getdyntend).

    Args:
        spec_state: State in spectral space.
        phis_grads: Surface geopotential gradients.
        dyn_config: Dynamics configuration.
        trans_config: Transform configuration.
        latitudes: Latitudes in radians.
        gauss_weights: Normalised Gaussian quadrature weights, shape (n_lat,).
            Required when pdryini is provided; used for global area integrals.
        pdryini: Initial global mean dry surface pressure (Pa).  When provided
            (and gauss_weights and dt are also provided), the dry-mass fixer is
            applied to the lnps spectral tendency after the dynamics step.
        dt: Timestep in seconds.  Required when pdryini is provided.

    Returns:
        SpectralTendencies: Tendencies in spectral space.
    """
    grid_state, grid_grads = spectral_to_grid(spec_state, trans_config)
    grid_tends = full_dynamics_step(
        grid_state, grid_grads, phis_grads, dyn_config, latitudes
    )
    spec_tends = grid_to_spectral_tendencies(grid_tends, trans_config)

    # Task 10.1: apply dry-mass fixer when enabled.
    # Mirrors Fortran run.f90 lines 349-356: after physics tendencies are
    # accumulated, dry_mass_fixer adjusts dlnpsspecdt so that the global
    # integral of dry surface pressure returns to pdryini.
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
        )
        spec_tends = SpectralTendencies(
            d_vorticity_d_t=spec_tends.d_vorticity_d_t,
            d_divergence_d_t=spec_tends.d_divergence_d_t,
            d_temperature_d_t=spec_tends.d_temperature_d_t,
            d_log_surface_pressure_d_t=corrected_lnps_tend,
            d_tracers_d_t=spec_tends.d_tracers_d_t,
        )

    return spec_tends
