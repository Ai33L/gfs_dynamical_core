import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from .dynamics import (
    DynamicsConfig,
    compute_dry_mass_fixer,
    compute_pressure_diagnostics,
    get_spectral_tendencies,
)
from .states import SpectralState, SpectralTendencies
from .transforms import TransformConfig, spectral_to_grid

jax_debug_step = 0


def dump_jax_intermediate(grid_state, spec_state, step, stage):
    if step > 100:
        return
    # Skip when running inside JIT — traced arrays cannot be converted to numpy
    if isinstance(grid_state.u, jax.core.Tracer):
        return
    filename = f"debug_data/jax_step_{step}_stage_{stage}.bin"
    if not os.path.isdir(os.path.dirname(filename)):
        return

    # Strictly match Fortran layout: (nlons, nlats, nlevs) or (nlons, nlats)
    # JAX u is (levels, lat, lon) -> transpose(2, 1, 0) -> (lon, lat, levels)
    u_f = np.asfortranarray(np.array(grid_state.u).transpose(2, 1, 0), dtype=np.float64)
    v_f = np.asfortranarray(np.array(grid_state.v).transpose(2, 1, 0), dtype=np.float64)
    t_f = np.asfortranarray(
        np.array(grid_state.temperature).transpose(2, 1, 0), dtype=np.float64
    )
    ps_f = np.asfortranarray(
        np.array(grid_state.log_surface_pressure).transpose(1, 0), dtype=np.float64
    )
    q_f = np.asfortranarray(
        np.array(grid_state.tracers).transpose(3, 2, 1, 0), dtype=np.float64
    )

    with open(filename, "wb") as f:
        f.write(u_f.tobytes(order="F"))
        f.write(v_f.tobytes(order="F"))
        f.write(t_f.tobytes(order="F"))
        f.write(ps_f.tobytes(order="F"))
        f.write(q_f.tobytes(order="F"))

        # Add spectral fields (rectangular layout, complex128)
        # We'll save them as (levels, L, 2L-1) or (L, 2L-1)
        f.write(np.array(spec_state.vorticity).astype(np.complex128).tobytes())
        f.write(np.array(spec_state.divergence).astype(np.complex128).tobytes())
        f.write(np.array(spec_state.temperature).astype(np.complex128).tobytes())
        f.write(
            np.array(spec_state.log_surface_pressure).astype(np.complex128).tobytes()
        )


@struct.dataclass
class StepperConfig:
    """Configuration for the IMEX Runge-Kutta time stepper."""

    # Explicit RK constants
    a21: float = 1.0
    a31: float = 0.25
    a32: float = 0.25
    b1: float = 1.0 / 6.0
    b2: float = 1.0 / 6.0
    b3: float = 2.0 / 3.0

    # Implicit RK constants
    aa21: float = 0.635
    aa22: float = 0.365
    aa31: float = 0.3175
    aa32: float = 0.0
    aa33: float = 0.1825
    bb1: float = 0.35
    bb2: float = 0.0
    bb3: float = 0.3
    bb4: float = 0.35

    explicit: bool = struct.field(pytree_node=False, default=False)

    # Linear diffusion
    disspec: jnp.ndarray = None  # (L, 2L-1)
    diff_prof: jnp.ndarray = None  # (n_lev,)
    dmp_prof: jnp.ndarray = None  # (n_lev,)

    # Semi-implicit matrices
    amhyb: jnp.ndarray = None  # (n_lev, n_lev)
    bmhyb: jnp.ndarray = None  # (n_lev, n_lev)
    tor_hyb: jnp.ndarray = None  # (n_lev,)
    svhyb: jnp.ndarray = None  # (n_lev,)
    d_hyb_m: jnp.ndarray = None  # (3, L, n_lev, n_lev)

    dt: float = 1200.0


# ---------------------------------------------------------------------------
# Semi-implicit matrix setup  (mirrors Fortran semimp_data.f90)
# ---------------------------------------------------------------------------

_REF_TEMP = 300.0  # K  — reference temperature for linearisation
_REF_PRESS = 800.0e2  # Pa — reference surface pressure for linearisation


def init_semi_implicit_matrices(
    dyn_config: DynamicsConfig,
    trans_config: TransformConfig,
    dt: float,
    aa22: float = 0.365,
    aa33: float = 0.1825,
    bb4: float = 0.35,
) -> dict:
    """Compute the IMEX semi-implicit matrices that match Fortran's ``init_semimpdata``.

    The linearised terms treated implicitly are:

    * **amhyb** — geopotential part of the linearised pressure-gradient force
      (operates on virtual-temperature spectral coefficients).
    * **bmhyb** — linearised energy-conversion term (operates on divergence
      spectral coefficients).
    * **tor_hyb** — ln(ps) contribution to the linearised PGF (multiplies
      ``lnpsspec``; combined with ``amhyb`` in the divergence equation).
    * **svhyb** — linearised ln(ps) tendency (multiplies divergence; continuity
      equation).
    * **d_hyb_m** — pre-inverted matrices ``(I + (c·dt)²·n(n+1)·Ym)⁻¹`` for
      each implicit RK coefficient ``c ∈ {aa22, aa33, bb4}`` and each
      spherical-harmonic degree ``n``.  Shape ``(3, L, n_lev, n_lev)``.

    The returned dict has keys
    ``amhyb, bmhyb, tor_hyb, svhyb, d_hyb_m`` — all plain ``jnp.ndarray``
    values ready to be passed to :class:`StepperConfig`.

    Parameters
    ----------
    dyn_config : DynamicsConfig
        Must contain ``ak, bk, rd, cp, radius`` (and derived ``rk``).
    trans_config : TransformConfig
        Provides ``L`` (band-limit).
    dt : float
        Time-step in seconds.
    aa22 : float
        Diagonal implicit coefficient for stage 2.  Must match the value in
        :class:`StepperConfig` that will use these matrices.  Default matches
        ``StepperConfig.aa22 = 0.365``.
    aa33 : float
        Diagonal implicit coefficient for stage 3.  Default matches
        ``StepperConfig.aa33 = 0.1825``.
    bb4 : float
        Diagonal implicit coefficient for the final combination stage.
        Default matches ``StepperConfig.bb4 = 0.35``.

    Returns
    -------
    dict[str, jnp.ndarray]
    """
    ak = np.asarray(dyn_config.ak, dtype=np.float64)
    bk = np.asarray(dyn_config.bk, dtype=np.float64)
    rd = float(dyn_config.rd)
    kappa = float(dyn_config.rk)  # R_d / C_p
    rerth = float(dyn_config.radius)
    n_lev = len(ak) - 1
    L = trans_config.L

    # --- reference-state pressure diagnostics (Fortran bottom-to-top) ------
    # Ensure ak_f, bk_f are top-to-bottom for the algorithm below
    # (index 0 = TOA)
    ak_f = ak[::-1]
    bk_f = bk[::-1]

    tref = np.full(n_lev, _REF_TEMP)
    pkref = np.empty(n_lev + 1)
    for k in range(n_lev + 1):
        pkref[k] = ak_f[k] + bk_f[k] * _REF_PRESS
    dpkref = np.empty(n_lev)
    for k in range(n_lev):
        dpkref[k] = pkref[k + 1] - pkref[k]
    alfaref = np.empty(n_lev)
    alfaref[0] = np.log(2.0)
    for k in range(1, n_lev):
        alfaref[k] = 1.0 - (pkref[k] / dpkref[k]) * np.log(pkref[k + 1] / pkref[k])

    # --- yecm: geopotential operator (upper-triangular + diagonal) ---------
    yecm = np.zeros((n_lev, n_lev))
    for irow in range(n_lev):
        yecm[irow, irow] = alfaref[irow] * rd
        for icol in range(irow + 1, n_lev):
            yecm[irow, icol] = rd * np.log(pkref[icol + 1] / pkref[icol])

    # --- tecm: energy-conversion operator (lower-triangular + diagonal) ----
    tecm = np.zeros((n_lev, n_lev))
    for irow in range(n_lev):
        tecm[irow, irow] = kappa * tref[irow] * alfaref[irow]
        for icol in range(irow):
            tecm[irow, icol] = (
                kappa * tref[irow] * dpkref[icol] / dpkref[irow]
            ) * np.log(pkref[irow + 1] / pkref[irow])

    # --- vecm, svhyb (continuity equation linearisation) -------------------
    vecm = dpkref / _REF_PRESS

    # Flip to bottom-to-top to match the JAX state (index 0 = surface)
    amhyb_f = np.zeros((n_lev, n_lev))
    bmhyb_f = np.zeros((n_lev, n_lev))
    svhyb_f = np.zeros(n_lev)
    for j in range(n_lev):
        svhyb_f[j] = vecm[n_lev - 1 - j]
        for k in range(n_lev):
            amhyb_f[k, j] = yecm[n_lev - 1 - k, n_lev - 1 - j]
            bmhyb_f[k, j] = tecm[n_lev - 1 - k, n_lev - 1 - j]

    amhyb_f = amhyb_f / rerth**2
    tor_hyb_f = rd * tref / rerth**2

    # --- ym = tor_hyb ⊗ svhyb + amhyb @ bmhyb  (wavenumber-independent) --
    ym = np.outer(tor_hyb_f, svhyb_f) + amhyb_f @ bmhyb_f

    # --- d_hyb_m: (I + (c·dt)²·n(n+1)·ym)⁻¹ for each stage & degree ------
    # Coefficients are taken from the function parameters (not hardcoded) so
    # that the matrices stay consistent with whichever StepperConfig uses them.
    consts = [aa22, aa33, bb4]

    # Fortran shape: (nlevs, nlevs, ntrunc+1, 3)
    # JAX  shape:    (3, L, nlevs, nlevs)   with L = ntrunc + 1
    d_hyb_m = np.zeros((3, L, n_lev, n_lev))
    rim = np.eye(n_lev)

    for stage_idx, const in enumerate(consts):
        for nn in range(L):
            n = nn  # degree
            rnn1 = n * (n + 1.0)
            mat = rim + (const * dt) ** 2 * rnn1 * ym
            d_hyb_m[stage_idx, nn] = np.linalg.inv(mat)

    return dict(
        amhyb=jnp.array(amhyb_f),
        bmhyb=jnp.array(bmhyb_f),
        tor_hyb=jnp.array(tor_hyb_f),
        svhyb=jnp.array(svhyb_f),
        d_hyb_m=jnp.array(d_hyb_m),
    )


# ---------------------------------------------------------------------------
# Hyper-diffusion & Rayleigh-damping setup  (mirrors Fortran setdampspec)
# ---------------------------------------------------------------------------


def init_diffusion_operators(
    dyn_config: DynamicsConfig,
    trans_config: TransformConfig,
    dt: float,
    number_of_damped_levels: int = 0,
    damping_timescale: float = 2.0 * 86400,
) -> dict:
    """Compute hyper-diffusion and upper-level Rayleigh damping operators.

    Mirrors Fortran ``setdampspec`` in ``dyn_init.f90``.

    Returns a dict with keys ``disspec, diff_prof, dmp_prof``.

    Parameters
    ----------
    dyn_config : DynamicsConfig
        Must contain ``ak, bk, radius, toa_pressure``.
    trans_config : TransformConfig
        Provides ``L`` (band-limit, = ntrunc + 1).
    dt : float
        Time-step in seconds (used only for documentation; the operators
        themselves are dt-independent).
    number_of_damped_levels : int
        Number of model levels from the top that are Rayleigh-damped.
    damping_timescale : float
        Rayleigh damping e-folding timescale in seconds.
    """
    ak = np.asarray(dyn_config.ak, dtype=np.float64)
    bk = np.asarray(dyn_config.bk, dtype=np.float64)
    rerth = float(dyn_config.radius)
    toa_pressure = float(dyn_config.toa_pressure)
    n_lev = len(ak) - 1
    L = trans_config.L
    ntrunc = trans_config.truncation

    # Python arrays are now bottom-to-top
    ak_f = ak
    bk_f = bk

    # Reference surface pressure for sigma computation
    pdryini = 1.0e5  # 1000 hPa default — only used for sigma levels
    # Sigma at interfaces (bottom-to-top in JAX convention)
    # ak_f[k] is surface interface at k=0.
    si = np.empty(n_lev + 1)
    for k in range(n_lev + 1):
        si[k] = (ak_f[k] - toa_pressure) / (pdryini - toa_pressure) + bk_f[k]
    # Sigma at mid-levels
    sl = np.empty(n_lev)
    for k in range(n_lev):
        sl[k] = 0.5 * (si[k] + si[k + 1])

    # --- ndiss, efold, fshk (GFS defaults) ---------------------------------
    ndiss = 8
    hdif_fac = 1.0
    hdif_fac2 = 1.0

    if ntrunc > 170:
        efold = 3600.0 / (hdif_fac2 * (ntrunc / 170.0) ** 4 * 1.1)
    elif ntrunc == 126 or ntrunc == 170:
        efold = 1.0 / (hdif_fac2 * 12.0e15 / rerth**4 * (80.0 * 81.0) ** 2)
    else:
        efold = 1.0 / (hdif_fac2 * 3.0e15 / rerth**4 * (80.0 * 81.0) ** 2)
    efold = 2.0 * efold  # mysterious factor 2 in deldifs.f

    if ntrunc > 170:
        fshk = 2.2 * hdif_fac
    elif ntrunc == 126:
        fshk = 1.5 * hdif_fac
    else:
        fshk = 1.0 * hdif_fac

    # --- diff_prof (height-dependent diffusion enhancement) ----------------
    # Fortran: diff_prof = sl**(log(1./fshk))
    diff_prof_f = sl ** np.log(1.0 / fshk)  # shape (n_lev,)

    # --- dmp_prof (Rayleigh damping) ----------------------------------------
    slrd0 = si[n_lev - number_of_damped_levels] if number_of_damped_levels > 0 else 0.0
    dmp_prof1 = 1.0 / damping_timescale
    dmp_prof_f = np.zeros(n_lev)
    for k in range(n_lev):
        if slrd0 > 0 and sl[k] < slrd0:
            dmp_prof_f[k] = dmp_prof1 * np.log(slrd0 / sl[k])

    # --- disspec (spectral hyper-diffusion operator) -----------------------
    # Fortran uses 1-D packed triangular layout; we need 2-D (L, 2L-1).
    # lap(l,m) = -l*(l+1);  Fortran's disspec = -(1/efold)*(lap/min(lap))^(ndiss/2)
    # min(lap) = -ntrunc*(ntrunc+1) (most negative)
    l_arr = np.arange(L, dtype=np.float64)
    m_arr = np.arange(-L + 1, L, dtype=np.float64)
    l_grid, m_grid = np.meshgrid(l_arr, m_arr, indexing="ij")

    lap_2d = -(l_grid * (l_grid + 1.0))
    min_lap = -(ntrunc * (ntrunc + 1.0))  # most negative value

    # Avoid division by zero at l=0 (lap=0 there, disspec=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(min_lap != 0, lap_2d / min_lap, 0.0)
    disspec_2d = -(1.0 / efold) * np.abs(ratio) ** (ndiss / 2)
    # l=0 has lap=0, so disspec should be 0 there
    disspec_2d[0, :] = 0.0

    # Zero out entries outside triangular truncation (|m| > l or l > ntrunc)
    mask = (l_grid <= ntrunc) & (np.abs(m_grid) <= l_grid)
    disspec_2d = np.where(mask, disspec_2d, 0.0)

    return dict(
        disspec=jnp.array(disspec_2d),
        diff_prof=jnp.array(diff_prof_f),
        dmp_prof=jnp.array(dmp_prof_f),
    )


def advance(
    state: SpectralState,
    phis_grads: tuple[jnp.ndarray, jnp.ndarray],
    dyn_config: DynamicsConfig,
    trans_config: TransformConfig,
    stepper_config: StepperConfig,
    latitudes: jnp.ndarray,
    gauss_weights: Optional[jnp.ndarray] = None,
    pdryini: Optional[float] = None,
) -> SpectralState:
    """
    Advances the spectral state by one timestep using 3-stage IMEX RK scheme.

    Args:
        state: Current spectral state.
        phis_grads: Surface geopotential gradients (dphis/dlambda, dphis/dphi).
        dyn_config: Dynamics configuration and physical constants.
        trans_config: Transform configuration (L, sampling, radius).
        stepper_config: IMEX RK stepper configuration.
        latitudes: Gaussian quadrature latitudes in radians, shape (n_lat,).
        gauss_weights: Normalised Gaussian quadrature weights summing to 1,
            shape (n_lat,).  Required to enable the dry-mass fixer.
        pdryini: Initial global mean dry surface pressure (Pa).  When both
            gauss_weights and pdryini are provided, the dry-mass fixer is
            applied to the lnps tendency at the final RK stage, matching
            Fortran run.f90 lines 349-356.
    """
    global jax_debug_step
    jax_debug_step += 1
    dt = stepper_config.dt
    L = trans_config.L

    l_arr = jnp.arange(L)
    # laplacian operator in spectral space is -l(l+1) / R^2, matching Fortran's lap(n)
    # But wait, Fortran's lap is computed via SHTNS, which is -l(l+1). Fortran code does `lap(n)`
    # actually Fortran lap is indeed -l(l+1). The / R^2 is applied explicitly or included?
    # Fortran: `ddivdtlin_orig(n,:) = -lap(n) * (matmul(amhyb, virtempspec) + tor_hyb * lnpsspec)`
    # Let's define lap = -l(l+1).
    lap = -l_arr * (l_arr + 1.0)

    def compute_linear_tendencies(div, temp, lnps):
        # Temp linear tendency: dtvdtlin = -matmul(bmhyb, div)
        # div is (n_lev, L, 2L-1), bmhyb is (n_lev, n_lev). We matmul over levs.
        dtvdtlin = -jnp.einsum("ij,j...->i...", stepper_config.bmhyb, div)

        # lnps linear tendency: dlnpsdtlin = -sum(svhyb * div)
        dlnpsdtlin = -jnp.einsum("i,i...->...", stepper_config.svhyb, div)

        # div linear tendency: ddivdtlin = -lap * (matmul(amhyb, temp) + tor_hyb * lnps)
        temp_term = jnp.einsum("ij,j...->i...", stepper_config.amhyb, temp)
        lnps_term = stepper_config.tor_hyb[:, None, None] * lnps[None, :, :]
        ddivdtlin = -lap[None, :, None] * (temp_term + lnps_term)

        return ddivdtlin, dtvdtlin, dlnpsdtlin

    def solve_implicit(div_expl, temp_expl, lnps_expl, coeff, stage_idx):
        # rhs = div_expl - coeff * dt * lap * (matmul(amhyb, temp_expl) + tor_hyb * lnps_expl)
        temp_term = jnp.einsum("ij,j...->i...", stepper_config.amhyb, temp_expl)
        lnps_term = stepper_config.tor_hyb[:, None, None] * lnps_expl[None, :, :]
        rhs = div_expl - coeff * dt * lap[None, :, None] * (temp_term + lnps_term)

        # div_new = matmul(d_hyb_m, rhs)
        # d_hyb_m has shape (3, L, n_lev, n_lev) -> indexed by stage_idx, l.
        # rhs is (n_lev, L, 2L-1).
        # We need to apply d_hyb_m[stage_idx, l] to rhs[:, l, m].
        # Using einsum: 'lij,jlm->ilm'
        d_mat = stepper_config.d_hyb_m[stage_idx]  # (L, n_lev, n_lev)
        div_new = jnp.einsum("lij,jlm->ilm", d_mat, rhs)

        # back substitute
        temp_new = temp_expl - coeff * dt * jnp.einsum(
            "ij,j...->i...", stepper_config.bmhyb, div_new
        )
        lnps_new = lnps_expl - coeff * dt * jnp.einsum(
            "i,i...->...", stepper_config.svhyb, div_new
        )

        return div_new, temp_new, lnps_new

    # --- Stage 1 ---
    tends_orig = get_spectral_tendencies(
        state, phis_grads, dyn_config, trans_config, latitudes
    )

    vort1 = state.vorticity + stepper_config.a21 * dt * tends_orig.d_vorticity_d_t
    tracers1 = state.tracers + stepper_config.a21 * dt * tends_orig.d_tracers_d_t

    if stepper_config.explicit:
        div1 = state.divergence + stepper_config.a21 * dt * tends_orig.d_divergence_d_t
        temp1 = (
            state.temperature + stepper_config.a21 * dt * tends_orig.d_temperature_d_t
        )
        lnps1 = (
            state.log_surface_pressure
            + stepper_config.a21 * dt * tends_orig.d_log_surface_pressure_d_t
        )

        ddivdtlin_orig = jnp.zeros_like(tends_orig.d_divergence_d_t)
        dtvdtlin_orig = jnp.zeros_like(tends_orig.d_temperature_d_t)
        dlnpsdtlin_orig = jnp.zeros_like(tends_orig.d_log_surface_pressure_d_t)
    else:
        ddivdtlin_orig, dtvdtlin_orig, dlnpsdtlin_orig = compute_linear_tendencies(
            state.divergence, state.temperature, state.log_surface_pressure
        )
        ddivspecdt_orig_nl = tends_orig.d_divergence_d_t - ddivdtlin_orig
        dtvspecdt_orig_nl = tends_orig.d_temperature_d_t - dtvdtlin_orig
        dlnpsspecdt_orig_nl = tends_orig.d_log_surface_pressure_d_t - dlnpsdtlin_orig

        div_expl = state.divergence + dt * (
            stepper_config.a21 * ddivspecdt_orig_nl
            + stepper_config.aa21 * ddivdtlin_orig
        )
        temp_expl = state.temperature + dt * (
            stepper_config.a21 * dtvspecdt_orig_nl + stepper_config.aa21 * dtvdtlin_orig
        )
        lnps_expl = state.log_surface_pressure + dt * (
            stepper_config.a21 * dlnpsspecdt_orig_nl
            + stepper_config.aa21 * dlnpsdtlin_orig
        )

        div1, temp1, lnps1 = solve_implicit(
            div_expl, temp_expl, lnps_expl, stepper_config.aa22, 0
        )

        # Override tendencies with nonlinear part for subsequent stages
        tends_orig = SpectralTendencies(
            d_vorticity_d_t=tends_orig.d_vorticity_d_t,
            d_divergence_d_t=ddivspecdt_orig_nl,
            d_temperature_d_t=dtvspecdt_orig_nl,
            d_log_surface_pressure_d_t=dlnpsspecdt_orig_nl,
            d_tracers_d_t=tends_orig.d_tracers_d_t,
        )

    state1 = SpectralState(
        vorticity=vort1,
        divergence=div1,
        temperature=temp1,
        log_surface_pressure=lnps1,
        tracers=tracers1,
    )

    # --- Stage 2 ---
    tends1 = get_spectral_tendencies(
        state1, phis_grads, dyn_config, trans_config, latitudes
    )

    vort2 = state.vorticity + dt * (
        stepper_config.a31 * tends_orig.d_vorticity_d_t
        + stepper_config.a32 * tends1.d_vorticity_d_t
    )
    tracers2 = state.tracers + dt * (
        stepper_config.a31 * tends_orig.d_tracers_d_t
        + stepper_config.a32 * tends1.d_tracers_d_t
    )

    if stepper_config.explicit:
        div2 = state.divergence + dt * (
            stepper_config.a31 * tends_orig.d_divergence_d_t
            + stepper_config.a32 * tends1.d_divergence_d_t
        )
        temp2 = state.temperature + dt * (
            stepper_config.a31 * tends_orig.d_temperature_d_t
            + stepper_config.a32 * tends1.d_temperature_d_t
        )
        lnps2 = state.log_surface_pressure + dt * (
            stepper_config.a31 * tends_orig.d_log_surface_pressure_d_t
            + stepper_config.a32 * tends1.d_log_surface_pressure_d_t
        )

        ddivdtlin1 = jnp.zeros_like(tends1.d_divergence_d_t)
        dtvdtlin1 = jnp.zeros_like(tends1.d_temperature_d_t)
        dlnpsdtlin1 = jnp.zeros_like(tends1.d_log_surface_pressure_d_t)
    else:
        ddivdtlin1, dtvdtlin1, dlnpsdtlin1 = compute_linear_tendencies(
            div1, temp1, lnps1
        )
        ddivspecdt1_nl = tends1.d_divergence_d_t - ddivdtlin1
        dtvspecdt1_nl = tends1.d_temperature_d_t - dtvdtlin1
        dlnpsspecdt1_nl = tends1.d_log_surface_pressure_d_t - dlnpsdtlin1

        div_expl = state.divergence + dt * (
            stepper_config.a31 * tends_orig.d_divergence_d_t
            + stepper_config.aa31 * ddivdtlin_orig
            + stepper_config.a32 * ddivspecdt1_nl
            + stepper_config.aa32 * ddivdtlin1
        )
        temp_expl = state.temperature + dt * (
            stepper_config.a31 * tends_orig.d_temperature_d_t
            + stepper_config.aa31 * dtvdtlin_orig
            + stepper_config.a32 * dtvspecdt1_nl
            + stepper_config.aa32 * dtvdtlin1
        )
        lnps_expl = state.log_surface_pressure + dt * (
            stepper_config.a31 * tends_orig.d_log_surface_pressure_d_t
            + stepper_config.aa31 * dlnpsdtlin_orig
            + stepper_config.a32 * dlnpsspecdt1_nl
            + stepper_config.aa32 * dlnpsdtlin1
        )

        div2, temp2, lnps2 = solve_implicit(
            div_expl, temp_expl, lnps_expl, stepper_config.aa33, 1
        )

        tends1 = SpectralTendencies(
            d_vorticity_d_t=tends1.d_vorticity_d_t,
            d_divergence_d_t=ddivspecdt1_nl,
            d_temperature_d_t=dtvspecdt1_nl,
            d_log_surface_pressure_d_t=dlnpsspecdt1_nl,
            d_tracers_d_t=tends1.d_tracers_d_t,
        )

    state2 = SpectralState(
        vorticity=vort2,
        divergence=div2,
        temperature=temp2,
        log_surface_pressure=lnps2,
        tracers=tracers2,
    )

    # --- Stage 3 ---
    tends2 = get_spectral_tendencies(
        state2, phis_grads, dyn_config, trans_config, latitudes
    )

    vort3 = state.vorticity + dt * (
        stepper_config.b1 * tends_orig.d_vorticity_d_t
        + stepper_config.b2 * tends1.d_vorticity_d_t
        + stepper_config.b3 * tends2.d_vorticity_d_t
    )
    tracers3 = state.tracers + dt * (
        stepper_config.b1 * tends_orig.d_tracers_d_t
        + stepper_config.b2 * tends1.d_tracers_d_t
        + stepper_config.b3 * tends2.d_tracers_d_t
    )

    if stepper_config.explicit:
        div3 = state.divergence + dt * (
            stepper_config.b1 * tends_orig.d_divergence_d_t
            + stepper_config.b2 * tends1.d_divergence_d_t
            + stepper_config.b3 * tends2.d_divergence_d_t
        )
        temp3 = state.temperature + dt * (
            stepper_config.b1 * tends_orig.d_temperature_d_t
            + stepper_config.b2 * tends1.d_temperature_d_t
            + stepper_config.b3 * tends2.d_temperature_d_t
        )
        lnps3 = state.log_surface_pressure + dt * (
            stepper_config.b1 * tends_orig.d_log_surface_pressure_d_t
            + stepper_config.b2 * tends1.d_log_surface_pressure_d_t
            + stepper_config.b3 * tends2.d_log_surface_pressure_d_t
        )
    else:
        ddivdtlin2, dtvdtlin2, dlnpsdtlin2 = compute_linear_tendencies(
            div2, temp2, lnps2
        )
        ddivspecdt2_nl = tends2.d_divergence_d_t - ddivdtlin2
        dtvspecdt2_nl = tends2.d_temperature_d_t - dtvdtlin2
        dlnpsspecdt2_nl = tends2.d_log_surface_pressure_d_t - dlnpsdtlin2

        div_expl = state.divergence + dt * (
            stepper_config.b1 * tends_orig.d_divergence_d_t
            + stepper_config.bb1 * ddivdtlin_orig
            + stepper_config.b2 * tends1.d_divergence_d_t
            + stepper_config.bb2 * ddivdtlin1
            + stepper_config.b3 * ddivspecdt2_nl
            + stepper_config.bb3 * ddivdtlin2
        )
        temp_expl = state.temperature + dt * (
            stepper_config.b1 * tends_orig.d_temperature_d_t
            + stepper_config.bb1 * dtvdtlin_orig
            + stepper_config.b2 * tends1.d_temperature_d_t
            + stepper_config.bb2 * dtvdtlin1
            + stepper_config.b3 * dtvspecdt2_nl
            + stepper_config.bb3 * dtvdtlin2
        )
        lnps_expl = state.log_surface_pressure + dt * (
            stepper_config.b1 * tends_orig.d_log_surface_pressure_d_t
            + stepper_config.bb1 * dlnpsdtlin_orig
            + stepper_config.b2 * tends1.d_log_surface_pressure_d_t
            + stepper_config.bb2 * dlnpsdtlin1
            + stepper_config.b3 * dlnpsspecdt2_nl
            + stepper_config.bb3 * dlnpsdtlin2
        )

        # Always run the implicit solve for the final stage.
        # When bb4 == 0 the solve reduces to the identity (d_hyb_m == I),
        # so this is safe and avoids a JIT-incompatible Python-level branch
        # on a traced scalar.
        div3, temp3, lnps3 = solve_implicit(
            div_expl, temp_expl, lnps_expl, stepper_config.bb4, 2
        )

    # Forward implicit linear damping/diffusion
    if stepper_config.disspec is not None and stepper_config.diff_prof is not None:
        disspec = stepper_config.disspec[None, :, :]
        diff_prof = stepper_config.diff_prof[:, None, None]
        dmp_prof = (
            stepper_config.dmp_prof[:, None, None]
            if stepper_config.dmp_prof is not None
            else jnp.zeros_like(diff_prof)
        )

        # denom = 1.0 - (disspec * diff_prof - dmp_prof) * dt
        denom_vort_div = 1.0 - (disspec * diff_prof - dmp_prof) * dt
        denom_temp_tracers = 1.0 - (disspec * diff_prof) * dt

        vort3 = vort3 / denom_vort_div
        div3 = div3 / denom_vort_div
        temp3 = temp3 / denom_temp_tracers
        tracers3 = tracers3 / denom_temp_tracers[None, :, :, :]

    final_state = SpectralState(
        vorticity=vort3,
        divergence=div3,
        temperature=temp3,
        log_surface_pressure=lnps3,
        tracers=tracers3,
    )

    # Post-RK dry-mass fixer: applied after the complete IMEX RK scheme,
    # matching Fortran run.f90 lines 357-360 (`if (massfix) then`).
    # This must NOT run inside the RK stages — doing so corrupts the implicit
    # gravity-wave solve by replacing the physical lnps tendency with ~0.
    if gauss_weights is not None and pdryini is not None:
        grid3, _ = spectral_to_grid(final_state, trans_config)
        press_diag = compute_pressure_diagnostics(
            grid3.log_surface_pressure.real, dyn_config
        )
        fixer_tend = compute_dry_mass_fixer(
            ps=press_diag.ps,
            tracers=grid3.tracers,
            dp=press_diag.dp,
            log_surface_pressure=final_state.log_surface_pressure,
            lnps_spectral_tend=jnp.zeros_like(final_state.log_surface_pressure),
            gauss_weights=gauss_weights,
            pdryini=pdryini,
            g=dyn_config.g,
            dt=dt,
            ntrunc=trans_config.truncation,
        )
        final_state = SpectralState(
            vorticity=final_state.vorticity,
            divergence=final_state.divergence,
            temperature=final_state.temperature,
            log_surface_pressure=final_state.log_surface_pressure + dt * fixer_tend,
            tracers=final_state.tracers,
        )

    return final_state
