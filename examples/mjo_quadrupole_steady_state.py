"""MJO upper-level quadrupole as a nonlinear steady-state solve on the GFS JAX core.

Reproduces the central result of Monteiro, Adames, Wallace & Sukhatme (2014,
GRL, "Interpreting the upper level structure of the Madden-Julian oscillation"):
impose an equator-straddling tropical heating and sweep a zonally-symmetric
subtropical westerly jet from rest toward realistic strength.  The upper-level
eddy response is expected to transition from the equatorially-trapped
Matsuno-Gill pattern, through a tilted Rossby wave train, into a compact
quadrupole of flanking Rossby gyres near 28 deg N/S as the jet strengthens.

Method (the point of the exercise): instead of time-integrating the dycore to a
steady state, we find the steady state DIRECTLY by minimising the total-tendency
residual ||F(X) + G(X)||^2 with a gradient-based optimizer (L-BFGS), taking the
gradient through the differentiable JAX core with jax.grad.  No time-stepping.

See docs/superpowers/specs/2026-06-12-mjo-quadrupole-steady-state-design.md.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")

import argparse
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optax
from sympl import get_constant, set_constant

import climt  # noqa: F401  (registers constants / grid helpers)
from gfs_dynamical_core.component_jax import GFSDynamicsJAX
from gfs_dynamical_core.jax.dynamics import DynamicsConfig, get_spectral_tendencies
from gfs_dynamical_core.jax.states import GridState, SpectralState
from gfs_dynamical_core.jax.transforms import (
    TransformConfig,
    get_gaussian_latitudes,
    grid_to_spectral,
    prebuild_kernels,
    s2_inverse,
    spectral_to_grid,
)

DAY = 86400.0


# ---------------------------------------------------------------------------
# Model configuration (constants + vertical coordinate from climt, as in
# examples/baroclinic_wave_jax.py, so the dycore is built exactly as in
# production use).
# ---------------------------------------------------------------------------
def build_config(L, n_lev=20):
    set_constant("reference_air_pressure", value=1e5, units="Pa")
    n_lon = 2 * L - 1
    n_lat = L
    grid = climt.get_grid(nx=n_lon, ny=n_lat, nz=n_lev)
    dycore = GFSDynamicsJAX()
    state = climt.get_default_state([dycore], grid_state=grid)

    ak = jnp.asarray(
        state["atmosphere_hybrid_sigma_pressure_a_coordinate_on_interface_levels"]
    )
    bk = jnp.asarray(
        state["atmosphere_hybrid_sigma_pressure_b_coordinate_on_interface_levels"]
    )
    dbk = bk[:-1] - bk[1:]
    ck = ak[:-1] * bk[1:] - ak[1:] * bk[:-1]
    rd = get_constant("gas_constant_of_dry_air", "J kg^-1 K^-1")
    cp = get_constant("heat_capacity_of_dry_air_at_constant_pressure", "J kg^-1 K^-1")

    dyn = DynamicsConfig(
        ak=ak, bk=bk, ck=ck, dbk=dbk, rk=rd / cp,
        toa_pressure=get_constant("top_of_model_pressure", "Pa"),
        radius=get_constant("planetary_radius", "m"),
        omega=get_constant("planetary_rotation_rate", "s^-1"),
        g=get_constant("gravitational_acceleration", "m s^-2"),
        rd=rd, rv=get_constant("gas_constant_of_vapor_phase", "J kg^-1 K^-1"),
        cp=cp, cvap=get_constant("heat_capacity_of_vapor_phase", "J kg^-1 K^-1"),
    )
    trans = TransformConfig(L=L, sampling="gl", radius=dyn.radius)
    prebuild_kernels(L, "gl")

    lats = get_gaussian_latitudes(L)            # radians, (n_lat,), N->S
    lons = 2.0 * np.pi * np.arange(n_lon) / n_lon  # radians, [0, 2pi)

    # Layer-mid sigma (bottom-to-top), reference ps = 1e5 Pa.
    ps_ref = 1e5
    sig_i = np.asarray(ak) / ps_ref + np.asarray(bk)
    sigma = 0.5 * (sig_i[:-1] + sig_i[1:])      # (n_lev,)
    return dyn, trans, lats, lons, sigma, n_lat, n_lon, n_lev


# ---------------------------------------------------------------------------
# Background jet, reference temperature, and heating anomaly.
# ---------------------------------------------------------------------------
def jet_profile(lats_rad, sigma):
    """Zonally + equatorially symmetric westerly jet, peak ~30 deg, upper-trop."""
    phi = lats_rad[None, :]                      # (1, n_lat) radians
    phi_jet = np.deg2rad(30.0)
    width = np.deg2rad(15.0)
    J = jnp.exp(-(((jnp.abs(phi) - phi_jet) / width) ** 2))   # (1, n_lat)
    V = jnp.exp(-(((sigma[:, None] - 0.20) / 0.12) ** 2))     # (n_lev, 1)
    return V * J                                 # (n_lev, n_lat), unit amplitude


SIGMA_TROP = 0.2          # tropopause at ~200 hPa (ps ref 1000 hPa)
T_SURF = 300.0
T_STRAT = 200.0


def reference_column(sigma):
    """Reference sounding: constant-lapse troposphere, ISOTHERMAL above 200 hPa.

    Linear in log-pressure from T_SURF at the surface to T_STRAT at the 200 hPa
    tropopause, then isothermal (T_STRAT) in the stratosphere.  The lid sets the
    vertical-mode structure so the heating can project onto the first baroclinic
    mode.
    """
    sig = jnp.asarray(sigma)
    zt = jnp.log(sig) / np.log(SIGMA_TROP)       # 0 at surface, 1 at 200 hPa, >1 above
    T_trop = T_SURF + (T_STRAT - T_SURF) * zt
    return jnp.where(sig >= SIGMA_TROP, T_trop, T_STRAT)     # (n_lev,)


def baroclinic_vertical_profile(sigma):
    """First-baroclinic-mode heating structure: half-sine in log-pressure confined
    to the troposphere (zero at surface and at 200 hPa, zero in the stratosphere),
    peaking near 450 hPa.  Single hump -> projects onto the first baroclinic mode."""
    sig = jnp.asarray(sigma)
    zt = jnp.log(sig) / np.log(SIGMA_TROP)
    W = jnp.sin(np.pi * zt)
    return jnp.where(sig >= SIGMA_TROP, W, 0.0)


def balanced_temperature(U_bg, sigma, lats_rad, dyn):
    """Temperature in gradient-wind + hydrostatic balance with the zonal jet.

    Integrates the meridional momentum balance for a steady, zonally-symmetric,
    v=0 state to get the balance-induced geopotential, then recovers temperature
    hydrostatically.  This makes the background a near-steady state of the dycore
    (F(X_bg) ~ 0), so the optimizer's residual is dominated by the heating rather
    than by the jet's own gradient-wind imbalance.

        dPhi/dphi = -(a*f + u*tan phi) * u            (gradient-wind, v=0)
        T = T_base - (1/Rd) * d(Phi_pert)/d(ln sigma)  (hydrostatic, dlnp~dlnsigma)

    Returns (n_lev, n_lat).
    """
    from scipy.integrate import cumulative_trapezoid

    a, Om, R = dyn.radius, dyn.omega, dyn.rd
    phi = np.asarray(lats_rad)                    # (n_lat,) N->S
    u = np.asarray(U_bg)                          # (n_lev, n_lat)
    sig = np.asarray(sigma)                       # (n_lev,)
    f = 2.0 * Om * np.sin(phi)
    dPhidphi = -(a * f[None, :] + u * np.tan(phi)[None, :]) * u   # (n_lev, n_lat)

    order = np.argsort(phi)                       # ascending S->N for integration
    phi_s = phi[order]
    G = cumulative_trapezoid(dPhidphi[:, order], phi_s, axis=1, initial=0.0)
    # Pin the balance-induced geopotential to zero at the equator.
    G0 = np.array([np.interp(0.0, phi_s, G[k]) for k in range(G.shape[0])])
    Phi_pert = (G - G0[:, None])[:, np.argsort(order)]            # back to N->S

    T_base = np.asarray(reference_column(sig))
    dPhi_dlnsig = np.gradient(Phi_pert, np.log(sig), axis=0)
    T = T_base[:, None] - dPhi_dlnsig / R
    return jnp.asarray(T)                         # (n_lev, n_lat)


def heating_anomaly(lats_rad, lons_rad, sigma, amp=2.0,
                    lon0_deg=90.0, Lx_deg=30.0, Ly_deg=10.0):
    """Equator-straddling Gaussian temperature anomaly, deep vertical profile.

    Zonal mean removed so the forcing is purely eddy (paper subtracts Q-bar)."""
    phi = lats_rad[:, None]                       # (n_lat, 1)
    lam = lons_rad[None, :]                        # (1, n_lon)
    lon0 = np.deg2rad(lon0_deg)
    Lx = np.deg2rad(Lx_deg)
    Ly = np.deg2rad(Ly_deg)
    dlam = jnp.arctan2(jnp.sin(lam - lon0), jnp.cos(lam - lon0))  # wrap to [-pi,pi]
    horiz = jnp.exp(-((dlam / Lx) ** 2) - ((phi / Ly) ** 2))      # (n_lat, n_lon)
    W = baroclinic_vertical_profile(sigma)        # first baroclinic mode, troposphere
    dT = amp * W[:, None, None] * horiz[None, :, :]   # (n_lev, n_lat, n_lon)
    dT = dT - jnp.mean(dT, axis=2, keepdims=True)     # remove zonal mean
    return dT


# ---------------------------------------------------------------------------
# Tendency, forcing, objective.
# ---------------------------------------------------------------------------
def make_solver(dyn, trans, lats, sigma, n_lat, n_lon, n_lev,
                tau_M_days=12.0, tau_T_days=12.0):
    tauM = tau_M_days * DAY
    tauT = tau_T_days * DAY
    phis_grads = (jnp.zeros((n_lat, n_lon)), jnp.zeros((n_lat, n_lon)))
    zeros3 = jnp.zeros((n_lev, n_lat, n_lon))
    tracers0 = jnp.zeros((1, n_lev, n_lat, n_lon))

    def theta_to_spec(theta):
        grid = GridState(
            u=theta["u"], v=theta["v"], temperature=theta["T"],
            vorticity=zeros3, divergence=zeros3,
            log_surface_pressure=theta["lnps"], tracers=tracers0,
        )
        return grid_to_spectral(grid, trans)

    def total_tendency(theta, vort_bg, Teq_spec, lnps_bg_spec):
        spec = theta_to_spec(theta)
        dyn_t = get_spectral_tendencies(spec, phis_grads, dyn, trans, lats)
        gv = -(spec.vorticity - vort_bg) / tauM
        gd = -(spec.divergence) / tauM
        gt = -(spec.temperature - Teq_spec) / tauT
        gl = -(spec.log_surface_pressure - lnps_bg_spec) / tauT
        return {
            "vort": dyn_t.d_vorticity_d_t + gv,
            "div": dyn_t.d_divergence_d_t + gd,
            "temp": dyn_t.d_temperature_d_t + gt,
            "lnps": dyn_t.d_log_surface_pressure_d_t + gl,
        }

    tracers0_spec = jnp.zeros((1, n_lev, trans.L, 2 * trans.L - 1), dtype=complex)

    def dict_to_spec(s):
        return SpectralState(
            vorticity=s["vort"], divergence=s["div"], temperature=s["temp"],
            log_surface_pressure=s["lnps"], tracers=tracers0_spec,
        )

    def total_tendency_spec(s, vort_bg, Teq_spec, lnps_bg_spec):
        """Square (state -> tendency, same pytree) form for the linearized solve."""
        spec = dict_to_spec(s)
        dyn_t = get_spectral_tendencies(spec, phis_grads, dyn, trans, lats)
        return {
            "vort": dyn_t.d_vorticity_d_t - (s["vort"] - vort_bg) / tauM,
            "div": dyn_t.d_divergence_d_t - (s["div"]) / tauM,
            "temp": dyn_t.d_temperature_d_t - (s["temp"] - Teq_spec) / tauT,
            "lnps": dyn_t.d_log_surface_pressure_d_t - (s["lnps"] - lnps_bg_spec) / tauT,
        }

    return theta_to_spec, total_tendency, total_tendency_spec, dict_to_spec, tauM, tauT


def build_background(U_max, dyn, trans, lats, sigma, n_lat, n_lon, n_lev,
                     theta_to_spec):
    """Background spectral fields and the bare-jet theta for a given jet strength."""
    U_bg = U_max * jet_profile(lats, sigma)                 # (n_lev, n_lat)
    u_bg = jnp.broadcast_to(U_bg[:, :, None], (n_lev, n_lat, n_lon))
    v_bg = jnp.zeros((n_lev, n_lat, n_lon))
    # Temperature in gradient-wind balance with the jet (near-steady background).
    T_col = balanced_temperature(U_bg, sigma, lats, dyn)    # (n_lev, n_lat)
    T_bg = jnp.broadcast_to(T_col[:, :, None], (n_lev, n_lat, n_lon))
    lnps_bg = jnp.full((n_lat, n_lon), np.log(1e5))
    theta_bg = {"u": u_bg, "v": v_bg, "T": T_bg, "lnps": lnps_bg}

    spec_bg = theta_to_spec(theta_bg)
    dT = heating_anomaly(lats, jnp.asarray(2 * np.pi * np.arange(n_lon) / n_lon),
                         sigma)
    theta_eq = dict(theta_bg, T=T_bg + dT)
    Teq_spec = theta_to_spec(theta_eq).temperature
    return theta_bg, spec_bg.vorticity, Teq_spec, spec_bg.log_surface_pressure


def make_objective(total_tendency, theta0, bg, tauM, tauT):
    """Weighted residual objective.

    Each variable's tendency is normalised by a *fixed* physical scale so the
    objective is dimensionless and balanced across variables, and consistent
    across the jet sweep.  We measure each residual in units of "fraction of a
    natural state scale changed per relaxation time":
        n_k = tau_k * residual_k / scale_k ,
    with scales chosen from typical eddy magnitudes.  Fixed scales avoid the
    ill-conditioning of normalising by the initial residual, which collapses to
    ~0 for variables (e.g. vorticity at rest) that have no tendency at theta0.
    """
    vort_bg, Teq_spec, lnps_bg_spec = bg
    # Normalise each variable by its own initial mean-square tendency so the
    # *active* equations (divergence, temperature) carry equal weight, instead
    # of letting the stiff gravity-wave (divergence) residual dominate.  A floor
    # at 1e-3 of the largest block caps variables whose tendency is ~0 at theta0
    # (vorticity, lnps at rest) so their weight cannot blow up.
    r0 = total_tendency(theta0, vort_bg, Teq_spec, lnps_bg_spec)
    m = {k: float(jnp.mean(jnp.abs(v) ** 2)) for k, v in r0.items()}
    floor = 1e-3 * max(m.values())
    coef = {k: 1.0 / (m[k] + floor) for k in m}

    def objective(theta):
        r = total_tendency(theta, vort_bg, Teq_spec, lnps_bg_spec)
        return 0.5 * sum(coef[k] * jnp.sum(jnp.abs(r[k]) ** 2) for k in r)

    def max_residual(theta):
        r = total_tendency(theta, vort_bg, Teq_spec, lnps_bg_spec)
        return {k: float(jnp.max(jnp.abs(v))) for k, v in r.items()}

    return objective, max_residual


def lbfgs_minimize(objective, theta0, maxiter=400, tol=1e-8, verbose=True,
                   memory_size=30):
    opt = optax.lbfgs(memory_size=memory_size)
    value_and_grad = optax.value_and_grad_from_state(objective)

    @jax.jit
    def step(theta, state):
        value, grad = value_and_grad(theta, state=state)
        updates, state = opt.update(
            grad, state, theta, value=value, grad=grad, value_fn=objective
        )
        theta = optax.apply_updates(theta, updates)
        return theta, state, value

    state = opt.init(theta0)
    theta = theta0
    j0 = float(objective(theta0))
    for i in range(maxiter):
        theta, state, value = step(theta, state)
        if verbose and (i % 25 == 0 or i == maxiter - 1):
            print(f"    iter {i:4d}   J/J0 = {float(value) / j0:.3e}")
        if float(value) / j0 < tol:
            if verbose:
                print(f"    converged at iter {i}, J/J0 = {float(value) / j0:.3e}")
            break
    return theta


def _pnorm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return float(jnp.sqrt(sum(jnp.sum(jnp.abs(a) ** 2) for a in leaves)))


# Characteristic eddy magnitudes per variable (for Jacobi-style nondimensional
# preconditioning of the tangent-linear system — without it the gravity-wave
# fast manifold makes the operator condition number ~1e6 and GMRES stagnates).
_VAR_SCALE = {"vort": 1e-5, "div": 1e-5, "temp": 1.0, "lnps": 1e-2}


def linearized_response(F_spec, X0, restart=150, maxiter=60, tol=1e-8, verbose=True):
    """Steady eddy response from the TANGENT-LINEAR model about X0 (spec 3A).

    `tlm = dF/dX|_{X0}` comes for free from `jax.linearize`; we solve the square
    Newton system `tlm . x = -F(X0)` matrix-free with GMRES.  The system is
    nondimensionalized per variable (right scaling S, left scaling 1/S) so the
    fast (gravity-wave) and slow (Rossby) blocks are balanced — essential for
    Krylov convergence.  No physics linearization is hand-coded; `tlm` is exact
    JAX autodiff.  Returns X_steady = X0 + x.
    """
    y0, tlm = jax.linearize(F_spec, X0)          # y0 = F(X0); tlm(dx) = J @ dx
    S = _VAR_SCALE

    def Jt(y):                                    # scaled operator: (1/S) J (S y)
        Sy = {k: y[k] * S[k] for k in y}
        t = tlm(Sy)
        return {k: t[k] / S[k] for k in t}

    rhs = {k: -y0[k] / S[k] for k in y0}         # (1/S) (-F(X0))
    y, _ = jax.scipy.sparse.linalg.gmres(
        Jt, rhs, tol=tol, atol=0.0, restart=restart, maxiter=maxiter,
        solve_method="batched",
    )
    rel = _pnorm(jax.tree_util.tree_map(lambda a, c: a - c, Jt(y), rhs)) / (
        _pnorm(rhs) + 1e-300)
    if verbose:
        print(f"    GMRES relative residual = {rel:.3e}")
    x = {k: y[k] * S[k] for k in y}
    X_steady = jax.tree_util.tree_map(lambda a, c: a + c, X0, x)
    return X_steady, rel


# ---------------------------------------------------------------------------
# Diagnostics.
# ---------------------------------------------------------------------------
def geopotential_height(grid, dyn):
    """Hydrostatic geopotential height Z(lev,lat,lon) [m] from T and lnps.

    Integrates Phi = phis + Rd * sum_below(T dlnp) upward (phis = 0), layer
    midpoints at the half-level.  Returns Z = Phi/g.
    """
    ps = jnp.exp(grid.log_surface_pressure)                # (n_lat, n_lon)
    p_i = dyn.ak[:, None, None] + dyn.bk[:, None, None] * ps[None]   # interfaces
    dlnp = jnp.log(jnp.maximum(p_i[:-1], 1e-10) / jnp.maximum(p_i[1:], 1e-10))
    contrib = dyn.rd * grid.temperature * dlnp             # per-layer Phi increment
    phi_below = jnp.cumsum(contrib, axis=0) - contrib       # Phi at lower interface
    phi_layer = phi_below + 0.5 * contrib                   # layer midpoint
    return phi_layer / dyn.g


def eddy_fields_from_spec(spec, trans, dyn):
    """Full 3-D eddy fields (streamfunction, geopotential height, u, v) from a
    SpectralState — shared by the nonlinear (theta) and linear (gmres) solvers."""
    L = trans.L
    R = dyn.radius
    l_arr = jnp.arange(L)
    fac = jnp.where(l_arr > 0, -(R ** 2) / (l_arr * (l_arr + 1)), 0.0)
    psi_lm = fac[:, None] * spec.vorticity                 # (n_lev, L, 2L-1)
    psi_lm = psi_lm.at[..., L - 1].set(0.0)                # drop m=0 (zonal mean)
    psi = jax.vmap(lambda f: s2_inverse(f, L, trans.sampling))(psi_lm).real

    grid, _ = spectral_to_grid(spec, trans)
    Z = geopotential_height(grid, dyn)

    def eddy(field):
        return field - jnp.mean(field, axis=-1, keepdims=True)

    return {
        "psi3d": np.asarray(eddy(psi)),
        "Z3d": np.asarray(eddy(Z)),
        "u3d": np.asarray(eddy(grid.u)),
        "v3d": np.asarray(eddy(grid.v)),
    }


def compute_eddy_fields(theta, theta_to_spec, trans, dyn):
    """Full 3-D eddy fields from the nonlinear (grid theta) solution."""
    return eddy_fields_from_spec(theta_to_spec(theta), trans, dyn)


def psi_mjo_ratio(Z_eddy, u_eddy, v_eddy):
    """Paper's ratio: max|geopotential| / max|wind|  (geopotential = g*Z)."""
    g = 9.80665
    wind = np.sqrt(u_eddy ** 2 + v_eddy ** 2)
    return float(np.max(np.abs(g * Z_eddy)) / (np.max(wind) + 1e-30))


def plot_sweep(results, lons_rad, lats_rad, sigma, lev_idx, outfile):
    from matplotlib.patches import Ellipse

    lon = np.rad2deg(np.asarray(lons_rad))
    lat = np.rad2deg(np.asarray(lats_rad))
    n = len(results)
    skl, skj = max(1, len(lon) // 24), max(1, len(lat) // 16)
    fig, axes = plt.subplots(n, 2, figsize=(15, 2.7 * n), squeeze=False)
    col_titles = ["eddy streamfunction (rotational response)",
                  "eddy geopotential height"]
    col_keys = ["psi3d", "Z3d"]
    col_labels = ["streamfunction (m^2/s)", "geopotential height (m)"]
    for row, res in enumerate(results):
        ue, ve = res["u3d"][lev_idx], res["v3d"][lev_idx]
        for col, key in enumerate(col_keys):
            ax = axes[row, col]
            field = res[key][lev_idx]
            m = np.max(np.abs(field)) + 1e-30
            c = ax.contourf(lon, lat, field, levels=np.linspace(-m, m, 21),
                            cmap="RdBu_r", extend="both")
            ax.quiver(lon[::skl], lat[::skj],
                      ue[::skj, ::skl], ve[::skj, ::skl],
                      width=0.002, color="k", alpha=0.55)
            ax.axhline(28, color="g", lw=0.6, ls="--")
            ax.axhline(-28, color="g", lw=0.6, ls="--")
            ax.add_patch(Ellipse((90, 0), 60, 20, fill=False,
                                 edgecolor="turquoise", lw=1.5))
            ax.set_ylim(-80, 80)
            ax.set_ylabel("lat")
            fig.colorbar(c, ax=ax, label=col_labels[col])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11)
        axes[row, 0].text(-0.16, 0.5,
                          f"U_max = {res['U_max']:.0f} m/s\n"
                          f"$\\Phi$/wind = {res['psi_mjo']:.0f}",
                          transform=axes[row, 0].transAxes, rotation=90,
                          va="center", ha="center", fontsize=10)
    for col in range(2):
        axes[-1, col].set_xlabel("longitude")
    fig.suptitle(
        f"Upper-level (sigma~{sigma[lev_idx]:.2f}) eddy response to equatorial "
        "heating at 90E vs subtropical jet strength\n"
        "nonlinear steady state via L-BFGS through the GFS JAX primitive-equation "
        "core (deep heating, full 3-D)",
        y=1.005, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(outfile, dpi=130, bbox_inches="tight")
    print(f"\nFigure written to {outfile}")


def plot_vertical(results, lons_rad, lats_rad, sigma, outfile):
    """Vertical structure: forcing/basic-state profiles + response cross-sections."""
    lon = np.rad2deg(np.asarray(lons_rad))
    lat = np.rad2deg(np.asarray(lats_rad))
    sig = np.asarray(sigma)
    p = sig * 1000.0                                  # approx pressure (hPa)
    res = results[-1]                                 # strongest jet
    u3d = res["u3d"]                                  # (n_lev, n_lat, n_lon)
    jeq = int(np.argmin(np.abs(lat)))                 # equator row
    ilon90 = int(np.argmin(np.abs(lon - 90.0)))       # heating longitude

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # (a) forcing + basic-state vertical profiles
    ax = axes[0]
    W = np.asarray(baroclinic_vertical_profile(sig))
    Tb = np.asarray(reference_column(sig))
    ax.plot(W / (np.max(np.abs(W)) + 1e-30), p, "r-o", ms=3, label="heating (mode 1)")
    ax.invert_yaxis()
    ax.set_ylabel("pressure (hPa)")
    ax.set_xlabel("normalized heating")
    ax.axhline(200, color="grey", ls="--", lw=0.8)
    ax.set_title("(a) first-baroclinic heating &\nbasic-state T")
    axT = ax.twiny()
    axT.plot(Tb, p, "b-s", ms=3, label="T_base")
    axT.set_xlabel("T_base (K)", color="b")
    axT.tick_params(axis="x", colors="b")
    ax.legend(loc="lower right", fontsize=8)

    # (b) equatorial longitude-pressure section of eddy u
    ax = axes[1]
    sec = u3d[:, jeq, :]
    m = np.max(np.abs(sec)) + 1e-30
    c = ax.contourf(lon, p, sec, levels=np.linspace(-m, m, 21), cmap="RdBu_r",
                    extend="both")
    ax.invert_yaxis()
    ax.axhline(200, color="grey", ls="--", lw=0.8)
    ax.axvline(90, color="turquoise", lw=1.2)
    ax.set_xlabel("longitude")
    ax.set_ylabel("pressure (hPa)")
    ax.set_title(f"(b) eddy u at equator  (U_max={res['U_max']:.0f} m/s)\n"
                 "baroclinic = sign reversal with height")
    fig.colorbar(c, ax=ax, label="eddy u (m/s)")

    # (c) meridional latitude-pressure section of eddy u at 90E
    ax = axes[2]
    sec = u3d[:, :, ilon90]
    m = np.max(np.abs(sec)) + 1e-30
    c = ax.contourf(lat, p, sec, levels=np.linspace(-m, m, 21), cmap="RdBu_r",
                    extend="both")
    ax.invert_yaxis()
    ax.axhline(200, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("latitude")
    ax.set_ylabel("pressure (hPa)")
    ax.set_title(f"(c) eddy u at 90E  (U_max={res['U_max']:.0f} m/s)")
    fig.colorbar(c, ax=ax, label="eddy u (m/s)")

    fig.tight_layout()
    fig.savefig(outfile, dpi=130, bbox_inches="tight")
    print(f"Vertical-structure figure written to {outfile}")


def plot_compare(res_lin, res_nl, lons_rad, lats_rad, sigma, lev_idx, outfile):
    """Side-by-side eddy streamfunction: linear (TLM/GMRES) vs nonlinear (L-BFGS)."""
    from matplotlib.patches import Ellipse

    lon = np.rad2deg(np.asarray(lons_rad))
    lat = np.rad2deg(np.asarray(lats_rad))
    n = len(res_lin)
    skl, skj = max(1, len(lon) // 24), max(1, len(lat) // 16)
    fig, axes = plt.subplots(n, 2, figsize=(15, 2.7 * n), squeeze=False)
    cols = [("linearized (tangent-linear + GMRES)", res_lin),
            ("nonlinear (L-BFGS residual min)", res_nl)]
    for row in range(n):
        for col, (title, res) in enumerate(cols):
            ax = axes[row, col]
            r = res[row]
            psi = r["psi3d"][lev_idx]
            m = np.max(np.abs(psi)) + 1e-30
            c = ax.contourf(lon, lat, psi, levels=np.linspace(-m, m, 21),
                            cmap="RdBu_r", extend="both")
            ax.quiver(lon[::skl], lat[::skj],
                      r["u3d"][lev_idx][::skj, ::skl],
                      r["v3d"][lev_idx][::skj, ::skl],
                      width=0.002, color="k", alpha=0.55)
            ax.axhline(28, color="g", lw=0.6, ls="--")
            ax.axhline(-28, color="g", lw=0.6, ls="--")
            ax.add_patch(Ellipse((90, 0), 60, 20, fill=False,
                                 edgecolor="turquoise", lw=1.5))
            ax.set_ylim(-80, 80)
            ax.set_ylabel("lat")
            fig.colorbar(c, ax=ax, label="streamfunction (m^2/s)")
            if row == 0:
                ax.set_title(title, fontsize=11)
        # relative L2 difference of the streamfunction between the two solutions
        a, b = res_lin[row]["psi3d"], res_nl[row]["psi3d"]
        reldiff = np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30)
        axes[row, 0].text(-0.17, 0.5,
                          f"U_max = {res_lin[row]['U_max']:.0f} m/s\n"
                          f"||nl-lin||/||lin|| = {reldiff:.2f}",
                          transform=axes[row, 0].transAxes, rotation=90,
                          va="center", ha="center", fontsize=9)
    for col in range(2):
        axes[-1, col].set_xlabel("longitude")
    fig.suptitle(
        f"Linear vs nonlinear steady eddy streamfunction (sigma~{sigma[lev_idx]:.2f}) "
        "— same heating, jet, damping\n"
        "tangent-linear GMRES solve about the background vs full nonlinear L-BFGS, "
        "GFS JAX core",
        y=1.005, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(outfile, dpi=130, bbox_inches="tight")
    print(f"\nComparison figure written to {outfile}")


def compare_linear_nonlinear(args, dyn, trans, lats, lons, sigma, n_lat, n_lon,
                             n_lev, lev_idx, theta_to_spec, total_tendency,
                             total_tendency_spec, dict_to_spec, tauM, tauT):
    """Solve both the tangent-linear (GMRES, 3A) and nonlinear (L-BFGS) steady
    states for each jet and compare."""
    res_lin, res_nl = [], []
    eddy_prev = None
    for U_max in args.u_sweep:
        print(f"\n=== U_max = {U_max:.0f} m/s ===")
        theta_bg, vort_bg, Teq_spec, lnps_bg_spec = build_background(
            U_max, dyn, trans, lats, sigma, n_lat, n_lon, n_lev, theta_to_spec
        )

        # ---- linearized (tangent-linear about the background) : F_spec(X)=0 ----
        spec_bg = theta_to_spec(theta_bg)
        X0 = {"vort": spec_bg.vorticity, "div": spec_bg.divergence,
              "temp": spec_bg.temperature, "lnps": spec_bg.log_surface_pressure}
        F_spec = lambda s: total_tendency_spec(s, vort_bg, Teq_spec, lnps_bg_spec)
        print("  [linear] tangent-linear GMRES solve ...")
        X_lin, _ = linearized_response(F_spec, X0)
        ef_lin = eddy_fields_from_spec(dict_to_spec(X_lin), trans, dyn)
        ef_lin["U_max"] = U_max
        ef_lin["psi_mjo"] = psi_mjo_ratio(ef_lin["Z3d"][lev_idx],
                                          ef_lin["u3d"][lev_idx],
                                          ef_lin["v3d"][lev_idx])
        res_lin.append(ef_lin)

        # ---- nonlinear (full residual minimization) ----
        print("  [nonlinear] L-BFGS residual minimization ...")
        theta0 = theta_bg if eddy_prev is None else {
            k: theta_bg[k] + eddy_prev[k] for k in theta_bg}
        objective, _ = make_objective(
            total_tendency, theta_bg, (vort_bg, Teq_spec, lnps_bg_spec), tauM, tauT)
        theta = lbfgs_minimize(objective, theta0, maxiter=args.maxiter, tol=1e-3)
        eddy_prev = {k: theta[k] - theta_bg[k] for k in theta_bg}
        ef_nl = compute_eddy_fields(theta, theta_to_spec, trans, dyn)
        ef_nl["U_max"] = U_max
        ef_nl["psi_mjo"] = psi_mjo_ratio(ef_nl["Z3d"][lev_idx],
                                         ef_nl["u3d"][lev_idx],
                                         ef_nl["v3d"][lev_idx])
        res_nl.append(ef_nl)

        reldiff = np.linalg.norm(ef_lin["psi3d"] - ef_nl["psi3d"]) / (
            np.linalg.norm(ef_lin["psi3d"]) + 1e-30)
        print(f"    psi_MJO  linear={ef_lin['psi_mjo']:.1f}  "
              f"nonlinear={ef_nl['psi_mjo']:.1f}   ||nl-lin||/||lin||={reldiff:.3f}")

    print("\n  U_max   psi_MJO(lin)  psi_MJO(nl)   reldiff")
    for rl, rn in zip(res_lin, res_nl):
        rd = np.linalg.norm(rl["psi3d"] - rn["psi3d"]) / (
            np.linalg.norm(rl["psi3d"]) + 1e-30)
        print(f"  {rl['U_max']:5.0f}   {rl['psi_mjo']:10.1f}  "
              f"{rn['psi_mjo']:10.1f}   {rd:7.3f}")

    out = args.out.replace(".png", "_compare.png")
    plot_compare(res_lin, res_nl, lons, lats, sigma, lev_idx, out)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--u-sweep", type=float, nargs="+",
                    default=[0.0, 8.0, 16.0, 24.0, 30.0])
    ap.add_argument("--tau-m", type=float, default=12.0, help="momentum relax (days)")
    ap.add_argument("--tau-t", type=float, default=12.0, help="thermal relax (days)")
    ap.add_argument("--plot-sigma", type=float, default=0.25,
                    help="sigma level for the horizontal maps (~250 hPa)")
    ap.add_argument("--out", type=str, default="mjo_quadrupole_sweep.png")
    ap.add_argument("--replot", type=str, default=None,
                    help="re-plot from a saved .npz without solving")
    ap.add_argument("--compare", action="store_true",
                    help="also solve the linearized (TLM/GMRES) response and "
                         "compare with the nonlinear one")
    args = ap.parse_args()

    if args.replot:
        replot(args.replot, args.out, plot_sigma=args.plot_sigma)
        return

    print(f"Building config (L={args.L}) ...")
    dyn, trans, lats, lons, sigma, n_lat, n_lon, n_lev = build_config(args.L)
    lev_idx = int(np.argmin(np.abs(np.asarray(sigma) - args.plot_sigma)))
    print(f"Upper-level diagnostic at level {lev_idx} (sigma={sigma[lev_idx]:.3f})")

    theta_to_spec, total_tendency, total_tendency_spec, dict_to_spec, tauM, tauT = (
        make_solver(dyn, trans, lats, sigma, n_lat, n_lon, n_lev,
                    tau_M_days=args.tau_m, tau_T_days=args.tau_t)
    )

    if args.compare:
        compare_linear_nonlinear(
            args, dyn, trans, lats, lons, sigma, n_lat, n_lon, n_lev, lev_idx,
            theta_to_spec, total_tendency, total_tendency_spec, dict_to_spec,
            tauM, tauT,
        )
        return

    results = []
    eddy_prev = None
    for U_max in args.u_sweep:
        print(f"\n=== U_max = {U_max:.0f} m/s ===")
        theta_bg, vort_bg, Teq_spec, lnps_bg_spec = build_background(
            U_max, dyn, trans, lats, sigma, n_lat, n_lon, n_lev, theta_to_spec
        )
        # Warm start: carry the previous eddy onto the new background jet.
        if eddy_prev is None:
            theta0 = theta_bg
        else:
            theta0 = {k: theta_bg[k] + eddy_prev[k] for k in theta_bg}

        objective, max_residual = make_objective(
            total_tendency, theta_bg, (vort_bg, Teq_spec, lnps_bg_spec), tauM, tauT
        )
        theta = lbfgs_minimize(objective, theta0, maxiter=args.maxiter, tol=1e-3)

        eddy_prev = {k: theta[k] - theta_bg[k] for k in theta_bg}
        mr = max_residual(theta)
        print(f"    max |residual|: " +
              ", ".join(f"{k}={v:.2e}" for k, v in mr.items()))

        ef = compute_eddy_fields(theta, theta_to_spec, trans, dyn)
        ratio = psi_mjo_ratio(ef["Z3d"][lev_idx], ef["u3d"][lev_idx],
                              ef["v3d"][lev_idx])
        ef.update({"U_max": U_max, "psi_mjo": ratio})
        results.append(ef)
        print(f"    psi_MJO = {ratio:.1f}")

    print("\n  U_max(m/s)   psi_MJO")
    for r in results:
        print(f"   {r['U_max']:6.0f}     {r['psi_mjo']:8.1f}")

    # Persist solved eddy fields so plotting can be re-run without re-solving.
    npz = args.out.replace(".png", ".npz")
    save = {"lons": np.asarray(lons), "lats": np.asarray(lats),
            "sigma": np.asarray(sigma), "lev_idx": lev_idx,
            "u_sweep": np.asarray([r["U_max"] for r in results]),
            "psi_mjo": np.asarray([r["psi_mjo"] for r in results])}
    for key in ("psi3d", "Z3d", "u3d", "v3d"):
        save[key] = np.stack([r[key] for r in results])
    np.savez_compressed(npz, **save)
    print(f"Solved fields saved to {npz}")

    plot_sweep(results, lons, lats, sigma, lev_idx, args.out)
    plot_vertical(results, lons, lats, sigma, args.out.replace(".png", "_vertical.png"))


def replot(npz_path, out, plot_sigma=None):
    """Re-make figures from a saved .npz without re-solving."""
    d = np.load(npz_path)
    lons, lats, sigma = d["lons"], d["lats"], d["sigma"]
    lev_idx = int(d["lev_idx"])
    if plot_sigma is not None:
        lev_idx = int(np.argmin(np.abs(np.asarray(sigma) - plot_sigma)))
    results = []
    for i, U in enumerate(d["u_sweep"]):
        results.append({"U_max": float(U), "psi_mjo": float(d["psi_mjo"][i]),
                        "psi3d": d["psi3d"][i], "Z3d": d["Z3d"][i],
                        "u3d": d["u3d"][i], "v3d": d["v3d"][i]})
    plot_sweep(results, lons, lats, sigma, lev_idx, out)
    plot_vertical(results, lons, lats, sigma, out.replace(".png", "_vertical.png"))


if __name__ == "__main__":
    main()
