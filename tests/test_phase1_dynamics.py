"""Phase 1 per-function dynamics tests.

Method (per ``debugging_code/test_harness_plan.md``): load Fortran's inputs
from ``debug_data/fortran_dyntend_call_NNNNN.bin``, run the JAX equivalent,
compare in grid space. Threshold: ``max|JAX - F| / max|F| < 1e-12``.

DynamicsConfig is built from the real climt default state (ak/bk and
constants) rather than mock values, so the config matches what produced the
dump bit-for-bit.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from fortran_loader import load_dyntend_dump

from gfs_dynamical_core.jax.dynamics import (
    DynamicsConfig,
    GridTendencies,
    VerticalVelocities,
    assemble_grid_tendencies,
    compute_energy_conversion,
    compute_pressure_diagnostics,
    compute_pressure_gradient_force,
    compute_vertical_advection,
    compute_vertical_velocities,
)
from gfs_dynamical_core.jax.states import GridGradients, GridState
from gfs_dynamical_core.jax.transforms import (
    TransformConfig,
    get_gaussian_latitudes,
    grid_to_spectral_tendencies,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
# Override with GFS_DUMP_DIR when the repo copy of the dumps has been
# regenerated at a different grid (any Fortran run from the repo root
# overwrites debug_data/fortran_dyntend_call_*.bin via the dump
# instrumentation in dyn_run.f90).
DUMP_DIR = Path(os.environ.get("GFS_DUMP_DIR", REPO_ROOT / "debug_data"))


def _build_config_from_climt(n_lev: int = 20, n_lat: int = 64, n_lon: int = 127):
    """Build a DynamicsConfig using the same climt default state that
    produced the dumps (see debugging_code/compare_full_run_cs.py)."""
    import climt
    from sympl import set_constant, get_constant
    from gfs_dynamical_core import GFSDynamicalCore

    set_constant("reference_air_pressure", value=1e5, units="Pa")
    grid = climt.get_grid(nx=n_lon, ny=n_lat, nz=n_lev)
    dycore = GFSDynamicalCore()
    state = climt.get_default_state([dycore], grid_state=grid)

    ak = jnp.asarray(np.asarray(
        state["atmosphere_hybrid_sigma_pressure_a_coordinate_on_interface_levels"]
    ))
    bk = jnp.asarray(np.asarray(
        state["atmosphere_hybrid_sigma_pressure_b_coordinate_on_interface_levels"]
    ))
    dbk = bk[:-1] - bk[1:]
    ck = ak[:-1] * bk[1:] - ak[1:] * bk[:-1]

    rd = get_constant("gas_constant_of_dry_air", "J kg^-1 K^-1")
    cp = get_constant("heat_capacity_of_dry_air_at_constant_pressure", "J kg^-1 K^-1")

    return DynamicsConfig(
        ak=ak,
        bk=bk,
        ck=ck,
        dbk=dbk,
        rk=rd / cp,
        toa_pressure=get_constant("top_of_model_pressure", "Pa"),
        radius=get_constant("planetary_radius", "m"),
        omega=get_constant("planetary_rotation_rate", "s^-1"),
        g=get_constant("gravitational_acceleration", "m s^-2"),
        rd=rd,
        rv=get_constant("gas_constant_of_vapor_phase", "J kg^-1 K^-1"),
        cp=cp,
        cvap=get_constant("heat_capacity_of_vapor_phase", "J kg^-1 K^-1"),
    )


@pytest.fixture(scope="module")
def dump1():
    path = DUMP_DIR / "fortran_dyntend_call_00001.bin"
    if not path.exists():
        pytest.skip(f"Fortran dump not present: {path}")
    return load_dyntend_dump(path)


@pytest.fixture(scope="module")
def dyn_config(dump1):
    h = dump1["header"]
    return _build_config_from_climt(n_lev=h.nlevs, n_lat=h.nlats, n_lon=h.nlons)


def _rel_err(jax_arr, f_arr, eps=1e-30):
    j = np.asarray(jax_arr)
    f = np.asarray(f_arr)
    denom = max(np.abs(f).max(), eps)
    return np.abs(j - f).max() / denom


def _max_abs(jax_arr, f_arr):
    return float(np.abs(np.asarray(jax_arr) - np.asarray(f_arr)).max())


def test_d1_1_pressure_diagnostics(dump1, dyn_config):
    """D1.1: pk, dp, prs, alfa, rlnp from JAX match Fortran dump at machine
    precision for all interior layers.

    Two intentional / benign TOA-only differences are documented (not bugs):

    * ``rlnp[-1]``: Fortran sets a sentinel ``99999.99`` (see
      ``pressure_data.f90:126``, "doesn't matter, should never be used");
      JAX sets 0.0. Assert the Fortran value is the sentinel and JAX is 0.
    * ``alfa[-1]``: both set to ``log(2)``, but Fortran uses the
      single-precision literal ``log(2.)`` (pressure_data.f90:125), so the
      stored value is ``float32(log(2))`` promoted to float64, giving
      ~1.9e-9 absolute difference vs JAX's double-precision ``jnp.log(2)``.
      JAX is the more precise representation.
    """
    lnps = jnp.asarray(dump1["grid_state"]["log_surface_pressure"])
    diag = compute_pressure_diagnostics(lnps, dyn_config)

    f = dump1["pressure"]

    tol = 1e-12

    assert _rel_err(diag.ps, f["ps"]) < tol, _rel_err(diag.ps, f["ps"])
    assert _rel_err(diag.pk, f["pk"]) < tol, _rel_err(diag.pk, f["pk"])
    assert _rel_err(diag.dp, f["dp"]) < tol, _rel_err(diag.dp, f["dp"])
    assert _rel_err(diag.prs, f["prs"]) < tol, _rel_err(diag.prs, f["prs"])

    # Interior layers: machine precision.
    assert _rel_err(diag.alfa[:-1], f["alfa"][:-1]) < tol, _rel_err(
        diag.alfa[:-1], f["alfa"][:-1]
    )
    assert _rel_err(diag.rlnp[:-1], f["rlnp"][:-1]) < tol, _rel_err(
        diag.rlnp[:-1], f["rlnp"][:-1]
    )

    # TOA alfa: both = log(2), Fortran stored as float32(log(2)) promoted.
    ja = np.asarray(diag.alfa[-1])
    fa = np.asarray(f["alfa"][-1])
    assert np.abs(ja - np.log(2.0)).max() < 1e-15
    assert np.abs(fa - np.float32(np.log(2.0))).max() < 1e-15

    # TOA rlnp: JAX=0, Fortran=99999.99 sentinel.
    assert np.allclose(diag.rlnp[-1], 0.0)
    assert np.asarray(f["rlnp"][-1]).max() > 1000.0


def _build_grid_state(dump) -> GridState:
    g = dump["grid_state"]
    return GridState(
        u=jnp.asarray(g["u"]),
        v=jnp.asarray(g["v"]),
        temperature=jnp.asarray(g["virtual_temperature"]),  # not used by D1.2
        vorticity=jnp.asarray(g["vorticity"]),
        divergence=jnp.asarray(g["divergence"]),
        log_surface_pressure=jnp.asarray(g["log_surface_pressure"]),
        tracers=jnp.asarray(g["tracers"]),
    )


def _build_grid_grads(dump) -> GridGradients:
    g = dump["gradients"]
    n_trac, n_lev, n_lat, n_lon = dump["grid_state"]["tracers"].shape
    return GridGradients(
        d_log_ps_d_phi=jnp.asarray(g["d_log_ps_d_phi"]),
        d_log_ps_d_lambda=jnp.asarray(g["d_log_ps_d_lambda"]),
        d_t_d_phi=jnp.asarray(g["d_t_d_phi"]),
        d_t_d_lambda=jnp.asarray(g["d_t_d_lambda"]),
        # Tracer gradients aren't dumped — unused by D1.2.
        d_tracers_d_phi=jnp.zeros((n_trac, n_lev, n_lat, n_lon)),
        d_tracers_d_lambda=jnp.zeros((n_trac, n_lev, n_lat, n_lon)),
    )


def test_d1_2_vertical_velocities(dump1, dyn_config):
    """D1.2: etadot, omega, d_log_ps_d_t from JAX match Fortran.

    Inputs are fed from the Fortran dump. Pressure diagnostics are computed
    by the JAX routine (verified in D1.1 to match Fortran except at the two
    documented TOA-only sentinel points). If this test reveals a top-layer
    discrepancy in etadot or omega that traces back to rlnp[-1] or alfa[-1],
    the TOA sentinel is leaking and must be audited.
    """
    grid_state = _build_grid_state(dump1)
    grid_grads = _build_grid_grads(dump1)
    lnps = grid_state.log_surface_pressure
    press_diag = compute_pressure_diagnostics(lnps, dyn_config)

    vv = compute_vertical_velocities(grid_state, grid_grads, press_diag, dyn_config)

    f = dump1["vertical_velocities"]

    tol = 1e-12
    err_dlnpsdt = _rel_err(vv.d_log_ps_d_t, f["d_log_ps_d_t"])
    err_etadot = _rel_err(vv.etadot, f["etadot"])

    assert err_dlnpsdt < tol, f"d_log_ps_d_t rel err = {err_dlnpsdt}"
    assert err_etadot < tol, f"etadot rel err = {err_etadot}"

    # Omega matches at machine precision in the interior; at the TOA layer
    # (k=-1) it inherits the documented D1.1 alfa[-1] float32(log(2))
    # artifact, producing ~2e-9 relative error. Assert interior matches at
    # machine precision and TOA residual is bounded by the artifact.
    err_omega_interior = _rel_err(vv.omega[:-1], f["omega"][:-1])
    assert err_omega_interior < tol, (
        f"omega interior rel err = {err_omega_interior}"
    )
    err_omega_toa = _rel_err(vv.omega[-1:], f["omega"][-1:])
    assert err_omega_toa < 1e-8, (
        f"omega TOA rel err = {err_omega_toa} — exceeds the expected "
        "alfa float32 artifact bound (~3e-9). Investigate."
    )


@pytest.mark.parametrize(
    "field,data_key,vadv_key",
    [
        ("u", "u", "vadv_u"),
        ("v", "v", "vadv_v"),
        ("T", "virtual_temperature", "vadv_t"),
    ],
)
def test_d1_3_to_d1_5_vertical_advection(dump1, field, data_key, vadv_key):
    """D1.3/D1.4/D1.5: vertical advection of u, v, T.

    All three call the same ``compute_vertical_advection`` helper. Inputs
    (data, etadot, dp) are taken from the Fortran dump so this isolates
    the advection routine from upstream pressure/vertical-velocity code.
    """
    data = jnp.asarray(dump1["grid_state"][data_key])
    etadot = jnp.asarray(dump1["vertical_velocities"]["etadot"])
    dp = jnp.asarray(dump1["pressure"]["dp"])

    vadv = compute_vertical_advection(data, etadot, dp)

    f_vadv = dump1["vertical_advection"][vadv_key]
    err = _rel_err(vadv, f_vadv)
    tol = 1e-12
    assert err < tol, f"{field} vadv rel err = {err}"


def test_d1_6_pressure_gradient_force(dump1, dyn_config):
    """D1.6: pgf_x, pgf_y from JAX match Fortran.

    Prime suspect per plan ("source of 4 prior bugs", top candidate for
    the 2.2% v-error). If this fails, decompose into the 5 additive terms
    (cofb_pressure, cofa, px2_factor, px3u/px3v, alfa term) to localise.

    Inputs from Fortran dump; pressure diagnostics from JAX (verified in
    D1.1). Surface geopotential gradients come from the Block-2 dump.
    """
    grid_grads = _build_grid_grads(dump1)
    virtual_temp = jnp.asarray(dump1["grid_state"]["virtual_temperature"])
    lnps = jnp.asarray(dump1["grid_state"]["log_surface_pressure"])
    press_diag = compute_pressure_diagnostics(lnps, dyn_config)
    phis_grads = (
        jnp.asarray(dump1["gradients"]["d_phis_d_lambda"]),
        jnp.asarray(dump1["gradients"]["d_phis_d_phi"]),
    )

    pgf_x, pgf_y = compute_pressure_gradient_force(
        virtual_temp, grid_grads, press_diag, dyn_config, phis_grads
    )

    f = dump1["pgf"]

    # pgf_x: for dry DCMIP initial conditions the zonal symmetry is near-exact
    # so max|pgf_x| ~ 1e-15 — relative error is meaningless. Compare in absolute
    # terms: require |JAX - F| below float-precision noise floor for values
    # of this magnitude.
    f_x_scale = max(float(np.abs(f["pgf_x"]).max()), 1e-15)
    abs_x = _max_abs(pgf_x, f["pgf_x"])
    assert abs_x < 1e-14, (
        f"pgf_x abs err = {abs_x} (field scale = {f_x_scale}, both ~noise). "
        "If the field scale grows at later steps this test should use "
        "relative error."
    )

    # pgf_y: interior layers machine precision. k=-1 (TOA) inherits the
    # documented D1.1 alfa[-1] float32(log(2)) artifact (verified by
    # substituting Fortran's alfa into the JAX PGF — TOA residual drops from
    # 2.4e-14 to 2.6e-18 at machine precision).
    err_y_interior = _rel_err(pgf_y[:-1], f["pgf_y"][:-1])
    assert err_y_interior < 1e-12, (
        f"pgf_y interior rel err = {err_y_interior}"
    )
    err_y_toa = _rel_err(pgf_y[-1:], f["pgf_y"][-1:])
    assert err_y_toa < 1e-11, (
        f"pgf_y TOA rel err = {err_y_toa} — exceeds expected alfa-artifact "
        "bound (~7e-12). Investigate."
    )


def test_d1_7_energy_conversion(dump1, dyn_config):
    """D1.7: kappa * omega * Tv / denom — energy conversion.

    Three-line formula: little scope for error, but easy to verify.
    """
    virtual_temp = jnp.asarray(dump1["grid_state"]["virtual_temperature"])
    omega = jnp.asarray(dump1["vertical_velocities"]["omega"])
    q = jnp.asarray(dump1["grid_state"]["tracers"][0])  # specific_humidity

    ec = compute_energy_conversion(omega, virtual_temp, q, dyn_config)

    err = _rel_err(ec, dump1["energy_conv"])
    assert err < 1e-12, f"energy_conv rel err = {err}"


def test_d1_8_tendency_assembly(dump1, dyn_config):
    """D1.8: u_flux, v_flux, temp_tend, ke from ``assemble_grid_tendencies``.

    This is the final grid-space step before the transform to spectral.
    All upstream inputs come from the Fortran dump (PGF, vertical velocities,
    energy_conv) so we isolate just the assembly arithmetic + the internal
    compute_vertical_advection call (already verified in D1.3-D1.5).
    """
    # grid_state: note temperature = virtual_temperature because Fortran
    # dynamics operate on Tv throughout and the dumped gradients are d(Tv).
    g = dump1["grid_state"]
    grid_state = GridState(
        u=jnp.asarray(g["u"]),
        v=jnp.asarray(g["v"]),
        temperature=jnp.asarray(g["virtual_temperature"]),
        vorticity=jnp.asarray(g["vorticity"]),
        divergence=jnp.asarray(g["divergence"]),
        log_surface_pressure=jnp.asarray(g["log_surface_pressure"]),
        tracers=jnp.asarray(g["tracers"]),
    )
    grid_grads = _build_grid_grads(dump1)

    # Use Fortran-dumped upstream quantities so upstream errors can't
    # contaminate this test.
    vv_f = dump1["vertical_velocities"]
    vvels = VerticalVelocities(
        omega=jnp.asarray(vv_f["omega"]),
        etadot=jnp.asarray(vv_f["etadot"]),
        d_log_ps_d_t=jnp.asarray(vv_f["d_log_ps_d_t"]),
    )
    # Pressure diagnostics: dp is needed for vadv; use dumped.
    lnps = grid_state.log_surface_pressure
    press_diag = compute_pressure_diagnostics(lnps, dyn_config)

    pgf = (
        jnp.asarray(dump1["pgf"]["pgf_x"]),
        jnp.asarray(dump1["pgf"]["pgf_y"]),
    )
    energy_conv = jnp.asarray(dump1["energy_conv"])

    n_lat = dump1["header"].nlats
    latitudes = get_gaussian_latitudes(n_lat)

    gt = assemble_grid_tendencies(
        grid_state, grid_grads, vvels, press_diag, pgf, energy_conv,
        dyn_config, latitudes,
    )

    f = dump1["grid_tendencies"]

    # ke = 0.5*(u^2+v^2) — trivial, exact.
    err_ke = _rel_err(gt.kinetic_energy, f["ke"])
    assert err_ke < 1e-12, f"ke rel err = {err_ke}"

    # temp_tend
    err_T = _rel_err(gt.temp_tend, f["temp_tend"])
    assert err_T < 1e-12, f"temp_tend rel err = {err_T}"

    # u_flux / v_flux — expect some accumulation; allow 1e-11 headroom.
    err_u = _rel_err(gt.u_flux, f["u_flux"])
    err_v = _rel_err(gt.v_flux, f["v_flux"])
    assert err_u < 1e-11, f"u_flux rel err = {err_u}"
    assert err_v < 1e-11, f"v_flux rel err = {err_v}"


def test_d1_9_spectral_tendencies(dump1, dyn_config):
    """D1.9: grid_to_spectral_tendencies — forward vector/scalar transforms
    of grid tendencies into spectral space.

    Feeds dumped grid tendencies (u_flux, v_flux, temp_tend, ke) and the
    dumped d_log_ps_d_t straight into ``grid_to_spectral_tendencies``, then
    compares against the dumped SHTNS-packed → s2fft-rectangular spectral
    tendencies. Isolates the forward transforms; upstream errors cannot
    contaminate.

    Tracer tendencies are not tested here — the Fortran dump does not
    include grid-space tracer tendencies, and the JAX path for tracers is a
    plain scalar forward transform (covered implicitly by ``d_temperature``).

    s2fft's forward transform has an O(1e-11) truncation floor for this
    grid size, so the tolerance is slightly looser than the grid-space
    D1.1-D1.8 tests.
    """
    f_grid = dump1["grid_tendencies"]
    f_vv = dump1["vertical_velocities"]
    f_grid_state = dump1["grid_state"]

    n_lat = dump1["header"].nlats
    n_lev = dump1["header"].nlevs

    # Build a minimal GridTendencies with dumped values for fields we test,
    # and zeros for tracer_tends (untested — no grid-space dump).
    ntrac = f_grid_state["tracers"].shape[0]
    n_lon = f_grid["u_flux"].shape[-1]
    grid_tends = GridTendencies(
        u_flux=jnp.asarray(f_grid["u_flux"]),
        v_flux=jnp.asarray(f_grid["v_flux"]),
        temp_tend=jnp.asarray(f_grid["temp_tend"]),
        log_ps_tend=jnp.asarray(f_vv["d_log_ps_d_t"]),
        tracer_tends=jnp.zeros((ntrac, n_lev, n_lat, n_lon)),
        kinetic_energy=jnp.asarray(f_grid["ke"]),
    )

    trans_config = TransformConfig(
        L=n_lat, sampling="gl", radius=dyn_config.radius, ntrunc=dump1["ntrunc"]
    )

    spec_jax = grid_to_spectral_tendencies(grid_tends, trans_config)
    f_spec = dump1["spectral_tendencies"]

    # s2fft forward-transform noise floor for this grid size is ~1e-11.
    tol = 1e-10

    err_vort = _rel_err(spec_jax.d_vorticity_d_t, f_spec["d_vorticity_d_t"])
    err_div = _rel_err(spec_jax.d_divergence_d_t, f_spec["d_divergence_d_t"])
    err_temp = _rel_err(spec_jax.d_temperature_d_t, f_spec["d_temperature_d_t"])
    err_lnps = _rel_err(
        spec_jax.d_log_surface_pressure_d_t,
        f_spec["d_log_surface_pressure_d_t"],
    )

    assert err_temp < tol, f"d_temperature_d_t rel err = {err_temp}"
    assert err_lnps < tol, f"d_log_surface_pressure_d_t rel err = {err_lnps}"
    assert err_vort < tol, f"d_vorticity_d_t rel err = {err_vort}"
    assert err_div < tol, f"d_divergence_d_t rel err = {err_div}"
