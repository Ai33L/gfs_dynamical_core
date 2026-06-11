"""Load Fortran ``getdyntend`` per-function intermediate dumps.

Paired with ``dump_dyntend_intermediates`` in
``gfs_dynamical_core/_lib/GFS/dyn_run.f90``. Converts every field from Fortran
conventions (column-major, latitude south->north, mixed BTU/TTB level orderings)
into the JAX conventions used throughout ``gfs_dynamical_core/jax/``:

  - 3D grid fields: ``(n_lev, n_lat, n_lon)``, bottom-to-top (k=0 is surface).
  - 2D grid fields: ``(n_lat, n_lon)``.
  - Interface fields: ``(n_lev+1, n_lat, n_lon)``, k=0 is surface.
  - Latitude axis: north-to-south (s2fft GL order) — matches Fortran's native
    order (shtns stores lats(j) = pi/2 at j=1 down to -pi/2 at j=nlats), so no
    lat flip is applied.
  - Spectral fields: converted from SHTNS packed to s2fft rectangular via
    :func:`spectral_converter.shtns_packed_to_s2fft_rect`.

Field-by-field level ordering of the raw Fortran arrays (see
``debugging_code/test_harness_plan.md`` Appendix A):

  BTU (k=1 surface): ug, vg, virtempg, divg, vrtg, tracerg, dlnpdtg,
                     vadvu, vadvv, vadvt, energy_conv, uflux, vflux,
                     temp_tend, ke.
  TTB (k=1 TOA):     pk, dpk, prs_layer, alfa, rlnp, etadot.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from spectral_converter import ndimspec, shtns_packed_to_s2fft_rect


_FIELDS_3D_BTU = (
    "u", "v", "virtual_temperature", "divergence", "vorticity",
    "omega", "vadv_u", "vadv_v", "vadv_t", "energy_conv",
    "u_flux", "v_flux", "temp_tend", "ke",
)
_FIELDS_3D_TTB = (
    "pk", "dp", "prs_layer", "alfa", "rlnp", "etadot",
)
_FIELDS_2D = (
    "log_surface_pressure", "surface_pressure", "d_log_ps_d_lambda",
    "d_log_ps_d_phi", "d_phis_d_lambda", "d_phis_d_phi",
    "d_log_ps_d_t",
)


@dataclass
class DumpHeader:
    nlons: int
    nlats: int
    nlevs: int
    ndimspec: int
    ntrac: int
    rkstage: int
    call_idx: int


def _btu_to_jax(arr: np.ndarray) -> np.ndarray:
    """Fortran BTU (nlons, nlats, nlevs) F-order -> JAX (nlevs, nlats, nlons).

    Fortran stores latitudes N->S (j=1 = NP, j=nlats = SP), matching climt's
    convention and JAX's ``get_gaussian_latitudes`` (also N->S). No lat flip.
    Level axis unchanged (k=0 surface).
    """
    return arr.transpose(2, 1, 0)


def _ttb_to_jax(arr: np.ndarray) -> np.ndarray:
    """Fortran TTB (nlons, nlats, nlevs[+1]) F-order -> JAX (nlevs[+1], nlats,
    nlons). Level axis flipped so k=0 -> surface (BTU); lat axis unchanged
    (stays N->S)."""
    return arr.transpose(2, 1, 0)[::-1, :, :]


def _2d_to_jax(arr: np.ndarray) -> np.ndarray:
    """Fortran (nlons, nlats) F-order -> JAX (nlats, nlons); lat unchanged (N->S)."""
    return arr.T


def _tracers_to_jax(arr: np.ndarray) -> np.ndarray:
    """Fortran (nlons, nlats, nlevs, ntrac) F-order -> JAX (ntrac, nlevs, nlats, nlons).

    Grid tracers are BTU in Fortran; level axis not flipped, lat axis not flipped.
    """
    return arr.transpose(3, 2, 1, 0)


class _StreamReader:
    def __init__(self, path: Path):
        self._buf = memoryview(path.read_bytes())
        self._pos = 0

    def read(self, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        n = int(np.prod(shape)) * np.dtype(dtype).itemsize
        chunk = bytes(self._buf[self._pos : self._pos + n])
        self._pos += n
        # Fortran writes column-major: reverse shape, read, then transpose.
        arr = np.frombuffer(chunk, dtype=dtype).copy()
        return arr.reshape(shape[::-1]).T

    def at_end(self) -> bool:
        return self._pos == len(self._buf)

    @property
    def remaining(self) -> int:
        return len(self._buf) - self._pos


def load_dyntend_dump(path: str | Path, L: int | None = None) -> dict:
    """Load one ``fortran_dyntend_call_NNNNN.bin`` dump file.

    Parameters
    ----------
    path : str | Path
        Path to the dump file.
    L : int, optional
        s2fft band-limit for spectral conversion. Defaults to ``nlats`` from
        the header (assumes GL sampling where ``n_lat = L``).

    Returns
    -------
    dict with the following keys (all arrays in JAX conventions unless noted):
        'header'                : DumpHeader
        'grid_state'            : dict of grid-space state
        'gradients'             : dict of horizontal gradients
        'pressure'              : dict of pressure diagnostics
        'vertical_velocities'   : dict (omega, etadot, d_log_ps_d_t)
        'pgf'                   : dict (pgf_x, pgf_y)
        'vertical_advection'    : dict (vadv_u, vadv_v, vadv_t)
        'energy_conv'           : ndarray
        'grid_tendencies'       : dict (u_flux, v_flux, temp_tend, ke)
        'spectral_tendencies'   : dict (s2fft rectangular, complex128)
    """
    path = Path(path)
    r = _StreamReader(path)

    header_raw = r.read((7,), np.int32)
    header = DumpHeader(
        nlons=int(header_raw[0]),
        nlats=int(header_raw[1]),
        nlevs=int(header_raw[2]),
        ndimspec=int(header_raw[3]),
        ntrac=int(header_raw[4]),
        rkstage=int(header_raw[5]),
        call_idx=int(header_raw[6]),
    )

    nlons, nlats, nlevs = header.nlons, header.nlats, header.nlevs
    ntrac, nd = header.ntrac, header.ndimspec
    if L is None:
        L = nlats
    # Derive ntrunc from packed length so we don't have to read it separately.
    ntrunc = _solve_ntrunc_from_ndimspec(nd)

    shape3d = (nlons, nlats, nlevs)
    shape3d_i = (nlons, nlats, nlevs + 1)
    shape2d = (nlons, nlats)
    shape_tracers = (nlons, nlats, nlevs, ntrac)
    shape_spec = (nd, nlevs)
    shape_spec_2d = (nd,)
    shape_spec_tracers = (nd, nlevs, ntrac)

    # Block 1: Grid state
    ug = _btu_to_jax(r.read(shape3d, np.float64))
    vg = _btu_to_jax(r.read(shape3d, np.float64))
    tv = _btu_to_jax(r.read(shape3d, np.float64))
    div = _btu_to_jax(r.read(shape3d, np.float64))
    vort = _btu_to_jax(r.read(shape3d, np.float64))
    lnps = _2d_to_jax(r.read(shape2d, np.float64))
    tracers = _tracers_to_jax(r.read(shape_tracers, np.float64))

    # Block 2: Gradients
    dtdx = _btu_to_jax(r.read(shape3d, np.float64))
    dtdy = _btu_to_jax(r.read(shape3d, np.float64))
    dlnpsdx = _2d_to_jax(r.read(shape2d, np.float64))
    dlnpsdy = _2d_to_jax(r.read(shape2d, np.float64))
    dphisdx = _2d_to_jax(r.read(shape2d, np.float64))
    dphisdy = _2d_to_jax(r.read(shape2d, np.float64))

    # Block 3: Pressure diagnostics (stored TTB in Fortran)
    pk = _ttb_to_jax(r.read(shape3d_i, np.float64))
    dp = _ttb_to_jax(r.read(shape3d, np.float64))
    prs_layer = _ttb_to_jax(r.read(shape3d, np.float64))
    alfa = _ttb_to_jax(r.read(shape3d, np.float64))
    rlnp = _ttb_to_jax(r.read(shape3d, np.float64))
    ps = _2d_to_jax(r.read(shape2d, np.float64))

    # Block 4: Vertical velocities
    etadot = _ttb_to_jax(r.read(shape3d_i, np.float64))
    omega = _btu_to_jax(r.read(shape3d, np.float64))
    dlnpsdt = _2d_to_jax(r.read(shape2d, np.float64))

    # Block 5: PGF
    pgf_x = _btu_to_jax(r.read(shape3d, np.float64))
    pgf_y = _btu_to_jax(r.read(shape3d, np.float64))

    # Block 6: Vertical advection
    vadv_u = _btu_to_jax(r.read(shape3d, np.float64))
    vadv_v = _btu_to_jax(r.read(shape3d, np.float64))
    vadv_t = _btu_to_jax(r.read(shape3d, np.float64))

    # Block 7: Energy conversion
    energy_conv = _btu_to_jax(r.read(shape3d, np.float64))

    # Block 8: Assembled grid tendencies
    u_flux = _btu_to_jax(r.read(shape3d, np.float64))
    v_flux = _btu_to_jax(r.read(shape3d, np.float64))
    temp_tend = _btu_to_jax(r.read(shape3d, np.float64))
    ke = _btu_to_jax(r.read(shape3d, np.float64))

    # Block 9: Spectral tendencies (SHTNS packed -> s2fft rectangular)
    dvrt_packed = r.read(shape_spec, np.complex128)
    ddiv_packed = r.read(shape_spec, np.complex128)
    dtemp_packed = r.read(shape_spec, np.complex128)
    dlnps_packed = r.read(shape_spec_2d, np.complex128)
    dtracers_packed = r.read(shape_spec_tracers, np.complex128)

    if not r.at_end():
        raise ValueError(
            f"Unexpected trailing bytes in {path}: {r.remaining} left."
        )

    # Convert packed -> rect and move level axis to the front (JAX convention).
    dvrt_rect = shtns_packed_to_s2fft_rect(dvrt_packed, ntrunc, L).transpose(2, 0, 1)
    ddiv_rect = shtns_packed_to_s2fft_rect(ddiv_packed, ntrunc, L).transpose(2, 0, 1)
    dtemp_rect = shtns_packed_to_s2fft_rect(dtemp_packed, ntrunc, L).transpose(2, 0, 1)
    dlnps_rect = shtns_packed_to_s2fft_rect(dlnps_packed, ntrunc, L)
    # tracers: (ndimspec, nlevs, ntrac) -> (L, 2L-1, nlevs, ntrac)
    dtrc_rect = shtns_packed_to_s2fft_rect(dtracers_packed, ntrunc, L)
    # -> JAX (ntrac, nlevs, L, 2L-1)
    dtrc_rect = dtrc_rect.transpose(3, 2, 0, 1)

    return {
        "header": header,
        "ntrunc": ntrunc,
        "L": L,
        "grid_state": {
            "u": ug, "v": vg, "virtual_temperature": tv,
            "divergence": div, "vorticity": vort,
            "log_surface_pressure": lnps, "tracers": tracers,
        },
        "gradients": {
            "d_t_d_lambda": dtdx, "d_t_d_phi": dtdy,
            "d_log_ps_d_lambda": dlnpsdx, "d_log_ps_d_phi": dlnpsdy,
            "d_phis_d_lambda": dphisdx, "d_phis_d_phi": dphisdy,
        },
        "pressure": {
            "ps": ps, "pk": pk, "dp": dp, "prs": prs_layer,
            "alfa": alfa, "rlnp": rlnp,
        },
        "vertical_velocities": {
            "omega": omega, "etadot": etadot, "d_log_ps_d_t": dlnpsdt,
        },
        "pgf": {"pgf_x": pgf_x, "pgf_y": pgf_y},
        "vertical_advection": {
            "vadv_u": vadv_u, "vadv_v": vadv_v, "vadv_t": vadv_t,
        },
        "energy_conv": energy_conv,
        "grid_tendencies": {
            "u_flux": u_flux, "v_flux": v_flux,
            "temp_tend": temp_tend, "ke": ke,
        },
        "spectral_tendencies": {
            "d_vorticity_d_t": dvrt_rect,
            "d_divergence_d_t": ddiv_rect,
            "d_temperature_d_t": dtemp_rect,
            "d_log_surface_pressure_d_t": dlnps_rect,
            "d_tracers_d_t": dtrc_rect,
        },
    }


def _solve_ntrunc_from_ndimspec(nd: int) -> int:
    # ndimspec = (ntrunc+1)(ntrunc+2)/2  =>  ntrunc = (-3 + sqrt(1+8*nd))/2
    disc = 1 + 8 * nd
    s = int(round(disc ** 0.5))
    if s * s != disc:
        raise ValueError(f"ndimspec={nd} is not a triangular number")
    ntrunc = (s - 3) // 2
    if ndimspec(ntrunc) != nd:
        raise ValueError(f"ndimspec={nd} inconsistent with ntrunc={ntrunc}")
    return ntrunc
