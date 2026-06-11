"""Reference spherical-harmonic transforms via the SHTNS Python bindings.

Matches the configuration of the Fortran dycore (SHT_NO_CS_PHASE, Gauss-Legendre)
and exposes operations mirroring ``gfs_dynamical_core/jax/transforms.py`` so a
test can compare outputs directly.

Grid convention
---------------
Both SHTNS (with SHT_PHI_CONTIGUOUS) and s2fft's GL sampling store spatial
arrays as ``(n_lat, n_lon)`` with latitudes north-to-south. Inputs and outputs
of this wrapper use that convention.

Spectral convention
-------------------
SHTNS natively stores coefficients in packed-triangular form of length
``(ntrunc+1)*(ntrunc+2)/2``. This wrapper offers two forms:

  ``*_packed`` methods return/accept SHTNS-native packed arrays.
  ``*_rect``   methods return/accept s2fft rectangular ``(L, 2L-1)`` arrays.

The rect versions apply the Condon-Shortley phase correction via
:mod:`spectral_converter` so results can be compared directly against s2fft
outputs (which include CS phase).

Vector and gradient conventions
-------------------------------
Following the Fortran dycore (``_lib/GFS/shtns.f90``):

  * ``vector_inverse`` (vort, div -> u, v):
        u, v = SHsphtor_to_spat( invlap * R * vort,  invlap * R * div )
    where ``invlap = -1 / (l(l+1))`` for l > 0 and 0 for l = 0.

  * ``vector_forward`` (u, v -> vort, div):
        slm, tlm = spat_to_SHsphtor(u, v); scale both by (lap/R) = -l(l+1)/R
        vort = slm*lap/R;  div = tlm*lap/R

  * ``gradient`` of a scalar f:
        u, v = SHsphtor_to_spat(0, f_packed) / R
"""

from __future__ import annotations

import numpy as np
import shtns

from spectral_converter import (
    ndimspec,
    s2fft_rect_to_shtns_packed,
    shtns_packed_to_s2fft_rect,
)


class SHTNSReference:
    """Thin wrapper over :mod:`shtns` with convenient packed/rect accessors."""

    def __init__(self, L: int, ntrunc: int | None = None, radius: float = 6371000.0):
        if ntrunc is None:
            # Default matches TransformConfig.truncation (Fortran GFS convention).
            n_lon = 2 * L - 1
            ntrunc = int(n_lon / 3 - 2)
        if ntrunc >= L:
            raise ValueError(f"ntrunc ({ntrunc}) must be < L ({L})")
        self.L = L
        self.ntrunc = ntrunc
        self.radius = radius
        self.n_lat = L
        self.n_lon = 2 * L - 1

        self._sh = shtns.sht(ntrunc, ntrunc)
        self._sh.set_grid(
            nlat=self.n_lat,
            nphi=self.n_lon,
            flags=shtns.sht_gauss | shtns.SHT_PHI_CONTIGUOUS | shtns.SHT_NO_CS_PHASE,
        )
        # l(l+1) eigenvalues of the horizontal Laplacian (SHTNS stores them
        # scaled by radius^2 internally, but we use the bare form so we can
        # mirror Fortran's explicit ``lap`` / ``invlap`` scaling).
        l_arr = self._sh.l.astype(np.int64)
        ll1 = l_arr * (l_arr + 1)
        self._lap = -ll1.astype(np.float64)   # negative convention (matches Fortran)
        self._invlap = np.zeros_like(self._lap)
        nonzero = self._lap != 0
        self._invlap[nonzero] = 1.0 / self._lap[nonzero]

    # -------------- latitude / longitude grids --------------------------
    @property
    def cos_theta(self) -> np.ndarray:
        """cos(colatitude) at each Gauss node, ordered north->south."""
        return np.asarray(self._sh.cos_theta)

    @property
    def latitudes(self) -> np.ndarray:
        """Geographic latitudes in radians, north->south."""
        return np.arcsin(self.cos_theta)

    @property
    def gauss_weights_half(self) -> np.ndarray:
        """Northern-hemisphere Gauss weights (matches SHTNS convention)."""
        return np.asarray(self._sh.gauss_wts())

    # -------------- scalar transforms -----------------------------------
    def scalar_forward_packed(self, grid: np.ndarray) -> np.ndarray:
        """(n_lat, n_lon) grid -> SHTNS packed spectral."""
        return np.asarray(self._sh.analys(np.ascontiguousarray(grid)))

    def scalar_inverse_packed(self, spec_packed: np.ndarray) -> np.ndarray:
        """SHTNS packed spectral -> (n_lat, n_lon) grid."""
        return np.asarray(self._sh.synth(np.ascontiguousarray(spec_packed)))

    def scalar_forward_rect(self, grid: np.ndarray) -> np.ndarray:
        """(n_lat, n_lon) grid -> s2fft rectangular spectral."""
        packed = self.scalar_forward_packed(grid)
        return shtns_packed_to_s2fft_rect(packed, self.ntrunc, self.L)

    def scalar_inverse_rect(self, spec_rect: np.ndarray) -> np.ndarray:
        """s2fft rectangular spectral -> (n_lat, n_lon) grid."""
        packed = s2fft_rect_to_shtns_packed(spec_rect, self.L, self.ntrunc)
        return self.scalar_inverse_packed(packed)

    # -------------- vector (u, v) <-> (vort, div) -----------------------
    def vector_inverse_packed(
        self, vort_packed: np.ndarray, div_packed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """vort, div (packed) -> (u, v) grids, following Fortran ``getuv``."""
        slm = self._invlap * self.radius * vort_packed  # fortran passes vrt as slm
        tlm = self._invlap * self.radius * div_packed   # and div as tlm
        u = self._sh.spat_array()
        v = self._sh.spat_array()
        self._sh.SHsphtor_to_spat(
            np.ascontiguousarray(slm.astype(np.complex128)),
            np.ascontiguousarray(tlm.astype(np.complex128)),
            u,
            v,
        )
        return np.asarray(u), np.asarray(v)

    def vector_forward_packed(
        self, u: np.ndarray, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """(u, v) grids -> (vort, div) packed, following Fortran ``getvrtdivspec``."""
        slm = self._sh.spec_array()
        tlm = self._sh.spec_array()
        self._sh.spat_to_SHsphtor(
            np.ascontiguousarray(u),
            np.ascontiguousarray(v),
            slm,
            tlm,
        )
        slm = np.asarray(slm) * (self._lap / self.radius)
        tlm = np.asarray(tlm) * (self._lap / self.radius)
        # Fortran treats slm as vort and tlm as div; mirror that mapping here.
        vort_packed = slm
        div_packed = tlm
        return vort_packed, div_packed

    def vector_inverse_rect(
        self, vort_rect: np.ndarray, div_rect: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        vp = s2fft_rect_to_shtns_packed(vort_rect, self.L, self.ntrunc)
        dp = s2fft_rect_to_shtns_packed(div_rect, self.L, self.ntrunc)
        return self.vector_inverse_packed(vp, dp)

    def vector_forward_rect(
        self, u: np.ndarray, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        vp, dp = self.vector_forward_packed(u, v)
        return (
            shtns_packed_to_s2fft_rect(vp, self.ntrunc, self.L),
            shtns_packed_to_s2fft_rect(dp, self.ntrunc, self.L),
        )

    # -------------- gradient of a scalar --------------------------------
    def gradient_packed(
        self, scalar_packed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gradient (d/dlambda, d/dphi) following Fortran ``getgrad``."""
        zeros = np.zeros_like(scalar_packed)
        u = self._sh.spat_array()
        v = self._sh.spat_array()
        self._sh.SHsphtor_to_spat(
            np.ascontiguousarray(zeros.astype(np.complex128)),
            np.ascontiguousarray(scalar_packed.astype(np.complex128)),
            u,
            v,
        )
        u = np.asarray(u) / self.radius
        v = np.asarray(v) / self.radius
        return u, v

    def gradient_rect(
        self, scalar_rect: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        sp = s2fft_rect_to_shtns_packed(scalar_rect, self.L, self.ntrunc)
        return self.gradient_packed(sp)
