"""Convert between SHTNS packed-triangular and s2fft rectangular spectral layouts.

SHTNS packed layout
    1D complex array of length ``ndimspec = (ntrunc+1)*(ntrunc+2)/2``.
    Modes stored only for ``0 <= m <= l <= ntrunc``. For a real field, the
    negative-m modes are implied by f_{l,-m} = (-1)**m * conj(f_{l,m}).
    Ordered as (m outer loop, l inner loop starting at l=m):
        (0,0), (0,1), ..., (0,ntrunc),
        (1,1), (1,2), ..., (1,ntrunc),
        ...
        (ntrunc,ntrunc)

s2fft rectangular layout
    2D complex array of shape ``(L, 2L-1)`` indexed as ``flm[l, m+L-1]``,
    with ``m`` ranging over ``-(L-1) .. +(L-1)``.

Condon-Shortley phase
    Fortran SHTNS is initialized with ``SHT_NO_CS_PHASE``. s2fft includes the
    (-1)**m Condon-Shortley phase by default. The converters apply (-1)**m to
    the positive-m modes when going shtns -> s2fft (and the inverse direction
    undoes it).
"""

from __future__ import annotations

import numpy as np


def ndimspec(ntrunc: int) -> int:
    """Number of packed-triangular modes for a triangular truncation ``ntrunc``."""
    return (ntrunc + 1) * (ntrunc + 2) // 2


def shtns_packed_to_s2fft_rect(
    packed: np.ndarray, ntrunc: int, L: int
) -> np.ndarray:
    """Convert SHTNS packed triangular -> s2fft rectangular (L, 2L-1).

    Supports input shapes ``(ndimspec,)`` or ``(ndimspec, *trailing)``; the
    packed axis must be the first axis.
    """
    expected = ndimspec(ntrunc)
    if packed.shape[0] != expected:
        raise ValueError(
            f"packed length {packed.shape[0]} != ndimspec({ntrunc}) = {expected}"
        )
    if L - 1 < ntrunc:
        raise ValueError(f"L-1 ({L - 1}) must be >= ntrunc ({ntrunc})")

    trailing = packed.shape[1:]
    rect = np.zeros((L, 2 * L - 1) + trailing, dtype=np.complex128)

    idx = 0
    for m in range(ntrunc + 1):
        cs_factor = (-1) ** m
        for l in range(m, ntrunc + 1):
            coeff = packed[idx]
            j_pos = m + L - 1
            rect[l, j_pos] = cs_factor * coeff
            if m > 0:
                j_neg = -m + L - 1
                rect[l, j_neg] = np.conj(coeff)
            idx += 1
    assert idx == expected
    return rect


def s2fft_rect_to_shtns_packed(
    rect: np.ndarray, L: int, ntrunc: int
) -> np.ndarray:
    """Convert s2fft rectangular (L, 2L-1) -> SHTNS packed triangular.

    Input may have arbitrary trailing axes; the first two axes must be
    ``(L, 2L-1)``.
    """
    if rect.shape[:2] != (L, 2 * L - 1):
        raise ValueError(
            f"rect must have shape (L={L}, 2L-1={2*L-1}, ...); got {rect.shape}"
        )
    if L - 1 < ntrunc:
        raise ValueError(f"L-1 ({L - 1}) must be >= ntrunc ({ntrunc})")

    trailing = rect.shape[2:]
    packed = np.zeros((ndimspec(ntrunc),) + trailing, dtype=np.complex128)

    idx = 0
    for m in range(ntrunc + 1):
        cs_factor = (-1) ** m
        j_pos = m + L - 1
        for l in range(m, ntrunc + 1):
            # Undo the CS phase we applied in the forward converter.
            packed[idx] = cs_factor * rect[l, j_pos]
            idx += 1
    return packed
