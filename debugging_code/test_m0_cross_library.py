"""Feed Fortran spectral coefficients to s2fft inverse and compare with Fortran grid output.
Isolates whether the inverse vector transform differs between SHTNS and s2fft for m=0."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np
import jax.numpy as jnp
import s2fft
from pathlib import Path

L = 64
NLONS = 2 * L - 1
NLATS = L
NLEVS = 20
NTRAC = 1
NTRUNC = int(NLONS / 3 - 2)
NDIMSPEC = (NTRUNC + 1) * (NTRUNC + 2) // 2
radius = 6371000.0

def shtns_packed_to_s2fft(packed_coeffs, L, ntrunc):
    flm = np.zeros((L, 2 * L - 1), dtype=np.complex128)
    idx = 0
    for m in range(ntrunc + 1):
        for l in range(m, ntrunc + 1):
            if l < L:
                j_pos = m + L - 1
                j_neg = -m + L - 1
                coeff = packed_coeffs[idx]
                cs_factor = (-1) ** m
                flm[l, j_pos] = cs_factor * coeff
                if m > 0:
                    flm[l, j_neg] = np.conj(coeff)
            idx += 1
    return flm

# Load Fortran dump
data = Path("debug_data/fortran_step_1_stage_0.bin").read_bytes()
offset = 0
def read_array(shape, dtype=np.float64):
    global offset
    n = int(np.prod(shape)) * np.dtype(dtype).itemsize
    arr = np.frombuffer(data[offset:offset+n], dtype=dtype).copy()
    offset += n
    return arr.reshape(shape[::-1]).T

ug = read_array((NLONS, NLATS, NLEVS))
vg = read_array((NLONS, NLATS, NLEVS))
virtempg = read_array((NLONS, NLATS, NLEVS))
lnpsg = read_array((NLONS, NLATS))
tracerg = read_array((NLONS, NLATS, NLEVS, NTRAC))
vrtspec = read_array((NDIMSPEC, NLEVS), dtype=np.complex128)
divspec = read_array((NDIMSPEC, NLEVS), dtype=np.complex128)
virtempspec = read_array((NDIMSPEC, NLEVS), dtype=np.complex128)
lnpsspec = read_array((NDIMSPEC,), dtype=np.complex128)
assert offset == len(data)

# Reorder to (lev, lat, lon), flip lat N->S
for name, arr in [("ug", ug), ("vg", vg), ("virtempg", virtempg)]:
    globals()[name] = arr.transpose(2, 1, 0)[:, ::-1, :]
lnpsg = lnpsg.T[::-1, :]
vrtspec = vrtspec.T
divspec = divspec.T

l_arr = np.arange(L)
l_factor = np.sqrt(l_arr * (l_arr + 1))
inv_l_factor = np.where(l_arr > 0, 1.0 / l_factor, 0.0)

# Pick a representative level
k = 10

# Convert Fortran spectral to s2fft format
vrt_s2fft = shtns_packed_to_s2fft(vrtspec[k], L, NTRUNC)
div_s2fft = shtns_packed_to_s2fft(divspec[k], L, NTRUNC)

# Apply s2fft spin-1 inverse (same formula as transforms.py)
F1_lm = inv_l_factor[:, None] * (div_s2fft + 1j * vrt_s2fft) * radius
f_spin1 = s2fft.inverse_jax(jnp.array(F1_lm), L, spin=1, sampling="gl")
u_jax = np.array(f_spin1.imag)
v_jax = np.array(-f_spin1.real)

u_fort = ug[k]
v_fort = vg[k]

print("=== Fortran spectral -> s2fft inverse vs Fortran grid (level %d) ===" % k)
for name, j, f in [("u (eastward)", u_jax, u_fort), ("v (northward)", v_jax, v_fort)]:
    diff = np.abs(j - f)
    mx = max(np.abs(j).max(), np.abs(f).max())
    zm_j = j.mean(axis=1)
    zm_f = f.mean(axis=1)
    eddy_j = j - zm_j[:, None]
    eddy_f = f - zm_f[:, None]
    print(f"\n{name}:")
    print(f"  total max|diff|  = {diff.max():.6e}  (max|val|={mx:.2f})")
    print(f"  zonal mean diff  = {np.abs(zm_j - zm_f).max():.6e}")
    print(f"  eddy diff        = {np.abs(eddy_j - eddy_f).max():.6e}")

    # Ratio of zonal means where they're significant
    strong = np.abs(zm_f) > 1.0
    if strong.any():
        ratios = zm_j[strong] / zm_f[strong]
        print(f"  zm ratio (jax/fort) where |zm|>1: mean={ratios.mean():.8f}, "
              f"range=[{ratios.min():.8f}, {ratios.max():.8f}]")

# Now check: does SHTNS use the SAME spectral coefficients but a different
# inverse formula? The Fortran getuv does:
#   Ψ_lm = invlap * R * vrtspec = -R/(l(l+1)) * vrtspec
#   Χ_lm = invlap * R * divspec = -R/(l(l+1)) * divspec
#   sphtor_to_spat(Ψ, Χ) -> (ugrid, vgrid)
#
# Our s2fft formula does:
#   F1_lm = R/sqrt(l(l+1)) * (div + i*vort)
#   spin-1 inverse -> f_spin1
#   u = imag(f_spin1), v = -real(f_spin1)
#
# If sphtor_to_spat(Ψ,Χ) = our result, both should match.
# If they don't match for m=0, there's a normalization difference in the
# spin-1 basis functions vs the vector SH basis functions.

# Also check scalar transform comparison
tmp_s2fft = shtns_packed_to_s2fft(virtempspec[k], L, NTRUNC)
t_jax = np.array(s2fft.inverse_jax(jnp.array(tmp_s2fft), L, spin=0, sampling="gl").real)
t_fort = virtempg[k]
print(f"\nTemperature (scalar, control):")
print(f"  max|diff| = {np.abs(t_jax - t_fort).max():.6e}  (max|val|={np.abs(t_fort).max():.2f})")
print(f"  zm diff   = {np.abs(t_jax.mean(axis=1) - t_fort.mean(axis=1)).max():.6e}")
