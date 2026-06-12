"""
Compare gradient computation: SHTNS vs JAX spin-1.
Uses L=64 and ntrunc=40 to match the actual dycore configuration.
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'

import numpy as np
import jax.numpy as jnp
import s2fft
import shtns

ntrunc = 40
L = 64  # matches dycore: n_lat = L = 64
R = 6.371e6

# SHTNS setup matching Fortran dycore
sh = shtns.sht(ntrunc, ntrunc)
nlat, nphi = sh.set_grid(nlat=64, nphi=128, flags=shtns.SHT_NO_CS_PHASE)

# --- Test spectral field (m=0 only, mimics JW06 T profile) ---
T_lm = np.zeros(sh.nlm, dtype=complex)
T_lm[sh.idx(0, 0)] = 300.0
T_lm[sh.idx(2, 0)] = -15.0
T_lm[sh.idx(4, 0)] = 5.0
T_lm[sh.idx(6, 0)] = -2.0
T_lm[sh.idx(8, 0)] = 0.5

# === SHTNS gradient → back to spectral ===
grad_theta, grad_phi = sh.synth_grad(T_lm)
grad_y_shtns = -grad_theta / R
grad_y_lm_shtns = sh.analys(grad_y_shtns)

# === JAX spin-1 gradient → back to spectral ===
# Convert to s2fft format (L=64, so array is 64 x 127)
T_lm_s2fft = np.zeros((L, 2*L - 1), dtype=complex)
for l in range(ntrunc + 1):
    for m in range(0, l + 1):
        val = T_lm[sh.idx(l, m)]
        T_lm_s2fft[l, L - 1 + m] = val
        if m > 0:
            T_lm_s2fft[l, L - 1 - m] = (-1)**m * np.conj(val)

l_arr = jnp.arange(L)
l_factor = jnp.sqrt(l_arr * (l_arr + 1))
F1_t_lm = -l_factor[:, None] * jnp.array(T_lm_s2fft)
f_spin1 = np.array(s2fft.inverse_jax(F1_t_lm, L, spin=1, sampling="gl"))
grad_y_jax = -f_spin1.real / R

# Forward transform back to spectral
grad_y_lm_jax = np.array(s2fft.forward_jax(jnp.array(grad_y_jax), L, sampling="gl"))

# === Compare m=0 spectral coefficients ===
print(f"{'l':>3} {'SHTNS':>16} {'JAX':>16} {'ratio':>10} {'diff':>14}")
print("-" * 73)
for l in range(min(ntrunc + 1, 15)):
    v_sh = grad_y_lm_shtns[sh.idx(l, 0)].real
    v_jx = grad_y_lm_jax[l, L - 1].real
    ratio = v_sh / v_jx if abs(v_jx) > 1e-20 else float('nan')
    diff = v_sh - v_jx
    print(f"{l:3d} {v_sh:16.8e} {v_jx:16.8e} {ratio:10.6f} {diff:14.6e}")

diffs = [abs(grad_y_lm_shtns[sh.idx(l, 0)].real - grad_y_lm_jax[l, L-1].real) for l in range(ntrunc+1)]
vals = [abs(grad_y_lm_shtns[sh.idx(l, 0)].real) for l in range(ntrunc+1)]
print(f"\nmax|diff| = {max(diffs):.6e}, max|val| = {max(vals):.6e}, rel = {max(diffs)/max(vals):.6e}")

# === Also compare grid-space zonal mean directly ===
# Both use 64-latitude GL grids; longitudes differ (128 vs 127) but zm is the same
zm_shtns = grad_y_shtns.mean(axis=1)
zm_jax = grad_y_jax.mean(axis=1)
print(f"\nGrid zm shapes: SHTNS={zm_shtns.shape}, JAX={zm_jax.shape}")
if zm_shtns.shape == zm_jax.shape:
    zm_diff = zm_shtns - zm_jax
    print(f"max|zm_diff| = {np.abs(zm_diff).max():.6e}")
    print(f"rel_err = {np.abs(zm_diff).max() / np.abs(zm_shtns).max():.6e}")
    # Print ratio profile
    print(f"\n{'lat_idx':>7} {'SHTNS':>14} {'JAX':>14} {'ratio':>10}")
    for i in range(0, 64, 4):
        r = zm_shtns[i] / zm_jax[i] if abs(zm_jax[i]) > 1e-20 else float('nan')
        print(f"{i:7d} {zm_shtns[i]:14.6e} {zm_jax[i]:14.6e} {r:10.6f}")
