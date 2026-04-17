import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np
import jax.numpy as jnp
from datetime import timedelta
from pathlib import Path
import climt
from sympl import set_constant

from gfs_dynamical_core import GFSDynamicalCore
from gfs_dynamical_core.component_jax import GFSDynamicsJAX
from gfs_dynamical_core.jax.states import GridState
from gfs_dynamical_core.jax.transforms import grid_to_spectral

L = 64
NLONS = 2 * L - 1
NLATS = L
NLEVS = 20
NTRAC = 1
NTRUNC = int(NLONS / 3 - 2)
NDIMSPEC = (NTRUNC + 1) * (NTRUNC + 2) // 2

set_constant("reference_air_pressure", value=1e5, units="Pa")

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

def load_fortran_dump(path):
    data = Path(path).read_bytes()
    offset = 0

    def read_array(shape, dtype=np.float64):
        nonlocal offset
        n = int(np.prod(shape)) * np.dtype(dtype).itemsize
        arr = np.frombuffer(data[offset : offset + n], dtype=dtype)
        offset += n
        arr = arr.reshape(shape[::-1]).T 
        return arr

    result = {}
    result["ug"] = read_array((NLONS, NLATS, NLEVS))
    result["vg"] = read_array((NLONS, NLATS, NLEVS))
    result["virtempg"] = read_array((NLONS, NLATS, NLEVS))
    result["lnpsg"] = read_array((NLONS, NLATS))
    result["tracerg"] = read_array((NLONS, NLATS, NLEVS, NTRAC))

    result["vrtspec"] = read_array((NDIMSPEC, NLEVS), dtype=np.complex128)
    result["divspec"] = read_array((NDIMSPEC, NLEVS), dtype=np.complex128)
    result["virtempspec"] = read_array((NDIMSPEC, NLEVS), dtype=np.complex128)
    result["lnpsspec"] = read_array((NDIMSPEC,), dtype=np.complex128)

    assert offset == len(data), f"Did not consume all data: {offset} != {len(data)}"

    for name in ["ug", "vg", "virtempg"]:
        arr = result[name]
        arr = arr.transpose(2, 1, 0)
        arr = arr[:, ::-1, :]
        result[name] = arr

    result["lnpsg"] = result["lnpsg"].T[::-1, :]

    arr = result["tracerg"]
    arr = arr.transpose(3, 2, 1, 0)
    arr = arr[:, :, ::-1, :]
    result["tracerg"] = arr

    for name in ["vrtspec", "divspec", "virtempspec"]:
        result[name] = result[name].T 

    return result

import sys
import shutil

def run_case(case_name):
    print(f"\n{'='*80}\nCASE: {case_name}\n{'='*80}")
    
    # Delete old dumps
    for p in Path("debug_data").glob("fortran_step_*.bin"):
        p.unlink()

    grid = climt.get_grid(nx=NLONS, ny=NLATS, nz=NLEVS)
    
    # ── Fortran ──
    dycore_f = GFSDynamicalCore()
    state_f = climt.get_default_state([dycore_f], grid_state=grid)
    
    lon = np.asarray(grid['longitude'])  # (NLATS, NLONS)
    lat = np.asarray(grid['latitude'])   # (NLATS, NLONS)
    
    u = np.zeros((NLEVS, NLATS, NLONS))
    v = np.zeros((NLEVS, NLATS, NLONS))
    T = np.ones((NLEVS, NLATS, NLONS)) * 300.0
    ps = np.ones((NLATS, NLONS)) * 1e5
    q = np.zeros((1, NLEVS, NLATS, NLONS))
    
    if case_name == "zonal_u_cos_lat":
        u[:] = 10.0 * np.cos(lat)[None, :, :]
    elif case_name == "zonal_u_sin_lat":
        u[:] = 10.0 * np.sin(lat)[None, :, :]
    elif case_name == "m1_u_wave":
        u[:] = 10.0 * np.cos(lat)[None, :, :] * np.cos(lon)[None, :, :]
    elif case_name == "m2_u_wave":
        u[:] = 10.0 * np.cos(lat)[None, :, :] * np.cos(2*lon)[None, :, :]
    elif case_name == "solid_body_v":
        v[:] = 10.0 * np.cos(lat)[None, :, :]
    elif case_name == "m1_v_wave":
        v[:] = 10.0 * np.cos(lat)[None, :, :] * np.sin(lon)[None, :, :]
    elif case_name == "T_wave":
        T[:] = 300.0 + 10.0 * np.cos(lat)[None, :, :] * np.cos(lon)[None, :, :]
    elif case_name == "ps_wave":
        ps[:] = 1e5 + 1000.0 * np.cos(lat) * np.cos(lon)
        
    state_f["eastward_wind"][:] = u
    state_f["northward_wind"][:] = v
    state_f["air_temperature"][:] = T
    state_f["surface_air_pressure"][:] = ps
    state_f["specific_humidity"][:] = q[0]
    
    timestep = timedelta(minutes=5)
    
    # We must ensure debug dumps are written. 
    # Fortran does it every `debug_step_count` steps. 
    # It starts at 0, dumps at step 1.
    dycore_f(state_f, timestep=timestep)
    
    # ── JAX ──
    dycore_j = GFSDynamicsJAX()
    state_j = climt.get_default_state([dycore_j], grid_state=grid)
    state_j["eastward_wind"][:] = u
    state_j["northward_wind"][:] = v
    state_j["air_temperature"][:] = T
    state_j["surface_air_pressure"][:] = ps
    state_j["specific_humidity"][:] = q[0]
    
    dycore_j(state_j, timestep=timestep)
    trans_config = dycore_j.trans_config
    
    grid0_j = GridState(
        u=jnp.array(u),
        v=jnp.array(v),
        temperature=jnp.array(T),
        vorticity=jnp.zeros_like(u),
        divergence=jnp.zeros_like(u),
        log_surface_pressure=jnp.log(jnp.array(ps)),
        tracers=jnp.array(q)
    )
    spec_j = grid_to_spectral(grid0_j, trans_config)
    
    dump_path = Path("debug_data/fortran_step_1_stage_0.bin")
    if not dump_path.exists():
        print("ERROR: Fortran debug dump not found.")
        return
        
    f_s0 = load_fortran_dump(dump_path)
    
    # Let's pick level 10
    k = 10
    vrt_f = shtns_packed_to_s2fft(f_s0["vrtspec"][k], L, NTRUNC)
    vrt_j = np.array(spec_j.vorticity[k])
    
    div_f = shtns_packed_to_s2fft(f_s0["divspec"][k], L, NTRUNC)
    div_j = np.array(spec_j.divergence[k])
    
    temp_f = shtns_packed_to_s2fft(f_s0["virtempspec"][k], L, NTRUNC)
    temp_j = np.array(spec_j.temperature[k])
    
    print(f"Comparing level {k}:")
    for name, f_arr, j_arr in [("Vorticity", vrt_f, vrt_j), ("Divergence", div_f, div_j), ("Temperature", temp_f, temp_j)]:
        diff = np.abs(f_arr - j_arr)
        max_f = np.abs(f_arr).max()
        max_j = np.abs(j_arr).max()
        max_diff = diff.max()
        
        print(f"  {name:15s}: max|F|={max_f:.4e}, max|J|={max_j:.4e}, max|diff|={max_diff:.4e}")
        
        if max_f > 1e-10 or max_j > 1e-10:
            threshold = 1e-5 * max(max_f, max_j)
            count = 0
            for l in range(L):
                for m in range(L):
                    j_idx = m + L - 1
                    val_f = f_arr[l, j_idx]
                    val_j = j_arr[l, j_idx]
                    if np.abs(val_f) > threshold or np.abs(val_j) > threshold:
                        if np.abs(val_f) > 0:
                            ratio_complex = val_j / val_f
                            print(f"    l={l:2d}, m={m:2d}: F={val_f:10.3e}, J={val_j:10.3e}, ratio={np.abs(ratio_complex):.4f}, phase={np.angle(ratio_complex):.2f}")
                        else:
                            print(f"    l={l:2d}, m={m:2d}: F={val_f:10.3e}, J={val_j:10.3e}, ratio=inf")
                        count += 1
                        if count > 5: break
                if count > 5: break

if __name__ == '__main__':
    cases = [
        "zonal_u_cos_lat",
        "zonal_u_sin_lat",
        "m1_u_wave",
        "m2_u_wave",
        "solid_body_v",
        "m1_v_wave",
        "T_wave",
        "ps_wave"
    ]
    for c in cases:
        run_case(c)
