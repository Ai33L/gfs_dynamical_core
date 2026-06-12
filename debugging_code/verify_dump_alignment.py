"""Quick sanity-check after fixing the pk pointer-shape bug in the Fortran dump.

Loads debug_data/fortran_dyntend_call_00001.bin via tests/fortran_loader.py and
prints value ranges for the fields that were previously misaligned (psg, dlnpsdt,
alfa, rlnp) plus a few sanity checks on the rest.

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate climt && \\
    python debugging_code/verify_dump_alignment.py
"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from fortran_loader import load_dyntend_dump  # noqa: E402


def _stats(name, arr):
    a = np.asarray(arr)
    print(f"  {name:28s} shape={tuple(a.shape)!s:<24s}"
          f" min={a.min(): .3e}  max={a.max(): .3e}  mean={a.mean(): .3e}")


def main():
    path = ROOT / "debug_data" / "fortran_dyntend_call_00001.bin"
    print(f"Loading {path}")
    size = path.stat().st_size
    print(f"  file size: {size} bytes  (expected 34213100)")
    d = load_dyntend_dump(path)
    h = d["header"]
    print(f"  header: nlons={h.nlons} nlats={h.nlats} nlevs={h.nlevs}"
          f" ndimspec={h.ndimspec} ntrac={h.ntrac}"
          f" rkstage={h.rkstage} call_idx={h.call_idx}")

    print("\nBlock 1 — grid state")
    _stats("u",    d["grid_state"]["u"])
    _stats("v",    d["grid_state"]["v"])
    _stats("Tv",   d["grid_state"]["virtual_temperature"])
    _stats("lnps", d["grid_state"]["log_surface_pressure"])

    print("\nBlock 2 — gradients")
    _stats("dT/dlambda",    d["gradients"]["d_t_d_lambda"])
    _stats("dlnps/dlambda", d["gradients"]["d_log_ps_d_lambda"])
    _stats("dphis/dlambda", d["gradients"]["d_phis_d_lambda"])

    print("\nBlock 3 — pressure diagnostics  (was the bad block)")
    _stats("pk",        d["pressure"]["pk"])
    _stats("dp",        d["pressure"]["dp"])
    _stats("prs_layer", d["pressure"]["prs"])
    _stats("alfa",      d["pressure"]["alfa"])
    _stats("rlnp",      d["pressure"]["rlnp"])
    _stats("ps",        d["pressure"]["ps"])

    print("\nBlock 4 — vertical velocities")
    _stats("etadot",  d["vertical_velocities"]["etadot"])
    _stats("omega",   d["vertical_velocities"]["omega"])
    _stats("dlnpsdt", d["vertical_velocities"]["d_log_ps_d_t"])

    print("\nBlock 5 — PGF")
    _stats("pgf_x", d["pgf"]["pgf_x"])
    _stats("pgf_y", d["pgf"]["pgf_y"])

    print("\nBlock 6 — vertical advection")
    _stats("vadv_u", d["vertical_advection"]["vadv_u"])
    _stats("vadv_t", d["vertical_advection"]["vadv_t"])

    print("\nBlock 7/8 — energy & tendencies")
    _stats("energy_conv", d["energy_conv"])
    _stats("u_flux",      d["grid_tendencies"]["u_flux"])
    _stats("temp_tend",   d["grid_tendencies"]["temp_tend"])
    _stats("ke",          d["grid_tendencies"]["ke"])

    print("\nBlock 9 — spectral tendencies (|coef| stats)")
    for k, v in d["spectral_tendencies"].items():
        a = np.abs(np.asarray(v))
        print(f"  {k:28s} shape={tuple(a.shape)!s:<24s}"
              f" |coef| min={a.min(): .3e}  max={a.max(): .3e}")

    print("\nSanity gates (must pass):")
    ps = d["pressure"]["ps"]
    ps_ok = 5e4 < ps.mean() < 1.1e5
    print(f"  ps mean in (5e4, 1.1e5) Pa:      {ps_ok}  (mean={ps.mean():.3e})")
    lnps = d["grid_state"]["log_surface_pressure"]
    lnps_ok = 10.5 < lnps.mean() < 11.7
    print(f"  lnps mean in (10.5, 11.7):       {lnps_ok}  (mean={lnps.mean():.3e})")
    pk_top_bot_ok = d["pressure"]["pk"][-1].max() > d["pressure"]["pk"][0].max()
    print(f"  pk[top-of-JAX-array] > pk[surface]?  {pk_top_bot_ok}  "
          "(JAX level 0 = surface, so pk[-1] is TOA which should be smaller — this prints the literal comparison)")
    print(f"    pk[k=0 surface]  max = {d['pressure']['pk'][0].max(): .3e}")
    print(f"    pk[k=nlevs TOA] max = {d['pressure']['pk'][-1].max(): .3e}")
    print(f"  alfa[surface level] mean = {d['pressure']['alfa'][0].mean(): .3e}  (expect O(0.1-1))")
    print(f"  dlnpsdt max |val|        = {np.abs(d['vertical_velocities']['d_log_ps_d_t']).max(): .3e}")


if __name__ == "__main__":
    main()
