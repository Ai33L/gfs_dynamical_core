import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re, os

SNAP="/tmp/gfs_compare_scratch/snapshots"
OUT="/Users/joymonteiro/github/gfs_dynamical_core/debugging_code/evidence"
cos_t,_=np.polynomial.legendre.leggauss(64)
lats=90.0-np.degrees(np.flip(np.arccos(cos_t)))
lons=np.linspace(0,360,127,endpoint=False)

def maps(step, day):
    d=np.load(f"{SNAP}/snap_{step:05d}.npz")
    fps=d["f_ps"].astype(np.float64)/100; jps=d["j_ps"].astype(np.float64)/100
    fT=d["f_T"].astype(np.float64)[18]; jT=d["j_T"].astype(np.float64)[18]
    fig,axes=plt.subplots(3,2,figsize=(15,10),sharex=True,sharey=True)
    pa_f=fps-fps.mean(axis=1,keepdims=True); pa_j=jps-jps.mean(axis=1,keepdims=True)
    ta_f=fT-fT.mean(axis=1,keepdims=True);  ta_j=jT-jT.mean(axis=1,keepdims=True)
    v1=max(np.abs(pa_f).max(),1); v2=max(np.abs(ta_f).max(),1)
    specs=[(pa_f,f"Fortran ps anomaly (hPa)",v1),(ta_f,"Fortran T' @ ~950 hPa (K)",v2),
           (pa_j,"JAX (fixed) ps anomaly",v1),(ta_j,"JAX (fixed) T'",v2),
           (pa_j-pa_f,"difference (J-F) ps",v1),(ta_j-ta_f,"difference (J-F) T'",v2)]
    for ax,(fld,title,vm) in zip(axes.flat,specs):
        im=ax.pcolormesh(lons,lats,fld,cmap="RdBu_r",vmin=-vm,vmax=vm)
        ax.set_title(title,fontsize=10); plt.colorbar(im,ax=ax,shrink=0.85)
    axes[0,0].set_ylim(0,90)  # NH where the wave is
    fig.suptitle(f"DCMIP 4.1 baroclinic wave, day {day:.1f} (step {step}) — Fortran vs fixed JAX",fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/5_maps_day{day:.0f}.png",dpi=120); plt.close(fig)
    print(f"day {day}: max|ps_J-ps_F| = {np.abs(jps-fps).max():.3f} hPa, max|T_J-T_F| = {np.abs(jT-fT).max():.3f} K")

maps(2400, 2400*5/1440)
maps(2880, 2880*5/1440)
maps(3450, 3450*5/1440)
