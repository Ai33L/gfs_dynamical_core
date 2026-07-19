"""Render the K-draw sigma-convergence figure for the tutorial (pure matplotlib).

Contrasts the single-draw M1 sweep (12 cases x 1 truth draw, settles ~2% high)
with the K-draw run (32 cases x 32 draws, overshoot-and-relax, bias removed).
Reads the two resumable checkpoints under _artifacts/; writes
figures/fig_convergence_multidraw.png. No JAX beyond unpickling SPPTParams.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = Path(__file__).parent / "_artifacts"
OUT = Path(__file__).parent / "fig_convergence_multidraw.png"


def _sigma(ckpt):
    h = pickle.load(open(ART / ckpt, "rb"))["history"]
    steps = [r["step"] for r in h]
    sig = [r["sigma"] for r in h]
    return steps, sig


def main():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(0.5, ls="--", c="0.4", lw=1, label=r"true $\sigma=0.5$")
    ax.axhline(0.2, ls=":", c="0.6", lw=1)

    s1, y1 = _sigma("sweep_ckpt.pkl")
    ax.plot(s1, y1, "-o", ms=3, c="#c44", lw=1.4,
            label=r"single draw, $N_c{=}12$, $K{=}1$")
    l1 = sum(y1[-10:]) / len(y1[-10:])
    ax.annotate(f"last-10 mean {l1:.3f}", (s1[-1], y1[-1]), fontsize=8,
                color="#c44", xytext=(4, 4), textcoords="offset points")

    s2, y2 = _sigma("convergence_ckpt.pkl")
    ax.plot(s2, y2, "-o", ms=3, c="#268", lw=1.4,
            label=r"K-draw, $N_c{=}32$, $K{=}32$")
    l2 = sum(y2[-10:]) / len(y2[-10:])
    ax.annotate(f"last-10 mean {l2:.3f}", (s2[-1], y2[-1]), fontsize=8,
                color="#268", xytext=(4, -12), textcoords="offset points")

    ax.set_xlabel("optimiser step")
    ax.set_ylabel(r"recovered $\sigma$")
    ax.set_title(r"K-draw truth removes the finite-sample bias in $\sigma$")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_ylim(0.15, 0.60)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT.name}: single-draw last-10 {l1:.4f}, K-draw last-10 {l2:.4f}")


if __name__ == "__main__":
    main()
