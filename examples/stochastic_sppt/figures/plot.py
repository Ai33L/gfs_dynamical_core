"""Render the M1 figures from saved artifacts (pure numpy/matplotlib -- no JAX).

Kept separate from the heavy compute (EXPERIMENT_LOG practice: split slow
JAX rollouts from fast plotting) so figures can be re-rendered instantly.

Reads figures/_artifacts/{sweep_ckpt,calib_results}.pkl and writes PNGs into
figures/ for the tutorial README to embed.
"""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
ART = HERE / "_artifacts"
TRUE_SIGMA, INIT_SIGMA = 0.5, 0.2
FIELD_LABEL = {"u850": "u850", "v850": "v850", "t500": "t500",
               "vort500": "vort500", "ps": "ps"}


def plot_convergence():
    with open(ART / "sweep_ckpt.pkl", "rb") as f:
        hist = pickle.load(f)["history"]
    steps = [h["step"] for h in hist]
    sigma = [h["sigma"] for h in hist]
    loss = [h["loss"] for h in hist]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax0.axhline(TRUE_SIGMA, color="k", ls="--", lw=1, label=f"true σ = {TRUE_SIGMA}")
    ax0.axhline(INIT_SIGMA, color="0.6", ls=":", lw=1, label=f"start σ = {INIT_SIGMA}")
    ax0.plot(steps, sigma, "-o", ms=3, color="C0")
    if len(sigma) >= 10:
        plateau = np.mean(sigma[-10:])
        ax0.annotate(f"last-10 mean\nσ ≈ {plateau:.3f}", xy=(steps[-1], sigma[-1]),
                     xytext=(0.55, 0.35), textcoords="axes fraction",
                     fontsize=9, color="C0")
    ax0.set_xlabel("optimizer step")
    ax0.set_ylabel("recovered σ (grid-point std)")
    ax0.set_title("M1: σ recovery (τ, ℓ frozen)")
    ax0.legend(loc="lower right", fontsize=8)
    ax0.grid(alpha=0.3)

    ax1.plot(steps, loss, "-o", ms=3, color="C3")
    ax1.set_xlabel("optimizer step")
    ax1.set_ylabel("almost-fair CRPS (batch)")
    ax1.set_title("Training loss (batch-stochastic)")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "fig_convergence.png", dpi=130)
    plt.close(fig)
    print("wrote fig_convergence.png")


def _load_calib():
    with open(ART / "calib_results.pkl", "rb") as f:
        return pickle.load(f)


def plot_spread_error(r):
    fields, leads = r["fields"], r["lead_days"]
    x = np.arange(len(fields))
    w = 0.8 / len(leads)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for j, ld in enumerate(leads):
        vals = [r["spread_error"][k][j] for k in fields]
        ax.bar(x + j * w, vals, w, label=f"{ld} d")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="calibrated (=1)")
    ax.set_xticks(x + w * (len(leads) - 1) / 2)
    ax.set_xticklabels([FIELD_LABEL[k] for k in fields])
    ax.set_ylabel("spread–error ratio")
    ax.set_ylim(0, 1.4)
    ax.set_title(f"Spread–error consistency at recovered σ = {r['sigma']:.3f}")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "fig_spread_error.png", dpi=130)
    plt.close(fig)
    print("wrote fig_spread_error.png")


def plot_afcrps(r):
    fields, leads = r["fields"], r["lead_days"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for k in fields:
        ax.plot(leads, r["afcrps"][k], "-o", ms=4, label=FIELD_LABEL[k])
    ax.set_xlabel("lead time (days)")
    ax.set_ylabel("afCRPS / truth-std  (normalized)")
    ax.set_title("Skill vs lead (per-field normalized)")
    ax.set_xticks(leads)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "fig_afcrps_vs_lead.png", dpi=130)
    plt.close(fig)
    print("wrote fig_afcrps_vs_lead.png")


def plot_rank_hist(r):
    rh = r["rank_hist"]
    counts = np.array(rh["counts"])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(np.arange(len(counts)), counts, color="C0", alpha=0.85)
    ax.axhline(counts.mean(), color="k", ls="--", lw=1,
               label=f"uniform = {counts.mean():.0f}")
    ax.set_xlabel("rank of truth among members")
    ax.set_ylabel("count")
    ax.set_title(f"Rank histogram — {rh['field']} at {rh['lead_day']} d")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "fig_rank_hist.png", dpi=130)
    plt.close(fig)
    print("wrote fig_rank_hist.png")


def plot_anom(r):
    an = r["anom"]
    member = np.array(an["member"])
    truth = np.array(an["truth"])
    lo, hi = np.percentile(member, [0.5, 99.5])
    bins = np.linspace(lo, hi, 41)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(member, bins=bins, density=True, alpha=0.5, color="C0", label="member anomalies")
    ax.hist(truth, bins=bins, density=True, histtype="step", lw=2, color="C3",
            label="truth anomalies")
    ax.set_xlabel(f"{an['field']} anomaly vs ensemble mean")
    ax.set_ylabel("density")
    ax.set_title(f"Truth looks like a member — {an['field']} at {an['lead_day']} d")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "fig_anomaly_overlay.png", dpi=130)
    plt.close(fig)
    print("wrote fig_anomaly_overlay.png")


def main():
    if (ART / "sweep_ckpt.pkl").exists():
        plot_convergence()
    else:
        print("(no sweep_ckpt.pkl yet -- skipping convergence figure)")
    if (ART / "calib_results.pkl").exists():
        r = _load_calib()
        plot_spread_error(r)
        plot_afcrps(r)
        plot_rank_hist(r)
        plot_anom(r)
    else:
        print("(no calib_results.pkl yet -- skipping calibration figures)")


if __name__ == "__main__":
    main()
