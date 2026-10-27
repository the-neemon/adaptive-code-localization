"""Figures for the final report, from data/results (Phase A and held-out, nDCG@300).

Run from the repository root:  python reports/final/make_figures.py
Writes reports/final/figures/*.pdf.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from acl import analysis, heldout, tier2

OUT = Path(__file__).parent / "figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, RED = "#2a78d6", "#e34948"                 # validated categorical pair (light surface)
RAMP = ["#86b6ef", "#2a78d6", "#104281"]          # validated ordinal blue ramp (low -> high)
LABEL = {"stop": "stop (seed)", "expand_inherit": "expand(inherit)", "expand_imports": "expand(imports)",
         "expand_calls": "expand(calls)", "dense": "dense", "fusion": "fusion"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Times New Roman"],
    "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})


def fig_fixed_vs_oracle(d: analysis.Data) -> None:
    acts = d.actions
    names = [LABEL[a] for a in acts] + ["per-issue oracle"]
    vals = [d.score[a] for a in acts] + [np.max(np.stack([d.score[a] for a in acts]), axis=0)]
    means = [v.mean() for v in vals]
    cis = [analysis.ci(v, d.repo) for v in vals]
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    y = np.arange(len(names))[::-1]
    colors = [BLUE] * len(acts) + [RED]
    ax.barh(y, means, height=0.62, color=colors, zorder=2)
    for yi, m, (lo, hi) in zip(y, means, cis):
        ax.plot([lo, hi], [yi, yi], color=INK, lw=0.8, zorder=3)
        ax.text(hi + 0.006, yi, f"{m:.3f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(y, names)
    ax.set_xlabel("mean nDCG@300, 500 issues (95% repo-bootstrap CI)")
    ax.set_xlim(0, 0.42)
    ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fixed_vs_oracle.pdf")


def fig_helps_hurts(d: analysis.Data) -> None:
    acts = [a for a in d.actions if a != "stop"]
    helps = [analysis.helps_hurts(d.delta(a))[0] * 100 for a in acts]
    hurts = [analysis.helps_hurts(d.delta(a))[1] * 100 for a in acts]
    fig, ax = plt.subplots(figsize=(3.3, 1.75))
    y = np.arange(len(acts))[::-1]
    ax.barh(y, [-h for h in hurts], height=0.6, color=RED, zorder=2, label="hurts (gain < 0)")
    ax.barh(y, helps, height=0.6, color=BLUE, zorder=2, label="helps (gain > 0)")
    for yi, h, u in zip(y, helps, hurts):
        ax.text(h + 1.5, yi, f"{h:.0f}%", va="center", fontsize=7, color=INK)
        ax.text(-u - 1.5, yi, f"{u:.0f}%", va="center", ha="right", fontsize=7, color=INK)
    ax.axvline(0, color=INK2, lw=0.6, zorder=3)
    ax.set_yticks(y, [LABEL[a] for a in acts])
    ax.set_xlim(-68, 68)
    ax.set_xticks([-50, -25, 0, 25, 50], ["50%", "25%", "0", "25%", "50%"])
    ax.set_xlabel("share of issues, nDCG@300 vs seed (rest: no change)")
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7,
              handlelength=1.0, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "helps_hurts.pdf")


def fig_degree(d: analysis.Data) -> None:
    acts = ["expand_calls", "expand_imports", "dense", "fusion"]
    deg = d.feats["top_degree"]
    q1, q2 = np.quantile(deg, [1 / 3, 2 / 3])
    groups = [(f"degree $\\leq$ {q1:.0f}", deg <= q1), (f"{q1:.0f} < degree $\\leq$ {q2:.0f}", (deg > q1) & (deg <= q2)),
              (f"degree > {q2:.0f} (hub)", deg > q2)]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    x = np.arange(len(acts))
    w = 0.26
    for k, ((name, m), color) in enumerate(zip(groups, RAMP)):
        means = [d.delta(a)[m].mean() for a in acts]
        cis = [analysis.ci(d.delta(a)[m], d.repo[m]) for a in acts]
        xs = x + (k - 1) * (w + 0.02)
        ax.bar(xs, means, width=w, color=color, zorder=2, label=f"{name} (n={m.sum()})")
        for xi, (lo, hi) in zip(xs, cis):
            ax.plot([xi, xi], [lo, hi], color=INK, lw=0.7, zorder=3)
    ax.axhline(0, color=INK2, lw=0.6, zorder=3)
    ax.set_xticks(x, [LABEL[a].replace("(", "\n(") for a in acts])
    ax.set_ylabel("gain in nDCG@300 over the seed")
    ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.tick_params(axis="x", length=0)
    ax.legend(title="graph degree of the top seed hit", title_fontsize=7, fontsize=6.5, frameon=False,
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.0, columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "degree_effect.pdf")


def fig_heldout(d: analysis.Data) -> None:
    """Gain over the fixed action on the 444 held-out issues, with the oracle as a reference line."""
    h = heldout.held_out_choices(d)
    held, choice = h["held"], dict(h["choice"])
    choice["tier2"] = tier2.held_out_choices(d)[0]
    s = {k: heldout.scores_of(d, c, held) for k, c in choice.items()}
    idx = np.flatnonzero(held)
    oracle = np.array([max(d.score[a][i] for a in d.actions) for i in idx]).mean() - s["fixed"].mean()
    rows = [("stop", "seed only (stop)", INK2), ("router", "Tier 1 router (up to 5 rules)", BLUE),
            ("router_1", "Tier 1, first rule only", BLUE), ("tier2", "Tier 2 classifier", BLUE)]
    fig, ax = plt.subplots(figsize=(3.3, 1.8))
    y = np.arange(len(rows))[::-1]
    for yi, (k, name, color) in zip(y, rows):
        diff = s[k] - s["fixed"]
        lo, hi = np.quantile(analysis.boot_means(diff, d.repo[held]), [0.025, 0.975])
        ax.plot([lo, hi], [yi, yi], color=color, lw=1.6, solid_capstyle="round", zorder=3)
        ax.plot(diff.mean(), yi, "o", ms=5, color=color, mec="white", mew=1.0, zorder=4)
        # Values sit in a right margin, clear of the intervals.
        ax.text(1.03, yi, f"{diff.mean():+.4f}", transform=ax.get_yaxis_transform(), va="center",
                ha="left", fontsize=7, color=INK)
    ax.axvline(0, color=INK2, lw=0.8, zorder=2)
    ax.set_yticks(y, [r[1] for r in rows])
    ax.set_xlim(-0.03, 0.012)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("gain in nDCG@300 over fusion (95% CI)")
    print(f"held-out oracle gain over fusion: {oracle:+.4f}")
    ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.3, rect=(0, 0, 0.86, 1))
    fig.savefig(OUT / "heldout.pdf")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = analysis.load(300, "ndcg")
    fig_fixed_vs_oracle(d)
    fig_helps_hurts(d)
    fig_degree(d)
    fig_heldout(d)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
