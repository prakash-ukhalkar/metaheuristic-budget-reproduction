"""Generate every manuscript figure.

Figures are numbered in the order they are first referred to in the text, as
required by the journal. Each is written as both PNG (screen) and PDF (vector,
the journal's preferred submission format). Fonts are sized for 8-12 pt at the
final printed width. Colour is never the only carrier of information: every
line has a distinct dash pattern and marker, and the one heat map uses a
monotonic grey scale with the exact counts printed in each cell.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from analysis import (BASELINES, ORDER, TEST_GROUP, TRAIN_PROBLEMS,
                      load, pairwise_wilcoxon, friedman_nemenyi)
from problems import PROBLEM_MAP

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)
BUDGET = 15000

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "font.family": "DejaVu Sans", "pdf.fonttype": 42,
})
MARK = {"GWO": "o", "WOA": "s", "SCA": "^", "SSA": "v", "HHO": "D", "AOA": "P",
        "DE": "X", "PSO": "*", "LSHADE": "<", "CMAES": ">", "RS": "h"}
DASH = {"GWO": "-", "WOA": "--", "SCA": "-.", "SSA": ":", "HHO": "-", "AOA": "--",
        "DE": "-.", "PSO": ":", "LSHADE": "-", "CMAES": "--", "RS": "-."}
COL = {a: ("black" if a in TEST_GROUP else "0.45") for a in ORDER}


def save(fig, stem):
    fig.savefig(f"{FIG}/{stem}.png")
    fig.savefig(f"{FIG}/{stem}.pdf")
    plt.close(fig)


# --------------------------------------------------------------- Figure 1
def fig1_design():
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)

    def box(x, y, w, h, t):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07",
                                    fc="white", ec="k", lw=0.9))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.5)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->",
                                     mutation_scale=8, lw=0.8, color="k"))

    box(0.1, 3.55, 2.2, 1.05, "8 design problems\nverified against\npublished optima")
    box(0.1, 2.00, 2.2, 1.05, "11 algorithms\n6 metaphor-based\n5 baselines")
    box(0.1, 0.45, 2.2, 1.05, "3 constraint\nschemes: Deb,\nstatic, $\\epsilon$")
    box(3.05, 2.70, 2.1, 1.05, "matched-budget\ntuning on 3\ntraining problems")
    box(3.05, 1.15, 2.1, 1.05, "author-default\nhyperparameters")
    box(5.95, 1.60, 1.9, 1.85, "51 seeded runs\n15,000 evaluations\nshared evaluator\nseeds paired\nacross algorithms")
    box(8.30, 1.60, 1.6, 1.85, "Friedman\n+ Nemenyi\n\nWilcoxon\n+ Holm\n\n$\\hat{A}_{12}$")
    for y in (4.07, 2.52, 0.97):
        arrow(2.32, y, 3.00, 3.05 if y > 3.5 else (2.20 if y < 1.5 else 2.60))
    arrow(5.17, 3.20, 5.90, 2.90)
    arrow(5.17, 1.65, 5.90, 2.10)
    arrow(7.88, 2.52, 8.25, 2.52)
    save(fig, "fig1_design")


# --------------------------------------------------------------- Figure 2
def fig2_cd(ranks, cd):
    """Critical difference diagram with labels held clear of the leader lines."""
    names, vals = list(ranks.index), ranks.values
    lo, hi = np.floor(vals.min()) - 0.5, np.ceil(vals.max()) + 0.5
    pad = 1.35                       # horizontal room reserved for text
    half = int(np.ceil(len(names) / 2))
    row_h = 0.34
    top = 0.0
    bottom = -row_h * (half + 1.6)

    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    ax.set_xlim(hi + pad, lo - pad)          # reversed: rank 1 on the right
    ax.set_ylim(bottom, 1.05)
    ax.axis("off")

    ax.hlines(top, lo, hi, color="k", lw=1.1)
    for tck in np.arange(np.ceil(lo), np.floor(hi) + 1):
        ax.vlines(tck, top, top + 0.07, color="k", lw=1.0)
        ax.text(tck, top + 0.13, f"{int(tck)}", ha="center", va="bottom", fontsize=8)

    ax.plot([lo, lo - cd], [top + 0.62, top + 0.62], color="k", lw=3.0,
            solid_capstyle="butt")
    ax.text(lo - cd / 2, top + 0.68, f"critical difference = {cd:.2f}",
            ha="center", va="bottom", fontsize=8)

    # Row order must keep each connector's leader line clear of every other
    # row's text. The label margin for the "left" (best-rank) group sits
    # below the group's minimum value, so the shallowest row must hold the
    # smallest value (ascending). The margin for the "right" (worst-rank)
    # group sits above the group's maximum value, so it must be reversed
    # (descending) or every deeper row's leader line crosses the shallower
    # rows' text on its way out to the margin.
    left_group = sorted(range(half), key=lambda i: vals[i])
    right_group = sorted(range(half, len(names)), key=lambda i: -vals[i])

    # Screen-space (points) offset from the leader line's endpoint, so the
    # gap is unaffected by the reversed, non-uniform x/y data scaling.
    for slot, i in enumerate(left_group):
        nm, v = names[i], vals[i]
        y = top - row_h * (slot + 1)
        x_end = lo - 0.10
        # single diagonal leader: touches the text zone only at its own row,
        # so it cannot cross through another row's label text.
        ax.plot([v, x_end], [top, y], color="k", lw=0.7)
        ax.annotate(f"{nm} ({v:.2f})", xy=(x_end, y), xycoords="data",
                    xytext=(-6, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8)

    for slot, i in enumerate(right_group):
        nm, v = names[i], vals[i]
        y = top - row_h * (slot + 1)
        x_end = hi + 0.10
        ax.plot([v, x_end], [top, y], color="k", lw=0.7)
        ax.annotate(f"{nm} ({v:.2f})", xy=(x_end, y), xycoords="data",
                    xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8)
    save(fig, "fig2_critical_difference")


# --------------------------------------------------------------- Figure 3
def fig3_wtl(R):
    M = np.zeros((len(TEST_GROUP), len(BASELINES), 3))
    for i, a in enumerate(TEST_GROUP):
        for j, b in enumerate(BASELINES):
            s = R[(R.test == a) & (R.base == b)].outcome.value_counts()
            M[i, j] = [s.get("win", 0), s.get("tie", 0), s.get("loss", 0)]
    net = M[:, :, 0] - M[:, :, 2]

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    im = ax.imshow(net, cmap="Greys_r", vmin=-9, vmax=9)
    ax.set_xticks(range(len(BASELINES)), BASELINES)
    ax.set_yticks(range(len(TEST_GROUP)), TEST_GROUP)
    for i in range(len(TEST_GROUP)):
        for j in range(len(BASELINES)):
            w, t, l = M[i, j].astype(int)
            ax.text(j, i, f"{w}/{t}/{l}", ha="center", va="center", fontsize=8,
                    color="white" if net[i, j] < -2 else "black")
    ax.set_xlabel("baseline"); ax.set_ylabel("metaphor-based algorithm")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("net wins (wins − losses)", fontsize=8)
    save(fig, "fig3_win_tie_loss")


# --------------------------------------------------------------- Figure 4
def fig4_ecdf(dm):
    fes = np.linspace(BUDGET / 100, BUDGET, 100)
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for a in ORDER:
        sub = dm[dm.algorithm == a]
        T = np.array([np.array(x) for x in sub["trace"]])
        fs = sub["fstar"].values[:, None]
        gap = np.abs(T - fs) / np.abs(fs)
        frac = (np.where(np.isfinite(gap), gap, np.inf) <= 0.01).mean(axis=0)
        ax.plot(fes, frac, DASH[a], marker=MARK[a], markevery=14, ms=4, lw=1.2,
                color=COL[a], label=a)
    ax.set_xlabel("function evaluations")
    ax.set_ylabel("fraction of runs within 1% of published best")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, handlelength=3.0)
    save(fig, "fig4_ecdf")


# --------------------------------------------------------------- Figure 5
def fig5_convergence(dm, problems):
    fes = np.linspace(BUDGET / 100, BUDGET, 100)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
    for ax, pn in zip(axes.ravel(), problems):
        sub = dm[dm.problem == pn]
        fstar = PROBLEM_MAP[pn].known_f
        for a in ORDER:
            T = np.array([np.array(x) for x in sub[sub.algorithm == a]["trace"]])
            gap = np.abs(T - fstar) / abs(fstar)
            med = np.median(np.where(np.isfinite(gap), gap, 10.0), axis=0)
            ax.plot(fes, np.maximum(med, 1e-12), DASH[a], marker=MARK[a],
                    markevery=18, ms=3, lw=0.9, color=COL[a], label=a)
        ax.set_yscale("log"); ax.set_title(pn)
        ax.set_xlabel("function evaluations")
        ax.set_ylabel("median relative gap")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.07),
               frameon=False, handlelength=3.0)
    fig.tight_layout()
    save(fig, "fig5_convergence")


# --------------------------------------------------------------- Figure 6
def fig6_box(dm, problems):
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    data = [np.maximum(dm[(dm.algorithm == a) & (dm.problem.isin(problems))]["gap"].values, 1e-9)
            for a in ORDER]
    bp = ax.boxplot(data, tick_labels=ORDER, showfliers=True, patch_artist=True,
                    widths=0.6, flierprops=dict(marker="+", ms=3, mew=0.6),
                    medianprops=dict(color="k", lw=1.2))
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor("0.92" if ORDER[i] in TEST_GROUP else "0.6")
        b.set_edgecolor("k"); b.set_linewidth(0.7)
    ax.axvline(len(TEST_GROUP) + 0.5, color="k", lw=0.8, ls=":")
    ax.text(3.5, ax.get_ylim()[1], "metaphor-based", ha="center", va="top", fontsize=8)
    ax.text(9.0, ax.get_ylim()[1], "baselines", ha="center", va="top", fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("relative gap to published best")
    ax.set_xticklabels(ORDER, rotation=45, ha="right")
    save(fig, "fig6_boxplot")


# --------------------------------------------------------------- Figure 7
def fig7_tuning(dm, dt, heldout):
    def mg(df):
        return (df[df.problem.isin(heldout)]
                .groupby(["algorithm", "problem"])["gap"].median()
                .unstack()[heldout].mean(axis=1))
    d0, d1 = mg(dm), mg(dt)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for a in ORDER:
        ax.plot([0, 1], [max(d0[a], 1e-6), max(d1[a], 1e-6)], DASH[a],
                marker=MARK[a], ms=5, lw=1.2, color=COL[a], label=a)
    ax.set_xticks([0, 1], ["author defaults", "matched-budget tuning"])
    ax.set_yscale("log"); ax.set_xlim(-0.08, 1.08)
    ax.set_ylabel("mean median relative gap (5 held-out problems)")
    ax.legend(ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, handlelength=3.0)
    save(fig, "fig7_tuning")


# --------------------------------------------------------------- Figure 8
def fig8_constraint(dm, ds, de, problems):
    sets = {"Deb feasibility rules": dm[dm.problem.isin(problems)],
            "static penalty": ds, "$\\epsilon$-constrained": de}
    vals = {k: [max(v.groupby("algorithm")["gap"].mean().get(a, np.nan), 1e-8) for a in ORDER]
            for k, v in sets.items()}
    x = np.arange(len(ORDER)); w = 0.27
    hatch = ["", "///", "..."]; shade = ["0.30", "0.60", "0.85"]
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    for i, (k, v) in enumerate(vals.items()):
        ax.bar(x + (i - 1) * w, v, w, label=k, hatch=hatch[i],
               color=shade[i], edgecolor="k", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_xticks(x, ORDER, rotation=45, ha="right")
    ax.set_ylabel("mean relative gap")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    save(fig, "fig8_constraint_scheme")


def main():
    dm, dt = load("main"), load("tuned")
    ds, de = load("cons", "static"), load("cons", "eps")
    allp = list(dm.problem.unique())
    scale = [p for p in allp if p != "GearTrain"]
    heldout = [p for p in scale if p not in TRAIN_PROBLEMS]
    consp = list(ds.problem.unique())
    _, _, _, ranks, cd = friedman_nemenyi(dm, allp)
    R = pairwise_wilcoxon(dm, allp)

    fig1_design()
    fig2_cd(ranks, cd)
    fig3_wtl(R)
    fig4_ecdf(dm)
    fig5_convergence(dm, ["WeldedBeam", "SpeedReducer", "PressureVessel", "TensionSpring"])
    fig6_box(dm, scale)
    fig7_tuning(dm, dt, heldout)
    fig8_constraint(dm, ds, de, consp)
    print("figures 1-8 written as PNG + PDF")


if __name__ == "__main__":
    main()
