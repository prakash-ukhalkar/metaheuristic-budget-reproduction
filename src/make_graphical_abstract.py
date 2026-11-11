"""Graphical abstract (landscape, 1328 x 531 px or larger).  Reads results/analysis.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
an = json.load(open(os.path.join(HERE, "..", "results", "analysis.json")))
OUT = os.path.join(HERE, "..", "figures")

LABEL = {"LSHADE": "L-SHADE", "CMAES": "CMA-ES", "RS": "Random search"}
BASE = ["DE", "PSO", "LSHADE", "CMAES", "RS"]
BASE_C, MET_C = "#1F4E79", "#C0392B"
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})

fig = plt.figure(figsize=(13.28, 5.31), dpi=150)
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.15], wspace=0.42, left=0.03, right=0.985, top=0.80, bottom=0.17)

# (1) protocol in words
ax = fig.add_subplot(gs[0]); ax.axis("off")
steps = ["19 constrained design problems", "23 algorithms,\none shared evaluator",
         "equal budget, paired seeds,\nthree constraint schemes", "tuning, budget and\nimplementation checks"]
for i, t in enumerate(steps):
    y = 0.84 - i * 0.235
    ax.add_patch(plt.Rectangle((0.0, y - 0.085), 1.0, 0.17, fc="#F4F6F8", ec="#34495E", lw=1.2, transform=ax.transAxes))
    ax.text(0.5, y, t, ha="center", va="center", fontsize=11, transform=ax.transAxes)
    if i < len(steps) - 1:
        ax.annotate("", xy=(0.5, y - 0.150), xytext=(0.5, y - 0.090), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#34495E", lw=1.4))
ax.set_title("Matched-budget protocol", fontsize=13, loc="left", fontweight="bold")

# (2) mean ranks
ax = fig.add_subplot(gs[1])
ranks = an["friedman_core"]["ranks"]
names = list(ranks)
y = np.arange(len(names))[::-1]
ax.barh(y, [ranks[n] for n in names], color=[BASE_C if n in BASE else MET_C for n in names], height=0.7)
ax.set_yticks(y); ax.set_yticklabels([LABEL.get(n, n) for n in names])
for yi, n in zip(y, names):
    ax.text(ranks[n] + 0.12, yi, f"{ranks[n]:.1f}", va="center", fontsize=10)
ax.set_xlabel("Friedman mean rank (lower is better)")
ax.set_xlim(0, 11.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("Reference algorithms lead", fontsize=13, loc="left", fontweight="bold")
ax.text(0.98, 0.86, "blue: reference algorithms\nred: metaphor-based", transform=ax.transAxes, ha="right", va="center", fontsize=10)

# (3) win / tie / loss of the metaphor algorithms against the references
ax = fig.add_subplot(gs[2])
w = an["wtl_by_algorithm"]
algs = list(w)
tot = {a: sum(w[a].values()) for a in algs}
y = np.arange(len(algs))[::-1]
left = np.zeros(len(algs))
for key, c, nm in (("win", "#2E8B57", "wins"), ("tie", "#BBBBBB", "ties"), ("loss", MET_C, "losses")):
    v = np.array([w[a][key] / tot[a] * 100 for a in algs])
    ax.barh(y, v, left=left, color=c, height=0.7, label=nm)
    left += v
ax.set_yticks(y); ax.set_yticklabels(algs)
ax.set_xlim(0, 100); ax.set_xlabel("Share of paired tests vs reference algorithms (%)")
ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
tw = an["wtl_all_baselines"]
ax.set_title("", fontsize=13)
fig.text(0.985, 0.955, f"Metaphor-based algorithms: {tw['win']} wins, {tw['tie']} ties, {tw['loss']} losses "
         f"in {sum(tw.values())} Holm-corrected paired tests", ha="right", va="top", fontsize=11.5, fontweight="bold")
fig.savefig(os.path.join(OUT, "Graphical_Abstract.png"), dpi=150)
fig.savefig(os.path.join(OUT, "Graphical_Abstract.pdf"))
print("graphical abstract written")
