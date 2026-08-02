import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from analysis import load, ORDER, TEST_GROUP, TRAIN_PROBLEMS, MARKERS, LINES, FIG

dm, dt = load("main"), load("tuned")
SCALE = [p for p in dm["problem"].unique() if p != "GearTrain"]
HO = [p for p in SCALE if p not in TRAIN_PROBLEMS]

def medgap(df, probs):
    return df[df.problem.isin(probs)].groupby(["algorithm","problem"])["gap"].median().unstack()[probs].mean(axis=1)

d0, d1 = medgap(dm, HO), medgap(dt, HO)
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for i, a in enumerate(ORDER):
    y0, y1 = max(d0[a], 1e-6), max(d1[a], 1e-6)
    ax.plot([0, 1], [y0, y1], LINES[i], marker=MARKERS[i], ms=5, lw=1.3,
            color="k" if a in TEST_GROUP else "0.55", label=a)
    ax.annotate(a, (1.02, y1), fontsize=7, va="center")
ax.set_xticks([0, 1], ["author defaults", "matched-budget tuning"])
ax.set_yscale("log"); ax.set_xlim(-0.1, 1.3)
ax.set_ylabel("mean median relative gap (5 held-out problems)")
ax.legend(ncol=2, fontsize=7, loc="lower left")
fig.savefig(f"{FIG}/fig6_tuning.png", bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 3.8))
data = [np.maximum(dm[(dm.algorithm==a) & (dm.problem.isin(SCALE))]["gap"].values, 1e-9) for a in ORDER]
bp = ax.boxplot(data, tick_labels=ORDER, showfliers=True, patch_artist=True,
                flierprops=dict(marker="+", ms=3))
for i, b in enumerate(bp["boxes"]):
    b.set_facecolor("0.88" if ORDER[i] in TEST_GROUP else "0.55")
ax.set_yscale("log"); ax.set_ylabel("relative gap to published best")
ax.set_xticklabels(ORDER, rotation=45, ha="right")
fig.savefig(f"{FIG}/fig8_boxplot.png", bbox_inches="tight"); plt.close(fig)

# Fig 1: study design schematic
fig, ax = plt.subplots(figsize=(7.2, 3.4)); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 5)
def box(x, y, w, h, t, fs=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                 fc="white", ec="k", lw=0.9))
    ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=fs)
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->",
                 mutation_scale=9, lw=0.8, color="k"))
box(0.1, 3.5, 2.3, 1.1, "9 design problems\nverified against\npublished optima")
box(0.1, 1.9, 2.3, 1.1, "11 algorithms\n6 metaphor-based\n5 baselines")
box(0.1, 0.3, 2.3, 1.1, "3 constraint\nschemes: Deb,\nstatic, $\\epsilon$")
box(3.1, 2.6, 2.2, 1.1, "matched-budget\ntuning\n(3 training problems)")
box(3.1, 1.0, 2.2, 1.1, "author-default\nhyperparameters")
box(6.0, 1.8, 1.9, 1.9, "25 seeded runs\n15,000 FEs\nshared evaluator\npaired seeds")
box(8.4, 1.8, 1.5, 1.9, "Friedman\n+ Nemenyi\nWilcoxon\n+ Holm\n$\\hat{A}_{12}$")
for y in (4.05, 2.45, 0.85): arrow(2.4, y, 3.05, 2.2 if y != 1.55 else y)
arrow(5.3, 3.15, 5.95, 2.9); arrow(5.3, 1.55, 5.95, 2.3); arrow(7.9, 2.75, 8.35, 2.75)
fig.savefig(f"{FIG}/fig1_design.png", bbox_inches="tight", dpi=200); plt.close(fig)
print("regenerated fig1, fig6, fig8")
print("default:", {a: round(d0[a],5) for a in ORDER})
print("tuned  :", {a: round(d1[a],5) for a in ORDER})
