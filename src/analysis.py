"""Statistical analysis and figures.

Primary metric: relative gap to the published best-known objective,
    gap = (f_best - f*) / |f*|
Infeasible runs are assigned the penalty value PENALTY so that they rank
below every feasible run.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from problems import PROBLEM_MAP

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

PENALTY = 10.0
TEST_GROUP = ["GWO", "WOA", "SCA", "SSA", "HHO", "AOA"]
BASELINES = ["DE", "PSO", "LSHADE", "CMAES", "RS"]
ORDER = TEST_GROUP + BASELINES
TRAIN_PROBLEMS = ["PressureVessel", "CantileverBeam", "GearTrain"]

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 200, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})
MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h"]
LINES = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--", "-."]


EXCLUDED = ["DiscBrake"]   # formulation under-constrained; see manuscript 4.5


def load(tag, scheme="deb"):
    rows = json.load(open(f"{OUT}/{tag}_{scheme}.json"))
    df = pd.DataFrame(rows)
    df = df[~df["problem"].isin(EXCLUDED)].reset_index(drop=True)
    df["fstar"] = df["problem"].map(lambda p: PROBLEM_MAP[p].known_f)
    gap = (df["best_f"] - df["fstar"]) / df["fstar"].abs()
    df["gap"] = np.where(df["feasible"], np.clip(gap, 0, PENALTY), PENALTY)
    return df


def summary_table(df):
    g = df.groupby(["problem", "algorithm"])
    t = g.agg(best=("best_f", "min"), median=("best_f", "median"),
              mean=("best_f", "mean"), std=("best_f", "std"),
              worst=("best_f", "max"), feas_rate=("feasible", "mean"),
              mean_gap=("gap", "mean")).reset_index()
    return t


def vargha_delaney(a, b):
    """A12 effect size: P(a < b) + 0.5 P(a == b) for minimisation."""
    a, b = np.asarray(a), np.asarray(b)
    m, n = len(a), len(b)
    r = stats.rankdata(np.concatenate([a, b]))[:m].sum()
    return (r / m - (m + 1) / 2) / n


def friedman_nemenyi(df, problems):
    """Friedman on the problem x algorithm matrix of mean gaps."""
    M = (df[df["problem"].isin(problems)]
         .groupby(["problem", "algorithm"])["gap"].mean().unstack()[ORDER])
    stat, p = stats.friedmanchisquare(*[M[c].values for c in M.columns])
    ranks = M.rank(axis=1).mean(axis=0)
    k, N = M.shape[1], M.shape[0]
    # Nemenyi critical difference at alpha = 0.05
    q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
           9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}[k]
    cd = q05 * np.sqrt(k * (k + 1) / (6.0 * N))
    return M, stat, p, ranks.sort_values(), cd


def pairwise_wilcoxon(df, problems):
    """Paired Wilcoxon per problem, each test algorithm vs each baseline,
    Holm-corrected across all comparisons."""
    recs = []
    for pn in problems:
        sub = df[df["problem"] == pn]
        piv = sub.pivot_table(index="run", columns="algorithm", values="gap")
        for a in TEST_GROUP:
            for b in BASELINES:
                x, y = piv[a].values, piv[b].values
                if np.allclose(x, y):
                    pval, statv = 1.0, np.nan
                else:
                    try:
                        statv, pval = stats.wilcoxon(x, y)
                    except ValueError:
                        statv, pval = np.nan, 1.0
                recs.append(dict(problem=pn, test=a, base=b, stat=statv, p=pval,
                                 A12=vargha_delaney(x, y),
                                 median_diff=np.median(x) - np.median(y)))
    R = pd.DataFrame(recs)
    order = np.argsort(R["p"].values)
    m = len(R)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        v = (m - i) * R["p"].values[idx]
        running = max(running, v)
        adj[idx] = min(1.0, running)
    R["p_holm"] = adj
    R["outcome"] = np.where(R["p_holm"] >= 0.05, "tie",
                            np.where(R["median_diff"] < 0, "win", "loss"))
    return R


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_cd(ranks, cd, path):
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    names = list(ranks.index)
    vals = ranks.values
    lo, hi = np.floor(vals.min()) - 0.5, np.ceil(vals.max()) + 0.5
    ax.set_xlim(hi, lo)
    ax.set_ylim(0, 1)
    ax.hlines(0.85, lo, hi, color="k", lw=1)
    for t in np.arange(np.ceil(lo), np.floor(hi) + 1):
        ax.vlines(t, 0.85, 0.90, color="k", lw=1)
        ax.text(t, 0.95, f"{int(t)}", ha="center", fontsize=8)
    half = int(np.ceil(len(names) / 2))
    for i, (nm, v) in enumerate(zip(names, vals)):
        if i < half:
            y = 0.72 - 0.11 * i
            ax.plot([v, v, lo + 0.15], [0.85, y, y], color="k", lw=0.8)
            ax.text(lo + 0.12, y, f"{nm} ({v:.2f})", ha="right", va="center", fontsize=8)
        else:
            y = 0.72 - 0.11 * (i - half)
            ax.plot([v, v, hi - 0.15], [0.85, y, y], color="k", lw=0.8)
            ax.text(hi - 0.12, y, f"{nm} ({v:.2f})", ha="left", va="center", fontsize=8)
    ax.plot([lo + 0.3, lo + 0.3 + cd], [0.99, 0.99], color="k", lw=2.5)
    ax.text(lo + 0.3 + cd / 2, 1.02, f"CD = {cd:.2f}", ha="center", fontsize=8)
    ax.axis("off")
    fig.savefig(path)
    plt.close(fig)


def fig_ecdf(df, path, budget=15000):
    """Anytime ECDF: fraction of (run, problem) pairs within 1% of f* vs budget."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    fes = np.linspace(budget / 100, budget, 100)
    for i, a in enumerate(ORDER):
        sub = df[df["algorithm"] == a]
        T = np.array([np.array(t) for t in sub["trace"]])
        fstar = sub["fstar"].values[:, None]
        gap = np.abs(T - fstar) / np.abs(fstar)
        gap = np.where(np.isfinite(gap), gap, np.inf)
        frac = (gap <= 0.01).mean(axis=0)
        ax.plot(fes, frac, LINES[i], marker=MARKERS[i], markevery=12, ms=4,
                lw=1.2, label=a)
    ax.set_xlabel("function evaluations")
    ax.set_ylabel("fraction of runs within 1% of published best")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(ncol=2, fontsize=8, loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def fig_convergence(df, path, problems, budget=15000):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    fes = np.linspace(budget / 100, budget, 100)
    for ax, pn in zip(axes.ravel(), problems):
        sub = df[df["problem"] == pn]
        fstar = PROBLEM_MAP[pn].known_f
        for i, a in enumerate(ORDER):
            T = np.array([np.array(t) for t in sub[sub["algorithm"] == a]["trace"]])
            gap = np.abs(T - fstar) / abs(fstar)
            gap = np.where(np.isfinite(gap), gap, PENALTY)
            med = np.median(gap, axis=0)
            ax.plot(fes, np.maximum(med, 1e-12), LINES[i], marker=MARKERS[i],
                    markevery=15, ms=3, lw=1.0, label=a)
        ax.set_yscale("log")
        ax.set_title(pn, fontsize=9)
        ax.set_xlabel("function evaluations")
        ax.set_ylabel("median relative gap")
    axes[0, 0].legend(ncol=3, fontsize=6.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_wtl(R, path):
    M = np.zeros((len(TEST_GROUP), len(BASELINES), 3))
    for i, a in enumerate(TEST_GROUP):
        for j, b in enumerate(BASELINES):
            s = R[(R["test"] == a) & (R["base"] == b)]["outcome"].value_counts()
            M[i, j] = [s.get("win", 0), s.get("tie", 0), s.get("loss", 0)]
    net = M[:, :, 0] - M[:, :, 2]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    im = ax.imshow(net, cmap="RdBu", vmin=-9, vmax=9)
    ax.set_xticks(range(len(BASELINES)), BASELINES)
    ax.set_yticks(range(len(TEST_GROUP)), TEST_GROUP)
    for i in range(len(TEST_GROUP)):
        for j in range(len(BASELINES)):
            w, t, l = M[i, j].astype(int)
            ax.text(j, i, f"{w}/{t}/{l}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("baseline")
    ax.set_ylabel("metaphor-based algorithm")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="net wins (win - loss)")
    fig.savefig(path)
    plt.close(fig)


def fig_tuning(dmain, dtuned, path, problems):
    a_def = (dmain[dmain["problem"].isin(problems)]
             .groupby("algorithm")["gap"].mean())
    a_tun = (dtuned[dtuned["problem"].isin(problems)]
             .groupby("algorithm")["gap"].mean())
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for i, a in enumerate(ORDER):
        y0, y1 = max(a_def[a], 1e-8), max(a_tun[a], 1e-8)
        ax.plot([0, 1], [y0, y1], LINES[i], marker=MARKERS[i], ms=5, lw=1.2, label=a)
    ax.set_xticks([0, 1], ["author defaults", "matched-budget tuning"])
    ax.set_yscale("log")
    ax.set_ylabel("mean relative gap (held-out problems)")
    ax.set_xlim(-0.15, 1.5)
    ax.legend(ncol=2, fontsize=8, loc="center right")
    fig.savefig(path)
    plt.close(fig)


def fig_constraint(dcons_static, dcons_eps, dmain, path, problems):
    schemes = {"Deb rules": dmain, "static penalty": dcons_static,
               "epsilon-constrained": dcons_eps}
    vals = {k: [max(v[v["problem"].isin(problems)]
                    .groupby("algorithm")["gap"].mean()[a], 1e-8) for a in ORDER]
            for k, v in schemes.items()}
    x = np.arange(len(ORDER))
    w = 0.26
    hatches = ["", "//", ".."]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, (k, v) in enumerate(vals.items()):
        ax.bar(x + (i - 1) * w, v, w, label=k, hatch=hatches[i],
               edgecolor="k", linewidth=0.5, color=f"{0.35 + 0.25 * i}")
    ax.set_yscale("log")
    ax.set_xticks(x, ORDER, rotation=45, ha="right")
    ax.set_ylabel("mean relative gap")
    ax.legend(fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def fig_box(df, path, problems):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    data = [np.maximum(df[(df["algorithm"] == a) & (df["problem"].isin(problems))]["gap"].values, 1e-12)
            for a in ORDER]
    bp = ax.boxplot(data, labels=ORDER, showfliers=True, patch_artist=True,
                    flierprops=dict(marker="+", ms=3))
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor("0.85" if ORDER[i] in TEST_GROUP else "0.6")
    ax.set_yscale("log")
    ax.set_ylabel("relative gap to published best")
    ax.set_xticklabels(ORDER, rotation=45, ha="right")
    fig.savefig(path)
    plt.close(fig)


def main():
    dmain = load("main")
    dtuned = load("tuned")
    heldout = [p for p in dmain["problem"].unique() if p not in TRAIN_PROBLEMS]
    allp = list(dmain["problem"].unique())

    summary_table(dmain).to_csv(f"{OUT}/table_summary_default.csv", index=False)
    summary_table(dtuned).to_csv(f"{OUT}/table_summary_tuned.csv", index=False)

    M, stat, p, ranks, cd = friedman_nemenyi(dmain, allp)
    Mh, stath, ph, ranksh, cdh = friedman_nemenyi(dmain, heldout)
    R = pairwise_wilcoxon(dmain, allp)
    R.to_csv(f"{OUT}/table_wilcoxon.csv", index=False)

    res = dict(
        friedman_all=dict(stat=float(stat), p=float(p), cd=float(cd),
                          ranks=ranks.to_dict(), n_problems=len(allp)),
        friedman_heldout=dict(stat=float(stath), p=float(ph), cd=float(cdh),
                              ranks=ranksh.to_dict(), n_problems=len(heldout)),
        feasibility=dmain.groupby("algorithm")["feasible"].mean().to_dict(),
        mean_gap_default=dmain.groupby("algorithm")["gap"].mean().to_dict(),
        mean_gap_tuned=dtuned.groupby("algorithm")["gap"].mean().to_dict(),
        wtl={a: R[R["test"] == a]["outcome"].value_counts().to_dict() for a in TEST_GROUP},
    )

    fig_cd(ranks, cd, f"{FIG}/fig2_critical_difference.png")
    fig_ecdf(dmain, f"{FIG}/fig3_ecdf.png")
    fig_convergence(dmain, f"{FIG}/fig4_convergence.png",
                    ["WeldedBeam", "SpeedReducer", "PressureVessel", "TensionSpring"])
    fig_wtl(R, f"{FIG}/fig5_win_tie_loss.png")
    fig_tuning(dmain, dtuned, f"{FIG}/fig6_tuning.png", heldout)
    fig_box(dmain, f"{FIG}/fig8_boxplot.png", allp)

    if os.path.exists(f"{OUT}/cons_static.json"):
        ds = load("cons", "static")
        de = load("cons", "eps")
        cp = list(ds["problem"].unique())
        fig_constraint(ds, de, dmain, f"{FIG}/fig7_constraint_scheme.png", cp)
        res["constraint_scheme"] = dict(
            deb=dmain[dmain["problem"].isin(cp)].groupby("algorithm")["gap"].mean().to_dict(),
            static=ds.groupby("algorithm")["gap"].mean().to_dict(),
            eps=de.groupby("algorithm")["gap"].mean().to_dict())
        Ms, _, _, rs, _ = friedman_nemenyi(ds, cp)
        Me, _, _, re_, _ = friedman_nemenyi(de, cp)
        Md, _, _, rd, _ = friedman_nemenyi(dmain[dmain["problem"].isin(cp)], cp)
        res["rank_by_scheme"] = dict(deb=rd.to_dict(), static=rs.to_dict(), eps=re_.to_dict())

    json.dump(res, open(f"{OUT}/analysis.json", "w"), indent=2, default=float)
    print(json.dumps(res, indent=2, default=float)[:4000])


if __name__ == "__main__":
    main()
