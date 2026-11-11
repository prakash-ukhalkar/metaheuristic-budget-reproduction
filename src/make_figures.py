"""Manuscript figures (v2).  Reads results/analysis.json and the raw sweeps."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis as A

FIG = os.path.join(A.HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": False, "figure.dpi": 200,
                     "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
LABEL = {"LSHADE": "L-SHADE", "CMAES": "CMA-ES", "RS": "Random search"}
MET_C, BASE_C, EXT_C = "#C0392B", "#1F4E79", "#E69F00"
lab = lambda a: LABEL.get(a, a.replace("mp_", "mp-"))
# line charts: one colour and marker per algorithm (Okabe-Ito based); solid = baselines, dashed = metaphor-based
STY = {"DE": ("#0072B2", "o"), "PSO": ("#56B4E9", "s"), "LSHADE": ("#000000", "^"), "CMAES": ("#009E73", "D"),
       "RS": ("#888888", "x"), "GWO": ("#D55E00", "o"), "WOA": ("#CC79A7", "s"), "SCA": ("#E69F00", "^"),
       "SSA": ("#8C564B", "D"), "HHO": ("#A6761D", "v"), "AOA": ("#6A3D9A", "P")}
kind = lambda a: BASE_C if a in A.BASELINES else (EXT_C if a in A.EXT or a.startswith("mp_") else MET_C)


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf")
    fig.savefig(f"{FIG}/{name}.png")
    plt.close(fig)


def fig_design():
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.axis("off")
    boxes = [
        (0.00, 0.62, 0.31, 0.34, "19 problems\n9 engineering design\n10 CEC 2006 (2 to 20 variables)\npublished optimum reproduced"),
        (0.00, 0.08, 0.31, 0.40, "23 algorithms\n6 metaphor-based (own code)\n6 same, third-party code\n6 further metaphor-based\n5 reference algorithms"),
        (0.39, 0.30, 0.25, 0.40, "One shared evaluator\nbudget metering\nbound repair, violation\nanytime trace\n3 constraint schemes"),
        (0.70, 0.70, 0.30, 0.26, "Protocol factors\n5,000 to 150,000 evaluations\ntuning: 32 vs 128 settings\nheld-out problems"),
        (0.70, 0.36, 0.30, 0.26, "Paired seeds\n31 to 51 runs per pair\nFriedman, Wilcoxon + Holm,\neffect size, bootstrap"),
        (0.70, 0.02, 0.30, 0.26, "Robustness checks\nsecond implementation\nbest-of-both implementation\nexhaustive formulation audit"),
    ]
    for x, y, w, h, t in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fc="#F4F6F8", ec="#34495E", lw=1.0))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.4)
    for (x0, y0, x1, y1) in [(0.31, 0.79, 0.39, 0.58), (0.31, 0.28, 0.39, 0.42),
                             (0.64, 0.50, 0.70, 0.82), (0.64, 0.50, 0.70, 0.49), (0.64, 0.42, 0.70, 0.15)]:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#34495E"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "Figure_1")


def cd_diagram(ranks, cd, ax, ci=None):
    names = list(ranks)
    r = np.array([ranks[n] for n in names])
    k = len(names)
    lo, hi = 1, k
    ax.set_xlim(hi + 0.4, lo - 0.4)      # best (low rank) on the right? keep best on the left:
    ax.set_xlim(lo - 0.4, hi + 0.4)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y0 = 0.62
    ax.plot([lo, hi], [y0, y0], "k", lw=1)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [y0, y0 + 0.02], "k", lw=1)
        ax.text(t, y0 + 0.05, str(t), ha="center", va="bottom", fontsize=8)
    half = (k + 1) // 2
    for i, (n, v) in enumerate(zip(names, r)):
        up = i < half
        ytxt = 0.40 - 0.045 * i if up else 0.40 - 0.045 * (k - 1 - i) - 0.0
        ytxt = y0 - 0.08 - 0.055 * i if i < half else y0 - 0.08 - 0.055 * (k - 1 - i)
        x_txt = lo - 0.3 if i < half else hi + 0.3
        ha = "right" if i < half else "left"
        ax.plot([v, v, x_txt + (0.05 if i < half else -0.05)], [y0, ytxt, ytxt], color="#555", lw=0.7)
        ax.text(x_txt, ytxt, f"{lab(n)}  ({v:.2f})", ha=ha, va="center", fontsize=8,
                color=kind(n))
    # cliques of algorithms not separated by the critical difference
    ybar = y0 + 0.17
    placed = []
    i = 0
    while i < k:
        j = i
        while j + 1 < k and r[j + 1] - r[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in placed):
            ax.plot([r[i] - 0.02, r[j] + 0.02], [ybar, ybar], "k", lw=2.2)
            ybar += 0.035
            placed.append((i, j))
        i += 1
    ax.plot([lo, lo + cd], [y0 + 0.40, y0 + 0.40], "k", lw=1.5)
    ax.text(lo + cd / 2, y0 + 0.43, f"CD = {cd:.2f}", ha="center", fontsize=8)


def fig_cd(an):
    f = an["friedman_core"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    cd_diagram(f["ranks"], f["cd"], ax)
    ax.set_ylim(-0.02, 1.15)
    save(fig, "Figure_2")


def fig_wtl(an):
    P = pd.read_csv(f"{A.TAB}/paired_wilcoxon_core.csv")
    bases = ["DE", "LSHADE", "CMAES", "PSO", "RS"]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    M = np.zeros((6, 5))
    txt = {}
    for i, a in enumerate(A.METAPHOR):
        for j, b in enumerate(bases):
            s = P[(P.test == a) & (P.base == b)].outcome.value_counts()
            w, t, l = s.get("win", 0), s.get("tie", 0), s.get("loss", 0)
            M[i, j] = w - l
            txt[i, j] = f"{w}/{t}/{l}"
    im = ax.imshow(M, cmap="RdBu", vmin=-19, vmax=19, aspect="auto")
    for (i, j), t in txt.items():
        ax.text(j, i, t, ha="center", va="center", fontsize=8,
                color="white" if abs(M[i, j]) > 11 else "black")
    ax.set_xticks(range(5)); ax.set_xticklabels([lab(b) for b in bases])
    ax.set_yticks(range(6)); ax.set_yticklabels(A.METAPHOR)
    fig.colorbar(im, ax=ax, label="wins - losses")
    save(fig, "Figure_3")


def fig_heat():
    main = A.load_tag("main")
    ref = A.Ref(main)
    main["ngap"] = ref.ngap(main)
    M = A.med_matrix(main[main.run < 51], A.CORE, A.PNAMES)
    order = list(M.mean(0).rank().sort_values().index)
    Z = np.log10(np.clip(M[order].T.values, 1e-6, 10))
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    im = ax.imshow(Z, cmap="viridis_r", aspect="auto", vmin=-6, vmax=1)
    ax.set_xticks(range(len(A.PNAMES))); ax.set_xticklabels(A.PNAMES, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([lab(a) for a in order])
    fig.colorbar(im, ax=ax, label="log10 median normalised gap")
    save(fig, "Figure_4")


def fig_anytime():
    tr = A.load_traces("main")
    main = A.load_tag("main")
    ref = A.Ref(main)
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    for ax, tau in zip(axs, (1e-2, 1e-4)):
        xs = np.linspace(150, 15000, 100)
        curves = {}
        for r in tr:
            if r["algorithm"] not in A.CORE or r["run"] >= 51:
                continue
            fs = A.FSTAR[r["problem"]]
            den = max(abs(ref.med[r["problem"]] - fs), 1e-300)
            t = np.array([np.nan if v is None else v for v in r["trace"]], dtype=float)
            hit = ((t - fs) / den) <= tau
            curves.setdefault(r["algorithm"], []).append(hit)
        for a in A.CORE:
            c = np.mean(curves[a], axis=0)
            ax.plot(xs, c, color=STY[a][0], ls="-" if a in A.BASELINES else "--", lw=1.4,
                    marker=STY[a][1], ms=3.2, markevery=12, label=lab(a))
        ax.set_xlabel("function evaluations"); ax.set_title(f"normalised gap <= {tau:g}", fontsize=9)
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("fraction of runs (19 problems)")
    axs[1].legend(fontsize=6.5, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "Figure_5")


def fig_impl(an):
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.2), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axs[0]
    ip = an["impl_pairs"]
    y = np.arange(6)
    a1 = [ip[a]["ours_better"] for a in A.METAPHOR]
    a2 = [ip[a]["equivalent"] for a in A.METAPHOR]
    a3 = [ip[a]["twin_better"] for a in A.METAPHOR]
    ax.barh(y, a1, color=MET_C, label="in-house better")
    ax.barh(y, a2, left=a1, color="#BBBBBB", label="no difference")
    ax.barh(y, a3, left=np.array(a1) + np.array(a2), color=EXT_C, label="third-party better")
    ax.set_yticks(y); ax.set_yticklabels(A.METAPHOR); ax.invert_yaxis()
    ax.set_xlabel("problems (of 19)"); ax.legend(fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    ax = axs[1]
    f = an["friedman_oracle_impl"]["ranks"]
    names = list(f)
    ax.barh(range(len(names)), [f[n] for n in names],
            color=[BASE_C if n in A.BASELINES else MET_C for n in names])
    ax.set_yticks(range(len(names))); ax.set_yticklabels([lab(n) for n in names]); ax.invert_yaxis()
    ax.set_xlabel("Friedman mean rank (best implementation)")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.2, top=0.86, wspace=0.55)
    save(fig, "Figure_6")


def fig_tuning(an):
    t = an.get("tuning_mean_median_ngap")
    if not t:
        return
    algs = A.CORE[:-1]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    w = 0.26
    x = np.arange(len(algs))
    for i, (k, c, nm) in enumerate([("default", "#BBBBBB", "default settings"), ("tuned32", "#5D8AA8", "tuned, 32 configurations"),
                                    ("tuned128", "#1F4E79", "tuned, 128 configurations")]):
        if k in t:
            ax.bar(x + (i - 1) * w, [t[k]["heldout"][a] for a in algs], w, color=c, label=nm)
    ax.set_xticks(x); ax.set_xticklabels([lab(a) for a in algs], rotation=30)   # linear axis: bars start at zero
    ax.set_ylabel("mean median normalised gap\n(held-out problems)"); ax.legend(fontsize=7)
    save(fig, "Figure_7")


def fig_sens(an):
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for ax, key, title, xl in [(axs[0], "friedman_schemes", "constraint-handling scheme", None),
                               (axs[1], "friedman_budgets", "evaluation budget", None)]:
        if key not in an:
            continue
        d = an[key]
        keys = list(d)
        for a in A.CORE:
            ax.plot(range(len(keys)), [d[k]["ranks"][a] for k in keys], marker=STY[a][1], ms=4,
                    color=STY[a][0], ls="-" if a in A.BASELINES else "--", lw=1.2, label=lab(a))
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys if key == "friedman_schemes" else [f"{int(k):,}" for k in keys])
        ax.set_xlabel(title); ax.set_ylabel("Friedman mean rank"); ax.invert_yaxis()
        ax.grid(alpha=0.25)
    axs[1].legend(fontsize=6, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "Figure_8")


def fig_ext(an):
    f = an["friedman_extended"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    cd_diagram(f["ranks"], f["cd"], ax)
    ax.set_ylim(-0.4, 1.15)
    save(fig, "Figure_9")


if __name__ == "__main__":
    an = json.load(open(f"{A.RES}/analysis.json"))
    fig_design(); fig_cd(an); fig_wtl(an); fig_heat(); fig_anytime()
    fig_impl(an); fig_tuning(an); fig_sens(an); fig_ext(an)
    print("figures written")
