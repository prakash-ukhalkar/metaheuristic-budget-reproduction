"""Statistical analysis for the v2 study.  Writes results/analysis.json and tables/*.csv.

Performance measure ("normalised gap"), scale-free so that every problem,
including GearTrain (f* ~ 1e-12) and G03 (f* < 0), can enter the aggregates:

    ngap = (f_best - f*) / (median_RS - f*)

where median_RS is the median final objective of uniform random search on the
same problem at the same budget (main sweep).  0 = reached the published best,
1 = as good as random search's median, infeasible run = PENALTY.

Rank statistics use the per-problem MEDIAN ngap over runs.
"""
import glob
import json
import os
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from problems import PROBLEM_MAP, PROBLEMS

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
TAB = os.path.join(RES, "tables")
os.makedirs(TAB, exist_ok=True)

PENALTY = 10.0
METAPHOR = ["GWO", "WOA", "SCA", "SSA", "HHO", "AOA"]
BASELINES = ["DE", "PSO", "LSHADE", "CMAES", "RS"]
CORE = METAPHOR + BASELINES
TWINS = ["mp_" + a for a in METAPHOR]
EXT = ["MFO", "FFA", "BA", "EO", "SMA", "MPA"]
PNAMES = [p.name for p in PROBLEMS]
TRAIN = ["PressureVessel", "CantileverBeam", "G04", "G08"]
HELDOUT = [p for p in PNAMES if p not in TRAIN]
FSTAR = {p.name: p.known_f for p in PROBLEMS}


def load_tag(tag):
    rows = []
    for fn in glob.glob(os.path.join(RES, tag, "*.json")):
        rows += json.load(open(fn))
    df = pd.DataFrame(rows)
    return df.drop(columns=["trace"]) if len(df) else df


def load_traces(tag):
    rows = []
    for fn in glob.glob(os.path.join(RES, tag, "*.json")):
        rows += json.load(open(fn))
    return rows


class Ref:
    """Random-search reference per problem for the normalised gap."""

    RS_MIN_FEASIBLE = 0.5   # random search must be feasible in at least this share of runs

    def __init__(self, main):
        allrs = main[main.algorithm == "RS"]
        rs = allrs[allrs.feasible]
        self.med = rs.groupby("problem")["best_f"].median().to_dict()
        frac = allrs.groupby("problem")["feasible"].mean().to_dict()
        # Fixed rule, independent of any compared algorithm: where random search is rarely
        # feasible (equality constraints, tiny feasible region) its median is not a usable
        # scale, so the scale falls back to |f*| (a relative error).
        self.fallback = sorted(p for p in FSTAR if frac.get(p, 0.0) < self.RS_MIN_FEASIBLE)
        for p in self.fallback:
            self.med[p] = FSTAR[p] + max(abs(FSTAR[p]), 1e-8)
        self.rs_feasible_share = {p: float(frac.get(p, 0.0)) for p in FSTAR}

    def ngap(self, df):
        fs = df["problem"].map(FSTAR)
        den = (df["problem"].map(self.med) - fs).abs().clip(lower=1e-300)
        g = (df["best_f"] - fs) / den
        return np.where(df["feasible"], np.clip(g, -1.0, PENALTY), PENALTY)


# ------------------------------------------------------------------ statistics
def nemenyi_cd(k, n, alpha=0.05):
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2.0)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


def med_matrix(df, algs, problems, col="ngap", nruns=None):
    d = df[df.algorithm.isin(algs) & df.problem.isin(problems)]
    if nruns:
        d = d[d.run < nruns]
    return d.groupby(["problem", "algorithm"])[col].median().unstack()[algs].loc[problems]


def friedman(M):
    stat, p = stats.friedmanchisquare(*[M[c].values for c in M.columns])
    ranks = M.rank(axis=1, method="average").mean(axis=0)
    return dict(chi2=float(stat), p=float(p), n=int(M.shape[0]), k=int(M.shape[1]),
                cd=nemenyi_cd(M.shape[1], M.shape[0]), ranks=ranks.sort_values().to_dict())


def boot_rank_ci(M, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    R = M.rank(axis=1).values
    n = R.shape[0]
    bs = np.array([R[rng.integers(0, n, n)].mean(0) for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return {a: (float(l), float(h)) for a, l, h in zip(M.columns, lo, hi)}


def a12(x, y):
    """Vargha-Delaney: P(x < y) + 0.5 P(x == y); > 0.5 means x (lower) is better."""
    x, y = np.asarray(x), np.asarray(y)
    m, n = len(x), len(y)
    r = stats.rankdata(np.concatenate([x, y]))[:m].sum()
    return 1.0 - (r / m - (m + 1) / 2) / n      # rankdata favours larger; flip


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx])
        adj[idx] = min(1.0, run)
    return adj


def paired_tests(df, tests, bases, problems, nruns=None):
    """Paired Wilcoxon signed-rank per (problem, test, base); Holm over all."""
    recs = []
    for pn in problems:
        d = df[df.problem == pn]
        if nruns:
            d = d[d.run < nruns]
        piv = d.pivot_table(index="run", columns="algorithm", values="ngap")
        for a in tests:
            for b in bases:
                x, y = piv[a].values, piv[b].values
                ok = np.isfinite(x) & np.isfinite(y)
                x, y = x[ok], y[ok]
                if np.allclose(x, y):
                    pv = 1.0
                else:
                    try:
                        pv = float(stats.wilcoxon(x, y).pvalue)
                    except ValueError:
                        pv = 1.0
                recs.append(dict(problem=pn, test=a, base=b, p=pv,
                                 A12_base_better=a12(y, x),
                                 med_diff=float(np.median(x) - np.median(y))))
    R = pd.DataFrame(recs)
    R["p_holm"] = holm(R["p"].values)
    R["outcome"] = np.where(R.p_holm >= 0.05, "tie", np.where(R.med_diff < 0, "win", "loss"))
    return R


def wtl(R):
    c = R.outcome.value_counts()
    return dict(win=int(c.get("win", 0)), tie=int(c.get("tie", 0)), loss=int(c.get("loss", 0)))


# ------------------------------------------------------------------ main
def main():
    out = {}
    main_df = load_tag("main")
    ref = Ref(main_df)
    main_df["ngap"] = ref.ngap(main_df)
    out["rs_median_ref"] = ref.med
    out["rs_reference_fallback_problems"] = ref.fallback
    out["rs_feasible_share"] = ref.rs_feasible_share

    # ---- admission check 2: nothing beats the published optimum -----------
    allrows = [main_df]
    for tag in glob.glob(os.path.join(RES, "*", "")):
        t = os.path.basename(os.path.dirname(tag))
        if t not in ("main", "tables", "") and not t.startswith("tune"):   # tune* hold tuner state, not runs
            d = load_tag(t)
            if len(d):
                allrows.append(d)
    A = pd.concat(allrows)
    A = A[A.feasible & (A.violation <= 1e-8)]
    adm = []
    for pn in PNAMES:
        fs = FSTAR[pn]
        best = float(A[A.problem == pn].best_f.min())
        beat = (fs - best) / max(abs(fs), 1e-12)
        adm.append(dict(problem=pn, published=fs, best_found=best, rel_beat=float(beat),
                        n_feasible=int((A.problem == pn).sum())))
    out["admission"] = adm
    pd.DataFrame(adm).to_csv(f"{TAB}/admission.csv", index=False)

    # ---- core comparison -------------------------------------------------
    core = main_df[main_df.algorithm.isin(CORE) & (main_df.run < 51)]
    M = med_matrix(core, CORE, PNAMES)
    M.to_csv(f"{TAB}/median_ngap_core.csv")
    fr = friedman(M)
    fr["ci"] = boot_rank_ci(M)
    out["friedman_core"] = fr

    # Demsar-style signed-rank across problems on per-problem medians, Holm
    srecs = []
    for a in METAPHOR:
        for b in BASELINES:
            d = M[a] - M[b]
            try:
                pv = float(stats.wilcoxon(M[a], M[b]).pvalue) if np.any(d != 0) else 1.0
            except ValueError:
                pv = 1.0
            srecs.append(dict(test=a, base=b, p=pv, n_better=int((d < 0).sum()),
                              n_worse=int((d > 0).sum())))
    S = pd.DataFrame(srecs)
    S["p_holm"] = holm(S.p.values)
    S.to_csv(f"{TAB}/signed_rank_across_problems.csv", index=False)
    out["signed_rank"] = dict(
        n_significant_losses=int(((S.p_holm < 0.05) & (S.n_worse > S.n_better)).sum()),
        n_significant_wins=int(((S.p_holm < 0.05) & (S.n_better > S.n_worse)).sum()),
        n_tests=len(S))

    P = paired_tests(core, METAPHOR, BASELINES, PNAMES)
    P.to_csv(f"{TAB}/paired_wilcoxon_core.csv", index=False)
    out["wtl_all_baselines"] = wtl(P)
    out["wtl_vs_baseline"] = {b: wtl(P[P.base == b]) for b in BASELINES}
    out["wtl_by_algorithm"] = {a: wtl(P[P.test == a]) for a in METAPHOR}
    out["median_A12_baseline_better"] = float(P.A12_base_better.median())
    top3 = ["DE", "LSHADE", "CMAES"]
    out["wtl_vs_top3"] = wtl(P[P.base.isin(top3)])
    # excluding random search from the baseline set
    out["wtl_excl_RS"] = wtl(P[P.base != "RS"])

    # ---- robustness of the core ranking to the normalisation ----------------
    # GearTrain's random-search scale is ~1e-9, so its gaps are dominated by clipping; the four
    # fallback problems use |f*| as the scale. Recompute the ranking without each group.
    def ranks_without(drop):
        keep = [p for p in PNAMES if p not in drop]
        r = friedman(M.loc[keep])
        return dict(n=len(keep), chi2=r["chi2"], cd=r["cd"], ranks=r["ranks"])
    out["robust_no_geartrain"] = ranks_without(["GearTrain"])
    out["robust_no_fallback"] = ranks_without(ref.fallback)
    out["robust_no_geartrain_no_fallback"] = ranks_without(["GearTrain"] + ref.fallback)
    base_order = list(fr["ranks"])
    out["robust_rank_order_kendall_tau"] = {
        k: float(stats.kendalltau([fr["ranks"][a] for a in CORE], [out[k]["ranks"][a] for a in CORE])[0])
        for k in ("robust_no_geartrain", "robust_no_fallback", "robust_no_geartrain_no_fallback")}
    pt = paired_tests(core, METAPHOR, BASELINES, [p for p in PNAMES if p != "GearTrain"])
    out["robust_wtl_no_geartrain"] = wtl(pt)

    sub = core[core.algorithm.isin(CORE)]
    hit = (sub.assign(h=(sub.ngap <= 1e-4)).groupby("algorithm")
           .agg(hit=("h", "mean"), feas=("feasible", "mean")))
    out["hit_rate"] = hit["hit"].to_dict()
    out["feas_rate"] = hit["feas"].to_dict()
    out["median_ngap_overall"] = (core.groupby("algorithm")["ngap"].median()).to_dict()

    # ---- extended algorithm set (31 paired runs) ---------------------------
    ext_algs = CORE + EXT
    m31 = main_df[main_df.run < 31]
    M2 = med_matrix(m31, ext_algs, PNAMES)
    out["friedman_extended"] = friedman(M2)
    P2 = paired_tests(m31, EXT, BASELINES, PNAMES, nruns=31)
    P2.to_csv(f"{TAB}/paired_wilcoxon_ext.csv", index=False)
    out["wtl_ext_vs_baselines"] = wtl(P2)
    out["wtl_ext_vs_top3"] = wtl(P2[P2.base.isin(top3)])
    out["wtl_ext_by_algorithm"] = {a: wtl(P2[P2.test == a]) for a in EXT}

    # ---- implementation sensitivity -----------------------------------------
    imp = []
    for a in METAPHOR:
        recs = paired_tests(m31, [a], ["mp_" + a], PNAMES, nruns=31)
        recs["alg"] = a
        imp.append(recs)
    I = pd.concat(imp)
    I.to_csv(f"{TAB}/implementation_pairs.csv", index=False)
    out["impl_pairs"] = {
        a: dict(ours_better=int(((I.alg == a) & (I.outcome == "win")).sum()),
                equivalent=int(((I.alg == a) & (I.outcome == "tie")).sum()),
                twin_better=int(((I.alg == a) & (I.outcome == "loss")).sum()))
        for a in METAPHOR}
    out["impl_total"] = dict(ours_better=int((I.outcome == "win").sum()),
                             equivalent=int((I.outcome == "tie").sum()),
                             twin_better=int((I.outcome == "loss").sum()), n=len(I))
    base_alg = CORE[6:]
    Mt = med_matrix(m31, TWINS + base_alg, PNAMES)
    out["friedman_twins_baselines"] = friedman(Mt)
    Mb = med_matrix(m31, METAPHOR + TWINS + base_alg, PNAMES)
    out["friedman_both_impls"] = friedman(Mb)
    # oracle: for every algorithm and problem keep the better of the two implementations
    Mo = pd.DataFrame({a: np.minimum(Mb[a], Mb["mp_" + a]) for a in METAPHOR})
    for b in base_alg:
        Mo[b] = Mb[b]
    out["friedman_oracle_impl"] = friedman(Mo)
    # paired tests of the twin group against baselines (same protocol as the core)
    Pt = paired_tests(m31, TWINS, BASELINES, PNAMES, nruns=31)
    out["wtl_twins_vs_baselines"] = wtl(Pt)
    out["wtl_twins_vs_top3"] = wtl(Pt[Pt.base.isin(top3)])
    # and the in-house group on the same 31 runs, for like-for-like comparison
    Po = paired_tests(m31, METAPHOR, BASELINES, PNAMES, nruns=31)
    out["wtl_ours_vs_baselines_31"] = wtl(Po)
    out["wtl_ours_vs_top3_31"] = wtl(Po[Po.base.isin(top3)])

    # ---- tuning ---------------------------------------------------------------
    tun = {}
    for n_init in (32, 128):
        d = load_tag(f"tuned{n_init}")
        if not len(d):
            continue
        d["ngap"] = ref.ngap(d)
        tun[n_init] = d
    if tun:
        dflt = main_df[main_df.algorithm.isin(CORE)]
        tab = {}
        for name, d in [("default", dflt)] + [(f"tuned{k}", v) for k, v in tun.items()]:
            tab[name] = {
                "heldout": med_matrix(d, CORE, HELDOUT, nruns=31).mean(0).to_dict(),
                "train": med_matrix(d, CORE, TRAIN, nruns=31).mean(0).to_dict()}
        out["tuning_mean_median_ngap"] = tab
        for k, d in tun.items():
            Mh = med_matrix(d, CORE, HELDOUT, nruns=31)
            out[f"friedman_tuned{k}_heldout"] = friedman(Mh)
            Pk = paired_tests(d, METAPHOR, BASELINES, HELDOUT, nruns=31)
            out[f"wtl_tuned{k}_heldout"] = wtl(Pk)
            Pk.to_csv(f"{TAB}/paired_wilcoxon_tuned{k}.csv", index=False)
            out[f"wtl_tuned{k}_vs_top3"] = wtl(Pk[Pk.base.isin(top3)])
        Md = med_matrix(dflt, CORE, HELDOUT, nruns=31)
        out["friedman_default_heldout"] = friedman(Md)
        # per-algorithm paired effect of tuning (held-out): default vs tuned
        eff = {}
        for k, d in tun.items():
            rows = []
            for a in CORE[:-1]:
                x = dflt[(dflt.algorithm == a) & dflt.problem.isin(HELDOUT) & (dflt.run < 31)]
                y = d[(d.algorithm == a) & d.problem.isin(HELDOUT) & (d.run < 31)]
                mx = x.groupby("problem").ngap.median()
                my = y.groupby("problem").ngap.median()
                better = int((my < mx).sum()); worse = int((my > mx).sum())
                try:
                    pv = float(stats.wilcoxon(mx, my).pvalue)
                except ValueError:
                    pv = 1.0
                rows.append(dict(alg=a, improved=better, worsened=worse, p=pv))
            eff[k] = rows
        out["tuning_effect"] = eff

    # ---- constraint-handling schemes -------------------------------------------
    cons = {"deb": main_df[main_df.algorithm.isin(CORE)]}
    for sc in ("static", "eps"):
        d = load_tag(f"cons_{sc}")
        if len(d):
            d["ngap"] = ref.ngap(d)
            cons[sc] = d
    if len(cons) > 1:
        rk = {}
        for sc, d in cons.items():
            Ms = med_matrix(d, CORE, PNAMES, nruns=31)
            f = friedman(Ms)
            rk[sc] = f
            if sc != "deb":
                Ps = paired_tests(d, METAPHOR, BASELINES, PNAMES, nruns=31)
                out[f"wtl_cons_{sc}"] = wtl(Ps)
                out[f"wtl_cons_{sc}_vs_top3"] = wtl(Ps[Ps.base.isin(top3)])
        out["friedman_schemes"] = rk
        sc_names = list(rk)
        tau = {}
        for a, b in itertools.combinations(sc_names, 2):
            ra = [rk[a]["ranks"][x] for x in CORE]
            rb = [rk[b]["ranks"][x] for x in CORE]
            tau[f"{a}~{b}"] = float(stats.kendalltau(ra, rb).statistic)
        out["scheme_kendall_tau"] = tau

    # ---- budget -------------------------------------------------------------------
    bud = {15000: main_df[main_df.algorithm.isin(CORE)]}
    for b in (5000, 50000, 150000):
        d = load_tag(f"budget{b}")
        if len(d):
            d["ngap"] = ref.ngap(d)
            bud[b] = d
    if len(bud) > 1:
        rk = {}
        for b, d in sorted(bud.items()):
            Mbud = med_matrix(d, CORE, PNAMES, nruns=15)
            rk[b] = friedman(Mbud)
            Pb = paired_tests(d, METAPHOR, BASELINES, PNAMES, nruns=15)
            rk[b]["wtl"] = wtl(Pb)
            rk[b]["wtl_vs_top3"] = wtl(Pb[Pb.base.isin(top3)])
        out["friedman_budgets"] = {str(k): v for k, v in rk.items()}
        ks = sorted(rk)
        out["budget_kendall_tau"] = {
            f"{a}~{b}": float(stats.kendalltau([rk[a]["ranks"][x] for x in CORE],
                                               [rk[b]["ranks"][x] for x in CORE]).statistic)
            for a, b in itertools.combinations(ks, 2)}

    json.dump(out, open(f"{RES}/analysis.json", "w"), indent=1, default=float)
    print("wrote analysis.json")
    return out


if __name__ == "__main__":
    main()
