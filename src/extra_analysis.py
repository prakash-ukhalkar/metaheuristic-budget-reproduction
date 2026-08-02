"""Fidelity, budget-sensitivity and headline statistics for the revised manuscript."""
import json, os, numpy as np, pandas as pd
from analysis import load, ORDER, TEST_GROUP, BASELINES, TRAIN_PROBLEMS, friedman_nemenyi, pairwise_wilcoxon
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
res = {}

dm, dt = load("main"), load("tuned")
allp = list(dm.problem.unique())
scale = [p for p in allp if p != "GearTrain"]
ho = [p for p in scale if p not in TRAIN_PROBLEMS]

_, st, pv, ranks, cd = friedman_nemenyi(dm, allp)
R = pairwise_wilcoxon(dm, allp)
res["friedman"] = dict(stat=float(st), p=float(pv), cd=float(cd), ranks=ranks.to_dict())
res["wtl_total"] = R.outcome.value_counts().to_dict()
res["median_A12"] = float(R.A12.median())
res["wtl_vs"] = {b: R[R.base == b].outcome.value_counts().to_dict() for b in BASELINES}
res["wtl_by_alg_vs_CMAES"] = {a: R[(R.test==a)&(R.base=="CMAES")].outcome.value_counts().to_dict() for a in TEST_GROUP}
res["wtl_by_alg_vs_RS"] = {a: R[(R.test==a)&(R.base=="RS")].outcome.value_counts().to_dict() for a in TEST_GROUP}
res["hit_rate"] = (dm[dm.problem.isin(scale)].assign(h=lambda d: d.gap<=1e-4)
                   .groupby("algorithm")["h"].mean().reindex(ORDER).to_dict())
res["feasibility"] = dm.groupby("algorithm")["feasible"].mean().to_dict()
med = dm[dm.problem.isin(scale)].groupby(["algorithm","problem"])["gap"].median().unstack()
tun = dt[dt.problem.isin(scale)].groupby(["algorithm","problem"])["gap"].median().unstack()
res["tuning"] = {a: dict(default=float(med.loc[a, ho].mean()), tuned=float(tun.loc[a, ho].mean()))
                 for a in ORDER}
res["n_runs"] = int(dm.groupby(["algorithm","problem"]).size().max())

for sc in ("static", "eps"):
    d = load("cons", sc); cp = list(d.problem.unique())
    _, _, _, r, _ = friedman_nemenyi(d, cp)
    res.setdefault("rank_by_scheme", {})[sc] = r.to_dict()
cp = list(load("cons","static").problem.unique())
_, _, _, rdeb, _ = friedman_nemenyi(dm[dm.problem.isin(cp)], cp)
res["rank_by_scheme"]["deb"] = rdeb.to_dict()

for b in (5000, 50000):
    f = f"{OUT}/budget{b}_deb.json"
    if os.path.exists(f):
        d = load(f"budget{b}")
        bp = list(d.problem.unique())
        _, _, _, r, _ = friedman_nemenyi(d, bp)
        res.setdefault("rank_by_budget", {})[str(b)] = r.to_dict()
_, _, _, r15, _ = friedman_nemenyi(dm[dm.problem.isin(["WeldedBeam","SpeedReducer","PressureVessel","TensionSpring"])],
                                   ["WeldedBeam","SpeedReducer","PressureVessel","TensionSpring"])
res.setdefault("rank_by_budget", {})["15000"] = r15.to_dict()

if os.path.exists(f"{OUT}/fidelity.json"):
    F = pd.DataFrame(json.load(open(f"{OUT}/fidelity.json")))
    res["fidelity"] = dict(
        n_pairs=int(len(F)),
        n_agree=int((~(F.p_ranksum < 0.05)).sum()),
        median_rel_diff=float(F.rel_diff.median()),
        by_alg={a: dict(agree=int((~(F[F.algorithm==a].p_ranksum<0.05)).sum()),
                        n=int((F.algorithm==a).sum()),
                        median_rel_diff=float(F[F.algorithm==a].rel_diff.median()))
                for a in F.algorithm.unique()},
        reference_nfe_mean={a: float(F[F.algorithm==a].mean_nfe_reference.mean())
                            for a in F.algorithm.unique()})
json.dump(res, open(f"{OUT}/headline_v2.json","w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float)[:2500])
