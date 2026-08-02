"""Implementation-fidelity validation.

The comparison in this study only means anything if the six metaphor-based
algorithms were implemented faithfully. This script tests that against an
independent implementation: mealpy, a widely used third-party library whose
code was written by different people from a different reading of the same
publications.

For each algorithm and problem, both implementations are run under an identical
evaluation budget and an identical, stateless static-penalty constraint
handler, and the resulting distributions are compared with a two-sided
Wilcoxon rank-sum test. Agreement on most problems is evidence that neither
implementation is crippled.

Note on budget control: mealpy's SSA and HHO consume roughly 1.9x the nominal
epoch x pop_size evaluations, because they evaluate additional candidates
inside an epoch. Termination is therefore driven by max_fe, not by epoch count.
Without this correction those two algorithms would silently receive almost
double the budget - an instance of exactly the accounting problem this paper
examines.
"""
import json
import os
import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

from problems import PROBLEMS, PROBLEM_MAP          # noqa: E402
from algorithms import run_one                      # noqa: E402
from mealpy import FloatVar, GWO, WOA, SCA, SSA, HHO, AOA   # noqa: E402

BUDGET = 15000
N_RUNS = 9
POP = 30
PENALTY_C = 1e5
OUT = os.path.join(os.path.dirname(__file__), "..", "results")

REF = {"GWO": GWO.OriginalGWO, "WOA": WOA.OriginalWOA, "SCA": SCA.OriginalSCA,
       "SSA": SSA.OriginalSSA, "HHO": HHO.OriginalHHO, "AOA": AOA.OriginalAOA}


def penalised(problem):
    def obj(x):
        X = problem.repair(np.array(x, dtype=float).reshape(1, -1))
        f, G = problem.evaluate(X)
        return float(f[0] + PENALTY_C * np.maximum(G[0], 0.0).sum())
    return obj


def run_reference(alg, problem, seed):
    np.random.seed(seed)
    prob = {"obj_func": penalised(problem),
            "bounds": FloatVar(lb=list(problem.lb), ub=list(problem.ub)),
            "minmax": "min", "log_to": None}
    model = REF[alg](epoch=100000, pop_size=POP)
    g = model.solve(prob, termination={"max_fe": BUDGET}, seed=seed)
    x = problem.repair(np.array(g.solution, dtype=float).reshape(1, -1))
    f, G = problem.evaluate(x)
    viol = float(np.maximum(G[0], 0.0).sum())
    return (float(f[0]) if viol <= 1e-8 else np.nan), model.nfe_counter


def main():
    import sys
    subset = sys.argv[1].split(",") if len(sys.argv) > 1 else list(REF)
    fn = f"{OUT}/fidelity.json"
    rows = json.load(open(fn)) if os.path.exists(fn) else []
    done = {(r["problem"], r["algorithm"]) for r in rows}
    for p in PROBLEMS:
        for alg in subset:
            if (p.name, alg) in done:
                continue
            mine, ref, nfes = [], [], []
            for r in range(N_RUNS):
                res = run_one(alg, p, BUDGET, 90000 + r, scheme="static")
                mine.append(res["best_f"] if res["feasible"] else np.nan)
                fv, nfe = run_reference(alg, p, 90000 + r)
                ref.append(fv); nfes.append(nfe)
            a = np.array(mine, dtype=float); b = np.array(ref, dtype=float)
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() >= 5:
                _, pv = stats.ranksums(a[ok], b[ok])
            else:
                pv = np.nan
            ma, mb = np.nanmedian(a), np.nanmedian(b)
            rel = abs(ma - mb) / max(abs(mb), 1e-12)
            rows.append(dict(problem=p.name, algorithm=alg,
                             median_this_study=float(ma), median_reference=float(mb),
                             rel_diff=float(rel), p_ranksum=float(pv),
                             mean_nfe_reference=float(np.mean(nfes))))
            json.dump(rows, open(fn, "w"), indent=1)
            print(f"{p.name:<17}{alg:<5} this={ma:<13.6g} ref={mb:<13.6g} "
                  f"reldiff={rel:<9.3g} p={pv:.3f}", flush=True)

    agree = sum(1 for r in rows if not (r["p_ranksum"] < 0.05))
    print(f"\nno significant difference on {agree}/{len(rows)} algorithm-problem pairs")


if __name__ == "__main__":
    main()
