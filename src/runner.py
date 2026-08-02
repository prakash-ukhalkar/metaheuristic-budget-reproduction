"""Experiment driver.

Phases
  tune    matched-budget random-search tuning on a disjoint training set
  main    default hyperparameters, all algorithms x all problems
  tuned   tuned hyperparameters, all algorithms x all problems
  cons    constraint-handling sensitivity (static penalty, epsilon-constrained)

Seeding: run r on problem index j uses seed 10000*j + r for EVERY algorithm,
so comparisons across algorithms are paired by seed.
"""
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from problems import PROBLEMS, PROBLEM_MAP           # noqa: E402
from algorithms import ALGORITHMS, SPACES, DEFAULTS, run_one   # noqa: E402

BUDGET = 15000
N_RUNS = 51
TRAIN_PROBLEMS = ["PressureVessel", "CantileverBeam", "GearTrain"]
HELDOUT = [p.name for p in PROBLEMS if p.name not in TRAIN_PROBLEMS]
CONS_SUBSET = ["WeldedBeam", "SpeedReducer", "TensionSpring", "ThreeBarTruss"]
BUDGET_SUBSET = ["WeldedBeam", "SpeedReducer", "PressureVessel", "TensionSpring"]
TUNE_INIT = 32   # successive-halving racing budget
TUNE_SEEDS = 8

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)


def sample_config(alg, rng):
    cfg = {}
    for k, spec in SPACES[alg].items():
        kind, lo, hi = spec
        cfg[k] = int(rng.integers(lo, hi + 1)) if kind == "int" else float(rng.uniform(lo, hi))
    return cfg


def normalised_score(res, problem):
    """Scale-free score: relative gap to the published best, +10 if infeasible."""
    if not res["feasible"]:
        return 10.0
    kf = problem.known_f
    denom = max(abs(kf), 1e-8)
    return min(10.0, abs(res["best_f"] - kf) / denom)


def phase_tune():
    """Matched-budget tuning by successive halving (a simple racing scheme).

    Every algorithm gets the identical protocol: 32 sampled configurations are
    raced on the three training problems, the worst half eliminated at each
    round while the per-configuration seed count doubles, until one survives.
    Total cost is identical across algorithms by construction.
    """
    rng = np.random.default_rng(20260801)
    best, log = {}, []
    for alg in ALGORITHMS:
        if not SPACES[alg]:
            best[alg] = {}
            continue
        cands = [sample_config(alg, rng) for _ in range(TUNE_INIT)]
        seeds = 1
        rnd = 0
        while len(cands) > 1:
            scores = []
            for ci, cfg in enumerate(cands):
                s = []
                for pn in TRAIN_PROBLEMS:
                    p = PROBLEM_MAP[pn]
                    for r in range(seeds):
                        res = run_one(alg, p, BUDGET, 777000 + 97 * rnd + 31 * ci + r,
                                      params=cfg)
                        s.append(normalised_score(res, p))
                scores.append(float(np.mean(s)))
            keep = np.argsort(scores)[:max(1, len(cands) // 2)]
            log.append(dict(alg=alg, round=rnd, n_configs=len(cands), seeds=seeds,
                            best_score=float(np.min(scores))))
            cands = [cands[i] for i in keep]
            seeds = min(TUNE_SEEDS, seeds * 2)
            rnd += 1
        best[alg] = cands[0]
        print(f"[tune] {alg:<8} cfg={cands[0]}", flush=True)
    json.dump(best, open(f"{OUT}/tuned_params.json", "w"), indent=2)
    json.dump(log, open(f"{OUT}/tuning_log.json", "w"), indent=2)


def sweep(tag, params_map, problems, scheme="deb"):
    fn = f"{OUT}/{tag}_{scheme}.json"
    rows = json.load(open(fn)) if os.path.exists(fn) and os.environ.get("RESUME") else []
    done = {r["problem"] for r in rows}
    t0 = time.time()
    for j, pn in enumerate(problems):
        if pn in done:
            continue
        p = PROBLEM_MAP[pn]
        for alg in ALGORITHMS:
            for r in range(N_RUNS):
                seed = 10000 * j + r
                res = run_one(alg, p, BUDGET, seed,
                              params=params_map.get(alg), scheme=scheme)
                rows.append(dict(phase=tag, scheme=scheme, problem=pn, algorithm=alg,
                                 run=r, seed=seed, feasible=res["feasible"],
                                 best_f=res["best_f"] if res["feasible"] else None,
                                 violation=res["best_viol"],
                                 trace=res["trace"]))
        json.dump(rows, open(fn, "w"))
        print(f"[{tag}/{scheme}] {pn} done  ({time.time()-t0:.0f}s)", flush=True)
    return rows


if __name__ == "__main__":
    phase = sys.argv[1]
    if phase == "tune":
        phase_tune()
    elif phase == "main":
        sweep("main", {a: DEFAULTS.get(a, {}) for a in ALGORITHMS},
              [p.name for p in PROBLEMS])
    elif phase == "tuned":
        tp = json.load(open(f"{OUT}/tuned_params.json"))
        lim = int(sys.argv[2]) if len(sys.argv) > 2 else 99
        sweep("tuned", tp, [p.name for p in PROBLEMS][:lim])
    elif phase == "budget":
        d = {a: DEFAULTS.get(a, {}) for a in ALGORITHMS}
        import runner as _self
        for b in ([int(x) for x in (sys.argv[2].split(",") if len(sys.argv)>2 else ["5000","50000"])]):
            _self.BUDGET = b
            globals()["BUDGET"] = b
            rows = []
            for j, pn in enumerate(BUDGET_SUBSET):
                p = PROBLEM_MAP[pn]
                for alg in ALGORITHMS:
                    for r in range(15):
                        res = run_one(alg, p, b, 10000 * j + r, params=d.get(alg))
                        rows.append(dict(phase=f"budget{b}", scheme="deb", problem=pn,
                                         algorithm=alg, run=r, seed=10000 * j + r,
                                         feasible=res["feasible"],
                                         best_f=res["best_f"] if res["feasible"] else None,
                                         violation=res["best_viol"], trace=res["trace"]))
                print(f"[budget{b}] {pn} done", flush=True)
            json.dump(rows, open(f"{OUT}/budget{b}_deb.json", "w"))
    elif phase == "cons":
        d = {a: DEFAULTS.get(a, {}) for a in ALGORITHMS}
        for sc in ("static", "eps"):
            sweep("cons", d, CONS_SUBSET, scheme=sc)
    print("PHASE COMPLETE:", phase, flush=True)
