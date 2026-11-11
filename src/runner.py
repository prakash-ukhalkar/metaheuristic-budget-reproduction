"""Experiment driver (v2): parallel, resumable, one result file per job.

Phases
  tune     successive-halving racing, once with 32 and once with 128 initial
           configurations (the 4x run answers "was the tuning budget enough?")
  main     default hyper-parameters, every algorithm x every problem, Deb rules
  tuned    tuned hyper-parameters (tune32 / tune128), core algorithms
  cons     constraint-handling sensitivity (static penalty, epsilon-constrained)
  budget   evaluation-budget sensitivity (5k, 50k, 150k), core algorithms

Seeding: run r on problem index j uses seed 10000*j + r for EVERY algorithm, so
cross-algorithm comparisons are paired by seed.  Tuning seeds (777000+) are
disjoint from evaluation seeds.

Usage:  python runner.py <phase> [args]      (set WORKERS=n to change the pool size)
"""
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

warnings.filterwarnings("ignore")

from problems import PROBLEMS, PROBLEM_MAP                       # noqa: E402
from algorithms import ALGORITHMS, CORE, SPACES, DEFAULTS, run_one   # noqa: E402

BUDGET = 15000
N_RUNS_CORE = 51          # in-house algorithms and baselines
N_RUNS_MEALPY = 31        # third-party algorithms (about 40x slower per run)
N_RUNS_SENS = 31          # constraint-scheme sensitivity
N_RUNS_BUDGET = 15        # budget sensitivity
CKPT_RUNS = 5             # checkpoint a sweep job every this many completed runs
CKPT_CANDS = 8            # checkpoint a tuning race every this many candidates
TRAIN_PROBLEMS = ["PressureVessel", "CantileverBeam", "G04", "G08"]
HELDOUT = [p.name for p in PROBLEMS if p.name not in TRAIN_PROBLEMS]
TUNE_SEEDS = 8
PIDX = {p.name: j for j, p in enumerate(PROBLEMS)}
WORKERS = int(os.environ.get("WORKERS", 7))

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)


def n_runs_for(alg, default=N_RUNS_CORE):
    return min(default, N_RUNS_MEALPY) if alg not in CORE else default


# ------------------------------------------------------------------ tuning
def sample_config(alg, rng):
    cfg = {}
    for k, (kind, lo, hi) in SPACES[alg].items():
        cfg[k] = int(rng.integers(lo, hi + 1)) if kind == "int" else float(rng.uniform(lo, hi))
    return cfg


def score(res, problem):
    """Scale-free tuning score: relative gap to the published best, 10 if infeasible."""
    if not res["feasible"]:
        return 10.0
    return min(10.0, abs(res["best_f"] - problem.known_f) / max(abs(problem.known_f), 1e-8))


def _write_json(fn, obj):
    """Atomic write: a crash mid-write never leaves a truncated checkpoint."""
    with open(fn + ".tmp", "w") as fh:
        json.dump(obj, fh)
    for attempt in range(20):        # Windows: indexers/antivirus can briefly lock the file
        try:
            os.replace(fn + ".tmp", fn)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.5)


def tune_one(alg, n_init, seed):
    """Successive-halving racing for one algorithm; identical protocol for all.

    Resumable: state (surviving candidates, round, per-candidate scores so far) is
    checkpointed every CKPT_CANDS candidates and after every round; a finished
    algorithm is stored in tune<N>/<alg>.json and skipped on restart.  Every
    evaluation seed is a pure function of (round, candidate, run), so a resumed
    race is identical to an uninterrupted one."""
    d = os.path.join(OUT, f"tune{n_init}")
    os.makedirs(d, exist_ok=True)
    done_fn, st_fn = os.path.join(d, f"{alg}.json"), os.path.join(d, f"{alg}.state.json")
    if os.path.exists(done_fn):
        r = json.load(open(done_fn))
        return alg, r["cfg"], r["log"], r["cost"]
    if os.path.exists(st_fn):
        st = json.load(open(st_fn))
        cands, seeds, rnd, log, cost, scores = (st[k] for k in
                                                ("cands", "seeds", "rnd", "log", "cost", "scores"))
    else:
        rng = np.random.default_rng(seed)
        cands = [sample_config(alg, rng) for _ in range(n_init)]
        seeds, rnd, log, cost, scores = 1, 0, [], 0, []

    def save():
        _write_json(st_fn, dict(cands=cands, seeds=seeds, rnd=rnd, log=log, cost=cost, scores=scores))

    while len(cands) > 1:
        for ci in range(len(scores), len(cands)):
            cfg, s = cands[ci], []
            for pn in TRAIN_PROBLEMS:
                p = PROBLEM_MAP[pn]
                for r in range(seeds):
                    res = run_one(alg, p, BUDGET, 777000 + 97 * rnd + 31 * ci + r, params=cfg)
                    s.append(score(res, p))
                    cost += BUDGET
            scores.append(float(np.mean(s)))
            if (ci + 1) % CKPT_CANDS == 0:
                save()
        keep = np.argsort(scores)[:max(1, len(cands) // 2)]
        log.append(dict(alg=alg, n_init=n_init, round=rnd, n_configs=len(cands),
                        seeds=seeds, best_score=float(np.min(scores))))
        cands = [cands[i] for i in keep]
        seeds = min(TUNE_SEEDS, seeds * 2)
        rnd += 1
        scores = []
        save()
    _write_json(done_fn, dict(cfg=cands[0], log=log, cost=cost))
    return alg, cands[0], log, cost


def phase_tune(n_init):
    algs = [a for a in CORE if SPACES[a]]
    best, log, costs = {a: {} for a in CORE}, [], {}
    with ProcessPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(tune_one, a, n_init, 20260801 + n_init) for a in algs]
        for f in as_completed(futs):
            a, cfg, lg, cost = f.result()
            best[a], costs[a] = cfg, cost
            log += lg
            print(f"[tune{n_init}] {a:<7} {cfg}", flush=True)
    json.dump(best, open(f"{OUT}/tuned_params_{n_init}.json", "w"), indent=2)
    json.dump(dict(log=log, evaluations_per_algorithm=costs),
              open(f"{OUT}/tuning_log_{n_init}.json", "w"), indent=1)


# ------------------------------------------------------------------ sweeps
def job(tag, pn, alg, budget, scheme, params, n_runs):
    d = os.path.join(OUT, tag)
    os.makedirs(d, exist_ok=True)
    fn = os.path.join(d, f"{pn}__{alg}.json")
    if os.path.exists(fn):
        return fn, 0.0
    t0 = time.time()
    p = PROBLEM_MAP[pn]
    part = fn + ".part"                  # partial checkpoint: completed runs so far
    rows = json.load(open(part)) if os.path.exists(part) else []
    for r in range(len(rows), n_runs):   # seeds depend only on (problem, run) -> resume is exact
        seed = 10000 * PIDX[pn] + r
        res = run_one(alg, p, budget, seed, params=params, scheme=scheme)
        rows.append(dict(problem=pn, algorithm=alg, run=r, seed=seed,
                         feasible=res["feasible"],
                         best_f=res["best_f"] if res["feasible"] else None,
                         violation=res["best_viol"], fes=res["fes"],
                         trace=[None if not np.isfinite(v) else float(v) for v in res["trace"]]))
        if (r + 1) % CKPT_RUNS == 0 and r + 1 < n_runs:
            _write_json(part, rows)
    _write_json(fn, rows)
    if os.path.exists(part):
        os.remove(part)
    return fn, time.time() - t0


def run_jobs(jobs, label):
    t0, n = time.time(), len(jobs)
    # slowest jobs first so the pool drains evenly
    jobs = sorted(jobs, key=lambda j: j[2] not in CORE, reverse=True)
    failed = []
    with ProcessPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(job, *j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
            except Exception as e:           # one bad job must not discard the others
                failed.append((futs[f][1], futs[f][2], repr(e)))
                print(f"[{label}] FAILED {futs[f][1]}/{futs[f][2]}: {e!r}", flush=True)
            if i % 10 == 0 or i == n:
                print(f"[{label}] {i}/{n}  ({time.time()-t0:.0f}s)", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} job(s) failed; finished jobs are kept, rerun to retry")


def tuned_params(n_init):
    return json.load(open(f"{OUT}/tuned_params_{n_init}.json"))


PHASE_MARKERS = ["main", "tune_32", "tune_128", "tuned_32", "tuned_128", "cons", "budget"]
_LOCK = None


def acquire_lock():
    """Exclusive, blocking, crash-safe lock: the OS drops it when the process dies,
    so two runners can never work on the same results folder at once."""
    global _LOCK
    _LOCK = open(os.path.join(OUT, "runner.lock"), "a+")
    announced = False
    while True:
        try:
            if os.name == "nt":
                import msvcrt
                _LOCK.seek(0)
                msvcrt.locking(_LOCK.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError:
            if not announced:
                print("another runner holds the lock; waiting", flush=True)
                announced = True
            time.sleep(30)


def is_done(m):
    """Phase marker present; the main sweep also counts as done when every job file exists."""
    if os.path.exists(os.path.join(OUT, "done_" + m)):
        return True
    if m == "main":
        d = os.path.join(OUT, "main")
        return os.path.isdir(d) and sum(f.endswith(".json") for f in os.listdir(d)) >= len(PROBLEMS) * len(ALGORITHMS)
    return False


def marker(phase, args):
    return os.path.join(OUT, "done_" + "_".join([phase] + args))


def main():
    phase = sys.argv[1]
    names = [p.name for p in PROBLEMS]
    if phase == "status":
        for m in PHASE_MARKERS:
            print(f"{m:<10}", "done" if is_done(m) else "pending")
        for d in sorted(os.listdir(OUT)):
            fp = os.path.join(OUT, d)
            if os.path.isdir(fp):
                fs = os.listdir(fp)
                print(f"  {d}: {sum(f.endswith('.json') and not f.endswith('.state.json') for f in fs)} files, "
                      f"{sum(f.endswith('.part') or f.endswith('.state.json') for f in fs)} partial")
        return
    acquire_lock()
    mk = marker(phase, sys.argv[2:3] if phase in ("tune", "tuned") else [])
    if os.path.exists(mk):
        print("already complete:", phase, flush=True)
        print("PHASE COMPLETE:", phase, flush=True)
        return
    if phase == "tune":
        phase_tune(int(sys.argv[2]))
    elif phase == "main":
        jobs = [("main", pn, a, BUDGET, "deb", DEFAULTS.get(a), n_runs_for(a))
                for pn in names for a in ALGORITHMS]
        run_jobs(jobs, "main")
    elif phase == "tuned":
        n_init = int(sys.argv[2])
        tp = tuned_params(n_init)
        jobs = [(f"tuned{n_init}", pn, a, BUDGET, "deb", tp[a], N_RUNS_CORE)
                for pn in names for a in CORE]
        run_jobs(jobs, f"tuned{n_init}")
    elif phase == "cons":
        jobs = [(f"cons_{sc}", pn, a, BUDGET, sc, DEFAULTS.get(a), N_RUNS_SENS)
                for sc in ("static", "eps") for pn in names for a in CORE]
        run_jobs(jobs, "cons")
    elif phase == "budget":
        budgets = [int(b) for b in (sys.argv[2].split(",") if len(sys.argv) > 2
                                    else ["5000", "50000", "150000"])]
        jobs = [(f"budget{b}", pn, a, b, "deb", DEFAULTS.get(a), N_RUNS_BUDGET)
                for b in budgets for pn in names for a in CORE]
        run_jobs(jobs, "budget")
    else:
        raise SystemExit("unknown phase")
    if phase == "budget" and len(sys.argv) > 2:      # custom budget list: not the full phase
        print("PHASE COMPLETE:", phase, flush=True)
        return
    open(mk, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    if all(is_done(m) for m in PHASE_MARKERS):
        open(os.path.join(OUT, "ALL_PHASES_COMPLETE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("PHASE COMPLETE:", phase, flush=True)


if __name__ == "__main__":
    main()
