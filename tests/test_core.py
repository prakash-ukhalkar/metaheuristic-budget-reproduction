"""Automated checks for the experimental machinery.  Run:  python -m pytest tests -q
(or  python tests/test_core.py  when pytest is not installed).

These tests guard the properties the paper's claims rest on:
  - every algorithm, in-house or third-party, receives exactly the same evaluation budget;
  - runs are reproducible from the seed and independent of the process they run in;
  - the constraint-handling schemes behave as defined;
  - every problem reproduces its published optimum, feasibly;
  - the statistical helpers return known values.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from problems import PROBLEMS, PROBLEM_MAP                              # noqa: E402
from algorithms import ALGORITHMS, Evaluator, run_one                   # noqa: E402
import analysis as A                                                    # noqa: E402

SMALL = 600           # small budget keeps the suite fast


def test_every_algorithm_uses_exactly_the_budget():
    """The headline fairness claim: no algorithm gets more or fewer evaluations."""
    p = PROBLEM_MAP["WeldedBeam"]
    for name in ALGORITHMS:
        res = run_one(name, p, SMALL, seed=1)
        assert res["fes"] == SMALL, f"{name} used {res['fes']} evaluations, expected {SMALL}"
        assert len(res["trace"]) == 100, f"{name} trace has {len(res['trace'])} points"


def test_evaluator_returns_inf_past_budget():
    p = PROBLEM_MAP["TensionSpring"]
    ev = Evaluator(p, budget=10)
    out = ev(np.tile((p.lb + p.ub) / 2, (15, 1)))
    assert ev.fes == 10
    assert np.isfinite(out[:10]).all() and np.isinf(out[10:]).all()
    assert np.isinf(ev(np.atleast_2d((p.lb + p.ub) / 2))).all()       # nothing evaluated once exhausted
    assert ev.fes == 10


def test_runs_are_reproducible_from_seed():
    p = PROBLEM_MAP["G04"]
    for name in ("GWO", "DE", "CMAES", "RS", "mp_GWO"):
        a, b = run_one(name, p, SMALL, seed=7), run_one(name, p, SMALL, seed=7)
        assert a["best_f"] == b["best_f"] and a["trace"] == b["trace"], name


def test_best_f_is_never_worse_over_the_trace():
    p = PROBLEM_MAP["PressureVessel"]
    res = run_one("DE", p, 3000, seed=3)
    t = [v for v in res["trace"] if v is not None and np.isfinite(v)]
    assert all(t[i + 1] <= t[i] + 1e-12 for i in range(len(t) - 1)), "best-so-far must be non-increasing"


def test_constraint_schemes_define_feasibility_as_documented():
    p = PROBLEM_MAP["G06"]                                         # two inequality constraints
    infeasible = np.array([[14.0, 1.0]])                                # violates the first constraint
    best = run_one("DE", p, 3000, seed=1)                               # published optimum is rounded; use a verified feasible point
    assert best["feasible"]
    feasible = np.atleast_2d(best["best_x"])
    for scheme in ("deb", "static", "eps"):
        ev = Evaluator(p, 100, scheme=scheme)
        ev(feasible)
        assert np.isfinite(ev.best_f) and ev.best_viol <= 1e-8, scheme
    ev = Evaluator(p, 100, scheme="deb")
    f_ok = ev(feasible)[0]
    f_bad = ev(infeasible)[0]
    assert f_bad >= f_ok, "under Deb's rules an infeasible point never beats a feasible one"
    # static penalty: fitness grows with violation
    ev = Evaluator(p, 100, scheme="static")
    _, G = p.evaluate(p.repair(infeasible.copy()))
    assert ev(infeasible)[0] > p.evaluate(p.repair(infeasible.copy()))[0][0]


def test_every_problem_reproduces_its_published_optimum():
    for p in PROBLEMS:
        x = p.repair(np.atleast_2d(p.known_x).astype(float))
        f, G = p.evaluate(x)
        viol = np.maximum(G, 0).sum()
        tol = 5e-3 if p.name == "G03" else 1e-3          # G03 equality tolerance 1e-4 is built into the constraint
        assert viol <= tol, f"{p.name}: published point infeasible (violation {viol})"
        rel = abs(f[0] - p.known_f) / max(abs(p.known_f), 1e-8)
        assert rel <= 1e-3 or abs(f[0] - p.known_f) <= 1e-9, f"{p.name}: f={f[0]} vs published {p.known_f}"


def test_clutch_brake_uses_the_Zplus1_mass_form():
    """Regression test for the v1 transcription error (mass used Z instead of Z+1)."""
    p = PROBLEM_MAP["DiscBrake"]
    f, G = p.evaluate(np.array([[70.0, 90.0, 1.0, 1000.0, 2.0]]))
    assert abs(f[0] - 0.2352424) < 1e-6


def test_statistics_helpers_against_known_values():
    # Holm: textbook example
    p = np.array([0.01, 0.04, 0.03])
    assert np.allclose(A.holm(p), [0.03, 0.06, 0.06])
    # Nemenyi CD for k=4, n large matches Demsar's q_0.05 = 2.569
    cd = A.nemenyi_cd(4, 1)
    assert abs(cd - 2.569 * np.sqrt(4 * 5 / 6)) < 0.02
    # Friedman: perfectly consistent ranking gives mean ranks 1..k
    import pandas as pd
    M = pd.DataFrame(np.tile(np.arange(1.0, 5.0), (6, 1)), columns=list("abcd"))
    fr = A.friedman(M)
    assert abs(fr["chi2"] - 18.0) < 1e-9                       # 12n/(k(k+1)) * sum(R^2) - 3n(k+1) = 18 for n=6, k=4
    assert fr["ranks"] == {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}


def test_normalised_gap_rule():
    """0 at the published optimum; 1 at the random-search median; infeasible gets the penalty."""
    import pandas as pd
    rows = []
    for r in range(10):
        rows.append(dict(problem="G06", algorithm="RS", run=r, feasible=True, best_f=-3000.0))
    df = pd.DataFrame(rows)
    ref = A.Ref(df)
    probe = pd.DataFrame([dict(problem="G06", best_f=A.FSTAR["G06"], feasible=True),
                          dict(problem="G06", best_f=-3000.0, feasible=True),
                          dict(problem="G06", best_f=np.nan, feasible=False)])
    g = ref.ngap(probe)
    assert abs(g[0]) < 1e-9 and abs(g[1] - 1.0) < 1e-9 and g[2] == A.PENALTY


def test_fallback_scale_when_random_search_is_infeasible():
    import pandas as pd
    df = pd.DataFrame([dict(problem="G03", algorithm="RS", run=r, feasible=False, best_f=np.nan) for r in range(10)])
    ref = A.Ref(df)
    assert "G03" in ref.fallback
    probe = pd.DataFrame([dict(problem="G03", best_f=A.FSTAR["G03"] + abs(A.FSTAR["G03"]), feasible=True)])
    assert abs(ref.ngap(probe)[0] - 1.0) < 1e-9                  # scale is |f*|


if __name__ == "__main__":
    n = 0
    for k, v in sorted(globals().items()):
        if k.startswith("test_") and callable(v):
            v()
            n += 1
            print("ok", k)
    print(n, "tests passed")
