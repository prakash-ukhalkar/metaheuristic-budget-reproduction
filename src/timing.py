"""Internal computational overhead per 15,000-evaluation run (single process, idle machine)."""
import json, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from problems import PROBLEM_MAP
from algorithms import ALGORITHMS, DEFAULTS, run_one

PROBS = ["WeldedBeam", "G07", "G01"]
out = {}
for a in ALGORITHMS:
    ts = []
    for pn in PROBS:
        for r in range(3):
            t0 = time.perf_counter()
            run_one(a, PROBLEM_MAP[pn], 15000, 123 + r, params=DEFAULTS.get(a))
            ts.append(time.perf_counter() - t0)
    out[a] = float(np.mean(ts))
    print(f"{a:<8}{out[a]:.3f}s", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "timing.json"), "w"), indent=1)
