"""Verify each problem implementation against its published best-known solution.

A problem passes only if the published solution is (near-)feasible and the
objective value we compute matches the published value to within 0.5%.
Problems that fail are excluded from the study.
"""
import numpy as np
from problems import PROBLEMS

FEAS_TOL = 1e-4
REL_TOL = 5e-3

rows = []
for p in PROBLEMS:
    x = p.known_x.reshape(1, -1).astype(float)
    f, G = p.evaluate(x)
    fv = float(f[0])
    viol = float(np.maximum(G[0], 0.0).sum())
    rel = abs(fv - p.known_f) / max(abs(p.known_f), 1e-12)
    ok_f = rel <= REL_TOL
    ok_g = viol <= FEAS_TOL
    rows.append((p.name, p.dim, p.n_con, p.known_f, fv, rel, viol, ok_f and ok_g))

print(f"{'problem':<18}{'D':>3}{'m':>4}{'published f':>16}{'computed f':>16}"
      f"{'rel.err':>12}{'violation':>14}  status")
print("-" * 96)
for name, d, m, kf, fv, rel, viol, ok in rows:
    print(f"{name:<18}{d:>3}{m:>4}{kf:>16.7g}{fv:>16.7g}{rel:>12.3e}{viol:>14.3e}"
          f"  {'PASS' if ok else 'FAIL'}")

n_pass = sum(r[-1] for r in rows)
print("-" * 96)
print(f"{n_pass}/{len(rows)} problems verified")
