"""
Algorithm implementations.

Test group (metaphor-based, 2014-2021), transcribed from the pseudocode in the
original publications:
    GWO  Mirjalili, Mirjalili & Lewis (2014)
    WOA  Mirjalili & Lewis (2016)
    SCA  Mirjalili (2016)
    SSA  Mirjalili et al. (2017)
    HHO  Heidari et al. (2019)
    AOA  Abualigah et al. (2021)

Baseline group:
    DE      Storn & Price (1997), DE/rand/1/bin
    PSO     Clerc & Kennedy (2002), constriction factor
    LSHADE  Tanabe & Fukunaga (2014), success-history DE with linear
            population size reduction
    CMAES   Hansen & Ostermeier (2001), via the `cma` package
    RS      uniform random search (performance floor)

Every algorithm consumes function evaluations through the same Evaluator
object, so the budget accounting is identical by construction.
"""
import math
import numpy as np
import cma


# --------------------------------------------------------------------------
# Constraint handling + evaluation budget accounting
# --------------------------------------------------------------------------
class Evaluator:
    """Scalarises (f, violation) into a single fitness and meters the budget.

    scheme:
      'deb'     Deb (2000) feasibility rules, expressed as a scalar transform:
                feasible -> f ; infeasible -> f_ref + violation, where f_ref is
                the worst objective value among feasible points seen so far.
      'static'  f + C * violation
      'eps'     as 'deb' but violations below a decaying threshold eps count
                as feasible (Takahama & Sakai, 2006)
    """

    def __init__(self, problem, budget, scheme="deb", C=1e5, eps0=1e-2, n_trace=100):
        self.p = problem
        self.budget = budget
        self.scheme = scheme
        self.C = C
        self.eps0 = eps0
        self.fes = 0
        self.f_ref = 0.0
        self.seen_feasible = False
        self.best_f = np.inf
        self.best_viol = np.inf
        self.best_x = None
        self.trace_at = np.linspace(budget / n_trace, budget, n_trace).astype(int)
        self.trace = []
        self._ti = 0

    @property
    def exhausted(self):
        return self.fes >= self.budget

    def remaining(self):
        return max(0, self.budget - self.fes)

    def __call__(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        r = self.remaining()
        if r <= 0:
            return np.full(X.shape[0], np.inf)
        if X.shape[0] > r:
            out = np.full(X.shape[0], np.inf)
            out[:r] = self(X[:r])
            return out

        Xr = self.p.repair(X.copy())
        f, G = self.p.evaluate(Xr)
        viol = np.maximum(G, 0.0).sum(axis=1)
        f = np.where(np.isfinite(f), f, 1e30)
        viol = np.where(np.isfinite(viol), viol, 1e30)

        eps = 0.0
        if self.scheme == "eps":
            prog = min(1.0, self.fes / (0.5 * self.budget))
            eps = self.eps0 * (1.0 - prog) ** 3
        feas = viol <= eps

        if self.scheme == "static":
            fit = f + self.C * viol
        else:
            if feas.any():
                cand = f[feas].max()
                self.f_ref = cand if not self.seen_feasible else max(self.f_ref, cand)
                self.seen_feasible = True
            fit = np.where(feas, f, self.f_ref + viol)

        # track the best strictly-feasible point found (violation <= 1e-8)
        strict = viol <= 1e-8
        if strict.any():
            i = int(np.argmin(np.where(strict, f, np.inf)))
            if f[i] < self.best_f:
                self.best_f, self.best_viol, self.best_x = float(f[i]), float(viol[i]), Xr[i].copy()
        elif not np.isfinite(self.best_f):
            i = int(np.argmin(viol))
            if viol[i] < self.best_viol:
                self.best_viol, self.best_x = float(viol[i]), Xr[i].copy()

        self.fes += X.shape[0]
        while self._ti < len(self.trace_at) and self.fes >= self.trace_at[self._ti]:
            self.trace.append(self.best_f)
            self._ti += 1
        return fit

    def finish(self):
        while self._ti < len(self.trace_at):
            self.trace.append(self.best_f)
            self._ti += 1
        return dict(best_f=self.best_f, best_viol=self.best_viol,
                    best_x=None if self.best_x is None else self.best_x.tolist(),
                    feasible=bool(np.isfinite(self.best_f)),
                    trace=list(self.trace), fes=self.fes)


def _init_pop(rng, n, p):
    return rng.uniform(p.lb, p.ub, size=(n, p.dim))


# --------------------------------------------------------------------------
# Test group: metaphor-based metaheuristics
# --------------------------------------------------------------------------
def GWO(ev, rng, pop=30):
    p, lb, ub = ev.p, ev.p.lb, ev.p.ub
    X = _init_pop(rng, pop, p)
    fit = ev(X)
    order = np.argsort(fit)
    A_, B_, D_ = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()
    fA, fB, fD = fit[order[0]], fit[order[1]], fit[order[2]]
    while not ev.exhausted:
        a = 2.0 - 2.0 * ev.fes / ev.budget
        new = np.empty_like(X)
        for k, leader in enumerate((A_, B_, D_)):
            r1 = rng.random((pop, p.dim)); r2 = rng.random((pop, p.dim))
            A = 2 * a * r1 - a
            C = 2 * r2
            Dl = np.abs(C * leader - X)
            new = (leader - A * Dl) if k == 0 else new + (leader - A * Dl)
        X = np.clip(new / 3.0, lb, ub)
        fit = ev(X)
        for i in range(pop):
            if fit[i] < fA:
                fD, D_ = fB, B_.copy(); fB, B_ = fA, A_.copy(); fA, A_ = fit[i], X[i].copy()
            elif fit[i] < fB:
                fD, D_ = fB, B_.copy(); fB, B_ = fit[i], X[i].copy()
            elif fit[i] < fD:
                fD, D_ = fit[i], X[i].copy()
    return ev.finish()


def WOA(ev, rng, pop=30, b=1.0):
    p, lb, ub = ev.p, ev.p.lb, ev.p.ub
    X = _init_pop(rng, pop, p)
    fit = ev(X)
    gi = int(np.argmin(fit)); gx, gf = X[gi].copy(), fit[gi]
    while not ev.exhausted:
        a = 2.0 - 2.0 * ev.fes / ev.budget
        a2 = -1.0 - ev.fes / ev.budget
        new = np.empty_like(X)
        for i in range(pop):
            r1, r2 = rng.random(p.dim), rng.random(p.dim)
            A = 2 * a * r1 - a
            C = 2 * r2
            if rng.random() < 0.5:
                if np.abs(A).mean() < 1:
                    new[i] = gx - A * np.abs(C * gx - X[i])
                else:
                    j = rng.integers(pop)
                    new[i] = X[j] - A * np.abs(C * X[j] - X[i])
            else:
                l = (a2 - 1) * rng.random(p.dim) + 1
                new[i] = np.abs(gx - X[i]) * np.exp(b * l) * np.cos(2 * np.pi * l) + gx
        X = np.clip(new, lb, ub)
        fit = ev(X)
        i = int(np.argmin(fit))
        if fit[i] < gf:
            gf, gx = fit[i], X[i].copy()
    return ev.finish()


def SCA(ev, rng, pop=30, a=2.0):
    p, lb, ub = ev.p, ev.p.lb, ev.p.ub
    X = _init_pop(rng, pop, p)
    fit = ev(X)
    gi = int(np.argmin(fit)); gx, gf = X[gi].copy(), fit[gi]
    while not ev.exhausted:
        r1 = a - a * ev.fes / ev.budget
        r2 = 2 * np.pi * rng.random((pop, p.dim))
        r3 = 2 * rng.random((pop, p.dim))
        r4 = rng.random((pop, p.dim))
        d = np.abs(r3 * gx - X)
        X = np.where(r4 < 0.5, X + r1 * np.sin(r2) * d, X + r1 * np.cos(r2) * d)
        X = np.clip(X, lb, ub)
        fit = ev(X)
        i = int(np.argmin(fit))
        if fit[i] < gf:
            gf, gx = fit[i], X[i].copy()
    return ev.finish()


def SSA(ev, rng, pop=30):
    p, lb, ub = ev.p, ev.p.lb, ev.p.ub
    X = _init_pop(rng, pop, p)
    fit = ev(X)
    gi = int(np.argmin(fit)); gx, gf = X[gi].copy(), fit[gi]
    while not ev.exhausted:
        t = ev.fes / ev.budget
        c1 = 2.0 * np.exp(-(4.0 * t) ** 2)
        new = X.copy()
        half = max(1, pop // 2)
        c2 = rng.random((half, p.dim)); c3 = rng.random((half, p.dim))
        new[:half] = np.where(c3 < 0.5,
                              gx + c1 * ((ub - lb) * c2 + lb),
                              gx - c1 * ((ub - lb) * c2 + lb))
        new[half:] = 0.5 * (X[half:] + X[half - 1:pop - 1])
        X = np.clip(new, lb, ub)
        fit = ev(X)
        i = int(np.argmin(fit))
        if fit[i] < gf:
            gf, gx = fit[i], X[i].copy()
    return ev.finish()


def HHO(ev, rng, pop=30, beta=1.5):
    p, lb, ub = ev.p, ev.p.lb, ev.p.ub
    X = _init_pop(rng, pop, p)
    fit = ev(X)
    gi = int(np.argmin(fit)); gx, gf = X[gi].copy(), fit[gi]
    sigma = ((math.gamma(1 + beta) * np.sin(np.pi * beta / 2)) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)

    def levy(d):
        u = rng.normal(0, sigma, d); v = rng.normal(0, 1, d)
        return 0.01 * u / (np.abs(v) ** (1 / beta))

    while not ev.exhausted:
        E1 = 2 * (1 - ev.fes / ev.budget)
        Xm = X.mean(axis=0)
        new = X.copy()
        for i in range(pop):
            E0 = 2 * rng.random() - 1
            E = E1 * E0
            if abs(E) >= 1:                       # exploration
                q = rng.random()
                if q >= 0.5:
                    j = rng.integers(pop)
                    new[i] = X[j] - rng.random() * np.abs(X[j] - 2 * rng.random() * X[i])
                else:
                    new[i] = (gx - Xm) - rng.random() * (lb + rng.random() * (ub - lb))
            else:                                  # exploitation
                r = rng.random()
                J = 2 * (1 - rng.random())
                if r >= 0.5 and abs(E) >= 0.5:
                    new[i] = (gx - X[i]) - E * np.abs(J * gx - X[i])
                elif r >= 0.5 and abs(E) < 0.5:
                    new[i] = gx - E * np.abs(gx - X[i])
                elif r < 0.5 and abs(E) >= 0.5:
                    Y = gx - E * np.abs(J * gx - X[i])
                    new[i] = Y + rng.random(p.dim) * levy(p.dim)
                else:
                    Y = gx - E * np.abs(J * gx - Xm)
                    new[i] = Y + rng.random(p.dim) * levy(p.dim)
        X = np.clip(new, lb, ub)
        fit = ev(X)
        i = int(np.argmin(fit))
        if fit[i] < gf:
            gf, gx = fit[i], X[i].copy()
    return ev.finish()


def AOA(ev, rng, pop=30, alpha=5.0, mu=0.5):
    p, lb, ub = ev.p, ev.p.lb, ev.p.ub
    X = _init_pop(rng, pop, p)
    fit = ev(X)
    gi = int(np.argmin(fit)); gx, gf = X[gi].copy(), fit[gi]
    eps = 1e-12
    while not ev.exhausted:
        t = ev.fes / ev.budget
        MOA = 0.2 + t * (0.9 - 0.2)
        MOP = 1.0 - t ** (1.0 / alpha)
        r1 = rng.random((pop, p.dim))
        r2 = rng.random((pop, p.dim))
        r3 = rng.random((pop, p.dim))
        span = mu * (ub - lb) + lb
        div = gx / (MOP + eps) * span
        mul = gx * MOP * span
        sub = gx - MOP * span
        add = gx + MOP * span
        explore = np.where(r2 > 0.5, div, mul)
        exploit = np.where(r3 > 0.5, sub, add)
        X = np.where(r1 > MOA, explore, exploit)
        X = np.clip(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), lb, ub)
        fit = ev(X)
        i = int(np.argmin(fit))
        if fit[i] < gf:
            gf, gx = fit[i], X[i].copy()
    return ev.finish()


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
def DE(ev, rng, pop=50, F=0.5, CR=0.9):
    p, lb, ub = ev.p, ev.p.lb, ev.p.ub
    X = _init_pop(rng, pop, p)
    fit = ev(X)
    while not ev.exhausted:
        idx = np.array([rng.choice(pop, 3, replace=False) for _ in range(pop)])
        V = X[idx[:, 0]] + F * (X[idx[:, 1]] - X[idx[:, 2]])
        cross = rng.random((pop, p.dim)) < CR
        jrand = rng.integers(p.dim, size=pop)
        cross[np.arange(pop), jrand] = True
        U = np.clip(np.where(cross, V, X), lb, ub)
        fu = ev(U)
        better = fu < fit
        X[better], fit[better] = U[better], fu[better]
    return ev.finish()


def PSO(ev, rng, pop=40, w=0.729, c1=1.494, c2=1.494):
    p, lb, ub = ev.p, ev.p.lb, ev.p.ub
    X = _init_pop(rng, pop, p)
    span = ub - lb
    V = rng.uniform(-span, span, size=(pop, p.dim)) * 0.1
    fit = ev(X)
    P, fP = X.copy(), fit.copy()
    gi = int(np.argmin(fP)); gx, gf = P[gi].copy(), fP[gi]
    while not ev.exhausted:
        r1 = rng.random((pop, p.dim)); r2 = rng.random((pop, p.dim))
        V = w * V + c1 * r1 * (P - X) + c2 * r2 * (gx - X)
        V = np.clip(V, -0.5 * span, 0.5 * span)
        X = np.clip(X + V, lb, ub)
        fit = ev(X)
        imp = fit < fP
        P[imp], fP[imp] = X[imp], fit[imp]
        i = int(np.argmin(fP))
        if fP[i] < gf:
            gf, gx = fP[i], P[i].copy()
    return ev.finish()


def LSHADE(ev, rng, pop=None, H=6, p_best=0.11):
    p, lb, ub = ev.p, ev.p.lb, ev.p.ub
    Ninit = pop if pop is not None else 18 * p.dim
    Nmin = 4
    X = _init_pop(rng, Ninit, p)
    fit = ev(X)
    MF = np.full(H, 0.5); MCR = np.full(H, 0.5); k = 0
    archive = []
    while not ev.exhausted:
        N = len(X)
        r = rng.integers(0, H, size=N)
        CR = np.clip(rng.normal(MCR[r], 0.1), 0, 1)
        F = np.clip(MF[r] + 0.1 * np.tan(np.pi * (rng.random(N) - 0.5)), 1e-6, 1.0)
        order = np.argsort(fit)
        npb = max(2, int(p_best * N))
        pb = X[order[rng.integers(0, npb, size=N)]]
        pool = np.vstack([X] + ([np.array(archive)] if archive else []))
        r1 = X[rng.integers(0, N, size=N)]
        r2 = pool[rng.integers(0, len(pool), size=N)]
        V = X + F[:, None] * (pb - X) + F[:, None] * (r1 - r2)
        V = np.where(V < lb, (X + lb) / 2, V)
        V = np.where(V > ub, (X + ub) / 2, V)
        cross = rng.random((N, p.dim)) < CR[:, None]
        jr = rng.integers(p.dim, size=N)
        cross[np.arange(N), jr] = True
        U = np.where(cross, V, X)
        fu = ev(U)
        better = fu < fit
        if better.any():
            for i in np.where(better)[0]:
                archive.append(X[i].copy())
            w = np.abs(fit[better] - fu[better]); w = w / (w.sum() + 1e-30)
            MF[k] = np.sum(w * F[better] ** 2) / max(np.sum(w * F[better]), 1e-30)
            MCR[k] = np.sum(w * CR[better])
            k = (k + 1) % H
        X[better], fit[better] = U[better], fu[better]
        while len(archive) > N:
            archive.pop(rng.integers(len(archive)))
        Nnew = int(round(Ninit - (Ninit - Nmin) * ev.fes / ev.budget))
        Nnew = max(Nmin, Nnew)
        if Nnew < len(X):
            keep = np.argsort(fit)[:Nnew]
            X, fit = X[keep], fit[keep]
    return ev.finish()


def CMAES(ev, rng, sigma_frac=0.3, popsize=None):
    p, lb, ub = ev.p, ev.p.lb, ev.p.ub
    span = ub - lb
    x0 = (lb + ub) / 2.0
    opts = {"bounds": [list(lb), list(ub)], "verbose": -9, "verb_log": 0,
            "seed": int(rng.integers(1, 2 ** 31 - 1)), "maxfevals": ev.budget}
    if popsize:
        opts["popsize"] = int(popsize)
    es = cma.CMAEvolutionStrategy(list(x0), float(sigma_frac * span.mean()), opts)
    while not ev.exhausted and not es.stop():
        sols = es.ask()
        vals = ev(np.array(sols))
        es.tell(sols, [float(v) for v in vals])
    return ev.finish()


def RS(ev, rng, batch=100):
    p = ev.p
    while not ev.exhausted:
        n = min(batch, ev.remaining())
        ev(_init_pop(rng, n, p))
    return ev.finish()


TEST_GROUP = {"GWO": GWO, "WOA": WOA, "SCA": SCA, "SSA": SSA, "HHO": HHO, "AOA": AOA}
BASELINES = {"DE": DE, "PSO": PSO, "LSHADE": LSHADE, "CMAES": CMAES, "RS": RS}
ALGORITHMS = {**TEST_GROUP, **BASELINES}

# hyperparameter search spaces for the matched-budget tuner
SPACES = {
    "GWO": {"pop": ("int", 10, 100)},
    "WOA": {"pop": ("int", 10, 100), "b": ("float", 0.5, 2.0)},
    "SCA": {"pop": ("int", 10, 100), "a": ("float", 1.0, 4.0)},
    "SSA": {"pop": ("int", 10, 100)},
    "HHO": {"pop": ("int", 10, 100), "beta": ("float", 1.1, 1.9)},
    "AOA": {"pop": ("int", 10, 100), "alpha": ("float", 3.0, 10.0), "mu": ("float", 0.1, 0.9)},
    "DE": {"pop": ("int", 20, 150), "F": ("float", 0.3, 1.0), "CR": ("float", 0.1, 1.0)},
    "PSO": {"pop": ("int", 20, 150), "w": ("float", 0.4, 0.9),
            "c1": ("float", 0.5, 2.5), "c2": ("float", 0.5, 2.5)},
    "LSHADE": {"pop": ("int", 30, 400), "H": ("int", 3, 12), "p_best": ("float", 0.05, 0.3)},
    "CMAES": {"sigma_frac": ("float", 0.05, 0.5), "popsize": ("int", 8, 60)},
    "RS": {},
}

DEFAULTS = {
    "GWO": {"pop": 30}, "WOA": {"pop": 30, "b": 1.0}, "SCA": {"pop": 30, "a": 2.0},
    "SSA": {"pop": 30}, "HHO": {"pop": 30, "beta": 1.5},
    "AOA": {"pop": 30, "alpha": 5.0, "mu": 0.5},
    "DE": {"pop": 50, "F": 0.5, "CR": 0.9},
    "PSO": {"pop": 40, "w": 0.729, "c1": 1.494, "c2": 1.494},
    "LSHADE": {"H": 6, "p_best": 0.11},
    "CMAES": {"sigma_frac": 0.3}, "RS": {},
}


def run_one(alg, problem, budget, seed, params=None, scheme="deb"):
    rng = np.random.default_rng(seed)
    ev = Evaluator(problem, budget, scheme=scheme)
    kw = dict(DEFAULTS.get(alg, {}))
    if params:
        kw.update(params)
    return ALGORITHMS[alg](ev, rng, **kw)
