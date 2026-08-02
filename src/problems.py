"""
Constrained engineering design problem suite.

Each problem exposes:
    name, dim, lb, ub, n_con
    evaluate(X) -> (f, G)   X: (n, dim);  f: (n,);  G: (n, n_con) in g(x) <= 0 form
    repair(X)   -> X        enforces integer / discrete-step variables
    known_x, known_f        published best-known solution, used for verification

All formulations follow the standard versions used in the constrained
metaheuristic literature. Every problem is verified against its published
optimum in verify_problems.py before any experiment is run.
"""
import numpy as np

PI = np.pi


class Problem:
    name = "base"
    dim = 0
    n_con = 0
    lb = None
    ub = None
    known_x = None
    known_f = None
    integer_idx = ()
    step_idx = {}   # {index: step_size}

    def repair(self, X):
        X = np.clip(X, self.lb, self.ub)
        if self.integer_idx:
            idx = list(self.integer_idx)
            X[:, idx] = np.round(X[:, idx])
        for j, s in self.step_idx.items():
            X[:, j] = np.round(X[:, j] / s) * s
        return np.clip(X, self.lb, self.ub)

    def evaluate(self, X):
        raise NotImplementedError


class PressureVessel(Problem):
    """Kannan & Kramer (1994). Ts, Th are multiples of 0.0625 in."""
    name = "PressureVessel"
    dim = 4
    n_con = 4
    lb = np.array([0.0625, 0.0625, 10.0, 10.0])
    ub = np.array([6.1875, 6.1875, 200.0, 200.0])
    step_idx = {0: 0.0625, 1: 0.0625}
    known_x = np.array([0.8125, 0.4375, 42.0984456, 176.6365958])
    known_f = 6059.714335

    def evaluate(self, X):
        Ts, Th, R, L = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        f = (0.6224 * Ts * R * L + 1.7781 * Th * R ** 2
             + 3.1661 * Ts ** 2 * L + 19.84 * Ts ** 2 * R)
        g1 = -Ts + 0.0193 * R
        g2 = -Th + 0.00954 * R
        g3 = -PI * R ** 2 * L - (4.0 / 3.0) * PI * R ** 3 + 1296000.0
        g4 = L - 240.0
        return f, np.column_stack([g1, g2, g3, g4])


class TensionSpring(Problem):
    """Arora (1989) tension/compression spring."""
    name = "TensionSpring"
    dim = 3
    n_con = 4
    lb = np.array([0.05, 0.25, 2.0])
    ub = np.array([2.0, 1.3, 15.0])
    known_x = np.array([0.051689, 0.356718, 11.288966])
    known_f = 0.012665

    def evaluate(self, X):
        d, D, N = X[:, 0], X[:, 1], X[:, 2]
        f = (N + 2.0) * D * d ** 2
        g1 = 1.0 - (D ** 3 * N) / (71785.0 * d ** 4)
        g2 = ((4.0 * D ** 2 - d * D) / (12566.0 * (D * d ** 3 - d ** 4))
              + 1.0 / (5108.0 * d ** 2) - 1.0)
        g3 = 1.0 - 140.45 * d / (D ** 2 * N)
        g4 = (d + D) / 1.5 - 1.0
        return f, np.column_stack([g1, g2, g3, g4])


class WeldedBeam(Problem):
    """Coello (2000) welded beam design."""
    name = "WeldedBeam"
    dim = 4
    n_con = 7
    lb = np.array([0.125, 0.1, 0.1, 0.125])
    ub = np.array([2.0, 10.0, 10.0, 2.0])
    known_x = np.array([0.205730, 3.470489, 9.036624, 0.205730])
    known_f = 1.724852

    def evaluate(self, X):
        h, l, t, b = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        P, L, E, G = 6000.0, 14.0, 30e6, 12e6
        tmax, smax, dmax = 13600.0, 30000.0, 0.25

        f = 1.10471 * h ** 2 * l + 0.04811 * t * b * (14.0 + l)

        M = P * (L + l / 2.0)
        R = np.sqrt(l ** 2 / 4.0 + ((h + t) / 2.0) ** 2)
        J = 2.0 * (np.sqrt(2.0) * h * l * (l ** 2 / 12.0 + ((h + t) / 2.0) ** 2))
        tau1 = P / (np.sqrt(2.0) * h * l)
        tau2 = M * R / J
        tau = np.sqrt(tau1 ** 2 + 2.0 * tau1 * tau2 * (l / (2.0 * R)) + tau2 ** 2)
        sigma = 6.0 * P * L / (b * t ** 2)
        delta = 4.0 * P * L ** 3 / (E * b * t ** 3)
        Pc = ((4.013 * E * np.sqrt(t ** 2 * b ** 6 / 36.0) / L ** 2)
              * (1.0 - t / (2.0 * L) * np.sqrt(E / (4.0 * G))))

        g1 = tau - tmax
        g2 = sigma - smax
        g3 = h - b
        g4 = 0.10471 * h ** 2 + 0.04811 * t * b * (14.0 + l) - 5.0
        g5 = 0.125 - h
        g6 = delta - dmax
        g7 = P - Pc
        return f, np.column_stack([g1, g2, g3, g4, g5, g6, g7])


class SpeedReducer(Problem):
    """Golinski speed reducer, 7 variables, 11 constraints. x3 integer."""
    name = "SpeedReducer"
    dim = 7
    n_con = 11
    lb = np.array([2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0])
    ub = np.array([3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5])
    integer_idx = (2,)
    known_x = np.array([3.5, 0.7, 17.0, 7.3, 7.7153199, 3.3502147, 5.2866545])
    known_f = 2994.4244658

    def evaluate(self, X):
        x1, x2, x3, x4, x5, x6, x7 = [X[:, i] for i in range(7)]
        f = (0.7854 * x1 * x2 ** 2 * (3.3333 * x3 ** 2 + 14.9334 * x3 - 43.0934)
             - 1.508 * x1 * (x6 ** 2 + x7 ** 2)
             + 7.4777 * (x6 ** 3 + x7 ** 3)
             + 0.7854 * (x4 * x6 ** 2 + x5 * x7 ** 2))
        g1 = 27.0 / (x1 * x2 ** 2 * x3) - 1.0
        g2 = 397.5 / (x1 * x2 ** 2 * x3 ** 2) - 1.0
        g3 = 1.93 * x4 ** 3 / (x2 * x3 * x6 ** 4) - 1.0
        g4 = 1.93 * x5 ** 3 / (x2 * x3 * x7 ** 4) - 1.0
        g5 = np.sqrt((745.0 * x4 / (x2 * x3)) ** 2 + 16.9e6) / (110.0 * x6 ** 3) - 1.0
        g6 = np.sqrt((745.0 * x5 / (x2 * x3)) ** 2 + 157.5e6) / (85.0 * x7 ** 3) - 1.0
        g7 = x2 * x3 / 40.0 - 1.0
        g8 = 5.0 * x2 / x1 - 1.0
        g9 = x1 / (12.0 * x2) - 1.0
        g10 = (1.5 * x6 + 1.9) / x4 - 1.0
        g11 = (1.1 * x7 + 1.9) / x5 - 1.0
        return f, np.column_stack([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11])


class ThreeBarTruss(Problem):
    name = "ThreeBarTruss"
    dim = 2
    n_con = 3
    lb = np.array([1e-8, 1e-8])
    ub = np.array([1.0, 1.0])
    known_x = np.array([0.78867513, 0.40824829])
    known_f = 263.8958434

    def evaluate(self, X):
        x1, x2 = X[:, 0], X[:, 1]
        L, P, sigma = 100.0, 2.0, 2.0
        f = (2.0 * np.sqrt(2.0) * x1 + x2) * L
        den = np.sqrt(2.0) * x1 ** 2 + 2.0 * x1 * x2
        g1 = (np.sqrt(2.0) * x1 + x2) / den * P - sigma
        g2 = x2 / den * P - sigma
        g3 = 1.0 / (np.sqrt(2.0) * x2 + x1) * P - sigma
        return f, np.column_stack([g1, g2, g3])


class CantileverBeam(Problem):
    name = "CantileverBeam"
    dim = 5
    n_con = 1
    lb = np.full(5, 0.01)
    ub = np.full(5, 100.0)
    known_x = np.array([6.0160, 5.3092, 4.4943, 3.5015, 2.15266])
    known_f = 1.3399576

    def evaluate(self, X):
        f = 0.0624 * X.sum(axis=1)
        g1 = (61.0 / X[:, 0] ** 3 + 37.0 / X[:, 1] ** 3 + 19.0 / X[:, 2] ** 3
              + 7.0 / X[:, 3] ** 3 + 1.0 / X[:, 4] ** 3) - 1.0
        return f, g1.reshape(-1, 1)


class GearTrain(Problem):
    """Sandgren (1990). All four variables integer teeth counts."""
    name = "GearTrain"
    dim = 4
    n_con = 1
    lb = np.full(4, 12.0)
    ub = np.full(4, 60.0)
    integer_idx = (0, 1, 2, 3)
    known_x = np.array([49.0, 19.0, 16.0, 43.0])
    known_f = 2.7009e-12

    def evaluate(self, X):
        a, b, c, d = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        f = (1.0 / 6.931 - (b * c) / (a * d)) ** 2
        g1 = np.full(X.shape[0], -1.0)   # unconstrained apart from bounds
        return f, g1.reshape(-1, 1)


class IBeamDeflection(Problem):
    """Vertical deflection of an I-beam."""
    name = "IBeamDeflection"
    dim = 4
    n_con = 2
    lb = np.array([10.0, 10.0, 0.9, 0.9])
    ub = np.array([50.0, 80.0, 5.0, 5.0])
    known_x = np.array([50.0, 80.0, 0.9, 2.321675])
    known_f = 0.0130741

    def evaluate(self, X):
        b, h, tw, tf = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        term = (tw * (h - 2.0 * tf) ** 3) / 12.0 + (b * tf ** 3) / 6.0 \
            + 2.0 * b * tf * ((h - tf) / 2.0) ** 2
        f = 5000.0 / term
        g1 = 2.0 * b * tf + tw * (h - 2.0 * tf) - 300.0
        num = 18.0 * h * 1e4
        den = tw * (h - 2.0 * tf) ** 3 + 2.0 * b * tf * (4.0 * tf ** 2 + 3.0 * h * (h - 2.0 * tf))
        num2 = 15.0 * b * 1e3
        den2 = (h - 2.0 * tf) * tw ** 3 + 2.0 * tf * b ** 3
        g2 = num / den + num2 / den2 - 6.0
        return f, np.column_stack([g1, g2])


class DiscBrake(Problem):
    """Multiple disc clutch brake. Discrete variables."""
    name = "DiscBrake"
    dim = 5
    n_con = 8
    lb = np.array([60.0, 90.0, 1.0, 600.0, 2.0])
    ub = np.array([80.0, 110.0, 3.0, 1000.0, 9.0])
    step_idx = {0: 1.0, 1: 1.0, 2: 0.5, 3: 1.0}
    integer_idx = (4,)
    known_x = np.array([70.0, 90.0, 1.0, 810.0, 3.0])
    known_f = 0.23524245

    def evaluate(self, X):
        ri, ro, t, F, Z = [X[:, i] for i in range(5)]
        Mf, Ms, Iz, n, Vsrmax = 3.0, 40.0, 55.0, 250.0, 10.0
        s, delta, Lmax, mu, pmax = 1.5, 0.5, 30.0, 0.6, 1.0
        DR = 20.0   # minimum radial width ri to ro (mm)
        rho = 0.0000078
        f = PI * (ro ** 2 - ri ** 2) * t * Z * rho
        Rsr = 2.0 / 3.0 * (ro ** 3 - ri ** 3) / (ro ** 2 - ri ** 2)
        A = PI * (ro ** 2 - ri ** 2)
        prz = F / A
        Vsr = PI * Rsr * n / 30000.0
        Mh = (2.0 / 3.0) * mu * F * Z * (ro ** 3 - ri ** 3) / ((ro ** 2 - ri ** 2) * 1000.0)
        Tmax = Iz * PI * n / (30.0 * (Mh + Mf))
        g1 = -(ro - ri - DR)
        g2 = -(Lmax - (Z + 1.0) * (t + delta))
        g3 = prz - pmax
        g4 = prz * Vsr - pmax * Vsrmax
        g5 = Vsr - Vsrmax
        g6 = Ms - Mh
        g7 = F - 1000.0
        g8 = Tmax - 15.0
        return f, np.column_stack([g1, g2, g3, g4, g5, g6, g7, g8])


PROBLEMS = [
    PressureVessel(), TensionSpring(), WeldedBeam(), SpeedReducer(),
    ThreeBarTruss(), CantileverBeam(), GearTrain(), IBeamDeflection(),
    DiscBrake(),
]

PROBLEM_MAP = {p.name: p for p in PROBLEMS}
