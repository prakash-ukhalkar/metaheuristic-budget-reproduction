"""CEC 2006 constrained benchmark (Liang et al. 2006), subset used in v2.

Added to widen dimension (up to 20), constraint type (nonlinear, linear,
equality, disjoint feasible region) and objective landscape beyond the eight
classic mechanical-design problems.  Equality constraints use |h| - 1e-4 <= 0.
"""
import numpy as np
from problems import Problem

PI = np.pi


class G01(Problem):
    name, dim, n_con = "G01", 13, 9
    lb = np.zeros(13)
    ub = np.array([1.0] * 9 + [100.0] * 3 + [1.0])
    known_x = np.array([1.0] * 9 + [3.0, 3.0, 3.0, 1.0])
    known_f = -15.0

    def evaluate(self, X):
        x = X
        f = 5 * x[:, :4].sum(1) - 5 * (x[:, :4] ** 2).sum(1) - x[:, 4:13].sum(1)
        g = [2*x[:, 0]+2*x[:, 1]+x[:, 9]+x[:, 10]-10, 2*x[:, 0]+2*x[:, 2]+x[:, 9]+x[:, 11]-10,
             2*x[:, 1]+2*x[:, 2]+x[:, 10]+x[:, 11]-10, -8*x[:, 0]+x[:, 9], -8*x[:, 1]+x[:, 10],
             -8*x[:, 2]+x[:, 11], -2*x[:, 3]-x[:, 4]+x[:, 9], -2*x[:, 5]-x[:, 6]+x[:, 10],
             -2*x[:, 7]-x[:, 8]+x[:, 11]]
        return f, np.column_stack(g)


class G02(Problem):
    """n = 20; best-known -0.803619 (Liang et al. 2006)."""
    name, dim, n_con = "G02", 20, 2
    lb = np.zeros(20)
    ub = np.full(20, 10.0)
    known_x = np.array([3.16246061572185, 3.12833142812967, 3.09479212988791, 3.06145059523469,
                        3.02792915885555, 2.99382606701730, 2.95866871765285, 2.92184227312450,
                        0.49482511456933, 0.48835711005490, 0.48231642711865, 0.47664475092742,
                        0.47129550835493, 0.46623099264167, 0.46142004984199, 0.45683664767217,
                        0.45245876903267, 0.44826762241853, 0.44424700958760, 0.44038285956317])
    known_f = -0.803619104

    def evaluate(self, X):
        i = np.arange(1, 21)
        c = np.cos(X)
        f = -np.abs((c ** 4).sum(1) - 2 * (c ** 2).prod(1)) / np.sqrt((i * X ** 2).sum(1))
        return f, np.column_stack([0.75 - X.prod(1), X.sum(1) - 7.5 * 20])


class G03(Problem):
    """n = 10, one equality (|h| <= 1e-4); best-known -1.0005001 for that tolerance."""
    name, dim, n_con = "G03", 10, 1
    lb = np.zeros(10)
    ub = np.ones(10)
    known_x = np.full(10, 1 / np.sqrt(10))
    known_f = -1.0005001

    def evaluate(self, X):
        f = -(np.sqrt(10.0) ** 10) * X.prod(1)
        return f, (np.abs((X ** 2).sum(1) - 1.0) - 1e-4).reshape(-1, 1)


class G04(Problem):
    name, dim, n_con = "G04", 5, 6
    lb = np.array([78.0, 33.0, 27.0, 27.0, 27.0])
    ub = np.array([102.0, 45.0, 45.0, 45.0, 45.0])
    known_x = np.array([78.0, 33.0, 29.995256025682, 45.0, 36.775812905788])
    known_f = -30665.539

    def evaluate(self, X):
        x1, x2, x3, x4, x5 = X.T
        f = 5.3578547 * x3 ** 2 + 0.8356891 * x1 * x5 + 37.293239 * x1 - 40792.141
        u = 85.334407 + 0.0056858 * x2 * x5 + 0.0006262 * x1 * x4 - 0.0022053 * x3 * x5
        v = 80.51249 + 0.0071317 * x2 * x5 + 0.0029955 * x1 * x2 + 0.0021813 * x3 ** 2
        w = 9.300961 + 0.0047026 * x3 * x5 + 0.0012547 * x1 * x3 + 0.0019085 * x3 * x4
        return f, np.column_stack([-u, u - 92, 90 - v, v - 110, 20 - w, w - 25])


class G06(Problem):
    name, dim, n_con = "G06", 2, 2
    lb = np.array([13.0, 0.0])
    ub = np.array([100.0, 100.0])
    known_x = np.array([14.095, 0.84296])
    known_f = -6961.81388

    def evaluate(self, X):
        x1, x2 = X.T
        f = (x1 - 10) ** 3 + (x2 - 20) ** 3
        return f, np.column_stack([-(x1 - 5) ** 2 - (x2 - 5) ** 2 + 100,
                                   (x1 - 6) ** 2 + (x2 - 5) ** 2 - 82.81])


class G07(Problem):
    name, dim, n_con = "G07", 10, 8
    lb = np.full(10, -10.0)
    ub = np.full(10, 10.0)
    known_x = np.array([2.171996, 2.363683, 8.773926, 5.095984, 0.9906548,
                        1.430574, 1.321644, 9.828726, 8.280092, 8.375927])
    known_f = 24.3062091

    def evaluate(self, X):
        x = X.T
        f = (x[0]**2 + x[1]**2 + x[0]*x[1] - 14*x[0] - 16*x[1] + (x[2]-10)**2 + 4*(x[3]-5)**2
             + (x[4]-3)**2 + 2*(x[5]-1)**2 + 5*x[6]**2 + 7*(x[7]-11)**2 + 2*(x[8]-10)**2
             + (x[9]-7)**2 + 45)
        g = [-105 + 4*x[0] + 5*x[1] - 3*x[6] + 9*x[7], 10*x[0] - 8*x[1] - 17*x[6] + 2*x[7],
             -8*x[0] + 2*x[1] + 5*x[8] - 2*x[9] - 12,
             3*(x[0]-2)**2 + 4*(x[1]-3)**2 + 2*x[2]**2 - 7*x[3] - 120,
             5*x[0]**2 + 8*x[1] + (x[2]-6)**2 - 2*x[3] - 40,
             x[0]**2 + 2*(x[1]-2)**2 - 2*x[0]*x[1] + 14*x[4] - 6*x[5],
             0.5*(x[0]-8)**2 + 2*(x[1]-4)**2 + 3*x[4]**2 - x[5] - 30,
             -3*x[0] + 6*x[1] + 12*(x[8]-8)**2 - 7*x[9]]
        return f, np.column_stack(g)


class G08(Problem):
    name, dim, n_con = "G08", 2, 2
    lb = np.zeros(2)
    ub = np.full(2, 10.0)
    known_x = np.array([1.2279713, 4.2453733])
    known_f = -0.0958250414

    def evaluate(self, X):
        x1, x2 = X.T
        f = -np.sin(2 * PI * x1) ** 3 * np.sin(2 * PI * x2) / (x1 ** 3 * (x1 + x2))
        return f, np.column_stack([x1 ** 2 - x2 + 1, 1 - x1 + (x2 - 4) ** 2])


class G09(Problem):
    name, dim, n_con = "G09", 7, 4
    lb = np.full(7, -10.0)
    ub = np.full(7, 10.0)
    known_x = np.array([2.330499, 1.951372, -0.4775414, 4.365726, -0.6244870, 1.038131, 1.594227])
    known_f = 680.6300573

    def evaluate(self, X):
        x = X.T
        f = ((x[0]-10)**2 + 5*(x[1]-12)**2 + x[2]**4 + 3*(x[3]-11)**2 + 10*x[4]**6
             + 7*x[5]**2 + x[6]**4 - 4*x[5]*x[6] - 10*x[5] - 8*x[6])
        g = [-127 + 2*x[0]**2 + 3*x[1]**4 + x[2] + 4*x[3]**2 + 5*x[4],
             -282 + 7*x[0] + 3*x[1] + 10*x[2]**2 + x[3] - x[4],
             -196 + 23*x[0] + x[1]**2 + 6*x[5]**2 - 8*x[6],
             4*x[0]**2 + x[1]**2 - 3*x[0]*x[1] + 2*x[2]**2 + 5*x[5] - 11*x[6]]
        return f, np.column_stack(g)


class G10(Problem):
    name, dim, n_con = "G10", 8, 6
    lb = np.array([100.0, 1000.0, 1000.0] + [10.0] * 5)
    ub = np.array([10000.0] * 3 + [1000.0] * 5)
    known_x = np.array([579.3167, 1359.943, 5110.071, 182.0174, 295.5985, 217.9799,
                        286.4162, 395.5979])
    known_f = 7049.2480205

    def evaluate(self, X):
        x = X.T
        f = x[0] + x[1] + x[2]
        g = [-1 + 0.0025*(x[3] + x[5]), -1 + 0.0025*(x[4] + x[6] - x[3]), -1 + 0.01*(x[7] - x[4]),
             -x[0]*x[5] + 833.33252*x[3] + 100*x[0] - 83333.333,
             -x[1]*x[6] + 1250*x[4] + x[1]*x[3] - 1250*x[3],
             -x[2]*x[7] + 1250000 + x[2]*x[4] - 2500*x[4]]
        return f, np.column_stack(g)


class G12(Problem):
    """Disjoint feasible region: 9^3 spheres of radius 0.25."""
    name, dim, n_con = "G12", 3, 1
    lb = np.zeros(3)
    ub = np.full(3, 10.0)
    known_x = np.array([5.0, 5.0, 5.0])
    known_f = -1.0
    _grid = np.array(np.meshgrid(*[np.arange(1, 10)] * 3)).reshape(3, -1).T.astype(float)

    def evaluate(self, X):
        f = -(100 - ((X - 5) ** 2).sum(1)) / 100.0
        d = ((X[:, None, :] - self._grid[None]) ** 2).sum(2) - 0.0625
        return f, d.min(1).reshape(-1, 1)


CEC_PROBLEMS = [G01(), G02(), G03(), G04(), G06(), G07(), G08(), G09(), G10(), G12()]
