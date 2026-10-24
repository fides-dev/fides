from harness import *

inf = np.inf
P = {}


# quadratic with many bounds active: target outside box
def shifted_quad(n, cond, shift, seed=0):
    fq, xs, fo = make_quadratic(n, cond, seed)
    # minimize over box [-1,1]; xs is random normal so some outside
    lb = -np.ones(n)
    ub = np.ones(n)
    r = run_scipy(fq, np.zeros(n), lb, ub, 'trust-constr')
    r2 = run_scipy(fq, np.zeros(n), lb, ub, 'L-BFGS-B')
    return fq, np.zeros(n) + 0.01, lb, ub, min(r['f'], r2['f'])


P['boxquad10_c1e2'] = shifted_quad(10, 1e2, 0)
P['boxquad30_c1e4'] = shifted_quad(30, 1e4, 0)
P['boxquad30_c1e8'] = shifted_quad(30, 1e8, 0)


# all bounds active: f = sum((x-3)^2), box [-1,1] -> optimum x=1
def allactive(n):
    def fun(x):
        return np.sum((x - 3) ** 2), 2 * (x - 3), 2 * np.eye(n)

    return fun, np.zeros(n), -np.ones(n), np.ones(n), n * 4.0


P['allactive5'] = allactive(5)
P['allactive50'] = allactive(50)


# bound active but one variable enters from far: start right next to lb
def near_bound(n):
    def fun(x):
        return np.sum((x - 3) ** 2), 2 * (x - 3), 2 * np.eye(n)

    return fun, -np.ones(n) + 1e-10, -np.ones(n), np.ones(n), n * 4.0


P['start_at_lb5'] = near_bound(5)

# rosenbrock in a box that cuts the valley: optimum on bound
P['rosen2_box_cut'] = (
    rosen_nd,
    [-1.2, 0.5],
    np.array([-2.0, -2.0]),
    np.array([0.5, 2.0]),
    None,
)
P['rosen10_box_cut'] = (
    rosen_nd,
    np.r_[[-1.2, 1] * 5],
    np.full(10, -2.0),
    np.full(10, 0.8),
    None,
)
# rosen with tight box around optimum; optimum interior but close to bounds
P['rosen2_tight'] = (
    rosen_nd,
    [-1.2, 1.0],
    np.array([-1.5, -1.5]),
    np.array([1.0001, 1.0001]),
    0.0,
)


# linear objective over box: degenerate hessian
def linear(n):
    c = np.linspace(-1, 1, n)
    c[c == 0] = 0.5

    def fun(x):
        return c @ x, c, np.zeros((n, n))

    xo = np.where(c > 0, -1.0, 1.0)
    return fun, np.zeros(n), -np.ones(n), np.ones(n), c @ xo


P['linear_box5'] = linear(5)
P['linear_box1'] = (
    lambda x: (x[0], np.array([1.0]), np.zeros((1, 1))),
    np.array([0.3]),
    np.array([-1.0]),
    np.array([1.0]),
    -1.0,
)
# 1D problems
P['rosen1d_like'] = (
    lambda x: (
        (x[0] - 2) ** 4,
        np.array([4 * (x[0] - 2) ** 3]),
        np.array([[12 * (x[0] - 2) ** 2]]),
    ),
    np.array([-3.0]),
    np.array([-5.0]),
    np.array([5.0]),
    0.0,
)
P['quartic1d_bound'] = (
    lambda x: (
        (x[0] - 2) ** 4,
        np.array([4 * (x[0] - 2) ** 3]),
        np.array([[12 * (x[0] - 2) ** 2]]),
    ),
    np.array([-3.0]),
    np.array([-5.0]),
    np.array([1.0]),
    1.0,
)
# one-sided bounds (inf on other side)
P['rosen10_lb_only'] = (
    rosen_nd,
    np.r_[[-1.2, 1] * 5],
    np.full(10, -1.5),
    np.full(10, inf),
    None,
)
# huge box: bounds far away, should behave like unconstrained
P['rosen10_hugebox'] = (
    rosen_nd,
    np.r_[[-1.2, 1] * 5],
    np.full(10, -1e6),
    np.full(10, 1e6),
    None,
)
# asymmetric: tiny box width
P['quad5_tinybox'] = (
    lambda x: (np.sum((x - 0.5) ** 2), 2 * (x - 0.5), 2 * np.eye(5)),
    np.full(5, 1e-7),
    np.zeros(5),
    np.full(5, 2e-7),
    5 * (0.5 - 2e-7) ** 2,
)
# box optimum on a vertex via indefinite objective: maximize distance -> concave
P['concave_box10'] = (
    lambda x: (-np.sum(x**2), -2 * x, -2 * np.eye(10)),
    np.linspace(-0.1, 0.1, 10),
    -np.ones(10),
    2 * np.ones(10),
    -40.0,
)
P['concave_box10_zero'] = (
    lambda x: (-np.sum(x**2), -2 * x, -2 * np.eye(10)),
    np.zeros(10),
    -np.ones(10),
    2 * np.ones(10),
    -40.0,
)
# exp with lb
P['expover3_box'] = (
    exp_overflow,
    np.array([600.0, 600.0, 600.0]),
    np.full(3, -100.0),
    np.full(3, 700.0),
    None,
)
# cosine mixture in box
P['cosmix10_box'] = (
    cosine_mixture,
    np.ones(10) * 0.3,
    np.zeros(10),
    np.ones(10),
    None,
)

for k, v in list(P.items()):
    fun, x0, lb, ub, fo = v
    if fo is None:
        r = run_scipy(fun, x0, lb, ub, 'trust-constr')
        r2 = run_scipy(fun, x0, lb, ub, 'L-BFGS-B')
        P[k] = (fun, x0, lb, ub, np.nanmin([r['f'], r2['f']]))

rows = battery(P, baselines=('trust-constr', 'L-BFGS-B'))
import pickle

pickle.dump(rows, open('b2.pkl', 'wb'))
