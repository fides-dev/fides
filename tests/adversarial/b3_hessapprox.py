# flake8: noqa -- exploratory benchmark script, see README.md
from harness import *

from fides import BB, BFGS, DFP, SR1, Broyden

inf = np.inf


def ub_(n):
    return np.full(n, inf)


def lb_(n):
    return np.full(n, -inf)


P = {}
P['rosen2'] = (rosen_nd, [-1.2, 1], lb_(2), ub_(2), 0.0)
P['rosen10'] = (
    rosen_nd,
    np.r_[[-1.2, 1] * 5],
    lb_(10),
    ub_(10),
    3.98662,
)  # local min
P['rosen30'] = (rosen_nd, np.r_[[-1.2, 1] * 15], lb_(30), ub_(30), 3.98662)
fq, xs, fo = make_quadratic(20, 1e6)
P['quad20_c1e6'] = (fq, np.zeros(20), lb_(20), ub_(20), fo)
P['powell_sing'] = (powell_singular, [3, -1, 0, 1], lb_(4), ub_(4), 0.0)
P['quartic_flat5'] = (quartic_flat, np.ones(5) * 2, lb_(5), ub_(5), 0.0)
P['beale_far'] = (beale, [-4.5, 4.5], lb_(2), ub_(2), 0.0)
P['trid50'] = (
    trid,
    np.zeros(50),
    lb_(50),
    ub_(50),
    -50 * (50 + 4) * (50 - 1) / 6,
)
P['dixon20'] = (dixon_price, np.ones(20) * 2, lb_(20), ub_(20), 0.0)
P['rosen2_x_1e-6_1e6'] = (
    varscaled(rosen_nd, [1e6, 1e-6]),
    [-1.2e-6, 1e6],
    lb_(2),
    ub_(2),
    0.0,
)
P['rosen10_box_cut'] = (
    rosen_nd,
    np.r_[[-1.2, 1] * 5],
    np.full(10, -2.0),
    np.full(10, 0.8),
    None,
)
P['cosmix10'] = (cosine_mixture, np.ones(10) * 0.3, lb_(10), ub_(10), -1.0)


def allactive(n):
    def fun(x):
        return np.sum((x - 3) ** 2), 2 * (x - 3), 2 * np.eye(n)

    return fun, np.zeros(n), -np.ones(n), np.ones(n), n * 4.0


P['allactive20'] = allactive(20)
for k, v in list(P.items()):
    fun, x0, lb, ub, fo = v
    if fo is None:
        r = run_scipy(fun, x0, lb, ub, 'trust-constr')
        r2 = run_scipy(fun, x0, lb, ub, 'L-BFGS-B')
        P[k] = (fun, x0, lb, ub, np.nanmin([r['f'], r2['f']]))

OPT = {
    k: OPTSETS[k]
    for k in [
        'default(2D,reflect)',
        'full,reflect',
        'scg,reflect',
        '2D,truncate',
        '2D,delta1e-6',
    ]
}
rows = battery(
    P,
    optsets=OPT,
    hessians=(BFGS, SR1, BB, Broyden, DFP),
    baselines=('L-BFGS-B',),
)
import pickle

pickle.dump(rows, open('b3.pkl', 'wb'))
