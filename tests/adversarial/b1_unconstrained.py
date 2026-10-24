from harness import *

inf = np.inf


def ub_(n):
    return np.full(n, inf)


def lb_(n):
    return np.full(n, -inf)


P = {}
P['rosen2'] = (rosen_nd, [-1.2, 1], lb_(2), ub_(2), 0.0)
P['rosen10'] = (rosen_nd, np.r_[[-1.2, 1] * 5], lb_(10), ub_(10), 0.0)
P['rosen100'] = (rosen_nd, np.r_[[-1.2, 1] * 50], lb_(100), ub_(100), 0.0)
fq, xs, fo = make_quadratic(20, 1e6)
P['quad20_c1e6'] = (fq, np.zeros(20), lb_(20), ub_(20), fo)
fq, xs, fo = make_quadratic(20, 1e12)
P['quad20_c1e12'] = (fq, np.zeros(20), lb_(20), ub_(20), fo)
fq, xs, fo = make_quadratic(50, 1e8, rot=False)
P['quad50_diag_c1e8'] = (fq, np.zeros(50), lb_(50), ub_(50), fo)
P['powell_sing'] = (powell_singular, [3, -1, 0, 1], lb_(4), ub_(4), 0.0)
P['quartic_flat5'] = (quartic_flat, np.ones(5) * 2, lb_(5), ub_(5), 0.0)
P['abspow1.5_5'] = (abs_pow, np.ones(5) * 2, lb_(5), ub_(5), 0.0)
P['beale'] = (beale, [1, 1], lb_(2), ub_(2), 0.0)
P['beale_far'] = (beale, [-4.5, 4.5], lb_(2), ub_(2), 0.0)
P['cosmix10'] = (cosine_mixture, np.ones(10) * 0.3, lb_(10), ub_(10), -1.0)
P['saddle10'] = (
    saddle_quadratic(10),
    np.zeros(10) + 1e-3,
    lb_(10),
    ub_(10),
    np.nan,
)
P['trid50'] = (
    trid,
    np.zeros(50),
    lb_(50),
    ub_(50),
    -50 * (50 + 4) * (50 - 1) / 6,
)
P['dixon20'] = (dixon_price, np.ones(20) * 2, lb_(20), ub_(20), 0.0)
# scale pathologies
P['rosen2_fx1e-10'] = (scaled(rosen_nd, 1e-10), [-1.2, 1], lb_(2), ub_(2), 0.0)
P['rosen2_fx1e10'] = (scaled(rosen_nd, 1e10), [-1.2, 1], lb_(2), ub_(2), 0.0)
P['rosen2_x_1e-6_1e6'] = (
    varscaled(rosen_nd, [1e6, 1e-6]),
    [-1.2e-6, 1e6],
    lb_(2),
    ub_(2),
    0.0,
)
P['rosen10_x_mixed'] = (
    varscaled(rosen_nd, np.logspace(-4, 4, 10)),
    np.r_[[-1.2, 1] * 5] / np.logspace(-4, 4, 10),
    lb_(10),
    ub_(10),
    0.0,
)
P['rosen2_wronghess10x'] = (
    wrong_hessian(rosen_nd, 10.0),
    [-1.2, 1],
    lb_(2),
    ub_(2),
    0.0,
)
P['rosen2_wronghess0.1x'] = (
    wrong_hessian(rosen_nd, 0.1),
    [-1.2, 1],
    lb_(2),
    ub_(2),
    0.0,
)
P['rosen2_noisy1e-6'] = (noisy(rosen_nd, 1e-6), [-1.2, 1], lb_(2), ub_(2), 0.0)
P['expover3'] = (
    exp_overflow,
    np.array([0.0, 0.0, 0.0]),
    lb_(3),
    ub_(3),
    np.nan,
)
P['expover3_far'] = (
    exp_overflow,
    np.array([600.0, 600.0, 600.0]),
    lb_(3),
    ub_(3),
    np.nan,
)

# compute saddle/expover fopt via scipy for reference
for k in ['saddle10', 'expover3', 'expover3_far']:
    fun, x0, lb, ub, _ = P[k]
    r = run_scipy(fun, x0, lb, ub, 'trust-exact')
    P[k] = (fun, x0, lb, ub, r['f'])

rows = battery(P)
import pickle

pickle.dump(rows, open('b1.pkl', 'wb'))
