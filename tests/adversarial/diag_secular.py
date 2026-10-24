"""Instrument the secular-equation solve: how often does Newton fail over to brentq / hard case?"""

import scipy.optimize as so
from harness import *

import fides.subproblem as sp

stats = dict(
    newton_calls=0,
    newton_fail=0,
    newton_iters=0,
    brentq_calls=0,
    brentq_fail=0,
    hard=0,
    nd_calls=0,
    lam=[],
)
_newton, _brentq = so.newton, so.brentq


def newton(f, x0, fprime, **kw):
    stats['newton_calls'] += 1
    try:
        r, res = _newton(
            f,
            x0,
            fprime,
            full_output=True,
            disp=False,
            **{k: v for k, v in kw.items() if k != 'tol'},
            tol=kw['tol'],
        )
        stats['newton_iters'] += res.iterations
        if not res.converged:
            stats['newton_fail'] += 1
            raise RuntimeError('nc')
        stats['lam'].append(r)
        return r
    except RuntimeError:
        stats['newton_fail'] += 1
        raise


def brentq(*a, **kw):
    stats['brentq_calls'] += 1
    try:
        return _brentq(*a, **kw)
    except Exception:
        stats['brentq_fail'] += 1
        raise


sp.newton, sp.brentq = newton, brentq
_solve = sp.solve_nd_trust_region_subproblem


def solve(b, g, delta, logger=None):
    stats['nd_calls'] += 1
    s, t = _solve(b, g, delta, logger)
    if t == 'hard':
        stats['hard'] += 1
    return s, t


sp.solve_nd_trust_region_subproblem = solve
import fides.steps as st

st.solve_nd_trust_region_subproblem = solve

inf = np.inf
fq6, _, fo6 = make_quadratic(20, 1e6)
fq12, _, fo12 = make_quadratic(20, 1e12)
P = {
    'quad20_c1e6': (fq6, np.zeros(20)),
    'quad20_c1e12': (fq12, np.zeros(20)),
    'rosen100': (rosen_nd, np.r_[[-1.2, 1] * 50]),
    'rosen10_x_mixed': (
        varscaled(rosen_nd, np.logspace(-4, 4, 10)),
        np.r_[[-1.2, 1] * 5] / np.logspace(-4, 4, 10),
    ),
    'expover3_far': (exp_overflow, np.array([600.0, 600.0, 600.0])),
    'rosen2_fx1e10': (scaled(rosen_nd, 1e10), [-1.2, 1]),
    'trid50': (trid, np.zeros(50)),
    'dixon20': (dixon_price, np.ones(20) * 2),
}
for pn, (fun, x0) in P.items():
    for on in ['full,reflect', 'default(2D,reflect)']:
        for k in stats:
            stats[k] = [] if k == 'lam' else 0
        n = len(x0)
        r = run_fides(fun, x0, np.full(n, -inf), np.full(n, inf), OPTSETS[on])
        lam = np.array(stats['lam'])
        print(
            f'{pn:16s} {on:20s} it={r["it"]:4d} {r["flag"]:>15s} {r["t"] * 1e3:6.0f}ms | nd={stats["nd_calls"]:4d} newton={stats["newton_calls"]:4d} nfail={stats["newton_fail"]:4d} iters/call={stats["newton_iters"] / max(1, stats["newton_calls"]):5.1f} brentq={stats["brentq_calls"]:3d} bfail={stats["brentq_fail"]} hard={stats["hard"]:3d} lam∈[{lam.min() if len(lam) else np.nan:.1e},{lam.max() if len(lam) else np.nan:.1e}]'
        )
