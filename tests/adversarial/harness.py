# flake8: noqa -- exploratory benchmark script, see README.md
import logging
import sys
import time
import warnings

import numpy as np

sys.path.insert(
    0,
    __import__('os').path.join(
        __import__('os').path.dirname(__file__), '..', '..'
    ),
)
from problems import *
from scipy.optimize import minimize as spmin

from fides import ExitFlag, Optimizer, Options, StepBackStrategy, SubSpaceDim

warnings.filterwarnings('ignore')
logging.getLogger('fides').setLevel(logging.CRITICAL)


def run_fides(fun, x0, lb, ub, opts=None, hess=None, maxiter=2000):
    o = {Options.MAXITER: maxiter, Options.MAXTIME: 30}
    if opts:
        o.update(opts)
    t = time.time()
    try:
        if hess is None:
            opt = Optimizer(fun, ub, lb, verbose=logging.CRITICAL, options=o)
        else:
            opt = Optimizer(
                lambda x: fun(x)[:2],
                ub,
                lb,
                verbose=logging.CRITICAL,
                options=o,
                hessian_update=hess(),
            )
        opt.minimize(np.array(x0, dtype=float))
        return dict(
            f=opt.fval_min,
            x=opt.x_min,
            it=opt.iteration,
            flag=ExitFlag(opt.exitflag).name,
            t=time.time() - t,
            nfev=opt.iteration + 1,
            err='',
        )
    except Exception as e:
        return dict(
            f=np.nan,
            x=None,
            it=-1,
            flag='EXC',
            t=time.time() - t,
            nfev=-1,
            err=f'{type(e).__name__}: {e}'[:80],
        )


def run_scipy(fun, x0, lb, ub, method='trust-constr'):
    t = time.time()
    nfev = [0]

    def f(x):
        nfev[0] += 1
        return fun(x)[0]

    bounds = None
    if np.any(np.isfinite(lb)) or np.any(np.isfinite(ub)):
        bounds = list(zip(lb, ub))
    try:
        if method == 'trust-exact' and len(x0) > 50:
            method = 'L-BFGS-B'
        if method == 'trust-exact':
            r = spmin(
                f,
                x0,
                jac=lambda x: fun(x)[1],
                hess=lambda x: fun(x)[2],
                method=method,
                options={'maxiter': 2000},
            )
        elif method == 'L-BFGS-B':
            r = spmin(
                f,
                x0,
                jac=lambda x: fun(x)[1],
                method=method,
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-8},
            )
        else:
            r = spmin(
                f,
                x0,
                jac=lambda x: fun(x)[1],
                hess=lambda x: fun(x)[2],
                method=method,
                bounds=bounds,
                options={'maxiter': 2000, 'gtol': 1e-6, 'xtol': 1e-12},
            )
        return dict(f=r.fun, it=r.nit, nfev=nfev[0], t=time.time() - t)
    except Exception as e:
        return dict(
            f=np.nan, it=-1, nfev=-1, t=time.time() - t, err=str(e)[:60]
        )


OPTSETS = {
    'default(2D,reflect)': {},
    'full,reflect': {Options.SUBSPACE_DIM: SubSpaceDim.FULL},
    'scg,reflect': {Options.SUBSPACE_DIM: SubSpaceDim.STEIHAUG},
    '2D,single': {Options.STEPBACK_STRAT: StepBackStrategy.SINGLE_REFLECT},
    '2D,truncate': {Options.STEPBACK_STRAT: StepBackStrategy.TRUNCATE},
    '2D,mixed': {Options.STEPBACK_STRAT: StepBackStrategy.MIXED},
    '2D,refine': {Options.STEPBACK_STRAT: StepBackStrategy.REFINE},
    'full,truncate': {
        Options.SUBSPACE_DIM: SubSpaceDim.FULL,
        Options.STEPBACK_STRAT: StepBackStrategy.TRUNCATE,
    },
    'scg,truncate': {
        Options.SUBSPACE_DIM: SubSpaceDim.STEIHAUG,
        Options.STEPBACK_STRAT: StepBackStrategy.TRUNCATE,
    },
    '2D,delta1e-6': {Options.DELTA_INIT: 1e-6},
    '2D,delta1e6': {Options.DELTA_INIT: 1e6},
    '2D,theta0.5': {Options.THETA_MAX: 0.5},
    '2D,theta0.999': {Options.THETA_MAX: 0.999},
}


def fmt(r):
    if r.get('err'):
        return f'EXC {r["err"]}'
    return f'f={r["f"]:+.3e} it={r["it"]:5d} {r["flag"]:>15s} {r["t"] * 1e3:7.0f}ms'


def battery(
    problems, optsets=OPTSETS, hessians=(None,), baselines=('trust-exact',)
):
    rows = []
    for pname, (fun, x0, lb, ub, fopt) in problems.items():
        print(f'\n=== {pname}  (n={len(x0)}, fopt={fopt:.3e})')
        for bl in baselines:
            if bl == 'trust-exact' and (
                np.any(np.isfinite(lb)) or np.any(np.isfinite(ub))
            ):
                bl = 'trust-constr'
            r = run_scipy(fun, x0, lb, ub, bl)
            print(
                f'  {"scipy " + bl:28s} f={r["f"]:+.3e} it={r["it"]:5d} nfev={r["nfev"]:5d} {r["t"] * 1e3:7.0f}ms'
            )
        for hname, h in (
            [('exact', None)]
            if hessians == (None,)
            else [(hh.__name__ if hh else 'exact', hh) for hh in hessians]
        ):
            for oname, o in optsets.items():
                r = run_fides(fun, x0, lb, ub, o, h)
                gap = r['f'] - fopt if np.isfinite(r['f']) else np.nan
                bad = ''
                if not np.isfinite(gap) or gap > 1e-4 * max(1, abs(fopt)):
                    bad = '  <-- BAD'
                elif r['it'] > 500:
                    bad = '  <-- SLOW'
                print(
                    f'  {hname + ":" + oname:28s} {fmt(r)} gap={gap:+.1e}{bad}'
                )
                rows.append(
                    dict(
                        problem=pname,
                        hess=hname,
                        opts=oname,
                        **{k: v for k, v in r.items() if k != 'x'},
                        gap=gap,
                    )
                )
    return rows
