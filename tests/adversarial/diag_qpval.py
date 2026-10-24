# flake8: noqa -- exploratory benchmark script, see README.md
"""Check that step.qpval equals the quadratic model evaluated at the step actually taken."""

from harness import *
from numpy.linalg import norm

import fides.minimize as fm
from fides.subproblem import quadratic_form

records = []
orig = fm.trust_region


def patched(x, g, hess, scaling, delta, dv, theta, lb, ub, **kw):
    step = orig(x, g, hess, scaling, delta, dv, theta, lb, ub, **kw)
    # actual step in scaled coords
    D = scaling.diagonal()
    s_actual = step.s + step.s0
    ss_actual = s_actual / D
    q_actual = quadratic_form(step.shess, step.sg, ss_actual)
    ss_claim = step.ss + step.ss0
    records.append(
        dict(
            type=step.type,
            alpha=step.alpha,
            qp=step.qpval,
            q_actual=q_actual,
            nsx_claim=norm(ss_claim),
            nsx_actual=norm(ss_actual),
            delta=delta,
            viol=np.max(np.maximum(x + s_actual - ub, lb - x - s_actual)),
        )
    )
    return step


fm.trust_region = patched


def report(name, fun, x0, lb, ub, opts):
    records.clear()
    r = run_fides(fun, x0, lb, ub, opts)
    bad = [
        rec
        for rec in records
        if abs(rec['qp'] - rec['q_actual']) > 1e-8 * max(1, abs(rec['qp']))
        or rec['nsx_actual'] > rec['delta'] * (1 + 1e-6)
    ]
    print(f'{name:40s} {fmt(r)}  steps={len(records)} inconsistent={len(bad)}')
    for rec in bad[:4]:
        print(
            '     ',
            {
                k: (f'{v:.3e}' if isinstance(v, float) else v)
                for k, v in rec.items()
            },
        )


def allactive(n):
    def fun(x):
        return np.sum((x - 3) ** 2), 2 * (x - 3), 2 * np.eye(n)

    return fun, np.zeros(n), -np.ones(n), np.ones(n)


fq, xs, fo = make_quadratic(10, 1e2)
probs = {
    'allactive5': allactive(5),
    'boxquad10': (fq, np.zeros(10) + 0.01, -np.ones(10), np.ones(10)),
    'rosen2_box': (
        rosen_nd,
        np.array([-1.2, 0.5]),
        np.array([-2.0, -2.0]),
        np.array([0.5, 2.0]),
    ),
    'rosen10_box': (
        rosen_nd,
        np.r_[[-1.2, 0.5] * 5],
        np.full(10, -2.0),
        np.full(10, 0.8),
    ),
}
for pn, (fun, x0, lb, ub) in probs.items():
    for on in [
        'default(2D,reflect)',
        'full,reflect',
        'scg,reflect',
        'scg,truncate',
        '2D,truncate',
        '2D,mixed',
        '2D,refine',
        '2D,single',
    ]:
        report(f'{pn}/{on}', fun, x0, lb, ub, OPTSETS[on])
