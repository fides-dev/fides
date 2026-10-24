"""Fuzz solve_nd_trust_region_subproblem against an eigendecomposition brute-force reference."""

import logging
import sys
import time
import warnings

sys.path.insert(
    0,
    __import__('os').path.join(
        __import__('os').path.dirname(__file__), '..', '..'
    ),
)
import numpy as np
from numpy.linalg import norm

from fides.subproblem import quadratic_form, solve_nd_trust_region_subproblem

warnings.filterwarnings('ignore')
logging.getLogger('fides').setLevel(logging.CRITICAL)


def reference(B, g, delta):
    """Moré–Sorensen via eigh + bisection in log-lambda; handles hard case."""
    ev, V = np.linalg.eigh(B)
    w = V.T @ g
    lam_lo = max(0.0, -ev[0])

    def s_of(lam):
        d = ev + lam
        c = np.where(np.abs(d) > 0, -w / np.where(d == 0, 1, d), 0.0)
        return V @ c

    # interior?
    if ev[0] > 0 and norm(s_of(0)) <= delta:
        return s_of(0)
    # does norm at lam_lo already fit (hard case)?
    s0 = s_of(lam_lo) if lam_lo > 0 else None
    if lam_lo > 0:
        # components in null space of B + lam_lo I are undefined; set to 0
        mask = np.abs(ev + lam_lo) <= 1e-12 * max(1, np.abs(ev).max())
        c = np.where(mask, 0.0, -w / np.where(mask, 1, ev + lam_lo))
        s0 = V @ c
        if norm(s0) <= delta:
            # hard case: add null-space component to reach boundary
            v = V[:, mask][:, 0]
            sigma = np.sqrt(max(delta**2 - norm(s0) ** 2, 0))
            cands = [s0 + sigma * v, s0 - sigma * v]
            return min(cands, key=lambda s: quadratic_form(B, g, s))
    # bisection on lam in (lam_lo, hi) where norm(s(lam)) decreasing
    lo = lam_lo
    hi = max(lam_lo, 1e-300) * 2 + 1e-300
    hi = max(hi, norm(g) / delta + np.abs(ev).max())
    while norm(s_of(hi)) > delta:
        hi *= 10
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        if norm(s_of(mid)) > delta:
            lo = mid
        else:
            hi = mid
    s = s_of(hi)
    # polish: scale to boundary exactly
    return s


def make_case(rng, n, kind):
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    if kind == 'posdef':
        ev = rng.uniform(0.1, 10, n)
    elif kind == 'indef':
        ev = rng.uniform(-10, 10, n)
    elif kind == 'illcond':
        ev = np.logspace(-8, 8, n) * rng.choice([-1, 1], n)
    elif kind == 'illcond_pos':
        ev = np.logspace(-10, 6, n)
    elif kind in ('hard', 'nearhard'):
        ev = rng.uniform(-10, 10, n)
        ev[0] = -10
    elif kind == 'degenerate':
        ev = (
            np.r_[-1, -1, rng.uniform(0, 1, n - 2)]
            if n > 2
            else np.array([-1, -1.0])
        )
    elif kind == 'zero':
        ev = np.zeros(n)
    elif kind == 'semidef':
        ev = (
            np.r_[0, 0, rng.uniform(0.1, 1, n - 2)]
            if n > 2
            else np.array([0, 0.0])
        )
    elif kind == 'huge':
        ev = rng.uniform(1e10, 1e12, n) * rng.choice([-1, 1], n)
    elif kind == 'tiny':
        ev = rng.uniform(1e-12, 1e-10, n) * rng.choice([-1, 1], n)
    B = Q @ np.diag(ev) @ Q.T
    B = 0.5 * (B + B.T)
    g = rng.standard_normal(n)
    if kind == 'hard':
        g -= Q[:, 0] * (Q[:, 0] @ g)
    if kind == 'nearhard':
        g -= Q[:, 0] * (Q[:, 0] @ g)
        g += Q[:, 0] * 1e-9
    return B, g


rng = np.random.default_rng(0)
fails = []
total = 0
t0 = time.time()
KINDS = [
    'posdef',
    'indef',
    'illcond',
    'illcond_pos',
    'hard',
    'nearhard',
    'degenerate',
    'zero',
    'semidef',
    'huge',
    'tiny',
]
for kind in KINDS:
    for n in [2, 3, 10, 30]:
        for delta in [1e-8, 1e-3, 1.0, 1e3, 1e8]:
            for trial in range(4):
                total += 1
                B, g = make_case(rng, n, kind)
                gscale = rng.choice([1e-8, 1, 1e8])
                g = g * gscale
                try:
                    s, st = solve_nd_trust_region_subproblem(B, g, delta)
                    s = np.real(s)
                    q = quadratic_form(B, g, s)
                    sref = reference(B, g, delta)
                    qref = quadratic_form(B, g, sref)
                    viol = norm(s) / delta - 1
                    scale = max(abs(qref), abs(q), norm(g) * delta, 1e-300)
                    rel = (q - qref) / scale
                    sev = 'SEVERE' if (viol > 1e-2 or rel > 1e-2) else 'minor'
                    if not np.all(np.isfinite(s)):
                        fails.append(
                            (kind, n, delta, gscale, st, 'nonfinite', q, qref)
                        )
                    elif viol > 1e-6:
                        fails.append(
                            (
                                kind,
                                n,
                                delta,
                                gscale,
                                st,
                                f'normviol {viol:.1e} {sev}',
                                q,
                                qref,
                            )
                        )
                    elif rel > 1e-6:
                        fails.append(
                            (
                                kind,
                                n,
                                delta,
                                gscale,
                                st,
                                f'suboptimal {rel:.1e} {sev}',
                                q,
                                qref,
                            )
                        )
                except Exception as e:
                    fails.append(
                        (
                            kind,
                            n,
                            delta,
                            gscale,
                            '?',
                            f'EXC {type(e).__name__}:{str(e)[:40]}',
                            np.nan,
                            np.nan,
                        )
                    )
print(f'{len(fails)} failures out of {total} in {time.time() - t0:.0f}s')
from collections import Counter

for k, c in sorted(
    Counter(
        (f[0], f[4], f[5].split()[0], f[5].split()[-1]) for f in fails
    ).items()
):
    print('  ', k, c)
print()
for f in fails[:50]:
    print(
        f'{f[0]:12s} n={f[1]:2d} delta={f[2]:.0e} gscale={f[3]:.0e} type={f[4]:7s} {f[5]:25s} q={f[6]:+.6e} qref={f[7]:+.6e}'
    )
