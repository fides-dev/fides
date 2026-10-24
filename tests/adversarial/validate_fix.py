"""Validate root-cause diagnosis: monkeypatch candidate fixes into fides.subproblem and rerun failing cases.
Fixes: (1) slam treats el==0 with w!=0 as infinite norm (so secular(laminit)<0 and the root is searched);
       (2) hard case picks the sign of the eigenvector that minimizes the model and uses a relative tolerance
           for 'repeated' eigenvalues; (3) acceptance tolerances relative to delta."""

import sys

sys.path.insert(
    0,
    __import__('os').path.join(
        __import__('os').path.dirname(__file__), '..', '..'
    ),
)
import numpy as np
from numpy.linalg import norm
from scipy import linalg
from scipy.optimize import brentq, newton

import fides.subproblem as sp
from fides.subproblem import quadratic_form


def slam(lam, w, eigvals, eigvecs):
    c = w.copy()
    el = eigvals + lam
    nz = el != 0
    c[nz] /= el[nz]
    if np.any(~nz & (w != 0)):
        c[~nz & (w != 0)] = np.inf * np.sign(w[~nz & (w != 0)])
    return eigvecs.dot(c)


def secular(lam, w, eigvals, eigvecs, delta):
    if lam < -np.min(eigvals):
        return np.inf
    s = slam(lam, w, eigvals, eigvecs)
    sn = norm(s)
    if not np.isfinite(sn):
        return -1 / delta
    return 1 / sn - 1 / delta if sn > 0 else np.inf


def solve_nd(b, g, delta, logger=None):
    if delta == 0:
        return np.zeros(g.shape), 'zero'
    eigvals, eigvecs = linalg.eigh(b)  # symmetric -> eigh (orthonormal, real)
    w = -eigvecs.T.dot(g)
    jmin = eigvals.argmin()
    mineig = eigvals[jmin]
    rtol = 1e-10
    if mineig > 0:
        s = slam(0, w, eigvals, eigvecs)
        if norm(s) <= delta * (1 + rtol):
            return s, 'posdef'
        laminit = 0
    else:
        laminit = -mineig
    if secular(laminit, w, eigvals, eigvecs, delta) < 0:
        maxiter = 100
        try:
            r = newton(
                secular,
                laminit,
                sp.dsecular,
                tol=1e-12 * max(1, abs(laminit)),
                maxiter=maxiter,
                args=(w, eigvals, eigvecs, delta),
            )
            s = slam(r, w, eigvals, eigvecs)
            if norm(s) <= delta * (1 + rtol):
                return s, 'indef'
        except RuntimeError:
            pass
        try:
            xa = laminit
            xb = max(laminit, np.sqrt(np.spacing(1))) * 10
            while secular(xb, w, eigvals, eigvecs, delta) < 0 and maxiter > 0:
                xa = xb
                xb *= 10
                maxiter -= 1
            if maxiter > 0:
                r = brentq(
                    secular,
                    xa,
                    xb,
                    xtol=1e-12 * max(1, xb),
                    rtol=4 * np.spacing(1),
                    maxiter=200,
                    args=(w, eigvals, eigvecs, delta),
                )
                s = slam(r, w, eigvals, eigvecs)
                if norm(s) <= delta * (1 + rtol):
                    return s, 'indef'
        except RuntimeError:
            pass
    # hard case
    degenerate = np.abs(eigvals - mineig) <= 1e-12 * max(
        1.0, np.abs(eigvals).max()
    )
    w = w.copy()
    w[degenerate] = 0
    s = slam(-mineig, w, eigvals, eigvecs)
    sigma = np.sqrt(max(delta**2 - norm(s) ** 2, 0))
    v = eigvecs[:, jmin]
    cands = [s + sigma * v, s - sigma * v]
    s = min(cands, key=lambda c: quadratic_form(b, g, c))
    return s, 'hard'


def apply():
    import fides.steps as st

    sp.slam, sp.secular, sp.solve_nd_trust_region_subproblem = (
        slam,
        secular,
        solve_nd,
    )
    st.solve_nd_trust_region_subproblem = solve_nd


if __name__ == '__main__':
    import fuzz_subproblem2 as fz  # runs the baseline fuzz on import

    print('\n===== with candidate fixes =====')
    apply()
    rng = np.random.default_rng(0)
    fails = []
    total = 0
    for kind in fz.KINDS:
        for n in [2, 3, 10, 30]:
            for delta in [1e-8, 1e-3, 1.0, 1e3, 1e8]:
                for trial in range(4):
                    total += 1
                    B, g = fz.make_case(rng, n, kind)
                    gscale = rng.choice([1e-8, 1, 1e8])
                    g = g * gscale
                    s, st_ = solve_nd(B, g, delta)
                    q = quadratic_form(B, g, s)
                    sref = fz.reference(B, g, delta)
                    qref = quadratic_form(B, g, sref)
                    viol = norm(s) / delta - 1
                    scale = max(abs(qref), abs(q), norm(g) * delta, 1e-300)
                    rel = (q - qref) / scale
                    if not np.all(np.isfinite(s)) or viol > 1e-6 or rel > 1e-6:
                        fails.append(
                            (
                                kind,
                                n,
                                delta,
                                gscale,
                                st_,
                                viol,
                                rel,
                                'SEVERE'
                                if (viol > 1e-2 or rel > 1e-2)
                                else 'minor',
                            )
                        )
    from collections import Counter

    print(f'{len(fails)} failures out of {total}')
    for k, c in sorted(Counter((f[0], f[4], f[-1]) for f in fails).items()):
        print('  ', k, c)
    for f in [f for f in fails if f[-1] == 'SEVERE'][:10]:
        print('   SEVERE:', f)

    from harness import *

    inf = np.inf
    P = {
        'rosen2 scaled 1e2/1e-2': (
            varscaled(rosen_nd, [1e2, 1e-2]),
            np.array([-1.2, 1]) / np.array([1e2, 1e-2]),
            0.0,
        ),
        'rosen2 scaled 1e3/1e-3': (
            varscaled(rosen_nd, [1e3, 1e-3]),
            np.array([-1.2, 1]) / np.array([1e3, 1e-3]),
            0.0,
        ),
        'rosen10_x_mixed 1e-2..1e2': (
            varscaled(rosen_nd, np.logspace(-2, 2, 10)),
            np.r_[[-1.2, 1] * 5] / np.logspace(-2, 2, 10),
            3.98662,
        ),
        'rosen10_x_mixed 1e-4..1e4': (
            varscaled(rosen_nd, np.logspace(-4, 4, 10)),
            np.r_[[-1.2, 1] * 5] / np.logspace(-4, 4, 10),
            3.98662,
        ),
        'beale_far': (beale, np.array([-4.5, 4.5]), 0.0),
        'rosen100': (rosen_nd, np.r_[[-1.2, 1] * 50], 3.98662),
        'rosen2': (rosen_nd, np.array([-1.2, 1.0]), 0.0),
        'rosen10': (rosen_nd, np.r_[[-1.2, 1] * 5], 3.98662),
        'expover3_far': (exp_overflow, np.array([600.0, 600.0, 600.0]), 0.0),
    }
    for pn, (fun, x0, fo) in P.items():
        n = len(x0)
        for on in ['default(2D,reflect)', 'full,reflect']:
            r = run_fides(
                fun, x0, np.full(n, -inf), np.full(n, inf), OPTSETS[on]
            )
            print(
                f'FIXED {pn:28s} {on:20s} it={r["it"]:4d} {r["flag"]:>15s} f={r["f"]:+.3e}'
            )
