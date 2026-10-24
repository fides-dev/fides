# flake8: noqa -- exploratory benchmark script, see README.md
"""Adversarial test problems for fides. Each returns (fun, x0, lb, ub, fopt, name)
fun returns (f, g, h)."""

import numpy as np


def rosen_nd(x):
    xm, xp = x[:-1], x[1:]
    f = np.sum(100 * (xp - xm**2) ** 2 + (1 - xm) ** 2)
    g = np.zeros_like(x)
    g[:-1] += -400 * (xp - xm**2) * xm - 2 * (1 - xm)
    g[1:] += 200 * (xp - xm**2)
    n = len(x)
    h = np.zeros((n, n))
    d = np.zeros(n)
    d[:-1] += 1200 * xm**2 - 400 * xp + 2
    d[1:] += 200
    h += np.diag(d)
    od = -400 * xm
    h += np.diag(od, 1) + np.diag(od, -1)
    return f, g, h


def make_quadratic(n, cond, seed=0, rot=True):
    rng = np.random.default_rng(seed)
    ev = np.logspace(0, np.log10(cond), n)
    if rot:
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    else:
        Q = np.eye(n)
    A = Q @ np.diag(ev) @ Q.T
    A = 0.5 * (A + A.T)
    xs = rng.standard_normal(n)
    b = -A @ xs

    def fun(x):
        g = A @ x + b
        f = 0.5 * x @ A @ x + b @ x
        return f, g, A

    fopt = fun(xs)[0]
    return fun, xs, fopt


def scaled(fun, scale):
    """rescale objective value by a factor"""

    def f2(x):
        f, g, h = fun(x)
        return scale * f, scale * g, scale * h

    return f2


def varscaled(fun, s):
    """x = s * z, optimize over z"""
    s = np.asarray(s)

    def f2(z):
        f, g, h = fun(s * z)
        return f, s * g, (s[:, None] * h) * s[None, :]

    return f2


def powell_singular(x):
    r = np.array(
        [
            x[0] + 10 * x[1],
            np.sqrt(5) * (x[2] - x[3]),
            (x[1] - 2 * x[2]) ** 2,
            np.sqrt(10) * (x[0] - x[3]) ** 2,
        ]
    )
    J = np.array(
        [
            [1, 10, 0, 0],
            [0, 0, np.sqrt(5), -np.sqrt(5)],
            [0, 2 * (x[1] - 2 * x[2]), -4 * (x[1] - 2 * x[2]), 0],
            [
                2 * np.sqrt(10) * (x[0] - x[3]),
                0,
                0,
                -2 * np.sqrt(10) * (x[0] - x[3]),
            ],
        ]
    )
    f = 0.5 * r @ r
    g = J.T @ r
    h = J.T @ J
    # second-order terms
    h2 = np.zeros((4, 4))
    # r3 = (x1-2x2)^2 : hessian 2*[1,-2]^T[1,-2]
    v = np.array([0, 1, -2, 0.0])
    h2 += r[2] * 2 * np.outer(v, v)
    v = np.array([1, 0, 0, -1.0])
    h2 += r[3] * 2 * np.sqrt(10) * np.outer(v, v)
    return f, g, h + h2


def quartic_flat(x):
    f = np.sum(x**4)
    g = 4 * x**3
    h = np.diag(12 * x**2)
    return f, g, h


def abs_pow(x, p=1.5):
    a = np.abs(x)
    f = np.sum(a**p)
    g = p * np.sign(x) * a ** (p - 1)
    h = np.diag(p * (p - 1) * a ** (p - 2))
    return f, g, h


def beale(x):
    r = np.array(
        [
            1.5 - x[0] * (1 - x[1]),
            2.25 - x[0] * (1 - x[1] ** 2),
            2.625 - x[0] * (1 - x[1] ** 3),
        ]
    )
    J = np.array(
        [
            [-(1 - x[1]), x[0]],
            [-(1 - x[1] ** 2), 2 * x[0] * x[1]],
            [-(1 - x[1] ** 3), 3 * x[0] * x[1] ** 2],
        ]
    )
    f = 0.5 * r @ r
    g = J.T @ r
    h = J.T @ J
    h2 = np.zeros((2, 2))
    h2[0, 1] = h2[1, 0] = r[0] * 1 + r[1] * 2 * x[1] + r[2] * 3 * x[1] ** 2
    h2[1, 1] = r[1] * 2 * x[0] + r[2] * 6 * x[0] * x[1]
    return f, g, h + h2


def cosine_mixture(x):
    # many negative-curvature regions
    f = -0.1 * np.sum(np.cos(5 * np.pi * x)) + np.sum(x**2)
    g = 0.5 * np.pi * np.sin(5 * np.pi * x) + 2 * x
    h = np.diag(2.5 * np.pi**2 * np.cos(5 * np.pi * x) + 2)
    return f, g, h


def saddle_quadratic(n, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    ev = np.linspace(-1, 1, n)
    ev[ev == 0] = 0.1
    A = Q @ np.diag(ev) @ Q.T

    def fun(x):
        # quartic regularization keeps it bounded
        f = 0.5 * x @ A @ x + 0.01 * np.sum(x**4)
        g = A @ x + 0.04 * x**3
        h = A + np.diag(0.12 * x**2)
        return f, g, h

    return fun


def exp_overflow(x):
    # objective grows super fast: tests non-finite handling
    f = np.sum(np.exp(x)) + np.sum((x + 50) ** 2)
    g = np.exp(x) + 2 * (x + 50)
    h = np.diag(np.exp(x) + 2)
    return f, g, h


def noisy(fun, sigma, seed=0):
    rng = np.random.default_rng(seed)

    def f2(x):
        f, g, h = fun(x)
        return f * (1 + sigma * rng.standard_normal()), g, h

    return f2


def wrong_hessian(fun, factor):
    def f2(x):
        f, g, h = fun(x)
        return f, g, factor * h

    return f2


def trid(x):
    n = len(x)
    f = np.sum((x - 1) ** 2) - np.sum(x[1:] * x[:-1])
    g = 2 * (x - 1)
    g[1:] -= x[:-1]
    g[:-1] -= x[1:]
    h = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    return f, g, h


def dixon_price(x):
    n = len(x)
    i = np.arange(2, n + 1)
    f = (x[0] - 1) ** 2 + np.sum(i * (2 * x[1:] ** 2 - x[:-1]) ** 2)
    g = np.zeros(n)
    t = 2 * x[1:] ** 2 - x[:-1]
    g[0] = 2 * (x[0] - 1)
    g[1:] += i * 2 * t * 4 * x[1:]
    g[:-1] += i * 2 * t * (-1)
    h = np.zeros((n, n))
    h[0, 0] += 2
    d1 = i * (2 * (4 * x[1:]) ** 2 + 2 * t * 4)
    h[np.arange(1, n), np.arange(1, n)] += d1
    h[np.arange(0, n - 1), np.arange(0, n - 1)] += i * 2
    od = i * 2 * (-1) * 4 * x[1:]
    h[np.arange(1, n), np.arange(0, n - 1)] += od
    h[np.arange(0, n - 1), np.arange(1, n)] += od
    return f, g, h
