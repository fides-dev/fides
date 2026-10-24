import numpy as np
import pytest
from numpy.linalg import norm
from scipy import linalg
from scipy.spatial.transform import Rotation

from fides.steps import normalize
from fides.subproblem import (
    solve_1d_trust_region_subproblem,
    solve_nd_trust_region_subproblem,
)


@pytest.fixture
def subproblem():
    b = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 5.0],
        ]
    )

    g = np.array([1.0, 1.0, 1.0])
    return {'B': b, 'g': g}


def quad(s, b, g):
    return 0.5 * s.T.dot(b.dot(s)) + s.T.dot(g)


def is_local_quad_min(s, b, g):
    """
    make local perturbations to verify s is a local minimum of quad(s, b, g)
    """
    _, ev = linalg.eig(b)
    perturbs = np.array(
        [
            quad(s + eps * ev[:, iv], b, g)
            for iv in range(ev.shape[1])
            for eps in [1e-2, -1e-2]
        ]
    )
    return np.all((perturbs - quad(s, b, g)) > 0)


def is_bound_quad_min(s, b, g):
    """
    make local rotations to verify that s is a local minimum of quad(s, b,
    g) on the sphere of radius ||s||
    """
    perturbs = np.array(
        [
            quad(
                Rotation.from_rotvec(np.pi / 2 * eps * np.eye(3)[:, iv])
                .as_matrix()
                .dot(s),
                b,
                g,
            )
            for iv in range(b.shape[1])
            for eps in [1e-2, -1e-2]
        ]
    )
    return np.all((perturbs - quad(s, b, g)) > 0)


def test_convex_subproblem(subproblem):
    delta = 1.151
    s, case = solve_nd_trust_region_subproblem(
        subproblem['B'], subproblem['g'], delta
    )
    assert np.all(np.real(linalg.eig(subproblem['B'])[0]) > 0)
    assert norm(s) < delta
    assert case == 'posdef'
    assert is_local_quad_min(s, subproblem['B'], subproblem['g'])

    for alpha in [0, 0.5, -0.5]:
        sc = solve_1d_trust_region_subproblem(
            subproblem['B'], subproblem['g'], s, delta, alpha * s
        )[0]
        assert sc + alpha == 1


def test_nonconvex_subproblem(subproblem):
    subproblem['B'][0, 0] = -1.0
    delta = 1.0
    s, case = solve_nd_trust_region_subproblem(
        subproblem['B'], subproblem['g'], delta
    )
    assert np.any(np.real(linalg.eig(subproblem['B'])[0]) < 0)
    assert np.isclose(norm(s), delta, atol=1e-6, rtol=0)
    assert case == 'indef'
    assert is_bound_quad_min(s, subproblem['B'], subproblem['g'])
    assert not is_bound_quad_min(-s, subproblem['B'], subproblem['g'])

    for alpha in [0, 0.5, -0.5]:
        sc = solve_1d_trust_region_subproblem(
            subproblem['B'], subproblem['g'], s, delta, alpha * s
        )[0]
        assert np.isclose(sc + alpha, 1)


@pytest.mark.parametrize('minev', list(np.logspace(-1, -50, 50)))
def test_nonconvex_subproblem_eigvals(subproblem, minev):
    subproblem['B'][0, 0] = -minev
    delta = 1.0
    s, case = solve_nd_trust_region_subproblem(
        subproblem['B'], subproblem['g'], delta
    )
    assert np.any(np.real(linalg.eig(subproblem['B'])[0]) < 0)
    assert np.isclose(norm(s), delta, atol=1e-6, rtol=0)
    assert is_bound_quad_min(s, subproblem['B'], subproblem['g'])
    assert not is_bound_quad_min(-s, subproblem['B'], subproblem['g'])

    for alpha in [0, 0.5, -0.5]:
        sc = solve_1d_trust_region_subproblem(
            subproblem['B'], subproblem['g'], s, delta, alpha * s
        )[0]
        assert np.isclose(sc + alpha, 1)


def test_hard_indef_subproblem(subproblem):
    subproblem['B'][0, 0] = -1.0
    subproblem['g'][0] = 0.0
    delta = 0.1
    s, case = solve_nd_trust_region_subproblem(
        subproblem['B'], subproblem['g'], delta
    )
    assert np.any(np.real(linalg.eig(subproblem['B'])[0]) < 0)
    assert np.isclose(norm(s), delta, atol=1e-6, rtol=0)
    assert case == 'indef'
    assert is_bound_quad_min(s, subproblem['B'], subproblem['g'])

    snorm = s.copy()
    normalize(snorm)
    for alpha in [0, 0.5, -0.5]:
        assert np.isclose(
            solve_1d_trust_region_subproblem(
                subproblem['B'], subproblem['g'], snorm, delta, alpha * s
            )[0],
            (1 - alpha) * delta,
        )


def test_hard_hard_subproblem(subproblem):
    subproblem['B'][0, 0] = -1.0
    subproblem['g'][0] = 0.0
    delta = 0.5
    s, case = solve_nd_trust_region_subproblem(
        subproblem['B'], subproblem['g'], delta
    )
    assert np.any(np.real(linalg.eig(subproblem['B'])[0]) < 0)
    assert np.isclose(norm(s), delta, atol=1e-6, rtol=0)
    assert case == 'hard'
    assert is_bound_quad_min(s, subproblem['B'], subproblem['g'])

    snorm = s.copy()
    normalize(snorm)
    for alpha in [0, 0.5, -0.5]:
        assert np.isclose(
            solve_1d_trust_region_subproblem(
                subproblem['B'], subproblem['g'], snorm, delta, alpha * s
            )[0],
            (1 - alpha) * delta,
        )


def assert_feasible(s, delta, rtol=1e-10):
    assert np.all(np.isfinite(s))
    assert norm(s) <= delta * (1 + rtol)


def test_misclassified_hard_case():
    # 2D subspace problem from Rosenbrock with variables rescaled by
    # (1e2, 1e-2). The tiny negative eigenvalue is roundoff, the gradient
    # has a non-zero component along its eigenvector, so this is not a
    # hard case.
    b = np.array(
        [
            [-4.7566494908382605e-05, 3.0395448590083091e-14],
            [-3.3744289246449747e-14, 7.5064416569630057e06],
        ]
    )
    g = np.array([2.0314562384141317e-02, 2.6861692953018959e01])
    delta = 8.0
    s, case = solve_nd_trust_region_subproblem(b, g, delta)
    assert case == 'indef'
    assert_feasible(s, delta)
    assert np.isclose(norm(s), delta, rtol=1e-10)
    # moves against the gradient along the direction of negative curvature
    assert s[0] < 0
    assert quad(s, b, g) < -0.164


def test_hard_case_sign(subproblem, monkeypatch):
    # near hard case where root finding fails, the hard case fallback has
    # to pick the direction along the eigenvector that decreases the model
    import fides.subproblem as sp

    def fail(*args, **kwargs):
        raise RuntimeError()

    monkeypatch.setattr(sp, 'newton', fail)
    monkeypatch.setattr(sp, 'brentq', fail)
    b = subproblem['B']
    b[0, 0] = -1.0
    delta = 0.5
    for g0 in [1e-6, -1e-6]:
        g = subproblem['g'].copy()
        g[0] = g0
        s, case = solve_nd_trust_region_subproblem(b, g, delta)
        assert case == 'hard'
        assert np.isclose(norm(s), delta)
        assert np.sign(s[0]) == -np.sign(g0)
        s_flipped = s.copy()
        s_flipped[0] *= -1
        assert quad(s, b, g) < quad(s_flipped, b, g)


@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('gscale', [0.0, 1e-8, 1.0])
def test_repeated_eigenvalues(seed, gscale):
    # eigh returns slightly different eigenvalues for rotated -I
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((2, 2)))
    b = q.dot(np.diag([-1.0, -1.0])).dot(q.T)
    g = rng.standard_normal(2) * gscale
    delta = 1.0
    s, _ = solve_nd_trust_region_subproblem(b, g, delta)
    assert_feasible(s, delta)
    # optimum is -delta * g/||g|| (or any boundary point for g = 0)
    assert np.isclose(
        quad(s, b, g), -0.5 * delta**2 - delta * norm(g), rtol=1e-10
    )


@pytest.mark.parametrize(
    ('b', 'g', 'delta'),
    [
        # interior check accepted |s| <= delta + sqrt(eps)
        (np.eye(2), np.array([2e-8, 0.0]), 1e-8),
        # valid root rejected as |s| - delta > sqrt(eps), hard case fallback
        (
            np.diag(np.logspace(-10, 6, 10)),
            np.random.default_rng(1).standard_normal(10),
            1e8,
        ),
    ],
)
def test_relative_norm_tolerance(b, g, delta):
    s, case = solve_nd_trust_region_subproblem(b, g, delta)
    assert case == 'indef'
    assert_feasible(s, delta)
    assert np.isclose(norm(s), delta, rtol=1e-10)


@pytest.mark.parametrize('b', [np.zeros((2, 2)), np.diag([0.0, 1.0])])
def test_zero_eigenvalue(b):
    g = np.array([0.3, 0.4])
    delta = 1.0
    s, _ = solve_nd_trust_region_subproblem(b, g, delta)
    assert_feasible(s, delta)
    assert np.isclose(norm(s), delta)
    assert quad(s, b, g) < 0
    if not b.any():
        assert np.allclose(s, -delta * g / norm(g))


@pytest.mark.parametrize(
    ('b', 'g', 'delta'),
    [
        # quadratic with condition 1e6, lambda ~ 4e5 and ~ 4e8
        (np.diag(np.logspace(0, 6, 20)), np.ones(20) * 1e5, 1.0),
        (np.diag(np.logspace(0, 6, 20)), np.ones(20) * 1e8, 1.0),
        # lambda ~ 1e160
        (np.diag([1.0, 2.0]), np.array([1e150, 1e150]), 1e-10),
    ],
)
def test_large_lambda(b, g, delta, monkeypatch):
    import fides.subproblem as sp

    def no_brentq(*args, **kwargs):
        raise AssertionError('newton did not converge')

    monkeypatch.setattr(sp, 'brentq', no_brentq)
    s, case = solve_nd_trust_region_subproblem(b, g, delta)
    assert case == 'indef'
    assert_feasible(s, delta)
    assert np.isclose(norm(s), delta, rtol=1e-10)


@pytest.mark.parametrize(
    ('b', 'g', 'delta'),
    [
        (np.diag(np.logspace(0, 6, 20)), np.ones(20) * 1e4, 1.0),
        (np.diag([1.0, 2.0]), np.array([1e150, 1e150]), 1e-10),
        (np.diag([-1.0, 2.0]), np.array([1e-9, 1.0]), 0.5),
    ],
)
def test_brentq_fallback(b, g, delta, monkeypatch):
    # brentq has to find the root on its own, also for huge lambda
    import fides.subproblem as sp

    def fail(*args, **kwargs):
        raise RuntimeError()

    monkeypatch.setattr(sp, 'newton', fail)
    s, case = solve_nd_trust_region_subproblem(b, g, delta)
    assert case == 'indef'
    assert_feasible(s, delta)
    assert np.isclose(norm(s), delta, rtol=1e-10)


def test_nonsymmetric_hessian():
    # only the symmetric part enters the quadratic form
    b = np.array([[1.0, 4.0], [-2.0, 1.0]])
    g = np.array([1.0, -1.0])
    delta = 0.5
    s, _ = solve_nd_trust_region_subproblem(b, g, delta)
    s_sym, _ = solve_nd_trust_region_subproblem(0.5 * (b + b.T), g, delta)
    assert np.allclose(s, s_sym)


def reference_subproblem_solution(b, g, delta):
    """Eigendecomposition + bisection in lambda, handles the hard case."""
    ev, v = np.linalg.eigh(b)
    w = v.T @ g
    lam_lo = max(0.0, -ev[0])

    def s_of(lam):
        d = ev + lam
        c = np.where(np.abs(d) > 0, -w / np.where(d == 0, 1, d), 0.0)
        return v @ c

    if ev[0] > 0 and norm(s_of(0)) <= delta:
        return s_of(0)
    if lam_lo > 0:
        # components in null space of B + lam_lo I are undefined; set to 0
        mask = np.abs(ev + lam_lo) <= 1e-12 * max(1, np.abs(ev).max())
        c = np.where(mask, 0.0, -w / np.where(mask, 1, ev + lam_lo))
        s0 = v @ c
        if norm(s0) <= delta:
            # hard case: add null-space component to reach boundary
            vmin = v[:, mask][:, 0]
            sigma = np.sqrt(max(delta**2 - norm(s0) ** 2, 0))
            cands = [s0 + sigma * vmin, s0 - sigma * vmin]
            return min(cands, key=lambda s: quad(s, b, g))
    lo = lam_lo
    hi = max(lam_lo, 1e-300) * 2 + 1e-300
    hi = max(hi, norm(g) / delta + np.abs(ev).max())
    while norm(s_of(hi)) > delta:
        hi *= 10
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if mid in (lo, hi):
            break
        if norm(s_of(mid)) > delta:
            lo = mid
        else:
            hi = mid
    return s_of(hi)


def make_subproblem(rng, n, kind):
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
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
        ev = np.r_[-1.0, -1.0, rng.uniform(0, 1, n - 2)]
    elif kind == 'zero':
        ev = np.zeros(n)
    elif kind == 'semidef':
        ev = np.r_[0.0, 0.0, rng.uniform(0.1, 1, n - 2)]
    elif kind == 'huge':
        ev = rng.uniform(1e10, 1e12, n) * rng.choice([-1, 1], n)
    elif kind == 'tiny':
        ev = rng.uniform(1e-12, 1e-10, n) * rng.choice([-1, 1], n)
    b = q @ np.diag(ev) @ q.T
    b = 0.5 * (b + b.T)
    g = rng.standard_normal(n)
    if kind in ('hard', 'nearhard'):
        g -= q[:, 0] * (q[:, 0] @ g)
    if kind == 'nearhard':
        g += q[:, 0] * 1e-9
    return b, g


@pytest.mark.parametrize(
    'kind',
    [
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
    ],
)
def test_subproblem_fuzz(kind):
    rng = np.random.default_rng(0)
    for n in [2, 3, 10, 30]:
        for delta in [1e-8, 1e-3, 1.0, 1e3, 1e8]:
            for _ in range(4):
                b, g = make_subproblem(rng, n, kind)
                g *= rng.choice([1e-8, 1, 1e8])
                s, _ = solve_nd_trust_region_subproblem(b, g, delta)
                assert_feasible(s, delta, rtol=1e-6)
                s_ref = reference_subproblem_solution(b, g, delta)
                q, q_ref = quad(s, b, g), quad(s_ref, b, g)
                scale = max(abs(q_ref), abs(q), norm(g) * delta)
                # eigenvalues of B are only accurate up to n*eps*||B||,
                # which limits the accuracy of model values for
                # ill-conditioned B
                atol = n * np.spacing(1) * norm(b, 2) * delta**2
                assert q - q_ref <= 1e-6 * scale + atol


def test_1d_boundary_solution_degenerate_start():
    """
    a start point on the trust region boundary with orthogonal search
    direction leaves s0 as the only feasible point (tau = 0), previously
    0/0 produced nan
    """
    from fides.subproblem import get_1d_trust_region_boundary_solution

    b = np.eye(2)
    g = np.ones(2)
    s = np.array([1.0, 0.0])
    for s0 in [np.array([0.0, 1.0]), np.array([0.0, 1.0 + 1e-16])]:
        tau = get_1d_trust_region_boundary_solution(b, g, s, s0, 1.0)
        assert tau == 0.0
        tau = solve_1d_trust_region_subproblem(b, g, s, 1.0, s0)
        assert np.all(tau == 0.0)
    # s0 marginally outside the trust region due to roundoff
    tau = get_1d_trust_region_boundary_solution(
        b, g, s, np.array([0.0, 1.0 + 1e-8]), 1.0
    )
    assert np.isfinite(tau)
    # huge radii must not raise OverflowError
    with np.errstate(all='ignore'):
        get_1d_trust_region_boundary_solution(b, g, s, np.zeros(2), 1e155)
