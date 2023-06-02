import numpy as np
import pytest
from numpy.linalg import norm
from scipy import linalg
from scipy.spatial.transform import Rotation as R

from fides.steps import normalize
from fides.subproblem import (
    solve_1d_trust_region_subproblem,
    solve_nd_trust_region_subproblem,
)


@pytest.fixture
def subproblem():
    B = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 5.0],
        ]
    )

    g = np.array([1.0, 1.0, 1.0])
    return {'B': B, 'g': g}


def quad(s, B, g):
    return 0.5 * s.T.dot(B.dot(s)) + s.T.dot(g)


def is_local_quad_min(s, B, g):
    """
    make local perturbations to verify s is a local minimum of quad(s, B, g)
    """
    _, ev = linalg.eig(B)
    perturbs = np.array(
        [
            quad(s + eps * ev[:, iv], B, g)
            for iv in range(ev.shape[1])
            for eps in [1e-2, -1e-2]
        ]
    )
    return np.all((perturbs - quad(s, B, g)) > 0)


def is_bound_quad_min(s, B, g):
    """
    make local rotations to verify that s is a local minimum of quad(s, B,
    g) on the sphere of radius ||s||
    """
    perturbs = np.array(
        [
            quad(
                R.from_rotvec(np.pi / 2 * eps * np.eye(3)[:, iv])
                .as_matrix()
                .dot(s),
                B,
                g,
            )
            for iv in range(B.shape[1])
            for eps in [1e-2, -1e-2]
        ]
    )
    return np.all((perturbs - quad(s, B, g)) > 0)


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


@pytest.mark.parametrize("minev", list(np.logspace(-1, -50, 50)))
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
