from fides.subproblem import solve_trust_region_subproblem
from scipy.spatial.transform import Rotation as R

import numpy as np
from numpy.linalg import norm
import pytest
from scipy import linalg


@pytest.fixture
def subproblem():
    B = np.array([
        [1, 0, 0],
        [0, 3, 0],
        [0, 0, 5],
    ])

    g = np.array([1, 1, 1])
    return {'B': B, 'g': g}


def quad(s, B, g):
    return 0.5*s.T.dot(B.dot(s)) + s.T.dot(g)


def is_local_quad_min(s, B, g):
    """
    make local perturbations to verify s is a local minimum of quad(s, B, g)
    """
    _, ev = linalg.eig(B)
    perturbs = np.array([
        quad(s + eps*ev[:, iv], B, g)
        for iv in range(ev.shape[1])
        for eps in [1e-2, -1e-2]
    ])
    return np.all((perturbs - quad(s, B, g)) > 0)


def is_bound_quad_min(s, B, g):
    """
    make local rotations to verify that s is a local minimum of quad(s, B,
    g) on the sphere of radius ||s||
    """
    perturbs = np.array([
        quad(R.from_rotvec(np.pi/2 * eps*np.eye(3)[:, iv]).as_matrix().dot(s),
             B, g)
        for iv in range(B.shape[1])
        for eps in [1e-2, -1e-2]
    ])
    return np.all((perturbs - quad(s, B, g)) > 0)


def test_convex_subproblem(subproblem):
    delta = 1.151
    s, case = solve_trust_region_subproblem(subproblem['B'], subproblem['g'],
                                            delta)
    assert np.all(np.real(linalg.eig(subproblem['B'])[0]) > 0)
    assert norm(s) < delta
    assert case == 'posdef'
    assert is_local_quad_min(s, subproblem['B'], subproblem['g'])


def test_nonconvex_subproblem(subproblem):
    subproblem['B'][0, 0] = -1
    delta = 1.151
    s, case = solve_trust_region_subproblem(subproblem['B'], subproblem['g'],
                                            delta)
    assert np.any(np.real(linalg.eig(subproblem['B'])[0]) < 0)
    assert norm(s) == delta
    assert case == 'indef'
    assert is_bound_quad_min(s, subproblem['B'], subproblem['g'])


def test_hard_indef_subproblem(subproblem):
    subproblem['B'][0, 0] = -1
    subproblem['g'][0] = 0
    delta = 0.1
    s, case = solve_trust_region_subproblem(subproblem['B'], subproblem['g'],
                                            delta)
    assert np.any(np.real(linalg.eig(subproblem['B'])[0]) < 0)
    assert np.isclose(norm(s), delta, 1e-6)
    assert case == 'indef'
    assert is_bound_quad_min(s, subproblem['B'], subproblem['g'])


def test_hard_hard_subproblem(subproblem):
    subproblem['B'][0, 0] = -1
    subproblem['g'][0] = 0
    delta = 0.5
    s, case = solve_trust_region_subproblem(subproblem['B'], subproblem['g'],
                                            delta)
    assert np.any(np.real(linalg.eig(subproblem['B'])[0]) < 0)
    assert norm(s) == delta
    assert case == 'hard'
    assert is_bound_quad_min(s, subproblem['B'], subproblem['g'])


