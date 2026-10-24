"""
Tests for the trust-region step proposals, in particular robustness of the
Newton step computation (fides-dev/fides#32).
"""

import logging

import numpy as np
import pytest
import scipy.linalg
from scipy.sparse import csc_matrix

import fides
from fides import Optimizer, Options, SubSpaceDim
from fides.steps import TRStep2D, solve_newton


def rosen(x: np.ndarray, scale: float = 1.0):
    f = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    g = np.array(
        [
            -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
            200 * (x[1] - x[0] ** 2),
        ]
    )
    h = np.array(
        [
            [1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]],
            [-400 * x[0], 200.0],
        ]
    )
    return scale * f, scale * g, scale * h


def failing_lstsq(drivers: set[str]):
    """lstsq that raises the nonconvergence LinAlgError for given drivers"""
    real_lstsq = scipy.linalg.lstsq

    def lstsq(a, b, *args, lapack_driver=None, **kwargs):
        if lapack_driver in drivers:
            raise scipy.linalg.LinAlgError(
                'SVD did not converge in Linear Least Squares'
            )
        return real_lstsq(a, b, *args, lapack_driver=lapack_driver, **kwargs)

    return lstsq


@pytest.fixture
def spd_system():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((5, 5))
    shess = a @ a.T + 5 * np.eye(5)
    sg = rng.standard_normal(5)
    return shess, sg


def test_solve_newton(spd_system):
    shess, sg = spd_system
    s = solve_newton(shess, sg, logging.getLogger())
    np.testing.assert_allclose(shess @ s, -sg)


def test_solve_newton_gelsd_failure_uses_gelsy(
    spd_system, monkeypatch, caplog
):
    shess, sg = spd_system
    monkeypatch.setattr(fides.steps.linalg, 'lstsq', failing_lstsq({'gelsd'}))
    with caplog.at_level(logging.WARNING):
        s = solve_newton(shess, sg, logging.getLogger())
    np.testing.assert_allclose(shess @ s, -sg)
    assert 'gelsd' in caplog.text


def test_solve_newton_all_drivers_fail_uses_gradient(
    spd_system, monkeypatch, caplog
):
    shess, sg = spd_system
    monkeypatch.setattr(
        fides.steps.linalg, 'lstsq', failing_lstsq({'gelsd', 'gelsy'})
    )
    with caplog.at_level(logging.WARNING):
        s = solve_newton(shess, sg, logging.getLogger())
    np.testing.assert_array_equal(s, -sg)
    assert 'gradient' in caplog.text


def test_tr2d_step_with_lstsq_failure(spd_system, monkeypatch):
    shess, sg = spd_system
    n = len(sg)
    args = (
        np.zeros(n),
        sg,
        shess,
        csc_matrix(np.eye(n)),
        csc_matrix(np.zeros((n, n))),
        1.0,
        0.95,
        np.full(n, np.inf),
        np.full(n, -np.inf),
        logging.getLogger(),
    )
    reference = TRStep2D(*args)
    reference.calculate()

    monkeypatch.setattr(fides.steps.linalg, 'lstsq', failing_lstsq({'gelsd'}))
    step = TRStep2D(*args)
    step.calculate()

    np.testing.assert_allclose(step.s, reference.s)
    assert step.qpval == pytest.approx(reference.qpval)


def test_minimize_survives_lstsq_nonconvergence(monkeypatch):
    monkeypatch.setattr(fides.steps.linalg, 'lstsq', failing_lstsq({'gelsd'}))
    opt = Optimizer(
        rosen,
        ub=np.full(2, 5.0),
        lb=np.full(2, -5.0),
        options={Options.SUBSPACE_DIM: SubSpaceDim.TWO},
        verbose=logging.ERROR,
    )
    opt.minimize(np.array([-1.5, 2.0]))
    np.testing.assert_allclose(opt.x_min, np.ones(2), atol=1e-4)


def test_non_finite_scaled_hessian_raises():
    """
    Finite hessian and bounds, but the product scaling * hess * scaling
    overflows. This must not surface as an obscure error from lstsq.
    """
    opt = Optimizer(
        lambda x: rosen(x, scale=1e12),
        ub=np.full(2, 1e300),
        lb=np.full(2, -1e300),
        options={Options.SUBSPACE_DIM: SubSpaceDim.TWO},
        verbose=logging.ERROR,
    )
    with (
        np.errstate(all='ignore'),
        pytest.raises(RuntimeError, match='scaled hessian'),
    ):
        opt.minimize(np.array([-1.5, 2.0]))
