"""
Tests for the trust-region step proposals: internal consistency of the
selected step and robustness of the Newton step computation
(fides-dev/fides#32).
"""

import logging

import numpy as np
import pytest
import scipy.linalg
from numpy.linalg import norm
from scipy.sparse import csc_matrix

import fides
import fides.minimize
from fides import Optimizer, Options, StepBackStrategy, SubSpaceDim
from fides.steps import TRStep2D, solve_newton
from fides.subproblem import quadratic_form


def shifted_sphere(n):
    # minimum at x = 3, outside of the box, all bounds active at solution
    def fun(x):
        return np.sum((x - 3) ** 2), 2 * (x - 3), 2 * np.eye(n)

    return fun, np.zeros(n), -np.ones(n), np.ones(n)


def box_quadratic(n, indefinite):
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    ev = np.logspace(0, 2, n)
    if indefinite:
        ev[: n // 3] *= -1
    hess = q @ np.diag(ev) @ q.T
    xopt = rng.uniform(-3, 3, size=n)

    def fun(x):
        dx = x - xopt
        return 0.5 * dx @ hess @ dx, hess @ dx, hess

    return fun, np.full(n, 0.01), -np.ones(n), np.ones(n)


PROBLEMS = {
    'shifted_sphere': shifted_sphere(5),
    'box_quadratic': box_quadratic(10, indefinite=False),
    'box_quadratic_indefinite': box_quadratic(10, indefinite=True),
}


@pytest.mark.parametrize('problem', PROBLEMS)
@pytest.mark.parametrize('stepback', StepBackStrategy)
@pytest.mark.parametrize('subspace_dim', SubSpaceDim)
def test_step_consistency(problem, stepback, subspace_dim, monkeypatch):
    """
    The selected step must be internally consistent: qpval is the value of
    the quadratic model at the step that is actually taken, ss + ss0 is
    that step in scaled coordinates and it lies inside the trust region.
    """
    if (
        subspace_dim == SubSpaceDim.STEIHAUG
        and stepback == StepBackStrategy.REFINE
    ):
        pytest.skip('Steihaug steps are not compatible with refinement')
    fun, x0, lb, ub = PROBLEMS[problem]

    records = []
    trust_region = fides.minimize.trust_region

    def recording_trust_region(x, g, hess, scaling, delta, *args, **kwargs):
        step = trust_region(x, g, hess, scaling, delta, *args, **kwargs)
        records.append((x.copy(), scaling.diagonal(), delta, step))
        return step

    monkeypatch.setattr(fides.minimize, 'trust_region', recording_trust_region)

    opt = Optimizer(
        fun,
        ub=ub,
        lb=lb,
        verbose=logging.WARNING,
        options={
            Options.SUBSPACE_DIM: subspace_dim,
            Options.STEPBACK_STRAT: stepback,
            Options.MAXITER: 200,
        },
    )
    opt.minimize(x0)

    assert records
    for x, dscaling, delta, step in records:
        # x_new - x = s + s0, which avoids cancellation for tiny steps
        x_new = x + step.s + step.s0
        ss_taken = (step.s + step.s0) / dscaling
        assert np.all(x_new <= ub)
        assert np.all(x_new >= lb)
        assert norm(ss_taken) <= delta * (1 + 1e-6)
        np.testing.assert_allclose(
            step.ss + step.ss0, ss_taken, rtol=1e-8, atol=1e-12
        )
        q_taken = quadratic_form(step.shess, step.sg, ss_taken)
        assert step.qpval == pytest.approx(q_taken, rel=1e-8, abs=1e-12)

    # make sure step back was actually exercised, alternative steps are
    # only computed if the trust-region step hits the boundary
    assert any(
        step.alpha < 1 or step.type in ('g', 'trr', 'trt', 'ref')
        for *_, step in records
    )


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
