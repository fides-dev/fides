import logging

import numpy as np
import pytest
from numpy.linalg import norm

import fides
import fides.minimize
from fides import Optimizer, Options, StepBackStrategy, SubSpaceDim
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
