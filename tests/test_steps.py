import logging

import numpy as np
import pytest
from numpy.linalg import norm
from scipy.optimize import OptimizeResult
from scipy.sparse import csc_matrix

import fides
import fides.minimize
import fides.steps
from fides import ExitFlag, Optimizer, Options, StepBackStrategy, SubSpaceDim
from fides.steps import (
    RefinedStep,
    TRStep2D,
    TRStepFull,
    TRStepReflected,
    TRStepSteihaug,
    TRStepTruncated,
)
from fides.subproblem import quadratic_form

LOGGER = logging.getLogger('fides_test_steps')


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


def affine_scaling(x, g, lb, ub):
    # same as Optimizer.get_affine_scaling
    v = np.sign(g) + (g == 0)
    dv = np.zeros(x.shape)
    bounds = lb.copy()
    bounds[g < 0] = ub[g < 0]
    bounded = np.isfinite(bounds)
    v[bounded] = x[bounded] - bounds[bounded]
    dv[bounded] = 1
    return v, dv


def step_args(x, g, hess, lb, ub, delta, theta=0.95):
    v, dv = affine_scaling(x, g, lb, ub)
    scaling = csc_matrix(np.diag(np.sqrt(np.abs(v))))
    g_dscaling = csc_matrix(np.diag(np.abs(g) * dv))
    sg = scaling.dot(g)
    return (x, sg, hess, scaling, g_dscaling, delta, theta, ub, lb, LOGGER)


def box_quadratic_trials(trials):
    # random box-constrained quadratics, the listed trials made the reflected
    # step overshoot the bounds by roundoff
    rng = np.random.default_rng(0)
    for trial in range(max(trials) + 1):
        n = int(rng.integers(2, 8))
        q, _ = np.linalg.qr(rng.normal(size=(n, n)))
        ev = 10 ** rng.uniform(-1, 2, size=n)
        if rng.random() < 0.3:
            ev[: n // 3] *= -1
        hess = q @ np.diag(ev) @ q.T
        xopt = rng.uniform(-3, 3, size=n)
        lb = -rng.uniform(0.1, 2, size=n)
        ub = rng.uniform(0.1, 2, size=n)
        x0 = rng.uniform(lb, ub)
        if trial in trials:

            def fun(x, hess=hess, xopt=xopt):
                dx = x - xopt
                return 0.5 * dx @ hess @ dx, hess @ dx, hess

            yield trial, fun, x0, lb, ub


@pytest.mark.parametrize('subspace_dim', SubSpaceDim)
def test_reflected_step_starts_inside(subspace_dim):
    """
    The reflected step must start at a strictly feasible point, not at the
    breakpoint on the boundary, otherwise roundoff can push it outside.
    """
    n = 2
    x = np.array([0.9, 0.0])
    g = np.array([-1.0, -0.5])
    hess = np.array([[1.0, -2.0], [-2.0, 1.0]])
    lb, ub = -np.ones(n), np.ones(n)
    args = step_args(x, g, hess, lb, ub, delta=10.0)
    tr_step = {
        SubSpaceDim.TWO: TRStep2D,
        SubSpaceDim.FULL: TRStepFull,
        SubSpaceDim.STEIHAUG: TRStepSteihaug,
    }[subspace_dim](*args)
    tr_step.calculate()
    assert tr_step.alpha < 1
    rtr_step = TRStepReflected(*args[:-1], tr_step)
    rtr_step.calculate()
    assert np.all(rtr_step.x > lb)
    assert np.all(rtr_step.x < ub)
    x_new = x + rtr_step.s + rtr_step.s0
    assert np.all(x_new > lb)
    assert np.all(x_new < ub)


@pytest.mark.parametrize('subspace_dim', SubSpaceDim)
def test_reflection_does_not_exceed_bounds(subspace_dim):
    for _, fun, x0, lb, ub in box_quadratic_trials([63, 96, 237, 268]):
        opt = Optimizer(
            fun,
            ub=ub,
            lb=lb,
            verbose=logging.WARNING,
            options={
                Options.SUBSPACE_DIM: subspace_dim,
                Options.STEPBACK_STRAT: StepBackStrategy.REFLECT,
                Options.MAXITER: 100,
            },
        )
        opt.minimize(x0)
        assert opt.exitflag != ExitFlag.EXCEEDED_BOUNDARY


def test_truncated_step_uses_model_gradient_at_ss0():
    """
    The truncated step searches the remaining subspace starting from ss0,
    its coefficients must minimize q(ss0 + S c), not q(S c).
    """
    hess = np.array(
        [[28.0, 21.0, -24.0], [21.0, 20.0, -20.0], [-24.0, -20.0, 23.0]]
    )
    x = np.array([0.5, 0.0, 0.0])
    g = np.array([0.0, 3.0, -3.0])
    lb, ub = -np.ones(3), np.ones(3)
    args = step_args(x, g, hess, lb, ub, delta=100.0)
    tr_step = TRStepFull(*args)
    tr_step.calculate()
    assert tr_step.alpha < 1
    trt_step = TRStepTruncated(*args[:-1], tr_step)
    trt_step.calculate()
    subspace = trt_step.subspace
    assert subspace.shape[1] == 2
    # unconstrained minimizer in the subspace lies inside the trust region,
    # so the model gradient restricted to the subspace must vanish there
    ss = trt_step.ss0 + subspace.dot(trt_step.og_sc)
    assert norm(ss) < trt_step.delta
    grad = subspace.T.dot(trt_step.shess.dot(ss) + trt_step.sg)
    assert norm(grad) < 1e-10 * norm(trt_step.sg)


def test_truncated_subspace_orthonormal():
    """
    Truncation zeroes rows of the subspace, which has to be
    re-orthonormalized for the trust-region constraint to hold.
    """
    n = 3
    hess = np.eye(n)
    g = -np.array([1.0, 1.0, 0.0])
    x = np.array([0.99, 0.0, 0.0])
    lb, ub = -np.ones(n) * 10, np.ones(n)
    delta = 0.5
    args = step_args(x, g, hess, lb, ub, delta=delta)
    tr_step = TRStep2D(*args)
    tr_step.calculate()
    assert tr_step.subspace.shape[1] == 2
    assert tr_step.alpha < 1
    trt_step = TRStepTruncated(*args[:-1], tr_step)
    trt_step.calculate()
    subspace = trt_step.subspace
    np.testing.assert_allclose(
        subspace.T @ subspace, np.eye(subspace.shape[1]), atol=1e-12
    )
    assert np.all(subspace[tr_step.iminbr, :] == 0)
    assert norm(trt_step.ss + trt_step.ss0) <= delta * (1 + 1e-12)


def test_refined_step_zero_scaling():
    """
    A variable on its active bound has zero scaling, which must not break
    the bound constraints of the refinement.
    """
    n = 3
    hess = np.diag([1.0, 2.0, 3.0])
    g = np.array([-1.0, -1.0, -1.0])
    lb, ub = -np.ones(n), np.array([1000.0, 1.0, 1.0])
    x = np.array([1000.0, 0.9999, 0.5])
    args = step_args(x, g, hess, lb, ub, delta=1.0)
    assert args[3].diagonal()[0] == 0
    tr_step = TRStepFull(*args)
    tr_step.calculate()
    assert tr_step.alpha < 1
    ref_step = RefinedStep(*args[:-1], tr_step)
    ref_step.calculate()
    x_new = x + ref_step.s + ref_step.s0
    assert np.all(np.isfinite(x_new))
    assert np.all(x_new <= ub)
    assert np.all(x_new >= lb)
    assert norm(ref_step.ss) <= 1.0 * (1 + 1e-12)


def test_refined_step_failed_refinement(monkeypatch):
    """
    If trust-constr returns an infeasible point, the refined step has to
    fall back to the unrefined step.
    """
    hess = np.array([[1.0, -2.0, 0.0], [-2.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    g = np.array([-1.0, -0.5, -0.5])
    # wide bounds in the remaining coordinates such that step-back does not
    # mask an oversized step
    lb, ub = np.array([-1.0, -1e3, -1e3]), np.array([1.0, 1e3, 1e3])
    x = np.array([0.9, 0.0, 0.0])
    args = step_args(x, g, hess, lb, ub, delta=1.0)
    tr_step = TRStepFull(*args)
    tr_step.calculate()
    assert tr_step.alpha < 1

    def failed_minimize(*_, **__):
        # far outside the trust region, but inside the bounds
        return OptimizeResult(x=np.array([0.0, 0.0, 20.0]), success=False)

    monkeypatch.setattr(fides.steps, 'minimize', failed_minimize)
    ref_step = RefinedStep(*args[:-1], tr_step)
    ref_step.calculate()
    assert norm(ref_step.ss) <= 1.0 * (1 + 1e-12)
    np.testing.assert_allclose(ref_step.ss, tr_step.ss)
    assert ref_step.qpval == pytest.approx(tr_step.qpval)


def test_2d_step_1d_zero_hessian():
    """
    In one dimension with zero curvature the Newton direction is undefined,
    the step must still be finite and descend.
    """

    def cubic(x):
        return (
            float(x[0] ** 3 - x[0]),
            np.array([3 * x[0] ** 2 - 1]),
            np.array([[6 * x[0]]]),
        )

    x = np.zeros(1)
    lb, ub = np.full(1, -np.inf), np.full(1, np.inf)
    _, g, hess = cubic(x)
    step = TRStep2D(*step_args(x, g, hess, lb, ub, delta=1.0))
    step.calculate()
    assert np.all(np.isfinite(step.s))
    assert step.qpval < 0

    opt = Optimizer(
        cubic,
        ub=ub,
        lb=lb,
        verbose=logging.WARNING,
        options={Options.SUBSPACE_DIM: SubSpaceDim.TWO},
    )
    opt.minimize(x)
    assert opt.x[0] == pytest.approx(1 / np.sqrt(3), rel=1e-6)


def test_steihaug_zero_gradient():
    n = 3
    x = np.zeros(n)
    g = np.zeros(n)
    hess = np.diag([1.0, -1.0, 2.0])
    lb, ub = -np.ones(n), np.ones(n)
    step = TRStepSteihaug(*step_args(x, g, hess, lb, ub, delta=1.0))
    step.calculate()
    assert np.all(np.isfinite(step.s))
    assert np.all(np.isfinite(step.ss))
    assert np.isfinite(step.qpval)


@pytest.mark.parametrize('xb', [1.0, 1e3, 1e6])
def test_make_non_degenerate_large_x(xb):
    fun, _, lb, ub = shifted_sphere(2)
    opt = Optimizer(fun, ub=ub * xb, lb=lb * xb, verbose=logging.WARNING)
    opt.x = np.array([xb, -xb])
    opt.make_non_degenerate()
    assert opt.x[0] < xb
    assert opt.x[1] > -xb
