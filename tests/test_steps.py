"""
Tests for the trust-region step proposals: internal consistency of the
selected step and robustness of the Newton step computation
(fides-dev/fides#32).
"""

import logging

import numpy as np
import pytest
import scipy.linalg
import scipy.optimize
from numpy.linalg import norm
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
    solve_newton,
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


def refine_setup():
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
    return args, tr_step


def test_refined_step_marginally_infeasible(monkeypatch):
    """
    Refinement results marginally outside the trust region are pulled back
    onto the trust region boundary.
    """
    args, tr_step = refine_setup()

    def marginal_solve(*_, **__):
        return np.array([0.0, 0.0, 1.0 + 1e-8]), 1

    monkeypatch.setattr(
        fides.steps, 'solve_bounded_trust_region_subproblem', marginal_solve
    )
    ref_step = RefinedStep(*args[:-1], tr_step)
    ref_step.calculate()
    assert ref_step.alpha == 1.0
    assert norm(ref_step.ss) <= 1.0
    np.testing.assert_allclose(ref_step.ss, [0.0, 0.0, 1.0], rtol=1e-15)


def test_refined_step_failed_refinement(monkeypatch):
    """
    If the refinement returns an infeasible point, the refined step has to
    fall back to the unrefined step.
    """
    args, tr_step = refine_setup()

    def failed_solve(*_, **__):
        # far outside the trust region, but inside the bounds
        return np.array([0.0, 0.0, 20.0]), 1

    monkeypatch.setattr(
        fides.steps, 'solve_bounded_trust_region_subproblem', failed_solve
    )
    ref_step = RefinedStep(*args[:-1], tr_step)
    ref_step.calculate()
    assert norm(ref_step.ss) <= 1.0 * (1 + 1e-12)
    np.testing.assert_allclose(ref_step.ss, tr_step.ss)
    assert ref_step.qpval == pytest.approx(tr_step.qpval)


def recording_refinement(monkeypatch):
    # records the number of iterations of every refinement
    niters = []
    solve = fides.steps.solve_bounded_trust_region_subproblem

    def recording_solve(*args, **kwargs):
        sc, niter = solve(*args, **kwargs)
        niters.append(niter)
        return sc, niter

    monkeypatch.setattr(
        fides.steps, 'solve_bounded_trust_region_subproblem', recording_solve
    )
    return niters


def convex_refine_setup(scale):
    # convex quadratic where the trust-region step is cut short by the bounds
    rng = np.random.default_rng(2)
    n = 5
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    hess = scale * q @ np.diag(np.logspace(0, 2, n)) @ q.T
    g = scale * rng.normal(size=n) * 100
    lb, ub = -np.ones(n), np.ones(n)
    x = rng.uniform(-0.9, 0.9, size=n)
    args = step_args(x, g, hess, lb, ub, delta=5.0)
    tr_step = TRStepFull(*args)
    tr_step.calculate()
    assert tr_step.alpha < 1
    return args, tr_step


def reference_refinement(args, tr_step):
    # tightly converged solution of the refinement subproblem
    x, _, _, scaling, _, delta, theta, ub, lb, _ = args
    res = scipy.optimize.minimize(
        fun=lambda c: quadratic_form(tr_step.chess, tr_step.cg, c),
        jac=lambda c: tr_step.chess.dot(c) + tr_step.cg,
        hess=lambda c: tr_step.chess,
        x0=tr_step.sc,
        method='trust-constr',
        constraints=[
            scipy.optimize.NonlinearConstraint(
                fun=lambda c: c.dot(c),
                lb=-np.inf,
                ub=delta**2,
                jac=lambda c: 2 * c[None, :],
                hess=lambda c, v: 2 * v[0] * np.eye(len(c)),
            ),
            scipy.optimize.LinearConstraint(
                np.eye(len(x)),
                theta * (lb - x) / scaling.diagonal(),
                theta * (ub - x) / scaling.diagonal(),
            ),
        ],
        options={'maxiter': 1000, 'gtol': 1e-10, 'xtol': 1e-12},
    )
    return quadratic_form(tr_step.shess, tr_step.sg, res.x)


def test_refined_step_scale_invariant():
    """
    Refinement has to find the solution of the refinement subproblem
    regardless of the scale of the objective function (fides-dev/fides#80).
    """
    qpvals = []
    for scale in [1e-8, 1.0, 1e8]:
        args, tr_step = convex_refine_setup(scale)
        ref_step = RefinedStep(*args[:-1], tr_step)
        ref_step.calculate()
        qpvals.append(ref_step.qpval / scale)
    args, tr_step = convex_refine_setup(1.0)
    qpval_ref = reference_refinement(args, tr_step)
    assert qpval_ref < tr_step.qpval
    assert qpvals == pytest.approx([qpvals[1]] * 3, rel=1e-12)
    assert qpvals[1] <= qpval_ref + 1e-8 * abs(qpval_ref)


def test_refine_rosenbrock_iterations(monkeypatch):
    """
    Bounded Rosenbrock problem from fides-dev/fides#80, where every
    refinement ran 100 iterations of trust-constr.
    """

    def fun(x):
        return (
            scipy.optimize.rosen(x),
            scipy.optimize.rosen_der(x),
            scipy.optimize.rosen_hess(x),
        )

    n = 10
    niters = recording_refinement(monkeypatch)
    fvals = {}
    for stepback in [StepBackStrategy.REFLECT, StepBackStrategy.REFINE]:
        opt = Optimizer(
            fun,
            lb=-2 * np.ones(n),
            ub=0.8 * np.ones(n),
            verbose=logging.WARNING,
            options={
                Options.SUBSPACE_DIM: SubSpaceDim.FULL,
                Options.STEPBACK_STRAT: stepback,
            },
        )
        fvals[stepback], *_ = opt.minimize(np.linspace(-1.5, 0.5, n))
        assert opt.exitflag > 0
    assert niters
    assert max(niters) <= n
    assert fvals[StepBackStrategy.REFINE] == pytest.approx(
        fvals[StepBackStrategy.REFLECT], rel=1e-8
    )


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


def test_steihaug_iteration_cap():
    """
    on ill-conditioned problems with interior solution, floating point CG
    may stagnate above the residual tolerance. The iteration must be capped
    and the current iterate returned as step.
    """
    n = 100
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    hess = q @ np.diag(np.logspace(0, -16, n)) @ q.T
    hess = 0.5 * (hess + hess.T)
    g = rng.standard_normal(n)
    g *= 1e-20 / norm(g)
    x = np.zeros(n)
    lb, ub = np.full(n, -np.inf), np.full(n, np.inf)
    step = TRStepSteihaug(*step_args(x, g, hess, lb, ub, delta=1.0))
    step.calculate()
    assert np.all(np.isfinite(step.ss))
    assert norm(step.ss) <= 1.0 + 1e-10
    assert step.qpval < 0


@pytest.mark.parametrize(
    'g', [np.array([1e-200, 1e-200]), np.array([1e-200, 0.0])]
)
def test_steihaug_non_finite_curvature(g):
    """
    overflow in the CG recursion must not hang the loop or raise
    """
    n = 2
    x = np.zeros(n)
    hess = np.diag([1e120, 1e120])
    lb, ub = np.full(n, -np.inf), np.full(n, np.inf)
    step = TRStepSteihaug(*step_args(x, g, hess, lb, ub, delta=1.0))
    with np.errstate(all='ignore'):
        step.calculate()
    assert np.all(np.isfinite(step.ss))
    assert norm(step.ss) <= 1.0


def test_steihaug_tolerance_invariant_to_objective_scale():
    """
    The residual tolerance of the CG iteration must not depend on the scale
    of the objective function (fides-dev/fides#77). With the forcing term
    min(0.5, sqrt(|g|)), a single CG iteration (a steepest descent step) was
    accepted for large gradients, whereas small gradients yielded the
    Newton step.
    """
    hess = np.diag([1e6, 1.0])
    # one steepest descent step reduces the residual to ~0.1 * |g|
    g = hess @ np.array([1.0, 1e5])
    x = np.zeros(2)
    lb, ub = np.full(2, -np.inf), np.full(2, np.inf)
    steps = []
    for scale in 2.0 ** np.array([-40, 0, 40]):
        # gradient decreased by 1e4 since the first iteration, eta = 0.01
        step = TRStepSteihaug(
            *step_args(x, scale * g, scale * hess, lb, ub, delta=1e6),
            sgnorm0=scale * 1e4 * norm(g),
        )
        step.calculate()
        steps.append(step.ss)
    # power of two scales keep the computation exact
    for ss in steps:
        assert np.array_equal(ss, steps[0])
    # interior Newton step
    assert steps[0] == pytest.approx(-np.array([1.0, 1e5]))
