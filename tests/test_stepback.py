"""
Tests for the step-back strategies
"""

import logging

import numpy as np
import pytest
from numpy.linalg import norm
from scipy.sparse import csc_matrix

import fides.stepback
from fides.stepback import (
    REFLECT_BOUNCE_RTOL,
    is_negligible_bounce,
    stepback_reflect,
)
from fides.steps import TRStepFull, TRStepReflected

LOGGER = logging.getLogger('fides_test_stepback')


def tr_step_and_args(x, g, hess, lb, ub, delta, theta):
    """
    :return:
        trust-region step and the remaining arguments of stepback_reflect
    """
    # affine scaling according to Optimizer.get_affine_scaling
    bounds = np.where(g < 0, ub, lb)
    scaling = csc_matrix(np.diag(np.sqrt(np.abs(x - bounds))))
    g_dscaling = csc_matrix(np.diag(np.abs(g)))
    sg = scaling.dot(g)
    args = (x, sg, hess, scaling, g_dscaling, delta, theta, ub, lb)
    tr_step = TRStepFull(*args, LOGGER)
    tr_step.calculate()
    return tr_step, args


@pytest.fixture
def reflected_steps(monkeypatch):
    """
    records all reflected steps that are calculated
    """
    steps = []
    calculate = TRStepReflected.calculate

    def recording_calculate(self):
        steps.append(self)
        calculate(self)

    monkeypatch.setattr(TRStepReflected, 'calculate', recording_calculate)
    return steps


def test_reflect_stops_after_negligible_bounce(reflected_steps, monkeypatch):
    """
    Box [-1, 1]^n where all but the first variable sit at their upper bound
    and the gradient pushes them further into it. The model has negative
    curvature along the first variable, such that the trust-region step
    crosses ub[0] and every reflected step runs back into ub[0]. The
    segments of these bouncing steps shrink by a factor 1 - theta per
    reflection. Without a cutoff, all n reflections are computed, down to
    alpha ~ 1e-14 (fides-dev/fides#83).
    """
    n = 10
    lb, ub = -np.ones(n), np.ones(n)
    x = ub - 1e-8
    x[0] = 0.0
    hess = np.eye(n)
    hess[0, 0] = -4.0
    g = np.full(n, -0.1)
    g[0] = -1.0
    theta = 0.95
    tr_step, args = tr_step_and_args(x, g, hess, lb, ub, 10.0, theta)
    assert tr_step.alpha < 1
    assert list(tr_step.iminbr) == [0]

    steps = stepback_reflect(tr_step, *args)
    assert reflected_steps == steps
    for step in steps:
        assert list(step.iminbr) == [0]
        assert step.reflection_indices == {0}
    seg = np.array([norm(step.ss) for step in steps])
    np.testing.assert_allclose(seg[1:] / seg[:-1], 1 - theta, rtol=1e-3)
    assert not any(map(is_negligible_bounce, steps[:-1]))
    assert is_negligible_bounce(steps[-1])
    assert len(steps) == 7

    # without the cutoff, the recursion only ends at the maximum number of
    # reflections, the best step remains the same up to the tolerance
    monkeypatch.setattr(
        fides.stepback, 'is_negligible_bounce', lambda _: False
    )
    all_steps = stepback_reflect(tr_step, *args)
    assert len(all_steps) == n
    assert all_steps[-1].alpha < 1e-13
    best = min(steps, key=lambda step: step.qpval)
    best_all = min(all_steps, key=lambda step: step.qpval)
    assert best.qpval == pytest.approx(best_all.qpval, rel=REFLECT_BOUNCE_RTOL)
    s = best.ss + best.ss0
    s_all = best_all.ss + best_all.ss0
    assert norm(s - s_all) <= REFLECT_BOUNCE_RTOL * norm(s_all)


def test_reflect_continues_at_corner(reflected_steps):
    """
    A negligible reflected segment that is stopped by a different boundary
    than the one it was reflected at turns at a corner of the box. The next
    reflection may still yield a much better step.
    """
    lb, ub = -np.ones(3), np.ones(3)
    # through the coupling with x[0], the trust-region step pushes x[1] and
    # x[2] into their upper bounds, which are almost equally far away
    x = np.array([0.0, 0.9, 0.9 - 1e-10])
    hess = np.array([[2.0, -1.0, -1.0], [-1.0, 2.0, 0.0], [-1.0, 0.0, 2.0]])
    g = np.array([-3.0, 0.1, 0.1])
    tr_step, args = tr_step_and_args(x, g, hess, lb, ub, 10.0, 1 - 1e-12)
    assert list(tr_step.iminbr) == [1]

    steps = stepback_reflect(tr_step, *args)
    assert reflected_steps == steps
    assert len(steps) == 2
    corner, step = steps
    assert list(corner.iminbr) == [2]
    assert norm(corner.ss) < REFLECT_BOUNCE_RTOL * norm(corner.ss + corner.ss0)
    assert not is_negligible_bounce(corner)
    assert step.alpha == 1.0
    assert step.qpval < 1.4 * corner.qpval < 0
