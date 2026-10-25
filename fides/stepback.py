"""
Trust Region StepBack
-----------------------------
This module provides the machinery to combine various step-back strategies
that can be used to compute longer steps in case the initially proposed step
had to be truncated due to non-compliance with boundary constraints.
"""

import numpy as np
from numpy.linalg import norm
from scipy.sparse import csc_matrix

from .steps import Step, TRStepReflected, TRStepTruncated

#: length of a reflected segment relative to the length of the reflected
#: step, below which :func:`stepback_reflect` stops reflecting if the segment
#: ran back into the boundary it was reflected at
REFLECT_BOUNCE_RTOL = 1e-8


def is_negligible_bounce(rtr_step: TRStepReflected) -> bool:
    """
    Checks whether a reflected step ran back into the boundary it was
    reflected at, which happens if the minimizer along the reflected
    direction lies behind the reflection point, and whether its segment is
    negligibly short compared to the whole step.

    Further reflections then typically keep bouncing at the same boundary
    along two alternating directions. Each segment ends at a fraction theta
    of the remaining distance to the boundary and is therefore shorter than
    the previous one by a factor (1 - theta). Together, they extend the step
    by less than a fraction `REFLECT_BOUNCE_RTOL * (1 - theta) / theta`,
    but each costs a subproblem solve. A short segment that ends at a
    different boundary turns at a corner of the feasible set, the next
    reflection may still yield a substantially longer and better step.

    :param rtr_step:
        Reflected step, after calculation

    :return:
        True if further reflections are negligible
    """
    if not set(rtr_step.iminbr) <= rtr_step.reflection_indices:
        return False
    return norm(rtr_step.ss) <= REFLECT_BOUNCE_RTOL * norm(
        rtr_step.ss + rtr_step.ss0
    )


def stepback_reflect(
    tr_step: Step,
    x: np.ndarray,
    sg: np.ndarray,
    hess: np.ndarray,
    scaling: csc_matrix,
    g_dscaling: csc_matrix,
    delta: float,
    theta: float,
    ub: np.ndarray,
    lb: np.ndarray,
) -> list[Step]:
    """
    Compute new proposal steps according to a reflection strategy.
    Reflections are applied recursively until a reflected step no longer
    hits the boundary, at most `len(x)` times, or until the reflected
    steps start bouncing back and forth at the same boundary with negligible
    segment length (see :func:`is_negligible_bounce`).

    :param tr_step:
        Reference trust region step that will be reflected
    :param x:
        Current values of the optimization variables
    :param sg:
        Rescaled objective function gradient at x
    :param hess:
        (Approximate) objective function Hessian at x
    :param g_dscaling:
        Unscaled gradient multiplied by derivative of scaling
        transformation
    :param scaling:
        Scaling transformation according to distance to boundary
    :param delta:
        Trust region radius, note that this applies after scaling
        transformation
    :param theta:
        parameter regulating stepback
    :param lb:
        lower optimization variable boundaries
    :param ub:
        upper optimization variable boundaries

    :return:
        New proposal steps
    """
    rtr_step = TRStepReflected(
        x, sg, hess, scaling, g_dscaling, delta, theta, ub, lb, tr_step
    )
    rtr_step.calculate()
    steps = [rtr_step]
    for _ireflection in range(len(x) - 1):
        if rtr_step.alpha == 1.0 or is_negligible_bounce(rtr_step):
            break
        # recursively add more reflections
        rtr_old = rtr_step
        rtr_step = TRStepReflected(
            x, sg, hess, scaling, g_dscaling, delta, theta, ub, lb, rtr_old
        )
        rtr_step.calculate()
        steps.append(rtr_step)

    return steps


def stepback_truncate(
    tr_step: Step,
    x: np.ndarray,
    sg: np.ndarray,
    hess: np.ndarray,
    scaling: csc_matrix,
    g_dscaling: csc_matrix,
    delta: float,
    theta: float,
    ub: np.ndarray,
    lb: np.ndarray,
) -> list[Step]:
    """
    Compute new proposal steps according to a truncation strategy.

    :param tr_step:
        Reference trust region step that will be reflect
    :param x:
        Current values of the optimization variables
    :param sg:
        Rescaled objective function gradient at x
    :param hess:
        (Approximate) objective function Hessian at x
    :param g_dscaling:
        Unscaled gradient multiplied by derivative of scaling
        transformation
    :param scaling:
        Scaling transformation according to distance to boundary
    :param delta:
        Trust region radius, note that this applies after scaling
        transformation
    :param theta:
        parameter regulating stepback
    :param lb:
        lower optimization variable boundaries
    :param ub:
        upper optimization variable boundaries

    :return:
        New proposal steps
    """
    rtt_step = TRStepTruncated(
        x, sg, hess, scaling, g_dscaling, delta, theta, ub, lb, tr_step
    )
    rtt_step.calculate()
    steps = [rtt_step]
    while rtt_step.subspace.shape[1] > 0:
        if rtt_step.alpha == 1.0:
            break
        rtt_step = TRStepTruncated(
            x, sg, hess, scaling, g_dscaling, delta, theta, ub, lb, rtt_step
        )
        rtt_step.calculate()
        steps.append(rtt_step)

    return steps
