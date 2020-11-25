"""
Trust Region StepBack
-----------------------------
This module provides the machinery to combine various step-back strategies
that can be used to compute longer steps in case the initially proposed step
had to be truncated due to non-compliance with boundary constraints.
"""

import numpy as np
from scipy.sparse import csc_matrix
from typing import List, Sequence

from .steps import Step, TRStepReflected, TRStepTruncated, RefinedStep


def stepback_reflect(tr_step: Step,
                     x: np.ndarray,
                     sg: np.ndarray,
                     hess: np.ndarray,
                     scaling: csc_matrix,
                     g_dscaling: csc_matrix,
                     delta: float,
                     theta: float,
                     ub: np.ndarray,
                     lb: np.ndarray) -> List[Step]:
    """
    Compute new proposal steps according to a reflection strategy.

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
    rtr_step = TRStepReflected(x, sg, hess, scaling, g_dscaling, delta,
                               theta, ub, lb, tr_step)
    rtr_step.calculate()
    steps = [rtr_step]
    for ireflection in range(len(x) - 1):
        if rtr_step.alpha == 1.0:
            break
        # recursively add more reflections
        rtr_old = rtr_step
        rtr_step = TRStepReflected(x, sg, hess, scaling, g_dscaling, delta,
                                   theta, ub, lb, rtr_old)
        rtr_step.calculate()
        steps.append(rtr_step)

    return steps


def stepback_truncate(tr_step: Step,
                      x: np.ndarray,
                      sg: np.ndarray,
                      hess: np.ndarray,
                      scaling: csc_matrix,
                      g_dscaling: csc_matrix,
                      delta: float,
                      theta: float,
                      ub: np.ndarray,
                      lb: np.ndarray) -> List[Step]:
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
    rtt_step = TRStepTruncated(x, sg, hess, scaling, g_dscaling, delta,
                               theta, ub, lb, tr_step)
    rtt_step.calculate()
    steps = [rtt_step]
    while rtt_step.subspace.shape[1] > 0:
        if rtt_step.alpha == 1.0:
            break
        rtt_step = TRStepTruncated(x, sg, hess, scaling, g_dscaling, delta,
                                   theta, ub, lb, rtt_step)
        rtt_step.calculate()
        steps.append(rtt_step)

    return steps


def stepback_refine(steps: Sequence[Step],
                    x: np.ndarray,
                    sg: np.ndarray,
                    hess: np.ndarray,
                    scaling: csc_matrix,
                    g_dscaling: csc_matrix,
                    delta: float,
                    theta: float,
                    ub: np.ndarray,
                    lb: np.ndarray) -> List[Step]:
    """
    Refine a promising subset of the provided steps based on trust-constr
    optimization

    :param steps:
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
    min_qpval = np.nanmin([step.qpval for step in steps])
    ref_steps = [
        RefinedStep(x, sg, hess, scaling, g_dscaling, delta, theta, ub, lb,
                    step)
        for step in steps
        if ((step.alpha == 1.0 and step.type not in ['trnd', 'tr2d', 'grad'])
            or (step.alpha < 1.0 and step.qpval < min_qpval / 2))
    ]
    for step in ref_steps:
        step.calculate()
    return ref_steps
