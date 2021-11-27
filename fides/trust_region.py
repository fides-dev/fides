"""
Trust Region Step Evaluation
----------------------------
This module provides the machinery to evaluate different trust-region(
-reflective) step proposals and select among them based on to their
performance according to the quadratic approximation of the objective function
"""

import numpy as np
import logging

from scipy.sparse import csc_matrix

from .constants import SubSpaceDim, StepBackStrategy
from .steps import (
    Step, GradientStep, TRStep2D, TRStepFull, RefinedStep,
    TRStepReflected, TRStepSteihaug
)
from .stepback import stepback_reflect, stepback_truncate


def trust_region(x: np.ndarray,
                 g: np.ndarray,
                 hess: np.ndarray,
                 scaling: csc_matrix,
                 delta: float,
                 dv: np.ndarray,
                 theta: float,
                 lb: np.ndarray,
                 ub: np.ndarray,
                 subspace_dim: SubSpaceDim,
                 stepback_strategy: StepBackStrategy,
                 logger: logging.Logger) -> Step:
    """
    Compute a step according to the solution of the trust-region subproblem.
    If step-back is necessary, gradient and reflected trust region step are
    also evaluated in terms of their performance according to the local
    quadratic approximation

    :param x:
        Current values of the optimization variables
    :param g:
        Objective function gradient at x
    :param hess:
        (Approximate) objective function Hessian at x
    :param scaling:
        Scaling transformation according to distance to boundary
    :param delta:
        Trust region radius, note that this applies after scaling
        transformation
    :param dv:
        derivative of scaling transformation
    :param theta:
        parameter regulating stepback
    :param lb:
        lower optimization variable boundaries
    :param ub:
        upper optimization variable boundaries
    :param subspace_dim:
        Subspace dimension in which the subproblem will be solved. Larger
        subspaces require more compute time but can yield higher quality step
        proposals.
    :param stepback_strategy:
        Strategy that is applied when the proposed step exceeds the
        optimization boundary.
    :param logger:
        logging.Logger instance to be used for logging

    :return:
        s: proposed step,
    """
    sg = scaling.dot(g)
    # diag(g_k)*J^v_k Eq (2.5) [ColemanLi1994]
    g_dscaling = csc_matrix(np.diag(np.abs(g) * dv))

    step_options = {
        SubSpaceDim.TWO: TRStep2D,
        SubSpaceDim.FULL: TRStepFull,
        SubSpaceDim.STEIHAUG: TRStepSteihaug,
    }
    tr_step = step_options[subspace_dim](x, sg, hess, scaling, g_dscaling,
                                         delta, theta, ub, lb, logger)
    tr_step.calculate()

    # in case of truncation, we hit the boundary and we check both the
    # gradient and the reflected step, either of which could be better than the
    # TR step

    steps = [tr_step]
    if tr_step.alpha < 1.0 and len(g) > 1:
        g_step = GradientStep(x, sg, hess, scaling, g_dscaling, delta,
                              theta, ub, lb, logger)
        g_step.calculate()
        steps.append(g_step)
        if stepback_strategy == StepBackStrategy.SINGLE_REFLECT:
            rtr_step = TRStepReflected(x, sg, hess, scaling, g_dscaling, delta,
                                       theta, ub, lb, tr_step)
            rtr_step.calculate()
            steps.append(rtr_step)

        if stepback_strategy in [StepBackStrategy.REFLECT,
                                 StepBackStrategy.MIXED]:
            steps.extend(stepback_reflect(
                tr_step, x, sg, hess, scaling, g_dscaling, delta, theta, ub,
                lb
            ))

        if stepback_strategy in [StepBackStrategy.TRUNCATE,
                                 StepBackStrategy.MIXED]:
            steps.extend(stepback_truncate(
                tr_step, x, sg, hess, scaling, g_dscaling, delta, theta, ub,
                lb
            ))

        if stepback_strategy == StepBackStrategy.REFINE and \
                tr_step.subspace.shape[1] > 1:
            ref_step = RefinedStep(
                x, sg, hess, scaling, g_dscaling, delta, theta, ub, lb,
                tr_step
            )
            ref_step.calculate()
            steps.append(ref_step)

    if len(steps) > 1:
        rcountstrs = [str(step.reflection_count)
                      * int(step.reflection_count > 0)
                      for step in steps]
        logger.debug(' | '.join([
            f'{step.type + rcountstr}: [qp:'
            f' {step.qpval:.2E}, '
            f'a: {step.alpha:.2E}]'
            for rcountstr, step in zip(rcountstrs, steps)
        ]))

    qpvals = [step.qpval for step in steps]
    return steps[np.argmin(qpvals)]
