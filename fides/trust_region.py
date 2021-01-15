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
from .steps import Step, GradientStep, TRStep2D, TRStepFull, TRStepReflected
from .stepback import stepback_refine, stepback_reflect, stepback_truncate


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
                 refine_stepback: bool,
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
    :param refine_stepback:
        If set to True, proposed steps that are computed via the specified
        stepback_strategy will be refined via optimization.
    :param logger:
        logging.Logger instance to be used for logging

    :return:
        s: proposed step,
        ss: rescaled proposed step,
        qpval: expected function value according to local quadratic
        approximation,
        subspace: computed subspace for reuse if proposed step is not accepted,
        steptype: type of step that was selected for proposal
    """
    sg = scaling.dot(g)
    g_dscaling = csc_matrix(np.diag(np.abs(g) * dv))

    if subspace_dim == SubSpaceDim.TWO:
        tr_step = TRStep2D(
            x, sg, hess, scaling, g_dscaling, delta, theta, ub, lb, logger
        )
    elif subspace_dim == SubSpaceDim.FULL:
        tr_step = TRStepFull(
            x, sg, hess, scaling, g_dscaling, delta, theta, ub, lb, logger
        )
    else:
        raise ValueError('Invalid choice of subspace dimension.')
    tr_step.calculate()

    # in case of truncation, we hit the boundary and we check both the
    # gradient and the reflected step, either of which could be better than the
    # TR step

    steps = [tr_step]
    if tr_step.alpha < 1.0 and len(g) > 1:
        g_step = GradientStep(x, sg, hess, scaling, g_dscaling, delta,
                              theta, ub, lb, logger)
        g_step.calculate()

        steps = [g_step]

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
        if refine_stepback:
            steps.extend(stepback_refine(
                steps, x, sg, hess, scaling, g_dscaling, delta, theta, ub,
                lb
            ))

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
    return steps[int(np.argmin(qpvals))]
