"""
Trust Region Steps
------------------
This module provides the machinery to compute different trust-region(
-reflective) step proposals and select among them based on to their
performance according to the quadratic approximation of the objective function
"""

import numpy as np
import scipy.linalg as linalg

from numpy.linalg import norm
from scipy.sparse import csc_matrix
from scipy.optimize import Bounds, NonlinearConstraint, minimize

from .logging import logger
from .subproblem import (
    solve_1d_trust_region_subproblem, solve_nd_trust_region_subproblem
)
from .constants import SubSpaceDim, StepBackStrategy

from typing import List, Sequence, Union


def normalize(v: np.ndarray) -> None:
    """
    Inplace normalization of a vector

    :param v:
        vector to be normalized
    """
    nv = norm(v)
    if nv > 0:
        v[:] = v/nv  # change inplace


def quadratic_form(Q: np.ndarray, p: np.ndarray, x: np.ndarray) -> float:
    """
    Computes the quadratic form :math:`x^TQx + x^Tp`

    :param Q: Matrix
    :param p: Vector
    :param x: Input

    :return:
        Value of form
    """
    return 0.5 * x.T.dot(Q).dot(x) + p.T.dot(x)


class Step:
    """
    Base class for the computation of a proposal step

    :ivar x: Current state of optimization variables
    :ivar s: Proposed step
    :ivar sc: Coefficients in the 1D/2D subspace that defines the affine
        transformed step ss: `ss = subspace * sc`
    :ivar ss: Affine transformed step: `s = scaling * ss`
    :ivar og_s: `s` without step back
    :ivar og_sc: `st` without step back
    :ivar og_ss: `ss` without step back
    :ivar sg: Rescaled gradient `scaling * g`
    :ivar hess: Hessian of the objective function at `x`
    :ivar g_dscaling: `diag(g) * dscaling`
    :ivar delta: Trust region radius in the transformed space defined by
        scaling matrix
    :ivar theta: Controls step back, fraction of step to take if full
        step would reach breakpoint
    :ivar lb: Lower boundaries for x
    :ivar ub: Upper boundaries for x
    :ivar minbr: Maximal fraction of step s that can be taken to reach
        first breakpoint
    :ivar iminbr: Index of x that specifies the variable that will hit the
        breakpoint if a step minbr * s is taken
    :ivar qpval: Value of the quadratic subproblem for the proposed step
    :ivar shess: Matrix of the full quadratic problem
    :ivar cg: Projection of the g_hat to the subspace
    :ivar chess: Projection of the B to the subspace
    :ivar reflection_count: Number of reflections that were applied to
        obtain this step
    :ivar truncation_count: Number of reflections that were applied to
        obtain this step

    :cvar type: Identifier that allows identification of subclasses
    """
    type = 'step'

    def __init__(self,
                 x: np.ndarray,
                 sg: np.ndarray,
                 hess: np.ndarray,
                 scaling: csc_matrix,
                 g_dscaling: csc_matrix,
                 delta: float,
                 theta: float,
                 ub: np.ndarray,
                 lb: np.ndarray):
        """

        :param x:
            Reference point
        :param sg:
            Gradient in rescaled coordinates
        :param hess:
            Hessian in unscaled coordinates
        :param scaling:
            Matrix that defines scaling transformation
        :param g_dscaling:
            Unscaled gradient multiplied by derivative of scaling
            transformation
        :param delta:
            Trust region Radius in scaled coordinates
        :param theta:
            Stepback parameter that controls how close steps are allowed to
            get to the boundary
        :param ub:
            Upper boundary
        :param lb:
            Lower boundary

        """
        self.x: np.ndarray = x

        self.s: Union[np.ndarray, None] = None
        self.sc: Union[np.ndarray, None] = None
        self.ss: Union[np.ndarray, None] = None

        self.og_s: Union[np.ndarray, None] = None
        self.og_sc: Union[np.ndarray, None] = None
        self.og_ss: Union[np.ndarray, None] = None

        self.sg: np.ndarray = sg
        self.scaling: csc_matrix = scaling

        self.delta: float = delta
        self.theta: float = theta

        self.lb: np.ndarray = lb
        self.ub: np.ndarray = ub

        self.br: np.ndarray = np.ones(sg.shape)
        self.minbr: float = 1.0
        self.alpha: float = 1.0
        self.iminbr: np.ndarray = np.array([])

        self.qpval: float = 0.0

        self.shess: np.ndarray = np.asarray(scaling * hess * scaling
                                            + g_dscaling)

        self.cg: Union[np.ndarray, None] = None
        self.chess: Union[np.ndarray, None] = None
        self.subspace: Union[np.ndarray, None] = None

        self.s0: np.ndarray = np.zeros(sg.shape)
        self.ss0: np.ndarray = np.zeros(sg.shape)

        self.reflection_count: int = 0
        self.truncation_count: int = 0

    def step_back(self):
        """
        This function truncates the step based on the distance of the
        current point to the boundary.
        """
        # create copies of the calculated step
        self.og_s = self.s.copy()
        self.og_ss = self.ss.copy()
        self.og_sc = self.sc.copy()

        nonzero = np.abs(self.s) > 0
        self.br = np.inf * np.ones(self.s.shape)
        if np.any(nonzero):
            # br quantifies the distance to the boundary normalized
            # by the proposed step, this indicates the fraction of the step
            # that would put the respective variable at the boundary
            # This is defined in [Coleman-Li1996] (3.1)
            self.br[nonzero] = np.max(np.vstack([
                (self.ub[nonzero] - self.x[nonzero])/self.s[nonzero],
                (self.lb[nonzero] - self.x[nonzero])/self.s[nonzero]
            ]), axis=0)
        self.minbr = np.min(self.br)
        self.iminbr = np.where(self.br == self.minbr)[0]
        # compute the minimum of the step
        self.alpha = np.min([1, self.theta * self.minbr])

        self.s *= self.alpha
        self.sc *= self.alpha
        self.ss *= self.alpha

    def reduce_to_subspace(self) -> None:
        """
        This function projects the matrix shess and the vector sg to the
        subspace
        """
        self.chess = self.subspace.T.dot(self.shess.dot(self.subspace))
        self.cg = self.subspace.T.dot(self.sg)

    def compute_step(self) -> None:
        """
        Compute the step as solution to the trust region subproblem. Special
        code is used for the special case 1-dimensional subspace case
        """
        if self.subspace.shape[1] == 0:
            self.sc = np.empty((0, 0))
            self.ss = np.zeros(self.ss0.shape)
            self.s = np.zeros(self.s0.shape)
            return
        if self.subspace.shape[1] > 1:
            self.sc, _ = solve_nd_trust_region_subproblem(
                self.chess, self.cg,
                np.sqrt(max(self.delta ** 2 - norm(self.ss0) ** 2, 0.0))
            )
        else:
            self.sc = solve_1d_trust_region_subproblem(
                self.shess, self.sg, self.subspace[:, 0], self.delta, self.ss0
            )
        self.ss = self.subspace.dot(np.real(self.sc))
        self.s = self.scaling.dot(self.ss)

    def calculate(self):
        """
        Calculates step and the expected objective function value according to
        the quadratic approximation
        """
        self.reduce_to_subspace()
        self.compute_step()
        self.step_back()
        self.qpval = quadratic_form(self.shess, self.sg, self.ss + self.ss0)


class TRStepFull(Step):
    """
    This class provides the machinery to compute an exact solution of
    the trust region subproblem.
    """

    type = 'trnd'

    def __init__(self, x, sg, hess, scaling, g_dscaling, delta, theta,
                 ub, lb):
        super().__init__(x, sg, hess, scaling, g_dscaling, delta, theta,
                         ub, lb)
        self.subspace = np.eye(hess.shape[0])


class TRStep2D(Step):
    """
    This class provides the machinery to compute an approximate solution of
    the trust region subproblem according to a 2D subproblem
    """

    type = 'tr2d'

    def __init__(self, x, sg, hess, scaling, g_dscaling, delta, theta,
                 ub, lb):
        super().__init__(x, sg, hess, scaling, g_dscaling, delta, theta,
                         ub, lb)
        n = len(sg)

        s_newt = linalg.solve(hess, sg)
        posdef = s_newt.dot(hess.dot(s_newt)) > 0
        normalize(s_newt)

        if n > 1:
            if not posdef:
                # in this case we are in Case 2 of Fig 12 in
                # [Coleman-Li1994]
                logger.debug('Newton direction did not have negative '
                             'curvature adding scaling * np.sign(sg) to '
                             '2D subspace.')
                s_grad = scaling * np.sign(sg) + (sg == 0)
            else:
                s_grad = sg.copy()

            # orthonormalize, this ensures that S.T.dot(S) = I and we
            # can use S/S.T for transformation
            s_grad = s_grad - s_newt * (s_newt.dot(s_grad))
            normalize(s_grad)
            # if non-zero, add s_grad to subspace
            if np.any(s_grad != 0):
                self.subspace = np.vstack([s_newt, s_grad]).T
                return
            else:
                logger.debug('Singular subspace, continuing with 1D '
                             'subspace.')

        self.subspace = np.expand_dims(s_newt, 1)


class TRStepReflected(Step):
    """
    This class provides the machinery to compute a reflected step based on
    trust region subproblem solution that hit the boundaries.
    """

    type = 'trr'

    def __init__(self, x, sg, hess, scaling, g_dscaling, delta, theta,
                 ub, lb, step: Step):
        """
        :param step:
            Trust-region step that is reflected
        """
        super().__init__(x, sg, hess, scaling, g_dscaling, delta, theta,
                         ub, lb)

        alpha = min(step.minbr, 1)
        self.s0 = alpha * step.og_s + step.s0
        self.ss0 = alpha * step.og_ss + step.ss0
        # update x and at breakpoint
        self.x = x + self.s0

        # reflect the transformed step at the boundary
        nss = step.og_ss.copy()
        nss[step.iminbr] *= -1
        normalize(nss)
        self.subspace = np.expand_dims(nss, 1)
        self.reflection_count = step.reflection_count + 1


class TRStepTruncated(Step):
    """
    This class provides the machinery to compute a reduced step based on
    trust region subproblem solution that hit the boundaries.
    """

    type = 'trt'

    def __init__(self, x, sg, hess, scaling, g_dscaling, delta, theta,
                 ub, lb, step: Step):
        """
        :param step:
            Trust-region step that is reduced
        """
        super().__init__(x, sg, hess, scaling, g_dscaling, delta, theta,
                         ub, lb)

        self.s0 = step.s0.copy()
        self.ss0 = step.ss0.copy()
        iminbr = step.iminbr
        self.s0[iminbr] += step.theta * step.br[iminbr] * step.og_s[iminbr]
        self.ss0[iminbr] += step.theta * step.br[iminbr] * step.og_ss[iminbr]
        # update x and at breakpoint
        self.x = x + self.s0

        subspace = step.subspace.copy()
        subspace[iminbr, :] = 0
        # reduce subspace
        subspace = subspace[:, (subspace != 0).any(axis=0)]
        # normalize subspace
        for ix in range(subspace.shape[1]):
            normalize(subspace[:, ix])
        self.subspace = subspace

        self.truncation_count = step.truncation_count + len(iminbr)


class GradientStep(Step):
    """
    This class provides the machinery to compute a gradient step.
    """

    type = 'g'

    def __init__(self, x, sg, hess, scaling, g_dscaling, delta, theta,
                 ub, lb):
        super().__init__(x, sg, hess, scaling, g_dscaling, delta, theta,
                         ub, lb)
        s_grad = sg.copy()
        normalize(s_grad)
        self.subspace = np.expand_dims(s_grad, 1)


class RefinedStep(Step):
    """
    This class provides the machinery to refine a step based on interior
    point optimization
    """

    type = 'ref'

    def __init__(self, x, sg, hess, scaling, g_dscaling, delta, theta,
                 ub, lb, step):
        super().__init__(x, sg, hess, scaling, g_dscaling, delta, theta,
                         ub, lb)
        s_grad = sg.copy()
        normalize(s_grad)
        self.subspace = np.expand_dims(s_grad, 1)
        self.constraints = [
            NonlinearConstraint(
                fun=lambda xs: (norm(xs) - delta) * np.ones((1,)),
                jac=lambda xs: np.expand_dims(xs, 1).T / norm(xs),
                lb=np.zeros((1,)),
                ub=np.ones((1,)) * np.inf,
            )
        ]
        self.guess = step.ss + step.ss0
        self.bounds = Bounds(
            step.theta * (lb - x) / scaling.diagonal(),
            step.theta * (ub - x) / scaling.diagonal()
        )
        self.reflection_count = step.reflection_count
        self.truncation_count = step.truncation_count

    def calculate(self):
        res = minimize(fun=lambda s: quadratic_form(self.shess, self.sg, s),
                       jac=lambda s: self.shess.dot(s) + self.sg,
                       hess=lambda s: self.shess,
                       x0=self.guess,
                       method='trust-constr',
                       bounds=self.bounds,
                       constraints=self.constraints,
                       options={'verbose': 0, 'maxiter': 10})
        self.ss = res.x
        self.s = self.scaling.dot(res.x)
        self.sc = self.ss
        self.step_back()
        self.qpval = quadratic_form(self.shess, self.sg, self.ss)


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
                 refine_stepback: bool) -> Step:
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
            x, sg, hess, scaling, g_dscaling, delta, theta, ub, lb
        )
    elif subspace_dim == SubSpaceDim.FULL:
        tr_step = TRStepFull(
            x, sg, hess, scaling, g_dscaling, delta, theta, ub, lb,
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
                              theta, ub, lb)
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
