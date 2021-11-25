"""
Trust Region Step Calculation
-----------------------------
This module provides the machinery to calculate different trust-region(
-reflective) step proposals
"""


import numpy as np
import scipy.linalg as linalg

from numpy.linalg import norm
from scipy.sparse import csc_matrix
from scipy.optimize import NonlinearConstraint, LinearConstraint, minimize

from logging import Logger
from .subproblem import (
    solve_1d_trust_region_subproblem, solve_nd_trust_region_subproblem,
    get_1d_trust_region_boundary_solution, quadratic_form
)

from typing import Union


def normalize(v: np.ndarray) -> None:
    """
    Inplace normalization of a vector

    :param v:
        vector to be normalized
    """
    nv = norm(v)
    if nv > 0:
        v[:] = v/nv  # change inplace


class Step:
    """
    Base class for the computation of a proposal step

    :ivar x: Current state of optimization variables
    :ivar s: Proposed step
    :ivar sc: Coefficients in the 1D/2D subspace that defines the affine
        transformed step ss: `ss = subspace * sc`
    :ivar ss: Affine transformed step: `s = scaling * ss`
    :ivar og_s: `s` without step back
    :ivar og_sc: `sc` without step back
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
    :ivar reflection_indices: Indices of variables for which reflection was
        applied
    :ivar truncation_indices: Indices of variables for which truncation was
        applied

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
                 lb: np.ndarray,
                 logger: Logger):
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

        self.sg: np.ndarray = sg.copy()
        self.scaling: csc_matrix = scaling.copy()

        self.delta: float = delta
        self.theta: float = theta

        self.lb: np.ndarray = lb
        self.ub: np.ndarray = ub

        self.br: np.ndarray = np.ones(sg.shape)
        self.minbr: float = 1.0
        self.alpha: float = 1.0
        self.iminbr: np.ndarray = np.array([])

        self.qpval: float = 0.0

        # B_hat (Eq 2.5) [ColemanLi1996]
        self.shess: np.ndarray = np.asarray(scaling * hess * scaling
                                            + g_dscaling)

        self.cg: Union[np.ndarray, None] = None
        self.chess: Union[np.ndarray, None] = None
        self.subspace: Union[np.ndarray, None] = None

        self.s0: np.ndarray = np.zeros(sg.shape)
        self.ss0: np.ndarray = np.zeros(sg.shape)

        self.reflection_indices: set = set()
        self.truncation_indices: set = set()
        self.logger: Logger = logger

    @property
    def reflection_count(self) -> int:
        """
        Number of reflections that were applied to obtain this step
        :return:
            Number of reflections
        """
        return len(self.reflection_indices)

    @property
    def truncation_count(self) -> int:
        """
        Number of truncations that were applied to obtain this step
        :return:
            Number of truncations
        """
        return len(self.truncation_indices)

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
            # This is defined in [Coleman-Li1994] (3.1)
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
                np.sqrt(max(self.delta ** 2 - norm(self.ss0) ** 2, 0.0)),
                self.logger
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

    type = 'nd'

    def __init__(self, x, sg, hess, scaling, g_dscaling, delta, theta,
                 ub, lb, logger):
        super().__init__(x, sg, hess, scaling, g_dscaling, delta, theta,
                         ub, lb, logger)
        self.subspace = np.eye(hess.shape[0])


class TRStep2D(Step):
    """
    This class provides the machinery to compute an approximate solution of
    the trust region subproblem according to a 2D subproblem
    """

    type = '2d'

    def __init__(self, x, sg, hess, scaling, g_dscaling, delta, theta,
                 ub, lb, logger):
        super().__init__(x, sg, hess, scaling, g_dscaling, delta, theta,
                         ub, lb, logger)
        s_newt = - linalg.lstsq(self.shess, sg)[0]
        # lstsq only returns absolute ev values
        e, v = np.linalg.eig(self.shess)
        self.posdef = np.min(np.real(e)) > - np.spacing(1) * np.max(np.abs(e))

        if len(sg) == 1:
            s_newt = - sg[0]/self.shess[0]
            self.subspace = np.expand_dims(s_newt, 1)
            return

        self.newton = False

        if self.posdef:
            s_newt = - linalg.lstsq(self.shess, sg)[0]

            if norm(s_newt) < delta:
                # Case 0 of Fig 12 in [ColemanLi1994]
                normalize(s_newt)
                self.newton = True
                self.subspace = np.expand_dims(s_newt, 1)
                return

            # Case 1 of Fig 12 in [ColemanLi1994]
            s_grad = sg.copy()
            # orthonormalize, this ensures that S.T.dot(S) = I and we
            # can use S/S.T for transformation
        else:
            # Case 2 of Fig 12 in [ColemanLi1994]
            # Eigenvectors to negative eigenvalues are constraint
            # compatible according to Theorem 5 (3)
            s_newt = np.real(v[:, np.argmin(np.real(e))])
            s_grad = scaling.dot(np.sign(sg) + (sg == 0))

        normalize(s_newt)
        s_grad = s_grad - s_newt * s_newt.dot(s_grad)
        # if non-zero, add s_grad to subspace
        if norm(s_grad) > np.spacing(1):
            normalize(s_grad)
            self.subspace = np.vstack([s_newt, s_grad]).T
        else:
            # s_newt and s_grad are parallel but we already projected
            # s_grad, so use s_newt here
            self.subspace = np.expand_dims(s_newt, 1)


class CGStep(Step):
    """
    This class provides the machinery to compute an approximate solution of
    the trust region subproblem using the conjugate gradients methods
    """

    type = 'cg'

    def calculate(self):
        nsg = norm(self.sg)
        self.conj_grad(min(0.5, np.sqrt(nsg)) * nsg)
        self.s = self.scaling.dot(self.ss + self.ss0)
        self.step_back()
        self.qpval = quadratic_form(self.shess, self.sg, self.ss + self.ss0)

    def conj_grad(self, eps):
        """
        Compute step proposal using conjugate gradient method

        :param eps:
            tolerance for residual norm
        """
        raise NotImplementedError()


class TRStepSteihaug(CGStep):
    """
    This class provides the machinery to compute an approximate solution of
    the trust region subproblem using Steihaug's Method
    """

    type = 'cgs'

    def conj_grad(self, eps):
        z = np.zeros_like(self.sg)
        r = self.sg.copy()
        d = -self.sg.copy()
        if norm(r) < eps:
            self.ss = z
            return

        while True:
            bd = self.shess.dot(d)
            c = d.dot(bd)
            r2 = np.dot(r, r)
            alpha = r2 / c
            zp = z + alpha * d
            if c <= 0 or norm(zp) >= self.delta:
                self.subspace = np.expand_dims(d, 1)
                self.ss0 = z
                self.sc = get_1d_trust_region_boundary_solution(
                    self.shess, self.sg, self.subspace[:, 0], self.ss0,
                    self.delta
                ) * np.ones((1,))
                self.ss = self.subspace.dot(self.sc)
                return
            rp = r + alpha*bd
            rp2 = np.dot(rp, rp)
            if np.sqrt(rp2) < eps:
                normalize(d)
                self.subspace = np.expand_dims(d, 1)
                self.sc = zp.dot(d) * np.ones((1,))
                self.ss = self.subspace.dot(self.sc)
                self.ss0 = zp - self.ss
                return
            beta = rp2 / r2

            d = -rp + beta * d
            z = zp.copy()
            r = rp.copy()


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
                         ub, lb, step.logger)

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

        for iminbr in step.iminbr:
            self.reflection_indices.add(iminbr)


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
                         ub, lb, step.logger)

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

        for iminbr in step.iminbr:
            self.truncation_indices.add(iminbr)


class GradientStep(Step):
    """
    This class provides the machinery to compute a gradient step.
    """

    type = 'g'

    def __init__(self, x, sg, hess, scaling, g_dscaling, delta, theta,
                 ub, lb, logger):
        super().__init__(x, sg, hess, scaling, g_dscaling, delta, theta,
                         ub, lb, logger)
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
                         ub, lb, step.logger)
        self.subspace: np.ndarray = step.subspace.copy()
        self.chess: np.ndarray = step.chess.copy()
        self.cg: np.ndarray = step.cg.copy()
        self.constraints = [
            NonlinearConstraint(
                fun=lambda xc: (norm(self.subspace.dot(xc)) - delta) *
                np.ones((1,)),
                jac=lambda xc:
                np.expand_dims(self.subspace.dot(xc), 1).T.dot(self.subspace) /
                norm(self.subspace.dot(xc)),
                lb=-np.ones((1,)) * np.inf,
                ub=np.zeros((1,)),
            ),
            LinearConstraint(
                A=self.subspace,
                lb=self.theta * (lb - x) / scaling.diagonal(),
                ub=self.theta * (ub - x) / scaling.diagonal()
            )
        ]
        self.guess: np.ndarray = step.sc.copy()
        self.qpval0: float = step.qpval
        self.reflection_indices: int = step.reflection_indices
        self.truncation_indices: int = step.truncation_indices

    def calculate(self):
        res = minimize(fun=lambda c: quadratic_form(self.chess, self.cg, c),
                       jac=lambda c: self.chess.dot(c) + self.cg,
                       hess=lambda c: self.chess,
                       x0=self.guess,
                       method='trust-constr',
                       constraints=self.constraints,
                       options={'verbose': 0, 'maxiter': 100,
                                'gtol': 0, 'xtol': 0})
        self.sc = res.x
        self.ss = self.subspace.dot(self.sc)
        self.s = self.scaling.dot(self.ss)
        self.step_back()
        self.qpval = quadratic_form(self.shess, self.sg, self.ss)
