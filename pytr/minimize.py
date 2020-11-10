import time

import numpy as np
import logging
from numpy.linalg import norm
from scipy.sparse import csc_matrix
from .trust_region import trust_region_reflective
from .hessian_approximation import HessianApproximation
from .defaults import MAXITER
from .logging import logger

from typing import Callable, Dict, Optional


class Optimizer:
    def __init__(self, fun: Callable,
                 ub: np.ndarray,
                 lb: np.ndarray,
                 verbose: Optional[int] = logging.DEBUG,
                 options: Optional[Dict] = None,
                 hessian_update: Optional[HessianApproximation] = None):
        self.fun = fun
        self.lb = lb
        self.ub = ub

        if options is None:
            self.options = {}
        else:
            self.options = options

        self.delta = 10

        self.x = np.empty(ub.shape)
        self.fval = np.nan
        self.grad = np.empty(ub.shape)
        self.hess = np.empty((ub.shape[0], ub.shape[0]))

        self.hessian_update = hessian_update

        self.starttime = np.nan
        self.iteration = 0
        self.converged = False
        logger.setLevel(verbose)

    def minimize(self, x0: np.ndarray):
        """
        Minimize the objective function

        :param x0: initial guess

        :return:
            fval: final function value
            x: final optimization variable values
            grad: final gradient
            hess: final Hessian (approximation)
        """
        self.starttime = time.time()
        self.iteration = 0

        self.x = np.array(x0).copy()
        self.make_non_degenerate()
        if self.hessian_update is None:
            self.fval, self.grad, self.hess = self.fun(self.x)
        else:
            self.fval, self.grad = self.fun(self.x)
            self.hess = self.hessian_update.get_mat()

        tr_space = None

        self.converged = False

        while self.check_continue():
            self.iteration += 1

            v, dv = self.get_affine_scaling()

            scaling = csc_matrix(np.diag(np.sqrt(np.abs(v))))
            theta = max(.95, 1 - norm(v * self.grad, np.inf))

            step_x, step_sx, qppred, tr_space, step_type = \
                trust_region_reflective(
                    self.x, self.grad, self.hess, scaling, tr_space,
                    self.delta, dv, theta, self.lb, self.ub
                )

            x_new = self.x + step_x

            if self.hessian_update is None:
                fval_new, grad_new, hess_new = self.fun(x_new)
            else:
                fval_new, grad_new = self.fun(x_new)

            accepted = self.update_tr_radius(
                fval_new, grad_new, step_sx, dv, qppred
            )

            if (self.iteration - 1) % 10 == 0:
                self.log_header()
            self.log_step(accepted, step_type, norm(step_x))
            self.check_convergence(fval_new, x_new, grad_new)

            if accepted:
                if self.hessian_update is not None:
                    self.hessian_update.update(step_x, grad_new - self.grad)
                    self.hess = self.hessian_update.get_mat()
                else:
                    self.hess = hess_new
                self.fval = fval_new
                self.x = x_new
                self.grad = grad_new
                tr_space = None

        return self.fval, self.x, self.grad, self.hess

    def update_tr_radius(self, fval, grad, step_sx, dv, qppred) -> bool:
        nsx = norm(step_sx)
        if not np.isfinite(fval):
            self.delta = np.min([self.delta / 4, nsx / 4])
            return False
        else:
            qpval = step_sx.dot(dv * np.abs(grad) * step_sx) / 2
            ratio = (fval + qpval - self.fval) / qppred

            # values as proposed in algorithm 4.1 in Nocedal & Wright
            if ratio >= 0.75:
                self.delta = 2 * self.delta
            elif ratio <= .25 or nsx < self.delta * 0.9:
                self.delta = np.min([self.delta / 4, nsx / 4])
            return ratio >= .25

    def check_convergence(self, fval, x, grad):
        converged = False

        fatol = self.options.get('fatol', 0)
        frtol = self.options.get('frtol', 0)
        xatol = self.options.get('xatol', 0)
        xrtol = self.options.get('xrtol', 0)
        gtol = self.options.get('gtol', 1e-6)
        gnorm = norm(grad)

        if np.isclose(fval, self.fval, atol=fatol, rtol=frtol):
            logger.info(
                f'Stopping as function difference '
                f'{np.abs(self.fval - fval)} was smaller than specified '
                f'tolerances (atol={fatol}, rtol={frtol})'
            )
            converged = True

        elif np.isclose(x, self.x, atol=xatol, rtol=xrtol).all():
            logger.info(
                f'Stopping as step was smaller than specified tolerances ('
                f'atol={xatol}, rtol={xrtol})'
            )
            converged = True

        elif gnorm <= gtol:
            logger.info(
                f'Stopping as gradient satisfies convergence criteria: '
                f'||g|| < {gtol}'
            )
            converged = True

        self.converged = converged

    def check_continue(self):
        """
        Checks whether minimization should continue based on convergence,
        iteration count and remaining computational budget
        """

        if self.converged:
            return False

        maxiter = self.options.get('maxiter', MAXITER)
        if self.iteration > maxiter:
            logger.error(
                f'Stopping as maximum number of iterations {maxiter} was '
                f'exceeded.'
            )
            return False

        time_elapsed = time.time() - self.starttime
        maxtime = self.options.get('maxtime', np.inf)
        time_remaining = maxtime - time_elapsed
        avg_iter_time = time_elapsed/(self.iteration + (self.iteration == 0))
        if time_remaining < avg_iter_time:
            logger.error(
                f'Stopping as maximum runtime {maxtime} is expected to be '
                f'exceeded in the next iteration.'
            )
            return False

        if self.delta < np.sqrt(np.spacing(1)):
            logger.error(
                'Stopping as trust region radius is smaller that machine '
                'precision.'
            )
            return False

        return True

    def make_non_degenerate(self, eps=1e2 * np.spacing(1)):
        """
        Ensures that x is non-degenerate, this should only be necessary for
        initial points.

        :param eps: degeneracy threshold
        """
        if np.min(np.abs(self.ub - self.x)) < eps or \
                np.min(np.abs(self.x - self.lb)) < eps:
            upperi = (self.ub - self.x) < eps
            loweri = (self.x - self.lb) < eps
            self.x[upperi] = self.x[upperi] - eps
            self.x[loweri] = self.x[loweri] + eps

    def get_affine_scaling(self):
        """
        Computes the vector v and dv, the diagonal of it's Jacobian. For the
        definition of v, see Definition 2 in [Coleman-Li1994]

        :return:
            v scaling vector
            dv diagonal of the Jacobian of v wrt x
        """
        # this implements no scaling for variables that are not constrained by
        # bounds ((iii) and (iv) in Definition 2)
        v = np.sign(self.grad) + (self.grad == 0)
        dv = np.zeros(self.x.shape)

        # this implements scaling for variables that are constrained by
        # bounds ( i and ii in Definition 2) bounds is equal to lb if grad <
        # 0 ub if grad >= 0
        bounds = ((1 + v)*self.lb + (1 - v)*self.ub)/2
        bounded = np.isfinite(bounds)
        v[bounded] = self.x[bounded] - bounds[bounded]
        dv[bounded] = 1
        return v, dv

    def log_step(self, accepted: bool, steptype: str, normdx: float):
        iterspaces = max(len(str(self.options.get('maxiter', MAXITER))), 5) - \
            len(str(self.iteration))
        steptypespaces = 4 - len(steptype)
        logger.info(f'{" " * iterspaces}{self.iteration}'
                    f' | {self.fval:.4E}'
                    f' | {self.delta:.2E}'
                    f' | {norm(self.grad):.2E}'
                    f' | {normdx:.2E}'
                    f' | {steptype}{" " * steptypespaces}'
                    f' | {int(accepted)}')

    def log_header(self):
        iterspaces = len(str(self.options.get('maxiter', MAXITER))) - 5
        logger.info(f'{" " * iterspaces} iter '
                    f'|    fval    |  delta   | ||step|| |  ||g||   '
                    f'| step | '
                    f'accept')
