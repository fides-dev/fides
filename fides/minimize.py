"""
Minimization
------------
This module performs the optimization given a step proposal.
"""

import time

import numpy as np
import logging
from numpy.linalg import norm
from scipy.sparse import csc_matrix
from .trust_region import trust_region_reflective
from .hessian_approximation import HessianApproximation
from .defaults import MAXITER
from .logging import logger

from typing import Callable, Dict, Optional, Tuple


class Optimizer:
    """
    Performs optimization

    :ivar fun: objective function
    :ivar funargs: keyword arguments that are passed to the function
    :ivar lb: lower optimization boundaries
    :ivar ub: upper optimization boundaries
    :ivar options: options that configure convergence checks
    :ivar delta: trust region radius
    :ivar x: current optimization variables
    :ivar fval: objective function value at x
    :ivar grad: objective function gradient at x
    :ivar hess: objective function Hessian (approximation) at x
    :ivar hessian_update: object that performs hessian updates
    :ivar starttime: time at which optimization was started
    :ivar iteration: current iteration
    :ivar converged: flag indicating whether optimization has converged
    """
    def __init__(self, fun: Callable,
                 ub: np.ndarray,
                 lb: np.ndarray,
                 verbose: Optional[int] = logging.DEBUG,
                 options: Optional[Dict] = None,
                 funargs: Optional[Dict] = None,
                 hessian_update: Optional[HessianApproximation] = None):
        """
        Create an optimizer object

        :param fun:
            This is the objective function, if no `hessian_update` is
            provided, this function must return a tuple (fval, grad),
            otherwise this function must return a tuple (fval, grad, Hessian)
        :param ub:
            Upper optimization boundaries. Individual entries can be set to
            np.inf for respective variable to have no upper bound
        :param lb:
            Lower optimization boundaries. Individual entries can be set to
            -np.inf for respective variable to have no lower bound
        :param verbose:
            Verbosity level, pick from logging.[DEBUG,INFO,WARNING,ERROR]
        :param options:
            Options that control termination of optimization.
            See `minimize` for details.
        :param funargs:
            Additional keyword arguments that are to be passed to fun for
            evaluation
        :param hessian_update:
            Subclass of :py:class:`HessianApproximation` that performs the
            hessian update strategy.
        """
        self.fun: Callable = fun
        if funargs is None:
            funargs = {}
        self.funargs: Dict = funargs

        self.lb: np.ndarray = np.array(lb)
        self.ub: np.ndarray = np.array(ub)

        if options is None:
            options = {}

        self.options: Dict = options

        self.delta = 10

        self.x: np.ndarray = np.empty(ub.shape)
        self.fval: float = np.nan
        self.grad: np.ndarray = np.empty(ub.shape)
        self.hess: np.ndarray = np.empty((ub.shape[0], ub.shape[0]))

        self.hessian_update: HessianApproximation = hessian_update

        self.starttime: float = np.nan
        self.iteration: int = 0
        self.converged: bool = False
        logger.setLevel(verbose)

    def minimize(self, x0: np.ndarray):
        """
        Minimize the objective function the interior trust-region reflective
        algorithm described by [ColemanLi1994] and [ColemanLi1996]
        Convergence with respect to function value is achieved when
        :math:`|f_{k+1} - f_k_|` < options[`fatol`] - :math:`f_k` options[
        `frtol`]. Similarly,  convergence with respect to optimization
        variables is achieved when :math:`||x_{k+1} - x_k||` < options[
        `xatol`] - :math:`x_k` options[`xrtol`].  Convergence with respect
        to the gradient is achieved when :math:`||g_k||` <
        options[`gtol`].  Other than that, optimization can be terminated
        when iterations exceed options[`maxiter`] or the elapsed time is
        expected to exceed options[`maxtime`].

        :param x0:
            initial guess

        :returns:
            final function value, final optimization variable values,
            final gradient, final Hessian (approximation)
        """
        self.starttime = time.time()
        self.iteration = 0

        self.x = np.array(x0).copy()
        self.make_non_degenerate()
        self.check_in_bounds()
        if self.hessian_update is None:
            self.fval, self.grad, self.hess = self.fun(self.x, **self.funargs)
        else:
            self.fval, self.grad = self.fun(self.x, **self.funargs)
            self.hess = self.hessian_update.get_mat()

        self.check_finite()

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
                fval_new, grad_new, hess_new = self.fun(x_new, **self.funargs)
            else:
                fval_new, grad_new = self.fun(x_new, **self.funargs)

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
                self.check_in_bounds(x_new)
                self.fval = fval_new
                self.x = x_new
                self.grad = grad_new
                self.check_finite()
                self.make_non_degenerate()

                tr_space = None

        return self.fval, self.x, self.grad, self.hess

    def update_tr_radius(self, fval, grad, step_sx, dv, qppred) -> bool:
        """
        Update the trust region radius

        :param fval:
            new function value if step defined by step_sx is taken
        :param grad:
            new gradient value if step defined by step_sx is taken
        :param step_sx:
            proposed scaled step
        :param dv:
            derivative of scaling vector v wrt x
        :param qppred:
            predicted objective function value according to the quadratic
            approximation

        :return:
            flag indicating whether the proposed step should be accepted
        """
        nsx = norm(step_sx)
        if not np.isfinite(fval):
            self.delta = np.min([self.delta / 4, nsx / 4])
            return False
        else:
            qpval = 0.5 * step_sx.dot(dv * np.abs(grad) * step_sx)
            ratio = (fval + qpval - self.fval) / qppred

            # values as proposed in algorithm 4.1 in Nocedal & Wright
            if ratio >= 0.75 and nsx > self.delta * 0.9:
                self.delta = 2 * self.delta
            elif ratio <= .25 or nsx < self.delta * 0.9 \
                    or fval > self.fval * 1.1:
                self.delta = np.min([self.delta / 4, nsx / 4])
            return ratio >= .25 and fval < self.fval * 1.1

    def check_convergence(self, fval, x, grad) -> None:
        """
        Check whether optimization has converged.

        :param fval:
            updated objective function value
        :param x:
            updated optimization variables
        :param grad:
            updated objective function gradient
        """
        converged = False

        fatol = self.options.get('fatol', 1e-6)
        frtol = self.options.get('frtol', 0)
        xatol = self.options.get('xatol', 0)
        xrtol = self.options.get('xrtol', 0)
        gtol = self.options.get('gtol', np.sqrt(np.spacing(1)))
        gnorm = norm(grad)

        if np.isclose(fval, self.fval, atol=fatol, rtol=frtol):
            logger.info(
                f'Stopping as function difference '
                f'{np.abs(self.fval - fval)} was smaller than specified '
                f'tolerances (atol={fatol:.2E}, rtol={frtol:.2E})'
            )
            converged = True

        elif np.isclose(x, self.x, atol=xatol, rtol=xrtol).all():
            logger.info(
                f'Stopping as step was smaller than specified tolerances ('
                f'atol={xatol:.2E}, rtol={xrtol:.2E})'
            )
            converged = True

        elif gnorm <= gtol:
            logger.info(
                f'Stopping as gradient norm satisfies convergence criteria: '
                f'{gnorm:.2E} < {gtol:.2E}'
            )
            converged = True

        self.converged = converged

    def check_continue(self) -> bool:
        """
        Checks whether minimization should continue based on convergence,
        iteration count and remaining computational budget

        :return:
            flag indicating whether minimization should continue
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

        if self.delta < np.spacing(1):
            logger.error(
                'Stopping as trust region radius is smaller that machine '
                'precision.'
            )
            return False

        return True

    def make_non_degenerate(self, eps=1e2 * np.spacing(1)) -> None:
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

    def get_affine_scaling(self) -> Tuple[np.ndarray, np.ndarray]:
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
        """
        Prints diagnostic information about the current step to the log

        :param accepted:
            flag indicating whether the current step was accepted
        :param steptype:
            identifier how the current step was computed
        :param normdx:
            norm of the current step
        """
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
        """
        Prints the header for diagnostic information, should complement
        :py:func:`Optimizer.log_step`.
        """
        iterspaces = len(str(self.options.get('maxiter', MAXITER))) - 5
        logger.info(f'{" " * iterspaces} iter '
                    f'|    fval    |  delta   | ||step|| |  ||g||   '
                    f'| step | '
                    f'accept')

    def check_finite(self):
        """
        Checks whether objective function value, gradient and Hessian (
        approximation) have finite values and optimization can continue.

        :raises:
            RuntimeError if any of the variables have non-finite entries
        """

        if self.iteration == 0:
            pointstr = 'at initial point.'
        else:
            pointstr = f'at iteration {self.iteration}.'

        if not np.isfinite(self.fval):
            raise RuntimeError(f'Encountered non-finite function {self.fval} '
                               f'value {pointstr}')

        if not np.isfinite(self.grad).all():
            ix = np.where(np.logical_not(np.isfinite(self.grad)))
            raise RuntimeError('Encountered non-finite gradient entries'
                               f' {self.grad[ix]} for indices {ix} '
                               f'{pointstr}')

        if not np.isfinite(self.hess).all():
            ix = np.where(np.logical_not(np.isfinite(self.hess)))
            raise RuntimeError('Encountered non-finite gradient hessian'
                               f' {self.hess[ix]} for indices {ix} '
                               f'{pointstr}')

    def check_in_bounds(self, x: Optional[np.ndarray] = None):
        """
        Checks whether the current optimization variables are all within the
        specified boundaries

        :raises:
            RuntimeError if any of the variables are not within boundaries
        """
        if x is None:
            x = self.x

        if self.iteration == 0:
            pointstr = 'at initial point.'
        else:
            pointstr = f'at iteration {self.iteration}.'

        for ref, sign, name in zip([self.ub, self.lb],
                                   [-1.0, 1.0],
                                   ['upper bounds', 'lower bounds']):
            diff = sign * (ref - x)
            if not np.all(diff <= 0):
                ix = np.where(diff > 0)
                raise RuntimeError(f'Exceeded upper bounds for indices {ix} by'
                                   f'{diff[ix]} {pointstr}')
