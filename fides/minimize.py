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
from .trust_region import trust_region, Step
from .hessian_approximation import (
    HessianApproximation, StructuredApproximation, HybridFixed, FX,
    IterativeHessianApproximation, TSSM, GNSBFGS
)
from .constants import Options, ExitFlag, DEFAULT_OPTIONS
from .logging import create_logger

from typing import Callable, Dict, Optional, Tuple, Union


class FunEvaluator:
    def __init__(self, fun: Callable, nargout: int, resfun: bool,
                 funargs: dict):
        self.fun = fun
        self.nargout = nargout
        self.resfun = resfun
        if funargs is None:
            funargs = {}
        self.funargs = funargs

    def __call__(self, x: np.ndarray):
        ret = self.fun(x, **self.funargs)

        if not isinstance(ret, tuple) or len(ret) != self.nargout:
            nargout = len(ret) if isinstance(ret, tuple) else 1
            raise ValueError(f'Provided function returned {nargout} values, '
                             f'but was expected to return {self.nargout}.'
                             f'Please make sure the provided function is '
                             f'compatible with the employed Hessian '
                             f'Approximation Scheme. If no Hessian '
                             f'Approximation Scheme is employed, the function '
                             f'needs to return 3 values (fval, grad, hess).')

        if self.resfun:
            res, sres = ret
            return Funout(fval=0.5 * res.T.dot(res), grad=res.T.dot(sres),
                          hess=sres.T.dot(sres), res=res, sres=sres, x=x)
        else:
            if self.nargout == 3:
                fval, grad, hess = ret
                return Funout(fval=fval, grad=grad, hess=hess, x=x)
            else:
                fval, grad = ret
                return Funout(fval=fval, grad=grad, x=x)


class Funout:
    def __init__(self, fval: float, grad: np.ndarray, x: np.ndarray,
                 hess: Optional[np.ndarray] = None,
                 res: Optional[np.ndarray] = None,
                 sres: Optional[np.ndarray] = None,):
        self.fval = fval
        self.grad = grad
        self.hess = hess
        self.x = x
        self.res = res
        self.sres = sres

    def checkdims(self):
        if not np.isscalar(self.fval):
            raise ValueError('Provided objective function must return a '
                             'scalar!')
        if not self.grad.ndim == 1:
            raise ValueError('Provided objective function must return a '
                             'gradient vector with x.ndim == 1, was '
                             f'{self.grad.ndim}!')
        if not len(self.grad) == len(self.x):
            raise ValueError('Provided objective function must return a '
                             'gradient vector of the same shape as x, '
                             f'x has {len(self.x)} entries but gradient has '
                             f'{len(self.grad)}!')

        if self.hess is None:
            return

        if not self.hess.ndim == 2:
            raise ValueError('Provided objective function must return a '
                             'Hessian matrix with x.ndim == 2, was '
                             f'{self.hess.ndim}!')

        if not self.hess.shape[0] == self.hess.shape[1]:
            raise ValueError('Provided objective function must return a '
                             'square Hessian matrix!')

        if not self.hess.shape[0] == len(self.x):
            raise ValueError('Provided objective function must return a '
                             'square Hessian matrix with same dimension as x. '
                             f'x has {len(self.x)} entries but Hessian has '
                             f'{self.hess.shape[0]}!')

    def __repr__(self):
        return f'Funout(fval={self.fval}, grad={self.grad}, hess={self.hess})'


class Optimizer:
    """
    Performs optimization

    :ivar fevaler: FunctionEvaluator instance
    :ivar lb: Lower optimization boundaries
    :ivar ub: Upper optimization boundaries
    :ivar options: Options that configure convergence checks
    :ivar delta_iter: Trust region radius that was used for the current step
    :ivar delta: Updated trust region radius
    :ivar x: Current optimization variables
    :ivar fval: Objective function value at x
    :ivar grad: Objective function gradient at x
    :ivar x_min: Optimal optimization variables
    :ivar fval_min: Objective function value at x_min
    :ivar grad_min: Objective function gradient at x_min
    :ivar hess: Objective function Hessian (approximation) at x
    :ivar hessian_update: Object that performs hessian updates
    :ivar starttime: Time at which optimization was started
    :ivar iteration: Current iteration
    :ivar converged: Flag indicating whether optimization has converged
    :ivar exitflag: ExitFlag to indicate reason for termination
    :ivar verbose: Verbosity level for logging
    :ivar logger: logger instance
    """
    def __init__(self,
                 fun: Callable,
                 ub: np.ndarray,
                 lb: np.ndarray,
                 verbose: Optional[int] = logging.DEBUG,
                 options: Optional[Dict] = None,
                 funargs: Optional[Dict] = None,
                 hessian_update: Optional[HessianApproximation] = None,
                 resfun: bool = False):
        """
        Create an optimizer object

        :param fun:
            This is the objective function, if no `hessian_update` is
            provided, this function must return a tuple (fval, grad),
            otherwise this function must return a tuple (fval, grad, Hessian).
            If the argument resfun is True, this function must return a tuple
            (res, sres) instead, where `sres` is the derivative of res.
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
            Subclass of :py:class:`fides.hessian_update.HessianApproximation`
            that performs the hessian updates in every iteration.
        :param resfun:
            Boolean flag indicating whether fun returns function values
            (False, default) or residuals (True).
        """
        nargout = 3 if hessian_update is None or (
            hessian_update.requires_hess and not hessian_update.requires_resfun
        ) else 2

        self.fevaler = FunEvaluator(fun=fun, nargout=nargout, resfun=resfun,
                                    funargs=funargs)

        if hessian_update is not None and \
                resfun != hessian_update.requires_resfun:
            raise ValueError(f'Hessian update scheme {type(hessian_update)} '
                             f'requires an objective function that returns '
                             f'(residual, residual derivative). Please make'
                             f'sure that is the case and then call this '
                             f'function with argument resfun set to `True`.')

        self.lb: np.ndarray = np.array(lb)
        self.ub: np.ndarray = np.array(ub)

        if options is None:
            options = {}

        for option in options:
            try:
                Options(option)
            except ValueError:
                raise ValueError(f'{option} is not a valid options field.')

        self.options: Dict = options

        self.delta: float = self.get_option(Options.DELTA_INIT)
        self.delta_iter: float = self.delta

        self.tr_ratio: float = 1

        self.x: np.ndarray = np.empty(ub.shape)
        self.fval: float = np.inf
        self.grad: np.ndarray = np.empty(ub.shape)
        self.hess: np.ndarray = np.empty((ub.shape[0], ub.shape[0]))
        self.x_min = self.x
        self.fval_min = self.fval
        self.grad_min = self.grad

        self.hessian_update: Union[HessianApproximation, None] = hessian_update
        self.iterations_since_tr_update: int = 0
        self.hybrid_switch: bool = False

        self.starttime: float = np.nan
        self.iteration: int = 0
        self.converged: bool = False
        self.exitflag: ExitFlag = ExitFlag.DID_NOT_RUN
        self.verbose: int = verbose
        self.logger: Union[logging.Logger, None] = None

    def _reset(self):
        self.starttime = time.time()
        self.iteration = 0
        self.converged = False
        self.delta = self.get_option(Options.DELTA_INIT)
        self.delta_iter = self.delta
        self.fval_min = np.inf
        self.logger = create_logger(self.verbose)
        self.hybrid_switch = False

    def minimize(self, x0: np.ndarray):
        """
        Minimize the objective function using the interior trust-region
        reflective algorithm described by [ColemanLi1994] and [ColemanLi1996]
        Convergence with respect to function value is achieved when
        math:`|f_{k+1} - f_k|` < options[`fatol`] - :math:`f_k` options[
        `frtol`]. Similarly, convergence with respect to optimization
        variables is achieved when :math:`||x_{k+1} - x_k||` < options[
        `xtol`] :math:`x_k` (note that this is checked in transformed
        coordinates that account for distance to boundaries).  Convergence
        with respect to the gradient is achieved when :math:`||g_k||` <
        options[`gatol`] or `||g_k||` < options[`grtol`] * `f_k`. Other than
        that, optimization can be terminated when iterations exceed
        options[ `maxiter`] or the elapsed time is expected to exceed
        options[`maxtime`] on the next iteration.

        :param x0:
            initial guess

        :returns:
            fval: final function value,
            x: final optimization variable values,
            grad: final gradient,
            hess: final Hessian (approximation)
        """
        self._reset()

        self.x = np.array(x0).copy()
        if self.x.ndim > 1:
            raise ValueError('x0 must be a vector with x.ndim == 1!')
        self.make_non_degenerate()
        self.check_in_bounds()

        funout = self.fevaler(self.x)

        self.fval, self.grad = funout.fval, funout.grad
        if self.hessian_update is not None:
            self.hessian_update.init_mat(len(self.x), funout.hess)
            self.hess = self.hessian_update.get_mat()
        else:
            self.hess = funout.hess

        funout.checkdims()

        self.track_minimum(funout)
        self.log_header()
        self.log_step_initial()

        self.check_finite(funout)

        self.converged = False

        while self.check_continue():
            self.iteration += 1
            self.delta_iter = self.delta

            v, dv = self.get_affine_scaling()

            scaling = csc_matrix(np.diag(np.sqrt(np.abs(v))))
            theta = max(self.get_option(Options.THETA_MAX),
                        1 - norm(v * self.grad, np.inf))

            self.check_finite()

            step = \
                trust_region(
                    self.x, self.grad, self.hess, scaling,
                    self.delta_iter, dv, theta, self.lb, self.ub,
                    subspace_dim=self.get_option(Options.SUBSPACE_DIM),
                    stepback_strategy=self.get_option(Options.STEPBACK_STRAT),
                    refine_stepback=self.get_option(Options.REFINE_STEPBACK),
                    use_scaled_gradient=self.get_option(
                        Options.SCALED_GRADIENT
                    ),
                    logger=self.logger
                )

            x_new = self.x + step.s + step.s0
            funout_new = self.fevaler(x_new)

            if np.isfinite(funout_new.fval):
                self.check_finite(funout_new)

            accepted = self.update_tr_radius(funout_new, step, dv)

            if self.iteration % 10 == 0:
                self.log_header()
            self.log_step(accepted, step, funout_new)
            self.check_convergence(step, funout_new)

            # track minimum independently of whether we accept the step or not
            self.track_minimum(funout_new)

            if accepted:
                self.update(step, funout_new, funout)
                funout = funout_new

        return self.fval, self.x, self.grad, self.hess

    def track_minimum(self, funout: Funout) -> None:
        """
        Function that tracks the optimization variables that have minimal
        function value independent of whether the step is accepted or not.

        :param funout:
            Function output generated by a :py:class:`FunEvaluator`
            evaluated at new x

        """
        if np.isfinite(funout.fval) and funout.fval < self.fval_min:
            self.x_min = funout.x
            self.fval_min = funout.fval
            self.grad_min = funout.grad

    def update(self,
               step: Step,
               funout_new: Funout,
               funout: Funout) -> None:
        """
        Update self according to employed step

        :param step:
            Employed step

        :param funout:
            Function output generated by a :py:class:`FunEvaluator` for new
            variables before step is taken

        :param funout_new:
            Function output generated by a :py:class:`FunEvaluator` for new
            variables after step is taken
        """
        if self.hessian_update is not None:
            s = step.s + step.s0
            y = funout_new.grad - self.grad

            if isinstance(self.hessian_update, IterativeHessianApproximation):
                self.hessian_update.update(s=s, y=y)
            elif isinstance(self.hessian_update, HybridFixed):
                self.hessian_update.update(
                    s=s, y=y, hess=funout_new.hess,
                    iter_since_tr_update=self.iterations_since_tr_update
                )
            elif isinstance(self.hessian_update, FX):
                # Equation (1.16)
                # A = sres
                # M = hess
                # \delta = s
                # r = res
                gamma = funout_new.hess.dot(s) + \
                        (funout_new.sres - funout.sres).T.dot(funout_new.res)
                self.hessian_update.update(delta=s, gamma=gamma,
                                           r=funout_new.res, rprev=funout.res,
                                           hess=funout_new.hess)
            elif isinstance(self.hessian_update, StructuredApproximation):
                # SSM: Equation (43) in [Dennis et al 1989]
                yb = (funout_new.sres - funout.sres).T.dot(funout_new.res)
                if isinstance(self.hessian_update, (TSSM, GNSBFGS)):
                    # TSSM: Equation (2.5) in [Huschens 1994]
                    # GNSBFGS: Equation (2.1) in [Zhou & Chen 2010]
                    yb *= norm(funout_new.res)/norm(funout.res)
                self.hessian_update.update(s=s, y=y, yb=yb, r=funout_new.res,
                                           hess=funout_new.hess)
            else:
                raise NotImplementedError

            self.hess = self.hessian_update.get_mat()
        else:
            self.hess = funout_new.hess
        self.check_in_bounds(funout_new.x)
        self.fval = funout_new.fval
        self.x = funout_new.x
        self.grad = funout_new.grad
        self.make_non_degenerate()

    def update_tr_radius(self,
                         funout: Funout,
                         step: Step,
                         dv: np.ndarray) -> bool:
        """
        Update the trust region radius

        :param funout:
            Function output generated by a :py:class:`FunEvaluator` for new
            variables after step is taken

        :param step:
            step

        :param dv:
            derivative of scaling vector v wrt x

        :return:
            flag indicating whether the proposed step should be accepted
        """
        fval, grad = funout.fval, funout.grad
        stepsx = step.ss + step.ss0
        nsx = norm(stepsx)
        self.iterations_since_tr_update += 1
        if not np.isfinite(fval):
            self.tr_ratio = 0
            self.delta = np.nanmin([
                self.delta * self.get_option(Options.GAMMA1),
                nsx / 4
            ])
            self.iterations_since_tr_update = 0
            return False
        else:
            qpval = 0.5 * stepsx.dot(dv * np.abs(grad) * stepsx)
            self.tr_ratio = (fval + qpval - self.fval) / step.qpval

            interior_solution = nsx < self.delta_iter * 0.9

            # values as proposed in algorithm 4.1 in Nocedal & Wright
            if self.tr_ratio >= self.get_option(Options.ETA) \
                    and not interior_solution and step.qpval <= 0:
                # increase radius
                self.delta = self.get_option(Options.GAMMA2) * self.delta
                self.iterations_since_tr_update = 0
            elif self.tr_ratio <= self.get_option(Options.MU) or \
                    step.qpval > 0:
                # decrease radius
                self.delta = np.nanmin([
                    self.delta * self.get_option(Options.GAMMA1),
                    nsx / 4
                ])
                self.iterations_since_tr_update = 0
            return self.tr_ratio > 0.0 and step.qpval <= 0

    def check_convergence(self, step: Step, funout: Funout) -> None:
        """
        Check whether optimization has converged.

        :param step:
            update to optimization variables

        :param funout:
            Function output generated by a :py:class:`FunEvaluator`
        """
        converged = False
        fval, grad = funout.fval, funout.grad

        fatol = self.get_option(Options.FATOL)
        frtol = self.get_option(Options.FRTOL)
        xtol = self.get_option(Options.XTOL)
        gatol = self.get_option(Options.GATOL)
        grtol = self.get_option(Options.GRTOL)
        gnorm = norm(grad)
        stepsx = step.ss + step.ss0
        nsx = norm(stepsx)

        if self.delta <= self.delta_iter and \
                np.abs(fval - self.fval) < fatol + frtol*np.abs(self.fval):
            self.exitflag = ExitFlag.FTOL
            self.logger.warning(
                'Stopping as function difference '
                f'{np.abs(self.fval - fval):.2E} was smaller than specified '
                f'tolerances (atol={fatol:.2E}, rtol={frtol:.2E})'
            )
            converged = True

        elif self.iteration > 1 and nsx < xtol:
            self.exitflag = ExitFlag.XTOL
            self.logger.warning(
                'Stopping as norm of step '
                f'{nsx} was smaller than specified '
                f'tolerance (tol={xtol:.2E})'
            )
            converged = True

        elif gnorm <= gatol:
            self.exitflag = ExitFlag.GTOL
            self.logger.warning(
                'Stopping as gradient norm satisfies absolute convergence '
                f'criteria: {gnorm:.2E} < {gatol:.2E}'
            )
            converged = True

        elif gnorm <= grtol * np.abs(self.fval):
            self.exitflag = ExitFlag.GTOL
            self.logger.warning(
                'Stopping as gradient norm satisfies relative convergence '
                f'criteria: {gnorm:.2E} < {grtol:.2E} * '
                f'{np.abs(self.fval):.2E}'
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

        maxiter = self.get_option(Options.MAXITER)
        if self.iteration >= maxiter:
            self.exitflag = ExitFlag.MAXITER
            self.logger.warning(
                f'Stopping as maximum number of iterations {maxiter} was '
                f'exceeded.'
            )
            return False

        time_elapsed = time.time() - self.starttime
        maxtime = self.get_option(Options.MAXTIME)
        time_remaining = maxtime - time_elapsed
        avg_iter_time = time_elapsed/(self.iteration + (self.iteration == 0))
        if time_remaining < avg_iter_time:
            self.exitflag = ExitFlag.MAXTIME
            self.logger.warning(
                f'Stopping as maximum runtime {maxtime} is expected to be '
                f'exceeded in the next iteration.'
            )
            return False

        if self.delta < np.spacing(1):
            self.exitflag = ExitFlag.DELTA_TOO_SMALL
            self.logger.warning(
                f'Stopping as trust region radius {self.delta:.2E} is '
                f'smaller than machine precision.'
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
        Computes the vector v and dv, the diagonal of its Jacobian. For the
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

    def log_step(self, accepted: bool, step: Step, funout: Funout):
        """
        Prints diagnostic information about the current step to the log

        :param accepted:
            flag indicating whether the current step was accepted

        :param step:
            proposal step

        :param funout:
            Function output generated by a :py:class:`FunEvaluator`
        """
        normdx = norm(step.s + step.s0)

        iterspaces = max(len(str(self.get_option(Options.MAXITER))), 5) - \
            len(str(self.iteration))
        steptypespaces = 4 - len(step.type)
        reflspaces, trunspaces = [
            4 - len(str(count))
            for count in [step.reflection_count, step.truncation_count]
        ]

        fval = funout.fval
        if not np.isfinite(fval):
            fval = self.fval
        self.logger.info(
            f'{" " * iterspaces}{self.iteration}'
            f' | {fval if accepted else self.fval:+.3E}'
            f' | {(fval - self.fval):+.2E}'
            f' | {step.qpval:+.2E}'
            f' | {self.tr_ratio:+.2E}'
            f' | {self.delta_iter:.2E}'
            f' | {norm(self.grad):.2E}'
            f' | {normdx:.2E}'
            f' | {step.theta:.2E}'
            f' | {step.alpha:.2E}'
            f' | {step.type}{" " * steptypespaces}'
            f' | {" " * reflspaces}{step.reflection_count}'
            f' | {" " * trunspaces}{step.truncation_count}'
            f' | {int(accepted)}'
        )

    def log_step_initial(self):
        """
        Prints diagnostic information about the initial step to the log
        """

        iterspaces = max(len(str(self.get_option(Options.MAXITER))), 5) - \
            len(str(self.iteration))
        self.logger.info(
            f'{" " * iterspaces}{self.iteration}'
            f' | {self.fval:+.3E}'
            f' |    NaN   '
            f' |    NaN   '
            f' |    NaN   '
            f' | {self.delta:.2E}'
            f' | {norm(self.grad):.2E}'
            f' |   NaN   '
            f' |   NaN   '
            f' |   NaN   '
            f' | NaN '
            f' | NaN '
            f' | NaN '
            f' | {int(np.isfinite(self.fval))}'
        )

    def log_header(self):
        """
        Prints the header for diagnostic information, should complement
        :py:func:`Optimizer.log_step`.
        """
        iterspaces = len(str(self.get_option(Options.MAXITER))) - 5

        self.logger.info(
            f'{" " * iterspaces} iter '
            f'|    fval    | fval diff | pred diff | tr ratio  '
            f'|  delta   |  ||g||   | ||step|| |  theta   |  alpha   '
            f'| step | refl | trun | accept'
        )

    def check_finite(self, funout: Optional[Funout] = None):
        """
        Checks whether objective function value, gradient and Hessian (
        approximation) have finite values and optimization can continue.

        :param funout:
            Function output generated by a :py:class:`FunEvaluator`

        :raises:
            RuntimeError if any of the variables have non-finite entries
        """

        if self.iteration == 0:
            pointstr = 'at initial point.'
        else:
            pointstr = f'at iteration {self.iteration}.'

        if funout is not None:
            fval, grad, hess = funout.fval, funout.grad, funout.hess
        else:
            fval, grad, hess = self.fval, self.grad, self.hess

        if not np.isfinite(fval):
            self.exitflag = ExitFlag.NOT_FINITE
            raise RuntimeError(f'Encountered non-finite function {self.fval} '
                               f'value {pointstr}')

        if not np.isfinite(grad).all():
            self.exitflag = ExitFlag.NOT_FINITE
            ix = np.where(np.logical_not(np.isfinite(grad)))
            raise RuntimeError('Encountered non-finite gradient entries'
                               f' {grad[ix]} for indices {ix} '
                               f'{pointstr}')

        if hess is None:
            return

        if not np.isfinite(hess).all():
            self.exitflag = ExitFlag.NOT_FINITE
            ix = np.where(np.logical_not(np.isfinite(hess)))
            raise RuntimeError('Encountered non-finite gradient hessian'
                               f' {hess[ix]} for indices {ix} '
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
                ix = np.where(diff > 0)[0]
                self.exitflag = ExitFlag.EXCEEDED_BOUNDARY
                raise RuntimeError(f'Exceeded {name} for indices {ix} by '
                                   f'{diff[ix]} {pointstr}')

    def get_option(self, option):
        if option not in Options:
            raise ValueError(f'{option} is not a valid option name.')

        if option not in DEFAULT_OPTIONS:
            raise ValueError(f'{option} is missing its default option.')

        return self.options.get(option, DEFAULT_OPTIONS.get(option))
