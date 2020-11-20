"""
Constants
-----------
This module provides a central place to define native python enums and
constants that are used in multiple other modules
"""

import enum
import numpy as np


class Options(str, enum.Enum):
    """
    Defines all the fields that can be specified in Options to
    :py:class:`Optimizer`
    """
    MAXITER = 'maxiter'  #: maximum number of allowed iterations
    MAXTIME = 'maxtime'  #: maximum amount of walltime in seconds
    FATOL = 'fatol'  #: absolute tolerance for convergence based on fval
    FRTOL = 'frtol'  #: relative tolerance for convergence based on fval
    XATOL = 'xatol'  #: absolute tolerance for convergence based on x
    XRTOL = 'xrtol'  #: relative tolerance for convergence based on x
    GATOL = 'gatol'  #: absolute tolerance for convergence based on grad
    GRTOL = 'grtol'  #: relative tolerance for convergence based on grad
    SUBSPACE_DIM = 'subspace_solver'  #: trust region subproblem subspace
    STEPBACK_STRAT = 'stepback_strategy'  #: method to use for stepback
    THETA_MAX = 'theta_max'  #: maximal fraction of step that would hit bounds
    DELTA_INIT = 'delta_init'  #: initial trust region radius
    MU = 'mu'  # acceptance threshold for trust region ratio
    ETA = 'eta'  # trust region increase threshold for trust region ratio
    GAMMA1 = 'gamma1'  # factor by which trust region radius will be decreased
    GAMMA2 = 'gamma2'  # factor by which trust region radius will be increased
    REFINE_STEPBACK = 'refine_stepback'  # whether


class SubSpaceDim(str, enum.Enum):
    r"""
    Defines the possible choices of subspace dimension in which the
    subproblem will be solved.
    """
    TWO = '2D'  #: Two dimensional Newton/Gradient subspace
    FULL = 'full'  #: Full :math:`\mathbb{R}^n`


class StepBackStrategy(str, enum.Enum):
    """
    Defines the possible choices of search refinement if proposed step
    reaches optimization boundary
    """
    SINGLE_REFLECT = 'reflect_single'  #: single reflection at boundary
    REFLECT = 'reflect'  #: recursive reflections at boundary
    TRUNCATE = 'truncate'  #: truncate step at boundary and resolve subproblem
    MIXED = 'mixed'  #: mix reflections and truncations


DEFAULT_OPTIONS = {
    Options.MAXITER: 1e3,
    Options.MAXTIME: np.inf,
    Options.FATOL: 1e-6,
    Options.FRTOL: 0,
    Options.XATOL: 0,
    Options.XRTOL: 0,
    Options.GATOL: 1e-6,
    Options.GRTOL: 0,
    Options.SUBSPACE_DIM: SubSpaceDim.FULL,
    Options.STEPBACK_STRAT: StepBackStrategy.REFLECT,
    Options.THETA_MAX: 0.95,
    Options.DELTA_INIT: 1.0,
    Options.MU: 0.25,  # [NodedalWright2006]
    Options.ETA: 0.75,  # [NodedalWright2006]
    Options.GAMMA1: 1/4,  # [NodedalWright2006]
    Options.GAMMA2: 2,  # [NodedalWright2006]
    Options.REFINE_STEPBACK: False,
}


class ExitFlag(int, enum.Enum):
    """
    Defines possible exitflag values for the optimizer to indicate why
    optimization exited. Negative value indicate errors while positive
    values indicate convergence.
    """
    DID_NOT_RUN = 0  #: Optimizer did not run
    MAXITER = -1  #: Reached maximum number of allowed iterations
    MAXTIME = -2  #: Expected to reach maximum allowed time in next iteration
    NOT_FINITE = -3  #: Encountered non-finite fval/grad/hess
    EXCEEDED_BOUNDARY = -4  #: Exceeded specified boundaries
    FTOL = 1  #: Converged according to fval difference
    XTOL = 2  #: Converged according to x difference
    GTOL = 3  #: Converged according to gradient norm
