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
    MAXITER = 'maxiter'
    MAXTIME = 'maxtime'
    FATOL = 'fatol'
    FRTOL = 'frtol'
    XATOL = 'xatol'
    XRTOL = 'xrtol'
    GATOL = 'gatol'
    GRTOL = 'grtol'
    SUBSPACE_DIM = 'subspace_solver'
    STEPBACK_STRAT = 'stepback_strategy'


class SubSpaceDim(str, enum.Enum):
    r"""
    Defines the possible choices of subspace dimension in which the
    subproblem will be solved.

    `2D`: Two dimensional subspace spanned by gradient and Newton search
    direction
    `full`: Full :math:`\mathbb{R}^n`
    """
    TWO = '2D'
    FULL = 'full'


class StepBackStrategy(str, enum.Enum):
    """
    Defines the possible choices of search refinement if proposed step
    reaches optimization boundary

    `reflect`: reflect step at boundary
    `reduce`: truncate step at boundary and search remaining subspace
    """
    REFLECT = 'reflect'
    TRUNCATE = 'reduce'


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
    Options.STEPBACK_STRAT: StepBackStrategy.TRUNCATE,
}


class ExitFlag(int, enum.Enum):
    """
    Defines possible exitflag values for the optimizer to indicate why
    optimization exited. Negative value indicate errors while positive
    values indicate convergence.
    """
    DID_NOT_RUN = 0
    MAXITER = -1
    MAXTIME = -2
    NOT_FINITE = -3
    EXCEEDED_BOUNDARY = -4
    FTOL = 1
    XTOL = 2
    GTOL = 3
    SMALL_DELTA = 4