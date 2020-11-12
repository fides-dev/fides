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
    SUBSPACE_SOLVER = 'subspace_solver'


class SubSpaceDim(str, enum.Enum):
    """
    Defines the possible choices of subspace dimension in which the
    subproblem will be solved
    """
    TWO = '2D'
    FULL = 'full'


DEFAULT_OPTIONS = {
    Options.MAXITER: 1e3,
    Options.MAXTIME: np.inf,
    Options.FATOL: 1e-6,
    Options.FRTOL: 0,
    Options.XATOL: 0,
    Options.XRTOL: 0,
    Options.GATOL: 1e-6,
    Options.GRTOL: 0,
    Options.SUBSPACE_SOLVER: SubSpaceDim.FULL,
}
