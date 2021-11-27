"""
Constants
-----------
This module provides a central place to define native python enums and
constants that are used in multiple other modules
"""

import enum
import numpy as np

from numbers import Real, Integral
from pathlib import PosixPath, WindowsPath

from typing import Dict


class Options(str, enum.Enum):
    """
    Defines all the fields that can be specified in Options to
    :py:class:`Optimizer`
    """
    MAXITER = 'maxiter'  #: maximum number of allowed iterations
    MAXTIME = 'maxtime'  #: maximum amount of walltime in seconds
    FATOL = 'fatol'  #: absolute tolerance for convergence based on fval
    FRTOL = 'frtol'  #: relative tolerance for convergence based on fval
    XTOL = 'xtol'  #: tolerance for convergence based on x
    GATOL = 'gatol'  #: absolute tolerance for convergence based on grad
    GRTOL = 'grtol'  #: relative tolerance for convergence based on grad
    SUBSPACE_DIM = 'subspace_solver'  #: trust region subproblem subspace
    STEPBACK_STRAT = 'stepback_strategy'  #: method to use for stepback
    THETA_MAX = 'theta_max'  #: maximal fraction of step that would hit bounds
    DELTA_INIT = 'delta_init'  #: initial trust region radius
    MU = 'mu'  #: acceptance threshold for trust region ratio
    ETA = 'eta'  #: trust region increase threshold for trust region ratio
    GAMMA1 = 'gamma1'  #: factor by which trust region radius will be decreased
    GAMMA2 = 'gamma2'  #: factor by which trust region radius will be increased
    HISTORY_FILE = 'history_file'  #: when set, statistics for each start will
    # be saved to the specified file


class SubSpaceDim(str, enum.Enum):
    r"""
    Defines the possible choices of subspace dimension in which the
    subproblem will be solved.
    """
    TWO = '2D'  #: Two dimensional Newton/Gradient subspace
    FULL = 'full'  #: Full :math:`\mathbb{R}^n`
    STEIHAUG = 'scg'  #: CG subspace via Steihaug's method


class StepBackStrategy(str, enum.Enum):
    """
    Defines the possible choices of search refinement if proposed step
    reaches optimization boundary
    """
    SINGLE_REFLECT = 'reflect_single'  #: single reflection at boundary
    REFLECT = 'reflect'  #: recursive reflections at boundary
    TRUNCATE = 'truncate'  #: truncate step at boundary and re-solve
    # restricted subproblem
    MIXED = 'mixed'  #: mix reflections and truncations
    REFINE = 'refine'  #: perform optimization to refine step


DEFAULT_OPTIONS = {
    Options.MAXITER: 1e3,
    Options.MAXTIME: np.inf,
    Options.FATOL: 1e-8,
    Options.FRTOL: 1e-8,
    Options.XTOL: 0,
    Options.GATOL: 1e-6,
    Options.GRTOL: 0,
    Options.SUBSPACE_DIM: SubSpaceDim.TWO,
    Options.STEPBACK_STRAT: StepBackStrategy.REFLECT,
    Options.THETA_MAX: 0.95,
    Options.DELTA_INIT: 1.0,
    Options.MU: 0.25,  # [NodedalWright2006]
    Options.ETA: 0.75,  # [NodedalWright2006]
    Options.GAMMA1: 1/4,  # [NodedalWright2006]
    Options.GAMMA2: 2,  # [NodedalWright2006]
    Options.HISTORY_FILE: None,
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
    DELTA_TOO_SMALL = -5  #: Trust Region Radius too small to proceed
    FTOL = 1  #: Converged according to fval difference
    XTOL = 2  #: Converged according to x difference
    GTOL = 3  #: Converged according to gradient norm


def validate_options(options: Dict):
    """Check if the chosen options are valid"""
    expected_types = {
        Options.MAXITER: Integral,
        Options.MAXTIME: Real,
        Options.FATOL: Real,
        Options.FRTOL: Real,
        Options.XTOL: Real,
        Options.GATOL: Real,
        Options.GRTOL: Real,
        Options.SUBSPACE_DIM: (SubSpaceDim, str),
        Options.STEPBACK_STRAT: (StepBackStrategy, str),
        Options.THETA_MAX: Real,
        Options.DELTA_INIT: Real,
        Options.MU: Real,
        Options.ETA: Real,
        Options.GAMMA1: Real,
        Options.GAMMA2: Real,
        Options.HISTORY_FILE: (str, PosixPath, WindowsPath),
    }
    for option_key, option_value in options.items():
        try:
            option = Options(option_key)
        except ValueError:
            raise ValueError(f'{option_key} is not a valid options field.')

        if option_key is Options.SUBSPACE_DIM:
            option_value = SubSpaceDim(option_value)

        if option_key is Options.STEPBACK_STRAT:
            option_value = StepBackStrategy(option_value)

        expected_type = expected_types[option]
        if not isinstance(option_value, expected_type):
            if expected_type == Integral and int(option_value) == option_value:
                continue
            raise TypeError(f'Type mismatch for option {option_key}. '
                            f'Expected {expected_type} but got '
                            f'{type(option_value)}')
