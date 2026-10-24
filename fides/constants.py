"""
Constants
-----------
This module provides a central place to define native python enums and
constants that are used in multiple other modules
"""

import enum
from numbers import Integral, Real
from pathlib import PosixPath, WindowsPath

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
    Options.GAMMA1: 1 / 4,  # [NodedalWright2006]
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


def validate_options(options: dict):
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
            raise ValueError(
                f'{option_key} is not a valid options field.'
            ) from None

        # compare the converted key, plain string keys would otherwise skip
        # the value checks
        enum_types = {
            Options.SUBSPACE_DIM: SubSpaceDim,
            Options.STEPBACK_STRAT: StepBackStrategy,
        }
        if option in enum_types:
            enum_type = enum_types[option]
            try:
                option_value = enum_type(option_value)
            except ValueError:
                raise ValueError(
                    f'{option_value!r} is not a valid value for option '
                    f'{option.value}. Valid values are '
                    f'{[e.value for e in enum_type]}.'
                ) from None

        expected_type = expected_types[option]
        if not isinstance(option_value, expected_type):
            if expected_type == Integral and _is_integral(option_value):
                pass
            else:
                raise TypeError(
                    f'Type mismatch for option {option_key}. '
                    f'Expected {expected_type} but got '
                    f'{type(option_value)}'
                )

        if option in OPTION_RANGES:
            lo, hi, lo_open, hi_open = OPTION_RANGES[option]
            if (
                np.isnan(option_value)
                or option_value < lo
                or option_value > hi
                or (lo_open and option_value == lo)
                or (hi_open and option_value == hi)
            ):
                raise ValueError(
                    f'Option {option.value} must be in the interval '
                    f'{"(" if lo_open else "["}{lo}, {hi}'
                    f'{")" if hi_open else "]"}, but was {option_value}.'
                )

    mu = options.get(Options.MU, DEFAULT_OPTIONS[Options.MU])
    eta = options.get(Options.ETA, DEFAULT_OPTIONS[Options.ETA])
    if mu > eta:
        raise ValueError(
            f'Option {Options.MU.value} ({mu}) must not exceed option '
            f'{Options.ETA.value} ({eta}).'
        )


def _is_integral(value) -> bool:
    """Check whether a (float) value has an integral value"""
    try:
        return int(value) == value
    except (TypeError, ValueError, OverflowError):
        return False


#: admissible ranges for numeric options as (lower, upper, lower_open,
#: upper_open). Options outside these ranges break the trust-region
#: iteration (e.g. a radius that never shrinks or steps onto the boundary).
OPTION_RANGES = {
    Options.MAXITER: (0, np.inf, False, False),
    Options.MAXTIME: (0, np.inf, False, False),
    Options.FATOL: (0, np.inf, False, False),
    Options.FRTOL: (0, np.inf, False, False),
    Options.XTOL: (0, np.inf, False, False),
    Options.GATOL: (0, np.inf, False, False),
    Options.GRTOL: (0, np.inf, False, False),
    Options.THETA_MAX: (0, 1, True, False),
    Options.DELTA_INIT: (0, np.inf, True, True),
    Options.MU: (0, 1, False, True),
    Options.ETA: (0, 1, False, True),
    Options.GAMMA1: (0, 1, True, True),
    Options.GAMMA2: (1, np.inf, True, True),
}
