"""
Fides
-----------
Fides is an interior trust-region reflective optimizer
"""

from .constants import ExitFlag, Options, StepBackStrategy, SubSpaceDim
from .hessian_approximation import (
    BB,
    BFGS,
    BG,
    DFP,
    FX,
    GNSBFGS,
    SR1,
    SSM,
    TSSM,
    Broyden,
    HybridFixed,
    HybridFraction,
)
from .logging import create_logger

# flake8: noqa
from .minimize import Optimizer
from .version import __version__
