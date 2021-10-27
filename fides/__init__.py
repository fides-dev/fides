"""
Fides
-----------
Fides is an interior trust-region reflective optimizer
"""

# flake8: noqa
from .minimize import Optimizer
from .hessian_approximation import (
    SR1, BFGS, DFP, FX, HybridFixed, GNSBFGS, BB, BG, Broyden, SSM, TSSM
)
from .logging import create_logger
from .version import __version__
from .constants import Options, SubSpaceDim, StepBackStrategy, ExitFlag
