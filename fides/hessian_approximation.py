"""
Hessian Update Strategies
-------------------------
This module provides various generic Hessian approximation strategies that
can be employed when the calculating the exact Hessian or an approximation
is computationally too demandind.
"""


from typing import Optional
import numpy as np


class HessianApproximation:
    """
    Abstract class from which Hessian update strategies should subclass
    """
    def __init__(self, dim, hess_init: Optional[np.ndarray] = None):
        if hess_init is None:
            hess_init = np.eye(dim)
        self._hess = hess_init.copy()

    def update(self, s, y):
        raise NotImplementedError()  # pragma : no cover

    def get_mat(self) -> np.ndarray:
        return self._hess


class SR1(HessianApproximation):
    """
    Symmetric Rank 1 update strategy. This updating strategy may yield
    indefinite hessian approximations.
    """
    def update(self, s, y):
        z = y - self._hess.dot(s)
        self._hess += np.outer(z, z.T)/z.T.dot(s)


class BFGS(HessianApproximation):
    """
    Broyden-Fletcher-Goldfarb-Shanno update strategy. This is a rank 2
    update strategy that always yields positive-semidefinite hessian
    approximations.
    """
    def update(self, s, y):
        b = y.T.dot(s)
        if b <= 0:
            return

        z = self._hess.dot(s)
        a = s.T.dot(z)
        self._hess += - np.outer(z, z.T) / a + np.outer(y, y.T) / b


class DFP(HessianApproximation):
    """
    Davidon-Fletcher-Powell update strategy. This is a rank 2
    update strategy that always yields positive-semidefinite hessian
    approximations. It usually does not perform as well as the BFGS
    strategy, but included for the sake of completeness.
    """
    def update(self, s, y):
        curv = y.T.dot(s)
        if curv <= 0:
            return
        mat1 = np.eye(self._hess.shape[0]) - np.outer(y, s.T) / curv
        mat2 = np.eye(self._hess.shape[0]) - np.outer(s, y.T) / curv

        self._hess = mat1.dot(self._hess).dot(mat2) + np.outer(y, y.T)/curv
