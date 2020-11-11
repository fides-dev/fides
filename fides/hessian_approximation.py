from typing import Optional
import numpy as np


class HessianApproximation:
    def __init__(self, dim, hess_init: Optional[np.ndarray] = None):
        if hess_init is None:
            hess_init = np.eye(dim)
        self._hess = hess_init.copy()

    def update(self, s, y):
        raise NotImplementedError()

    def get_mat(self) -> np.ndarray:
        return self._hess


class SR1(HessianApproximation):
    def update(self, s, y):
        z = y - self._hess.dot(s)
        self._hess += np.outer(z, z.T)/z.T.dot(s)


class BFGS(HessianApproximation):
    def update(self, s, y):
        b = y.T.dot(s)
        if b <= 0:
            return

        z = self._hess.dot(s)
        a = s.T.dot(z)
        self._hess += - np.outer(z, z.T) / a + np.outer(y, y.T) / b


class DFP(HessianApproximation):
    def update(self, s, y):
        curv = y.T.dot(s)
        if curv <= 0:
            return
        z = y - self._hess.dot(s)

        self._hess += np.outer(z, y.T)/curv + np.outer(y, z.T)/curv \
            - z.T.dot(s)/(curv ** 2) * np.outer(y, y.T)
