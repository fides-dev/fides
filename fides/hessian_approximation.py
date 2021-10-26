"""
Hessian Update Strategies
-------------------------
This module provides various generic Hessian approximation strategies that
can be employed when the calculating the exact Hessian or an approximation
is computationally too demanding.
"""


from typing import Optional
import numpy as np
import warnings
from numpy.linalg import norm


class HessianApproximation:
    """
    Abstract class from which Hessian update strategies should subclass
    """
    def __init__(self, init_with_hess: Optional[bool] = False):
        """
        Create a Hessian update strategy instance

        :param init_with_hess:
            Whether the hybrid update strategy should be initialized
            according to the user-provided objective function
        """
        self._hess: np.ndarray = np.empty(0)
        self.init_with_hess = init_with_hess

    def init_mat(self, dim: int, hess: Optional[np.ndarray] = None):
        """
        Initializes this approximation instance and checks the dimensionality

        :param dim:
            dimension of optimization variables

        :param hess:
            user provided initialization
        """
        if hess is None or not self.init_with_hess:
            self._hess = np.eye(dim)
        else:
            if hess.shape[0] != dim:
                raise ValueError('Initial approximation had inconsistent '
                                 f'dimension, was {hess.shape[0]}, '
                                 f'but should be {dim}.')
            self._hess = hess.copy()

    def get_mat(self) -> np.ndarray:
        """
        Getter for the Hessian approximation
        :return:
        """
        return self._hess

    def set_mat(self, mat: np.ndarray):
        """
        Getter for the Hessian approximation
        :return:
        """
        if mat.shape != self._hess.shape:
            raise ValueError('Passed matrix had inconsistent '
                             f'shape, was {mat.shape}, '
                             f'but should be {self._hess.shape}.')
        self._hess = mat

    @property
    def requires_resfun(self):
        return False  # pragma: no cover

    @property
    def requires_hess(self):
        return False  # pragma: no cover


class IterativeHessianApproximation(HessianApproximation):
    """
    Iterative update schemes that only use s and y values for update.
    """
    def update(self, s, y):
        raise NotImplementedError()  # pragma: no cover


class Broyden(IterativeHessianApproximation):
    """
    BroydenClass Update scheme as described in [Nocedal & Wright](
    http://dx.doi.org/10.1007/b98874) Chapter 6.3. This is a
    generalization of BFGS/DFP methods where the parameter :math:`phi`
    controls the convex combination between the two. This is a rank 2 update
    strategy that preserves positive-semidefiniteness and symmetry (if
    :math:`\\phi \\in [0,1]`).

    This scheme only works with a function that returns (fval, grad)

    :parameter phi:
        convex combination parameter interpolating between BFGS (phi==0) and
        DFP (phi==1).
    """
    def __init__(self, phi: float, init_with_hess: Optional[bool] = False):
        self.phi = phi
        if phi < 0 or phi > 1:
            warnings.warn('Setting phi to values outside the interval [0, 1]'
                          'will not guarantee that positive definiteness is '
                          'preserved during updating.')
        super(Broyden, self).__init__(init_with_hess)

    def update(self, s, y):
        if y.T.dot(s) <= 0:
            return
        self._hess += broyden_class_update(y, s, self._hess, self.phi)


class BFGS(Broyden):
    """
    Broyden-Fletcher-Goldfarb-Shanno update strategy. This is a rank 2
    update strategy that preserves symmetry and positive-semidefiniteness.

    This scheme only works with a function that returns (fval, grad)
    """
    def __init__(self, init_with_hess: Optional[bool] = False):
        super(BFGS, self).__init__(phi=0.0, init_with_hess=init_with_hess)


class DFP(Broyden):
    """
    Davidon-Fletcher-Powell update strategy. This is a rank 2
    update strategy that preserves symmetry and positive-semidefiniteness.

    This scheme only works with a function that returns (fval, grad)
    """
    def __init__(self, init_with_hess: Optional[bool] = False):
        super(DFP, self).__init__(phi=1.0, init_with_hess=init_with_hess)


class PSB(IterativeHessianApproximation):
    """
    Powell-symmetric-Broyden update strategy as introduced in
    [Powell 1970](https://doi.org/10.1016/B978-0-12-597050-1.50006-3).
    This is a rank 2 update strategy that preserves symmetry and
    positive-semidefiniteness.

    This scheme only works with a function that returns (fval, grad)
    """
    def update(self, s, y):
        self._hess += broyden_class_update(y, s, self._hess, v=s)


class SR1(IterativeHessianApproximation):
    """
    Symmetric Rank 1 update strategy as described in
    [Nocedal & Wright](http://dx.doi.org/10.1007/b98874) Chapter 6.2.
    This is a rank 1 update  strategy that preserves symmetry but does not
    preserve positive-semidefiniteness.

    This scheme only works with a function that returns (fval, grad)
    """
    def update(self, s, y):
        z = y - self._hess.dot(s)
        d = z.T.dot(s)

        # [NocedalWright2006] (6.26) reject if update degenerate
        if np.abs(d) >= 1e-8 * norm(s) * norm(z):
            self._hess += np.outer(z, z.T) / d


class BG(IterativeHessianApproximation):
    """
    Broydens "good" method as introduced in
    [Broyden 1965](https://doi.org/10.1090%2FS0025-5718-1965-0198670-6).
    This is a rank 1 update strategy that does not preserve symmetry or
    positive definiteness.

    This scheme only works with a function that returns (fval, grad)
    """
    def update(self, s, y):
        self._hess += np.outer(y - self._hess.dot(s), s.T) / s.T.dot(s)


class BB(IterativeHessianApproximation):
    """
    Broydens "bad" method as introduced in
    [Broyden 1965](https://doi.org/10.1090%2FS0025-5718-1965-0198670-6).
    This is a rank 1 update strategy that does not preserve symmetry or
    positive definiteness.

    This scheme only works with a function that returns (fval, grad)
    """
    def update(self, s, y):
        b = y.T.dot(s)
        if b <= 0:
            return
        self._hess += np.outer(y - self._hess.dot(s), s.T) / b


class HybridSwitchApproximation(HessianApproximation):
    def __init__(self, happ: IterativeHessianApproximation = BFGS()):
        """
        Create a Hybrid Hessian update strategy that switches between an
        iterative approximation and a dynamic approximation

        :param happ:
            Iterative Hessian Approximation
        """
        self.hessian_update = happ
        super(HybridSwitchApproximation, self).__init__()

    def init_mat(self, dim: int, hess: Optional[np.ndarray] = None):
        self.hessian_update.init_mat(dim, hess)
        super(HybridSwitchApproximation, self).init_mat(dim, hess)

    def get_mat(self) -> np.ndarray:
        return self.hessian_update.get_mat()

    def requires_hess(self):
        return True  # pragma: no cover


class HybridFixed(HybridSwitchApproximation):
    def __init__(self,
                 happ: IterativeHessianApproximation = BFGS(),
                 switch_iteration: Optional[int] = 20):
        """
        Switch from a dynamic approximation that to the user provided
        iterative scheme after a fixed number of iterations. The iterative
        scheme is initialized and updated from the beginning, but only
        employed after the specified number of iterations.

        This scheme only works with a function that returns (fval, grad, hess)

        :param switch_iteration:
            Iteration after which this approximation is used
        """
        self.switch_iteration: int = switch_iteration
        super(HybridFixed, self).__init__(happ)
        self._switched = False

    def update(self, s, y, hess, iter_since_tr_update):
        self.hessian_update.update(s, y)
        self._hess = hess
        if self._switched:
            return
        self._switched = iter_since_tr_update >= self.switch_iteration

    def get_mat(self) -> np.ndarray:
        if self._switched:
            return self.hessian_update.get_mat()
        else:
            return self._hess


class FX(HybridSwitchApproximation):
    def __init__(self,
                 happ: IterativeHessianApproximation = BFGS(),
                 hybrid_tol: Optional[float] = 0.2):
        r"""
        Hybrid method HY2 as introduced by
        [Fletcher & Xu 1986](https://doi.org/10.1093/imanum/7.3.371). This
        approximation scheme employs a dynamic approximation as long as
        function values satisfy :math:`\frac{f_k - f_{k+1}}{f_k} < \epsilon`
        and employs the iterative scheme applied to the last dynamic
        approximation if not.

        This scheme only works with a function that returns (fval, grad, hess)

        :param hybrid_tol:
            switch tolerance :math:`\epsilon`
        """
        self.hybrid_tol = hybrid_tol
        super(FX, self).__init__(happ)

    def update(self, delta, gamma, r, rprev, hess):
        # Equation (3.5)
        ratio = (rprev.dot(rprev) - r.dot(r))/rprev.dot(rprev)
        if ratio >= self.hybrid_tol:
            self.hessian_update.set_mat(hess)
        else:
            self.hessian_update.update(delta, gamma)

    def requires_resfun(self):
        return True  # pragma: no cover


def _bfgs_vector(s, y, mat):
    u = mat.dot(s)
    c = u.T.dot(s)
    b = y.T.dot(s)
    rho = np.sqrt(b / c)
    if not np.isfinite(rho):
        rho = 0
    return y + np.sqrt(rho)*u


def _psb_vector(s, y, mat):
    return s


def _dfp_vector(s, y, mat):
    return y


class StructuredApproximation(HessianApproximation):
    vector_routines = {
        'BFGS': _bfgs_vector,
        'PSB': _psb_vector,
        'DFP': _dfp_vector,
    }

    def __init__(self,
                 update_method: Optional[str] = 'BFGS'):
        """
        This is the base class for structured secant methods (SSM). SSMs
        approximate the hessian by combining the Gauss-Newton component C(x)
        and an iteratively updated component that approximates the
        difference S to the true Hessian.
        """
        self.A: np.ndarray = np.empty(0)
        if update_method not in self.vector_routines:
            raise ValueError(f'Unknown update method {update_method}. Known '
                             f'methods are {self.vector_routines.keys()}.')

        self.vector_routine = self.vector_routines[update_method]
        super(StructuredApproximation, self).__init__(init_with_hess=True)

    def init_mat(self, dim: int, hess: Optional[np.ndarray] = None):
        self.A = np.eye(dim) * np.spacing(1)
        super(StructuredApproximation, self).init_mat(dim, hess)

    def update(self, s: np.ndarray, y: np.ndarray, r: np.ndarray,
               hess: np.ndarray, yb: np.ndarray):
        raise NotImplementedError()  # pragma: no cover

    def requires_resfun(self):
        return True  # pragma: no cover

    def requires_hess(self):
        return True  # pragma: no cover


class SSM(StructuredApproximation):
    """
    Structured Secant Method as introduced by
    [Dennis et al 1989](https://doi.org/10.1007/BF00962795), which is
    compatible with BFGS, DFP and PSB update schemes.

    This scheme only works with a function that returns (res, sres)
    """

    def update(self, s: np.ndarray, y: np.ndarray, r: np.ndarray,
               hess: np.ndarray, yb: np.ndarray):
        # B^S = A + C(x_+)
        Bs = hess + self.A
        # y^S = y^# + C(x_+)*s
        ys = yb + hess.dot(s)
        # Equation (8-10) + Equation (13)
        v = self.vector_routine(s, ys, Bs)
        # Equation (13)
        self.A += broyden_class_update(yb, s, self.A, v=v)
        # B_+ = C(x_+) + A + BFGS update A (=A_+)
        self._hess = hess + self.A


class TSSM(StructuredApproximation):
    """
    Totally Structured Secant Method as introduced by
    [Huschens 1994](https://doi.org/10.1137/0804005), which uses a
    self-adjusting update method for the second order term.

    This scheme only works with a function that returns (res, sres)
    """

    def update(self, s: np.ndarray, y: np.ndarray, r: np.ndarray,
               hess: np.ndarray, yb: np.ndarray):
        # Equation (2.7)
        Bs = hess + norm(r) * self.A
        # Equation (2.6)
        ys = hess.dot(s) + norm(r) * yb
        # Equation (2.8)
        v = self.vector_routine(s, ys, Bs)
        # Equation (2.8)
        self.A += broyden_class_update(yb, s, self.A, v=v)
        # Equation (2.9)
        self._hess = hess + norm(r) * self.A


class GNSBFGS(StructuredApproximation):
    def __init__(self, hybrid_tol: float = 1e-6):
        """
        Hybrid Gauss-Newton Structured BFGS method as introduced by
        [Zhou & Chen 2010](https://doi.org/10.1137/090748470),
        which combines ideas of hybrid switching methods and structured
        secant methods.

        This scheme only works with a function that returns (res, sres)

        :parameter hybrid_tol:
            switching tolerance that controls switching between update methods
        """
        self.hybrid_tol: float = hybrid_tol
        super(GNSBFGS, self).__init__('BFGS')

    def update(self, s: np.ndarray, y: np.ndarray, r: np.ndarray,
               hess: np.ndarray, yb: np.ndarray):
        # Equation (2.1)
        ys = yb * norm(r)
        ratio = ys.T.dot(s)/s.dot(s)
        if ratio > self.hybrid_tol:
            # Equation (2.3)
            self.A += broyden_class_update(ys, s, self.A, phi=1.0)
            # Equation (2.2)
            self._hess = hess + self.A
        else:
            # Equation (2.2)
            self._hess = hess + norm(r) * np.eye(len(y))


def broyden_class_update(y, s, mat, phi=None, v=None):
    """
    Scale free implementation of the broyden class update scheme. This can
    either be called by using a phi parameter that interpolates between BFGS
    (phi=0) and DFP (phi=1) or by using the weighting vector v that allows
    implementation of PSB (v=s), DFP (v=y) and BFGS (V=y+rho*B*s).

    :param y:
        difference in gradient
    :param s:
        search direction in previous step
    :param mat:
        current hessian approximation
    :param phi:
        convex combination parameter. Must not pass this parameter at the same
        time as v.
    :param v:
        weighting vector. Must not pass this parameter at the same time as phi.
    """
    u = mat.dot(s)
    c = u.T.dot(s)
    b = y.T.dot(s)

    if b <= 0:
        return np.zeros(mat.shape)

    if v is None and phi is not None:
        bfgs = phi == 0
    elif v is not None and phi is None:
        bfgs = False
    else:
        raise ValueError('Exactly one of the values of phi and v must be '
                         'provided.')

    if bfgs:  # BFGS
        return np.outer(y, y.T) / b - np.outer(u, u.T) / c

    if v is None:
        rho = np.sqrt(b / c)
        if not np.isfinite(rho):
            rho = 0
        v = y + (1-phi) * rho * u

    z = y - mat.dot(s)
    d = v.T.dot(s)
    a = z.T.dot(s) / (d ** 2)

    update = (np.outer(z, v.T) + np.outer(v, z.T)) / d - a * np.outer(v, v.T)

    return update
