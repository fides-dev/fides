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
        self._diff: np.ndarray = np.empty(0)
        self.init_with_hess = init_with_hess

    def init_mat(self, dim: int, hess: Optional[np.ndarray] = None) -> None:
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
        self._diff = np.zeros_like(self._hess)

    def get_mat(self) -> np.ndarray:
        """
        Getter for the Hessian approximation

        :return:
            Hessian approximation
        """
        return self._hess.copy()

    def get_diff(self) -> np.ndarray:
        """
        Getter for the Hessian approximation update

        :return:
            Hessian approximation update
        """
        return self._diff.copy()

    def set_mat(self, mat: np.ndarray) -> None:
        """
        Setter for the Hessian approximation

        :param mat:
            Hessian approximation
        """
        if mat.shape != self._hess.shape:
            raise ValueError('Passed matrix had inconsistent '
                             f'shape, was {mat.shape}, '
                             f'but should be {self._hess.shape}.')
        self._hess = mat.copy()

    @property
    def requires_resfun(self):
        return False  # pragma: no cover

    @property
    def requires_hess(self):
        return False  # pragma: no cover

    def _update_hess_and_store_diff(self, hess):
        self._diff = hess - self._hess
        self._hess = hess.copy()


class IterativeHessianApproximation(HessianApproximation):
    """
    Iterative update schemes that only use s and y values for update.
    """
    def update(self, s: np.ndarray, y: np.ndarray) -> None:
        """
        Update the Hessian approximation

        :param s:
            step in optimization variables

        :param y:
            step in gradient
        """
        self._compute_update(s, y)
        self._apply_update()

    def _compute_update(self, s: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError()  # pragma: no cover

    def _apply_update(self) -> None:
        self._hess += self._diff


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
    :parameter enforce_curv_cond:
        boolean that controls whether the employed broyden class update
        should attempt to preserve positive definiteness. If set to True,
        updates from steps that violate the curvature condition will be
        discarded.
    """
    def __init__(self, phi: float, init_with_hess: Optional[bool] = False,
                 enforce_curv_cond: Optional[bool] = True):
        self.phi = phi
        self.enforce_curv_cond = enforce_curv_cond
        if phi < 0 or phi > 1:
            warnings.warn('Setting phi to values outside the interval [0, 1]'
                          'will not guarantee that positive definiteness is '
                          'preserved during updating.')
        super(Broyden, self).__init__(init_with_hess)

    def _compute_update(self, s: np.ndarray, y: np.ndarray):
        self._diff = broyden_class_update(y, s, self._hess, self.phi,
                                          self.enforce_curv_cond)


class BFGS(Broyden):
    """
    Broyden-Fletcher-Goldfarb-Shanno update strategy. This is a rank 2
    update strategy that preserves symmetry and positive-semidefiniteness.

    This scheme only works with a function that returns (fval, grad)
    """
    def __init__(self, init_with_hess: Optional[bool] = False,
                 enforce_curv_cond: Optional[bool] = True):
        super(BFGS, self).__init__(phi=0.0, init_with_hess=init_with_hess,
                                   enforce_curv_cond=enforce_curv_cond)


class DFP(Broyden):
    """
    Davidon-Fletcher-Powell update strategy. This is a rank 2
    update strategy that preserves symmetry and positive-semidefiniteness.

    This scheme only works with a function that returns (fval, grad)
    """
    def __init__(self, init_with_hess: Optional[bool] = False,
                 enforce_curv_cond: Optional[bool] = True):
        super(DFP, self).__init__(phi=1.0, init_with_hess=init_with_hess,
                                  enforce_curv_cond=enforce_curv_cond)


class SR1(IterativeHessianApproximation):
    """
    Symmetric Rank 1 update strategy as described in
    [Nocedal & Wright](http://dx.doi.org/10.1007/b98874) Chapter 6.2.
    This is a rank 1 update  strategy that preserves symmetry but does not
    preserve positive-semidefiniteness.

    This scheme only works with a function that returns (fval, grad)
    """
    def _compute_update(self, s: np.ndarray, y: np.ndarray):
        z = y - self._hess.dot(s)
        d = z.T.dot(s)

        # [NocedalWright2006] (6.26) reject if update degenerate
        if np.abs(d) >= np.sqrt(np.spacing(1)) * norm(s) * norm(z):
            self._diff = np.outer(z, z.T) / d
        else:
            self._diff = np.zeros_like(self._hess)


class BG(IterativeHessianApproximation):
    """
    Broydens "good" method as introduced in
    [Broyden 1965](https://doi.org/10.1090%2FS0025-5718-1965-0198670-6).
    This is a rank 1 update strategy that does not preserve symmetry or
    positive definiteness.

    This scheme only works with a function that returns (fval, grad)
    """
    def _compute_update(self, s: np.ndarray, y: np.ndarray):
        z = y - self._hess.dot(s)
        self._diff = np.outer(z, s.T) / s.T.dot(s)


class BB(IterativeHessianApproximation):
    """
    Broydens "bad" method as introduced in
    [Broyden 1965](https://doi.org/10.1090%2FS0025-5718-1965-0198670-6).
    This is a rank 1 update strategy that does not preserve symmetry or
    positive definiteness.

    This scheme only works with a function that returns (fval, grad)
    """
    def update(self, s: np.ndarray, y: np.ndarray) -> None:
        b = y.T.dot(s)
        z = y - self._hess.dot(s)
        if b <= 0:
            self._diff = np.zeros_like(self._hess)
        else:
            self._diff = np.outer(z, s.T) / b


class HybridApproximation(HessianApproximation):
    def __init__(self, happ: IterativeHessianApproximation = BFGS()):
        """
        Create a Hybrid Hessian update strategy that switches between an
        iterative approximation and a dynamic approximation

        :param happ:
            Iterative Hessian Approximation
        """
        self.hessian_update = happ
        super(HybridApproximation, self).__init__()

    def init_mat(self, dim: int, hess: Optional[np.ndarray] = None):
        self.hessian_update.init_mat(dim, hess)
        super(HybridApproximation, self).init_mat(dim, hess)

    def requires_hess(self):
        return True  # pragma: no cover


class HybridSwitchApproximation(HybridApproximation):
    def _switched_update(self, s: np.ndarray, y: np.ndarray,
                         hess: np.ndarray):
        self.hessian_update.update(s, y)
        if self._switched:
            new_hess = self.hessian_update.get_mat()
        else:
            new_hess = hess
        self._update_hess_and_store_diff(new_hess)


class HybridFixed(HybridSwitchApproximation):
    def __init__(self,
                 happ: IterativeHessianApproximation = BFGS(),
                 switch_iteration: Optional[int] = 20):
        """
        Switch from a dynamic approximation to the user provided iterative
        scheme after a fixed number of successive iterations without
        trust-region update. The switching is non-reversible. The iterative
        scheme is initialized and updated rom the beginning, but only
        employed after the specified number of iterations.

        This scheme only works with a function that returns (fval, grad, hess)

        :param switch_iteration:
            Number of iterations without trust region update after which
            switch occurs.
        """
        self.switch_iteration: int = switch_iteration
        super(HybridFixed, self).__init__(happ)
        self._switched = False

    def update(self, s: np.ndarray, y: np.ndarray, hess: np.ndarray,
               iter_since_tr_update: int):
        self._switched_update(s, y, hess)
        if not self._switched:
            self._switched = iter_since_tr_update >= self.switch_iteration


class HybridFraction(HybridSwitchApproximation):
    def __init__(self,
                 happ: IterativeHessianApproximation = BFGS(),
                 switch_threshold: Optional[float] = 0.8):
        """
        Switch from a dynamic approximation to the user provided iterative
        scheme as soon as the fraction of iterations where the step is
        accepted but the trust region is not update exceeds the user provided
        threshold.Threshold check is only performed after 25 iterations.
        The switching is  non-reversible. The iterative scheme is
        initialized and updated rom the beginning, but only employed after
        the switching.

        This scheme only works with a function that returns (fval, grad, hess)

        :param switch_threshold:
            Threshold for fraction of iterations where step is accepted but
            trust region is not updated, which when exceeded triggers switch
            of approximation.
        """
        self.switch_threshold: float = switch_threshold
        super(HybridFraction, self).__init__(happ)
        self._switched = False

    def update(self, s: np.ndarray, y: np.ndarray, hess: np.ndarray,
               tr_nonupdates: int, iterations: int):
        self._switched_update(s, y, hess)
        if not self._switched and iterations > 25:
            self._switched = tr_nonupdates/iterations > self.switch_threshold


class FX(HybridApproximation):
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

    def update(self, delta: np.ndarray, gamma: np.ndarray,
               r: np.ndarray, rprev: np.ndarray, hess: np.ndarray) -> None:
        """
        Update the Hessian approximation

        :param delta:
            step in optimization variables

        :param gamma:
            step in gradient

        :param r:
            residuals after current step

        :param rprev:
            residuals befor current step

        :param hess:
            user-provided (Gauss-Newton) Hessian approximation
        """
        # Equation (3.5)
        ratio = (rprev.dot(rprev) - r.dot(r))/rprev.dot(rprev)
        if ratio >= self.hybrid_tol:
            self._diff = hess - self.hessian_update.get_mat()
            self.hessian_update.set_mat(hess)
        else:
            self.hessian_update.update(delta, gamma)
            self._diff = self.hessian_update.get_diff()

    def get_mat(self) -> np.ndarray:
        return self.hessian_update.get_mat()

    @property
    def requires_resfun(self) -> bool:
        return True  # pragma: no cover


class StructuredApproximation(HessianApproximation):
    def __init__(self, phi: Optional[float] = 0.0,
                 enforce_curv_cond: Optional[bool] = True):
        """
        This is the base class for structured secant methods (SSM). SSMs
        approximate the hessian by combining the Gauss-Newton component C(x)
        and an iteratively updated component that approximates the
        difference S to the true Hessian.

        :parameter phi:
            convex combination parameter interpolating between BFGS (phi==0)
            and DFP (phi==1) update schemes.

        :parameter enforce_curv_cond:
            boolean that controls whether the employed broyden class update
            should attempt to preserve positive definiteness. If set to True,
            updates from steps that violate the curvature condition will be
            discarded.
        """
        self.A: np.ndarray = np.empty(0)
        self.phi = phi
        self.enforce_curv_cond = enforce_curv_cond
        self._structured_diff = np.empty(0)
        if phi < 0 or phi > 1:
            warnings.warn('Setting phi to values outside the interval [0, 1]'
                          'will not guarantee that positive definiteness is '
                          'preserved during updating.')
        super(StructuredApproximation, self).__init__(init_with_hess=True)

    def init_mat(self, dim: int, hess: Optional[np.ndarray] = None):
        self.A = np.eye(dim) * np.spacing(1)
        self._structured_diff = np.zeros_like(self.A)
        super(StructuredApproximation, self).init_mat(dim, hess)

    def update(self, s: np.ndarray, y: np.ndarray, r: np.ndarray,
               hess: np.ndarray, yb: np.ndarray) -> None:
        """
        Update the structured approximation

        :param s:
            step in optimization parameters
        :param y:
            step in gradient parameters
        :param r:
            residual vector
        :param hess:
            user-provided (Gauss-Newton) Hessian approximation
        :param yb:
            approximation to A*s, where A is structured approximation matrix
        """
        raise NotImplementedError()  # pragma: no cover

    @property
    def requires_resfun(self):
        return True  # pragma: no cover

    @property
    def requires_hess(self):
        return True  # pragma: no cover

    def get_structured_diff(self) -> np.ndarray:
        return self._structured_diff


class SSM(StructuredApproximation):
    """
    Structured Secant Method as introduced by
    [Dennis et al 1989](https://doi.org/10.1007/BF00962795), which is
    compatible with BFGS, DFP update schemes.

    This scheme only works with a function that returns (res, sres)
    """

    def update(self, s: np.ndarray, y: np.ndarray, r: np.ndarray,
               hess: np.ndarray, yb: np.ndarray) -> None:
        # B^S = A + C(x_+)
        Bs = hess + self.A
        # y^S = y^# + C(x_+)*s
        ys = yb + hess.dot(s)
        # Equation (13)
        self._structured_diff = broyden_class_update(
            ys, s, Bs, phi=self.phi, enforce_curv_cond=self.enforce_curv_cond
        )
        self.A += self._structured_diff
        # B_+ = C(x_+) + A + BFGS update A (=A_+)
        self._update_hess_and_store_diff(hess + self.A)


class TSSM(StructuredApproximation):
    """
    Totally Structured Secant Method as introduced by
    [Huschens 1994](https://doi.org/10.1137/0804005), which uses a
    self-adjusting update method for the second order term.

    This scheme only works with a function that returns (res, sres)
    """

    def update(self, s: np.ndarray, y: np.ndarray, r: np.ndarray,
               hess: np.ndarray, yb: np.ndarray) -> None:
        # Equation (2.7)
        Bs = hess + norm(r) * self.A
        # Equation (2.6)
        ys = hess.dot(s) + yb
        # Equation (2.10)
        self._structured_diff = broyden_class_update(
            ys, s, Bs, phi=self.phi, enforce_curv_cond=self.enforce_curv_cond
        )/norm(r)
        self.A += self._structured_diff
        # Equation (2.9)
        self._update_hess_and_store_diff(hess + norm(r) * self.A)


class GNSBFGS(StructuredApproximation):
    def __init__(self, hybrid_tol: float = 1e-6,
                 enforce_curv_cond: bool = True):
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
        super(GNSBFGS, self).__init__(phi=0.0,
                                      enforce_curv_cond=enforce_curv_cond)

    def update(self, s: np.ndarray, y: np.ndarray, r: np.ndarray,
               hess: np.ndarray, yb: np.ndarray) -> None:
        # Equation (2.1)
        ratio = yb.T.dot(s)/s.dot(s)
        if ratio > self.hybrid_tol:
            # Equation (2.3)
            self._structured_diff = broyden_class_update(
                yb, s, self.A, phi=self.phi,
                enforce_curv_cond=self.enforce_curv_cond
            )
            self.A += self._structured_diff
            # Equation (2.2)
            self._update_hess_and_store_diff(hess + self.A)
        else:
            # Equation (2.2)
            self._structured_diff = np.zeros_like(self.A)
            self._update_hess_and_store_diff(hess + norm(r) * np.eye(len(y)))


def broyden_class_update(y: np.ndarray,
                         s: np.ndarray,
                         mat: np.ndarray,
                         phi: float = 0.0,
                         enforce_curv_cond: bool = True) -> np.ndarray:
    """
    Scale free implementation of the broyden class update scheme.

    :param y:
        difference in gradient
    :param s:
        search direction in previous step
    :param mat:
        current hessian approximation
    :param phi:
        convex combination parameter, interpolates between BFGS (phi=0) and
        DFP (phi=1).
    :parameter enforce_curv_cond:
        boolean that controls whether the employed broyden class update
        should attempt to preserve positive definiteness. If set to True,
        updates from steps that violate the curvature condition will be
        discarded.
    """
    u = mat.dot(s)
    c = u.T.dot(s)
    b = y.T.dot(s)

    if b <= 0 and enforce_curv_cond:
        return np.zeros_like(mat)

    update = np.outer(y, y.T) / b - np.outer(u, u.T) / c

    if phi != 0.0:
        v = y / b - u / c
        update += phi * c * np.outer(v, v.T)

    return update
