"""
Subproblem Solvers
------------------
This module provides the machinery to solve 1- and N-dimensional
trust-region subproblems.
"""

import logging
import math

import numpy as np
from numpy.linalg import norm
from scipy import linalg
from scipy.optimize import brentq, newton


def quadratic_form(q: np.ndarray, p: np.ndarray, x: np.ndarray) -> float:
    """
    Computes the quadratic form :math:`x^TQx + x^Tp`

    :param q: Matrix
    :param p: Vector
    :param x: Input

    :return:
        Value of form
    """
    return 0.5 * x.T.dot(q).dot(x) + p.T.dot(x)


def solve_1d_trust_region_subproblem(
    b: np.ndarray, g: np.ndarray, s: np.ndarray, delta: float, s0: np.ndarray
) -> np.ndarray:
    """
    Solves the special case of a one-dimensional subproblem

    :param b:
        Hessian of the quadratic subproblem
    :param g:
        Gradient of the quadratic subproblem
    :param s:
        Vector defining the one-dimensional search direction
    :param delta:
        Norm boundary for the solution of the quadratic subproblem
    :param s0:
        reference point from where search is started, also counts towards
        norm of step

    :return:
        Proposed step-length
    """
    if delta == 0.0:
        return delta * np.ones((1,))

    if np.array_equal(s, np.zeros_like(s)):
        return np.zeros((1,))

    a = 0.5 * b.dot(s).dot(s)
    if not isinstance(a, float):
        a = a[0, 0]
    b_coef = s.T.dot(b.dot(s0) + g)

    minq = -b_coef / (2 * a)
    if a > 0 and norm(minq * s + s0) <= delta:
        # interior solution
        tau = minq
    else:
        tau = get_1d_trust_region_boundary_solution(b, g, s, s0, delta)

    return tau * np.ones((1,))


def get_1d_trust_region_boundary_solution(b, g, s, s0, delta):
    a = np.dot(s, s)
    b_coef = 2 * np.dot(s0, s)
    c = np.dot(s0, s0) - delta**2

    aux = b_coef + math.copysign(np.sqrt(b_coef**2 - 4 * a * c), b_coef)
    ts = [-aux / (2 * a), -2 * c / aux]
    qs = [quadratic_form(b, g, s0 + t * s) for t in ts]
    return ts[np.argmin(qs)]


#: relative tolerance on the norm of a boundary or interior step with respect
#: to the trust region radius
NORM_RTOL = 1e-10
#: relative tolerance with respect to the largest absolute eigenvalue below
#: which eigenvalues are considered equal to the smallest eigenvalue in the
#: hard case, roughly the roundoff error of the eigenvalue decomposition
EIG_RTOL = 10 * np.spacing(1)


def solve_nd_trust_region_subproblem(
    b: np.ndarray,
    g: np.ndarray,
    delta: float,
    logger: logging.Logger | None = None,
) -> tuple[np.ndarray, str]:
    r"""
    This function exactly solves the n-dimensional subproblem.

    :math:`argmin_s\{s^T B s + s^T g = 0: ||s|| <= \Delta, s \in \mathbb{
    R}^n\}`

    The solution is characterized by the equation
    :math:`-(B + \lambda I)s = g`. If B is positive definite, the solution can
    be obtained by :math:`\lambda = 0`$` if :math:`Bs = -g` satisfies
    :math:`||s|| <= \Delta`. If B is indefinite or :math:`Bs = -g`
    satisfies :math:`||s|| > \Delta` and an appropriate :math:`\lambda` has
    to be  identified via 1D rootfinding of the secular equation

    :math:`\phi(\lambda) = \frac{1}{||s(\lambda)||} - \frac{1}{\Delta} = 0`

    with :math:`s(\lambda)` computed according to an eigenvalue decomposition
    of B. The eigenvalue decomposition, although being more expensive than a
    cholesky decomposition, has the advantage that eigenvectors are
    invariant to changes in :math:`\lambda` and eigenvalues are linear in
    :math:`\lambda`, so factorization only has to be performed once. We perform
    the linesearch via Newton's algorithm and Brent-Q as fallback, both
    restricted to an interval that is known to contain the root.
    The hard case is treated separately and serves as general fallback.

    :param B:
        Hessian of the quadratic subproblem, only the symmetric part
        :math:`(B + B^T)/2` is used
    :param g:
        Gradient of the quadratic subproblem
    :param delta:
        Norm boundary for the solution of the quadratic subproblem
    :param logger:
        Logger instance to be used for logging

    :return:
        s: Selected step,
        step_type: Type of solution that was obtained

    """
    if logger is None:
        logger = logging.getLogger('fides')

    if delta == 0:
        return np.zeros(g.shape), 'zero'

    # See Nocedal & Wright 2006 for details
    # INITIALIZATION

    # instead of a cholesky factorization, we go with an eigenvalue
    # decomposition, which works pretty well for n=2. Only the symmetric part
    # of B enters the quadratic form, symmetrizing guarantees real
    # eigenvalues and orthonormal eigenvectors also for non-symmetric
    # hessian approximations
    eigvals, eigvecs = linalg.eigh(0.5 * (b + b.T))
    w = -eigvecs.T.dot(g)
    jmin = eigvals.argmin()
    mineig = eigvals[jmin]

    # since B symmetric eigenvecs V are orthonormal
    # B + lambda I = V * (E + lambda I) * V.T
    # inv(B + lambda I) = V * inv(E + lambda I) * V.T
    # w = V.T * g
    # s(lam) = V * w./(eigvals + lam)
    # ds(lam) = - V * w./((eigvals + lam)**2)
    # \phi(lam) = 1/||s(lam)|| - 1/delta
    # \phi'(lam) = - s(lam).T*ds(lam)/||s(lam)||^3

    # POSITIVE DEFINITE
    if mineig > 0:  # positive definite
        s = slam(0, w, eigvals, eigvecs)  # s = - self.cB\self.cg_hat
        if norm(s) <= delta * (1 + NORM_RTOL):  # CASE 0
            logger.debug('Interior subproblem solution')
            return s, 'posdef'
        else:
            muinit = mineig
    else:
        muinit = 0

    # The root is searched for in the shifted variable mu = lam + mineig,
    # which, in contrast to lam, retains full relative precision close to the
    # pole at lam = -mineig.
    shifted = eigvals - mineig
    args = (w, shifted, eigvecs, delta)

    # INDEFINITE CASE
    # note that this includes what Nocedal calls the "hard case" but with
    # ||s|| > delta, so the provided formula is not applicable.
    # ||s(mu = 0)|| is infinite if g is not orthogonal to the eigenspace of
    # mineig, so we always end up here in that case
    if secular(muinit, *args) < 0:
        # the root is contained in [mu_lo, mu_hi] since
        # |w_i|/(shifted_i + mu) <= ||s(mu)|| <= ||w||/mu
        # [More & Sorensen 1983]
        mu_lo = max(muinit, np.max(np.abs(w) / delta - shifted))
        mu_hi = max(mu_lo, norm(w) / delta)
        # relative changes in ||s(mu)|| are bounded by relative changes in mu
        mutol = max(1e-12 * mu_lo, np.finfo(float).tiny)
        # phi is concave and increasing, so newton iterates started at
        # mu_lo increase monotonically towards the root, unless mu_lo
        # is a pole
        if np.isfinite(norm(_slam_coefs(mu_lo, w, shifted))):
            try:
                r = newton(
                    secular,
                    mu_lo,
                    dsecular,
                    tol=mutol,
                    rtol=1e-12,
                    maxiter=100,
                    args=args,
                )
                s = slam(r, w, shifted, eigvecs)
                if norm(s) <= delta * (1 + NORM_RTOL):
                    logger.debug(
                        'Found boundary subproblem solution via newton'
                    )
                    return s, 'indef'
            except RuntimeError:
                pass
        try:
            if secular(mu_hi, *args) <= 0:
                # mu_hi is the root up to roundoff
                r = mu_hi
            else:
                r = brentq(
                    secular,
                    mu_lo,
                    mu_hi,
                    xtol=mutol,
                    rtol=4 * np.spacing(1),
                    maxiter=500,
                    args=args,
                )
            s = slam(r, w, shifted, eigvecs)
            if norm(s) <= delta * (1 + NORM_RTOL):
                logger.debug('Found boundary subproblem solution via brentq')
                return s, 'indef'
        except (RuntimeError, ValueError):
            pass  # may end up here due to ill-conditioning, treat as hard case

    # HARD CASE (gradient is orthogonal to eigenspace of smallest eigenvalue)
    # eigenvalues that only differ from mineig due to roundoff are treated as
    # a single eigenspace
    degenerate = shifted <= EIG_RTOL * np.max(np.abs(eigvals))
    wmin = w[degenerate]
    w[degenerate] = 0
    s = slam(0, w, shifted, eigvecs)
    if norm(s) > delta:
        # not actually a hard case, but rootfinding failed. fall back to a
        # step that at least satisfies the trust region constraint.
        s *= delta / norm(s)
    # we know that ||s + sigma*v|| = delta for any normalized v in the
    # eigenspace of mineig, since s is orthogonal to that eigenspace, so we
    # can just add the difference in norm. The model value changes by
    # sigma * g.T * v, which is minimal if v is the normalized projection of
    # -g onto the eigenspace.
    sigma = np.sqrt(max(delta**2 - norm(s) ** 2, 0))
    if norm(wmin) > 0:
        v = eigvecs[:, degenerate].dot(wmin) / norm(wmin)
    else:
        v = eigvecs[:, jmin]
    s = s + sigma * v
    logger.debug('Found boundary 2D subproblem solution via hard case')
    return s, 'hard'


def _slam_coefs(lam: float, w: np.ndarray, eigvals: np.ndarray) -> np.ndarray:
    r"""
    Computes the coefficients of :math:`s(\lambda)` in the eigenbasis of B.
    Coefficients are infinite where :math:`\lambda` is the negative of an
    eigenvalue and the respective entry of w is non-zero.
    """
    el = eigvals + lam
    c = np.zeros(w.shape)
    nz = el != 0
    c[nz] = w[nz] / el[nz]
    pole = ~nz & (w != 0)
    c[pole] = np.copysign(np.inf, w[pole])
    return c


def _dslam_coefs(lam: float, w: np.ndarray, eigvals: np.ndarray) -> np.ndarray:
    r"""
    Computes the coefficients of
    :math:`\frac{\partial s(\lambda)}{\partial \lambda}` in the eigenbasis of
    B.
    """
    el = eigvals + lam
    dc = -_slam_coefs(lam, w, eigvals)
    nz = el != 0
    # avoids overflow in el**2
    dc[nz] /= el[nz]
    return dc


def slam(
    lam: float, w: np.ndarray, eigvals: np.ndarray, eigvecs: np.ndarray
) -> np.ndarray:
    r"""
    Computes the solution :math:`s(\lambda)` as subproblem solution according
    to

    :math:`-(B + \lambda I)s = g`

    :param lam:
        :math:`\lambda`
    :param w:
        precomputed eigenvector coefficients for -g
    :param eigvals:
        precomputed eigenvalues of B
    :param eigvecs:
        precomputed eigenvectors of B

    :return:
        :math:`s(\lambda)`, non-finite if :math:`B + \lambda I` is singular
        and -g has a component in its null space
    """
    return eigvecs.dot(_slam_coefs(lam, w, eigvals))


def dslam(lam: float, w: np.ndarray, eigvals: np.ndarray, eigvecs: np.ndarray):
    r"""
    Computes the derivative of the solution :math:`s(\lambda)` with respect to
    lambda, where :math:`s` is the subproblem solution according to

    :math:`-(B + \lambda I)s = g`

    :param lam:
        :math:`\lambda`
    :param w:
        precomputed eigenvector coefficients for -g
    :param eigvals:
        precomputed eigenvalues of B
    :param eigvecs:
        precomputed eigenvectors of B

    :return:
        :math:`\frac{\partial s(\lambda)}{\partial \lambda}`
    """
    return eigvecs.dot(_dslam_coefs(lam, w, eigvals))


def secular(
    lam: float,
    w: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    delta: float,
):
    r"""
    Secular equation

    :math:`\phi(\lambda) = \frac{1}{||s||} - \frac{1}{\Delta}`

    Subproblem solutions are given by the roots of this equation

    :param lam:
        :math:`\lambda`
    :param w:
        precomputed eigenvector coefficients for -g
    :param eigvals:
        precomputed eigenvalues of B
    :param eigvecs:
        precomputed eigenvectors of B, assumed to be orthonormal so that
        norms can be computed in the eigenbasis
    :param delta:
        trust region radius :math:`\Delta`

    :return:
        :math:`\phi(\lambda)`
    """
    if lam < -np.min(eigvals):
        return np.inf  # safeguard to implement boundary
    sn = norm(_slam_coefs(lam, w, eigvals))
    if sn > 0:
        return 1 / sn - 1 / delta
    else:
        return np.inf


def dsecular(
    lam: float,
    w: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    delta: float,
):
    r"""
    Derivative of the secular equation

    :math:`\phi(\lambda) = \frac{1}{||s||} - \frac{1}{\Delta}`

    with respect to :math:`\lambda`

    :param lam:
        :math:`\lambda`
    :param w:
        precomputed eigenvector coefficients for -g
    :param eigvals:
        precomputed eigenvalues of B
    :param eigvecs:
        precomputed eigenvectors of B, assumed to be orthonormal so that
        norms can be computed in the eigenbasis
    :param delta:
        trust region radius :math:`\Delta`

    :return:
        :math:`\frac{\partial \phi(\lambda)}{\partial \lambda}`
    """
    c = _slam_coefs(lam, w, eigvals)
    dc = _dslam_coefs(lam, w, eigvals)
    sn = norm(c)
    if sn > 0:
        # avoids under/overflow in sn**3
        return -(c / sn).dot(dc / sn) / sn
    else:
        return np.inf
