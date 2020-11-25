"""
Subproblem Solvers
------------------
This module provides the machinery to solve 1- and N-dimensional
trust-region subproblems.
"""

import logging

import numpy as np
from numpy.linalg import norm

from scipy import linalg
from scipy.optimize import newton, brentq

from typing import Tuple, Union


def solve_1d_trust_region_subproblem(B: np.ndarray,
                                     g: np.ndarray,
                                     s: np.ndarray,
                                     delta: float,
                                     s0: np.ndarray) -> np.ndarray:
    """
    Solves the special case of a one-dimensional subproblem

    :param B:
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

    a = 0.5 * B.dot(s).dot(s)
    if not isinstance(a, float):
        a = a[0, 0]
    b = s.T.dot(g)

    minq = - b / (2 * a)
    if a > 0 and norm(minq * s + s0) <= delta:
        # interior solution
        tau = minq
    else:
        nrms0 = norm(s0)
        if nrms0 == 0:
            tau = - delta * np.sign(b)
        elif nrms0 >= delta:
            tau = 0
        else:
            tau = brentq(lambda q: 1/norm(q * s + s0) - 1/delta,
                         a=0, b=2*delta, xtol=1e-12, maxiter=100)

    return tau * np.ones((1,))


def solve_nd_trust_region_subproblem(
        B: np.ndarray, g: np.ndarray, delta: float,
        logger: Union[logging.Logger, None] = None
) -> Tuple[np.ndarray, str]:
    r"""
    This function exactly solves the n-dimensional subproblem.

    :math:`argmin_s\{s^T B s + s^T g = 0: ||s|| <= \Delta, s \in \mathbb{
    R}^n\}`

    The  solution to is characterized by the equation
    :math:`-(B + \lambda I)s = g`. If B is positive definite, the solution can
    be obtained by :math:`\lambda = 0`$` if :math:`Bs = -g` satisfies
    :math:`||s|| <= \Delta`. If B is indefinite or :math:`Bs = -g`
    satisfies :math:`||s|| > \Delta` and an approppriate :math:`\lambda` has
    to be  identified via 1D rootfinding of the secular equation

    :math:`\phi(\lambda) = \frac{1}{||s(\lambda)||} - \frac{1}{\Delta} = 0`

    with :math:`s(\lambda)` computed according to an eigenvalue decomposition
    of B. The eigenvalue decomposition, although being more expensive than a
    cholesky decomposition, has the advantage that eigenvectors are
    invariant to changes in :math:`\lambda` and eigenvalues are linear in
    :math:`\lambda`, so factorization only has to be performed once. We perform
    the linesearch via Newton's algorithm and Brent-Q as fallback.
    The hard case is treated seperately and serves as general fallback.

    :param B:
        Hessian of the quadratic subproblem
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
    # decomposition, which works pretty well for n=2
    eigvals, eigvecs = linalg.eig(B)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    w = - eigvecs.T.dot(g)
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
        s = np.real(slam(0, w, eigvals, eigvecs))  # s = - self.cB\self.cg_hat
        if norm(s) <= delta + np.sqrt(np.spacing(1)):  # CASE 0
            logger.debug('Interior subproblem solution')
            return s, 'posdef'
        else:
            laminit = 0
    else:
        laminit = -mineig

    # INDEFINITE CASE
    # note that this includes what Nocedal calls the "hard case" but with
    # ||s|| > delta, so the provided formula is not applicable,
    # the respective w should be close to 0 anyways
    if secular(laminit, w, eigvals, eigvecs, delta) < 0:
        maxiter = 100
        try:
            r = newton(secular, laminit, dsecular, tol=1e-12, maxiter=maxiter,
                       args=(w, eigvals, eigvecs, delta),)
            s = slam(r, w, eigvals, eigvecs)
            if norm(s) <= delta + 1e-12:
                logger.debug('Found boundary subproblem solution via newton')
                return s, 'indef'
        except RuntimeError:
            pass
        try:
            xa = laminit
            xb = (laminit + np.sqrt(np.spacing(1))) * 10
            # search to the right for a change of sign
            while secular(xb, w, eigvals, eigvecs, delta) < 0 and \
                    maxiter > 0:
                xa = xb
                xb = xb * 10
                maxiter -= 1
            if maxiter > 0:
                r = brentq(secular, xa, xb, xtol=1e-12, maxiter=maxiter,
                           args=(w, eigvals, eigvecs, delta))
                s = slam(r, w, eigvals, eigvecs)
                if norm(s) <= delta + np.sqrt(np.spacing(1)):
                    logger.debug(
                        'Found boundary subproblem solution via brentq'
                    )
                    return s, 'indef'
        except RuntimeError:
            pass  # may end up here due to ill-conditioning, treat as hard case

    # HARD CASE (gradient is orthogonal to eigenvector to smallest eigenvalue)
    w[(eigvals-mineig) == 0] = 0
    s = slam(-mineig, w, eigvals, eigvecs)
    # we know that ||s(lam) + sigma*v_jmin|| = delta, since v_jmin is
    # orthonormal, we can just substract the difference in norm to get
    # the right length.

    sigma = np.sqrt(max(delta ** 2 - norm(s) ** 2, 0))
    s = s + sigma * eigvecs[:, jmin]
    logger.debug('Found boundary 2D subproblem solution via hard case')
    return s, 'hard'


def slam(lam: float,
         w: np.ndarray,
         eigvals: np.ndarray,
         eigvecs: np.ndarray) -> np.ndarray:
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
        :math:`s(\lambda)`
    """
    c = w.copy()
    el = eigvals + lam
    c[el != 0] /= el[el != 0]
    return eigvecs.dot(c)


def dslam(lam: float,
          w: np.ndarray,
          eigvals: np.ndarray,
          eigvecs: np.ndarray):
    r"""
    Computes the derivative of the solution :math:`s(\lambda)` with respect to
    lambda, where :math:`s` is the ubproblem solution according to

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
    c = w.copy()
    el = eigvals + lam
    c[el != 0] /= - np.power(el[el != 0], 2)
    c[(el == 0) & (c != 0)] = np.inf
    return eigvecs.dot(c)


def secular(lam: float,
            w: np.ndarray,
            eigvals: np.ndarray,
            eigvecs: np.ndarray,
            delta: float):
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
        precomputed eigenvectors of B
    :param delta:
        trust region radius :math:`\Delta`

    :return:
        :math:`\phi(\lambda)`
    """
    if lam < -np.min(eigvals):
        return np.inf  # safeguard to implement boundary
    s = slam(lam, w, eigvals, eigvecs)
    sn = norm(s)
    if sn > 0:
        return 1 / sn - 1 / delta
    else:
        return np.inf


def dsecular(lam: float, w: np.ndarray, eigvals: np.ndarray,
             eigvecs: np.ndarray, delta: float):
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
        precomputed eigenvectors of B
    :param delta:
        trust region radius :math:`\Delta`

    :return:
        :math:`\frac{\partial \phi(\lambda)}{\partial \lambda}`
    """
    s = slam(lam, w, eigvals, eigvecs)
    ds = dslam(lam, w, eigvals, eigvecs)
    sn = norm(s)
    if sn > 0:
        return - s.T.dot(ds) / (norm(s) ** 3)
    else:
        return np.inf
