import numpy as np
from numpy.linalg import norm

from scipy import linalg
from scipy.optimize import newton, brentq

from .logging import logger


def solve_trust_region_subproblem(B: np.ndarray,
                                  g: np.ndarray,
                                  delta: float):
    """
    This function exactly solves the two dimensional subproblem.
    $$ armgin_s{s^T B s + s^T g = 0, ||s|| <= \Delta, s in \mathbb{R}^2} $$
    The  solution to is characterized by the equation
    $-(B + \lamda I)s = g$. If B is positive definite, the solution can
    be obtained by $\lambda = 0$ if $Bs = -g$ satisfies ||s|| <= \Delta$.
    If B is indefinite or $Bs = -g$ satisfies ||s|| > delta$, lambda has to
    be identified by linesearch over the secular equation
    $$\phi(\lambda) = \frac{1}{||s(\lambda)||} - \frac{1}{\Delta} = 0$$
    with s(\lambda) computed according to an eigenvalue decomposition of B.
    The eigenvalue decomposition, although being more expensive than a
    cholesky decomposition, has the advantage that eigenvectors are
    invariant to changes in \lambda and eigenvalues are linear in
    \lambda, so factorization only has to be performed once. We perform
    the linesearch via Newtons equation. Hard case (B indefinite) is
    treated seperately.

    :param B: Hessian of the quadratic subproblem
    :param g: gradient of the quadratic subproblem
    :param delta: norm boundary for the solution of the quadratic subproblem
    """
    # See Nocedal & Wright 2006 for details
    # INITIALIZATION

    # instead of a cholesky factorization, we go with an eigenvalue
    # decomposition, which works pretty well for n=2
    eigvals, eigvecs = linalg.eig(B)
    eigvals = np.real(eigvals)
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
    if mineig > np.sqrt(np.spacing(1)):  # positive definite
        s = slam(0, w, eigvals, eigvecs)  # s = - self.cB\self.cg_hat
        if norm(s) <= delta + np.sqrt(np.spacing(1)):  # CASE 0
            logger.debug('Interior 2D subproblem solution')
            return s, 'posdef'
        else:
            laminit = 0
    else:
        # add a little wiggle room so we don't end up at the pole
        laminit = -mineig + np.sqrt(np.spacing(1))

    # INDEFINITE CASE (
    # note that this includes what Nocedal calls the "hard case" but with
    # ||s|| > delta, so the provided formula is not applicable,
    # the respective w should be close to 0 anyways
    if secular(laminit, w, eigvals, eigvecs, delta) < 0:
        maxiter = 100
        try:
            r = newton(secular, laminit, dsecular, tol=1e-12, maxiter=maxiter,
                       args=(w, eigvals, eigvecs, delta),)
            s = slam(r, w, eigvals, eigvecs)
            logger.debug('Found boundary 2D subproblem solution via newton')
            return s, 'indef'
        except RuntimeError:
            pass
        try:
            xf = laminit * 2
            # search to the right for a change of sign
            while secular(xf, w, eigvals, eigvecs, delta) < 0 and \
                    maxiter > 0:
                xf = laminit * 2
                maxiter -= 1
            r = brentq(secular, xf/2, xf, xtol=1e-12, maxiter=maxiter,
                       args=(w, eigvals, eigvecs, delta))
            s = slam(r, w, eigvals, eigvecs)
            logger.debug('Found boundary 2D subproblem solution via brentq')
            return s, 'indef'
        except RuntimeError:
            pass  # may end up here due to ill-conditioning, treat as hard case

    # HARD CASE (gradient is orthogonal to eigenvector to smallest eigenvalue)
    w[w == w[jmin]] = 0
    s = slam(-mineig, w, eigvals, eigvecs)
    s[np.isnan(s)] = 0
    # we know that ||s(lam) + sigma*v_jmin|| = delta, since v_jmin is
    # orthonormal, we can just substract the difference in norm to get
    # the right length.

    sigma = np.sqrt(delta ** 2 - norm(s) ** 2)
    s = s + sigma * eigvecs[:, jmin]
    logger.debug('Found boundary 2D subproblem solution via hard case')
    return s, 'hard'


def slam(lam: float, w: np.ndarray, eigvals: np.ndarray, eigvecs: np.ndarray):
    c = w.copy()
    el = eigvals + lam
    c[el != 0] /= el[el != 0]
    c[(el == 0) & (c != 0)] = np.inf
    return eigvecs.dot(c)


def dslam(lam: float, w: np.ndarray, eigvals: np.ndarray, eigvecs: np.ndarray):
    c = w.copy()
    el = eigvals + lam
    c[el != 0] /= - np.power(el[el != 0], 2)
    c[(el == 0) & (c != 0)] = np.inf
    return eigvecs.dot(c)


def secular(lam: float, w: np.ndarray, eigvals: np.ndarray,
            eigvecs: np.ndarray, delta: float):
    if lam < -np.min(eigvals):
        return np.inf  # safeguard to implement boundary
    s = slam(lam, w, eigvals, eigvecs)
    return 1 / norm(s) - 1 / delta


def dsecular(lam: float, w: np.ndarray, eigvals: np.ndarray,
             eigvecs: np.ndarray, _):
    s = slam(lam, w, eigvals, eigvecs)
    ds = dslam(lam, w, eigvals, eigvecs)
    return - s.T.dot(ds) / (norm(s) ** 3)
