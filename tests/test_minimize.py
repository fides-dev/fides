from fides import Optimizer, BFGS, SR1, DFP, SubSpaceDim, StepBackStrategy
import numpy as np

import logging
import pytest
import fides


def rosengrad(x):
    f = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    g = np.array([-400 * (x[1] - x[0] ** 2) * x[0] - 2 * (1 - x[0]),
                  200 * (x[1] - x[0] ** 2)])

    return f, g


def rosenboth(x):
    f, g = rosengrad(x)

    h = np.array([[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]],
                  [-400 * x[0], 200]])

    return f, g, h


def finite_bounds_include_optimum():
    lb = np.array([-2, -1.5])
    ub = np.array([1.5, 2])

    x0 = np.zeros(lb.shape)
    return lb, ub, x0


def finite_bounds_exlude_optimum():
    lb = np.array([-2, -1.5])
    ub = np.array([0.99, 0.99])

    x0 = (lb + ub) / 2
    return lb, ub, x0


def unbounded_and_init():
    lb = np.array([-np.inf, -np.inf])
    ub = np.array([np.inf, np.inf])

    x0 = np.zeros(lb.shape)
    return lb, ub, x0


@pytest.mark.parametrize("stepback", [StepBackStrategy.REFLECT,
                                      StepBackStrategy.TRUNCATE])
@pytest.mark.parametrize("subspace_dim", [SubSpaceDim.FULL,
                                          SubSpaceDim.TWO])
@pytest.mark.parametrize("bounds_and_init", [finite_bounds_include_optimum(),
                                             unbounded_and_init(),
                                             finite_bounds_exlude_optimum()])
@pytest.mark.parametrize("fun, happ", [
    (rosenboth, None),
    (rosenboth, None),
    (rosengrad, SR1()),
    (rosengrad, BFGS()),
    (rosengrad, DFP()),
])
def test_minimize_hess_approx(bounds_and_init, fun, happ, subspace_dim,
                              stepback):
    lb, ub, x0 = bounds_and_init

    opt = Optimizer(
        fun, ub=ub, lb=lb, verbose=logging.INFO,
        hessian_update=happ if happ is not None else None,
        options={fides.Options.FATOL: 0,
                 fides.Options.SUBSPACE_DIM: subspace_dim,
                 fides.Options.STEPBACK_STRAT: stepback,
                 fides.Options.MAXITER: 1e3}
    )
    opt.minimize(x0)
    assert opt.fval >= opt.fval_min
    if opt.fval == opt.fval_min:
        assert np.isclose(opt.grad, opt.grad_min).all()
        assert np.isclose(opt.x, opt.x_min).all()
    if np.all(ub > 1):
        assert np.isclose(opt.x, [1, 1]).all()
        assert np.isclose(opt.grad, np.zeros(opt.x.shape), atol=1e-6).all()
