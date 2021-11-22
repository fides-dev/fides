from fides import (
    Optimizer, BFGS, SR1, DFP, BB, BG, Broyden, GNSBFGS, HybridFixed,
    HybridFraction, FX, SSM, TSSM, SubSpaceDim, StepBackStrategy, Options
)
import numpy as np

import logging
import pytest
import fides
import time
import h5py
import os


def rosen(x):
    f = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    return f


def rosengrad(x):
    f = rosen(x)

    g = np.array([-400 * (x[1] - x[0] ** 2) * x[0] - 2 * (1 - x[0]),
                  200 * (x[1] - x[0] ** 2)])

    return f, g


def rosenboth(x):
    f, g = rosengrad(x)

    h = np.array([[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]],
                  [-400 * x[0], 200]])

    return f, g, h


def fletcher(x):
    res = np.array([np.sqrt(10.2)*x[0],
                    np.sqrt(10.8)*x[1],
                    4.6-x[0]**2,
                    4.9-x[1]**2])

    sres = np.array([[np.sqrt(10.2), 0],
                     [0, np.sqrt(10.8)],
                     [-2*x[0], 0],
                     [0, -2*x[1]]])

    return res, sres


def rosenrandomfail(x):
    f, g, h = rosenboth(x)

    p = 1/4  # elementwise probability for nan

    if np.random.choice(a=[True, False], p=[p, 1-p]):
        f = np.nan

    g[np.random.choice(a=[True, False], size=g.shape, p=[p, 1-p])] = np.nan

    h[np.random.choice(a=[True, False], size=h.shape, p=[p, 1-p])] = np.nan

    return f, g, h


def rosenwrongf(x):
    f, g, h = rosenboth(x)

    return np.ones((1, 1)) * f, g, h


def rosentransg(x):
    f, g, h = rosenboth(x)

    return f, np.expand_dims(g, 1).T, h


def rosenexpandg(x):
    f, g, h = rosenboth(x)

    return f, np.expand_dims(g, 1), h


def rosenshortg(x):
    f, g, h = rosenboth(x)

    return f, g[0], h


def rosenshorth(x):
    f, g, h = rosenboth(x)

    return f, g, h[0, 0] * np.ones((1, 1))


def rosennonsquarh(x):
    f, g, h = rosenboth(x)

    return f, g, h[0, :]


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
                                      StepBackStrategy.SINGLE_REFLECT,
                                      StepBackStrategy.TRUNCATE,
                                      StepBackStrategy.MIXED,
                                      StepBackStrategy.REFINE])
@pytest.mark.parametrize("subspace_dim", [SubSpaceDim.STEIHAUG,
                                          SubSpaceDim.FULL,
                                          SubSpaceDim.TWO])
@pytest.mark.parametrize("bounds_and_init", [unbounded_and_init(),
                                             finite_bounds_include_optimum(),
                                             finite_bounds_exlude_optimum()])
@pytest.mark.parametrize("fun, happ", [
    (rosenboth, None),  # 0
    (rosengrad, SR1()),  # 1
    (rosengrad, BFGS()),  # 2
    (rosengrad, DFP()),  # 3
    (rosengrad, BG()),  # 4
    (rosengrad, BB()),  # 5
    (rosengrad, Broyden(0.5)),  # 6
    (rosenboth, HybridFixed(BFGS())),  # 7
    (rosenboth, HybridFixed(SR1())),  # 8
    (rosenboth, HybridFixed(BFGS(init_with_hess=True))),  # 9
    (rosenboth, HybridFixed(SR1(init_with_hess=True))),  # 10
    (rosenboth, HybridFraction(BFGS())),  # 11
    (rosenboth, HybridFraction(SR1())),  # 12
    (rosenboth, HybridFraction(BFGS(init_with_hess=True))),  # 13
    (rosenboth, HybridFraction(SR1(init_with_hess=True))),  # 14
    (fletcher, FX(BFGS())),  # 15
    (fletcher, FX(SR1())),  # 16
    (fletcher, FX(BFGS(init_with_hess=True))),  # 17
    (fletcher, FX(SR1(init_with_hess=True))),  # 18
    (fletcher, SSM(0.0)),  # 19
    (fletcher, SSM(0.5)),  # 20
    (fletcher, SSM(1.0)),  # 21
    (fletcher, TSSM(0.0)),  # 22
    (fletcher, TSSM(0.5)),  # 23
    (fletcher, TSSM(1.0)),  # 24
    (fletcher, GNSBFGS()),  # 25
])
def test_minimize_hess_approx(bounds_and_init, fun, happ, subspace_dim,
                              stepback):
    lb, ub, x0 = bounds_and_init

    if (x0 == 0).all() and fun is fletcher:
        x0 += 1

    kwargs = dict(
        fun=fun, ub=ub, lb=lb, verbose=logging.WARNING,
        hessian_update=happ if happ is not None else None,
        options={fides.Options.FATOL: 0,
                 fides.Options.FRTOL: 1e-12 if fun is fletcher else 1e-8,
                 fides.Options.SUBSPACE_DIM: subspace_dim,
                 fides.Options.STEPBACK_STRAT: stepback,
                 fides.Options.MAXITER: 2e2},
        resfun=happ.requires_resfun if happ is not None else False
    )
    if not (subspace_dim == fides.SubSpaceDim.STEIHAUG and
            stepback == fides.StepBackStrategy.REFINE):
        opt = Optimizer(**kwargs)
    else:
        with pytest.raises(ValueError):
            Optimizer(**kwargs)
        return
    opt.minimize(x0)
    assert opt.fval >= opt.fval_min

    if fun is fletcher:
        xsol = [0, 0]
    else:
        xsol = [1, 1]

    if opt.fval == opt.fval_min:
        assert np.isclose(opt.grad, opt.grad_min).all()
        assert np.isclose(opt.x, opt.x_min).all()
    if np.all(ub > 1) and not isinstance(happ, BB):  # bad broyden is bad
        assert np.isclose(opt.x, xsol,
                          atol=1e-4 if fun is fletcher else 1e-6).all()
        assert np.isclose(opt.grad, np.zeros(opt.x.shape),
                          atol=1e-4 if fun is fletcher else 1e-6).all()


@pytest.mark.parametrize("stepback", [StepBackStrategy.REFLECT,
                                      StepBackStrategy.TRUNCATE])
@pytest.mark.parametrize("subspace_dim", [SubSpaceDim.FULL,
                                          SubSpaceDim.TWO])
def test_multistart(subspace_dim, stepback):
    lb, ub, x0 = finite_bounds_exlude_optimum()
    fun = rosenboth

    opt = Optimizer(
        fun, ub=ub, lb=lb, verbose=logging.INFO,
        options={fides.Options.FATOL: 0,
                 fides.Options.SUBSPACE_DIM: subspace_dim,
                 fides.Options.STEPBACK_STRAT: stepback,
                 fides.Options.MAXITER: 1e3}
    )
    for _ in range(int(1e2)):
        x0 = np.random.random(x0.shape) * (ub-lb) + lb
        opt.minimize(x0)
        assert opt.fval >= opt.fval_min
        if opt.fval == opt.fval_min:
            assert np.isclose(opt.grad, opt.grad_min).all()
            assert np.isclose(opt.x, opt.x_min).all()
        if np.all(ub > 1):
            assert np.isclose(opt.x, [1, 1]).all()
            assert np.isclose(opt.grad, np.zeros(opt.x.shape), atol=1e-6).all()


def test_multistart_randomfail():
    lb, ub, x0 = finite_bounds_exlude_optimum()
    fun = rosenrandomfail

    opt = Optimizer(
        fun, ub=ub, lb=lb, verbose=logging.INFO,
        options={fides.Options.FATOL: 0,
                 fides.Options.MAXITER: 1e3}
    )

    for _ in range(int(1e2)):
        with pytest.raises(RuntimeError):
            x0 = np.random.random(x0.shape) * (ub - lb) + lb
            opt.minimize(x0)


@pytest.mark.parametrize("fun", [rosennonsquarh, rosenwrongf, rosenshorth,
                                 rosentransg, rosenshortg, rosenexpandg])
def test_wrong_dim(fun):
    lb, ub, x0 = finite_bounds_exlude_optimum()

    opt = Optimizer(
        fun, ub=ub, lb=lb, verbose=logging.INFO,
        options={fides.Options.FATOL: 0,
                 fides.Options.MAXITER: 1e3}
    )

    with pytest.raises(ValueError):
        x0 = np.random.random(x0.shape) * (ub - lb) + lb
        opt.minimize(x0)


def test_hess_and_hessian_update():
    lb, ub, x0 = finite_bounds_exlude_optimum()
    fun = rosenboth

    opt = Optimizer(
        fun, ub=ub, lb=lb, verbose=logging.INFO,
        options={fides.Options.FATOL: 0},
        hessian_update=DFP()
    )

    with pytest.raises(ValueError):
        opt.minimize(x0)


def test_no_grad():
    lb, ub, x0 = finite_bounds_exlude_optimum()
    fun = rosen

    opt = Optimizer(
        fun, ub=ub, lb=lb, verbose=logging.INFO,
        options={fides.Options.FATOL: 0},
        hessian_update=DFP()
    )

    with pytest.raises(ValueError):
        opt.minimize(x0)


def test_wrong_x():
    lb, ub, x0 = finite_bounds_exlude_optimum()
    fun = rosen

    opt = Optimizer(
        fun, ub=ub, lb=lb, verbose=logging.INFO,
        options={fides.Options.FATOL: 0},
        hessian_update=DFP()
    )

    with pytest.raises(ValueError):
        opt.minimize(np.expand_dims(x0, 1))


def test_maxiter_maxtime():
    lb, ub, x0 = finite_bounds_exlude_optimum()
    fun = rosengrad

    opt = Optimizer(
        fun, ub=ub, lb=lb, verbose=logging.INFO,
        options={fides.Options.FATOL: 0},
        hessian_update=DFP()
    )
    tstart = time.time()
    opt.minimize(x0)
    t_elapsed = time.time() - tstart

    maxiter = opt.iteration - 1
    maxtime = t_elapsed/10

    opt.options[fides.Options.MAXITER] = maxiter
    opt.minimize(x0)
    assert opt.exitflag == fides.ExitFlag.MAXITER
    del opt.options[fides.Options.MAXITER]

    opt.options[fides.Options.MAXTIME] = maxtime
    opt.minimize(x0)
    assert opt.exitflag == fides.ExitFlag.MAXTIME
    del opt.options[fides.Options.MAXTIME]


def test_history():
    lb, ub, x0 = finite_bounds_exlude_optimum()
    fun = fletcher

    h5file = 'history.h5'

    opt = Optimizer(
        fun, ub=ub, lb=lb, verbose=logging.INFO,
        options={fides.Options.FATOL: 0,
                 fides.Options.HISTORY_FILE: h5file},
        hessian_update=GNSBFGS(),
        resfun=True
    )
    opt.minimize(x0)
    opt.minimize(x0)
    with h5py.File(h5file, 'r') as f:
        assert len(f.keys()) == 2  # one group per optimization

    # create new optimizer to check we are appending
    opt = Optimizer(
        fun, ub=ub, lb=lb, verbose=logging.INFO,
        options={fides.Options.FATOL: 0,
                 fides.Options.HISTORY_FILE: h5file},
        hessian_update=GNSBFGS(),
        resfun=True
    )
    opt.minimize(x0)
    opt.minimize(x0)
    opt.minimize(x0)
    with h5py.File(h5file, 'r') as f:
        assert len(f.keys()) == 5
    os.remove(h5file)


def test_wrong_options():
    lb, ub, x0 = finite_bounds_exlude_optimum()
    fun = rosenboth

    with pytest.raises(ValueError):
        Optimizer(
            fun, ub=ub, lb=lb, verbose=logging.INFO,
            options={'option_doesnt_exist': 1}
        )
    with pytest.raises(TypeError):
        Optimizer(
            fun, ub=ub, lb=lb, verbose=logging.INFO,
            options={Options.FATOL: 'not a number'}
        )
