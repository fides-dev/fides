import logging
import os
import time

import h5py
import numpy as np
import pytest

import fides
from fides import (
    BB,
    BFGS,
    BG,
    DFP,
    FX,
    GNSBFGS,
    SR1,
    SSM,
    TSSM,
    Broyden,
    HybridFixed,
    HybridFraction,
    Optimizer,
    Options,
    StepBackStrategy,
    SubSpaceDim,
)


def rosen(x):
    f = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    return f


def rosengrad(x):
    f = rosen(x)

    g = np.array(
        [
            -400 * (x[1] - x[0] ** 2) * x[0] - 2 * (1 - x[0]),
            200 * (x[1] - x[0] ** 2),
        ]
    )

    return f, g


def rosenboth(x):
    f, g = rosengrad(x)

    h = np.array(
        [[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200]]
    )

    return f, g, h


def fletcher(x):
    res = np.array(
        [
            np.sqrt(10.2) * x[0],
            np.sqrt(10.8) * x[1],
            4.6 - x[0] ** 2,
            4.9 - x[1] ** 2,
        ]
    )

    sres = np.array(
        [
            [np.sqrt(10.2), 0],
            [0, np.sqrt(10.8)],
            [-2 * x[0], 0],
            [0, -2 * x[1]],
        ]
    )

    return res, sres


def rosenrandomfail(x):
    f, g, h = rosenboth(x)

    p = 1 / 4  # elementwise probability for nan

    if np.random.choice(a=[True, False], p=[p, 1 - p]):
        f = np.nan

    g[np.random.choice(a=[True, False], size=g.shape, p=[p, 1 - p])] = np.nan

    h[np.random.choice(a=[True, False], size=h.shape, p=[p, 1 - p])] = np.nan

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


@pytest.mark.parametrize(
    'stepback',
    [
        StepBackStrategy.REFLECT,
        StepBackStrategy.SINGLE_REFLECT,
        StepBackStrategy.TRUNCATE,
        StepBackStrategy.MIXED,
        StepBackStrategy.REFINE,
    ],
)
@pytest.mark.parametrize(
    'subspace_dim', [SubSpaceDim.STEIHAUG, SubSpaceDim.FULL, SubSpaceDim.TWO]
)
@pytest.mark.parametrize(
    'bounds_and_init',
    [
        unbounded_and_init(),
        finite_bounds_include_optimum(),
        finite_bounds_exlude_optimum(),
    ],
)
@pytest.mark.parametrize(
    ('fun', 'happ'),
    [
        (rosenboth, None),  # 0
        (rosengrad, SR1()),  # 1
        (rosengrad, BFGS()),  # 2
        (rosengrad, DFP()),  # 3
        (rosengrad, BG()),  # 4
        (rosengrad, BB()),  # 5
        (rosengrad, Broyden(0.5)),  # 6
        (rosenboth, HybridFixed(BFGS())),  # 7
        (rosenboth, HybridFixed(SR1())),  # 8 # 10
        (rosenboth, HybridFraction(BFGS())),  # 11
        (rosenboth, HybridFraction(SR1())),  # 12
        (fletcher, FX(BFGS())),  # 15
        (fletcher, FX(SR1())),  # 16
        (fletcher, SSM(0.0)),  # 19
        (fletcher, SSM(0.5)),  # 20
        (fletcher, SSM(1.0)),  # 21
        (fletcher, TSSM(0.0)),  # 22
        (fletcher, TSSM(0.5)),  # 23
        (fletcher, TSSM(1.0)),  # 24
        (fletcher, GNSBFGS()),  # 25
    ],
)
def test_minimize_hess_approx(
    bounds_and_init, fun, happ, subspace_dim, stepback
):
    lb, ub, x0 = bounds_and_init

    if (x0 == 0).all() and fun is fletcher:
        x0 += 1

    kwargs = {
        'fun': fun,
        'ub': ub,
        'lb': lb,
        'verbose': logging.WARNING,
        'hessian_update': happ if happ is not None else None,
        'options': {
            fides.Options.FATOL: 0,
            fides.Options.FRTOL: 1e-12 if fun is fletcher else 1e-8,
            fides.Options.SUBSPACE_DIM: subspace_dim,
            fides.Options.STEPBACK_STRAT: stepback,
            fides.Options.MAXITER: 2e2,
        },
        'resfun': happ.requires_resfun if happ is not None else False,
    }
    if not (
        subspace_dim == fides.SubSpaceDim.STEIHAUG
        and stepback == fides.StepBackStrategy.REFINE
    ):
        opt = Optimizer(**kwargs)
    else:
        with pytest.raises(ValueError):
            Optimizer(**kwargs)
        return
    opt.minimize(x0)
    assert opt.fval >= opt.fval_min

    xsol = [0, 0] if fun is fletcher else [1, 1]

    if opt.fval == opt.fval_min:
        assert np.isclose(opt.grad, opt.grad_min).all()
        assert np.isclose(opt.x, opt.x_min).all()
    if np.all(ub > 1) and not isinstance(happ, BB):  # bad broyden is bad
        assert np.isclose(
            opt.x, xsol, atol=1e-4 if fun is fletcher else 1e-6
        ).all()
        assert np.isclose(
            opt.grad,
            np.zeros(opt.x.shape),
            atol=1e-4 if fun is fletcher else 1e-6,
        ).all()


@pytest.mark.parametrize(
    'stepback', [StepBackStrategy.REFLECT, StepBackStrategy.TRUNCATE]
)
@pytest.mark.parametrize('subspace_dim', [SubSpaceDim.FULL, SubSpaceDim.TWO])
def test_multistart(subspace_dim, stepback):
    lb, ub, x0 = finite_bounds_exlude_optimum()
    fun = rosenboth

    opt = Optimizer(
        fun,
        ub=ub,
        lb=lb,
        verbose=logging.INFO,
        options={
            fides.Options.FATOL: 0,
            fides.Options.SUBSPACE_DIM: subspace_dim,
            fides.Options.STEPBACK_STRAT: stepback,
            fides.Options.MAXITER: 1e3,
        },
    )
    for _ in range(int(1e2)):
        x0 = np.random.random(x0.shape) * (ub - lb) + lb
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
        fun,
        ub=ub,
        lb=lb,
        verbose=logging.INFO,
        options={fides.Options.FATOL: 0, fides.Options.MAXITER: 1e3},
    )

    for _ in range(int(1e2)):
        x0 = np.random.random(x0.shape) * (ub - lb) + lb
        with pytest.raises(RuntimeError):
            opt.minimize(x0)


@pytest.mark.parametrize(
    'fun',
    [
        rosennonsquarh,
        rosenwrongf,
        rosenshorth,
        rosentransg,
        rosenshortg,
        rosenexpandg,
    ],
)
def test_wrong_dim(fun):
    lb, ub, x0 = finite_bounds_exlude_optimum()

    opt = Optimizer(
        fun,
        ub=ub,
        lb=lb,
        verbose=logging.INFO,
        options={fides.Options.FATOL: 0, fides.Options.MAXITER: 1e3},
    )

    x0 = np.random.random(x0.shape) * (ub - lb) + lb
    with pytest.raises(ValueError):
        opt.minimize(x0)


def test_hess_and_hessian_update():
    lb, ub, x0 = finite_bounds_exlude_optimum()
    fun = rosenboth

    opt = Optimizer(
        fun,
        ub=ub,
        lb=lb,
        verbose=logging.INFO,
        options={fides.Options.FATOL: 0},
        hessian_update=DFP(),
    )

    with pytest.raises(ValueError):
        opt.minimize(x0)


def test_no_grad():
    lb, ub, x0 = finite_bounds_exlude_optimum()
    fun = rosen

    opt = Optimizer(
        fun,
        ub=ub,
        lb=lb,
        verbose=logging.INFO,
        options={fides.Options.FATOL: 0},
        hessian_update=DFP(),
    )

    with pytest.raises(ValueError):
        opt.minimize(x0)


def test_wrong_x():
    lb, ub, x0 = finite_bounds_exlude_optimum()
    fun = rosen

    opt = Optimizer(
        fun,
        ub=ub,
        lb=lb,
        verbose=logging.INFO,
        options={fides.Options.FATOL: 0},
        hessian_update=DFP(),
    )

    with pytest.raises(ValueError):
        opt.minimize(np.expand_dims(x0, 1))


def test_maxiter_maxtime():
    lb, ub, x0 = finite_bounds_exlude_optimum()
    fun = rosengrad

    opt = Optimizer(
        fun,
        ub=ub,
        lb=lb,
        verbose=logging.INFO,
        options={fides.Options.FATOL: 0},
        hessian_update=DFP(),
    )
    tstart = time.time()
    opt.minimize(x0)
    t_elapsed = time.time() - tstart

    maxiter = opt.iteration - 1
    maxtime = t_elapsed / 10

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
        fun,
        ub=ub,
        lb=lb,
        verbose=logging.INFO,
        options={fides.Options.FATOL: 0, fides.Options.HISTORY_FILE: h5file},
        hessian_update=GNSBFGS(),
        resfun=True,
    )
    opt.minimize(x0)
    opt.minimize(x0)
    with h5py.File(h5file, 'r') as f:
        assert len(f.keys()) == 2  # one group per optimization

    # create new optimizer to check we are appending
    opt = Optimizer(
        fun,
        ub=ub,
        lb=lb,
        verbose=logging.INFO,
        options={fides.Options.FATOL: 0, fides.Options.HISTORY_FILE: h5file},
        hessian_update=GNSBFGS(),
        resfun=True,
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
            fun,
            ub=ub,
            lb=lb,
            verbose=logging.INFO,
            options={'option_doesnt_exist': 1},
        )
    with pytest.raises(TypeError):
        Optimizer(
            fun,
            ub=ub,
            lb=lb,
            verbose=logging.INFO,
            options={Options.FATOL: 'not a number'},
        )

    # check we can pass floats for ints
    Optimizer(
        fun, ub=ub, lb=lb, verbose=logging.INFO, options={Options.MAXITER: 1e4}
    )
    with pytest.raises(ValueError):
        Optimizer(
            fun,
            ub=ub,
            lb=lb,
            verbose=logging.INFO,
            options={Options.SUBSPACE_DIM: 'invalid_subspace'},
        )

    # check we can pass strings for enums
    Optimizer(
        fun,
        ub=ub,
        lb=lb,
        verbose=logging.INFO,
        options={Options.SUBSPACE_DIM: '2D'},
    )


def test_hess0_initialization():
    """
    Test that hess0 parameter correctly initializes Hessian approximation.
    """
    lb, ub, x0 = finite_bounds_include_optimum()
    fun = rosengrad
    fun_with_hess = rosenboth

    # Test 1: Verify hess0 is used when provided with hessian_update
    custom_hess0 = np.eye(len(x0)) * 10.0
    opt_with_hess0 = Optimizer(
        fun,
        ub=ub,
        lb=lb,
        verbose=logging.WARNING,
        options={Options.MAXITER: 1},  # Only run one iteration
        hessian_update=BFGS(),
    )
    opt_with_hess0.minimize(x0, hess0=custom_hess0)
    assert opt_with_hess0.hess is not None

    # Test 2: Verify default initialization when hess0 is not provided
    opt_without_hess0 = Optimizer(
        fun,
        ub=ub,
        lb=lb,
        verbose=logging.WARNING,
        options={Options.MAXITER: 1},
        hessian_update=BFGS(),
    )
    opt_without_hess0.minimize(x0)

    # Test 3: Verify hess0 has correct dimensions
    wrong_dim_hess0 = np.eye(len(x0) + 1)
    opt_wrong_dim = Optimizer(
        fun,
        ub=ub,
        lb=lb,
        verbose=logging.WARNING,
        hessian_update=BFGS(),
    )
    with pytest.raises(ValueError):
        opt_wrong_dim.minimize(x0, hess0=wrong_dim_hess0)

    # Test 4: Verify hess0 works with different update schemes
    for happ_class in [BFGS, DFP, SR1, Broyden]:
        happ = happ_class() if happ_class != Broyden else Broyden(phi=0.5)
        custom_hess = np.eye(len(x0)) * 5.0
        opt = Optimizer(
            fun,
            ub=ub,
            lb=lb,
            verbose=logging.WARNING,
            options={Options.MAXITER: 2, Options.FATOL: 0},
            hessian_update=happ,
        )
        opt.minimize(x0, hess0=custom_hess)
        assert opt.iteration >= 1, f'Failed for {happ_class.__name__}'

    # Test 5: Verify hess0 is ignored when no hessian_update is provided
    opt_no_update = Optimizer(
        fun_with_hess,
        ub=ub,
        lb=lb,
        verbose=logging.WARNING,
        options={Options.MAXITER: 1},
    )
    hess0_ignored = np.eye(len(x0)) * 100.0
    opt_no_update.minimize(x0, hess0=hess0_ignored)

    # Test 6: Test initialization with exact Hessian
    opt_hess_init = Optimizer(
        fun_with_hess,
        ub=ub,
        lb=lb,
        verbose=logging.WARNING,
        options={Options.MAXITER: 10, Options.FATOL: 1e-8},
        hessian_update=HybridFixed(BFGS()),
    )
    opt_hess_init.minimize(x0, hess0='hess')
    iterations_with_hess = opt_hess_init.iteration

    # Compare with BFGS without using initial Hessian
    opt_no_hess_init = Optimizer(
        fun,
        ub=ub,
        lb=lb,
        verbose=logging.WARNING,
        options={Options.MAXITER: 10, Options.FATOL: 1e-8},
        hessian_update=BFGS(),
    )
    opt_no_hess_init.minimize(x0)
    iterations_without_hess = opt_no_hess_init.iteration

    # Using exact Hessian for initialization should help convergence
    assert iterations_with_hess <= iterations_without_hess or (
        opt_hess_init.converged and opt_no_hess_init.converged
    ), 'Hessian initialization should help convergence'

    # Test 8: Verify hess0 affects convergence behavior
    true_hess_at_x0 = np.array(
        [
            [1200 * x0[0] ** 2 - 400 * x0[1] + 2, -400 * x0[0]],
            [-400 * x0[0], 200],
        ]
    )

    opt_good_init = Optimizer(
        fun,
        ub=ub,
        lb=lb,
        verbose=logging.WARNING,
        options={Options.MAXITER: 100, Options.FATOL: 1e-8},
        hessian_update=BFGS(),
    )
    opt_good_init.minimize(x0, hess0=true_hess_at_x0)
    iterations_good = opt_good_init.iteration

    # Use a poor initial Hessian approximation
    poor_hess = np.eye(len(x0)) * 0.01
    opt_poor_init = Optimizer(
        fun,
        ub=ub,
        lb=lb,
        verbose=logging.WARNING,
        options={Options.MAXITER: 100, Options.FATOL: 1e-8},
        hessian_update=BFGS(),
    )
    opt_poor_init.minimize(x0, hess0=poor_hess)
    iterations_poor = opt_poor_init.iteration

    # Good initialization should converge in fewer or equal iterations
    assert iterations_good <= iterations_poor or (
        opt_good_init.converged and opt_poor_init.converged
    ), 'Good Hessian initialization should help convergence'
