import logging
import os
import re
import signal
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


def test_history_reused_start_id(tmp_path):
    # a reused start_id must fail before the optimization runs, not when
    # writing the history afterwards, which would discard the result
    h5file = tmp_path / 'history.h5'
    calls = []

    def fun(x):
        calls.append(x)
        return rosenboth(x)

    lb, ub, x0 = finite_bounds_exlude_optimum()
    opt = Optimizer(
        fun,
        ub=ub,
        lb=lb,
        verbose=logging.WARNING,
        options={Options.HISTORY_FILE: h5file},
    )
    opt.minimize(x0, start_id='run')
    calls.clear()
    with pytest.raises(ValueError, match='start_id'):
        opt.minimize(x0, start_id='run')
    assert not calls

    opt.minimize(x0, start_id='run2')
    with h5py.File(h5file, 'r') as f:
        assert set(f.keys()) == {'run', 'run2'}


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


@pytest.mark.parametrize(
    'options',
    [
        {'stepback_strategy': 'reflct'},
        {'subspace_solver': 'foo'},
        {Options.STEPBACK_STRAT: 'reflct'},
        {Options.SUBSPACE_DIM: 'foo'},
    ],
)
def test_invalid_enum_options(options):
    # plain string keys used to skip the validation of the value, an invalid
    # stepback strategy then silently disabled stepback
    lb, ub, _ = finite_bounds_exlude_optimum()
    with pytest.raises(ValueError, match='not a valid value'):
        Optimizer(rosenboth, ub=ub, lb=lb, options=options)


@pytest.mark.parametrize('stepback', [s.value for s in StepBackStrategy])
@pytest.mark.parametrize('subspace_dim', [s.value for s in SubSpaceDim])
def test_string_options(subspace_dim, stepback):
    lb, ub, x0 = finite_bounds_exlude_optimum()
    options = {
        'subspace_solver': subspace_dim,
        'stepback_strategy': stepback,
        'maxiter': 5,
    }
    if subspace_dim == 'scg' and stepback == 'refine':
        with pytest.raises(ValueError, match='refinement'):
            Optimizer(rosenboth, ub=ub, lb=lb, options=options)
        return
    opt = Optimizer(
        rosenboth, ub=ub, lb=lb, verbose=logging.WARNING, options=options
    )
    opt.minimize(x0)
    assert opt.get_option(Options.SUBSPACE_DIM) == SubSpaceDim(subspace_dim)
    assert opt.get_option(Options.STEPBACK_STRAT) == StepBackStrategy(stepback)


@pytest.mark.parametrize(
    'happ',
    [
        HybridFixed(BFGS(), switch_iteration=2),
        HybridFraction(BFGS(), switch_threshold=-1.0),
    ],
)
def test_hybrid_reuse_across_runs(happ):
    """
    Reusing a hybrid approximation instance (e.g., in a multistart) must
    give the same result as a fresh instance, i.e., the switch from the
    dynamic to the iterative approximation must not carry over.
    """
    lb, ub, x0 = unbounded_and_init()
    x0 = np.array([-1.2, 1.0])
    opt = Optimizer(
        rosenboth,
        ub=ub,
        lb=lb,
        verbose=logging.WARNING,
        hessian_update=happ,
        options={Options.FATOL: 0},
    )
    opt.minimize(x0)
    iterations = opt.iteration
    assert happ._switched
    opt.minimize(x0)
    assert opt.iteration == iterations


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


def test_x_names_wrong_length():
    lb, ub, _ = finite_bounds_exlude_optimum()
    with pytest.raises(ValueError, match='x_names'):
        Optimizer(rosenboth, ub=ub, lb=lb, x_names=['x'])


def rosennangrad(x):
    f, g, h = rosenboth(x)
    g[1] = np.nan
    return f, g, h


def rosennanhess(x):
    f, g, h = rosenboth(x)
    h[0, 1] = np.nan
    return f, g, h


@pytest.mark.parametrize(
    ('fun', 'x0', 'x_names', 'expected'),
    [
        (rosenboth, [0.0, 5.0], None, 'upper bounds for indices [1] by'),
        (
            rosenboth,
            [0.0, 5.0],
            ['a', 'b'],
            "upper bounds for ['b'] (indices [1]) by",
        ),
        (
            rosenboth,
            [-5.0, 0.0],
            np.array(['a', 'b']),
            "lower bounds for ['a'] (indices [0]) by",
        ),
        (rosennangrad, [0.0, 0.0], None, 'for indices [1] at'),
        (rosennangrad, [0.0, 0.0], ['a', 'b'], "for ['b'] (indices [1])"),
        (rosennanhess, [0.0, 0.0], ['a', 'b'], "for [('a', 'b')] (indices"),
    ],
)
def test_x_names_in_messages(fun, x0, x_names, expected):
    lb, ub, _ = finite_bounds_exlude_optimum()
    opt = Optimizer(fun, ub=ub, lb=lb, x_names=x_names)
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        opt.minimize(np.array(x0))


def saddle_quartic(n=10, seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    ev = np.linspace(-1, 1, n)
    ev[ev == 0] = 0.1
    a = q @ np.diag(ev) @ q.T

    def fun(x):
        f = 0.5 * x @ a @ x + 0.01 * np.sum(x**4)
        g = a @ x + 0.04 * x**3
        h = a + np.diag(0.12 * x**2)
        return f, g, h

    return fun


def test_no_ftol_on_tr_limited_step():
    # tiny steps that are limited by a tiny trust region radius must not
    # trigger function value convergence at a saddle point
    n = 10
    opt = Optimizer(
        saddle_quartic(n),
        ub=np.full(n, np.inf),
        lb=np.full(n, -np.inf),
        verbose=logging.WARNING,
        options={Options.DELTA_INIT: 1e-6, Options.MAXITER: 1000},
    )
    opt.minimize(1e-3 * np.ones(n))
    assert opt.iteration > 1
    assert opt.fval < -30
    assert np.linalg.norm(opt.grad) < 1e-6


@pytest.mark.parametrize('delta_init', [1e-6, 1.0, 1e6])
@pytest.mark.parametrize('stepback', list(StepBackStrategy))
def test_gtol_all_bounds_active(stepback, delta_init):
    # optimum at the upper bound with large gradient, scaled gradient
    # v * g vanishes there and should trigger GTOL
    n = 5
    opt = Optimizer(
        lambda x: (np.sum((x - 3) ** 2), 2 * (x - 3), 2 * np.eye(n)),
        ub=np.ones(n),
        lb=-np.ones(n),
        verbose=logging.WARNING,
        options={
            Options.DELTA_INIT: delta_init,
            Options.STEPBACK_STRAT: stepback,
            Options.MAXITER: 1000,
        },
    )
    opt.minimize(np.zeros(n))
    assert opt.exitflag > 0
    assert np.isclose(opt.fval, 4 * n, rtol=1e-6)


def test_concave_box_zero_gradient_start_accepts_first_step():
    # the ratio augmentation must use the gradient at the current point
    # (zero here), otherwise the exact first step is rejected
    n = 10
    opt = Optimizer(
        lambda x: (-np.sum(x**2), -2 * x, -2 * np.eye(n)),
        ub=2 * np.ones(n),
        lb=-np.ones(n),
        verbose=logging.WARNING,
        options={Options.MAXITER: 1},
    )
    opt.minimize(np.zeros(n))
    assert opt.tr_ratio > opt.get_option(Options.ETA)
    assert opt.fval < 0


@pytest.mark.parametrize('bounds', [(-1, 2), (-2, 1)])
@pytest.mark.parametrize('subspace_dim', [SubSpaceDim.TWO, SubSpaceDim.FULL])
@pytest.mark.parametrize('delta_init', [1e-6, 1.0, 1e6])
def test_concave_box_zero_gradient_start(delta_init, subspace_dim, bounds):
    # x0 = 0 is a saddle point with zero gradient, so the first step goes
    # along an eigenvector of -2 * I with arbitrary sign. If that step ends on
    # a bound, the other variables still have zero gradient, i.e., the scaled
    # gradient vanishes at a saddle point and GTOL must not fire. Every vertex
    # of the box is a strict local minimizer and no other point is, which
    # vertex is reached depends on the eigenvector sign.
    n = 10
    lb, ub = bounds[0] * np.ones(n), bounds[1] * np.ones(n)
    opt = Optimizer(
        lambda x: (-np.sum(x**2), -2 * x, -2 * np.eye(n)),
        ub=ub,
        lb=lb,
        verbose=logging.WARNING,
        options={
            Options.DELTA_INIT: delta_init,
            Options.SUBSPACE_DIM: subspace_dim,
            Options.MAXITER: 1000,
        },
    )
    opt.minimize(np.zeros(n))
    assert opt.exitflag > 0
    assert np.all(np.isclose(opt.x, lb) | np.isclose(opt.x, ub))


def test_gtol_ignores_curvature_of_hessian_approximation():
    # a hessian approximation may have negative curvature that the function
    # does not have, only exact hessians block GTOL at stationary points
    n = 2
    opt = Optimizer(
        lambda x: (np.sum(x**2), 2 * x),
        ub=np.ones(n),
        lb=-np.ones(n),
        verbose=logging.WARNING,
        hessian_update=SR1(),
        options={Options.MAXITER: 1000},
    )
    opt.minimize(np.zeros(n), hess0=-np.eye(n))
    assert opt.exitflag == fides.ExitFlag.GTOL
    assert opt.iteration == 1


def test_logger_reused_across_runs(capsys):
    # repeated runs must not register a new logger and handler each time
    lb, ub, x0 = finite_bounds_exlude_optimum()
    opt = Optimizer(rosenboth, ub=ub, lb=lb, verbose=logging.INFO)
    opt.minimize(x0)
    logger = opt.logger
    n_loggers = len(logging.Logger.manager.loggerDict)
    out_first = capsys.readouterr().err
    assert out_first

    for _ in range(3):
        opt.minimize(x0)
        assert capsys.readouterr().err.count('\n') == out_first.count('\n')
    assert opt.logger is logger
    assert len(logger.handlers) == 1
    assert len(logging.Logger.manager.loggerDict) == n_loggers

    # a changed verbosity applies to the next run
    opt.verbose = logging.WARNING
    opt.minimize(x0)
    assert logger.level == logging.WARNING
    assert not capsys.readouterr().err


def exp_overflow(x):
    f = np.sum(np.exp(x)) + np.sum((x + 50) ** 2)
    g = np.exp(x) + 2 * (x + 50)
    h = np.diag(np.exp(x) + 2)
    return f, g, h


@pytest.mark.skipif(
    not hasattr(signal, 'SIGALRM'), reason='requires signal.SIGALRM'
)
@pytest.mark.parametrize('subspace_dim', list(SubSpaceDim))
def test_huge_gradient_and_hessian(subspace_dim):
    # gradient and hessian entries ~e^600 overflow r'r and d'Bd in the
    # Steihaug CG recursion, which then looped forever on nan. MAXTIME is only
    # checked between iterations, so enforce a hard runtime cap via SIGALRM.
    def timeout(signum, frame):
        raise TimeoutError('minimize did not terminate')

    n = 3
    opt = Optimizer(
        exp_overflow,
        ub=np.full(n, np.inf),
        lb=np.full(n, -np.inf),
        verbose=logging.WARNING,
        options={Options.SUBSPACE_DIM: subspace_dim, Options.MAXITER: 2000},
    )
    handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(60)
    try:
        with np.errstate(over='ignore'):
            opt.minimize(np.full(n, 600.0))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, handler)
    assert opt.exitflag in (fides.ExitFlag.GTOL, fides.ExitFlag.FTOL)
    assert opt.fval < 1e-12
    assert np.allclose(opt.x, -50)


def test_gtol_not_triggered_by_rejected_step():
    """
    A rejected step must not trigger gradient based convergence: the
    gradient criterion has to be evaluated at the point the optimizer
    actually stays at, not at the rejected trial point.
    """

    def plateau(x):
        # f = 1 - exp(-x^2): the gradient vanishes on the plateau far from
        # the minimum at 0, where f is larger than anywhere near 0
        e = np.exp(-(x[0] ** 2))
        return float(1 - e), 2 * x * e

    opt = Optimizer(
        plateau,
        ub=np.array([np.inf]),
        lb=np.array([-np.inf]),
        verbose=logging.INFO,
        hessian_update=BFGS(),
        options={Options.DELTA_INIT: 10.0},
    )
    # tiny initial Hessian => first step is radius limited and lands on
    # the plateau at x = 9.5, where f is larger and the step is rejected
    fval, x, grad, _ = opt.minimize(np.array([-0.5]), hess0=np.array([[1e-3]]))

    assert opt.exitflag == fides.ExitFlag.GTOL
    assert np.linalg.norm(grad) <= opt.get_option(Options.GATOL)
    assert np.abs(x[0]) < 1e-3
    assert opt.iteration > 1


def test_xtol_not_triggered_by_rejected_step():
    """
    A cascade of rejected steps shrinks the trust region and thereby the
    proposed step, this must not be mistaken for convergence in x.
    """

    def wrong_gradient(x):
        # gradient with the wrong sign: every proposed step increases f
        return float(np.sum(x**2)), -2 * x, 2 * np.eye(len(x))

    opt = Optimizer(
        wrong_gradient,
        ub=np.full(2, np.inf),
        lb=np.full(2, -np.inf),
        verbose=logging.INFO,
        options={Options.XTOL: 1e-6},
    )
    fval, x, grad, _ = opt.minimize(np.array([1.0, 1.0]))

    assert opt.exitflag != fides.ExitFlag.XTOL
    assert opt.exitflag < 0
    assert not opt.converged


def test_gtol_at_initial_point():
    """
    Starting at a stationary point, the (zero) step is rejected but
    gradient based convergence must still be detected at the current point.
    """

    def quadratic(x):
        return float(np.sum(x**2)), 2 * x, 2 * np.eye(len(x))

    opt = Optimizer(
        quadratic,
        ub=np.full(2, np.inf),
        lb=np.full(2, -np.inf),
        verbose=logging.INFO,
    )
    fval, x, grad, _ = opt.minimize(np.zeros(2))

    assert opt.exitflag == fides.ExitFlag.GTOL
    assert opt.iteration == 1
    assert np.all(x == 0)


def _quadratic(x):
    return float(np.sum(x**2)), 2 * x, 2 * np.eye(len(x))


def test_bounds_and_x0_as_lists():
    """lists and integer arrays for bounds and x0 must work like arrays"""
    opt = Optimizer(_quadratic, ub=[2, 2], lb=[-2, -2], verbose=logging.ERROR)
    fval, x, _, _ = opt.minimize([1, 1])
    assert opt.lb.dtype == float
    assert x.dtype == float
    assert np.allclose(x, 0, atol=1e-6)


def test_integer_x0_on_bound_is_shifted():
    """
    the non-degeneracy shift must not be truncated away for integer x0
    """
    opt = Optimizer(_quadratic, ub=[2, 2], lb=[0, 0], verbose=logging.ERROR)
    opt.minimize(np.array([0, 0]), hess0=None)
    assert opt.x.dtype == float


@pytest.mark.parametrize(
    ('lb', 'ub', 'x0', 'match'),
    [
        (np.zeros(2), np.ones(3), np.ones(2), 'lb has 2 entries'),
        (np.zeros(2), np.array([1, np.nan]), np.ones(2), 'NaN'),
        (np.ones(2), np.zeros(2), np.ones(2), 'lb exceeds ub'),
        (np.zeros((2, 1)), np.ones((2, 1)), np.ones(2), 'ndim == 1'),
        (np.zeros(2), np.ones(2), np.full(3, 0.5), 'x0 has 3 entries'),
    ],
)
def test_invalid_bounds_and_x0(lb, ub, x0, match):
    with pytest.raises(ValueError, match=match):
        Optimizer(_quadratic, ub=ub, lb=lb, verbose=logging.ERROR).minimize(x0)


def test_resfun_without_hessian_update():
    """
    resfun=True without a Hessian update scheme performs a Gauss-Newton
    optimization with the Hessian approximation computed from sres
    """
    lb, ub, x0 = finite_bounds_include_optimum()
    opt = Optimizer(
        fletcher,
        ub=ub,
        lb=lb,
        verbose=logging.ERROR,
        resfun=True,
        # Gauss-Newton converges linearly on this large-residual problem
        options={Options.FATOL: 0, Options.FRTOL: 1e-12},
    )
    fval, x, grad, hess = opt.minimize(x0 + 0.5)
    assert opt.exitflag > 0
    assert np.allclose(x, [0, 0], atol=1e-4)
    assert hess.shape == (2, 2)


def test_resfun_with_incompatible_hessian_update():
    lb, ub, _ = finite_bounds_include_optimum()
    with pytest.raises(ValueError, match='does not support'):
        Optimizer(
            fletcher,
            ub=ub,
            lb=lb,
            verbose=logging.ERROR,
            resfun=True,
            hessian_update=BFGS(),
        )


def test_hess0_validation():
    lb, ub, x0 = finite_bounds_include_optimum()
    # 'hess' with a function that does not return a hessian
    opt = Optimizer(
        rosengrad, ub=ub, lb=lb, verbose=logging.ERROR, hessian_update=BFGS()
    )
    with pytest.raises(ValueError, match="hess0='hess' requires"):
        opt.minimize(x0, hess0='hess')
    with pytest.raises(ValueError, match="string 'hess'"):
        opt.minimize(x0, hess0='identity')
    with pytest.raises(ValueError, match='inconsistent shape'):
        opt.minimize(x0, hess0=np.ones((2, 3)))


def test_missing_hessian_without_update_scheme():
    lb, ub, x0 = finite_bounds_include_optimum()
    opt = Optimizer(
        lambda x: (*rosengrad(x), None), ub=ub, lb=lb, verbose=logging.ERROR
    )
    with pytest.raises(ValueError, match='must return a Hessian'):
        opt.minimize(x0)


def test_array_like_function_outputs():
    """
    zero-dimensional arrays (as returned by autodiff frameworks) and lists
    must be accepted as function outputs
    """
    lb, ub, x0 = finite_bounds_include_optimum()

    def fun(x):
        f, g, h = rosenboth(x)
        return np.array(f), list(g), h.tolist()

    opt = Optimizer(fun, ub=ub, lb=lb, verbose=logging.ERROR)
    fval, x, grad, hess = opt.minimize(x0)
    assert isinstance(fval, float)
    assert grad.dtype == float
    assert hess.dtype == float
    assert np.allclose(x, [1, 1], atol=1e-6)

    def fun_vector_fval(x):
        f, g, h = rosenboth(x)
        return np.array([f]), g, h

    with pytest.raises(ValueError, match='scalar'):
        Optimizer(fun_vector_fval, ub=ub, lb=lb).minimize(x0)


def test_verbose_none():
    lb, ub, x0 = finite_bounds_include_optimum()
    opt = Optimizer(rosenboth, ub=ub, lb=lb, verbose=None)
    opt.minimize(x0)
    assert opt.logger.level == logging.INFO


@pytest.mark.parametrize(
    ('option', 'value'),
    [
        (Options.GAMMA1, 1.0),
        (Options.GAMMA1, 0.0),
        (Options.GAMMA2, 1.0),
        (Options.THETA_MAX, 1.5),
        (Options.THETA_MAX, 0.0),
        (Options.DELTA_INIT, 0.0),
        (Options.DELTA_INIT, -1.0),
        (Options.MU, 0.9),  # exceeds default eta
        (Options.ETA, 0.1),  # below default mu
        (Options.FATOL, -1e-8),
        (Options.MAXITER, -1),
        (Options.MAXTIME, np.nan),
    ],
)
def test_option_out_of_range(option, value):
    lb, ub, _ = finite_bounds_include_optimum()
    with pytest.raises(ValueError, match=option.value):
        Optimizer(rosenboth, ub=ub, lb=lb, options={option: value})


def test_option_boundary_values_accepted():
    lb, ub, _ = finite_bounds_include_optimum()
    Optimizer(
        rosenboth,
        ub=ub,
        lb=lb,
        options={
            Options.THETA_MAX: 1.0,
            Options.MU: 0.0,
            Options.ETA: 0.0,
            Options.XTOL: 0,
            Options.MAXITER: 0,
            Options.MAXTIME: np.inf,
        },
    )
    with pytest.raises(TypeError):
        Optimizer(rosenboth, ub=ub, lb=lb, options={Options.MAXITER: np.inf})


def test_project_into_bounds():
    opt = Optimizer(
        _quadratic,
        ub=np.array([1.0, 1e6, np.inf, 2.0, 1.0]),
        lb=np.array([-1.0, -1e6, -np.inf, 2.0, 1.0 - 1e-15]),
        verbose=logging.ERROR,
    )
    mlb, mub = opt.degeneracy_margins()
    assert mub[0] == 1e2 * np.spacing(1.0)
    assert mub[1] == 1e2 * np.spacing(1e6)
    assert mlb[2] == mub[2] == 0
    # fixed variable and narrow box: margins shrink to half the width
    assert mlb[3] == mub[3] == 0
    assert np.isclose(mlb[4], 0.5e-15)
    assert mlb[4] == mub[4]

    x = np.array([1.0, -1e6 + 1e-11, 1e300, 2.0, 1.0])
    xp = opt.project_into_bounds(x)
    assert xp[0] == 1.0 - mub[0]
    assert xp[1] == -1e6 + mlb[1]
    assert xp[2] == 1e300
    assert xp[3] == 2.0
    assert opt.lb[4] <= xp[4] <= opt.ub[4]
    # interior points (and projected points) are not changed
    xi = np.array([0.5, 0.0, 0.0, 2.0, xp[4]])
    np.testing.assert_array_equal(opt.project_into_bounds(xi), xi)
    # roundoff-level violations are projected, larger ones are errors
    xr = np.array([1.0 + np.spacing(1.0), 0.0, 0.0, 2.0, 1.0])
    assert opt.project_into_bounds(xr)[0] == 1.0 - mub[0]
    with pytest.raises(RuntimeError, match='Exceeded upper bounds'):
        opt.project_into_bounds(np.array([1.0 + 1e-10, 0.0, 0.0, 2.0, 1.0]))


def test_returned_fval_matches_returned_x():
    """
    the non-degeneracy shift must be applied before the function is
    evaluated, otherwise the returned function value and gradient belong to
    a different point than the returned x
    """

    def fun(x):
        d = x - 2.0
        return float(0.5e8 * d.dot(d)), 1e8 * d, 1e8 * np.eye(len(x))

    opt = Optimizer(
        fun,
        ub=np.array([1.0]),
        lb=np.array([-1.0]),
        verbose=logging.ERROR,
        options={Options.FATOL: 0, Options.FRTOL: 0},
    )
    fval, x, grad, _ = opt.minimize(np.zeros(1))
    assert opt.exitflag == fides.ExitFlag.GTOL
    assert fval == fun(x)[0]
    np.testing.assert_array_equal(grad, fun(x)[1])
    assert x[0] < 1.0
    assert 1.0 - x[0] <= 2 * opt.degeneracy_margins()[1][0]


def test_gtol_at_margin_with_large_active_gradient():
    """
    a variable that sits on the degeneracy margin of the bound its gradient
    points towards is first-order optimal. The scaled gradient
    ``v * g = margin * g`` can exceed gatol for large gradients, which
    previously ended in DELTA_TOO_SMALL instead of GTOL
    """

    def fun(x):
        d = x - np.array([2.0, 0.5])
        scale = np.array([1e8, 1.0])
        return (
            float(0.5 * (scale * d).dot(d)),
            scale * d,
            np.diag(scale),
        )

    opt = Optimizer(
        fun,
        ub=np.ones(2),
        lb=-np.ones(2),
        verbose=logging.ERROR,
        options={Options.FATOL: 0, Options.FRTOL: 0},
    )
    fval, x, grad, _ = opt.minimize(np.zeros(2))
    assert opt.exitflag == fides.ExitFlag.GTOL
    assert x[0] < 1.0
    assert np.isclose(x[1], 0.5, atol=1e-6)

    # fuzz seed 417: bounded 3D quadratic with optimum at the boundary,
    # exact hessian, 2D/truncate ended in DELTA_TOO_SMALL at the optimum
    a_mat = np.array(
        [
            [254.47025190187892, -957.7507076730951, -89.35613001751094],
            [-957.7507076730951, 5567.74575936696, 610.7217365155356],
            [-89.35613001751095, 610.7217365155357, 94.36382195105466],
        ]
    )
    c = np.array([95295.05872705032, -788707.4313011734, 363651.61546168785])

    def quad(x):
        d = x - c
        g = a_mat @ d
        return 0.5 * d @ g, g, a_mat.copy()

    opt = Optimizer(
        quad,
        ub=np.array([np.inf, 2.129356493655654, -1.6491456920060925]),
        lb=np.array(
            [-1.8124686425827827, -1.1467974443098234, -2.72748591691]
        ),
        verbose=logging.ERROR,
        options={
            Options.SUBSPACE_DIM: SubSpaceDim.TWO,
            Options.STEPBACK_STRAT: StepBackStrategy.TRUNCATE,
        },
    )
    fval, x, grad, _ = opt.minimize(
        np.array(
            [0.1685990186836514, -1.1467974443065474, -1.9781779767178744]
        )
    )
    assert opt.exitflag == fides.ExitFlag.GTOL
    assert np.isclose(x[0], 2936056.4128618506, rtol=1e-6)
    assert np.all(x >= opt.lb)
    assert np.all(x <= opt.ub)


@pytest.mark.parametrize('subspace_dim', [SubSpaceDim.TWO, SubSpaceDim.FULL])
def test_reflected_step_roundoff_stays_in_bounds(subspace_dim):
    """
    fuzz seed 194: multiply reflected steps with a base point one ulp from
    the bound left the box by roundoff and aborted the run with
    EXCEEDED_BOUNDARY
    """
    from scipy.optimize import rosen, rosen_der

    lb = np.array(
        [
            -0.10513910090839354,
            -3.8847815401692245e-06,
            -1.0890332360853343e-06,
            -38.43034911505484,
            -777.5199314672093,
            -693.936612481771,
            -0.004468839218146271,
            -2654.231323851368,
            -3.3397342719787193e-06,
            -0.005747499251648356,
            -4.340574279133467e-05,
            -0.09619101501202056,
        ]
    )
    ub = np.array(
        [
            0.0003678849437744267,
            0.0031272252641871374,
            0.02583814866058093,
            69974.720994837,
            13302.0595926898,
            246641.273308562,
            33.50444796803692,
            4578.3918179615075,
            13911.60376445779,
            1.7827454502346366e-06,
            20419.003332247426,
            0.1965329469257042,
        ]
    )
    x0 = np.array(
        [
            -0.06134901875737717,
            0.00019035859215926556,
            0.009247738171127349,
            67678.10966826047,
            1135.3459417074257,
            16857.81267011807,
            13.997348539895688,
            884.4014586935302,
            6762.988218659159,
            -0.0046606882604397675,
            10996.565258243538,
            -0.09619101501202056,
        ]
    )
    opt = Optimizer(
        lambda x: (rosen(x), rosen_der(x)),
        ub=ub,
        lb=lb,
        verbose=logging.ERROR,
        hessian_update=SR1(),
        options={
            Options.MAXITER: 500,
            Options.SUBSPACE_DIM: subspace_dim,
            Options.STEPBACK_STRAT: StepBackStrategy.REFLECT,
        },
    )
    fval, x, grad, _ = opt.minimize(x0)
    assert np.all(x >= lb)
    assert np.all(x <= ub)
    assert opt.exitflag > 0
    assert np.isclose(fval, 10.913558700882975, rtol=1e-6)


@pytest.mark.parametrize(
    'stepback',
    [
        StepBackStrategy.SINGLE_REFLECT,
        StepBackStrategy.REFLECT,
        StepBackStrategy.TRUNCATE,
        StepBackStrategy.MIXED,
    ],
)
@pytest.mark.parametrize('subspace_dim', [SubSpaceDim.TWO, SubSpaceDim.FULL])
def test_several_active_bounds(subspace_dim, stepback):
    """
    gh-78: with reflect_single, the reflected segment is cut short by the
    second active bound. The step is then shorter than the radius, the
    radius was never increased and the run crawled to MAXITER.
    """
    hess = np.diag([2.0, 1.0, 1.5, 1.2])
    c = np.array([-1000.0, 1000.0, -1000.0, 1000.0])

    def fun(x):
        d = x - c
        g = hess @ d
        return 0.5 * d @ g, g, hess

    opt = Optimizer(
        fun,
        ub=np.full(4, np.inf),
        lb=np.array([-1.0, -np.inf, -1.0, -np.inf]),
        verbose=logging.WARNING,
        options={
            Options.MAXITER: 100,
            Options.SUBSPACE_DIM: subspace_dim,
            Options.STEPBACK_STRAT: stepback,
        },
    )
    fval, x, _, _ = opt.minimize(np.array([-1.0, 0.0, -1.0, 0.0]))
    assert opt.exitflag > 0
    assert np.isclose(fval, 1746501.75, rtol=1e-8)
    assert np.allclose(x, [-1.0, 1000.0, -1.0, 1000.0], rtol=1e-6)
