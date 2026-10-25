import contextlib
import io
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from fides import Optimizer


def fun(x):
    return np.sum((x - 1) ** 2), 2 * (x - 1), 2 * np.eye(len(x))


def make_optimizer(verbose=logging.WARNING):
    n = 2
    return Optimizer(
        fun, lb=-5 * np.ones(n), ub=5 * np.ones(n), verbose=verbose
    )


def fides_loggers():
    return [
        logger
        for name, logger in logging.Logger.manager.loggerDict.items()
        # loggers were named fides_<n> before #81
        if name.startswith('fides') and isinstance(logger, logging.Logger)
    ]


def n_fides_handlers():
    return sum(len(logger.handlers) for logger in fides_loggers())


def test_loggers_bounded():
    # one optimizer per start, as in pyPESTO, used to register one logger
    # and one handler per start
    n_loggers = len(fides_loggers())
    n_handlers = n_fides_handlers()
    for _ in range(50):
        make_optimizer().minimize(np.zeros(2))
    assert len(fides_loggers()) <= n_loggers + 1
    assert n_fides_handlers() <= n_handlers + 1


def test_loggers_bounded_when_minimize_raises():
    n_loggers = len(fides_loggers())
    for _ in range(20):
        opt = make_optimizer()
        # creates the logger, then fails on the dimension of x0
        with pytest.raises(ValueError, match='entries'):
            opt.minimize(np.zeros(3))
    del opt
    assert len(fides_loggers()) <= n_loggers + 1


def test_loggers_bounded_by_concurrent_optimizers():
    n_workers = 4

    def run(_):
        make_optimizer().minimize(np.zeros(2))

    n_loggers = len(fides_loggers())
    with ThreadPoolExecutor(n_workers) as executor:
        list(executor.map(run, range(40)))
    assert len(fides_loggers()) <= n_loggers + n_workers


def test_concurrent_optimizers_have_own_logger():
    # avoids lock contention between threads, see #13
    opts = [make_optimizer() for _ in range(5)]
    for opt in opts:
        opt.minimize(np.zeros(2))
    assert len({id(opt.logger) for opt in opts}) == len(opts)
    handlers = [handler for opt in opts for handler in opt.logger.handlers]
    assert len({id(handler) for handler in handlers}) == len(opts)
    # configurable through the fides logger
    fides_logger = logging.getLogger('fides')
    for opt in opts:
        assert opt.logger.name.startswith('fides.')
        assert opt.logger.parent is fides_logger


def test_verbose_output_by_default(capsys):
    opt = Optimizer(fun, lb=-5 * np.ones(2), ub=5 * np.ones(2))
    opt.minimize(np.zeros(2))
    err = capsys.readouterr().err
    assert 'fides(INFO)' in err
    assert 'iter' in err
    assert 'Stopping as' in err


def test_reused_logger_is_reset():
    opt = make_optimizer(verbose=logging.INFO)
    opt.minimize(np.zeros(2))
    logger = opt.logger
    logger.addHandler(logging.NullHandler())
    logger.addFilter(lambda record: False)
    logger.disabled = True
    del opt

    # the reused logger writes to the current sys.stderr
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        opt = make_optimizer(verbose=logging.INFO)
        opt.minimize(np.zeros(2))
    assert opt.logger is logger
    assert len(logger.handlers) == 1
    assert not logger.filters
    assert not logger.disabled
    assert 'fides(INFO)' in err.getvalue()


def test_propagation_can_be_disabled():
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        make_optimizer(verbose=logging.INFO).minimize(np.zeros(2))
        assert records
        records.clear()
        logging.getLogger('fides').propagate = False
        make_optimizer(verbose=logging.INFO).minimize(np.zeros(2))
        assert not records
    finally:
        logging.getLogger('fides').propagate = True
        root.removeHandler(handler)
