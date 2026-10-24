"""
Logging
-------
This module provides the machinery that is used to display progress of the
optimizer as well as debugging information

Every optimizer logs to its own logger ``fides.<n>``, which prints to
``sys.stderr``. These loggers propagate to the ``fides`` logger, so
applications that configure the root logger and do not want fides output
twice can call ``logging.getLogger('fides').propagate = False``.

:var logger:
    logging.Logger instance that can be used throughout fides
"""

import logging
import threading
import weakref

logger_count = 0
_logger_count_lock = threading.Lock()
# loggers whose owner has been garbage collected, handed out again by
# create_logger, so the number of loggers is bounded by the number of owners
# that exist at the same time
_free_loggers: list[logging.Logger] = []


def create_logger(level: int, owner: object | None = None) -> logging.Logger:
    """
    Creates a logger instance. To avoid unnecessary locks during
    multithreading, different logger instance should be created for every
    optimizer that may run concurrently.

    :param level:
        logging level

    :param owner:
        object that uses the logger. Once it is garbage collected, the
        logger is handed out again by subsequent calls. Without an owner,
        the logger is never reused.

    :return:
        logger instance
    """
    global logger_count
    try:
        logger = _free_loggers.pop()
    except IndexError:
        with _logger_count_lock:
            logger_count += 1
            # add logger count to differentiate between different fides
            # optimization instances and avoid deadlocks
            name = f'fides.{logger_count}'
        logger = logging.getLogger(name)
    else:
        # reset a reused logger to the state of a new one, its handler would
        # also still write to the sys.stderr from when it was created
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        logger.filters.clear()
        logger.propagate = True
        logger.disabled = False
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s fides(%(levelname)s) %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(level)
    if owner is not None:
        # the bound method keeps the pool alive during interpreter shutdown
        weakref.finalize(owner, _free_loggers.append, logger).atexit = False
    return logger
