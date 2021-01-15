"""
Logging
-------
This module provides the machinery that is used to display progress of the
optimizer as well as debugging information

:var logger:
    logging.Logger instance that can be used throughout fides
"""

import logging

logger_count = 0


def create_logger(level: int) -> logging.Logger:
    """
    Creates a logger instance. To avoid unnecessary locks during
    multithreading, different logger instance should be created for every

    :param level:
        logging level

    :return:
        logger instance
    """
    global logger_count
    logger_count += 1
    # add logger count to differentiate between different fides
    # optimization instances and avoid deadlocks
    logger = logging.getLogger(f'fides_{logger_count}')
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - fides - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(level)
    return logger
