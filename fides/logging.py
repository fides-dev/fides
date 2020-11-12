"""
Logging
-------
This module provides the machinery that is used to display progress of the
optimizer as well as debugging information
"""

import logging

logger = logging.getLogger('fides')
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
ch.setFormatter(formatter)
logger.addHandler(ch)
