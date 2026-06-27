"""
utils/logger.py
---------------
Provides a single `get_logger()` factory so every module in the project
shares the same log format and respects the same log level.

Using Python's built-in logging (rather than bare print statements) means
you can silence verbose output in production by setting LOG_LEVEL=WARNING
without touching any other file.
"""

import logging
import os
import sys
from datetime import datetime


def get_logger(name: str, log_to_file: bool = False) -> logging.Logger:
    """
    Return a named logger configured with a consistent format.

    Parameters
    ----------
    name : str
        Typically ``__name__`` from the calling module.
    log_to_file : bool
        When True a timestamped log file is also written to ``outputs/logs/``.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if the logger was already configured
    # (this happens when a module is imported more than once in the same run).
    if logger.handlers:
        return logger

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — always present
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    if log_to_file:
        log_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "outputs", "logs",
        )
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(log_dir, f"{timestamp}_{name}.log")
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
