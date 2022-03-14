"""This script's contains the different logger modules.

The logger allows for a better printing overview

__author__ = Louis Weyland
__date__   = 13/02/2022
"""
import logging


class Logger:
    """Logger class to print various statements."""

    def __init__(self, level="WARNING"):
        """Set the logger."""
        self.logger = logging.getLogger("logger")
        self.logger.setLevel("DEBUG")
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s"
        )
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def testing(self, integ: int):
        """Test some stuff."""
        return integ + 2
