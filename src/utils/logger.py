"""
This script's contains the different logger modules
which allows for a better printing overview

__author__ = Louis Weyland
__date__   = 13/02/2022
"""
import logging


class Logger:
    def __init__(self, level="WARNING"):
        # To debug the class, init the object with debug_level='DEBUG'
        # save the logging to a file
        self.logger = logging.getLogger("logger")
        self.logger.setLevel("DEBUG")
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s"
        )
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
