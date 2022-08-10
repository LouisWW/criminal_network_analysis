"""This file contains random functions such as creating a timestamp or get the directory.

__author__ = Louis Weyland
__date__   = 10/08/2022
"""
import datetime
import os


def timestamp() -> str:
    """Return a timestamp."""
    e = datetime.datetime.now()
    return e.strftime("%d-%m-%Y-%H-%M")


class DirectoryFinder:
    """Contain all the directories."""

    def __init__(self) -> None:
        """Get the directories."""
        path = os.path.dirname(os.path.realpath(__file__))
        self.main_dir = os.path.abspath(os.path.join(path, "../"))  # par_dir = ../src/
        self.result_dir = self.main_dir + "/results"
        self.result_dir_data = self.result_dir + "/data"
