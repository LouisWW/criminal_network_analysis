"""
This script's intention is to get the generate plots to
visualize the data generated in this project

__author__ = Louis Weyland
__date__   = 13/02/2022
"""
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import powerlaw
from cycler import cycler


class Plotter:
    """This class takes care of all the plotting generated in this project"""

    def __init__(self) -> None:

        # Making sure all the plots have the same parameters
        plt.rcParams["figure.figsize"] = (10, 7)
        plt.rcParams["figure.autolayout"] = True

        # Change the default color list
        mpl.rcParams["axes.prop_cycle"] = cycler(color="krbgmyc")
        mpl.rcParams["figure.dpi"] = 100
        mpl.rcParams["savefig.dpi"] = 300

    def plot_log_log(self, data: List[float], x_label: str, y_label: str) -> plt.Axes:
        """
        Sort the data in ascending way and plots the
        data in a log-log scale to better visualize the powerlaw
        effect
        """
        data = sorted(data, reverse=True)
        fit = powerlaw.Fit(data)

        # init object
        fig, ax = plt.subplots()

        # get scatter points
        x, y = powerlaw.pdf(data, linear_bins=True)
        ind = y > 0
        y = y[ind]
        x = x[:-1]
        x = x[ind]

        # plot
        ax.scatter(x, y, color="r")
        fit.power_law.plot_pdf(ax=ax, linestyle="--", color="k", label="Power law fit")
        fit.plot_pdf(ax=ax, original_data=True, color="b", label="PDF fit")

        ax.legend()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        return ax
