"""
This script's intention is generate plots.

More specifically, generated data is visualized.

__author__ = Louis Weyland
__date__   = 13/02/2022
"""
from typing import List

import graph_tool.all as gt
import matplotlib as mpl
import matplotlib.pyplot as plt
import powerlaw
from config.config import ConfigParser
from cycler import cycler


class Plotter(ConfigParser):
    """This class takes care of all the plotting generated in this project."""

    def __init__(self) -> None:
        """Inherit from Configparser and set default plotting param."""
        super().__init__()

        # Making sure all the plots have the same parameters
        plt.rcParams["figure.figsize"] = (10, 7)
        plt.rcParams["figure.autolayout"] = True

        # Change the default color list
        mpl.rcParams["axes.prop_cycle"] = cycler(color="krbgmyc")
        mpl.rcParams["figure.dpi"] = 100
        mpl.rcParams["savefig.dpi"] = 300

    def draw_network(self, network: gt.Graph) -> None:
        """Visualizes the Network."""
        # draw circular
        assert isinstance(network, gt.Graph), "network type is not from graph-tool"
        if self.args.draw_network == "c":
            g = gt.GraphView(network)
            state = gt.minimize_nested_blockmodel_dl(g)
            t = gt.get_hierarchy_tree(state)[0]
            tpos = pos = gt.radial_tree_layout(t, t.vertex(t.num_vertices() - 1))
            cts = gt.get_hierarchy_control_points(g, t, tpos)
            pos = g.own_property(tpos)
            b = state.levels[0].b
            shape = b.copy()
            shape.a %= 14
            gt.graph_draw(
                g,
                pos=pos,
                vertex_fill_color=b,
                vertex_shape=shape,
                edge_control_points=cts,
                edge_color=[0, 0, 0, 0.3],
                vertex_anchor=0,
            )

        elif self.args.draw_network == "n":
            gt.graph_draw(network)

    def plot_log_log(self, data: List[float], x_label: str, y_label: str) -> plt.Axes:
        """
        Plot the data in a log-log scale to visualize the powerlaw.

        Important: Sort the data in ascending way
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
