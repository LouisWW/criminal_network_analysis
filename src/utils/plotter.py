"""
This script's intention is generate plots.

More specifically, generated data is visualized.

__author__ = Louis Weyland
__date__   = 13/02/2022
"""
from typing import Any
from typing import DefaultDict
from typing import List

import graph_tool.all as gt
import matplotlib as mpl
import matplotlib.pyplot as plt
import powerlaw
from cycler import cycler
from src.config.config import ConfigParser


class Plotter(ConfigParser):
    """This class takes care of all the plotting generated in this project."""

    def __init__(self) -> None:
        """Inherit from Configparser and set default plotting param."""
        super().__init__()

        # Making sure all the plots have the same parameters
        plt.rcParams["figure.figsize"] = (5, 4)
        plt.rcParams["figure.autolayout"] = True

        # Change the default color list
        mpl.rcParams["axes.prop_cycle"] = cycler(color="krbgmyc")
        mpl.rcParams["figure.dpi"] = 100
        mpl.rcParams["savefig.dpi"] = 300

    def draw_network(
        self,
        network: gt.Graph,
        color_vertex_property: str = None,
    ) -> None:
        """Visualizes the Network.

        If vertex_property is given, then the vertex are colored based on their
        property
        """
        assert isinstance(network, gt.Graph), "network type is not from graph-tool"

        # Define pos to circumvent error produced by graph_tool
        pos = gt.sfdp_layout(network)

        # draw circular
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

        elif self.args.draw_network == "n" and color_vertex_property is None:
            gt.graph_draw(network, pos=pos)

        elif self.args.draw_network == "n" and color_vertex_property is not None:
            # Add a color map corresponding to the chosen vertex_property
            # if color_vertex_property is not None:
            network, _ = self.get_color_map(
                network, color_vertex_property=color_vertex_property
            )
            gt.graph_draw(
                network,
                pos=pos,
                vertex_fill_color=network.vertex_properties[color_vertex_property],
            )

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

    def get_color_map(
        self, network: gt.Graph, color_vertex_property: str = None
    ) -> gt.PropertyMap:
        """Define the color of the vertex based on the vertex property."""
        if color_vertex_property == "state_color":
            # c = red, h = blue, w = green
            color_map = {"c": (1, 0, 0, 1), "h": (0, 0, 1, 1), "w": (0, 1, 0, 1)}
            color_code = network.new_vertex_property("vector<double>")
            network.vertex_properties["state_color"] = color_code
            for v in network.vertices():
                color_code[v] = color_map[network.vertex_properties["state"][v]]
            return network, color_code

        elif color_vertex_property == "group_color":
            # For now this color map is create outside
            return network, None
        else:
            return None

    def plot_lines(
        self,
        dict_data: DefaultDict[str, List[int]],
        data_to_plot: List[str],
        *args: str,
        **kwargs: Any,
    ) -> plt.Axes:
        """Plot line graph from data  points.

        Args:
            dict_data (DefaultDict[str, List[int]]): Contains all the data
            data_to_plot (List[str]): Defines which data to choose from the dict_data

        Returns:
            plt.Axes: matplotlib axes object
        """
        _, ax = plt.subplots()
        if ax is None:
            ax = plt.gca()

        for data in data_to_plot:
            if data not in dict_data.keys():
                raise KeyError(f"Given key doens't exist,{dict_data.keys()=}")
            ax.plot(dict_data[data], label=data)
            if "plot_std" in kwargs:
                std = data.replace("mean", "std")
                ax.fill_between(
                    range(0, len(dict_data[data])),
                    dict_data[data] - dict_data[std],
                    dict_data[data] + dict_data[std],
                    alpha=0.5,
                )

        if "title" in kwargs:
            ax.set_title(kwargs["title"])
        if "xlabel" in kwargs:
            ax.set_xlabel(kwargs["xlabel"])
        if "ylabel" in kwargs:
            ax.set_ylabel(kwargs["ylabel"])

        # set legend
        ax.legend()
        return ax

    def plot_hist(
        self,
        dict_data: DefaultDict[str, List[Any]],
        data_to_plot: List[str],
        n_bins: int = None,
        *args: str,
        **kwargs: Any,
    ) -> plt.Axes:
        """Plot a histogram from data points.

        Args:
            dict_data (DefaultDict[str, List[Any]]): Contains all the data
            data_to_plot (List[str]): Defines which data to choose from the dict_data

        Returns:
            plt.Axes: matplotlib axes object
        """
        _, ax = plt.subplots()
        if ax is None:
            ax = plt.gca()

        for data in data_to_plot:
            if data not in dict_data.keys():
                raise KeyError(f"Given key doens't exist,{dict_data.keys()=}")
            ax.hist(dict_data[data], label=data, bins=n_bins)

        if "title" in kwargs:
            ax.set_title(kwargs["title"])
        if "xlabel" in kwargs:
            ax.set_xlabel(kwargs["xlabel"])
        if "ylabel" in kwargs:
            ax.set_ylabel(kwargs["ylabel"])

        # set legend
        ax.legend()
        return ax
