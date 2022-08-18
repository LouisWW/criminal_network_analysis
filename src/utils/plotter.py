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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import powerlaw
import seaborn as sns
from config.config import ConfigParser
from cycler import cycler
from PIL import Image
from PIL import PngImagePlugin
from utils.stats import get_correlation
from utils.tools import DirectoryFinder
from utils.tools import timestamp


class Plotter(ConfigParser):
    """This class takes care of all the plotting generated in this project."""

    def __init__(self) -> None:
        """Inherit from Configparser and set default plotting param."""
        super().__init__()

        # Making sure all the plots have the same parameters
        # plt.style.use('ggplot')
        plt.rcParams["figure.figsize"] = (15, 13)
        plt.rcParams["figure.autolayout"] = True
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["xtick.major.size"] = 5.0
        plt.rcParams["xtick.minor.size"] = 3.0
        plt.rcParams["ytick.major.size"] = 5.0
        plt.rcParams["ytick.minor.size"] = 3.0
        plt.rcParams["axes.labelsize"] = "x-large"
        plt.rcParams["axes.titlesize"] = "x-large"
        plt.rcParams["xtick.labelsize"] = "x-large"
        plt.rcParams["ytick.labelsize"] = "x-large"
        mpl.rcParams["lines.linewidth"] = 2
        mpl.rcParams["lines.markersize"] = 4

        # Change the default color list
        mpl.rcParams["axes.prop_cycle"] = cycler(color="gbrgmyc")
        mpl.rcParams["figure.dpi"] = 100
        mpl.rcParams["savefig.dpi"] = 300
        mpl.rcParams["axes.spines.top"] = False
        mpl.rcParams["axes.spines.right"] = False

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
            gt.graph_draw(
                network,
                pos=pos,
                output=f"{DirectoryFinder().result_dir_fig}{network.graph_properties.name}.png"
                if self.args.save
                else None,
            )

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
                output=f"{DirectoryFinder().result_dir_fig}{network.graph_properties.name}.png"
                if self.args.save
                else None,
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
        plt.tight_layout()

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
        y_data_to_plot: List[str],
        x_data_to_plot: str = None,
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

        for data in y_data_to_plot:
            if data not in dict_data.keys():
                raise KeyError(f"Given key doesn't exist,{dict_data.keys()=}")

            if x_data_to_plot:
                ax.plot(
                    dict_data[x_data_to_plot],
                    dict_data[data],
                    label=data.replace("_", " ").capitalize(),
                )
            else:
                ax.plot(dict_data[data], label=data.replace("_", " ").capitalize())
            if "plot_deviation" in kwargs:
                if kwargs["plot_deviation"] == "std":  # standard deviation
                    dev = data.replace("mean", "std")
                elif kwargs["plot_deviation"] == "sem":  # standard error of the mean
                    dev = data.replace("mean", "sem")

                if x_data_to_plot:
                    upper_dev = np.array(dict_data[data]) + np.array(dict_data[dev])
                    lower_dev = np.array(dict_data[data]) - np.array(dict_data[dev])

                    # if values are above 1 or below 0 is not possible
                    if "mean_ratio" in data:
                        upper_dev = np.where(upper_dev > 1, 1, upper_dev)
                        lower_dev = np.where(lower_dev < 0, 0, lower_dev)

                    ax.fill_between(
                        dict_data[x_data_to_plot],
                        lower_dev,
                        upper_dev,
                        alpha=0.5,
                    )
                else:
                    ax.fill_between(
                        range(0, len(dict_data[data])),
                        lower_dev,
                        upper_dev,
                        alpha=0.5,
                    )

        if "title" in kwargs:
            ax.set_title(kwargs["title"].replace("_", " ").capitalize())
        if "xlabel" in kwargs:
            ax.set_xlabel(kwargs["xlabel"].replace("_", " ").capitalize())
        if "ylabel" in kwargs:
            ax.set_ylabel(kwargs["ylabel"].replace("_", " ").capitalize())

        # set legend
        ax.legend()

        if self.args.save:
            fig_name = (
                DirectoryFinder().result_dir_fig
                + "population_ration_"
                + timestamp()
                + ".png"
            )
            plt.savefig(fig_name, dpi=300)
            if "meta_simulator" in kwargs:
                meta_str_sim = dict(
                    {
                        str(key): str(value)
                        for key, value in kwargs["meta_sim"].__dict__.items()
                    }
                )
                meta = PngImagePlugin.PngInfo()
                for x in meta_str_sim:
                    meta.add_text(x, meta_str_sim[x])
                im = Image.open(fig_name)
                im.save(fig_name, "png", pnginfo=meta)
        else:
            plt.show()
            return ax

    def plot_hist(
        self,
        dict_data: DefaultDict[str, List[Any]],
        y_data_to_plot: List[str],
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
        _, axs = plt.subplots(1, len(y_data_to_plot))
        if axs is None:
            axs = plt.gca()

        keys_diff_structure = list(dict_data.keys())

        for data, ax in zip(y_data_to_plot, axs):
            if data not in dict_data[keys_diff_structure[0]].keys():
                raise KeyError(
                    f"Given key doesn't exist,{dict_data[keys_diff_structure[0]].keys()=}"
                )

            color_list = iter(list(sns.color_palette("rocket")))
            for key_diff_structure in keys_diff_structure:
                color = next(color_list)
                sns.histplot(
                    dict_data[key_diff_structure][data],
                    kde=True,
                    color=color,
                    stat="probability",
                    label=key_diff_structure,
                    ax=ax,
                )

                if "title" in kwargs:
                    ax.set_title(data.replace("_", " ").capitalize())
                if "ylabel" in kwargs:
                    ax.set_ylabel("probability".capitalize())
                # set legend
                ax.legend()

        if self.args.save:
            fig_name = (
                DirectoryFinder().result_dir_fig
                + "correlation_fig"
                + "_"
                + timestamp()
                + ".png"
            )
            plt.savefig(fig_name, dpi=300, bbox_inches="tight")
            return ax
        else:
            plt.show()
            return ax

    def plot_phase_diag(
        self,
        grid: np.ndarray,
        x_range: np.ndarray,
        y_range: np.ndarray,
        param_x: str,
        param_y: str,
    ) -> plt.Axes:
        """Generate a phase diagram of param1 and param2.

        The colors correspond to the dominant status at the end of the run

        Args:
            grid (np.ndarray): 2-d array containing the dominant status for
                                    each combination of param1 and param2
            x_range (np.ndarray) : range of x
            y_range (np.ndarray) : range of
            param_x (str): parameter x of the model
            parma_y (str): parameter y of the model

        Returns:
            plt.Axes: phase diagram figure
        """
        # translate array into rgb code
        rgb_array = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=int)
        for x_i in range(0, grid.shape[0]):
            for y_i in range(0, grid.shape[1]):
                if grid[x_i, y_i] == "mean_ratio_criminal":
                    rgb_array[x_i, y_i, 0] = 255
                    rgb_array[x_i, y_i, 1] = 0
                    rgb_array[x_i, y_i, 2] = 0
                elif grid[x_i, y_i] == "mean_ratio_wolf":
                    rgb_array[x_i, y_i, 0] = 0
                    rgb_array[x_i, y_i, 1] = 0
                    rgb_array[x_i, y_i, 2] = 255
                elif grid[x_i, y_i] == "mean_ratio_honest":
                    rgb_array[x_i, y_i, 0] = 0
                    rgb_array[x_i, y_i, 1] = 255
                    rgb_array[x_i, y_i, 2] = 0

        _, ax = plt.subplots()
        ax.imshow(
            rgb_array, extent=[min(y_range), max(y_range), max(x_range), min(x_range)]
        )

        cmap = {1: [1, 0, 0, 1], 3: [0, 1, 0, 1], 2: [0, 0, 1, 1]}
        labels = {1: "criminal", 2: "lone wolf", 3: "honest"}
        # create patches as legend
        patches = [mpatches.Patch(color=cmap[i], label=labels[i]) for i in cmap]
        plt.legend(handles=patches)

        # Add the \ to the param to print it in latex format
        if param_x in ["beata_c", "beta_s", "beta_h", "delta", "tau", "gamma"]:
            param_x = "\\" + param_x
        if param_y in ["beata_c", "beta_s", "beta_h", "delta", "tau", "gamma"]:
            param_y = "\\" + param_y

        ax.set_xlabel(fr"${param_y}$")
        ax.set_ylabel(fr"${param_x}$")
        return ax

    def plot_lines_comparative(
        self,
        dict_data: DefaultDict[str, DefaultDict[str, List[int]]],
        y_data_to_plot: List[str],
        x_data_to_plot: str = None,
        *args: str,
        **kwargs: Any,
    ) -> plt.Axes:
        """Plot line graph from data points.

        Args:
            dict_data (DefaultDict[str, DefaultDict[str, List[int]]]): Contains all the data
            data_to_plot (List[str]): Defines which data to choose from the dict_data
        Returns:
            plt.Axes: matplotlib axes object
        """
        _, axs = plt.subplots(1, len(y_data_to_plot))
        if axs is None:
            axs = plt.gca()

        keys_diff_structure = list(dict_data.keys())

        for data, ax in zip(y_data_to_plot, axs):
            if data not in dict_data[keys_diff_structure[0]].keys():
                raise KeyError(
                    f"Given key doesn't exist,{dict_data[keys_diff_structure[0]].keys()=}"
                )

            color_list = iter(list(sns.color_palette("Set2")))
            for key_diff_structure in keys_diff_structure:
                color = next(color_list)
                deviation = data.replace("mean", "sem")
                ax.errorbar(
                    dict_data[key_diff_structure][x_data_to_plot],
                    dict_data[key_diff_structure][data],
                    yerr=dict_data[key_diff_structure][deviation],
                    color=color,
                    capsize=5,
                    label=key_diff_structure,
                )

                ax.set_xlabel("Rounds", weight="bold")
                ax.set_ylabel(data.replace("_", " ").capitalize(), weight="bold")
                # set legend
                ax.legend()
                ax.grid(alpha=0.5, linestyle=":")
                ax.set_aspect(1.5 * np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()))

        if self.args.save:
            fig_name = (
                DirectoryFinder().result_dir_fig
                + y_data_to_plot
                + "_"
                + timestamp()
                + ".png"
            )
            plt.savefig(fig_name, dpi=300, bbox_inches="tight")
            return ax
        else:
            plt.show()
            return ax

    def plot_lines_correlation(
        self,
        dict_data: DefaultDict[str, DefaultDict[str, List[int]]],
        y_data_to_plot: str,
        x_data_to_plot: str,
        *args: str,
        **kwargs: Any,
    ) -> plt.Axes:
        """Plot line graph from correlation points with correlatio factor.

        Args:
            dict_data (DefaultDict[str, DefaultDict[str, List[int]]]): Contains all the data
            data_to_plot (List[str]): Defines which data to choose from the dict_data
        Returns:
            plt.Axes: matplotlib axes object
        """
        keys = list(dict_data.keys())

        mpl.rcParams["axes.spines.top"] = True
        mpl.rcParams["axes.spines.right"] = True

        _, axs = plt.subplots(1, len(y_data_to_plot))

        if axs is None:
            axs = plt.gca()

        for centrality_measure, ax in zip(y_data_to_plot, axs):

            line_plot_style = iter(["k-", "k--", "k.-"])
            for key in keys:
                corr = get_correlation(
                    dict_data[key]["df_total"][x_data_to_plot],
                    dict_data[key]["df_total"][centrality_measure],
                )

                m, b = np.polyfit(
                    dict_data[key]["df_total"][x_data_to_plot],
                    dict_data[key]["df_total"][centrality_measure],
                    1,
                )

                ax.plot(
                    dict_data[key]["df_total"][x_data_to_plot],
                    m * dict_data[key]["df_total"][x_data_to_plot] + b,
                    next(line_plot_style),
                    label=key + f" --- {corr=}",
                )
                ax.set_xlabel("Criminal likelihood", weight="bold")
                ax.set_ylabel(centrality_measure.capitalize(), weight="bold")
                ax.patch.set_edgecolor("black")
                ax.patch.set_linewidth("2")
                ax.set_aspect(1.5 * np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()))
                ax.legend(loc="upper right", fontsize="x-small")

        plt.tight_layout()

        if self.args.save:
            fig_name = (
                DirectoryFinder().result_dir_fig
                + "correlation_fig"
                + "_"
                + timestamp()
                + ".png"
            )
            plt.savefig(fig_name, dpi=300, bbox_inches="tight")
            return ax
        else:
            plt.show()
            return ax
