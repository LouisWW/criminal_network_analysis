"""
This script's intention is generate plots.

More specifically, generated data is visualized.

__author__ = Louis Weyland
__date__   = 13/02/2022
"""
import pickle
from typing import Any
from typing import DefaultDict
from typing import List
from typing import Union

import graph_tool.all as gt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import powerlaw
import seaborn as sns
from config.config import ConfigParser
from cycler import cycler
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
        plt.rcParams["figure.figsize"] = (20, 12)
        plt.rcParams["figure.autolayout"] = True
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["xtick.major.size"] = 5.0
        plt.rcParams["xtick.minor.size"] = 3.0
        plt.rcParams["ytick.major.size"] = 5.0
        plt.rcParams["ytick.minor.size"] = 3.0
        plt.rcParams["axes.labelsize"] = "xx-large"
        plt.rcParams["axes.titlesize"] = "x-large"
        plt.rcParams["xtick.labelsize"] = "xx-large"
        plt.rcParams["ytick.labelsize"] = "xx-large"
        mpl.rcParams["lines.linewidth"] = 2
        mpl.rcParams["lines.markersize"] = 4

        # Change the default color list
        mpl.rcParams["axes.prop_cycle"] = cycler(color="gbrgmyc")
        mpl.rcParams["figure.dpi"] = 100
        mpl.rcParams["savefig.dpi"] = 300
        mpl.rcParams["axes.spines.top"] = False
        mpl.rcParams["axes.spines.right"] = False

    def save_figure(sefl, fig_name: str, axs: plt.Axes) -> None:
        """Save figures to png files and pickle data."""
        plt.savefig(fig_name, dpi=300, bbox_inches="tight")
        with open(fig_name.replace("png", "pkl"), "wb") as fig:
            pickle.dump(axs, fig)

    def load_figure(self, fig_name: str) -> plt.Axes:
        """Load a pickle saved figure and returns an axes."""
        with open(fig_name, "rb") as fid:
            ax = pickle.load(fid)
        return ax

    def draw_network(
        self,
        network: gt.Graph,
        color_vertex_property: str = None,
    ) -> None:
        """Visualizes the Network.

        If vertex_property is given, then the vertex are colored based on their
        property.
        
        Args:
            network (gt.Graph): population network containing honest/lone wovles/criminals.
            color_vertex_property (str, optional): property to show in the plot ("status" for now). 
                                                    Defaults to None.
        """        
        assert isinstance(network, gt.Graph), "network type is not from graph-tool"

        # Define pos to circumvent error produced by graph_tool
        pos = gt.sfdp_layout(network)

        # draw circular
        if self.args.draw_network == "c":
            g = gt.GraphView(network)
            status = gt.minimize_nested_blockmodel_dl(g)
            t = gt.get_hierarchy_tree(status)[0]
            tpos = pos = gt.radial_tree_layout(t, t.vertex(t.num_vertices() - 1))
            cts = gt.get_hierarchy_control_points(g, t, tpos)
            pos = g.own_property(tpos)
            b = status.levels[0].b
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
        """Plot the data in a log-log scale to visualize the powerlaw.

        Important: Sort the data in ascending way
    
        Args:
            data (List[float]): containing the centrality (avg. degree,...) of each node
            x_label (str): x label
            y_label (str): y label

        Returns:
            plt.Axes: returns the figure object
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
        if color_vertex_property == "status_color":
            # c = red, h = blue, w = green
            color_map = {"c": (1, 0, 0, 1), "h": (0, 0, 1, 1), "w": (0, 1, 0, 1)}
            color_code = network.new_vertex_property("vector<double>")
            network.vertex_properties["status_color"] = color_code
            for v in network.vertices():
                color_code[v] = color_map[network.vertex_properties["status"][v]]
            return network, color_code

        elif color_vertex_property == "group_color":
            # For now this color map is create outside
            return network, None
        else:
            return None

    def plot_violin(
        self,
        dict_data: DefaultDict[str, List[int]],
        *args: str,
        **kwargs: Any,
    ) -> Union[plt.Axes, np.ndarray, np.generic]:
        """Plot violin graph from data  points.

        Args:
            dict_data (DefaultDict[str, List[Any]]): Contains all the data
            data_to_plot (List[str]): Defines which data to choose from the dict_data

        Returns:
            plt.Axes: matplotlib axes object
        """
        # data needs to be in df

        df = pd.DataFrame(columns=["structure", "link", "data"])

        for structure in dict_data.keys():
            for link in dict_data[structure].keys():
                for data_points in dict_data[structure][link]:
                    df = df.append(
                        {"structure": structure, "link": link, "data": data_points},
                        ignore_index=True,
                    )

        df["structure"] = df["structure"].astype(str)
        df["link"] = df["link"].astype(int)
        df["data"] = df["data"].astype(float)

        _, ax = plt.subplots()

        ax = sns.violinplot(
            data=df,
            x="link",
            y="data",
            hue="structure",
            width=0.7,
            palette="Set2",
            ax=ax,
            cut=0,
        )

        if "xlabel" in kwargs:
            ax.set_xlabel(kwargs["xlabel"].capitalize(), weight="bold", fontsize=35)
        if "ylabel" in kwargs:
            ax.set_ylabel(kwargs["ylabel"].capitalize(), weight="bold", fontsize=35)

        if "density_conv" in kwargs:
            ax.set_xticks(
                np.arange(0, len(kwargs["density_conv"])), labels=kwargs["density_conv"]
            )

        # set legend
        ax.legend(fancybox=True, shadow=True, fontsize=35)
        ax.tick_params(labelsize=35)
        plt.tight_layout()
        if self.args.save:
            fig_name = (
                DirectoryFinder().result_dir_fig
                + "violin_plot"
                + "_"
                + timestamp()
                + ".png"
            )
            self.save_figure(fig_name, ax)
            return ax
        else:
            plt.show()
            return ax

    def plot_lines(
        self,
        dict_data: DefaultDict[str, List[int]],
        y_data_to_plot: List[str],
        x_data_to_plot: str,
        *args: str,
        **kwargs: Any,
    ) -> Union[plt.Axes, np.ndarray, np.generic]:
        """Plot line graph from data  points.

        Args:
            dict_data (DefaultDict[str, List[Any]]): Contains all the data
            data_to_plot (List[str]): Defines which data to choose from the dict_data

        Returns:
            plt.Axes: matplotlib axes object
        """
        keys_diff_structure = list(dict_data.keys())
        _, axs = plt.subplots(1, len(keys_diff_structure))
        if axs is None:
            axs = plt.gca()

        if not isinstance(axs, (np.ndarray, np.generic)):
            axs = np.array([axs])

        for key_diff_structure, ax in zip(keys_diff_structure, axs):
            for data in y_data_to_plot:
                if data not in dict_data[keys_diff_structure[0]].keys():
                    raise KeyError(
                        f"Given key doesn't exist,{dict_data[keys_diff_structure[0]].keys()=}"
                    )
                print(key_diff_structure, data)
                print(
                    dict_data[key_diff_structure][x_data_to_plot],
                    dict_data[key_diff_structure][data],
                )
                ax.plot(
                    dict_data[key_diff_structure][x_data_to_plot],
                    dict_data[key_diff_structure][data],
                    label=data.replace("_", " ").capitalize(),
                )

                if "plot_deviation" in kwargs:
                    if kwargs["plot_deviation"] == "std":  # standard deviation
                        dev = data.replace("mean", "std")
                    elif (
                        kwargs["plot_deviation"] == "sem"
                    ):  # standard error of the mean
                        dev = data.replace("mean", "sem")

                    upper_dev = np.array(
                        dict_data[key_diff_structure][data]
                    ) + np.array(dict_data[key_diff_structure][dev])
                    lower_dev = np.array(
                        dict_data[key_diff_structure][data]
                    ) - np.array(dict_data[key_diff_structure][dev])

                    # if values are above 1 or below 0 is not possible
                    if "mean_ratio" in data:
                        upper_dev = np.where(upper_dev > 1, 1, upper_dev)
                        lower_dev = np.where(lower_dev < 0, 0, lower_dev)

                    ax.fill_between(
                        dict_data[key_diff_structure][x_data_to_plot],
                        lower_dev,
                        upper_dev,
                        alpha=0.5,
                    )

            # set label to percentage
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            if "ylim" in kwargs:
                ax.set_ylim(kwargs["ylim"][0], kwargs["ylim"][1])
            else:
                ax.set_ylim(0, 1)
            ax.ticklabel_format(
                axis="x", style="sci", scilimits=(0, 0), useMathText="True"
            )
            if "title" in kwargs:
                ax.set_title(
                    key_diff_structure.replace("_", " ").capitalize(), weight="bold"
                )
            if "xlabel" in kwargs:
                ax.set_xlabel(
                    kwargs["xlabel"].replace("_", " ").capitalize(), weight="bold"
                )
            if "ylabel" in kwargs:
                ax.set_ylabel(
                    kwargs["ylabel"].replace("_", " ").capitalize(), weight="bold"
                )

            if "tick_size" in kwargs:
                ax.tick_params(labelsize=kwargs["tick_size"])

            if "axes_size" in kwargs:
                ax.xaxis.label.set_size(kwargs["axes_size"])
                ax.yaxis.label.set_size(kwargs["axes_size"])

            if "legend_size" in kwargs:
                ax.legend(fancybox=True, shadow=True, fontsize=kwargs["legend_size"])
            else:
                # set legend
                ax.legend(
                    fancybox=True,
                    shadow=True,
                )

            # only do it if multiple plots are made
            if "square_plot" in kwargs:
                ax.set_aspect(np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()))

        plt.tight_layout()

        if self.args.save:
            fig_name = (
                DirectoryFinder().result_dir_fig
                + "population_ratio_"
                + timestamp()
                + ".png"
            )
            self.save_figure(fig_name, axs)
            return axs
        else:
            plt.show()
            return axs

    def plot_hist(
        self,
        dict_data: DefaultDict[str, List[Any]],
        y_data_to_plot: List[str],
        *args: str,
        **kwargs: Any,
    ) -> Union[plt.Axes, np.ndarray, np.generic]:
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

            color_list = iter(list(sns.color_palette("Set2")))
            for key_diff_structure in keys_diff_structure:
                color = next(color_list)
                sns.kdeplot(
                    dict_data[key_diff_structure][data],
                    color=color,
                    common_norm=True,
                    multiple="stack",
                    alpha=0.8,
                    linewidth=0,
                    label=key_diff_structure,
                    ax=ax,
                )

                if "xlabel" in kwargs:
                    ax.set_xlabel(
                        data.replace("mean_", "").replace("_", " ").capitalize(),
                        weight="bold",
                    )
                    if "information" in data:
                        ax.set_xlabel("Flow of information", weight="bold")

                # set legend
                ax.legend(fancybox=True, shadow=True)
                ax.set_aspect(np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()))

            if "ylabel" in kwargs:
                axs[0].set_ylabel("Density".capitalize(), weight="bold")
                axs[1].set_ylabel(" ".capitalize(), weight="bold")
                axs[2].set_ylabel(" ".capitalize(), weight="bold")

        plt.tight_layout()

        if self.args.save:
            fig_name = (
                DirectoryFinder().result_dir_fig
                + "topo_meas_hist"
                + "_"
                + timestamp()
                + ".png"
            )
            self.save_figure(fig_name, axs)
            return axs
        else:
            plt.show()
            return axs

    def plot_phase_diag(
        self,
        dict_data: DefaultDict[str, List[int]],
        *args: str,
        **kwargs: Any,
    ) -> plt.Axes:
        """Generate a phase diagram of param1 and param2.

        The colors correspond to the dominant status at the end of the run

        Returns:
            plt.Axes: phase diagram figure
        """
        structures = list(dict_data.keys())
        cases = list(dict_data[structures[0]].keys())

        mpl.rcParams["axes.spines.top"] = True
        mpl.rcParams["axes.spines.right"] = True
        plt.rcParams["axes.labelsize"] = "xx-large"
        plt.rcParams["axes.titlesize"] = "xx-large"
        plt.rcParams["xtick.labelsize"] = "large"
        plt.rcParams["ytick.labelsize"] = "large"
        fig, axs = plt.subplots(len(cases), len(structures))

        for case, k in zip(cases, range(axs.shape[0])):

            # data 1 normalizer=iter([Normalize(0,0.8),Normalize(0,0.03),Normalize(0,0.035)])
            # data 2 normalizer=iter([Normalize(0,0.75),Normalize(0,0.025),Normalize(0,0.05)])

            normalizer = iter(
                [Normalize(0, 0.8), Normalize(0, 0.03), Normalize(0, 0.035)]
            )
            for structure, i in zip(structures, range(axs.shape[1])):

                divider = make_axes_locatable(axs[k, i])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                mat = axs[k, i].matshow(
                    dict_data[structure][case]["ratio_criminal"],
                    aspect="auto",
                    interpolation="bilinear",
                    norm=next(normalizer),
                    cmap="viridis",
                    extent=[
                        min(dict_data[structure][case]["x_range"]),
                        max(dict_data[structure][case]["x_range"]),
                        max(dict_data[structure][case]["y_range"]),
                        min(dict_data[structure][case]["y_range"]),
                    ],
                    # norm=normalizer
                )
                fig.colorbar(mat, cax=cax, orientation="vertical")

                # Add the \ to the param to print it in latex format
                if dict_data[structure][case]["param_x"] in [
                    "beata_c",
                    "beta_s",
                    "beta_h",
                    "delta",
                    "tau",
                    "gamma",
                ]:
                    dict_data[structure][case]["param_x"] = (
                        "\\" + dict_data[structure][case]["param_x"]
                    )
                if dict_data[structure][case]["param_y"] in [
                    "beata_c",
                    "beta_s",
                    "beta_h",
                    "delta",
                    "tau",
                    "gamma",
                ]:
                    dict_data[structure][case]["param_y"] = (
                        "\\" + dict_data[structure][case]["param_y"]
                    )

                param_y = dict_data[structure][case]["param_y"]
                param_x = dict_data[structure][case]["param_x"]
                axs[k, i].set_xlabel(fr"${param_x}$")
                axs[k, i].set_ylabel(fr"${param_y}$", rotation=0)
                axs[k, i].grid(b=None)

        for ax, structure in zip(axs[0], structures):
            ax.set_title(structure.capitalize(), weight="bold")

        # plt.tight_layout()
        if self.args.save:

            fig_name = (
                DirectoryFinder().result_dir_fig
                + "phase_diag_"
                + param_x
                + "_"
                + param_y
                + "_"
                + timestamp()
                + ".png"
            )
            self.save_figure(fig_name, axs)
        else:
            plt.show()
            return ax

    def plot_lines_comparative(
        self,
        dict_data: DefaultDict[str, DefaultDict[str, List[int]]],
        y_data_to_plot: List[str],
        x_data_to_plot: str = None,
        *args: str,
        **kwargs: Any,
    ) -> Union[plt.Axes, np.ndarray, np.generic]:
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
                ax.ticklabel_format(
                    axis="x", style="sci", scilimits=(0, 0), useMathText="True"
                )
                ax.set_ylabel(
                    data.replace("mean_", "").replace("_", " ").capitalize(),
                    weight="bold",
                )
                if "information" in data:
                    ax.set_ylabel("Flow of information", weight="bold")
                # set legend
                ax.legend(
                    fancybox=True,
                    shadow=True,
                )
                ax.grid(alpha=0.5, linestyle=":")
                ax.set_aspect(np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()))

        plt.tight_layout()

        if self.args.save:
            fig_name = (
                DirectoryFinder().result_dir_fig
                + "topological_meas"
                + "_"
                + timestamp()
                + ".png"
            )
            self.save_figure(fig_name, axs)
            return axs
        else:
            plt.show()
            return axs

    def plot_lines_correlation(
        self,
        dict_data: DefaultDict[str, DefaultDict[str, List[int]]],
        y_data_to_plot: str,
        x_data_to_plot: str,
        *args: str,
        **kwargs: Any,
    ) -> Union[plt.Axes, np.ndarray, np.generic]:
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

        fig, axs = plt.subplots(1, len(y_data_to_plot))

        if axs is None:
            axs = plt.gca()

        for centrality_measure, ax in zip(y_data_to_plot, axs):

            line_plot_style = iter(["k-", "k--", "k:"])
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
                    # label=key + f" --- {corr=}",
                    label=corr,
                )

                ax.xaxis.set_major_formatter(mtick.PercentFormatter())
                ax.set_xlabel("Criminal likelihood", weight="bold")
                ax.set_ylabel(centrality_measure.capitalize(), weight="bold")
                ax.patch.set_edgecolor("black")
                ax.patch.set_linewidth("2")
                ax.set_aspect(np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()))
                ax.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.05),
                    ncol=3,
                    fancybox=True,
                    shadow=True,
                )

        line_labels = ["Preferential", "Random", "Small-world"]
        fig.legend(
            ax.get_lines(),  # The line objects
            labels=line_labels,  # The labels for each line
            loc="upper center",  # Position of legend
            bbox_to_anchor=(0.5, 0.8),
            borderaxespad=0.1,  # Small spacing around legend box
            ncol=3,
            fontsize="large",
        )

        plt.tight_layout()

        if self.args.save:
            fig_name = (
                DirectoryFinder().result_dir_fig
                + "correlation_fig"
                + "_"
                + timestamp()
                + ".png"
            )
            self.save_figure(fig_name, axs)
            return axs
        else:
            plt.show()
            return axs

    def plot_lines_correlation_grid(
        self,
        dict_data: DefaultDict[str, DefaultDict[str, List[int]]],
        y_data_to_plot: str,
        x_data_to_plot: str,
        *args: str,
        **kwargs: Any,
    ) -> Union[plt.Axes, np.ndarray, np.generic]:
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

        fig, axs = plt.subplots(len(y_data_to_plot), len(keys))

        if axs is None:
            axs = plt.gca()

        for centrality_measure, i in zip(y_data_to_plot, range(axs.shape[0])):
            for key, k in zip(keys, range(0, axs.shape[1])):

                # filter the nodes that have not been affected at all by the simulation
                filtered_data = dict_data[key]["df_total"][
                    dict_data[key]["df_total"][x_data_to_plot] != 0
                ]
                # remove inf and NaN entries
                filtered_data = filtered_data.replace([np.inf, -np.inf], np.nan).dropna(
                    axis=1
                )

                corr = get_correlation(
                    filtered_data[x_data_to_plot],
                    filtered_data[centrality_measure],
                )

                m, b = np.polyfit(
                    filtered_data[x_data_to_plot],
                    filtered_data[centrality_measure],
                    1,
                )

                axs[i, k].plot(
                    filtered_data[x_data_to_plot],
                    m * filtered_data[x_data_to_plot] + b,
                    color="k",
                    label=corr,
                )

                axs[i, k].scatter(
                    filtered_data[x_data_to_plot],
                    filtered_data[centrality_measure],
                    s=0.5,
                    color="grey",
                )

                axs[i, k].set_xlim(0, 1)
                axs[i, k].xaxis.set_major_formatter(mtick.PercentFormatter())
                axs[i, k].patch.set_edgecolor("black")
                axs[i, k].patch.set_linewidth("2")
                axs[i, k].legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.05),
                    ncol=3,
                    fancybox=True,
                    shadow=True,
                )

        for ax, col in zip(axs[0], keys):
            ax.set_title(col.capitalize(), weight="bold")

        for ax, col in zip(axs[-1], keys):
            ax.set_xlabel("Criminal likelihood".capitalize(), weight="bold")

        for ax, row in zip(axs[:, 0], y_data_to_plot):
            ax.set_ylabel(row.capitalize(), weight="bold")

        plt.tight_layout()

        if self.args.save:
            fig_name = (
                DirectoryFinder().result_dir_fig
                + "correlation_grid_fig"
                + "_"
                + timestamp()
                + ".png"
            )
            self.save_figure(fig_name, axs)
            return axs
        else:
            plt.show()
            return axs
