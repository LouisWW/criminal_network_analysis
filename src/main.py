"""
This script contains regroups all the functions and pipelines needed to recreate the results.

__author__ = Louis Weyland
__date__   = 22/02/2022
"""
import datetime
import logging

import matplotlib.pyplot as plt
import numpy as np
from config.config import ConfigParser
from network_utils.network_converter import NetworkConverter
from network_utils.network_reader import NetworkReader
from network_utils.network_stats import NetworkStats
from PIL import Image
from PIL import PngImagePlugin
from simulators.meta_simulator import MetaSimulator
from simulators.sim_mart_vaq import SimMartVaq
from utils.plotter import Plotter
from utils.sensitivity_analysis import SensitivityAnalyser

# Catch the flags
args = ConfigParser().args
plotter = Plotter()


# Define logger output
logger = logging.getLogger("logger")
logger_handler = logging.StreamHandler()  # Handler for the logger
logger_handler.setFormatter(
    logging.Formatter(
        "[%(levelname)s]\n\t %(message)-100s ---- (%(asctime)s.%(msecs)03d) %(filename)s",
        datefmt="%H:%M:%S",
    )
)
logger.addHandler(logger_handler)
logger.propagate = False

if args.verbose:
    logger.setLevel(logging.INFO)


if args.sim_mart_vaq:
    """Simulate the simulation form
    Martinez-Vaquero, L. A., Dolci, V., & Trianni, V. (2019).
    Evolutionary dynamics of organised crime and terrorist networks. Scientific reports, 9(1), 1-10.
    """
    # Get actual criminal network
    nx_network = NetworkReader().get_data(args.read_data)
    logger.info(f"The data used is {nx_network.name}")

    # Add nodes to network
    # First convert to gt
    meta_sim = MetaSimulator(
        network_name=nx_network.name,
        attachment_method=args.attach_meth,
        ratio_honest=0.9,
        ratio_wolf=0.01,
        random_fit_init=False,
    )

    simulators = SimMartVaq(
        network=meta_sim.network,
        delta=0.7,  # no acting for wolfs
        gamma=0.8,
        tau=0.1,  # no fitness sharing between wolf to criminal
        beta_s=5,
        beta_h=5,
        beta_c=5,
        c_c=1,  # no benefits from criminals/ they still act
        r_c=1,
        c_w=1,
        r_w=1,
        r_h=1,
        temperature=10,
        mutation_prob=0.0001,  # only fermi function
    )
    data_collector = simulators.avg_play(
        network=simulators.network,
        rounds=200,
        n_groups=1,
        ith_collect=50,
        repetition=5,
        measure_topology=True,
    )

    ax_0 = plotter.plot_lines(
        dict_data=data_collector,
        y_data_to_plot=["mean_ratio_honest", "mean_ratio_wolf", "mean_ratio_criminal"],
        x_data_to_plot="mean_iteration",
        xlabel="Rounds",
        ylabel="Ratio",
        plot_std=True,
    )

    e = datetime.datetime.now()
    timestamp = e.strftime("%d-%m-%Y-%H-%M")
    simulators_str_dict = dict(
        {str(key): str(value) for key, value in simulators.__dict__.items()}
    )
    meta_str_sim = dict(
        {str(key): str(value) for key, value in meta_sim.__dict__.items()}
    )
    meta = PngImagePlugin.PngInfo()
    for x in simulators_str_dict:
        meta.add_text(x, simulators_str_dict[x])

    if args.save:
        fig_name = plotter.savig_dir + "population_ration_" + timestamp + ".png"
        plt.savefig(fig_name, dpi=300)
        # Add the meta data to it
        im = Image.open(fig_name)
        im.save(fig_name, "png", pnginfo=meta)
    else:
        plt.show()

    ax_1 = plotter.plot_lines(
        dict_data=data_collector,
        y_data_to_plot=[
            "mean_fitness_honest",
            "mean_fitness_wolf",
            "mean_fitness_criminal",
        ],
        x_data_to_plot="mean_iteration",
        xlabel="Rounds",
        ylabel="Average fitness",
    )

    if args.save:
        fig_name = plotter.savig_dir + "fitness_evol_" + timestamp + ".png"
        plt.savefig(fig_name, dpi=300)
        # Add the meta to it
        im = Image.open(fig_name)
        im.save(fig_name, "png", pnginfo=meta)
    else:
        plt.show()

if args.sensitivity_analysis:
    """Runs a sensitivity analysis on the given choice."""
    if args.sensitivity_analysis == "sim-mart-vaq":

        sa = SensitivityAnalyser()
        sobol_indices = sa.sim_mart_vaq_sa(
            output_value=args.output_value,
            problem=None,
            n_samples=args.n_samples,
            rounds=args.rounds,
        )

if args.phase_diagram:
    parameter_dict = {
        "1": "beta_s",
        "2": "beta_h",
        "3": "beta_c",
        "4": "delta",
        "5": "gamma",
        "6": "tau",
        "7": "r_w",
        "8": "r_c",
    }

    print("Which parameters to test? Please give a number:")
    for k, v in parameter_dict.items():
        print(f"{k:<2}|{v:<15}")
    param_x = input("Parameter 1:\n")
    param_y = input("Parameter 2:\n")
    x_input = input("Range of parameter 1:  example: 0,5\n")
    x_input_tpl = tuple(int(x) for x in x_input.split(","))
    y_input = input("Range of parameter 2:  example: 0,5\n")
    y_input_tpl = tuple(int(x) for x in y_input.split(","))

    assert param_x != param_y, " Parameter can't be the same!"
    # create a mesh grid
    nx, ny = (10, 10)
    x_range = np.linspace(x_input_tpl[0], x_input_tpl[1], nx)
    y_range = np.linspace(y_input_tpl[0], y_input_tpl[1], ny)
    grid = np.empty((nx, ny), dtype=object)

    # init simulation
    # Add nodes to network
    # First convert to gt
    # Get actual criminal network
    nx_network = NetworkReader().get_data(args.read_data)
    logger.info(f"The data used is {nx_network.name}")

    meta_sim = MetaSimulator(
        network_name=nx_network.name,
        ratio_honest=0.33,
        ratio_wolf=0.33,
        random_fit_init=False,
    )
    for x_i in range(0, nx):
        for y_i in range(0, ny):
            variable_dict = dict(
                zip(
                    [parameter_dict[param_x], parameter_dict[param_y]],
                    [x_range[x_i], y_range[y_i]],
                )
            )
            simulators = SimMartVaq(
                network=meta_sim.network,
                **variable_dict,
                mutation_prob=0.0001,  # only fermi function
            )
            data_collector = simulators.avg_play(
                network=simulators.network,
                rounds=7500,
                n_groups=1,
                ith_collect=7500,
                repetition=5,
            )

            # Only look at the ratio and get the status with the highest ratio at the end
            filtered_dict = {
                k: v
                for k, v in data_collector.items()
                if k in ["mean_ratio_criminal", "mean_ratio_wolf", "mean_ratio_honest"]
            }

            grid[x_i, y_i] = max(filtered_dict, key=lambda x: filtered_dict[x][-1])

    ax = plotter.plot_phase_diag(
        grid, x_range, y_range, parameter_dict[param_x], parameter_dict[param_y]
    )

    if args.save:
        e = datetime.datetime.now()
        timestamp = e.strftime("%d-%m-%Y-%H-%M")
        simulators_str_dict = dict(
            {str(key): str(value) for key, value in simulators.__dict__.items()}
        )
        meta_str_sim = dict(
            {str(key): str(value) for key, value in meta_sim.__dict__.items()}
        )
        meta = PngImagePlugin.PngInfo()
        for x in simulators_str_dict:
            meta.add_text(x, simulators_str_dict[x])

        fig_name = (
            plotter.savig_dir
            + "phase_diag_"
            + param_x
            + "_"
            + param_y
            + "_"
            + timestamp
            + ".png"
        )
        plt.savefig(fig_name, dpi=300)
        # Add the meta to it
        im = Image.open(fig_name)
        im.save(fig_name, "png", pnginfo=meta)
    else:
        plt.show()


if args.compare_simulations:
    logger.info(f"The data used is {args.read_data}")

    meta_sim_pref = MetaSimulator(
        network_name=args.read_data,
        ratio_honest=0.3,
        ratio_wolf=0.3,
        n_new_edges=2,
        attachment_method="preferential",
    )
    meta_sim_rand = MetaSimulator(
        network_name=args.read_data,
        ratio_honest=0.3,
        ratio_wolf=0.3,
        prob=0.01,
        attachment_method="random",
    )
    meta_sim_sw = MetaSimulator(
        network_name=args.read_data,
        ratio_honest=0.3,
        ratio_wolf=0.3,
        prob=0.6,
        k=4,
        attachment_method="small-world",
    )

    simulators_pref = SimMartVaq(network=meta_sim_pref.network)
    simulators_rand = SimMartVaq(network=meta_sim_rand.network)
    simulators_sw = SimMartVaq(network=meta_sim_sw.network)

    # Get overview of the new network
    complete_network_stats_pref = NetworkStats(
        NetworkConverter.gt_to_nk(simulators_pref.network)
    )
    complete_network_stats_rand = NetworkStats(
        NetworkConverter.gt_to_nk(simulators_rand.network)
    )
    complete_network_stats_sw = NetworkStats(
        NetworkConverter.gt_to_nk(simulators_sw.network)
    )

    complete_network_stats_pref.get_overview()
    complete_network_stats_rand.get_overview()
    complete_network_stats_sw.get_overview()

    data_collector_pref = simulators_pref.avg_play(
        network=simulators_pref.network,
        rounds=args.rounds,
        n_groups=1,
        repetition=5,
        ith_collect=20,
        measure_topology=True,
    )
    data_collector_rand = simulators_rand.avg_play(
        network=simulators_rand.network,
        rounds=args.rounds,
        n_groups=1,
        repetition=5,
        ith_collect=20,
        measure_topology=True,
    )
    data_collector_sw = simulators_sw.avg_play(
        network=simulators_sw.network,
        rounds=args.rounds,
        n_groups=1,
        repetition=5,
        ith_collect=20,
        measure_topology=True,
    )

    ax = plotter.plot_lines_comparative(
        {
            "preferential attachment": data_collector_pref,
            "random attaschment": data_collector_rand,
            "small world": data_collector_sw,
        },
        y_data_to_plot=["mean_" + "security_efficiency"],
        x_data_to_plot="mean_iteration",
        title="Testing the simulation",
        xlabel="rounds",
        ylabel="security_efficiency",
        plot_std="True",
    )

    plt.show()
