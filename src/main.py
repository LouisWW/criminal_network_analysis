"""
This script contains regroups all the functions and pipelines needed to recreate the results.

__author__ = Louis Weyland
__date__   = 22/02/2022
"""
import datetime
import logging

import matplotlib.pyplot as plt
from config.config import ConfigParser
from network_utils.network_reader import NetworkReader
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
        ratio_honest=0.9,
        ratio_wolf=0.01,
        random_fit_init=True,
    )

    simulators = SimMartVaq(
        network=meta_sim.network,
        delta=0.7,  # no acting for wolfs
        gamma=0.8,
        tau=0.2,  # no fitness sharing between wolf to criminal
        beta_s=10,
        beta_h=10,
        beta_c=10,
        c_c=4,  # no benefits from criminals/ they still act
        r_c=1,
        c_w=5,
        r_w=3,
        mutation_prob=0.0001,  # only fermi function
    )
    network, data_collector = simulators.play(
        network=simulators.network, rounds=50000, n_groups=5, ith_collect=5000
    )

    ax_0 = plotter.plot_lines(
        dict_data=data_collector,
        y_data_to_plot=["ratio_honest", "ratio_wolf", "ratio_criminal"],
        x_data_to_plot="iteration",
        title="Testing the simulation",
        xlabel="rounds",
        ylabel="ratio",
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

    ax_1 = plotter.plot_lines(
        dict_data=data_collector,
        y_data_to_plot=["fitness_honest", "fitness_wolf", "fitness_criminal"],
        x_data_to_plot="iteration",
        title="Testing the simulation",
        xlabel="rounds",
        ylabel="Average fitness",
    )

    if args.save:
        fig_name = plotter.savig_dir + "fitness_evol_" + timestamp + ".png"
        plt.savefig(fig_name, dpi=300)
        # Add the meta to it
        im = Image.open(fig_name)
        im.save(fig_name, "png", pnginfo=meta)

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
