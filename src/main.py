"""
This script contains regroups all the functions and pipelines needed to recreate the results.

__author__ = Louis Weyland
__date__   = 22/02/2022
"""
import logging

from config.config import ConfigParser
from network_utils.network_converter import NetworkConverter
from network_utils.network_reader import NetworkReader
from network_utils.network_stats import NetworkStats
from simulators.sim_mart_vaq import SimMartVaq
from utils.plotter import Plotter
from utils.sensitivity_analysis import SensitivityAnalyser

# Catch the flags
args = ConfigParser().args


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
    # Get stats about network_obj
    nk_network = NetworkConverter.nx_to_nk(nx_network)
    network_stats = NetworkStats(nk_network)
    network_stats.get_overview()

    # Add nodes to network
    # First convert to gt
    gt_network = NetworkConverter.nx_to_gt(nx_network)
    simulator = SimMartVaq(gt_network, ratio_honest=0.4, ratio_wolf=0.05)
    new_gt_network = simulator.initialise_network(gt_network)
    combined_nk_network = NetworkConverter.gt_to_nk(new_gt_network)

    # Get stats about network_obj
    network_stats = NetworkStats(combined_nk_network)
    network_stats.get_overview()

    Plotter().draw_network(new_gt_network, color_vertex_property="state")


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
