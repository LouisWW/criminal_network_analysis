"""
This script contains regroups all the functions and pipelines needed to recreate the results.

__author__ = Louis Weyland
__date__   = 22/02/2022
"""
import logging

from config.config import ConfigParser
from network_utils.network_combiner import NetworkCombiner
from network_utils.network_converter import NetworkConverter
from network_utils.network_reader import NetworkReader
from network_utils.network_stats import NetworkStats
from utils.plotter import Plotter

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
logger.setLevel(logging.INFO)


if args.draw_network:
    """
    Visualize the network.

    Default is circular representation
    """
    network = NetworkReader().read_montagna_phone_calls()
    # convert graph_obj
    network = NetworkConverter.nx_to_gt(network)
    Plotter().draw_network(network)


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
    combined_gt_network = NetworkCombiner.combine_by_preferential_attachment_faster(
        gt_network, new_nodes=10000, n_new_edges=10
    )
    combined_nk_network = NetworkConverter.gt_to_nk(combined_gt_network)

    # Get stats about network_obj
    network_stats = NetworkStats(combined_nk_network)
    network_stats.get_overview()
