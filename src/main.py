"""
This script contains regroups all the functions and pipelines needed to recreate the results.

__author__ = Louis Weyland
__date__   = 22/02/2022
"""
from config.config import ConfigParser
from network_reader import NetworkReader
from utils.graph_converter import NetworkConverter
from utils.plotter import Plotter

# catch the flags
args = ConfigParser().args


if args.draw_network:
    """
    Visualize the network.

    Default is circular representation
    """
    network_reader = NetworkReader()
    network_obj = network_reader.read_montagna_phone_calls()
    # convert graph_obj
    network_obj = NetworkConverter().nx2gt(network_obj)
    Plotter().draw_network(network_obj)
