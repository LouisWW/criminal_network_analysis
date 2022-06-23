"""This script's intention is to extract a network based on vertices' attributes.

__author__ = Louis Weyland
__date__   = 23/06/2022
"""
import graph_tool.all as gt


class NetworkExtractor:
    """Filter out a given sub-network form the total network."""

    def __init__(self) -> None:
        """For the moment nothing."""
        pass

    @staticmethod
    def filter_criminal_network(network: gt.Graph) -> gt.Graph:
        """Filter the criminal network out of the whole network."""
        for i in range(0, network.num_vertices()):
            if network.vp.state[network.vertex(i)] == "c":
                network.vp.filtering[network.vertex(i)] = 1
            else:
                network.vp.filtering[network.vertex(i)] = 0

        network.set_vertex_filter(network.vp.filtering)
        return network

    @staticmethod
    def un_filter_criminal_network(network: gt.Graph) -> gt.Graph:
        """Reset the filtering and return the total graph."""
        network.set_vertex_filter(None)
        return network
