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
        """Filter the criminal network out of the whole network.

        Args:
            network (gt.Graph): network having nodes with attribute status.

        Returns:
            gt.Graph: the network with nodes.status = 'c' and links to each other.
        """
        filtering = network.new_vertex_property("bool")
        filtering.a = (network.status == "c").astype(int)
        network.vertex_properties["filtering"] = filtering
        network.set_vertex_filter(network.vp.filtering)
        return network

    @staticmethod
    def un_filter_criminal_network(network: gt.Graph) -> gt.Graph:
        """Reset the filtering and return the total graph.

        Args:
            network (gt.Graph): Any gt.network which is filtered or not.

        Returns:
            gt.Graph: returns an unfiltered network.
        """
        network.set_vertex_filter(None)
        return network
