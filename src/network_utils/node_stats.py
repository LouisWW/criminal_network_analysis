"""This script's intention is to get the properties of all the individual nodes.

__author__ = Louis Weyland
__date__   = 13/02/2022
"""
from typing import Sequence
from typing import Tuple

import graph_tool.all as gt
import networkit as nk
import numpy as np
from graph_tool import EdgePropertyMap
from graph_tool import VertexPropertyMap


class NodeStats:
    """Define the properties of a node.

    The properties are katz,betweeness,eccentricity
    """

    def __init__(self) -> None:
        """Initialize the network as an attribute."""
        pass

    @staticmethod
    def get_katz(network: gt.Graph) -> VertexPropertyMap:
        """Get the katz score for each node."""
        # Initialize algorithm
        katz = gt.katz(network)
        return katz

    @staticmethod
    def get_betweenness(network: gt.Graph) -> Tuple[VertexPropertyMap, EdgePropertyMap]:
        """Return betweeness centrality."""
        btwn = gt.betweenness(network)
        return btwn

    @staticmethod
    def get_closeness(network: gt.Graph) -> VertexPropertyMap:
        """Return the closeness of a node."""
        closness = gt.closeness(network)
        return closness

    @staticmethod
    def get_eigenvector_centrality(
        network: gt.Graph,
    ) -> Tuple[float, VertexPropertyMap]:
        """Get the eigenvector centrality of a node."""
        eigen_v = gt.eigenvector(network)
        return eigen_v

    @staticmethod
    def get_central_dominance(
        network: gt.GraphPropertyMap, betweenness: VertexPropertyMap
    ) -> float:
        """Get the central point dominace."""
        vertex, _ = betweenness
        central_dominace = gt.central_point_dominance(network, vertex)
        return central_dominace

    def get_eccentricity(network: nk.Graph) -> Sequence[Tuple[int, int]]:
        """Return the eccentricity of the nodes."""
        eccentricity = np.zeros(network.numberOfNodes())
        # to append to the right idx in the list
        iterator = iter(range(0, network.numberOfNodes()))

        for node in network.iterNodes():
            eccentricity[next(iterator)] = nk.distance.Eccentricity.getValue(
                network, node
            )
        return eccentricity

    def get_degree(self) -> Sequence[Tuple[int, int]]:
        """Count the number of neighbor a node has."""
        raise NotImplementedError

    def get_local_clustering(self) -> Sequence[Tuple[int, float]]:
        """Get the local clustering of a node."""
        raise NotImplementedError

    def get_average_path(self) -> None:
        """Get average path over a node."""
        raise NotImplementedError
