"""This script's intention is to get the properties of all the individual nodes.

__author__ = Louis Weyland
__date__   = 13/02/2022
"""
from typing import Sequence
from typing import Tuple

import networkit as nk
import numpy as np
from network_utils.network_converter import NetworkConverter


class NodeStats:
    """Define the properties of a node.

    The properties are katz,betweeness,eccentricity
    """

    def __init__(self, network: nk.Graph) -> None:
        """Initialize the network as an attribute."""
        self.network = NetworkConverter.nx_to_nk(network)

    def get_katz(self) -> Sequence[Tuple[int, float]]:
        """Get the katz score for each node."""
        # Initialize algorithm
        katz = nk.centrality.KatzCentrality(self.network, 1e-3)
        katz.run()
        return katz.ranking()

    def get_eccentricity(self) -> Sequence[Tuple[int, int]]:
        """Return the eccentricity of the nodes."""
        eccentricity = np.zeros(self.network.numberOfNodes())
        # to append to the right idx in the list
        iterator = iter(range(0, self.network.numberOfNodes()))

        for node in self.network.iterNodes():
            eccentricity[next(iterator)] = nk.distance.Eccentricity.getValue(
                self.network, node
            )

        return eccentricity

    def get_betweenness(self) -> Sequence[Tuple[int, int]]:
        """Return betweeness centrality."""
        btwn = nk.centrality.Betweenness(self.network)
        btwn.run()
        return btwn.ranking()

    def get_closeness(self) -> Sequence[Tuple[int, float]]:
        """Return the closeness of a node."""
        raise NotImplementedError

    def get_degree(self) -> Sequence[Tuple[int, int]]:
        """Count the number of neighbor a node has."""
        raise NotImplementedError

    def get_local_clustering(self) -> Sequence[Tuple[int, float]]:
        """Get the local clustering of a node."""
        raise NotImplementedError

    def get_eigenvector_centrality(self) -> Sequence[Tuple[int, float]]:
        """Get the eigenvector centrality of a node."""
        raise NotImplementedError

    def get_average_path(self) -> None:
        """Get average path over a node."""
        raise NotImplementedError
