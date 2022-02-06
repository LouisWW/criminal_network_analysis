"""
This script's intention is to get the properities of an
given network

__author__ = Louis Weyland
__date__   = 5/02/2022
"""
import networkit as nk
import networkx as nx


class NetowrkStats:
    """Takes an network as input and returns its properties"""

    def __init__(self, network: nx.Graph) -> None:

        # convert grpah to networkit for faster computation
        self.network = nk.nxadapter.nx2nk(network)

    def get_overview(self) -> None:
        """Get an overview of the network"""
        nk.overview(self.network)


if __name__ == "__main__":

    from network_generator import NetworkGenerator

    network_generator = NetworkGenerator()
    network = network_generator.generate_random(n_nodes=200)

    network_stats = NetowrkStats(network)
    network_stats.get_overview()
