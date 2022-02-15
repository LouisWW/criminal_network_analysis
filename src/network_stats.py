"""
This script's intention is to get the properties of an
given network

__author__ = Louis Weyland
__date__   = 5/02/2022
"""
import logging
from typing import List
from typing import Tuple

import networkit as nk
import networkx as nx
import powerlaw

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s ---- (%(asctime)s.%(msecs)03d) %(filename)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("logger")


class NetworkStats:
    """Takes an network as input and returns its properties"""

    def __init__(self, network: nx.Graph) -> None:

        self.network = network

    def get_overview(self) -> None:
        """Get an overview of the network"""
        nk.overview(self.network)
        nk.community.detectCommunities(self.network)

    def get_connected_components(self) -> int:
        """Return the number of connected components"""
        cc = nk.components.ConnectedComponents(self.network)
        cc.run()
        n_component = cc.numberOfComponents()
        logger.info(f"Number of components = {n_component}")
        return n_component

    def get_degree_distribution(self) -> List[float]:
        """Gets the normalized degree distribution of a network"""
        return (
            nk.centrality.DegreeCentrality(self.network, normalized=True).run().scores()
        )

    def check_if_powerlaw(self, data: List[float]) -> Tuple[bool, float]:
        """
        Checks if a given data follows a powerlaw distribution
        Data needs to be sorted first!
        """
        data = sorted(data, reverse=True)
        distributions = ["exponential", "lognormal"]
        fit = powerlaw.Fit(data)

        for distribution in distributions:
            # if power_law value is negative than other distributions are preferred
            res = fit.distribution_compare("power_law", distribution)
            if res[0] < 0:
                logger.info(
                    f"{distribution} is preferred over powerlaw, results = {res}"
                )
                is_powerlaw = False
                break
            else:
                is_powerlaw = True
        return is_powerlaw, fit.alpha

    def get_community(self) -> int:
        """Gets the number of communities"""
        communities = nk.community.detectCommunities(self.network)
        logger.warning("Additional work needed here!")
        return communities

    def get_diameter(self) -> int:
        """
        Gets the diameter, longest possible path of a
        network
        """
        if self.get_connected_components() == 1:
            diam = nk.distance.Diameter(self.network, algo=1)
            diam.run()
            diameter = diam.getDiameter()[0]
            logger.info(f"Diameter = {diameter}")
        else:
            logger.warning("Graph must be connected! Otherwise distance == inf")
            return -1

    def get_radius(self) :
        """Get the radius of a graph"""
        raise NotImplementedError
 
    def get_scale_freeness(self):
        """Scale freeness as defined in M. Graph Theory"""
        raise NotImplementedError

    def get_density(self):
        """Get the relative density of a graph as
        defined in Scott J."""
        raise NotImplementedError


if __name__ == "__main__":

    from network_generator import NetworkGenerator

    network_generator = NetworkGenerator()
    network = network_generator.generate_barabasi_albert(n_nodes=1000)

    network_stats = NetworkStats(network)

    """
    data = network_stats.get_degree_distribution
    _, alpha = network_stats.check_if_powerlaw(data)
    from utils.plotter import Plotter

    plotter = Plotter()
    plotter.plot_log_log(data, "Degree", "P(X)")
    """
    # network_stats.get_community()
    x = network_stats.get_diameter()
