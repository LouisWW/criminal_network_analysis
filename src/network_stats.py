"""This script's intention is to get the properties of an given network.

__author__ = Louis Weyland
__date__   = 5/02/2022
"""
import logging
from typing import List
from typing import Tuple

import networkit as nk
import networkx as nx
import numpy as np
import powerlaw

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s ---- (%(asctime)s.%(msecs)03d) %(filename)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("logger")


class NetworkStats:
    """Take an network as input and returns its properties."""

    def __init__(self, network: nx.Graph) -> None:
        """Initialize the network as an attribute."""
        self.network = network

    def get_overview(self) -> None:
        """Get an overview of the network."""
        nk.overview(self.network)
        nk.community.detectCommunities(self.network)

    def get_connected_components(self) -> int:
        """Return the number of connected components."""
        cc = nk.components.ConnectedComponents(self.network)
        cc.run()
        n_component = cc.numberOfComponents()
        logger.info(f"Number of components = {n_component}")
        return n_component

    def get_degree_distribution(self) -> List[float]:
        """Get the normalized degree distribution of a network."""
        return (
            nk.centrality.DegreeCentrality(self.network, normalized=True).run().scores()
        )

    def check_if_powerlaw(self, data: List[float]) -> Tuple[bool, float]:
        """Check if a given data follows a powerlaw distribution.

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
        """Get the number of communities."""
        communities = nk.community.detectCommunities(self.network)
        logger.warning("Additional work needed here!")
        return communities

    def get_diameter(self) -> int:
        """Get the diameter, longest possible path of a network."""
        if self.get_connected_components() == 1:
            diam = nk.distance.Diameter(self.network, algo=1)
            diam.run()
            diameter = diam.getDiameter()[0]
            logger.info(f"Diameter = {diameter}")
            return diameter
        else:
            logger.warning("Graph must be connected! Otherwise distance == inf")
            return -1

    def get_radius(self) -> int:
        """Get the radius of a graph which is the minimum eccentricity."""
        # predefine the len of the list for speed
        eccentricity = np.zeros(self.network.numberOfNodes())
        # to append to the right idx in the list
        iterator = iter(range(0, self.network.numberOfNodes()))

        for node in self.network.iterNodes():
            eccentricity[next(iterator)] = self.get_eccentricity(node)

        radius = min(eccentricity)
        logger.info(f"Radius = {radius}")
        return radius

    def get_eccentricity(self, node) -> int:
        """Return the eccentricity of a node."""
        return nk.distance.Eccentricity.getValue(self.network, node)[1]

    def get_scale_freeness(self):
        """Scale freeness as defined in M. Graph Theory."""
        raise NotImplementedError

    def get_density(self) -> float:
        """Get the density of a network."""
        m = self.network.numberOfEdges()
        n = self.network.numberOfNodes()
        d = (2 * m) / n * (n - 1)
        return d

    def get_relative_density(self) -> float:
        """Get the relative density of a graph as defined in Scott J."""
        m = self.network.numberOfEdges()
        n = self.network.numberOfNodes()
        mean_degree = (2 * m) / n
        d = (n * mean_degree) / (n * (n - 1))
        return d


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
    # c =network_stats.get_community()
    # x = network_stats.get_diameter()
    # r = network_stats.get_radius()
    d = network_stats.get_relative_density()
