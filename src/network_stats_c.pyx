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


cdef class NetworkStats:
    """Take an network as input and returns its properties."""

    def __init__(self) -> None:
        """Initialize the network as an attribute."""

    def get_overview(self,network) -> None:
        """Get an overview of the network."""
        nk.overview(network)
        nk.community.detectCommunities(network)

    def get_connected_components(self,network) -> int:
        """Return the number of connected components."""
        cc = nk.components.ConnectedComponents(network)
        cc.run()
        n_component = cc.numberOfComponents()
        logger.info(f"Number of components = {n_component}")
        return n_component

    def get_degree_distribution(self,network) -> List[float]:
        """Get the normalized degree distribution of a network."""
        return (
            nk.centrality.DegreeCentrality(network, normalized=True).run().scores()
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

    def get_community(self,network) -> int:
        """Get the number of communities."""
        communities = nk.community.detectCommunities(network)
        logger.warning("Additional work needed here!")
        return communities

    def get_diameter(self,network) -> int:
        """Get the diameter, longest possible path of a network."""
        if self.get_connected_components() == 1:
            diam = nk.distance.Diameter(network, algo=1)
            diam.run()
            diameter = diam.getDiameter()[0]
            logger.info(f"Diameter = {diameter}")
            return diameter
        else:
            logger.warning("Graph must be connected! Otherwise distance == inf")
            return -1

    def get_radius(self,network) -> int:
        """Get the radius of a graph which is the minimum eccentricity."""
        # predefine the len of the list for speed
        eccentricity = np.zeros(network.numberOfNodes())
        # to append to the right idx in the list
        iterator = iter(range(0, network.numberOfNodes()))

        for node in network.iterNodes():
            eccentricity[next(iterator)] = self.get_eccentricity(network,node)

        radius = min(eccentricity)
        logger.info(f"Radius = {radius}")
        return radius

    def get_eccentricity(self,network, node: int) -> int:
        """Return the eccentricity of a node."""
        return nk.distance.Eccentricity.getValue(network, node)[1]

    def get_scale_freeness(self,network) -> None:
        """Scale freeness as defined in M. Graph Theory."""
        raise NotImplementedError

    def get_density(self,network) -> float:
        """Get the density of a network."""
        m = network.numberOfEdges()
        n = network.numberOfNodes()
        d = (2 * m) / n * (n - 1)
        logger.info(f"Density = {d}")
        return d

    def get_relative_density(self,network) -> float:
        """Get the relative density of a graph as defined in Scott J."""
        m = network.numberOfEdges()
        n = network.numberOfNodes()
        mean_degree = (2 * m) / n
        d = (n * mean_degree) / (n * (n - 1))
        logger.info(f"Relative Density = {d}")
        return d


    def fib(self,n: int = 50) -> int:
        if n <= 1:
            return n
        else:
            return self.fib(n - 2) + self.fib(n - 1)

    cpdef int fib_c(self, int n = 50):
        if n <= 1:
            return n
        else:
            return self.fib(n - 2) + self.fib(n - 1)