"""
This script's intention is to get the properities of an
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
    format="\n%(levelname)s: %(message)s ---- (%(asctime)s.%(msecs)03d) %(filename)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("logger")


class NetowrkStats:
    """Takes an network as input and returns its properties"""

    def __init__(self, network: nx.Graph) -> None:

        self.network = network

    def get_overview(self) -> None:
        """Get an overview of the network"""
        nk.overview(self.network)

    @property
    def get_connected_components(self) -> int:
        """Return the number of connected components"""
        cc = nk.components.ConnectedComponents(self.network)
        cc.run()
        return cc.numberOfComponents()

    @property
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


if __name__ == "__main__":

    from network_generator import NetworkGenerator

    network_generator = NetworkGenerator()
    network = network_generator.generate_barabasi_albert(n_nodes=10000)

    network_stats = NetowrkStats(network)
    data = network_stats.get_degree_distribution
    _, alpha = network_stats.check_if_powerlaw(data)

    from utils.plotter import Plotter

    plotter = Plotter()
    plotter.plot_log_log(data, "Degree", "P(X)")