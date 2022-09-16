"""This script's intention is to get the properties of an given network.

__author__ = Louis Weyland
__date__   = 5/02/2022
"""
import logging
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import networkit as nk
import numpy as np
import powerlaw
from tqdm import tqdm
from utils.suppress_stdout_stderr import suppress_stdout_stderr

logger = logging.getLogger("logger")


class NetworkStats:
    """Take an network as input and returns its properties."""

    def __init__(self, network: nk.Graph) -> None:
        """Initialize the network as an attribute."""
        assert isinstance(network, nk.Graph), "Given network type is not Networkit"
        self.network = network

    def get_overview(self) -> Dict[str, Union[int, float]]:
        """Get an overview of the network."""
        if logging.getLevelName(logger.level) == "INFO":
            print("----------------------------")
            nk.overview(self.network)
            nk.community.detectCommunities(self.network)

        overview = {
            "component": int(self.get_connected_components()),
            "is_powerlaw": self.check_if_powerlaw()[0],
            "density": self.get_density(),
            "secrecy": self.get_secrecy(),
            "nodes": int(self.network.numberOfNodes()),
            "edges": int(self.network.numberOfEdges()),
            "avg_degree": sum(
                nk.centrality.DegreeCentrality(self.network).run().scores()
            )
            / self.network.numberOfNodes(),
            "clustering_coeff": self.get_clustering_coefficient(),
            "relative_density": self.get_relative_density(),
            "diameter": int(self.get_diameter()),
            # "radius": int(self.get_radius()),
            "dispersion": self.get_degree_dispersion(),
            "avg_path_length": self.get_average_path_length(),
        }
        # self.get_efficiency()
        if logging.getLevelName(logger.level) == "INFO":
            print("----------------------------")

        return overview

    def get_connected_components(self) -> int:
        """Return the number of connected components."""
        cc = nk.components.ConnectedComponents(self.network)
        cc.run()
        n_component = cc.numberOfComponents()
        logger.info(f"Number of components = {n_component}")
        return n_component

    def get_degree_distribution(self, normalized: bool = True) -> List[float]:
        """Get the normalized degree distribution of a network."""
        return (
            nk.centrality.DegreeCentrality(self.network, normalized=normalized)
            .run()
            .scores()
        )

    def check_if_powerlaw(self, data: List[float] = None) -> Tuple[bool, float]:
        """Check if a given data follows a powerlaw distribution.

        Data needs to be sorted first!
        """
        if data is None:
            data = self.get_degree_distribution()
        data = sorted(data, reverse=True)
        distributions = ["exponential", "lognormal"]
        # Suppress the powerlaw info
        with suppress_stdout_stderr():
            fit = powerlaw.Fit(data)

        for distribution in distributions:
            # if power_law value is negative than other distributions are preferred
            res = fit.distribution_compare("power_law", distribution)
            if res[0] < 0:
                logger.info(
                    f"{distribution.title()} is preferred over powerlaw, p_value = {res[1]:.4f}"
                )
                is_powerlaw = False
                break
            else:
                is_powerlaw = True

        if is_powerlaw is True:
            logger.info(
                f"Network degree distribution follows a powerlaw, p_value = {res[1]:.4f}"
            )
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
            return int(diameter)
        else:
            logger.warning("Graph must be connected! Otherwise distance == inf")
            return -1

    def get_average_path_length(self) -> float:
        """Return the average_path_length."""
        # Initialse algorithm
        cc = nk.components.ConnectedComponents(self.network)
        cc.run()
        gcs = cc.extractLargestConnectedComponent(self.network, True)
        n_nodes = gcs.numberOfNodes()
        if n_nodes <= 1:
            return 0
        apsp = nk.distance.APSP(gcs)
        apsp.run()

        distances = apsp.getDistances(asarray=True)
        upper_triang = np.triu(distances, k=1)
        avg_path_length = upper_triang[np.nonzero(upper_triang)].mean()
        logger.info(f"Average path length is {avg_path_length=}")
        return avg_path_length

    def get_radius(self) -> int:
        """Get the radius of a graph which is the minimum eccentricity."""
        # predefine the len of the list for speed
        eccentricity = []

        for node in tqdm(
            self.network.iterNodes(),
            total=self.network.numberOfNodes(),
            desc="Getting the radius...",
            leave=False,
        ):
            eccentricity.append(self.get_eccentricity(node))

        radius = np.min(eccentricity)
        logger.info(f"Radius = {radius}")
        return int(radius)

    def get_eccentricity(self, node: int) -> int:
        """Return the eccentricity of a node."""
        return nk.distance.Eccentricity.getValue(self.network, node)[1]

    def get_scale_freeness(self) -> None:
        """Scale freeness as defined in M. Graph Theory."""
        raise NotImplementedError

    def get_density(self) -> float:
        """Get the density of a network."""
        m = self.network.numberOfEdges()
        n = self.network.numberOfNodes()
        d = (2 * m) / (n * (n - 1))
        logger.info(f"Density = {d}")
        return d

    def get_secrecy(self) -> float:
        """Get the secrecy of a network."""
        m = self.network.numberOfEdges()
        n = self.network.numberOfNodes()
        s = (n * (n - 1)) / (2 * m)
        logger.info(f"Secrecy = {s}")
        return s

    def get_relative_density(self) -> float:
        """Get the relative density of a graph as defined in Scott J."""
        n = self.network.numberOfNodes()
        mean_degree = np.mean(self.get_degree_distribution(normalized=False))
        d = (n * mean_degree) / (n * (n - 1))
        logger.info(f"Relative Density = {d}")
        return d

    def get_degree_dispersion(self) -> float:
        """Get the dipsersion coefficient <k^2>/<k>.

        If the Molloy-Reed criterion is highter > 2, an giant component exists.
        """
        k = self.get_degree_distribution(normalized=False)
        k_2 = list(map(lambda x: pow(x, 2), k))
        dispersion = np.mean(k_2) / np.mean(k)

        if dispersion > 2:
            logger.info(
                f"Dispersion criterion is {dispersion:.3f}>2 -> A giant component is present!"
            )
        elif dispersion <= 2:
            logger.info(
                f"Dispersion criterion is {dispersion:3.f}<=2 -> A giant component isn't exist!"
            )
        return float(dispersion)

    def get_efficiency(self) -> float:
        """Get the efficiency of a Network.

        Corresponds to communication efficiency
        """
        # Using the All-Pairs Shortest-Paths algorithm
        apsp = nk.distance.APSP(self.network)
        apsp.run()
        # Vector of list for each node to each node
        vector_of_dist = apsp.getDistances()
        # Merge all the lists in one array
        arr_dist = np.hstack(vector_of_dist)

        # get ride of 0 elements since they represent distance to themself
        # and get the inv
        arr_dist = arr_dist[arr_dist != 0]
        arr_inv_dist = np.reciprocal(arr_dist)

        n = self.network.numberOfNodes()
        efficiency = (1 / (n * (n - 1))) * np.sum(arr_inv_dist)

        logger.info(f"Efficiency = {efficiency}")
        return efficiency

    def get_clustering_coefficient(self) -> float:
        """Return the global clustering coefficient."""
        lcc = (
            nk.centrality.LocalClusteringCoefficient(self.network, True).run().scores()
        )
        return sum(lcc) / self.network.numberOfNodes()
