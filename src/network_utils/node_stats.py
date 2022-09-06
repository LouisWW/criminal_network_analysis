"""This script's intention is to get the topology and properties of all the individual nodes.

__author__ = Louis Weyland
__date__   = 13/02/2022
"""
import itertools
import logging
from typing import Sequence
from typing import Tuple

import graph_tool.all as gt
import networkit as nk
import numpy as np
from graph_tool import EdgePropertyMap
from graph_tool import VertexPropertyMap
from network_utils.network_converter import NetworkConverter

logger = logging.getLogger("logger")


class NodeStats:
    """Define the properties of a node.

    The properties are katz,betweeness,eccentricity.
    Additionally, compute some topology measures.
    """

    def __init__(self) -> None:
        """Initialize the network as an attribute."""
        pass

    @staticmethod
    def get_katz(network: gt.Graph) -> VertexPropertyMap:
        """Get the katz score for each node."""
        # Initialize algorithm
        katz = gt.katz(network)
        network.vertex_properties["katz"] = katz
        return network, katz

    @staticmethod
    def get_betweenness(network: gt.Graph) -> Tuple[VertexPropertyMap, EdgePropertyMap]:
        """Return betweeness centrality."""
        btwn, _ = gt.betweenness(network)
        network.vertex_properties["betweenness"] = btwn
        return network, btwn

    @staticmethod
    def get_closeness(network: gt.Graph) -> VertexPropertyMap:
        """Return the closeness of a node."""
        closeness = gt.closeness(network)
        network.vertex_properties["closeness"] = closeness
        return network, closeness

    @staticmethod
    def get_eigenvector_centrality(
        network: gt.Graph,
    ) -> Tuple[float, VertexPropertyMap]:
        """Get the eigenvector centrality of a node."""
        (
            _,
            eigen_v,
        ) = gt.eigenvector(network)
        network.vertex_properties["eigen_v"] = eigen_v
        return network, eigen_v

    @staticmethod
    def get_central_dominance(
        network: gt.GraphPropertyMap, betweenness: VertexPropertyMap
    ) -> float:
        """Get the central point dominace."""
        vertex = betweenness
        central_dominance = gt.central_point_dominance(network, vertex)
        return central_dominance

    # @staticmethod
    # def get_eccentricity(network: nk.Graph) -> Sequence[Tuple[int, int]]:
    #    """Return the eccentricity of the nodes."""
    #    eccentricity = np.empty(network.numberOfNodes())
    #    # to append to the right idx in the list
    #    iterator = iter(range(0, network.numberOfNodes()))
    #    for node in network.iterNodes():
    #         eccentricity[next(iterator)] = nk.distance.Eccentricity.getValue(
    #          network, node
    #                   )
    #     return eccentricity

    @staticmethod
    def get_density(network: gt.Graph) -> float:
        """Return the density.

        Metric has been defined in https://www.nature.com/articles/srep04238
        """
        n_edges = network.num_edges()
        n_nodes = network.num_vertices()
        if (n_nodes * (n_nodes - 1)) == 0:
            return 0
        return (2 * n_edges) / (n_nodes * (n_nodes - 1))

    @staticmethod
    def get_secrecy(network: gt.Graph) -> float:
        """Return the secrecy.

        Metric has been defined in https://www.nature.com/articles/srep04238
        """
        n_edges = network.num_edges()
        n_nodes = network.num_vertices()
        if (2 * n_edges) == 0:
            return -1
        return (n_nodes * (n_nodes - 1)) / (2 * n_edges)

    @staticmethod
    def get_flow_of_information(network: gt.Graph) -> float:
        """Return the flow of information.

        Metric has been defined in
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6214327/
        """
        n_nodes = network.num_vertices()
        if n_nodes <= 1:
            return 0
        shortest_dist_list = []
        source_target = list(itertools.combinations(network.get_vertices(), 2))
        for s, t in source_target:
            shortest_dist_list.append(
                gt.shortest_distance(network, source=s, target=t, directed=False)
            )
        sum_inv_shortest_dist = sum(1 / dist for dist in shortest_dist_list)
        return (1 / (n_nodes * (n_nodes - 1))) * sum_inv_shortest_dist

    def get_flow_of_information_faster(gt_network: gt.Graph) -> float:
        """Return the flow of information.

        Metric has been defined in
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6214327/
        """
        network = NetworkConverter.gt_to_nk(gt_network)
        n_nodes = network.numberOfNodes()
        if n_nodes <= 1:
            return 0
        apsp = nk.distance.APSP(network)
        apsp.run()

        distances = apsp.getDistances(asarray=True)
        inv_distances = np.reciprocal(distances)
        sum_inv_distances = np.triu(inv_distances, k=1).sum()
        return (1 / (n_nodes * (n_nodes - 1))) * sum_inv_distances

    @staticmethod
    def get_size_of_largest_component(network: gt.Graph) -> Tuple[int, int]:
        """Return the size of the largest component."""
        try:
            largest_component = gt.extract_largest_component(network)
        except:
            logger.info("Something didn't work with extracting of GCS")
            return 0, 0

        return largest_component.num_vertices(), largest_component.num_edges()

    def get_degree(self) -> Sequence[Tuple[int, int]]:
        """Count the number of neighbor a node has."""
        raise NotImplementedError

    def get_local_clustering(self) -> Sequence[Tuple[int, float]]:
        """Get the local clustering of a node."""
        raise NotImplementedError

    def get_average_path(self) -> None:
        """Get average path over a node."""
        raise NotImplementedError
