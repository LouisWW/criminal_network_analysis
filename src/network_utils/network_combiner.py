"""This script's intention is to combined a given network with a synthetical network.

__author__ = Louis Weyland
__date__   = 6/02/2022
"""
import graph_tool.all as gt
import numpy as np
from tqdm import tqdm


class NetworkCombiner:
    """Combines two network together.

    In other words, it creates/attach nodes to an existing network.
    """

    def __init__(self) -> None:
        """Init parameters."""
        pass

    @staticmethod
    def combine_by_preferential_attachment_faster(
        network: gt.Graph, new_nodes: int, n_new_edges: int
    ) -> gt.Graph:
        """Apply preferential attachment to a existing network."""
        # Get the number of nodes of the existing network
        print("Hello")
        for _ in tqdm(
            range(0, new_nodes),
            desc="Adding nodes to existing network using preferential attachment...",
            leave=False,
            disable=True,
        ):
            # Get the attachment prob distribution before adding the node
            nodes_probs = []
            n_edges = network.num_edges()
            # get the degree of the nodes
            node_degr = network.get_out_degrees(network.get_vertices())
            nodes_probs = node_degr / (2 * n_edges)

            new_edges = np.random.choice(
                network.get_vertices(), size=n_new_edges, p=nodes_probs
            )

            network.add_vertex(1)
            # Add the edges to the last added node network.vertex(network.num_vertices())-1
            for new_e in new_edges:
                network.add_edge(network.vertex(network.num_vertices() - 1), new_e)
        return network
