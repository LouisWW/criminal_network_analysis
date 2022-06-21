"""This script's intention is to combined a given network with a synthetical network.

__author__ = Louis Weyland
__date__   = 6/02/2022
"""
import graph_tool.all as gt
import numpy as np
import random
from tqdm import tqdm
import itertools


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
    
        
    @staticmethod
    def combine_by_random_attachment_faster(network: gt.Graph, new_nodes: int,prob:float) -> gt.Graph:
        """Generate a Erdös-Rény Random Network around the given network.
        
        The code is based on the pseudo-code described in 
        https://www.frontiersin.org/articles/10.3389/fncom.2011.00011/full
        """
        # Add new nodes
        network.add_vertex(n=new_nodes)
        n_number_of_nodes=network.num_vertices()
        random_number_list = np.random.rand(((n_number_of_nodes-new_nodes)*new_nodes)+
                                            (int((new_nodes*(new_nodes+1))/2)))
        accepted_edges_idx = len(np.argwhere(random_number_list<prob))
        possible_edge_list = list(itertools.product(range(0,n_number_of_nodes-new_nodes), range(n_number_of_nodes-new_nodes,n_number_of_nodes)))+\
        list(itertools.combinations(range(n_number_of_nodes-new_nodes,n_number_of_nodes),2))

        network.add_edge_list(random.sample(possible_edge_list, k=accepted_edges_idx))
        return network
                

        
        
        