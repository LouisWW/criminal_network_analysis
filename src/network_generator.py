"""
This script's intention is generate random, scale-free or
other kinds of networks using netowkx

__author__ = Louis Weyland
__date__   = 6/02/2022
"""
import matplotlib.pyplot as plt
import networkx as nx


class NetworkGenerator:
    """
    Generates desired random,scale-free or other sorts of networks
    from scratch using networkx
    """

    def __init__(self):
        pass

    def generate_barabasi_albert(
        self, n_nodes: int = 100, n_edges: int = 2
    ) -> nx.barabasi_albert_graph:
        """
        Generates a Barabasi-Albert Graph
        A. L. Barabási and R. Albert “Emergence of scaling in random networks”,
        Science 286, pp 509-512, 1999.
        """
        return nx.barabasi_albert_graph(n=n_nodes, m=n_edges)

    def generate_random(
        self, n_nodes: int = 100, prob: float = 0.3
    ) -> nx.erdos_renyi_graph:
        """
        Generates a random graph based
        P. Erdős and A. Rényi, On Random Graphs, Publ. Math. 6, 290 (1959).
        """
        return nx.erdos_renyi_graph(n=n_nodes, p=prob)


if __name__ == "__main__":

    network_generator = NetworkGenerator()
    network_obj = network_generator.generate_barabasi_albert(n_nodes=60, n_edges=2)
    # network_obj = network_generator.generate_random(n_nodes=10)
    nx.draw(network_obj)
    plt.show()
