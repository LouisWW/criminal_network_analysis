"""
This script's intention is generate random, scale-free or
other kinds of networks using netowkx

__author__ = Louis Weyland
__date__   = 6/02/2022
"""
import matplotlib.pyplot as plt
import networkit as nk


class NetworkGenerator:
    """
    Generates desired random,scale-free or other sorts of networks
    from scratch using networkx
    """

    def __init__(self):
        pass

    def generate_barabasi_albert(
        self, n_nodes: int = 100, n_edges: int = 2
    ) -> nk.generators.BarabasiAlbertGenerator:
        """
        Generates a Barabasi-Albert Graph
        A. L. Barabási and R. Albert “Emergence of scaling in random networks”,
        Science 286, pp 509-512, 1999.
        """
        return nk.generators.BarabasiAlbertGenerator(k=n_edges, nMax=n_nodes).generate()

    def generate_random(
        self, n_nodes: int = 100, prob: float = 0.3
    ) -> nk.generators.ErdosRenyiGenerator:
        """
        Generates a random graph based
        P. Erdős and A. Rényi, On Random Graphs, Publ. Math. 6, 290 (1959).
        """
        return nk.generators.ErdosRenyiGenerator(nNodes=n_nodes, prob=prob).generate()


if __name__ == "__main__":

    network_generator = NetworkGenerator()
    network_obj = network_generator.generate_barabasi_albert(n_nodes=60, n_edges=2)
    # network_obj = network_generator.generate_random(n_nodes=10)
    nk.viztasks.drawGraph(network_obj)
    plt.show()
