"""This script's intention is generate random, scale-free or other kinds of networks using netowkx.

__author__ = Louis Weyland
__date__   = 6/02/2022
"""
import networkit as nk


class NetworkGenerator:
    """Generate desired random,scale-free or other sorts of networks."""

    def __init__(self) -> None:
        """Nothing to return for now."""
        pass

    def generate_barabasi_albert(
        self, n_nodes: int = 100, n_edges: int = 2
    ) -> nk.generators.BarabasiAlbertGenerator:
        """
        Generate a Barabasi-Albert graph.

        A. L. Barabási and R. Albert “Emergence of scaling in random networks”,
        Science 286, pp 509-512, 1999.
        """
        return nk.generators.BarabasiAlbertGenerator(k=n_edges, nMax=n_nodes).generate()

    def generate_random(
        self, n_nodes: int = 100, prob: float = 0.3
    ) -> nk.generators.ErdosRenyiGenerator:
        """
        Generate a random graph.

        P. Erdős and A. Rényi, On Random Graphs, Publ. Math. 6, 290 (1959).
        """
        return nk.generators.ErdosRenyiGenerator(nNodes=n_nodes, prob=prob).generate()
