"""This script's intention is to simulate the evolution of a criminal network.

Martinez-Vaquero, L. A., Dolci, V., & Trianni, V. (2019).
Evolutionary dynamics of organised crime and terrorist networks. Scientific reports, 9(1), 1-10.

__author__ = Louis Weyland
__date__   = 11/04/2022
"""
import logging

import graph_tool.all as gt
import numpy as np
from network_utils.network_combiner import NetworkCombiner
from tqdm import tqdm

logger = logging.getLogger("logger")


class SimMartVaq:
    """Contain the framework to simulate the process."""

    def __init__(
        self, network: gt.Graph, ratio_honest: float = 0.7, ratio_wolf: float = 0.1
    ) -> None:
        """Init the network charaterisics."""
        # Check if data is coherent
        assert isinstance(network, gt.Graph), "Network should be of type gt."
        assert 0 < ratio_honest < 1, "Ratio needs to be (0,1)"
        assert 0 < ratio_wolf < 1, "Ratio needs to be (0,1)"
        assert 0 < ratio_wolf + ratio_honest < 1, "Togehter the ratio should be (0,1)"
        assert network.vp.state, "Network has no attribute state"

        self.ratio_honest = ratio_honest
        self.ratio_wolf = ratio_wolf
        self.ratio_criminal = 1 - self.ratio_honest - self.ratio_wolf

        # Network needs to have a base criminal network
        self.n_criminal = len(gt.find_vertex(network, network.vp.state, "c"))

        self.total_number_nodes = int(self.n_criminal / self.ratio_criminal)
        self.new_nodes = self.total_number_nodes - self.n_criminal
        logger.info(
            f"Given the ratio param, {self.new_nodes} \
                    nodes are added, total = {self.total_number_nodes} nodes!"
        )

        self.new_network = NetworkCombiner.combine_by_preferential_attachment_faster(
            network, new_nodes=self.new_nodes, n_new_edges=2
        )

        # Get all the agents with no states
        # Init either honest or lone wolf
        self.relative_ratio_honest = round(
            self.ratio_honest / (self.ratio_honest + self.ratio_wolf), 2
        )
        self.relative_ratio_wolf = round(
            self.ratio_wolf / (self.ratio_honest + self.ratio_wolf), 2
        )
        nodes_no_states = gt.find_vertex(network, network.vp.state, "")
        for i in tqdm(
            nodes_no_states, desc="Adding attriutes to nodes", total=self.new_nodes
        ):
            self.new_network.vp.state[self.new_network.vertex(i)] = np.random.choice(
                ["h", "w"], 1, p=[self.relative_ratio_honest, self.relative_ratio_wolf]
            )
