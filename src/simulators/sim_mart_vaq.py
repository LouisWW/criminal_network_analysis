"""This script's intention is to simulate the evolution of a criminal network.

Martinez-Vaquero, L. A., Dolci, V., & Trianni, V. (2019).
Evolutionary dynamics of organised crime and terrorist networks. Scientific reports, 9(1), 1-10.

__author__ = Louis Weyland
__date__   = 11/04/2022
"""
import logging
from typing import FrozenSet
from typing import List
from typing import Tuple

import graph_tool.all as gt
import numpy as np
from network_utils.network_combiner import NetworkCombiner
from tqdm import tqdm

logger = logging.getLogger("logger")


class SimMartVaq:
    """Contain the framework to simulate the process."""

    def __init__(
        self,
        network: gt.Graph,
        ratio_honest: float = 0.7,
        ratio_wolf: float = 0.1,
        n_new_edges: int = 2,
    ) -> None:
        """Init the network charaterisics."""
        # Define name of simulator
        self._name = "sim_mart_vaq"
        self.network = network

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
        assert self.n_criminal > 1, "The given network contains no criminals..."

        self.total_number_nodes = int(self.n_criminal / self.ratio_criminal)
        self.new_nodes = self.total_number_nodes - self.n_criminal

        # Init either honest or lone wolf
        self.relative_ratio_honest = round(
            self.ratio_honest / (self.ratio_honest + self.ratio_wolf), 2
        )
        self.relative_ratio_wolf = round(
            self.ratio_wolf / (self.ratio_honest + self.ratio_wolf), 2
        )

    @property
    def name(self) -> str:
        """Return the name of the simulator."""
        return self._name

    def initialise_network(self, network: gt.Graph, n_new_edges: int = 2) -> gt.Graph:
        """Add to the existing criminal network honests and lone wolfs.

        Thereby, the nodes are added based on the preferential attachment principle.
        Returns a network with new added nodes respecting the ratio of criminals/honest/wolfs.
        """
        logger.info(
            f"Given the ratio param, {self.new_nodes} \
                    nodes are added, total = {self.total_number_nodes} nodes!"
        )
        new_network = NetworkCombiner.combine_by_preferential_attachment_faster(
            network, new_nodes=self.new_nodes, n_new_edges=n_new_edges
        )

        # Get all the agents with no states
        nodes_no_states = gt.find_vertex(new_network, new_network.vp.state, "")
        for i in tqdm(
            nodes_no_states, desc="Adding attributes to nodes", total=self.new_nodes
        ):
            new_network.vp.state[new_network.vertex(i)] = np.random.choice(
                ["h", "w"], 1, p=[self.relative_ratio_honest, self.relative_ratio_wolf]
            )[0]

        return new_network

    def acting_stage(self, network: gt.Graph) -> None:
        """Correspond to the acting stage in the paper.

        Network is subdivided in to n groups.
        In each group, a person is selected.
        If selected person is a wolf or criminal,
        damage is inflicted on others.
        """
        pass

    def divide_in_groups(self, network: gt.Graph) -> Tuple[List[int], FrozenSet[int]]:
        """Divide the network in groups.

        Making use of the  minimize_blockmodel_dl func.
        For now, the number of groups can't be imposed.
        Returns a list with the group number/label.
        """
        partitions = gt.minimize_blockmodel_dl(network)
        groups = partitions.get_blocks()
        groups_label = frozenset(groups)

        return groups, groups_label
