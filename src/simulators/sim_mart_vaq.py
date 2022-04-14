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
        self.relative_ratio_honest = self.ratio_honest / (
            self.ratio_honest + self.ratio_wolf
        )
        self.relative_ratio_wolf = 1 - self.relative_ratio_honest

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

    def play(self, network: gt.Graph, rounds: int = 1, n_new_edges: int = 2) -> None:
        """Run the simulation.

        Network is subdivided in to n groups.
        In each group, a person is selected.
        If selected person is a wolf or criminal,
        damage is inflicted on others.
        """
        # Init a population
        network = self.initialise_network(network, n_new_edges)
        # Init fitness attribute
        network = self.init_fitness(network)

        # Run the simulation
        for i in tqdm(range(0, rounds), desc="Playing the rounds...", total=rounds):
            # Divide the network in random new groups
            mbr_list, group_numbers = self.divide_in_groups(network)
            logger.debug(f"The Network is divided in {len(group_numbers)} groups")

            # Go through each group
            for number in group_numbers:
                self.acting_stage(network, mbr_list, number)

    def acting_stage(
        self, network: gt.Graph, mbr_list: List[int], group_number: int
    ) -> None:
        """Correspond to the acting stage in the paper.

        Given an group, select on person and proceed to the acting.
        """
        # Get all the people from the same group
        group_member = gt.find_vertex(network, mbr_list, group_number)
        # Select one person
        slct_pers = np.random.choice(group_member, 1)
        # check the person status
        slct_pers_st = network.vp.state[network.vertex(slct_pers)]

        return network

    def divide_in_groups(self, network: gt.Graph) -> Tuple[List[int], FrozenSet[int]]:
        """Divide the network in groups.

        Making use of the  minimize_blockmodel_dl func.
        For now, the number of groups can't be imposed.
        Returns a list with the group number/label.
        """
        partitions = gt.minimize_blockmodel_dl(network)
        mbr_list = partitions.get_blocks()
        group_numbers = frozenset(mbr_list)

        return mbr_list, group_numbers

    def init_fitness(self, network: gt.Graph) -> gt.Graph:
        """Add the attribute fitness to the network."""
        if "fitness" in network.vp:
            return network
        else:
            fitness = network.new_vertex_property("double")
            network.vertex_properties["fitness"] = fitness

        return network
