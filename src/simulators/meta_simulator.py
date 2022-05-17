"""This script contains the MetaSimulator.

The MetaSimulator encapsules all the simulators.
It prepapres the network for the simulators. From this script,
the different models can be run.

__author__ = Louis Weyland
__date__ = 17/05/2022
"""
import logging
from typing import Tuple

import graph_tool.all as gt
import numpy as np
from network_utils.network_combiner import NetworkCombiner
from network_utils.network_converter import NetworkConverter
from network_utils.network_reader import NetworkReader
from tqdm import tqdm


logger = logging.getLogger("logger")


class MetaSimulator:
    """Encapsule all the simulators and perpares the network."""

    def __init__(
        self,
        network_name: str,
        ratio_honest: float = 0.7,
        ratio_wolf: float = 0.1,
        n_new_edges: int = 2,
        random_fit_init: bool = False,
    ) -> None:
        """Define the ratio of honest and criminals."""
        # Define name of simulator
        self._name = "meta_simulator"

        # Check if data is coherent
        assert 0 < ratio_honest < 1, "Ratio needs to be (0,1)"
        assert 0 < ratio_wolf < 1, "Ratio needs to be (0,1)"
        assert 0 < ratio_wolf + ratio_honest < 1, "Together the ratio should be (0,1)"

        self.ratio_honest = ratio_honest
        self.ratio_wolf = ratio_wolf
        self.ratio_criminal = 1 - self.ratio_honest - self.ratio_wolf

        self.network = self.prepare_network(network_name)

        # Network needs to have a base criminal network
        self.n_criminal = len(gt.find_vertex(self.network, self.network.vp.state, "c"))
        (
            self.new_nodes,
            self.total_number_nodes,
            self.relative_ratio_honest,
            self.relative_ratio_wolf,
        ) = self.compute_the_ratio(self.n_criminal)

        # Add the new nodes
        self.network = self.initialise_network(self.network, n_new_edges)

        # Init fitness
        self.network = self.init_fitness(self.network, random_fit_init)

        # Init filtering
        self.network = self.init_filtering(self.network)

    @property
    def name(self) -> str:
        """Return the name of the simulator."""
        return self._name

    def prepare_network(self, network_name: str) -> gt.Graph:
        """Get the network."""
        # Get actual criminal network
        nx_network = NetworkReader().get_data(network_name)
        logger.info(f"The data used is {nx_network.name}")

        # Convert to gt.Graph
        gt_network = NetworkConverter.nx_to_gt(nx_network)
        assert gt_network.vp.state, "Network has no attribute state"
        return gt_network

    def compute_the_ratio(self, n_criminal: int) -> Tuple[int, int, float, float]:
        """Compute the number of nodes to add given the number of criminals.

        Additionally computes the relative ratio for wolfs and honest to be added.
        """
        # Network needs to have a base criminal network
        assert n_criminal >= 1, "The given network contains no criminals..."

        total_number_nodes = int(n_criminal / self.ratio_criminal)
        new_nodes = total_number_nodes - n_criminal

        # Init either honest or lone wolf
        relative_ratio_honest = self.ratio_honest / (
            self.ratio_honest + self.ratio_wolf
        )
        relative_ratio_wolf = 1 - relative_ratio_honest
        return new_nodes, total_number_nodes, relative_ratio_honest, relative_ratio_wolf

    def initialise_network(self, network: gt.Graph, n_new_edges: int = 2) -> gt.Graph:
        """Add to the existing criminal network honest and lone wolfs.

        Thereby, the nodes are added based on the preferential attachment principle.
        Returns a network with new added nodes respecting the ratio of criminals/honest/wolfs.
        """
        new_network = NetworkCombiner.combine_by_preferential_attachment_faster(
            network, new_nodes=self.new_nodes, n_new_edges=n_new_edges
        )

        # Get all the agents with no states
        nodes_no_states = gt.find_vertex(new_network, new_network.vp.state, "")
        tq = tqdm(
            nodes_no_states,
            desc="Adding attributes to nodes",
            total=self.new_nodes,
            leave=False,
            disable=True,
        )
        for i in tq:
            new_network.vp.state[new_network.vertex(i)] = np.random.choice(
                ["h", "w"], 1, p=[self.relative_ratio_honest, self.relative_ratio_wolf]
            )[0]

        return new_network

    def init_fitness(self, network: gt.Graph, random_fit: bool) -> gt.Graph:
        """Add the attribute fitness to the network."""
        if "fitness" in network.vp:
            return network
        else:
            fitness = network.new_vertex_property("double")
            if random_fit:
                fitness.a = np.random.random(network.num_vertices())
            network.vertex_properties["fitness"] = fitness
        return network

    def init_filtering(self, network: gt.Graph) -> gt.Graph:
        """Add the filtering attribute.

        By doing so, we can later filter out the criminal network
        """
        if "filtering" in network.vp:
            return network
        else:
            filtering = network.new_vertex_property("bool")
            network.vertex_properties["filtering"] = filtering
        return network
