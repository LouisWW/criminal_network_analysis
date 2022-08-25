"""This script contains the MetaSimulator.

The MetaSimulator encapsules all the simulators.
It prepapres the network for the simulators. From this script,
the different models can be run.

__author__ = Louis Weyland
__date__ = 17/05/2022
"""
import logging
from copy import deepcopy
from typing import Any
from typing import DefaultDict
from typing import List
from typing import Tuple
from typing import Union

import graph_tool.all as gt
import numpy as np
from network_utils.network_combiner import NetworkCombiner
from network_utils.network_converter import NetworkConverter
from network_utils.network_reader import NetworkReader
from simulators.sim_mart_vaq import SimMartVaq
from tqdm import tqdm


logger = logging.getLogger("logger")


class MetaSimulator:
    """Encapsule all the simulators and prepares the network."""

    def __init__(
        self,
        network_name: str,
        attachment_method: str,
        ratio_honest: float = 0.7,
        ratio_wolf: float = 0.1,
        k: int = 2,
        prob: float = 0.4,
        random_fit_init: bool = False,
    ) -> None:
        """Define the ratio of honest and criminals.

        Args:
            network_name (str): _description_
            ratio_honest (float, optional): Honest ratio. Defaults to 0.7.
            ratio_wolf (float, optional): Wolf ratio. Defaults to 0.1.
            k (int, optional): Number of edges to add for preferential attachment/rnd/sw..
                                                            Defaults to 2.
            random_fit_init (bool, optional): Init random fintess. Defaults to False.
        """
        # Define name of simulator
        self._name = "meta_simulator"
        self.attachment_method = attachment_method

        # Check if data is coherent
        assert 0 < ratio_honest < 1, "Ratio needs to be (0,1)"
        assert 0 < ratio_wolf < 1, "Ratio needs to be (0,1)"
        assert 0 < ratio_wolf + ratio_honest < 1, "Together the ratio should be (0,1)"

        self.ratio_honest = ratio_honest
        self.ratio_wolf = ratio_wolf
        self.ratio_criminal = 1 - self.ratio_honest - self.ratio_wolf

        self.criminal_network = self.prepare_network(network_name)
        self.network_name = self.criminal_network.gp.name

        # Network needs to have a base criminal network
        self.n_criminal = len(
            gt.find_vertex(self.criminal_network, self.criminal_network.vp.state, "c")
        )
        (
            self.new_nodes,
            self.total_number_nodes,
            self.relative_ratio_honest,
            self.relative_ratio_wolf,
        ) = self.compute_the_ratio(self.n_criminal)

        # network property
        self.k = k
        self.prob = prob
        self.random_fit_init = random_fit_init

        self.network = self.create_population(self.criminal_network)

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

    def create_population(self, network: gt.Graph) -> gt.Graph:
        """Create the population."""
        # Add the new nodes
        network = self.initialise_network(network, self.prob, self.k)
        # Init fitness
        network = self.init_fitness(network, self.random_fit_init)
        # Init age
        network = self.init_age(network)
        # Init filtering
        network = self.init_filtering(network)
        return network

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

    def initialise_network(
        self, network: gt.Graph, prob: float = 0.3, k: int = 2
    ) -> gt.Graph:
        """Add to the existing criminal network honest and lone wolfs.

        Thereby, the nodes are added based on the preferential attachment principle.
        Returns a network with new added nodes respecting the ratio of criminals/honest/wolfs.
        """
        new_network = deepcopy(network)
        if self.attachment_method == "preferential":
            new_network = NetworkCombiner.combine_by_preferential_attachment_faster(
                new_network, new_nodes=self.new_nodes, k=k
            )[0]
        elif self.attachment_method == "random":
            new_network = NetworkCombiner.combine_by_random_attachment_faster(
                new_network, new_nodes=self.new_nodes, k=k
            )[0]
        elif self.attachment_method == "small-world":
            new_network = NetworkCombiner.combine_by_small_world_attachment(
                new_network, new_nodes=self.new_nodes, k=k, prob=prob
            )[0]
        else:
            raise RuntimeError(
                "Define a network attachment method : 'preferential','random','small-world'"
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
                fitness.a = np.random.uniform(-50, 50, network.num_vertices())
            network.vertex_properties["fitness"] = fitness
        return network

    def init_age(self, network: gt.Graph) -> gt.Graph:
        """Add the attribute age to the network."""
        if "age" in network.vp:
            return network
        else:
            age = network.new_vertex_property("double")
            network.vertex_properties["age"] = age
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

    def create_list_of_populations(self, repetition: int) -> List[gt.Graph]:
        """Create a list of n different populations."""
        list_of_population = []
        tq = tqdm(
            range(repetition),
            desc="Creating list of populations",
            total=repetition,
            leave=False,
            disable=True,
        )
        for _ in tq:
            list_of_population.append(self.create_population(self.criminal_network))
        return list_of_population

    def avg_play(
        self,
        rounds: int = 1,
        n_groups: int = 20,
        ith_collect: int = 20,
        repetition: int = 20,
        measure_topology: bool = False,
        measure_likelihood_corr: bool = False,
    ) -> DefaultDict[str, Union[DefaultDict[Any, Any], List[Any]]]:
        """Get the average results of the simulation given the parameters.

        Same as the avg_play function from sim_mart_vaq.py
        The difference is that instead of running the simulation n (number of repetition) times
        on the same network, the simulation is run once on n (number of repetition) networks
        Args:
            rounds (int, optional): Rounds to play in the simulation. Defaults to 1.
            n_groups (int,optional): Number of groups to form each round
            repetition (int, optional): number of repetition of the simulation. Defaults to 20.

        Returns:
            DefaultDict[Union[int, str], Union[DefaultDict,List[Any]]]:
                                                            Returns network and collected data.
        """
        # create n different populations
        list_of_population = self.create_list_of_populations(repetition)

        # init simulator and use a place holder population
        simulator = SimMartVaq(list_of_population[0])

        data_collector = simulator.avg_play(
            list_of_population,
            rounds=rounds,
            n_groups=n_groups,
            ith_collect=ith_collect,
            repetition=repetition,
            measure_topology=measure_topology,
            measure_likelihood_corr=measure_likelihood_corr,
        )
        return data_collector
