"""This script's intention is to simulate the evolution of a criminal network.

Martinez-Vaquero, L. A., Dolci, V., & Trianni, V. (2019).
Evolutionary dynamics of organised crime and terrorist networks. Scientific reports, 9(1), 1-10.

__author__ = Louis Weyland
__date__   = 11/04/2022
"""
import itertools
import logging
import random
from typing import Any
from typing import Dict
from typing import FrozenSet
from typing import List
from typing import Tuple

import graph_tool.all as gt
import numpy as np
from network_utils.network_combiner import NetworkCombiner
from sim_mart_vaq_helper_c import divide_network_fast_loop
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
        delta: int = 0,
        tau: int = 0,
        gamma: float = 0.5,
        beta_s: int = 0,
        beta_h: int = 10,
        beta_c: int = 400,
        c_w: int = 1,
        c_c: int = 1,
        c_k: int = 5,
        r_w: int = 1,
        r_c: int = 1,
        r_k: int = 1,
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
        assert self.n_criminal >= 1, "The given network contains no criminals..."

        self.total_number_nodes = int(self.n_criminal / self.ratio_criminal)
        self.new_nodes = self.total_number_nodes - self.n_criminal

        # Init either honest or lone wolf
        self.relative_ratio_honest = self.ratio_honest / (
            self.ratio_honest + self.ratio_wolf
        )
        self.relative_ratio_wolf = 1 - self.relative_ratio_honest

        # Set damaging and punishing parameters
        self.delta = delta
        self.tau = tau
        self.gamma = gamma
        self.beta_s = beta_s
        self.beta_h = beta_h
        self.beta_c = beta_c
        self.c_w = c_w
        self.c_c = c_c
        self.c_k = c_k
        self.r_w = r_w
        self.r_c = r_c
        self.r_k = r_k

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

    def play(
        self,
        network: gt.Graph,
        rounds: int = 1,
        n_new_edges: int = 2,
        min_grp: int = 5,
        max_grp: int = 20,
        radius: int = 3,
    ) -> None:
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
            dict_of_group = self.select_multiple_communities(
                network=network, radius=radius, min_grp=min_grp, max_grp=max_grp
            )
            logger.debug(f"The Network is divided in {len(dict_of_group)} groups")

            # Go through each group
            for group_number, group_members in dict_of_group.items():
                self.acting_stage(network, group_number, group_members)

    def acting_stage(
        self,
        network: gt.Graph,
        group_number: int,
        group_members: FrozenSet[int],
    ) -> Tuple[Any, int, str]:
        """Correspond to the acting stage in the paper.

        Given an group, select on person and proceed to the acting.
        """
        # Select one person
        slct_pers = np.random.choice(list(group_members), 1)
        # Check the person status
        slct_pers_status = network.vp.state[network.vertex(slct_pers)]

        if slct_pers_status == "h":
            return network, slct_pers, slct_pers_status
        elif slct_pers_status == "c":
            new_network = self.inflict_damage(
                network, group_members, slct_pers, slct_pers_status
            )
            return new_network, slct_pers, slct_pers_status
        elif slct_pers_status == "w":
            new_network = self.inflict_damage(
                network, group_members, slct_pers, slct_pers_status
            )
            return new_network, slct_pers, slct_pers_status
        else:
            raise KeyError("Person status didn't correspond to h/c/w...")

    def inflict_damage(
        self,
        network: gt.Graph,
        group_members: FrozenSet[int],
        slct_pers: int,
        slct_pers_status: str,
    ) -> gt.Graph:
        """Perform criminal activity.

        Rest of the group gets a damage inflicted
        """
        n_c, n_h, n_w, p_c, p_h, p_w = self.counting_status_proprotions(
            network=network, group_members=group_members
        )

        if slct_pers_status == "c":
            # Inflict damage to all the wolfs and honest
            for member in group_members:
                if (
                    network.vp.state[network.vertex(member)] == "h"
                    or network.vp.state[network.vertex(member)] == "w"
                ):
                    network.vp.fitness[network.vertex(member)] = (
                        network.vp.fitness[network.vertex(member)] - self.r_k * self.c_k
                    )
                elif network.vp.state[network.vertex(member)] == "c":
                    network.vp.fitness[network.vertex(member)] = network.vp.fitness[
                        network.vertex(member)
                    ] + (((n_h + n_w) * (self.r_k * self.c_k)) / n_c)

        elif slct_pers_status == "w":
            # Inflicting damage to everyone but himself
            for member in group_members:
                if member != slct_pers:
                    network.vp.fitness[network.vertex(member)] = network.vp.fitness[
                        network.vertex(member)
                    ] - (self.r_k * self.c_k)
                elif member == slct_pers:
                    network.vp.fitness[network.vertex(member)] = network.vp.fitness[
                        network.vertex(member)
                    ] + (len(group_members) - 1) * (self.r_k * self.c_k)

        else:
            raise KeyError("Person status didn't correspond to c/w...")

        return network

    def counting_status_proprotions(
        self, network: gt.Graph, group_members: FrozenSet[int]
    ) -> Tuple[int, int, int, float, float, float]:
        """Return the proportions of criminals,honest and wolfs."""
        # First get proportions of h/c/w within the group
        statuses = []
        size_group = len(group_members)
        for member in group_members:
            statuses.append(network.vp.state[network.vertex(member)])

        n_h = statuses.count("h")
        n_c = statuses.count("c")
        n_w = statuses.count("w")
        p_h = n_h / size_group
        p_c = n_c / size_group
        p_w = n_w / size_group
        return n_c, n_h, n_w, p_c, p_h, p_w

    def divide_in_groups(
        self, network: gt.Graph, min_group: int
    ) -> Tuple[List[int], FrozenSet[int]]:
        """Divide the network in groups.

        Making use of the  minimize_blockmodel_dl func.
        For now, the number of groups can't be imposed.
        Returns a list with the group number/label.
        """
        logger.warning("This function is deprecated!")
        partitions = gt.minimize_blockmodel_dl(
            network, multilevel_mcmc_args={"B_min": min_group}
        )
        mbr_list = partitions.get_blocks()
        group_numbers = frozenset(mbr_list)
        return list(mbr_list), group_numbers

    def act_divide_in_groups(
        self, network: gt.Graph, min_grp: int, max_grp: int
    ) -> gt.Graph:
        """Divide the network in groups.

        Actually divides in n_groups of connected components
        """
        logger.info("This function is deprecated")
        assert (
            2 <= min_grp <= max_grp
        ), f"Min number of groups must be between 2 and {max_grp}"
        assert (
            min_grp <= max_grp <= network.num_vertices()
        ), "Maximum group number can exceed network size"
        n_groups = np.random.randint(low=min_grp, high=max_grp + 1)
        logger.debug(f"Number of groups is {n_groups}")
        # Define group_numbers attribute
        # If it already exists it will overwrite it
        gpr_nbr = network.new_vertex_property("int32_t")
        network.vertex_properties["grp_nbr"] = gpr_nbr

        # Get a map of the nodes and its neighbours
        dict_nodes_and_neighbour = {}
        for v in network.iter_vertices():
            dict_nodes_and_neighbour[v] = list(network.iter_all_neighbors(v))

        # Set the seed, number needs to start from 1 because default is 0
        for v, group_number in zip(
            np.random.choice(
                list(range(0, network.num_vertices())),
                n_groups,
            ),
            range(1, n_groups + 1),
        ):
            network.vp.grp_nbr[network.vertex(v)] = group_number

        # Loop through the dict and assinging same value to its neigbours
        pbar = tqdm(total=len(dict_nodes_and_neighbour))
        while len(dict_nodes_and_neighbour) > 0:
            key_to_del = []
            # random order in order to avoid an group to grow too much
            nodes = list(dict_nodes_and_neighbour.keys())
            random.shuffle(nodes)
            for i in nodes:
                # if node has group number
                if network.vp.grp_nbr[network.vertex(i)] != 0:
                    neighbours = list(network.iter_all_neighbors(i))
                    for neighbour in neighbours:
                        if network.vp.grp_nbr[network.vertex(neighbour)] == 0:
                            network.vp.grp_nbr[
                                network.vertex(neighbour)
                            ] = network.vp.grp_nbr[network.vertex(i)]
                    # del key since all neighbours have a group number
                    key_to_del.append(i)

            # if key_to_del is empty
            # means the seed can not reach some isolated components
            if len(key_to_del) == 0:
                break
            else:
                for k in key_to_del:
                    dict_nodes_and_neighbour.pop(k, None)

            pbar.update(1)

        return network, n_groups

    def act_divide_in_groups_faster(
        self, network: gt.Graph, min_grp: int, max_grp: int
    ) -> gt.Graph:
        """Divide the network in groups.

        Actually divides in n_groups of connected components
        """
        assert (
            2 <= min_grp <= max_grp
        ), f"Min number of groups must be between 2 and {max_grp}"
        assert (
            min_grp <= max_grp <= network.num_vertices()
        ), "Maximum group number can exceed network size"
        n_groups = np.random.randint(low=min_grp, high=max_grp + 1)
        logger.debug(f"Number of groups is {n_groups}")
        # Define group_numbers attribute
        # If it already exists it will overwrite it
        gpr_nbr = network.new_vertex_property("int32_t")
        network.vertex_properties["grp_nbr"] = gpr_nbr

        # Get a map of the nodes and its neighbours
        dict_nodes_and_neighbour = {}
        for v in network.iter_vertices():
            dict_nodes_and_neighbour[v] = list(network.iter_all_neighbors(v))

        nodes_group = divide_network_fast_loop(
            dict_nodes_and_neighbour, n_groups, network.num_vertices()
        )

        for node, group_number in nodes_group.items():
            if group_number is not None:
                network.vp.grp_nbr[network.vertex(node)] = group_number
            elif group_number is None:
                # In order to make sure the result is the same as for the slower function
                network.vp.grp_nbr[network.vertex(node)] = 0

        return network, n_groups

    def select_multiple_communities(
        self, network: gt.Graph, radius: int, min_grp: int, max_grp: int
    ) -> Dict[int, FrozenSet[int]]:
        """Define the groups by randomly selecting one node and it's neighbours within radius r.

        This is done in an iterative fashion.Thus some groups can overlapp.
        """
        assert (
            2 <= min_grp <= max_grp
        ), f"Min number of groups must be between 2 and {max_grp}"
        assert (
            min_grp <= max_grp <= network.num_vertices()
        ), "Maximum group number can exceed network size"
        n_groups = np.random.randint(low=min_grp, high=max_grp + 1)
        logger.debug(f"Number of groups is {n_groups}")

        dict_of_groups = {}
        for n in range(1, n_groups + 1):
            seed = np.random.randint(0, network.num_vertices())
            dict_of_groups[n] = self.select_communities(network, radius, seed)

        return dict_of_groups

    def select_communities(
        self, network: gt.Graph, radius: int, seed: int
    ) -> FrozenSet[int]:
        """Select the neighbours and neighbours neighbours of a given node/seed.

        Args:
            network (gt.Graph): graph-tool network
            radius (int): how many neigbours to select(neighbours of neighbours of...)
            seed (int):  starting node
        """
        nbrs = {seed}
        all_neighbours = []
        for _ in range(radius):
            nbrs = {nbr for n in nbrs for nbr in network.iter_all_neighbors(n)}
            all_neighbours.append(list(nbrs))
        all_neighbours_list = list(itertools.chain.from_iterable(all_neighbours))
        return frozenset(all_neighbours_list)

    def init_fitness(self, network: gt.Graph) -> gt.Graph:
        """Add the attribute fitness to the network."""
        if "fitness" in network.vp:
            return network
        else:
            fitness = network.new_vertex_property("double")
            network.vertex_properties["fitness"] = fitness

        return network
