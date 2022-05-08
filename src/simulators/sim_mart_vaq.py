"""This script's intention is to simulate the evolution of a criminal network.

Martinez-Vaquero, L. A., Dolci, V., & Trianni, V. (2019).
Evolutionary dynamics of organised crime and terrorist networks. Scientific reports, 9(1), 1-10.

__author__ = Louis Weyland
__date__   = 11/04/2022
"""
import itertools
import logging
import math
import random
from collections import defaultdict
from copy import deepcopy
from typing import Any
from typing import DefaultDict
from typing import Dict
from typing import FrozenSet
from typing import List
from typing import Tuple

import graph_tool.all as gt
import numpy as np
from src.network_utils.network_combiner import NetworkCombiner
from src.simulators.sim_mart_vaq_helper_c import divide_network_fast_loop
from tqdm import tqdm

logger = logging.getLogger("logger")


class SimMartVaq:
    """Contain the framework to simulate the process."""

    def __init__(
        self,
        network: gt.Graph,
        ratio_honest: float = 0.7,
        ratio_wolf: float = 0.1,
        delta: float = 0.0,
        tau: float = 0.0,
        gamma: float = 0.5,
        beta_s: int = 0,
        beta_h: int = 10,
        beta_c: int = 400,
        c_w: int = 1,
        c_c: int = 1,
        r_w: int = 1,
        r_c: int = 1,
        temperature: float = 10,
        mutation_prob: float = 0.3,
    ) -> None:
        """Init the network charateristics.

        Args:
            network (gt.Graph): Initial criminal network
            ratio_honest (float, optional): Honest ratio. Defaults to 0.7.
            ratio_wolf (float, optional): Wolf ratio. Defaults to 0.1.
            delta (int, optional): Influence of criminals on the acting of the wolf. Defaults to 0.
            tau (int, optional):Influence of the wolf's action on criminals. Defaults to 0.
            gamma (float, optional): Punishment ratio for the other members of the criminal
                                                                    organization. Defaults to 0.5.
            beta_s (int, optional): State punishment value. Defaults to 0.
            beta_h (int, optional): Civil punishment value. Defaults to 10.
            beta_c (int, optional): Criminal punishment value. Defaults to 400.
            c_w (int, optional): Damage caused by wolf. Defaults to 1.
            c_c (int, optional): Damage caused by criminal. Defaults to 1.
            r_w (int, optional): Damage ratio for wolf. Defaults to 1.
            r_c (int, optional): Damage ratio for criminal. Defaults to 1.
            temperature (int, optional): Temperature used in the fermi function. Defaults to 10.
            mutation_prob (float, optional): Mutation probability to either randomly pick a
                                new state or switch state with an other agent. Defaults to 0.3.
        """
        # Define name of simulator
        self._name = "sim_mart_vaq"
        self.network = network

        # Check if data is coherent
        assert isinstance(network, gt.Graph), "Network should be of type gt."
        assert 0 < ratio_honest < 1, "Ratio needs to be (0,1)"
        assert 0 < ratio_wolf < 1, "Ratio needs to be (0,1)"
        assert 0 < ratio_wolf + ratio_honest < 1, "Together the ratio should be (0,1)"
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
        self.r_w = r_w
        self.r_c = r_c

        # Set the fermic temperature & mutation probability
        self.temperature = temperature
        self.mutation_prob = mutation_prob

    @property
    def name(self) -> str:
        """Return the name of the simulator."""
        return self._name

    def initialise_network(self, network: gt.Graph, n_new_edges: int = 2) -> gt.Graph:
        """Add to the existing criminal network honest and lone wolfs.

        Thereby, the nodes are added based on the preferential attachment principle.
        Returns a network with new added nodes respecting the ratio of criminals/honest/wolfs.
        """
        logger.info(
            f"Given the ratio param, {self.new_nodes}\
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
    ) -> Tuple[gt.Graph, DefaultDict[str, List[Any]]]:
        """Run the simulation.

        Network is subdivided in to n groups.
        In each group, a person is selected.
        If selected person is a wolf or criminal,
        damage is inflicted on others.
        """
        # collectors which collects the ratio and fitness over each iteration
        data_collector = defaultdict(
            list,
            {
                k: []
                for k in (
                    "ratio_honest",
                    "ratio_criminal",
                    "ratio_wolf",
                    "fitness_honest",
                    "fitness_criminal",
                    "fitness_wolf",
                )
            },
        )  # type: DefaultDict[str, List[Any]]
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
                # Acting stage
                network, slct_pers, slct_status = self.acting_stage(
                    network, group_members
                )
                # Investigation stage
                network = self.investigation_stage(
                    network, group_members, slct_pers, slct_status
                )
                # Evolutionary stage
                network = self.evolutionary_stage(network, group_members)

            # Collect the data
            _, _, _, p_c, p_h, p_w = self.counting_status_proprotions(
                network=network,
                group_members=frozenset(range(0, network.num_vertices())),
            )
            mean_h_fit, mean_c_fit, mean_w_fit = self.get_overall_fitness_distribution(
                network=network,
                group_members=list(range(0, network.num_vertices())),
            )
            data_collector["ratio_honest"].append(p_h)
            data_collector["ratio_wolf"].append(p_w)
            data_collector["ratio_criminal"].append(p_c)
            data_collector["fitness_honest"].append(mean_h_fit)
            data_collector["fitness_criminal"].append(mean_c_fit)
            data_collector["fitness_wolf"].append(mean_w_fit)

        return network, data_collector

    def investigation_stage(
        self,
        network: gt.Graph,
        group_members: FrozenSet[int],
        slct_pers: int,
        slct_status: str,
    ) -> gt.Graph:
        """Correspond to the investigation stage.

        Given an group, if the victimizer is found, a punishment is conducted
        """
        # Get the status proportions of the group
        _, _, _, p_c, p_h, _ = self.counting_status_proprotions(network, group_members)
        if slct_status == "h":
            # No victimizer ->  No punishment
            return network

        # If victimizer is found, penalties shouldn't 0.
        # state investigation
        state_penalty = self.conducting_investigation(
            group_members, slct_pers, self.beta_s
        )
        # civil investigation
        civil_penalty = (
            self.conducting_investigation(group_members, slct_pers, self.beta_h) * p_h
        )
        # criminal investigation
        criminal_penalty = (
            self.conducting_investigation(group_members, slct_pers, self.beta_c) * p_c
        )

        if slct_status == "c":
            # Punish the victimizer
            network.vp.fitness[network.vertex(slct_pers)] = (
                network.vp.fitness[network.vertex(slct_pers)]
                - state_penalty
                - civil_penalty
            )
            # Punish the partner in crime
            for member in group_members:
                if (
                    network.vp.state[network.vertex(member)] == "c"
                    and member != slct_pers
                ):
                    network.vp.fitness[network.vertex(member)] = network.vp.fitness[
                        network.vertex(member)
                    ] - self.gamma * (state_penalty + civil_penalty)

        elif slct_status == "w":
            # only punish a wolf if he dared to act
            if self.wolf_acting is True:
                # Punish the victimizer
                network.vp.fitness[network.vertex(slct_pers)] = (
                    network.vp.fitness[network.vertex(slct_pers)]
                    - state_penalty
                    - civil_penalty
                    - criminal_penalty
                )

        else:
            raise KeyError("slct_status should be either h/w/c...")

        return network

    def conducting_investigation(
        self, group_members: FrozenSet[int], slct_pers: int, penalty_score: int
    ) -> int:
        """Perform an state investigation.

        Pick a random person, if victimizer is found, penalty is returned
        """
        random_picked_person = np.random.choice(list(group_members), 1)
        if random_picked_person == slct_pers:
            # Found victimizer
            return penalty_score
        elif random_picked_person != slct_pers:
            # Victimizer not found
            return 0
        return None

    def acting_stage(
        self,
        network: gt.Graph,
        group_members: FrozenSet[int],
    ) -> Tuple[gt.Graph, int, str]:
        """Correspond to the acting stage in the paper.

        Given an group, select on person and proceed to the acting.
        """
        # Select one person
        slct_pers = np.random.choice(list(group_members), 1)[0]
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
                if network.vp.state[network.vertex(member)] in ["h", "w"]:
                    network.vp.fitness[network.vertex(member)] = (
                        network.vp.fitness[network.vertex(member)] - self.r_c * self.c_c
                    )
                elif network.vp.state[network.vertex(member)] == "c":
                    network.vp.fitness[network.vertex(member)] = network.vp.fitness[
                        network.vertex(member)
                    ] + (((n_h + n_w) * (self.r_c * self.c_c)) / n_c)

        elif slct_pers_status == "w":
            # Decide if lone wolf dares to act
            self.wolf_acting = False
            if np.random.uniform() >= 1 - self.delta * (1 - p_c):
                self.wolf_acting = True
            # Inflicting damage to everyone but himself
            if self.wolf_acting is True:
                for member in group_members:
                    if member != slct_pers and network.vp.state[
                        network.vertex(member)
                    ] in ["h", "w"]:
                        network.vp.fitness[network.vertex(member)] = network.vp.fitness[
                            network.vertex(member)
                        ] - (self.r_w * self.c_w)

                    elif (
                        member != slct_pers
                        and network.vp.state[network.vertex(member)] == "c"
                    ):
                        network.vp.fitness[network.vertex(member)] = (
                            network.vp.fitness[network.vertex(member)]
                            - (self.r_w * self.c_w)
                            + (
                                (
                                    self.tau
                                    * ((len(group_members) - 1) * (self.r_w * self.c_w))
                                )
                                / n_c
                            )
                        )

                    elif member == slct_pers:
                        network.vp.fitness[network.vertex(member)] = network.vp.fitness[
                            network.vertex(member)
                        ] + (len(group_members) - 1) * (self.r_w * self.c_w)

        else:
            raise KeyError("Person status didn't correspond to c/w...")

        return network

    def evolutionary_stage(
        self, network: gt.Graph, group_members: FrozenSet[int]
    ) -> gt.Graph:
        """Perform the evolutionary stage.

        Randomly picks a two players and performs either mutation
        or a role switch with a certain probability.
        """
        person_a, person_b = np.random.choice(list(group_members), 2)
        if np.random.rand() > self.mutation_prob:
            # Based on the fermi function will check if an interaction will happen
            network = self.interchange_roles(network, person_a, person_b)
        else:
            # Mutation will happen
            network = self.mutation(network, person_a)
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

    def get_overall_fitness_distribution(
        self, network: gt.Graph, group_members: List[int]
    ) -> Tuple[float, float, float]:
        """Get the mean fintess for the different states in a group."""
        h_fit = []
        c_fit = []
        w_fit = []
        for member in group_members:
            state = network.vp.state[network.vertex(member)]
            if state == "h":
                h_fit.append(network.vp.fitness[network.vertex(member)])
            elif state == "c":
                c_fit.append(network.vp.fitness[network.vertex(member)])
            elif state == "w":
                w_fit.append(network.vp.fitness[network.vertex(member)])
            else:
                raise KeyError("slct_status should be either h/w/c...")

        mean_h_fit = np.mean(h_fit)
        mean_c_fit = np.mean(c_fit)
        mean_w_fit = np.mean(w_fit)

        return mean_h_fit, mean_c_fit, mean_w_fit

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

        # Loop through the dict and assinging same value to its neighbours
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
            radius (int): how many neighbours to select(neighbours of neighbours of...)
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

    def mutation(self, network: gt.Graph, person: int) -> gt.Graph:
        """Perform mutation on a given individual."""
        network.vp.state[network.vertex(person)] = np.random.choice(["c", "h", "w"], 1)[
            0
        ]
        return network

    def interchange_roles(
        self, network: gt.Graph, person_a: int, person_b: int
    ) -> gt.Graph:
        """Interchange roles based on fermin function."""
        fitness_a = network.vp.fitness[network.vertex(person_a)]
        fitness_b = network.vp.fitness[network.vertex(person_b)]

        value_a = deepcopy(network.vp.state[network.vertex(person_a)])
        value_b = deepcopy(network.vp.state[network.vertex(person_b)])

        # Probability that b copies a
        if self.fermi_function(fitness_a, fitness_b):
            network.vp.state[network.vertex(person_b)] = value_a
        # Probability that a copies b
        if self.fermi_function(fitness_b, fitness_a):
            network.vp.state[network.vertex(person_a)] = value_b

        return network

    def fermi_function(self, w_j: float, w_i: float) -> bool:
        """Return the probability of changing their role."""
        prob = 1 / (np.exp(-(w_j - w_i) / self.temperature) + 1)
        if np.random.rand() > prob:
            return False
        else:
            return True
        return None

    def mean_group_size(self, radius: int, min_grp: int, max_grp: int) -> int:
        """Compute the mean average groupe size."""
        group_size_data_collector = defaultdict(
            list
        )  # type: DefaultDict[str, List[Any]]
        group_size_data_collector["group_size"]
        for _ in range(0, 100):
            group_dict = self.select_multiple_communities(
                self.network, radius=radius, min_grp=min_grp, max_grp=max_grp
            )
            for _, v in group_dict.items():
                group_size_data_collector["group_size"].append(len(v))
        return int(np.mean(group_size_data_collector["group_size"]))

    def get_analytical_solution(
        self, radius: int = 3, min_grp: int = 5, max_grp: int = 10
    ) -> Dict[str, float]:
        """Compute the analytical solution.

        It has to be noted that the analytical solution offers limited insight.
        First of all, it doesn't take into account the mutation factor. Second
        it assumes that it is possible to have a sub-population that is filled
        with criminals and wolfs. However, in case there is only one criminal in
        the whole population. The in a sub-population of N players, there can never
        be more than 1 criminal.
        """
        mean_fitness_dict = {"h": 0.0, "c": 0.0, "w": 0.0}

        # Mean group size
        N = self.mean_group_size(radius, min_grp, max_grp)
        Z = self.network.num_vertices()
        Z_c = int(self.ratio_criminal * Z)
        Z_h = int(self.ratio_honest * Z)
        Z_w = int(self.ratio_wolf * Z)

        for k in mean_fitness_dict.keys():
            for N_c_prime in range(0, N):
                for N_w_prime in range(0, N - N_c_prime - 1):
                    # Number of honest is N-n_c-n_w
                    N_h_prime = N - N_c_prime - N_w_prime

                    mean_field_approx = self.mean_field_approx(
                        p_h=N_h_prime / N, p_c=N_c_prime / N, N=N, N_w=N_w_prime
                    )

                    mean_fitness_dict[k] += self.hypergeometric_dist(
                        N_h_prime, N_c_prime, N_w_prime, Z_h, Z_c, Z_w, Z, N, k
                    ) * (mean_field_approx[k]["a"] + mean_field_approx[k]["a"])

        return mean_fitness_dict

    def hypergeometric_dist(
        self,
        N_h: int,
        N_c: int,
        N_w: int,
        Z_h: int,
        Z_c: int,
        Z_w: int,
        Z: int,
        N: int,
        indv: str,
    ) -> float:
        """Compute the hypergeometric distance."""
        if indv == "h":
            if N_h != 0:
                N_h = N_h - 1
                Z_h = Z_h - 1
        elif indv == "c":
            if N_c != 0:
                N_c = N_c - 1
                Z_c = Z_c - 1
        elif indv == "w":
            if N_w != 0:
                N_w = N_w - 1
                Z_w = Z_w - 1

        h = (math.comb(Z_h, N_h) * math.comb(Z_c, N_c) * math.comb(Z_w, N_w)) / (
            math.comb(Z - 1, N - 1)
        )
        return h

    def mean_field_approx(
        self, p_h: float, p_c: float, N: int, N_w: int
    ) -> Dict[str, Dict[str, float]]:
        """Compute the mean field approximation based on the formula of the paper."""
        # compute damage and benefits made by a criminal
        d_c = self.c_c * p_c
        b_c = self.r_c * self.c_c * (1 - p_c)

        # compute the damage and benefits made by a wolf
        p_w_prime = 1 - self.delta * (1 - p_c)
        d_w = self.c_w * p_w_prime * (N_w) / N
        d_w_prime = self.c_w * p_w_prime * (N_w - 1) / N
        b_w = self.r_w * self.c_w * (N - 1) * (1 / N) * p_w_prime

        # Fitness for honest,wolfs and criminals
        # after acting stage
        w_h_a = -d_c - d_w
        w_c_a = b_c + self.tau * b_w - d_w
        w_w_a = (1 - self.tau) * b_w - d_c - d_w_prime

        # Fitness after investigation stage
        w_h_i = 0
        w_c_i = (
            -(self.beta_s + self.beta_h * p_h)
            * p_c
            * (self.gamma * p_c + (1 - self.gamma) * (1 / N))
        )
        w_w_i = (
            -(self.beta_s + self.beta_h * p_h + self.beta_c * p_c)
            * (1 / N)
            * (1 / N)
            * p_w_prime
        )

        mean_field_approx = {
            "h": {"a": w_h_a, "i": w_h_i},
            "c": {"a": w_c_a, "i": w_c_i},
            "w": {"a": w_w_a, "i": w_w_i},
        }
        return mean_field_approx
