"""This script's intention is to simulate the evolution of a criminal network.

Martinez-Vaquero, L. A., Dolci, V., & Trianni, V. (2019).
Evolutionary dynamics of organised crime and terrorist networks. Scientific reports, 9(1), 1-10.

__author__ = Louis Weyland
__date__   = 11/04/2022
"""
import cProfile
import gc
import io
import itertools
import logging
import math
import multiprocessing
import pstats
import random
import warnings
from collections import defaultdict
from pstats import SortKey
from typing import Any
from typing import DefaultDict
from typing import Dict
from typing import FrozenSet
from typing import List
from typing import Tuple
from typing import Union

import graph_tool.all as gt
import numpy as np
import pandas as pd
from network_utils.network_extractor import NetworkExtractor
from network_utils.node_stats import NodeStats
from p_tqdm import p_umap
from tqdm import tqdm
from utils.mt_random_c import random_c
from utils.stats import concat_df
from utils.stats import get_mean_std_over_list

# suppress warnings
warnings.filterwarnings("ignore")
logger = logging.getLogger("logger")


class SimMartVaq:
    """Contain the framework to simulate the process."""

    def __init__(
        self,
        network: gt.Graph,
        delta: float = 0.7,
        tau: float = 0.1,
        gamma: float = 0.8,
        beta_s: int = 1,
        beta_h: int = 1,
        beta_c: int = 1,
        c_w: int = 1,
        c_c: int = 100,
        r_w: int = 1,
        r_c: int = 100,
        r_h: int = 0,
        temperature: float = 10,
        mutation_prob: float = 0.0001,
        execute: str = "parallel",
    ) -> None:
        """Init the network characteristics.

        Args:
            network (gt.Graph): Initial criminal network
            delta (int, optional): Influence of criminals on the acting of the wolf.
                                   Defaults to 0.7.
            tau (int, optional):Influence of the wolf's action on criminals. Defaults to 0.8.
            gamma (float, optional): Punishment ratio for the other members of the criminal
                                                                    organization. Defaults to 0.1.
            beta_s (int, optional): State punishment value. Defaults to 5.
            beta_h (int, optional): Civil punishment value. Defaults to 5.
            beta_c (int, optional): Criminal punishment value. Defaults to 5.
            c_w (int, optional): Damage caused by wolf. Defaults to 1.
            c_c (int, optional): Damage caused by criminal. Defaults to 1.
            r_w (int, optional): Reward ratio for wolf. Defaults to 1.
            r_c (int, optional): Reward ratio for criminal. Defaults to 1.
            r_h (int, optional): Bonus ratio for honest. Defaults to 1.
            temperature (int, optional): Temperature used in the fermi function. Defaults to 10.
            mutation_prob (float, optional): Mutation probability to either randomly pick a
                                new state or switch state with an other agent. Defaults to 0.0001.
            execute (str,optional): Defines if some process should run parallel or sequential.
                                    Default to parallel.
        """
        # Define name of simulator
        self._name = "sim_mart_vaq"
        self.network = network

        # Check if data is coherent
        assert isinstance(network, gt.Graph), "Network should be of type gt."
        assert network.vp.state, "Network has no attribute state"
        self.ratio_honest = (
            len(gt.find_vertex(network, network.vp.state, "h"))
            / self.network.num_vertices()
        )
        self.ratio_wolf = (
            len(gt.find_vertex(network, network.vp.state, "w"))
            / self.network.num_vertices()
        )
        self.ratio_criminal = (
            len(gt.find_vertex(network, network.vp.state, "c"))
            / self.network.num_vertices()
        )

        assert np.isclose(self.ratio_honest + self.ratio_wolf + self.ratio_criminal, 1)

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
        self.r_h = r_h

        # Set the fermic temperature & mutation probability
        self.temperature = temperature
        self.mutation_prob = mutation_prob

        # Set the execute, if parallel or sequential
        self.execute = execute

    @property
    def name(self) -> str:
        """Return the name of the simulator."""
        return self._name

    def play(
        self,
        network: gt.Graph,
        rounds: int = 1,
        n_groups: int = 1,
        ith_collect: int = 20,
        collect_fitness: bool = True,
        measure_topology: bool = False,
        measure_likelihood_corr: bool = False,
        show_no_bar: bool = True,
    ) -> Tuple[gt.Graph, DefaultDict[str, List[Any]]]:
        """Run the simulation.

        Network is subdivided in to n groups.
        In each group, a person is selected.
        If selected person is a wolf or criminal,
        damage is inflicted on others.
        """
        network.status = np.asarray(list(network.vp.state))
        network.fitness = np.zeros(network.num_vertices())
        network.age = np.zeros(network.num_vertices())

        # Creating profile object
        ob = cProfile.Profile()
        ob.enable()
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
                    "secrecy",
                    "flow_information",
                    "size_of_largest_component",
                    "df",
                )
            },
        )  # type: DefaultDict[str, List[Any]]

        # Run the simulation
        for i in tqdm(
            range(1, rounds + 1),
            desc="Playing the rounds...",
            total=rounds,
            leave=False,
            disable=False,
        ):
            # Divide the network in random new groups
            dict_of_group = self.slct_pers_n_neighbours(
                network=network, n_groups=n_groups, network_size=network.num_vertices()
            )

            # Go through each group
            for slct_pers, group_members in dict_of_group.items():

                # Check the person status
                slct_pers_status = network.status[slct_pers]

                # Acting stage
                network, slct_pers, slct_status = self.acting_stage(
                    network, slct_pers, slct_pers_status, group_members
                )
                # Investigation stage
                network = self.investigation_stage(
                    network, group_members, slct_pers, slct_status
                )

            # Divide the network in random new groups for evolutionary process
            dict_of_group_evol = self.slct_pers_n_neighbours(
                network=network,
                n_groups=n_groups,
                network_size=network.num_vertices(),
            )
            # Go through each group
            for slct_pers_evol, group_members_evol in dict_of_group_evol.items():
                # Evolutionary stage
                network = self.evolutionary_stage(
                    network, slct_pers_evol, group_members_evol
                )

            if measure_likelihood_corr:
                network = self.update_age(network)

            # update fitness decay
            network = self.update_fitness(network)

            # Collect the data
            if i % ith_collect == 0 or i == 1:
                _, _, _, p_c, p_h, p_w = self.counting_status_proportions(
                    network=network, group_members=np.arange(0, network.num_vertices())
                )

                data_collector["iteration"].append(i)
                data_collector["ratio_honest"].append(p_h)
                data_collector["ratio_wolf"].append(p_w)
                data_collector["ratio_criminal"].append(p_c)

                if collect_fitness:
                    (
                        mean_h_fit,
                        mean_c_fit,
                        mean_w_fit,
                    ) = self.get_overall_fitness_distribution(
                        network=network,
                        group_members=list(range(0, network.num_vertices())),
                    )
                    data_collector["fitness_honest"].append(mean_h_fit)
                    data_collector["fitness_criminal"].append(mean_c_fit)
                    data_collector["fitness_wolf"].append(mean_w_fit)

                if measure_topology:
                    # Extract the criminal network, the filtering is done on the network object
                    logger.info("Filtering the criminal_network")
                    NetworkExtractor.filter_criminal_network(network)

                    if network.num_vertices() == 0:
                        logger.info("No criminals in the network")
                        logger.info("Calculating the secrecy")
                        data_collector["secrecy"].append(-1)
                        data_collector["density"].append(0)
                        logger.info("Calculating the flow of information")
                        data_collector["flow_information"].append(0)
                        logger.info("Calculating the largest_component")
                        data_collector["size_of_largest_component"].append(0)

                    elif network.num_vertices() >= 1:
                        logger.info("Calculating the secrecy")
                        data_collector["secrecy"].append(NodeStats.get_secrecy(network))
                        data_collector["density"].append(NodeStats.get_density(network))
                        logger.info("Calculating the flow of information")

                        gsc = gt.extract_largest_component(network)
                        if self.execute == "parallel":
                            data_collector["flow_information"].append(
                                NodeStats.get_flow_of_information(gsc)
                            )
                        elif self.execute == "sequential":
                            # the faster version doesn't work sequentially
                            data_collector["flow_information"].append(
                                NodeStats.get_flow_of_information_faster(gsc)
                            )

                        logger.info("Calculating the largest_component")
                        data_collector["size_of_largest_component"].append(
                            NodeStats.get_size_of_largest_component(network)[0]
                        )

                    # Unfilter the network back to its initial configuration
                    logger.info("UnFiltering the criminal_network")
                    NetworkExtractor.un_filter_criminal_network(network)

        # Add a df with the likelihood of being a criminal and the node centrality
        if measure_likelihood_corr:
            data_collector["df"] = self.create_likelihood_corr_df(network)

        ob.disable()
        sec = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(ob, stream=sec).sort_stats(sortby)
        ps.print_stats("sim_mart_vaq.py")
        print(sec.getvalue())
        gc.collect()
        return network, data_collector

    def avg_play(
        self,
        network: Union[gt.Graph, List[gt.Graph]],
        rounds: int = 1,
        n_groups: int = 20,
        ith_collect: int = 20,
        repetition: int = 20,
        measure_topology: bool = False,
        measure_likelihood_corr: bool = False,
        collect_fitness: bool = False,
        show_no_bar: bool = False,
    ) -> DefaultDict[str, Union[DefaultDict[Any, Any], List[Any]]]:
        """Get the average results of the simulation given the parameters.

        Args:
            network (gt.Graph, List): population or list of population
            rounds (int, optional): Rounds to play in the simulation. Defaults to 1.
            n_groups (int,optional): Number of groups to form each round
            repetition (int, optional): number of repetition of the simulation. Defaults to 20.

        Returns:
            DefaultDict[Union[int, str], Union[DefaultDict,List[Any]]]:
                                                            Returns network and collected data.
        """
        # Running multiprocessing
        # If repetition are less than number of cores
        # then don't use all the cores
        if self.execute == "parallel":
            if repetition < multiprocessing.cpu_count() - 1:
                num_cpus = repetition
            else:
                num_cpus = multiprocessing.cpu_count() - 5

            if isinstance(network, gt.Graph):
                results = p_umap(
                    self.avg_play_help,
                    (
                        [
                            # arguments need to be in this order
                            (
                                network,
                                rounds,
                                n_groups,
                                ith_collect,
                                collect_fitness,
                                measure_topology,
                                measure_likelihood_corr,
                                show_no_bar,
                            )
                            for i in range(0, repetition)
                        ]
                    ),
                    **{"num_cpus": num_cpus, "desc": "Repeating simulation...."},
                )
            elif isinstance(network, list):
                results = p_umap(
                    self.avg_play_help,
                    (
                        [
                            # arguments need to be in this order
                            (
                                network[i],
                                rounds,
                                n_groups,
                                ith_collect,
                                collect_fitness,
                                measure_topology,
                                measure_likelihood_corr,
                                show_no_bar,
                            )
                            for i in range(0, repetition)
                        ]
                    ),
                    **{"num_cpus": num_cpus, "desc": "Repeating simulation...."},
                )

            # merge results in a dict
            data_collector = defaultdict(list)
            for i, k in enumerate(results):
                data_collector[str(i)] = k

        elif self.execute == "sequential":
            data_collector = defaultdict(list)
            for i in tqdm(
                range(0, repetition),
                desc="Repeating the simulation...",
                total=repetition,
                leave=False,
            ):
                data_collector[str(i)] = self.avg_play_help(
                    (
                        network[i],
                        rounds,
                        n_groups,
                        ith_collect,
                        collect_fitness,
                        measure_topology,
                        measure_likelihood_corr,
                        show_no_bar,
                    )
                )

        # Data over the different rounds is averaged and std is computed
        averaged_dict = get_mean_std_over_list(data_collector)
        averaged_dict = concat_df(averaged_dict, rounds)
        return averaged_dict

    def avg_play_help(self, tuple_of_variable: Any) -> DefaultDict[str, List[Any]]:
        """Help for the avg_play to return only the default dict."""
        # Set the seed each time, otherwise the simulation will be exactly the same
        random.seed()
        (
            network,
            rounds,
            n_groups,
            ith_collect,
            collect_fitness,
            measure_topology,
            measure_likelihood_corr,
            show_no_bar,
        ) = tuple_of_variable

        _, data_collector = self.play(
            network,
            rounds,
            n_groups,
            ith_collect,
            collect_fitness,
            measure_topology,
            measure_likelihood_corr,
            show_no_bar,
        )
        return data_collector

    def investigation_stage(
        self,
        network: gt.Graph,
        group_members: List[int],
        slct_pers: int,
        slct_status: str,
    ) -> Tuple[gt.Graph]:
        """Correspond to the investigation stage.

        Given an group, if the victimizer is found, a punishment is conducted
        """
        if slct_status == "h":
            # No victimizer ->  No punishment
            return network

        # Get the status proportions of the group
        _, _, _, p_c, p_h, _ = self.counting_status_proportions(network, group_members)

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
            # only punish a wolf if he dared to act
            if self.criminal_acting is True:
                # Punish the victimizer
                network.fitness[slct_pers] = (
                    network.fitness[slct_pers] - state_penalty - civil_penalty
                )
                # Punish the partner in crime
                for member in group_members:
                    if network.status[member] == "c" and member != slct_pers:
                        network.fitness[member] = network.fitness[
                            member
                        ] - self.gamma * (state_penalty + civil_penalty)

        elif slct_status == "w":
            # only punish a wolf if he dared to act
            if self.wolf_acting is True:
                # Punish the victimizer
                network.fitness[slct_pers] = (
                    network.fitness[slct_pers]
                    - state_penalty
                    - civil_penalty
                    - criminal_penalty
                )

        else:
            raise warnings.warn("slct_status should be either h/w/c...")

        return network

    def conducting_investigation(
        self, group_members: List[int], slct_pers: int, penalty_score: int
    ) -> int:
        """Perform an state investigation.

        Pick a random person, if victimizer is found, penalty is returned
        """
        random_picked_person = random.choice(list(group_members))
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
        slct_pers: int,
        slct_pers_status: str,
        group_members: List[int],
    ) -> Tuple[gt.Graph, int, str]:
        """Correspond to the acting stage in the paper.

        Given an group, select on person and proceed to the acting.
        """
        if slct_pers_status == "h":
            new_network = self.inflict_damage(
                network, group_members, slct_pers, slct_pers_status
            )
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
            raise warnings.warn(
                f"Person status {slct_pers_status} didn't correspond to h/c/w..."
            )

    def inflict_damage(
        self,
        network: gt.Graph,
        group_members: List[int],
        slct_pers: int,
        slct_pers_status: str,
    ) -> Tuple[gt.Graph]:
        """Perform criminal activity.

        Rest of the group gets a damage inflicted.
        If slct_pers is honest, the person gets bonus points for acting good!
        """
        if slct_pers_status != "h":
            n_c, n_h, n_w, p_c, p_h, p_w = self.counting_status_proportions(
                network=network, group_members=group_members
            )
        if slct_pers_status == "h":
            # Bonus points if law-abiding
            network.fitness[slct_pers] = network.fitness[slct_pers] + self.r_h
        elif slct_pers_status == "c":
            # Inflict damage to all the wolfs and honest
            # If only criminals are present in the group
            # then there is no acting
            self.criminal_acting = False
            # Honest and wolfs are present
            if p_h + p_w > 0:
                self.criminal_acting = True
            # Inflicting damage to everyone but himself
            if self.criminal_acting is True:
                for member in group_members:
                    if network.status[member] in ["h", "w"]:
                        network.fitness[member] = network.fitness[member] - self.c_c
                    elif network.status[member] == "c":
                        network.fitness[member] = network.fitness[member] + (
                            (self.r_c * self.c_c) / n_c
                        )

        elif slct_pers_status == "w":
            # Decide if lone wolf dares to act
            self.wolf_acting = False
            if random_c() >= 1 - self.delta * (1 - p_c):
                self.wolf_acting = True
            # Inflicting damage to everyone but himself
            if self.wolf_acting is True:
                for member in group_members:
                    if member != slct_pers and network.status[member] in ["h", "w"]:
                        network.fitness[member] = network.fitness[member] - self.c_w

                    elif member != slct_pers and network.status[member] == "c":
                        network.fitness[member] = (
                            network.fitness[member]
                            - self.c_w
                            + ((self.tau * (self.r_w * self.c_w)) / n_c)
                        )

                    elif member == slct_pers:
                        network.fitness[member] = network.fitness[member] + (
                            self.r_w * self.c_w
                        )

        else:
            raise warnings.warn("Person status didn't correspond to c/w...")

        return network

    def evolutionary_stage(
        self, network: gt.Graph, slct_person: int, group_members: List[int]
    ) -> Tuple[gt.Graph]:
        """Perform the evolutionary stage.

        Randomly picks a two players and performs either mutation
        or a role switch with a certain probability.
        """
        person_a = slct_person
        bucket_list = list(group_members)
        bucket_list.remove(person_a)
        if random_c() > self.mutation_prob and len(bucket_list) != 0:
            # Based on the fermi function will check if an interaction will happen
            person_b = random.choice(bucket_list)
            network = self.interchange_roles(network, person_a, person_b)
        else:
            # Mutation will happen
            network = self.mutation(network, person_a)
        return network

    def counting_status_proportions(
        self, network: gt.Graph, group_members: np.ndarray
    ) -> Tuple[int, int, int, float, float, float]:
        """Return the proportions of criminals,honest and wolfs."""
        # First get proportions of h/c/w within the group
        size_group = len(group_members)
        unique, counts = np.unique(network.status[group_members], return_counts=True)
        count_dict = dict(zip(unique, counts))

        if "h" in count_dict:
            n_h = count_dict["h"]
        else:
            n_h = 0

        if "c" in count_dict:
            n_c = count_dict["c"]
        else:
            n_c = 0

        if "w" in count_dict:
            n_w = count_dict["w"]
        else:
            n_w = 0

        p_h = n_h / size_group
        p_c = n_c / size_group
        p_w = n_w / size_group
        return n_c, n_h, n_w, p_c, p_h, p_w

    def get_overall_fitness_distribution(
        self, network: gt.Graph, group_members: List[int]
    ) -> Tuple[float, float, float]:
        """Get the mean fitness for the different states in a group."""
        h_idx = np.where(network.status == "h")
        c_idx = np.where(network.status == "c")
        w_idx = np.where(network.status == "w")

        mean_h_fit = np.mean(network.fitness[h_idx])
        mean_c_fit = np.mean(network.fitness[c_idx])
        mean_w_fit = np.mean(network.fitness[w_idx])
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
        n_groups = random.randint(min_grp, max_grp + 1)
        logger.debug(f"Number of groups is {n_groups}")

        dict_of_groups = {}
        for n in range(1, n_groups + 1):
            seed = random.randint(0, network.num_vertices())
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
            nbrs = {nbr for n in nbrs for nbr in network.get_all_neighbors(n)}
            all_neighbours.append(list(nbrs))
        all_neighbours_list = list(itertools.chain.from_iterable(all_neighbours))
        return frozenset(all_neighbours_list)

    def slct_pers_n_neighbours(
        self, network: gt.Graph, n_groups: int, network_size: int
    ) -> Dict[int, List[int]]:
        """Randomly select the protagonist (person who can act) and its neighbours.

        Args:
            n_groups (int): number of groups to form each round
            network_size (int): get the size of the network to include all the nodes

        Returns:
            Dict[int,List[int]]: key is the protagonist and value is protagonist + neighbors
        """
        communities = {}
        protagonists = random.sample(range(0, network_size), n_groups)
        for protagonist in protagonists:
            communities[protagonist] = [protagonist] + list(
                network.get_all_neighbors(protagonist)
            )

        return communities

    def mutation(self, network: gt.Graph, person: int) -> gt.Graph:
        """Perform mutation on a given individual."""
        network.status[person] = random.choice(["c", "h", "w"])
        return network

    def interchange_roles(
        self,
        network: gt.Graph,
        person_a: int,
        person_b: int,
    ) -> Tuple[gt.Graph]:
        """Interchange roles based on fermin function."""
        fitness_a = network.fitness[person_a]
        fitness_b = network.fitness[person_b]

        # Probability that a copies b
        if self.fermi_function(fitness_b, fitness_a):
            network.status[person_a] = network.status[person_b]

        return network

    def fermi_function(self, w_j: float, w_i: float) -> bool:
        """Return the probability of changing their role."""
        prob = 1 / (np.exp(-(w_j - w_i) / self.temperature) + 1)
        if random_c() > prob:
            return False
        else:
            return True
        return None

    def update_age(self, network: gt.Graph) -> gt.Graph:
        """Update the age of a criminal node.

        Basically, count how many rounds a node has a criminal status criminal
        """
        idx = np.where(network.status == "c")
        network.age[idx] = network.age[idx] + 1
        return network

    def update_fitness(self, network: gt.Graph) -> gt.Graph:
        """Update the fitness in a decay fashion."""
        network.fitness = network.fitness * 0.666
        return network

    def create_likelihood_corr_df(self, network: gt.Graph) -> pd.DataFrame:
        """Create a DataFrame of nodes likelihood of being a criminal and its characteristics."""
        org_num_threads = gt.openmp_get_num_threads()
        if self.execute == "parallel":
            gt.openmp_set_num_threads(1)
        network, _ = NodeStats.get_eigenvector_centrality(network)
        network, _ = NodeStats.get_betweenness(network)
        network, _ = NodeStats.get_closeness(network)
        network, _ = NodeStats.get_katz(network)

        if self.execute == "parallel":
            gt.openmp_set_num_threads(org_num_threads)

        df = pd.DataFrame(
            {
                "criminal_likelihood": network.age,
                "degree": network.get_total_degrees(
                    list(range(0, network.num_vertices()))
                ),
                "betweenness": network.btwn,
                "katz": network.katz,
                "closeness": network.closeness,
                "eigen vector": network.eigen_v,
            }
        )
        df = df.astype(float)
        return df

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
                    ) * (mean_field_approx[k]["a"] + mean_field_approx[k]["i"])

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
