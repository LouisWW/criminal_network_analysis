"""Test if the simulation from Martinez-Vaquero is running correctly."""
from copy import deepcopy

import graph_tool.all as gt
import numpy as np
import pytest
from hypothesis import assume
from hypothesis import given
from hypothesis import strategies as st
from simulators.sim_mart_vaq import SimMartVaq


class TestSimMartVaq:
    """Class for unit tests for  SimMartVaq."""

    @pytest.mark.essential
    def test_sim_mart_vaq(self, gt_network: gt.Graph) -> None:
        """Test if the initialization works."""
        org_size = gt_network.num_vertices()
        # Keep the ratio small so the test will be faster
        # by adding less nodes
        ratio_honest = 0.3
        ratio_wolf = 0.1

        simulators = SimMartVaq(gt_network, ratio_honest, ratio_wolf)

        # Test if the obj is init correctly
        assert isinstance(
            simulators, SimMartVaq
        ), "Simulator hasn't been init correctly"

        # Test if the ration is caluclated correctly
        assert simulators.ratio_criminal == 0.6, "Ratio is wrong."
        assert (
            simulators.n_criminal == org_size
        ), "Determined number of criminals is wrong."
        assert simulators.total_number_nodes == 158, "Ratio is wrong"
        assert simulators.new_nodes == 63, "Number of nodes to add is wrong"
        assert (
            round(simulators.relative_ratio_honest, 2) == 0.75
        ), "Relative ratio is wrong"
        assert (
            round(simulators.relative_ratio_wolf, 2) == 0.25
        ), "Relative ratio is wrong"

        # Try to change its name
        with pytest.raises(Exception):
            simulators.name = "New name"

    @pytest.mark.essential
    def test_sim_mart_vaq_wrong_init(self, gt_network: gt.Graph) -> None:
        """Test if a wrong init triggers the assert statements."""
        # With ratio_honest == 0
        with pytest.raises(Exception):
            assert SimMartVaq(gt_network, ratio_honest=0)

        # With ratio_wolf == 1.1
        with pytest.raises(Exception):
            assert SimMartVaq(gt_network, ratio_wolf=1.1)

        # With ratios not adding up more than 1
        with pytest.raises(Exception):
            assert SimMartVaq(gt_network, ratio_honest=0.8, ratio_wolf=0.3)
            assert SimMartVaq(gt_network, ratio_honest=0.8, ratio_wolf=0.2)
            assert SimMartVaq(gt_network, ratio_honest=-0.8, ratio_wolf=0.1)
            assert SimMartVaq(gt_network, ratio_honest=-0.8, ratio_wolf=0.2)

    @pytest.mark.essential
    def test_initialise_network(self, gt_network: gt.Graph) -> None:
        """Test if the init process is done corretly.

        More precisely tests if the ratio of c/h/w is correct
        """
        simulators = SimMartVaq(gt_network, ratio_honest=0.2, ratio_wolf=0.3)
        network = simulators.initialise_network(simulators.network)

        # Criminal network size should be 95
        # Honest and Wolf ration should be within a range given the init is stochastic
        assert (
            len(gt.find_vertex(network, network.vp.state, "c")) == 95
        ), "Criminal ratio not correct"
        assert (
            38 - 15 <= len(gt.find_vertex(network, network.vp.state, "h")) <= 38 + 15
        ), "Honest ratio not correct"
        assert (
            57 - 15 <= len(gt.find_vertex(network, network.vp.state, "w")) <= 57 + 15
        ), "Wolf ratio not correct"

    @pytest.mark.essential
    def test_divide_in_groups(self, gt_network: gt.Graph) -> None:
        """Test if the division into groups works correctly."""
        ratio_honest = np.random.uniform(0.1, 0.99)
        ratio_wolf = np.random.uniform(0.1, (1 - 0.99))
        simulators = SimMartVaq(gt_network, ratio_honest, ratio_wolf)
        simulators.network = simulators.initialise_network(simulators.network)
        groups, groups_label = simulators.divide_in_groups(
            simulators.network, min_group=3
        )

        assert groups is not np.empty, "Group list is empty"
        assert all(
            [isinstance(item, int) for item in groups]
        ), "Items in Group should be int"
        assert isinstance(groups_label, frozenset), "Group label should be frozenset"
        assert len(groups_label) > 0, "Group label is empty"
        assert all(
            [isinstance(item, int) for item in groups_label]
        ), "Items in Group label should be int"

    @pytest.mark.essential
    def test_act_divide_in_groups_faster(self, gt_network: gt.Graph) -> None:
        """Test if the division into groups works correctly."""
        ratio_honest = np.random.uniform(0.1, 0.80)
        ratio_wolf = np.random.uniform(0.1, (1 - 0.80))
        simulators = SimMartVaq(gt_network, ratio_honest, ratio_wolf)
        simulators.network = simulators.initialise_network(simulators.network)
        divided_network, n_groups = simulators.act_divide_in_groups_faster(
            simulators.network, min_grp=2, max_grp=2
        )

        # Group attribute needs to exist
        assert divided_network.vp.grp_nbr, "Attribute grp_nbr doesn't exist...."

        # Check if divided in two, every node should have a neighbour of the same group#
        # Pick random neighbors
        for i in range(0, 100):
            # Get random node
            x = np.random.uniform(1, divided_network.num_vertices())
            x_group = divided_network.vp.grp_nbr[divided_network.vertex(x)]
            neighbours = list(divided_network.iter_all_neighbors(x))
            neighbours_group = [
                divided_network.vp.grp_nbr[divided_network.vertex(neighbour)]
                for neighbour in neighbours
            ]
            assert x_group in neighbours_group, "No neighbour is of the same group..."

        # Check if recalled, the attributes are indeed reset
        # Set group number to avoid the chance to have same group number again
        divided_network_1, n_groups = simulators.act_divide_in_groups_faster(
            simulators.network, min_grp=100, max_grp=100
        )
        x_group_1 = divided_network_1.vp.grp_nbr[divided_network_1.vertex(25)]
        divided_network_2, n_groups = simulators.act_divide_in_groups_faster(
            divided_network_1, min_grp=100, max_grp=100
        )
        x_group_2 = divided_network_2.vp.grp_nbr[divided_network_2.vertex(25)]
        assert x_group_1 != x_group_2, "Attributes are not reset correctly"

    @pytest.mark.essential
    def test_act_divide_in_groups(self, gt_network: gt.Graph) -> None:
        """Test if the division into groups works correctly."""
        ratio_honest = np.random.uniform(0.1, 0.90)
        ratio_wolf = np.random.uniform(0.1, (1 - ratio_honest - 0.1))
        simulators = SimMartVaq(gt_network, ratio_honest, ratio_wolf)
        simulators.network = simulators.initialise_network(simulators.network)
        divided_network, n_groups = simulators.act_divide_in_groups(
            simulators.network, min_grp=2, max_grp=2
        )

        # Group attribute needs to exist
        assert divided_network.vp.grp_nbr, "Attribute grp_nbr doesn't exist...."

        # Check if divided in two, every node should have a neighbour of the same group#
        # Pick random neighbors
        for i in range(0, 100):
            # Get random node
            x = np.random.uniform(1, divided_network.num_vertices())
            x_group = divided_network.vp.grp_nbr[divided_network.vertex(x)]
            neighbours = list(divided_network.iter_all_neighbors(x))
            neighbours_group = [
                divided_network.vp.grp_nbr[divided_network.vertex(neighbour)]
                for neighbour in neighbours
            ]
            assert x_group in neighbours_group, "No neighbour is of the same group..."

        # Check if recalled, the attributes are indeed reset
        # Set group number to avoid the chance to have same group number again
        divided_network_1, n_groups = simulators.act_divide_in_groups(
            simulators.network, min_grp=100, max_grp=100
        )
        x_group_1 = divided_network_1.vp.grp_nbr[divided_network_1.vertex(25)]
        divided_network_2, n_groups = simulators.act_divide_in_groups(
            divided_network_1, min_grp=100, max_grp=100
        )
        x_group_2 = divided_network_2.vp.grp_nbr[divided_network_2.vertex(25)]
        assert x_group_1 != x_group_2, "Attributes are not reset correctly"

    @pytest.mark.essential
    def test_init_fitness(self, gt_network: gt.Graph) -> None:
        """Test if the init of the fitness attribute is done correctly."""
        simulators = SimMartVaq(gt_network)
        network = simulators.init_fitness(simulators.network)
        assert network.vp.fitness, "Fitness attribute doesn't exists..."

    @pytest.mark.essential
    def test_acting_stage(self, gt_network: gt.Graph) -> None:
        """Test if the acting stage process is working correctly."""
        # Set delta to 100 to make sure wolf will always act
        simulators = SimMartVaq(gt_network, ratio_wolf=0.2, ratio_honest=0.4, delta=100)
        network = simulators.init_fitness(simulators.network)
        network = simulators.initialise_network(network)
        # Network and network_aft_dmge are same object
        # To compare network create an independent copy
        untouched_network = deepcopy(network)
        min_grp = 5
        max_grp = 10

        dict_of_communities = simulators.select_multiple_communities(
            network=gt_network, radius=1, min_grp=min_grp, max_grp=max_grp
        )
        mbrs = dict_of_communities[min_grp]
        n_c, n_h, n_w, p_c, p_h, p_w = simulators.counting_status_proprotions(
            network=network, group_members=mbrs
        )

        # Select one group number from the all the numbers
        network_aft_dmge, slct_pers, slct_pers_status = simulators.acting_stage(
            network, mbrs
        )

        # select random node from group
        node = np.random.choice(list(mbrs), 1)
        if slct_pers_status == "h":
            assert list(network_aft_dmge.vp.fitness) == list(
                untouched_network.vp.fitness
            ), "Fitness should be unchanged..."

        elif slct_pers_status == "c":
            if network.vp.state[network.vertex(node)] == "c":
                assert network_aft_dmge.vp.fitness[
                    network_aft_dmge.vertex(node)
                ] == untouched_network.vp.fitness[untouched_network.vertex(node)] + (
                    ((n_h + n_w) * (simulators.r_c * simulators.c_c)) / n_c
                )
            elif network.vp.state[network.vertex(node)] in ["h", "w"]:
                assert network_aft_dmge.vp.fitness[
                    network_aft_dmge.vertex(node)
                ] == untouched_network.vp.fitness[untouched_network.vertex(node)] - (
                    simulators.r_c * simulators.c_c
                )

        elif slct_pers_status == "w":
            if node == slct_pers:
                assert network_aft_dmge.vp.fitness[
                    network_aft_dmge.vertex(node)
                ] == untouched_network.vp.fitness[untouched_network.vertex(node)] + (
                    len(mbrs) - 1
                ) * (
                    simulators.r_w * simulators.c_w
                )
            else:
                if network_aft_dmge.vp.state[network_aft_dmge.vertex(node)] in [
                    "h",
                    "w",
                ]:
                    assert network_aft_dmge.vp.fitness[
                        network_aft_dmge.vertex(node)
                    ] == untouched_network.vp.fitness[
                        untouched_network.vertex(node)
                    ] - (
                        simulators.r_w * simulators.c_w
                    )
                elif network_aft_dmge.vp.state[network_aft_dmge.vertex(node)] == "c":
                    assert network_aft_dmge.vp.fitness[
                        network_aft_dmge.vertex(node)
                    ] == untouched_network.vp.fitness[
                        untouched_network.vertex(node)
                    ] - (
                        simulators.r_w * simulators.c_w
                    ) + (
                        (
                            simulators.tau
                            * ((len(mbrs) - 1) * (simulators.r_w * simulators.c_w))
                        )
                        / n_c
                    )

                else:
                    raise KeyError(
                        f"Node should have status w/c instead of \
                        {network_aft_dmge.vp.state[network_aft_dmge.vertex(node)]}"
                    )

        else:
            assert slct_pers_status in [
                "c",
                "h",
                "w",
            ], "Returned status should be either c/h/w"

    @pytest.mark.essential
    def test_acting_stage_2(self, gt_network: gt.Graph) -> None:
        """Test if the acting stage process is working correclty.

        This time, lone wolf never act!
        """
        # Set delta to 0 to make sure wolf will never act
        simulators = SimMartVaq(gt_network, ratio_wolf=0.2, ratio_honest=0.4, delta=0)
        network = simulators.init_fitness(simulators.network)
        network = simulators.initialise_network(network)
        # Network and network_aft_dmge are same object
        # To compare network create an independent copy
        untouched_network = deepcopy(network)
        min_grp = 5
        max_grp = 10

        dict_of_communities = simulators.select_multiple_communities(
            network=gt_network, radius=1, min_grp=min_grp, max_grp=max_grp
        )
        mbrs = dict_of_communities[min_grp]
        n_c, n_h, n_w, p_c, p_h, p_w = simulators.counting_status_proprotions(
            network=network, group_members=mbrs
        )

        # Select one group number from the all the numbers
        network_aft_dmge, slct_pers, slct_pers_status = simulators.acting_stage(
            network, mbrs
        )

        # select random node from group
        node = np.random.choice(list(mbrs), 1)
        if slct_pers_status == "h":
            assert list(network_aft_dmge.vp.fitness) == list(
                untouched_network.vp.fitness
            ), "Fitness should be unchanged..."

        elif slct_pers_status == "c":
            if network.vp.state[network.vertex(node)] == "c":
                assert network_aft_dmge.vp.fitness[
                    network_aft_dmge.vertex(node)
                ] == untouched_network.vp.fitness[untouched_network.vertex(node)] + (
                    ((n_h + n_w) * (simulators.r_c * simulators.c_c)) / n_c
                )
            elif network.vp.state[network.vertex(node)] in ["h", "w"]:
                assert network_aft_dmge.vp.fitness[
                    network_aft_dmge.vertex(node)
                ] == untouched_network.vp.fitness[untouched_network.vertex(node)] - (
                    simulators.r_c * simulators.c_c
                )

        elif slct_pers_status == "w":
            assert list(network_aft_dmge.vp.fitness) == list(
                untouched_network.vp.fitness
            ), "Fitness should be unchanged..."

        else:
            assert slct_pers_status in [
                "c",
                "h",
                "w",
            ], "Returned status should be either c/h/w"

    @pytest.mark.essential
    def test_select_communities(self, gt_network: gt.Graph) -> None:
        """Test if the random select communites is working correctly."""
        simulators = SimMartVaq(gt_network)
        network = simulators.init_fitness(simulators.network)

        # Depth 1
        seed = np.random.randint(0, gt_network.num_vertices())
        community = simulators.select_communities(network, radius=1, seed=seed)
        nbrs = network.iter_all_neighbors(seed)
        assert list(nbrs) != community

        # Depth 2
        seed = np.random.randint(0, gt_network.num_vertices())
        community = simulators.select_communities(network, radius=2, seed=seed)
        nbrs = list(network.iter_all_neighbors(seed))
        scnd_degree_nbr: list = []
        for nbr in nbrs:
            scnd_degree_nbr = scnd_degree_nbr + list(network.get_all_neighbors(nbr))

        nbrs = nbrs + scnd_degree_nbr
        assert set(nbrs) == community

        # Depth 3
        seed = np.random.randint(0, gt_network.num_vertices())
        community = simulators.select_communities(network, radius=3, seed=seed)
        nbrs = list(network.iter_all_neighbors(seed))
        scnd_degree_nbr = []
        for nbr in nbrs:
            scnd_degree_nbr = scnd_degree_nbr + list(network.get_all_neighbors(nbr))

        third_degree_nbr: list = []
        for nbr in scnd_degree_nbr:
            third_degree_nbr = third_degree_nbr + list(network.get_all_neighbors(nbr))

        nbrs = nbrs + scnd_degree_nbr + third_degree_nbr
        assert set(nbrs) == community

    @pytest.mark.essential
    def test_select_multiple_communities(self, gt_network: gt.Graph) -> None:
        """Test if select_multiple_communities works correctly."""
        simulators = SimMartVaq(gt_network)
        network = simulators.init_fitness(simulators.network)
        min_grp = 5
        max_grp = 10
        dict_of_communities = simulators.select_multiple_communities(
            network=network, radius=1, min_grp=min_grp, max_grp=max_grp
        )

        assert len(dict_of_communities) != 0, "Dict of communities is empty..."
        assert (
            min_grp <= len(dict_of_communities) <= max_grp
        ), "Number of communites is not correct..."

        for k, v in dict_of_communities.items():
            assert isinstance(k, int)
            assert len(dict_of_communities[k]) >= 1, "Some communities are empty..."

    @pytest.mark.essential
    def test_counting_status_proportions(self, gt_network: gt.Graph) -> None:
        """Test if the counting works correctly."""
        simulators = SimMartVaq(gt_network)
        network = simulators.init_fitness(simulators.network)
        min_grp = 5
        max_grp = 10
        dict_of_communities = simulators.select_multiple_communities(
            network=network, radius=1, min_grp=min_grp, max_grp=max_grp
        )

        for k, v in dict_of_communities.items():
            n_h, n_c, n_w, p_c, p_h, p_w = simulators.counting_status_proprotions(
                network=network, group_members=v
            )

            assert 0 <= p_c <= 1, "Proportion is not calculated correctly"
            assert 0 <= p_h <= 1, "Proportion is not calculated correctly"
            assert 0 <= p_w <= 1, "Proportion is not calculated correctly"
            assert (
                pytest.approx(p_h + p_c + p_w, 0.1) == 1
            ), "Total ration should sum up to 1"

            assert isinstance(n_h, int), "Number should be an int"
            assert isinstance(n_c, int), "Number should be an int"
            assert isinstance(n_w, int), "Number should be an int"
            assert n_h + n_c + n_w == len(v), "Everyone should be c,h,w"

    @pytest.mark.essential
    def test_get_overall_fitness_distribution(
        self, create_gt_network: gt.Graph
    ) -> None:
        """Test if the get overall fitness distribution works."""
        simulators = SimMartVaq(network=create_gt_network)
        mean_h, mean_c, mean_w = simulators.get_overall_fitness_distribution(
            simulators.network, list(range(0, 5))
        )

        assert mean_h == 7, "Mean fitness is not correct..."
        assert mean_c == 10, "Mean fitness is not correct..."
        assert mean_w == 7, "Mean fitness is not correct..."

    @pytest.mark.essential
    def test_inflicting_damage(self, create_gt_network: gt.Graph) -> None:
        """Test if the inflicting damage function works correclty."""
        # set delta to 100 to make sure lone wolf acts
        simulators = SimMartVaq(
            create_gt_network, c_c=5, r_c=1, c_w=3, r_w=2, delta=100, tau=0.5
        )
        network = simulators.init_fitness(simulators.network)

        # What if the criminal is chosen
        node = 0
        network = simulators.inflict_damage(
            simulators.network,
            frozenset([0, 1, 2, 3, 4]),
            node,
            network.vp.state[network.vertex(node)],
        )
        assert network.vp.fitness[network.vertex(0)] == 30
        assert network.vp.fitness[network.vertex(2)] == 1
        assert network.vp.fitness[network.vertex(3)] == 6
        assert network.vp.fitness[network.vertex(4)] == -2

        # What if the criminal is chosen
        # Based on the previous fitness resulting from the criminal
        # activity above.
        node = 3
        network = simulators.inflict_damage(
            network,
            frozenset([0, 1, 2, 3, 4]),
            node,
            network.vp.state[network.vertex(node)],
        )
        assert network.vp.fitness[network.vertex(0)] == 36
        assert network.vp.fitness[network.vertex(2)] == -5
        assert network.vp.fitness[network.vertex(4)] == -8
        assert network.vp.fitness[network.vertex(3)] == 30

    @pytest.mark.essential
    def test_investigation_stage(self, create_gt_network: gt.Graph) -> None:
        """Test if the investigation stage is working correctly."""
        # Define the different penalties
        simulators = SimMartVaq(
            create_gt_network, gamma=0.5, beta_c=2, beta_s=3, beta_h=5
        )
        network = simulators.init_fitness(simulators.network)

        # What if the criminal is chosen
        # Triggers state and civilian punishment
        np.random.seed(2)
        node = 0
        network = simulators.investigation_stage(
            simulators.network,
            frozenset([0, 1, 2, 3, 4]),
            node,
            network.vp.state[network.vertex(node)],
        )
        assert network.vp.fitness[network.vertex(0)] == 5
        assert network.vp.fitness[network.vertex(1)] == 8
        assert network.vp.fitness[network.vertex(2)] == 6
        assert network.vp.fitness[network.vertex(3)] == 11
        assert network.vp.fitness[network.vertex(4)] == 3

        # What if the lone wolf is not chosen
        # Based on the previous fitness resulting from the criminal
        # activity above.
        np.random.seed(3)
        node = 3
        network = simulators.investigation_stage(
            network,
            frozenset([0, 1, 2, 3, 4]),
            node,
            network.vp.state[network.vertex(node)],
        )
        assert network.vp.fitness[network.vertex(0)] == 5
        assert network.vp.fitness[network.vertex(1)] == 8
        assert network.vp.fitness[network.vertex(2)] == 6
        assert network.vp.fitness[network.vertex(3)] == 11
        assert network.vp.fitness[network.vertex(4)] == 3

        # What if the lone wolf is chosen
        # Based on the previous fitness resulting from the criminal
        # activity above.
        np.random.seed(6)
        node = 3
        network = simulators.investigation_stage(
            network,
            frozenset([0, 1, 2, 3, 4]),
            node,
            network.vp.state[network.vertex(node)],
        )
        assert network.vp.fitness[network.vertex(0)] == 5
        assert network.vp.fitness[network.vertex(1)] == 8
        assert network.vp.fitness[network.vertex(2)] == 6
        assert network.vp.fitness[network.vertex(3)] == 10.6
        assert network.vp.fitness[network.vertex(4)] == 3

        # Check in case multiple criminals are present
        network.vp.state[network.vertex(4)] = "c"
        # Triggers state and civilian punishment
        np.random.seed(2)
        node = 0
        network = simulators.investigation_stage(
            simulators.network,
            frozenset([0, 1, 2, 3, 4]),
            node,
            network.vp.state[network.vertex(node)],
        )
        assert network.vp.fitness[network.vertex(0)] == 0
        assert network.vp.fitness[network.vertex(1)] == 8
        assert network.vp.fitness[network.vertex(2)] == 6
        assert network.vp.fitness[network.vertex(3)] == 10.6
        assert network.vp.fitness[network.vertex(4)] == 0.5

    @pytest.mark.essential
    def test_fermi_function(self, create_gt_network: gt.Graph) -> None:
        """Test if the fermi function is working correclty."""
        # The given network is just a placeholder
        simulators = SimMartVaq(create_gt_network)
        np.random.seed(0)
        assert simulators.fermi_function(40, 3), "Should be True"
        assert simulators.fermi_function(-40.1, 3) is False, "Should be False"

    @pytest.mark.essential
    @given(
        x=st.floats(allow_nan=False, allow_infinity=False),
        y=st.floats(allow_nan=False, allow_infinity=False),
        t=st.floats(allow_nan=False, allow_infinity=False),
    )
    def test_fermi_function_for_overflow(
        self, x: float, y: float, t: float, create_gt_network_session: gt.Graph
    ) -> None:
        """Test for fermi function if overflow still occurs.

        Hypothesis package is used to test random numbers
        """
        assume(t != 0)
        simulators = SimMartVaq(create_gt_network_session, temperature=t)

        # Since it returns a bool anyway it will trigger the assert statements
        # But her the goal is to trigger an overflow in the equation
        try:
            simulators.fermi_function(x, y)
        except RuntimeError as e:
            raise Exception(f"An error occurred with value {x},{y},{t}") from e

    @pytest.mark.essential
    def test_interchange_roles(self, create_gt_network: gt.Graph) -> None:
        """Test if the interchanging role is working."""
        simulators = SimMartVaq(create_gt_network, temperature=10)
        # Seed 2 will trigger that the roles are interchanged
        np.random.seed(2)
        network = simulators.interchange_roles(
            network=simulators.network, person_a=0, person_b=4
        )
        # Check if wolf turned criminal based on the criminal's fitness
        assert (
            network.vp.state[network.vertex(4)] == "c"
        ), "Wolf didn't copied criminal...."
        assert (
            network.vp.state[network.vertex(0)] == "w"
        ), "Criminal didn't copied wolf...."

    @pytest.mark.essential
    def test_mutation(self, create_gt_network: gt.Graph) -> None:
        """Test if the mutation works correclty."""
        simulators = SimMartVaq(create_gt_network, temperature=10)
        # The seed should turn nodes into a criminal,honest or wolf
        np.random.seed(0)
        node = 4
        network = simulators.mutation(simulators.network, person=node)
        assert (
            network.vp.state[network.vertex(node)] == "c"
        ), "Mutation didn't work properly"
        node = 0
        network = simulators.mutation(simulators.network, person=node)
        assert (
            network.vp.state[network.vertex(node)] == "h"
        ), "Mutation didn't work properly"
        node = 3
        network = simulators.mutation(simulators.network, person=node)
        assert (
            network.vp.state[network.vertex(node)] == "c"
        ), "Mutation didn't work properly"

    @pytest.mark.essential
    def test_evolutionary_stage(self, gt_network: gt.Graph) -> None:
        """Test if the evolutionary stage is working correctly."""
        simulators = SimMartVaq(
            gt_network, mutation_prob=0.3, temperature=10, ratio_honest=0.8
        )
        # Need to randomly change the fitness
        np.random.seed(0)
        network = simulators.initialise_network(simulators.network)
        network = simulators.init_fitness(simulators.network)
        np.random.seed(0)
        for i in range(0, network.num_vertices()):
            network.vp.fitness[network.vertex(i)] = np.random.randint(0, 200)

        # To compare it to the other object
        untouched_network = deepcopy(network)
        min_grp = 5
        max_grp = 10
        dict_of_communities = simulators.select_multiple_communities(
            network=network, radius=3, min_grp=min_grp, max_grp=max_grp
        )
        mbrs = dict_of_communities[min_grp]

        # Check if the players changed status
        # With seed 0, role interchange is triggered
        np.random.seed(0)
        network = simulators.evolutionary_stage(network, mbrs)

        assert (
            network.vp.state[network.vertex(74)]
            == untouched_network.vp.state[untouched_network.vertex(291)]
        ), "Interchange function didn't work properly"
        assert (
            network.vp.state[network.vertex(291)]
            == untouched_network.vp.state[untouched_network.vertex(291)]
        ), "Interchange function didn't work properly"

        # Check if the players changed status
        # With seed 5, mutation is triggered
        np.random.seed(10)
        network = simulators.evolutionary_stage(network, mbrs)
        assert (
            network.vp.state[network.vertex(515)] == "c"
        ), "Mutation function didn't work properly"

    @pytest.mark.essential
    def test_play(self, gt_network: gt.Graph) -> None:
        """Test if the play function is working."""
        # Play the simulation
        rounds = 50
        simulator = SimMartVaq(gt_network, ratio_honest=0.7, ratio_wolf=0.05)
        network, data_collector = simulator.play(simulator.network, rounds=rounds)

        # Check if the data_collectors collect at each round data
        assert (
            len(data_collector["ratio_honest"]) == rounds
        ), "Length of the collected data is not correct..."
        assert (
            len(data_collector["ratio_wolf"]) == rounds
        ), "Length of the collected data is not correct..."
        assert (
            len(data_collector["ration_criminal"]) == rounds
        ), "Length of the collected data is not correct..."
        assert (
            len(data_collector["fitness_honest"]) == rounds
        ), "Length of the collected data is not correct..."
        assert (
            len(data_collector["fitness_criminal"]) == rounds
        ), "Length of the collected data is not correct..."
        assert (
            len(data_collector["fitness_wolf"]) == rounds
        ), "Length of the collected data is not correct..."

        # Check if the data does indeed changes, checks the simulation is working
        assert (
            len(set(data_collector["ratio_honest"])) > 1
        ), "Collectors keep collecting same value..."
        assert (
            len(set(data_collector["ratio_wolf"])) > 1
        ), "Collectors keep collecting same value..."
        assert (
            len(set(data_collector["ration_criminal"])) > 1
        ), "Collectors keep collecting same value..."
        assert (
            len(set(data_collector["fitness_honest"])) > 1
        ), "Collectors keep collecting same value..."
        assert (
            len(set(data_collector["fitness_criminal"])) > 1
        ), "Collectors keep collecting same value..."
        assert (
            len(set(data_collector["fitness_wolf"])) > 1
        ), "Collectors keep collecting same value..."
