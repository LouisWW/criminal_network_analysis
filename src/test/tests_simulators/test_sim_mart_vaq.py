"""Test if the simulation from Martinez-Vaquero is running correctly."""
from copy import deepcopy

import graph_tool.all as gt
import numpy as np
import pytest
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
        np.random.seed(0)
        assert (
            len(gt.find_vertex(network, network.vp.state, "c")) == 95
        ), "Criminal ratio not correct"
        assert (
            38 - 10 <= len(gt.find_vertex(network, network.vp.state, "h")) <= 38 + 10
        ), "Honest ratio not correct"
        assert (
            57 - 10 <= len(gt.find_vertex(network, network.vp.state, "w")) <= 57 + 10
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
        ratio_honest = np.random.uniform(0.1, 0.99)
        ratio_wolf = np.random.uniform(0.1, (1 - 0.99))
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
        ratio_honest = np.random.uniform(0.1, 0.99)
        ratio_wolf = np.random.uniform(0.1, (1 - 0.99))
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
        """Test if the acting stage process is working correclty."""
        simulators = SimMartVaq(gt_network, ratio_wolf=0.2, ratio_honest=0.4)
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
            network, min_grp, mbrs
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
                assert network_aft_dmge.vp.fitness[
                    network_aft_dmge.vertex(node)
                ] == untouched_network.vp.fitness[untouched_network.vertex(node)] - (
                    simulators.r_w * simulators.c_w
                )
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
            assert p_h + p_c + p_w == 1, "Total ration should sum up to 1"

            assert isinstance(n_h, int), "Number should be an int"
            assert isinstance(n_c, int), "Number should be an int"
            assert isinstance(n_w, int), "Number should be an int"
            assert n_h + n_c + n_w == len(v), "Everyone should be c,h,w"

    def test_inflicting_damage(self, create_gt_network: gt.Graph) -> None:
        """Test if the inflicting damage function works correclty."""
        simulators = SimMartVaq(create_gt_network, c_c=5, r_c=1, c_w=3, r_w=2)
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
        assert network.vp.fitness[network.vertex(0)] == 24
        assert network.vp.fitness[network.vertex(2)] == -5
        assert network.vp.fitness[network.vertex(4)] == -8
        assert network.vp.fitness[network.vertex(3)] == 30
