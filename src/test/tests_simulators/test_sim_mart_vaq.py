"""Test if the simulation from Martinez-Vaquero is running correctly."""
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
        assert simulators.relative_ratio_honest == 0.75, "Relative ratio is wrong"
        assert simulators.relative_ratio_wolf == 0.25, "Relative ratio is worng"

        # Try to change its name
        with pytest.raises(Exception):
            simulators.name = "New name"

    @pytest.mark.essential
    def test_sim_mart_vaq_wrong_init(self, gt_network: gt.Graph) -> None:
        """Test if a wrong init triggers the assert statments."""
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
        simulators = SimMartVaq(gt_network)
        groups, groups_label = simulators.divide_in_groups(simulators.network)

        assert groups is not np.empty, "Group list is empty"
        assert all(
            [isinstance(item, int) for item in groups]
        ), "Items in Group should be int"
        assert isinstance(groups_label, frozenset), "Group label should be frozenset"
        assert len(groups_label) > 0, "Group label is empty"
        assert all(
            [isinstance(item, int) for item in groups_label]
        ), "Items in Group label should be int"

        # Make sure that the group membership are random each time
        # This will make sure that not always the same groups are selected
        # and that some behaviour might be able to spread over the network
        groups_1, _ = simulators.divide_in_groups(simulators.network)
        groups_2, _ = simulators.divide_in_groups(simulators.network)

        assert not (
            groups_1 == groups_2
        ).all(), "Every time the same groups are formed..."

    @pytest.mark.essential
    def test_init_fitness(self, gt_network: gt.Graph) -> None:
        """Test if the init of the fitness attribute is done correctly."""
        simulators = SimMartVaq(gt_network)
        network = simulators.init_fitness(simulators.network)
        assert network.vp.fitness, "Fitness attribute doesn't exists..."

    @pytest.mark.essential
    def test_acting_stage(self, gt_network: gt.Graph) -> None:
        """Test if the acting stage process is working correclty."""
        simulators = SimMartVaq(gt_network)
        network = simulators.init_fitness(simulators.network)
        mbr_list, group_numbers = simulators.divide_in_groups(network)

        # Select one group number from the all the numbers
        group_number = next(iter(group_numbers))
        network = simulators.acting_stage(network, mbr_list, group_number)
        raise NotImplementedError
