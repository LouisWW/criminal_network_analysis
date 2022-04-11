"""Test if the simulation from Martinez-Vaquero is running correctly."""
import graph_tool.all as gt
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
