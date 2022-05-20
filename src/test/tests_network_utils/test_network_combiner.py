"""Test if the combination between graphs works well."""
import warnings
from unittest import main

import graph_tool.all as gt
import pytest
from network_utils.network_combiner import NetworkCombiner
from network_utils.network_converter import NetworkConverter
from network_utils.network_reader import NetworkReader
from network_utils.network_stats import NetworkStats


class TestNetworkCombiner:
    """Class for unit tests for  NetworkConverter."""

    @pytest.mark.essential
    def test_combine_by_preferential_attachment_faster(self) -> None:
        """Test if combination works well."""
        n_nodes_to_add = 100
        network = NetworkReader().read_cunha()
        org_network_size = network.number_of_nodes()
        gt_network = NetworkConverter.nx_to_gt(network)

        new_gt_network = NetworkCombiner.combine_by_preferential_attachment_faster(
            gt_network, n_nodes_to_add, 2
        )
        assert isinstance(new_gt_network, gt.Graph), "Network wasn't created properly"

        # New network should have 100 new nodes
        assert (
            new_gt_network.num_vertices() == org_network_size + n_nodes_to_add
        ), "Number of new nodes is not correct!"

        # Test if the network is following a powerlaw distribution
        # First convert to Networkit
        nk_network = NetworkConverter.gt_to_nk(new_gt_network)
        network_stats = NetworkStats(nk_network)

        node_degree_dist = network_stats.get_degree_distribution(normalized=False)
        if network_stats.check_if_powerlaw(node_degree_dist)[0] is False:
            warnings.warn("Network should be scale-free")


if __name__ == "__main__":
    main()
