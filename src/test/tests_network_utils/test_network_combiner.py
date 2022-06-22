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
        network = NetworkReader().read_montagna_phone_calls()
        org_network_size = network.number_of_nodes()
        org_network_n_edges = network.number_of_edges()
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

        # Checking if the criminal network is intact
        filtering = new_gt_network.new_vertex_property("bool")
        new_gt_network.vertex_properties["filtering"] = filtering
        for i in range(0, new_gt_network.num_vertices()):
            if new_gt_network.vp.state[new_gt_network.vertex(i)] == "c":
                new_gt_network.vp.filtering[new_gt_network.vertex(i)] = 1
            else:
                new_gt_network.vp.filtering[new_gt_network.vertex(i)] = 0

        new_gt_network.set_vertex_filter(new_gt_network.vp.filtering)
        assert (
            new_gt_network.num_vertices() == org_network_size
        ), "Criminal network should be intact"
        assert (
            new_gt_network.num_edges() == org_network_n_edges
        ), "Criminal network should be intact"

    @pytest.mark.essential
    def test_combine_by_random_attachment_faster(self) -> None:
        """Test if the random attachment is working."""
        n_nodes_to_add = 100
        network = NetworkReader().read_montagna_phone_calls()
        org_network_size = network.number_of_nodes()
        org_network_n_edges = network.number_of_edges()
        gt_network = NetworkConverter.nx_to_gt(network)

        new_gt_network = NetworkCombiner.combine_by_random_attachment_faster(
            gt_network, n_nodes_to_add, 0.5
        )
        assert isinstance(new_gt_network, gt.Graph), "Network wasn't created properly"

        # New network should have 100 new nodes
        assert (
            new_gt_network.num_vertices() == org_network_size + n_nodes_to_add
        ), "Number of new nodes is not correct!"

        # Checking if the criminal network is intact
        filtering = new_gt_network.new_vertex_property("bool")
        new_gt_network.vertex_properties["filtering"] = filtering
        for i in range(0, new_gt_network.num_vertices()):
            if new_gt_network.vp.state[new_gt_network.vertex(i)] == "c":
                new_gt_network.vp.filtering[new_gt_network.vertex(i)] = 1
            else:
                new_gt_network.vp.filtering[new_gt_network.vertex(i)] = 0

        new_gt_network.set_vertex_filter(new_gt_network.vp.filtering)
        assert (
            new_gt_network.num_vertices() == org_network_size
        ), "Criminal network should be intact"
        assert (
            new_gt_network.num_edges() == org_network_n_edges
        ), "Criminal network should be intact"

    @pytest.mark.essential
    def test_combine_by_small_world_attachment(self) -> None:
        """Test if the random attachment is working."""
        n_nodes_to_add = 100
        network = NetworkReader().read_montagna_phone_calls()
        org_network_size = network.number_of_nodes()
        org_network_n_edges = network.number_of_edges()
        gt_network = NetworkConverter.nx_to_gt(network)

        new_gt_network = NetworkCombiner.combine_by_small_world_attachment(
            gt_network, n_nodes_to_add, 50, 0.5
        )
        assert isinstance(new_gt_network, gt.Graph), "Network wasn't created properly"

        # New network should have 100 new nodes
        assert (
            new_gt_network.num_vertices() == org_network_size + n_nodes_to_add
        ), "Number of new nodes is not correct!"

        # Checking if the criminal network is intact
        filtering = new_gt_network.new_vertex_property("bool")
        new_gt_network.vertex_properties["filtering"] = filtering
        for i in range(0, new_gt_network.num_vertices()):
            if new_gt_network.vp.state[new_gt_network.vertex(i)] == "c":
                new_gt_network.vp.filtering[new_gt_network.vertex(i)] = 1
            else:
                new_gt_network.vp.filtering[new_gt_network.vertex(i)] = 0

        new_gt_network.set_vertex_filter(new_gt_network.vp.filtering)
        assert (
            new_gt_network.num_vertices() == org_network_size
        ), "Criminal network should be intact"
        assert (
            new_gt_network.num_edges() == org_network_n_edges
        ), "Criminal network should be intact"


if __name__ == "__main__":
    main()
