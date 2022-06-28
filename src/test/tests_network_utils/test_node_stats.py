"""This script test if the function of the network_utils/node_stats work correctly.

__author__ = Louis Weyland
__date__ = 16/05/2022
"""
import graph_tool.all as gt
import numpy as np
import pytest
from network_utils.node_stats import NodeStats


class TestNodeStats:
    """Class for unit tests for NodeStats.

    The network under study is one created in conftest.py
    """

    @pytest.mark.essential
    def test_get_katz(self, bigger_gt_network: gt.Graph) -> None:
        """Test if the function return a katz value."""
        katz = NodeStats.get_katz(network=bigger_gt_network)
        assert np.any(katz.a), "Array contains only zeros..."

    @pytest.mark.essential
    def test_get_betweeness(self, bigger_gt_network: gt.Graph) -> None:
        """Test if the function return a betweeness value."""
        betweenness = NodeStats.get_betweenness(network=bigger_gt_network)
        assert np.any(betweenness[0].a), "Array contains only zeros..."

    @pytest.mark.essential
    def test_get_closeness(self, bigger_gt_network: gt.Graph) -> None:
        """Test if the function return a closeness value."""
        closeness = NodeStats.get_closeness(network=bigger_gt_network)
        assert np.any(closeness.a), "Array contains only zeros..."

    @pytest.mark.essential
    def test_get_eigenvector_centrality(self, bigger_gt_network: gt.Graph) -> None:
        """Test if the function return the eigenvalue centrality value."""
        eigen_v = NodeStats.get_eigenvector_centrality(network=bigger_gt_network)

        assert eigen_v[0] != 0, "Largest eigenvalue is zero...."
        assert np.any(eigen_v[1].a), "Array contains only zeros..."

    @pytest.mark.essential
    def test_central_dominance(self, bigger_gt_network: gt.Graph) -> None:
        """Test if the central dominace is computed corretly."""
        betweeness = NodeStats.get_betweenness(network=bigger_gt_network)
        central_dominance = NodeStats.get_central_dominance(
            network=bigger_gt_network, betweenness=betweeness
        )
        assert central_dominance != 0, "Central dominance is zero..."

    @pytest.mark.essential
    def test_get_security_efficiency_trade_off(
        self, create_gt_network: gt.Graph
    ) -> None:
        """Test if the the security_efficency trade off is computed correctly."""
        sec_eff = NodeStats.get_security_efficiency_trade_off(network=create_gt_network)
        assert sec_eff == 0.6, "Security-efficency score is not correct"

    @pytest.mark.essential
    def test_get_flow_of_information(self, create_gt_network: gt.Graph) -> None:
        """Test if the the flow of information is computed correctly."""
        flow_inf = NodeStats.get_flow_of_information(network=create_gt_network)
        assert flow_inf == 0.4, "The flow information score is not correct"

    @pytest.mark.essential
    def test_get_size_of_largest_component(self, gt_network: gt.Graph) -> None:
        """Test if the the size of the largest component is computed correctly."""
        n_nodes, n_edges = NodeStats.get_size_of_largest_component(network=gt_network)
        assert n_nodes == 84, "The size of the largest component is not correct"
        assert (
            n_edges == 112
        ), "The number of edges of the largest component is not correct"
        assert (
            gt_network.num_vertices() == 95
        ), "Thereby the initial network should be untouched"
