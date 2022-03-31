"""Test if all the function from NetworkStats work correctly."""
import networkx as nx
import pytest
from network_stats import NetworkStats


class TestNetworkStats:
    """Class for unit tests for  NetworkStats.

    The network under study is one created in conftest.py
    """

    @pytest.mark.essential
    def test_networkstats_init(self, create_networkx: nx.Graph) -> None:
        """Test if error is raised if wrong network type is given."""
        with pytest.raises(Exception):
            NetworkStats(create_networkx)

    @pytest.mark.essential
    def test_efficiency(self, network_stats_obj: NetworkStats) -> None:
        """Test if the efficiency is correct."""
        efficiency = network_stats_obj.get_efficiency()
        assert isinstance(efficiency, float), "Efficiency should be float"
        assert efficiency == 0.8, "Efficiency score is not correct"
