"""Test if all the function from NetworkStats work correctly."""
import networkx as nx
import pytest
from network_utils.network_stats import NetworkStats


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
    def test_get_efficiency(self, network_stats_obj: NetworkStats) -> None:
        """Test if the efficiency is correct."""
        efficiency = network_stats_obj.get_efficiency()
        assert isinstance(efficiency, float), "Efficiency should be float"
        assert efficiency == 0.8, "Efficiency score is not correct"

    @pytest.mark.essential
    def test_degree_dispersion(self, network_stats_obj: NetworkStats) -> None:
        """Test if the dispersion is correct."""
        dispersion = network_stats_obj.get_degree_dispersion()
        assert isinstance(dispersion, float), "Dispersion should be float"
        assert dispersion == 2.5, "Dispersion score is not correct"

    @pytest.mark.essential
    def test_get_density(self, network_stats_obj: NetworkStats) -> None:
        """Test if density is correct."""
        rel_density = network_stats_obj.get_relative_density()
        assert isinstance(rel_density, float), "Density should be float"
        assert rel_density == 0.6, "Density score is not correct"

    @pytest.mark.essential
    def test_get_relative_density(self, network_stats_obj: NetworkStats) -> None:
        """Test if density is correct."""
        density = network_stats_obj.get_density()
        assert isinstance(density, float), "Density should be float"
        assert density == 0.6, "Density score is not correct"

    @pytest.mark.essential
    def test_get_eccentricity(self, network_stats_obj: NetworkStats) -> None:
        """Test if the eccentricity is correct."""
        eccentricity = network_stats_obj.get_eccentricity(0)
        assert isinstance(eccentricity, int), "Eccentricity should be an int"
        assert eccentricity == 2, "Eccentricity is not correct."

    @pytest.mark.essential
    def test_get_radius(self, network_stats_obj: NetworkStats) -> None:
        """Test if radius is correct."""
        radius = network_stats_obj.get_radius()
        assert isinstance(radius, int), "Radius should be int"
        assert radius == 2, "Radius score is not correct"

    @pytest.mark.essential
    def test_get_diameter(self, network_stats_obj: NetworkStats) -> None:
        """Test if diameter is correct."""
        diameter = network_stats_obj.get_diameter()
        assert isinstance(diameter, int), "Diameter should be int"
        assert diameter == 2, "Diameter score is not correct"
