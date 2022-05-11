"""Test if all the function from NetworkStats work correctly."""
import warnings
from unittest import main

import networkit as nk
import networkx as nx
import pytest
from src.network_utils.network_stats import NetworkStats


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

    @pytest.mark.essential
    def test_check_if_powerlaw(
        self, scale_free_network: nk.Graph, random_network: nk.Graph
    ) -> None:
        """Test if the check_if_powerlaw_function is working.

        For this test, a scale-free network is used and an random
        one to test, true positive and true negative. The network form
        conftest is to small to tes that!
        """
        # Check if the scale-free network returns correct results
        network_stats_scf = NetworkStats(scale_free_network)
        nodes_degree_dist = network_stats_scf.get_degree_distribution(normalized=False)
        if network_stats_scf.check_if_powerlaw(nodes_degree_dist)[0] is False:
            warnings.warn("Network should be scale-free")

        # Check if the random network returns correct results
        network_stats_rdm = NetworkStats(random_network)
        nodes_degree_dist = network_stats_rdm.get_degree_distribution()
        if network_stats_rdm.check_if_powerlaw(nodes_degree_dist)[0] is True:
            warnings.warn("Network should not be scale-free")


if __name__ == "__main__":
    main()
