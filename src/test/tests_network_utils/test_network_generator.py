"""Test if the generating of network is done correctly."""
from unittest import main

import networkit as nk
import pytest
from network_utils.network_generator import NetworkGenerator


class TestNetworkGenerator:
    """Class for unit tests for  NetworkGenerator."""

    @pytest.mark.essential
    def test_generate_barabasi_albert(self) -> None:
        """Test if network is created properly."""
        network_obj = NetworkGenerator.generate_barabasi_albert(n_nodes=60, n_edges=2)
        assert isinstance(network_obj, nk.Graph), "network not created properly"
        assert network_obj.numberOfNodes() == 60, "Number of nodes is not correct!"

    @pytest.mark.essential
    def test_generate_random(self) -> None:
        """Test if network is created properly."""
        network_obj = NetworkGenerator.generate_random(n_nodes=60)
        assert isinstance(network_obj, nk.Graph), "network not created properly"
        assert network_obj.numberOfNodes() == 60, "Number of nodes is not correct!"


if __name__ == "__main__":
    main()
