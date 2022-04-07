"""Test if the reading of the data is done correctly."""
import networkx as nx
import pytest
from network_utils.network_reader import NetworkReader


class TestNetworkReader:
    """Class for unit tests for  NetworkReader."""

    @pytest.mark.essential
    def test_read_cunha(self) -> None:
        """Test if the cunha data is read correctly in nx format."""
        network_reader = NetworkReader()

        network_obj = network_reader.read_cunha()
        assert isinstance(network_obj, nx.Graph), "Network type is not correct!"

    @pytest.mark.essential
    def test_read_montagna_meetings(self) -> None:
        """Test if the montagna data is read correctly in nx format."""
        network_reader = NetworkReader()
        network_obj = network_reader.read_montagna_meetings()
        assert isinstance(network_obj, nx.Graph), "Network type is not correct!"

    @pytest.mark.essential
    def test_read_montagna_phone_calls(self) -> None:
        """Test if the montagna phone call data is read correctly in nx format."""
        network_reader = NetworkReader()
        network_obj = network_reader.read_montagna_phone_calls()
        assert isinstance(network_obj, nx.Graph), "Network type is not correct!"
