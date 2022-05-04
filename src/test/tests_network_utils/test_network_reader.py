"""Test if the reading of the data is done correctly."""
import networkx as nx
import pytest
from src.network_utils.network_reader import NetworkReader


class TestNetworkReader:
    """Class for unit tests for  NetworkReader."""

    @pytest.mark.essential
    def test_read_cunha(self) -> None:
        """Test if the cunha data is read correctly in nx format."""
        network_reader = NetworkReader()

        network_obj = network_reader.read_cunha()
        assert isinstance(network_obj, nx.Graph), "Network type is not correct!"

        # Check if attributes are set right for all the nodes
        assert (
            network_obj.nodes["d02e9bdc27a894e882fa0c9055c99722"]["state"] == "c"
        ), "Attributes are set wrong"
        assert (
            network_obj.nodes["adad9e1c91a7e0f63a139458941b1c66"]["state"] == "c"
        ), "Attributes are set wrong"

        assert network_obj.name == "cunha", "Network name is not correct"

    @pytest.mark.essential
    def test_read_montagna_meetings(self) -> None:
        """Test if the montagna data is read correctly in nx format."""
        network_reader = NetworkReader()
        network_obj = network_reader.read_montagna_meetings()
        assert isinstance(network_obj, nx.Graph), "Network type is not correct!"

        # Check if attributes are set right for all the nodes
        assert network_obj.nodes["N1"]["state"] == "c", "Attributes are set wrong"
        assert network_obj.nodes["N20"]["state"] == "c", "Attributes are set wrong"

        assert network_obj.name == "montagna_meetings", "Network name is not correct"

    @pytest.mark.essential
    def test_read_montagna_phone_calls(self) -> None:
        """Test if the montagna phone call data is read correctly in nx format."""
        network_reader = NetworkReader()
        network_obj = network_reader.read_montagna_phone_calls()
        assert isinstance(network_obj, nx.Graph), "Network type is not correct!"

        # Check if attributes are set right for all the nodes
        assert network_obj.nodes["N100"]["state"] == "c", "Attributes are set wrong"
        assert network_obj.nodes["N80"]["state"] == "c", "Attributes are set wrong"

        assert network_obj.name == "montagna_phone_calls", "Network name is not correct"
