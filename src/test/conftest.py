"""
This script's contains all the pytest fixtures.

The fixtures are a piece of code which can be used
for multiple tests as setup.
By doing so we can avoid to setup time-consuming lines of code
for multiple tests


__author__ = Louis Weyland
__date__   = 23/02/2022
"""
import networkx as nx
import pytest
from network_utils.network_converter import NetworkConverter
from network_utils.network_stats import NetworkStats


@pytest.fixture(scope="session")
def create_networkx() -> nx.Graph:
    """
    Create a well-known network with attributes.

    Since it is well-known we know how the outcome of different
    functions need to be!
    """
    network = nx.Graph()
    network.add_edge(0, 1)
    network.add_edge(0, 2)
    network.add_edge(0, 4)
    network.add_edge(1, 2)
    network.add_edge(2, 3)
    network.add_edge(3, 4)

    # set nodes attributes fitness
    network.nodes[0]["fitness"] = 10
    network.nodes[1]["fitness"] = 8
    network.nodes[2]["fitness"] = 6
    network.nodes[3]["fitness"] = 3
    network.nodes[4]["fitness"] = 11

    # set nodes attributes age
    network.nodes[0]["age"] = 30
    network.nodes[1]["age"] = 35
    network.nodes[2]["age"] = 20
    network.nodes[3]["age"] = 60
    network.nodes[4]["age"] = 4

    return network


@pytest.fixture(scope="session")
def network_stats_obj(create_networkx: nx.Graph) -> NetworkStats:
    """Create a NetworkStats object.

    Thereby the create_networkx is converted to nk
    """
    conv_graph = NetworkConverter.nx_to_nk(create_networkx)
    network_stats = NetworkStats(conv_graph)
    return network_stats
